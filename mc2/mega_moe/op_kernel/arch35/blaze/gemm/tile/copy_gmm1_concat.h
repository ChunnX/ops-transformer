/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/tile/copy_gm_to_l1.h"
#include "blaze/gemm/tile/tile_trait.h"

namespace MegaMoeImpl {

// Keep the tensor's layout/engine while carrying the local NZ copy policy through
// QGMM's K slices. Ordinary tensors still use the upstream concat copy unchanged.
template <typename Tensor>
struct Gmm1NzWeightView : Tensor {
    __aicore__ explicit inline Gmm1NzWeightView(const Tensor &tensor)
        : Tensor(tensor)
    {}

    template <typename Coord, typename Shape>
    __aicore__ inline auto slice(const Coord &coord, const Shape &shape) const
    {
        // Delegate indexing to Tensor; preserve the NZ copy tag on the returned view.
        auto slice = Tensor::slice(coord, shape);
        return Gmm1NzWeightView<decltype(slice)>(slice);
    }
};

// QGMM owns accumulation; MegaMoe owns the destination and valid output extent.
template <typename Tensor>
struct Gmm1OutputView {
    Tensor tensor;
};

template <typename Tensor>
__aicore__ inline auto MakeGmm1WeightView(const Tensor &tensor)
{
    using Pattern = asc::te::get_layout_pattern<typename Tensor::layout_type>;
    if constexpr (AscendC::Std::is_same_v<Pattern, asc::te::zn_layout_ptn>) {
        return Gmm1NzWeightView<Tensor>(tensor);
    } else {
        return tensor;
    }
}

// Source is a slice starting
// at gate[j], retaining the full GM strides. Its logical N is the COMPUTE width
// (gate + up); only its left half is read before jumping to up[j]. Destination
// already describes the concatenated compute tile. No global weight permutation.
template <typename Tensor>
__aicore__ inline uint64_t ConcatTensorRows(const Tensor &tensor)
{
    const auto &shape = AscendC::Std::get<0>(tensor.layout().shape());
    return AscendC::Std::get<0>(shape) * AscendC::Std::get<1>(shape);
}

template <typename Tensor>
__aicore__ inline uint64_t ConcatTensorColumns(const Tensor &tensor)
{
    const auto &shape = AscendC::Std::get<1>(tensor.layout().shape());
    return AscendC::Std::get<0>(shape) * AscendC::Std::get<1>(shape);
}

template <typename Dst, typename Src>
__aicore__ inline void CopyGmm1WeightConcatToUb(const Dst &dst, const Src &src, uint64_t halfN)
{
    using Pattern = asc::te::get_layout_pattern<typename Src::layout_type>;
    ASCENDC_ASSERT((AscendC::Std::is_same_v<Pattern, asc::te::zn_layout_ptn>),
                   { KERNEL_LOG(KERNEL_ERROR, "A8W4 prologue consumes packed C0=32 ZN weights"); });
    const uint64_t c0 = AscendC::Std::get<0>(AscendC::Std::get<0>(src.layout().shape()));
    const uint64_t columnBytes = c0 / 2U;
    const uint32_t blockCount = 2U * ConcatTensorRows(src) / c0;
    const uint32_t blockBytes = ConcatTensorColumns(src) / 2U * columnBytes;
    // Describe the physical NZ bursts as byte rows; the UB view writes them consecutively.
    auto shape = asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, blockCount),
                                     asc::te::make_shape(asc::te::_1{}, blockBytes));
    auto srcStride = asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, halfN * columnBytes),
                                          asc::te::make_stride(asc::te::_0{}, asc::te::_1{}));
    auto dstStride = asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, blockBytes),
                                          asc::te::make_stride(asc::te::_0{}, asc::te::_1{}));
    using ByteLayout = decltype(asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1U, 1U));
    using ByteLayoutTrait = asc::te::get_layout_trait<ByteLayout>;
    auto srcLayout = asc::te::make_pattern_layout<asc::te::nd_ext_layout_ptn, ByteLayoutTrait>(shape, srcStride);
    auto dstLayout = asc::te::make_pattern_layout<asc::te::nd_ext_layout_ptn, ByteLayoutTrait>(shape, dstStride);
    auto srcBytes = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ uint8_t *>(src.data().get())), srcLayout);
    auto dstBytes = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub>(reinterpret_cast<__ubuf__ uint8_t *>(dst.data().get())),
        dstLayout);
    srcBytes.set_l2_cache_hint(src.get_cache_mode());
    asc::te::copy(asc::te::make_copy(asc::te::copy_gm_to_ub{}).with(asc::te::gm_to_ub_params{0, 0, false}), dstBytes,
                  srcBytes);
}

template <typename Dst, typename Src>
__aicore__ inline void CopyGmm1ScaleConcatToL1(const Dst &dst, const Src &src, uint64_t halfN, uint64_t fullK)
{
    using Pattern = asc::te::get_layout_pattern<typename Src::layout_type>;
    ASCENDC_ASSERT((AscendC::Std::is_same_v<Pattern, asc::te::scaleb_dn_layout_ptn>),
                   { KERNEL_LOG(KERNEL_ERROR, "MegaMoe GMM1 scale concat expects ScaleBDN"); });
    const uint64_t singleN = ConcatTensorColumns(src) / 2U;
    const uint64_t scaleSpan = ConcatTensorRows(src);
    auto left = src.slice(asc::te::make_coord(uint64_t(0), uint64_t(0)), asc::te::make_shape(scaleSpan, singleN));
    Blaze::Gemm::Tile::CopyConcatGM2L1::Copy(dst, left, {halfN * 2U, fullK});
}

} // namespace MegaMoeImpl

namespace asc::te {

// These overloads match only MegaMoe's view types. Declare them before including
// QGMM, whose qualified copy calls resolve the overload set at template definition.
template <typename Dst, typename Src>
__aicore__ inline void copy(const copy_atom<copy_traits<Blaze::Gemm::Tile::CopyConcatGM2L1>> &copy, const Dst &dst,
                            const MegaMoeImpl::Gmm1NzWeightView<Src> &src)
{
    // Treat gate/up as twice as many K fractals with half the GM fractal stride.
    // Keep the native ZN pattern: ordinary ZN->ZN copy emits one align DMA without
    // the runtime compact-mode selection of an ND byte view. FP4 strides remain
    // in element units; tensor copy converts them to bytes.
    auto shape = src.layout().shape();
    auto srcStride = src.layout().stride();
    auto dstStride = srcStride;
    const auto c0 = AscendC::Std::get<0>(AscendC::Std::get<0>(shape));
    AscendC::Std::get<1>(AscendC::Std::get<0>(shape)) *= 2U;
    AscendC::Std::get<1>(AscendC::Std::get<0>(srcStride)) = (copy.params.n / 2U) * c0;
    AscendC::Std::get<1>(AscendC::Std::get<0>(dstStride)) = get_total_column_shape(src.layout()) * c0;
    using Trait = get_layout_trait<typename Src::layout_type>;
    auto srcLayout = make_pattern_layout<zn_layout_ptn, Trait>(shape, srcStride);
    auto dstLayout = make_pattern_layout<zn_layout_ptn, Trait>(shape, dstStride);
    auto srcConcat = make_tensor(src.data(), srcLayout);
    auto dstConcat = make_tensor(dst.data(), dstLayout);
    srcConcat.set_l2_cache_hint(src.get_cache_mode());
    asc::te::copy(make_copy(copy_gm_to_l1{}), dstConcat, srcConcat);
}

template <typename Atom, typename Dst, typename Src>
__aicore__ inline void copy(const copy_atom<Atom> &, const MegaMoeImpl::Gmm1OutputView<Dst> &dst, const Src &src)
{
    auto valid =
        src.slice(make_coord(uint64_t(0), uint64_t(0)),
                  make_shape(get_total_row_shape(dst.tensor.layout()), get_total_column_shape(dst.tensor.layout())));
    if constexpr (AscendC::Std::is_same_v<get_mem_location<Dst>, location::ub>) {
        copy(make_copy(copy_l0c_to_ub{}).with(l0c_to_ub_params(unit_flag_mode::enable_update)), dst.tensor, valid);
    } else {
        copy(make_copy(copy_l0c_to_gm{}).with(l0c_to_gm_params(unit_flag_mode::enable_update)), dst.tensor, valid);
    }
}

} // namespace asc::te

#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
