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
#include "blaze/gemm/tile/arch35/copy_gm_to_l1.h"
#include "../../../common/mega_moe_concat_geometry.h"

namespace MegaMoeImpl {

// Source is a slice starting
// at gate[j], retaining the full GM strides. Its logical N is the COMPUTE width
// (gate + up); only its left half is read before jumping to up[j]. Destination
// already describes the concatenated compute tile. No global weight permutation.
template <typename Tensor>
__aicore__ inline uint64_t ConcatTensorRows(const Tensor &tensor)
{
    const auto &shape = AscendC::Std::get<0>(tensor.Layout().Shape());
    return AscendC::Std::get<0>(shape) * AscendC::Std::get<1>(shape);
}

template <typename Tensor>
__aicore__ inline uint64_t ConcatTensorColumns(const Tensor &tensor)
{
    const auto &shape = AscendC::Std::get<1>(tensor.Layout().Shape());
    return AscendC::Std::get<0>(shape) * AscendC::Std::get<1>(shape);
}

template <typename Dst, typename Src>
__aicore__ inline void CopyGmm1WeightConcatToL1(const Dst &dst, const Src &src, uint64_t halfN)
{
    using Pattern = AscendC::Te::GetLayoutPattern<typename Src::layoutType>;
    using Element = asc::te::get_attribute_element_type<typename Src::elementType *>;
    constexpr bool packed = Blaze::Gemm::IsFp4<Element>();
    const uint64_t singleN = ConcatTensorColumns(src) / 2U;
    const uint64_t realK = ConcatTensorRows(src);
    const uint8_t cacheMode = src.Engine().GetCacheMode();
    if constexpr (AscendC::Std::is_same_v<Pattern, AscendC::Te::ZNLayoutPtn>) {
        const uint64_t c0 = AscendC::Std::get<0>(AscendC::Std::get<0>(src.Layout().Shape()));
        const auto copy = MakeGmm1ConcatZnCopy(realK, singleN, halfN, c0, packed ? 2U : 1U);
        asc_copy_gm2l1_align(reinterpret_cast<__cbuf__ uint8_t *>(dst.Data().Get()),
                             reinterpret_cast<__gm__ uint8_t *>(src.Data().Get()), copy.blockCount, copy.blockBytes, 0,
                             0, true, cacheMode, copy.sourceStrideBytes, copy.destinationStrideBytes);
    } else {
        static_assert(AscendC::Std::is_same_v<Pattern, AscendC::Te::DNExtLayoutPtn>,
                      "MegaMoe GMM1 weight concat supports DN and ZN weights");
        // GM is [2N, K], L1 is ZN [Ktile, 2*singleN]. Hardware matrix loop loads
        // gate/up as two DN matrices into adjacent N regions of the same L1 tile.
        const uint64_t fullK = AscendC::Std::get<1>(AscendC::Std::get<1>(src.Layout().Stride()));
        auto left = src.Slice(AscendC::Te::MakeCoord(uint64_t(0), uint64_t(0)), AscendC::Te::MakeShape(realK, singleN));
        Blaze::Gemm::Tile::CopyConcatGM2L1::Copy(dst, left, {halfN * 2U, fullK});
    }
}

template <typename Dst, typename Src>
__aicore__ inline void CopyGmm1ScaleConcatToL1(const Dst &dst, const Src &src, uint64_t halfN, uint64_t fullK)
{
    using Pattern = AscendC::Te::GetLayoutPattern<typename Src::layoutType>;
    static_assert(AscendC::Std::is_same_v<Pattern, AscendC::Te::ScaleBDNLayoutPtn>,
                  "MegaMoe GMM1 scale concat expects ScaleBDN");
    const uint64_t singleN = ConcatTensorColumns(src) / 2U;
    const uint64_t scaleSpan = ConcatTensorRows(src);
    auto left = src.Slice(AscendC::Te::MakeCoord(uint64_t(0), uint64_t(0)), AscendC::Te::MakeShape(scaleSpan, singleN));
    Blaze::Gemm::Tile::CopyConcatGM2L1::Copy(dst, left, {halfN * 2U, fullK});
}

} // namespace MegaMoeImpl
