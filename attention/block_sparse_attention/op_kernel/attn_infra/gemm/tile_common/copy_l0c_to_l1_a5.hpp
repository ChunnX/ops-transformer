/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_TILE_COPY_L0C_TO_L1_A5_HPP
#define GEMM_TILE_COPY_L0C_TO_L1_A5_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_arch.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l0c_to_dst.hpp"
#include "../../../tla/tensor.hpp"

#if (__CCE_AICORE__ == 310)
// Fixpipe L0C -> L1（A1）通路：NZ 保形输出（NZ2NZ），isToUB=false
constexpr AscendC::FixpipeConfig CFG_NZ_L1 = {AscendC::CO2Layout::NZ, false};
#endif

namespace NpuArch::Gemm::Tile {

// L0C -> L1，NZ 保形（CFG_NZ），无量化
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToL1Tla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::A1>,
    ScaleGranularity::NO_QUANT,
    ReluEnable_> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be L1");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        // NZ2NZ：目的 NZ 矩阵相邻 Z 排布的起始地址偏移 = mSize × C0_Size(32B) / sizeof(T)
        // （fixpipe_l0c2l1 样例：dstStride = baseM × c0Size，c0Size 单位 B；换算成 element 即本公式）
        intriParams.dstStride = tla::get<0>(dstTensor.shape()) * NpuArch::BYTE_PER_C0 / sizeof(ElementDst);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_NZ_L1>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

// L0C -> L1，NZ 保形（CFG_NZ），scalar 反量化（QF322F32_PRE）
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToL1Tla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::A1>,
    ScaleGranularity::PER_TENSOR,
    ReluEnable_> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_TENSOR>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint64_t deqScalar,
                                      uint8_t unitFlag = 0)
    {
        static_assert(
            TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be L1");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        // NZ2NZ：目的 NZ 矩阵相邻 Z 排布的起始地址偏移 = mSize × C0_Size(32B) / sizeof(T)
        // （fixpipe_l0c2l1 样例：dstStride = baseM × c0Size，c0Size 单位 B；换算成 element 即本公式）
        intriParams.dstStride = tla::get<0>(dstTensor.shape()) * NpuArch::BYTE_PER_C0 / sizeof(ElementDst);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.deqScalar = deqScalar;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_NZ_L1>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_COPY_L0C_TO_L1_A5_HPP
