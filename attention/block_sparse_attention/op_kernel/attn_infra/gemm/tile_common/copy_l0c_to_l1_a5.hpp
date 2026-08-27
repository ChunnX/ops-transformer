/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief L0C -> L1（CO1->C1）数据搬运：DataCopy 通路 + channelSplit。
 * 背景（950）：L0C fractal = 16×16 元素（fp32 每 C0 = 64B），L1 标准 NZ = 32B/C0（fp32 16×8）；
 * Fixpipe 的 channelSplit 仅 L0C->GM 通路有效，L0C->L1 无法做 16×16->16×8 切分；
 * DataCopyCO12Dst（CO1->C1）支持 channelSplit，是 L0C->L1 的正确通路（dst 位置标签必须为 C1）。
 * 反量化：CO1->C1 的 quantPre 无 float->float 枚举，保形浮点搬移（raw 域），
 * 记账改比值方案（diag(dm·vS_{t-1}/vS_t)，见 S_trans SetDiagonalDeltaMax）；deqScalar 本通路不生效。
 * 仅 FP8+DN+rescaleOnCube 场景使用（唯一调用点 block_mmad_pv_transposed_rescale_arch35.hpp 中间块分支）。
 */
#ifndef GEMM_TILE_COPY_L0C_TO_L1_A5_HPP
#define GEMM_TILE_COPY_L0C_TO_L1_A5_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_arch.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l0c_to_dst.hpp"
#include "../../../tla/tensor.hpp"

namespace NpuArch::Gemm::Tile {

#if (__CCE_AICORE__ == 310)
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToL1Tla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::C1>,
    ScaleGranularity::PER_TENSOR,
    ReluEnable_> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static_assert(AscendC::IsSameType<ElementSrc, float>::value &&
                  AscendC::IsSameType<ElementDst, float>::value,
        "DataCopy CO1->C1 channelSplit only supports float NZ output");

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::C1,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be L1");

        AscendC::DataCopyCO12DstParams intriParams;
        auto dstShape = dstTensor.shape();
        uint32_t nSize =
            static_cast<uint32_t>(tla::get<1, 0>(dstShape)) * tla::get<1, 1>(dstShape);
        uint32_t mSize =
            static_cast<uint32_t>(tla::get<0, 0>(dstShape)) * tla::get<0, 1>(dstShape);
        intriParams.nSize = nSize;
        intriParams.mSize = mSize;
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / tla::get<0, 0>(dstTensor.stride());
        intriParams.channelSplit = true;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_COPY_L0C_TO_L1_A5_HPP
