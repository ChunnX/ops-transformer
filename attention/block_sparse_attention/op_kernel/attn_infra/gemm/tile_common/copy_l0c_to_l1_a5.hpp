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
 * 反量化比值由DIAGDM记账，deqScalar在本通路不生效。
 * 仅FP8 ND RescaleOnCube的中间块用于将GO^T写入L1。
 */
#ifndef GEMM_TILE_COPY_L0C_TO_L1_A5_HPP
#define GEMM_TILE_COPY_L0C_TO_L1_A5_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_arch.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_copy_l0c_to_dst.hpp"
#include "../../../tla/tensor_bsa.hpp"

namespace NpuArch::Gemm::Tile {

#if (__CCE_AICORE__ == 310)
// L0C -> L1（C1），NZ 保形（channelSplit），DataCopy 通路（950 官方支持）
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
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be L1(C1)");

        AscendC::DataCopyCO12DstParams intriParams;
        // dst 为 zN 布局时 shape 每维 = (fractal 内元素数, fractal 数) 嵌套 tuple，
        // 总元素数 = 内层两分量乘积（与 block_epilogue_online_softmax_arch35_reg_low_prec.hpp 同款变参取法）
        auto dstShape = dstTensor.shape();
        uint32_t nSize =
            static_cast<uint32_t>(tla::get<1, 0>(dstShape)) * tla::get<1, 1>(dstShape);
        uint32_t mSize =
            static_cast<uint32_t>(tla::get<0, 0>(dstShape)) * tla::get<0, 1>(dstShape);
        // DataCopy（CO1->C1）官方语义：
        //   nSize：源 NZ N 方向大小（channelSplit 时 C0=8，nSize 须 8 的倍数）
        //   mSize：源 NZ M 方向大小
        //   srcStride：源 NZ 相邻 Z 排布起始地址偏移，单位 C0_Size（16×sizeof(T)），= mSize 对 16 向上取整
        //   dstStride：目的 NZ 相邻 Z 排布起始地址偏移，单位 datablock（32 字节）
        intriParams.nSize = nSize;   // qS（=qBaseTile_，8 的倍数）
        intriParams.mSize = mSize;   // D（=embed ∈ {64,128}，16 的倍数）
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        // dstStride（datablock 32B 单位）：L1 zN 每行恰 32B → 列块步长折合的 datablock 数 = RoundUp(mSize,16)，
        // 与下方 stride 比值数值恒等（[RoundUp(mSize,16)×ELE_NUM_PER_C0] / ELE_NUM_PER_C0 = RoundUp(mSize,16)）；
        // 直读写法：intriParams.dstStride = RoundUp(mSize, 16);
        // （D∈{64,128} 时 = mSize，与官方样例公式 baseM×c0Size×sizeof/ONE_BLK_SIZE 等价）
        intriParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / tla::get<0, 0>(dstTensor.stride());
        intriParams.channelSplit = true;  // L0C 每个 C0（64B，16×16 fp32）拆为 2 个 L1 datablock（32B，16×8）
        // quantPre/nz2ndEn/reluPre/unitFlag：保持默认（NoQuant 恒等、NZ2NZ、无 ReLU、无 unitFlag）
        // ——CO1->C1 无 float 反量化枚举；官方样例仅填四参数；deqScalar 本通路不生效
        // （raw 域保形，比值记账在 softmax 侧）

        // 偏移计算与指令无关（tla 纯地址计算）：整块搬 coord=(0,0) → offset=0；GetTile 子块也能正确算
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC DataCopy（CO1 -> C1）
        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_COPY_L0C_TO_L1_A5_HPP
