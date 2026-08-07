/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_TILE_COPY_GM_TO_L1_MX_SCALE_DN2NZ_A5_HPP
#define GEMM_TILE_COPY_GM_TO_L1_MX_SCALE_DN2NZ_A5_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_arch.hpp"

namespace NpuArch::Gemm::Tile {

// =============================================================================
// QK 专用：Host 沿 D 的 E8M0 scale GM→L1（Dn2Nz）
// -----------------------------------------------------------------------------
// Host 布局：[S, D//64, 2]，每 token scaleK = Ceil(D/64)*2 字节。
// 与 PV 的 MxScaleA(ColumnMajor,trans)→zZ（Nd2Nz、沿 S）完全不同，禁止混用。
//
// 忠实移植：
//   CopyQScaleGmToL1 / CopyKScaleGmToL1（稠密整段）
//   SparseKScaleBaseTileL1FullLoad 内每段 DataCopy（支持 B16 偏移）
// =============================================================================

/// Q/K-scale：GM uint8(E8M0) → L1，按 half 视图 Dn2Nz。
struct CopyGmToL1MxScaleDn2NzA5 {
    __aicore__ inline CopyGmToL1MxScaleDn2NzA5() {}

    // rows   : 本次搬的 token 行数（稠密 = m/n；稀疏段 = curDealtLen）
    // scaleK : Ceil(D/64)*2
    // srcB16Off / dstB16Off : 稀疏多段拼接时的 half 偏移；稠密传 0
    // headMul: 头内 token 步长因子。BNSD=1（头内 token 连续）；
    //          BSND/TND=N（GM [S/T,N,D//64,2]，同头相邻 token 隔 N 个 scale 行）。
    //          原写死 srcDValue=scaleRowHalf，仅 BNSD 正确；TND/BSND 多头时会读到邻头
    //          /错 token 的 scale。CX dst≠2^k 时少数 token 的 pair 内 scale 不等才暴露
    //          （mm1 仅若干 Q 列错）；dst=8/OCP 下 scale 近乎全等，缺陷被掩盖。
    __aicore__ inline void operator()(
        AscendC::LocalTensor<uint8_t> const &l1Scale,
        AscendC::GlobalTensor<uint8_t> const &gScale,
        uint32_t rows,
        uint32_t scaleK,
        uint32_t srcB16Off = 0,
        uint32_t dstB16Off = 0,
        uint32_t headMul = 1)
    {
        uint32_t scaleRowHalf = scaleK / 2;

        AscendC::GlobalTensor<half> gScaleB16;
        gScaleB16.SetGlobalBuffer((__gm__ half *)(gScale.GetPhyAddr()), 0x7FFFFFFFu);
        auto l1ScaleB16 = l1Scale.ReinterpretCast<half>();

        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        dn2nzParams.dValue = rows;
        dn2nzParams.nValue = scaleRowHalf;
        dn2nzParams.srcDnMatrixStride = 0;
        // 同头相邻 token 的 half 步长 = headMul * (D//64)
        dn2nzParams.srcDValue = headMul * scaleRowHalf;
        dn2nzParams.dstNzC0Stride = scaleRowHalf;
        dn2nzParams.dstNzNStride = 1;
        dn2nzParams.dstNzMatrixStride = 0;

        AscendC::DataCopy(l1ScaleB16[dstB16Off], gScaleB16[srcB16Off], dn2nzParams);
    }
};

} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_COPY_GM_TO_L1_MX_SCALE_DN2NZ_A5_HPP