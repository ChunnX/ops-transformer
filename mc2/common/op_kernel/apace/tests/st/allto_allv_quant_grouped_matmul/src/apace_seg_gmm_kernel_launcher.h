/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apace_seg_gmm_kernel_launcher.h
 * \brief segment 模式（rank-major 布局）GMM kernel 直驱入口，用于隔离验证 hcomm 路径的段式计算
 */

#pragma once

#include "apace/kernel/matmul/grouped_matmul/qgmm_mx_kernel.h"

using SegTilingData = Apace::hcommA2avQuantGmmTilingData;

__global__ __aicore__ void ApaceSegGmmKernel_E4M3(__gm__ SegTilingData *tilingGM, GM_ADDR aGM, GM_ADDR aScaleGM,
                                                  GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    using AType = fp8_e4m3fn_t;
    using BType = fp8_e4m3fn_t;
    using CType = bfloat16_t;
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using LayoutScaleA = AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<2>>;
    using LayoutScaleB = AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<2>>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutA, CType, LayoutC,
                                                    float, LayoutBias>;
    using GmmKernel = Apace::Kernel::QuantGroupedMatmulKernel<ProblemShape, BlockMmad, BlockScheduler>;
    __gm__ SegTilingData *td = tilingGM;
    typename GmmKernel::GMMTiling gmm;
    gmm.groupNum = td->segNum;
    gmm.n = td->n;
    gmm.k = td->k;
    gmm.baseM = td->baseM;
    gmm.baseN = td->baseN;
    gmm.baseK = td->baseK;
    gmm.kAL1 = td->kAL1;
    gmm.kBL1 = td->kBL1;
    gmm.scaleKAL1 = td->scaleKAL1;
    gmm.isBias = 0;
    gmm.dbL0C = 2;
    gmm.l1BufferStage = 2;
    typename GmmKernel::Params params;
    params.aBasePtr = reinterpret_cast<__gm__ AType *>(aGM);
    params.scaleABasePtr = reinterpret_cast<__gm__ fp8_e8m0_t *>(aScaleGM);
    params.bBasePtr = reinterpret_cast<__gm__ BType *>(bGM);
    params.scaleBBasePtr = reinterpret_cast<__gm__ fp8_e8m0_t *>(bScaleGM);
    params.cBasePtr = reinterpret_cast<__gm__ CType *>(cGM);
    params.groupList = reinterpret_cast<__gm__ const uint32_t *>(reinterpret_cast<GM_ADDR>(td) +
                                                                 offsetof(SegTilingData, segList));
    params.segAOff = reinterpret_cast<__gm__ const uint32_t *>(reinterpret_cast<GM_ADDR>(td) +
                                                               offsetof(SegTilingData, segAOff));
    params.segCOff = reinterpret_cast<__gm__ const uint32_t *>(reinterpret_cast<GM_ADDR>(td) +
                                                               offsetof(SegTilingData, segCOff));
    params.segW = td->rankSize;
    params.gmm = gmm;
    // 块级调试：每核写 [核id, 组id, mPos, nPos, blockM, blockN, aRowOff, cRowOff]×N 到 dbg 区
    GmmKernel op;
    op(params);
}
