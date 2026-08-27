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

#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "cgmct/prologue/block_prologue_finalize_routing.h"
#include "block_epilogue_finalize_routing_tensor_api.h"
#include "grouped_matmul_finalize_routing_tiling_data.h"
#include "kernel_gmm_finalize_routing_tensor_api.h"

template <typename LayoutA, typename LayoutB>
__aicore__ inline void grouped_matmul_finalize_routing_mx_tensor_api(GM_ADDR x, GM_ADDR w, GM_ADDR wScale, GM_ADDR bias,
                                                                     GM_ADDR xScale, GM_ADDR groupList,
                                                                     GM_ADDR sharedInput, GM_ADDR logit,
                                                                     GM_ADDR rowIndex, GM_ADDR offset, GM_ADDR y,
                                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    (void)offset;
    (void)workspace;
    REGISTER_TILING_DEFAULT(GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingTilingData);
    GET_TILING_DATA(tilingData, tiling);
    const auto &opTiling = tilingData.gmmFinalizeRoutingDataParams;
    const auto &mmTiling = tilingData.matmulTiling;

    using AType = DTYPE_X;
    using BType = DTYPE_W;
    using OutputType = DTYPE_Y;
    using MmadOutputType = float;
    using BiasType = bfloat16_t;
    using LogitType = DTYPE_LOGIT;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGmmFinalizeRoutingTensorApiMixMx>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, MmadOutputType,
                                                    AscendC::Te::NDExtLayoutPtn, BiasType, AscendC::Te::NDExtLayoutPtn>;
    using BlockEpilogue = BlockEpilogueFinalizeRoutingTensorApi<OutputType, MmadOutputType, LogitType>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, Scheduler>;

    if ASCEND_IS_AIV {
        using Prologue = Cgmct::Gemm::Block::BlockPrologueFinalizeRouting<OutputType, BiasType>;
        Prologue prologue;
        typename Prologue::Params prologueParams{
            sharedInput,           y, opTiling.sharedInputOffset, opTiling.sharedInputLen, mmTiling.N, opTiling.batch,
            opTiling.residualScale};
        prologue.Init(prologueParams);
        prologue();
    }
    AscendC::SyncAll();

    const uint32_t kAL1 = static_cast<uint32_t>(mmTiling.stepKa * mmTiling.baseK);
    const uint32_t kBL1 = static_cast<uint32_t>(mmTiling.stepKb * mmTiling.baseK);
    constexpr uint32_t scaleFactorMask = 0xffU;
    constexpr uint32_t scaleFactorBShift = 8U;
    const uint32_t scaleFactorA = mmTiling.mxTypePara & scaleFactorMask;
    const uint32_t scaleFactorB = (mmTiling.mxTypePara >> scaleFactorBShift) & scaleFactorMask;
    const uint64_t scaleKAL1 = static_cast<uint64_t>(scaleFactorA) * kAL1;
    const uint64_t scaleKBL1 = static_cast<uint64_t>(scaleFactorB) * kBL1;
    const uint64_t scaleKCandidate = scaleKAL1 > scaleKBL1 ? scaleKAL1 : scaleKBL1;
    const uint64_t problemK = static_cast<uint64_t>(mmTiling.Ka);
    const uint32_t scaleKL1 = static_cast<uint32_t>(scaleKCandidate < problemK ? scaleKCandidate : problemK);
    typename Kernel::GMMTiling gmmTiling{opTiling.groupNum,
                                         mmTiling.M,
                                         mmTiling.N,
                                         mmTiling.Ka,
                                         static_cast<uint32_t>(mmTiling.baseM),
                                         static_cast<uint32_t>(mmTiling.baseN),
                                         static_cast<uint32_t>(mmTiling.baseK),
                                         kAL1,
                                         kBL1,
                                         scaleKL1,
                                         static_cast<uint8_t>(mmTiling.dbL0C),
                                         opTiling.groupListType,
                                         opTiling.hasBias};
    typename BlockMmad::Params mmParams{x, w, nullptr, bias, xScale, wScale};
    typename BlockEpilogue::Params epilogueParams{y,     wScale,   xScale,         bias,
                                                  logit, rowIndex, mmTiling.baseM, mmTiling.baseN};
    typename Kernel::Params params{
        {gmmTiling.m, gmmTiling.n, gmmTiling.k, 1}, mmParams, epilogueParams, groupList, gmmTiling};
    Kernel kernel;
    kernel(params);
}
