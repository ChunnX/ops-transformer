/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_finalize_routing_pertoken_dequant.h
 * \brief
 */

#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_PERTOKEN_DEQUANT_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_PERTOKEN_DEQUANT_H

#include "cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h"
#include "cgmct/block/block_mmad_builder.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "cgmct/epilogue/block_epilogue_dequant_sequential_write.h"
#include "grouped_matmul_finalize_routing_tiling_data.h"
#include "gmm_fr_deterministic_a5.h"
using namespace GMMFRDeterministic;

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;

template <typename layoutA, typename layoutB, int scaleType, int rowIndType>
__aicore__ inline void grouped_matmul_finalize_routing_pertoken_dequant(
    GM_ADDR x, GM_ADDR w, GM_ADDR w_scale, GM_ADDR bias, GM_ADDR x_scale, GM_ADDR group_list, GM_ADDR share_input,
    GM_ADDR logit, GM_ADDR row_index, GM_ADDR offset, GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    auto gmmFinalizeRoutingQuantParams_ = tilingData.gmmFinalizeRoutingDataParams;
    auto matmulTiling_ = tilingData.matmulTiling;

    using L1TileShape = AscendC::Shape<Cgmct::Gemm::_0, Cgmct::Gemm::_0, Cgmct::Gemm::_0>;
    using L0TileShape = AscendC::Shape<Cgmct::Gemm::_0, Cgmct::Gemm::_0, Cgmct::Gemm::_0>;

    using AType = DTYPE_X;
    using BType = DTYPE_W;
    using CType = DTYPE_Y;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajorAlign;
    // 2 represents bf16 dtype
    using weightscaleType = std::conditional_t<scaleType == 2, bfloat16_t, float>;
    using BiasType = bfloat16_t;
    using LayoutBias = layout::RowMajor;
    using C1Type = std::conditional_t<std::is_same_v<AType, int8_t>, int32_t, float>; // matmul output dtype
    using xscaleType = float;
    using rowIndexType = std::conditional_t<rowIndType == 1, int32_t, int64_t>;
    using ProblemShape = Cgmct::Gemm::MatmulShape;
    using BlockScheduler = Cgmct::Gemm::GroupedMatmulAswtWithTailSplitScheduler;
    using BlockMmadBuilder =
        Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, C1Type, LayoutC, BiasType, LayoutBias, L1TileShape,
                                L0TileShape, BlockScheduler, MatmulMultiBlock<>,
                                Tile::TileCopy<Arch::DAV3510, Tile::CopyInAndCopyOutSplitMWithParams>>;

    using BlockPrologue = Cgmct::Gemm::Block::BlockPrologueFinalizeRouting<CType, BiasType>;

    using BlockEpilogueDequant =
        Cgmct::Gemm::Block::BlockEpilogueDequantFinalizeRouting<CType, C1Type, weightscaleType, xscaleType, BiasType,
                                                                rowIndexType>;

    using GmmKernel =
        Cgmct::Gemm::Kernel::KernelGmmFinalizeRoutingPertokenDequant<ProblemShape, BlockMmadBuilder, BlockPrologue,
                                                                     BlockEpilogueDequant, BlockScheduler>;
    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;

    GMMTiling gmmParams{gmmFinalizeRoutingQuantParams_.groupNum,
                        gmmFinalizeRoutingQuantParams_.groupListType,
                        matmulTiling_.baseM,
                        matmulTiling_.baseN,
                        matmulTiling_.baseK,
                        gmmFinalizeRoutingQuantParams_.hasBias};

    gmmParams.matmulTiling = &matmulTiling_;

    if (gmmFinalizeRoutingQuantParams_.deterministicFlag == 1) {
        // ============ Deterministic mode (W8A8 INT8 only) ============
        // Plan B: Use BlockEpilogueDequantSequentialWrite (no scatter, no AtomicAdd)
        // Epilogue writes sequentially to workspace by input row index.
        // After Cgmct Kernel completes, call FRDeterministicA5 to aggregate.

        // Deterministic buffer starts after Cgmct system workspace (16MB).
        // Cgmct Cube results go to UB, NOT to workspace (unlike A3).
        constexpr uint64_t SYS_WORKSPACE_SIZE = 16UL * 1024 * 1024;
        GM_ADDR deterBuffer = workspaceGM + SYS_WORKSPACE_SIZE;

        // Define deterministic kernel with sequential-write Epilogue
        using BlockEpilogueDeter =
            Cgmct::Gemm::Block::BlockEpilogueDequantSequentialWrite<CType, C1Type, weightscaleType, xscaleType,
                                                                     BiasType, rowIndexType>;
        using GmmKernelDeter =
            Cgmct::Gemm::Kernel::KernelGmmFinalizeRoutingPertokenDequant<ProblemShape, BlockMmadBuilder, BlockPrologue,
                                                                          BlockEpilogueDeter, BlockScheduler>;
        using ParamsDeter = typename GmmKernelDeter::Params;
        using GMMTilingDeter = typename GmmKernelDeter::GMMTiling;

        GMMTilingDeter gmmParamsDeter{gmmFinalizeRoutingQuantParams_.groupNum,
                                       gmmFinalizeRoutingQuantParams_.groupListType,
                                       matmulTiling_.baseM,
                                       matmulTiling_.baseN,
                                       matmulTiling_.baseK,
                                       gmmFinalizeRoutingQuantParams_.hasBias};
        gmmParamsDeter.matmulTiling = &matmulTiling_;

        ParamsDeter params = {
            {1, 1, 1, 1},
            {x, w, deterBuffer, bias, group_list},              // y -> workspace
            {share_input, deterBuffer,                           // prologue y -> workspace
             gmmFinalizeRoutingQuantParams_.sharedInputOffset,
             gmmFinalizeRoutingQuantParams_.sharedInputLen, matmulTiling_.N,
             gmmFinalizeRoutingQuantParams_.batch,
             gmmFinalizeRoutingQuantParams_.residualScale},
            {deterBuffer, w_scale, x_scale, bias, logit, row_index,  // epilogue y -> workspace (sequential write)
             matmulTiling_.baseM, matmulTiling_.baseN},
            gmmParamsDeter};

        // Execute Cgmct Kernel with sequential-write Epilogue (output to workspace)
        GmmKernelDeter gmmDeter;
        gmmDeter(params);

        // ============ Deterministic aggregation: workspace -> yGm ============
        GlobalTensor<float> deterBufferGm;
        deterBufferGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(deterBuffer));
        GlobalTensor<float> yGm;
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        GlobalTensor<rowIndexType> tokenRanksGm;
        tokenRanksGm.SetGlobalBuffer(reinterpret_cast<__gm__ rowIndexType*>(row_index));

        // Init SyncConfig
        SyncConfig syncConfig;
        syncConfig.windowSize = gmmFinalizeRoutingQuantParams_.deterWorkspaceSize /
                                (matmulTiling_.N * sizeof(float));
        syncConfig.lowBoundM = syncConfig.windowSize;
        uint64_t nTimes = Ceil(matmulTiling_.N, DETER_UB_SIZE / sizeof(float));
        syncConfig.baseN = Ceil(Ceil(matmulTiling_.N, nTimes), 128) * 128;

        // Init queBind (deterministic UB transfer buffer)
        TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queBind;
        TPipe deterPipe;
        deterPipe.InitBuffer(queBind, BUFFER_NUM, DETER_UB_SIZE);

        // Final aggregation: set curM to total rows, process all at once
        syncConfig.curM = matmulTiling_.M;
        FRDeterministicA5<float, rowIndexType>(
            syncConfig, deterBufferGm, yGm, tokenRanksGm, queBind,
            matmulTiling_.usedCoreNum, matmulTiling_.N);

    } else {
        // ============ Non-deterministic mode (unchanged) ============
        Params params = {
            {1, 1, 1, 1},                // problem shape
            {x, w, y, bias, group_list}, // BlockMmadParams
            {share_input, y, gmmFinalizeRoutingQuantParams_.sharedInputOffset,
             gmmFinalizeRoutingQuantParams_.sharedInputLen, matmulTiling_.N, gmmFinalizeRoutingQuantParams_.batch,
             gmmFinalizeRoutingQuantParams_.residualScale},                                          // prologue params
            {y, w_scale, x_scale, bias, logit, row_index, matmulTiling_.baseM, matmulTiling_.baseN}, // epilogue params
            gmmParams};
        GmmKernel gmm;
        gmm(params);
    }
}
#endif