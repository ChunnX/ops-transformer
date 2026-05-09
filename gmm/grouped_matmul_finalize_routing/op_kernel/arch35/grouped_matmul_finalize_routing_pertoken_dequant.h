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
#include "gmm_fr_deterministic_a5.h"
#include "grouped_matmul_finalize_routing_tiling_data.h"

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
    // scaleType==2 means bf16
    using weightscaleType = std::conditional_t<scaleType == 2, bfloat16_t, float>;
    using BiasType = bfloat16_t;
    using LayoutBias = layout::RowMajor;
    using C1Type = std::conditional_t<std::is_same_v<AType, int8_t>, int32_t, float>;
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

    // Deterministic: sequential write by mOffset, no scatter/AtomicAdd
    using BlockEpilogueDequantSeq =
        Cgmct::Gemm::Block::BlockEpilogueDequantSequentialWrite<CType, C1Type, weightscaleType, xscaleType, BiasType,
                                                                 rowIndexType>;

    using GmmKernel =
        Cgmct::Gemm::Kernel::KernelGmmFinalizeRoutingPertokenDequant<ProblemShape, BlockMmadBuilder, BlockPrologue,
                                                                     BlockEpilogueDequant, BlockScheduler>;

    // Deterministic kernel variant using SequentialWrite epilogue
    using GmmKernelDeterministic =
        Cgmct::Gemm::Kernel::KernelGmmFinalizeRoutingPertokenDequant<ProblemShape, BlockMmadBuilder, BlockPrologue,
                                                                      BlockEpilogueDequantSeq, BlockScheduler>;

    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;
    using ParamsDeterministic = typename GmmKernelDeterministic::Params;

    GMMTiling gmmParams{gmmFinalizeRoutingQuantParams_.groupNum,
                        gmmFinalizeRoutingQuantParams_.groupListType,
                        matmulTiling_.baseM,
                        matmulTiling_.baseN,
                        matmulTiling_.baseK,
                        gmmFinalizeRoutingQuantParams_.hasBias};

    gmmParams.matmulTiling = &matmulTiling_;

    // Deterministic constants (consistent with A3)
    constexpr uint32_t DETER_UB_SIZE = 12 * 1024;
    constexpr uint32_t BUFFER_NUM = 2;

    if (gmmFinalizeRoutingQuantParams_.deterministicFlag == 1) {
        // Deterministic mode (W8A8 INT8 only):
        //   Phase 1: Prologue zeros yGm + writes residual -> yGm
        //   Phase 2: SequentialWrite epilogue writes dequant -> workspace (no race)
        //   Phase 3: FRDeterministicA5 aggregates workspace -> yGm via AtomicAdd (one writer per outRow)

        constexpr uint64_t SYS_WORKSPACE_SIZE = 16UL * 1024 * 1024;
        GM_ADDR deterBuffer = workspaceGM + SYS_WORKSPACE_SIZE;

        // Must use GmmKernelDeterministic's own GMMTiling type (different kernel instantiations have incompatible nested types)
        using GMMTilingDeterministic = typename GmmKernelDeterministic::GMMTiling;

        GMMTilingDeterministic gmmParamsDeter{gmmFinalizeRoutingQuantParams_.groupNum,
                                              gmmFinalizeRoutingQuantParams_.groupListType,
                                              matmulTiling_.baseM,
                                              matmulTiling_.baseN,
                                              matmulTiling_.baseK,
                                              gmmFinalizeRoutingQuantParams_.hasBias};

        gmmParamsDeter.matmulTiling = &matmulTiling_;

        ParamsDeterministic params = {
            {1, 1, 1, 1},
            {x, w, deterBuffer, bias, group_list},              // BlockMmad: c -> workspace
            {share_input, y,                                     // Prologue: zeros + residual -> yGm
             gmmFinalizeRoutingQuantParams_.sharedInputOffset,
             gmmFinalizeRoutingQuantParams_.sharedInputLen, matmulTiling_.N,
             gmmFinalizeRoutingQuantParams_.batch,
             gmmFinalizeRoutingQuantParams_.residualScale},
            {deterBuffer, w_scale, x_scale, bias, logit, row_index,  // Epilogue: dequant -> workspace
             matmulTiling_.baseM, matmulTiling_.baseN},
            gmmParamsDeter};

        GmmKernelDeterministic gmm;
        gmm(params);

        // Aggregation: workspace -> yGm via rowIndex mapping + AtomicAdd
        GlobalTensor<float> deterBufferGm;
        deterBufferGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(deterBuffer));
        GlobalTensor<float> yGm;
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        GlobalTensor<rowIndexType> rowIndexGm;
        rowIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ rowIndexType*>(row_index));

        TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queBind;
        TPipe deterPipe;
        deterPipe.InitBuffer(queBind, BUFFER_NUM, DETER_UB_SIZE);

        uint32_t coreNumVec = matmulTiling_.usedCoreNum * GetTaskRation();
        uint64_t nTimes = Ceil(matmulTiling_.N, DETER_UB_SIZE / sizeof(float));
        uint64_t baseN = Ceil(Ceil(matmulTiling_.N, nTimes), 128) * 128;

        // totalM = total input rows across all groups
        uint64_t totalM = static_cast<uint64_t>(matmulTiling_.M);

        GMMFRDeterministic::SyncConfig syncConfig{};
        syncConfig.curM = totalM;
        syncConfig.lowBoundM = totalM;
        syncConfig.windowSize = totalM;
        syncConfig.baseN = baseN;

        GMMFRDeterministic::FRDeterministicA5<float, rowIndexType>(
            syncConfig, deterBufferGm, yGm, rowIndexGm, queBind,
            matmulTiling_.usedCoreNum, static_cast<uint32_t>(matmulTiling_.N));

    } else {
        // Non-deterministic mode
        Params params = {
            {1, 1, 1, 1},
            {x, w, y, bias, group_list},
            {share_input, y, gmmFinalizeRoutingQuantParams_.sharedInputOffset,
             gmmFinalizeRoutingQuantParams_.sharedInputLen, matmulTiling_.N, gmmFinalizeRoutingQuantParams_.batch,
             gmmFinalizeRoutingQuantParams_.residualScale},
            {y, w_scale, x_scale, bias, logit, row_index, matmulTiling_.baseM, matmulTiling_.baseN},
            gmmParams};
        GmmKernel gmm;
        gmm(params);
    }
}
#endif