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
        // Deterministic per-group mode (W8A8 INT8 only):
        //   Process each group individually (groupNum=1 per kernel call).
        //   This avoids cross-group state issues in the Cgmct kernel.
        //   Phase 1 (first group): Prologue zeros yGm + writes residual -> yGm
        //   Phase 2: SequentialWrite epilogue writes dequant -> workspace (no race)
        //   Phase 3: FRDeterministicA5 aggregates workspace -> yGm via AtomicAdd
        //   Repeat phases 2-3 for subsequent groups (prologue skipped).

        constexpr uint64_t SYS_WORKSPACE_SIZE = 16UL * 1024 * 1024;
        GM_ADDR deterBuffer = workspaceGM + SYS_WORKSPACE_SIZE;

        uint32_t groupNum = gmmFinalizeRoutingQuantParams_.groupNum;
        uint8_t groupListType = gmmFinalizeRoutingQuantParams_.groupListType;
        int32_t n = matmulTiling_.N;
        int32_t k = matmulTiling_.Ka;
        constexpr bool isTransB = BlockMmadBuilder::transB;

        // NZ format: single group weight size
        constexpr uint32_t WEIGHT_TILE_K_LARGE = 32U;
        constexpr uint32_t WEIGHT_TILE_N_LARGE = 32U;
        constexpr uint32_t WEIGHT_TILE_K_SMALL = 16U;
        constexpr uint32_t WEIGHT_TILE_N_SMALL = 16U;
        constexpr uint32_t WEIGHT_TILE_CAPACITY = 512U;
        uint64_t singleGroupWeightNZ = 0;
        if constexpr (BlockMmadBuilder::formatB == CubeFormat::NZ) {
            if (isTransB) {
                singleGroupWeightNZ = CeilDiv(static_cast<uint64_t>(k), WEIGHT_TILE_K_LARGE)
                                    * CeilDiv(static_cast<uint64_t>(n), WEIGHT_TILE_N_SMALL)
                                    * WEIGHT_TILE_CAPACITY;
            } else {
                singleGroupWeightNZ = CeilDiv(static_cast<uint64_t>(n), WEIGHT_TILE_N_LARGE)
                                    * CeilDiv(static_cast<uint64_t>(k), WEIGHT_TILE_K_SMALL)
                                    * WEIGHT_TILE_CAPACITY;
            }
        }

        // Prepare global tensors
        GlobalTensor<rowIndexType> rowIndexGm;
        rowIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ rowIndexType*>(row_index));
        GlobalTensor<int64_t> groupListFullGm;
        groupListFullGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list));
        GlobalTensor<float> deterBufferGm;
        deterBufferGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(deterBuffer));
        GlobalTensor<float> yGm;
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

        TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queBind;
        TPipe deterPipe;
        if ASCEND_IS_AIV {
            deterPipe.InitBuffer(queBind, BUFFER_NUM, DETER_UB_SIZE);
        }

        uint64_t nTimes = Ceil(n, DETER_UB_SIZE / sizeof(float));
        uint64_t baseN = Ceil(Ceil(n, nTimes), 128) * 128;

        using GMMTilingDeterministic = typename GmmKernelDeterministic::GMMTiling;

        uint64_t globalRowOffset = 0;
        uint64_t xRowOffset = 0;
        uint64_t wGroupOffset = 0;
        bool isFirstGroup = true;

        // Per-group loop: each group processed as a separate kernel invocation (groupNum=1)
        for (uint32_t gi = 0; gi < groupNum; gi++) {
            // Get this group's M from groupList
            int32_t groupM = 0;
            int32_t preOffsetInit = 0;
            if (groupListType == 1) {
                groupM = static_cast<int32_t>(groupListFullGm.GetValue(gi));
            } else {
                // cumulative mode
                int32_t val = static_cast<int32_t>(groupListFullGm.GetValue(gi));
                if (gi == 0) {
                    groupM = val;
                } else {
                    int32_t prev = static_cast<int32_t>(groupListFullGm.GetValue(gi - 1));
                    groupM = val - prev;
                    preOffsetInit = prev;
                }
            }

            // Skip groups with M=0 but maintain SyncAll count
            if (groupM == 0) {
                if (singleGroupWeightNZ > 0) {
                    wGroupOffset += singleGroupWeightNZ;
                } else {
                    wGroupOffset += static_cast<uint64_t>(n) * k;
                }
                SyncAll();
                GMMFRDeterministic::SyncConfig emptyConfig{};
                emptyConfig.curM = 0;
                emptyConfig.baseN = baseN;
                GMMFRDeterministic::FRDeterministicA5<float, rowIndexType>(
                    emptyConfig, deterBufferGm, yGm, rowIndexGm, queBind,
                    matmulTiling_.usedCoreNum, static_cast<uint32_t>(n), globalRowOffset);
                continue;
            }

            // Per-group matmulTiling: override M to this group's M
            TCubeTiling singleGroupTiling = matmulTiling_;
            singleGroupTiling.M = static_cast<uint64_t>(groupM);

            GMMTilingDeterministic gmmParamsSingle{1, groupListType,
                                                   matmulTiling_.baseM, matmulTiling_.baseN, matmulTiling_.baseK,
                                                   gmmFinalizeRoutingQuantParams_.hasBias};
            gmmParamsSingle.matmulTiling = &singleGroupTiling;
            gmmParamsSingle.preOffsetInit = preOffsetInit;

            // Per-group addresses
            GM_ADDR xGroup = x + xRowOffset * k;
            GM_ADDR wGroup = w + wGroupOffset;
            GM_ADDR biasGroup = bias + static_cast<uint64_t>(gi) * n * sizeof(bfloat16_t);
            GM_ADDR groupListSingle = group_list + gi * sizeof(int64_t);
            GM_ADDR wScaleGroup = w_scale + static_cast<uint64_t>(gi) * n * sizeof(float);
            GM_ADDR xScaleGroup = x_scale + xRowOffset * sizeof(float);
            GM_ADDR logitGroup = logit + xRowOffset * sizeof(float);

            uint32_t prologueBatch = isFirstGroup ? gmmFinalizeRoutingQuantParams_.batch : 0;
            uint32_t prologueSharedInputOffset = isFirstGroup ? gmmFinalizeRoutingQuantParams_.sharedInputOffset : 0;
            uint32_t prologueSharedInputLen = isFirstGroup ? gmmFinalizeRoutingQuantParams_.sharedInputLen : 0;

            ParamsDeterministic singleParams = {
                {1, 1, 1, 1},
                {xGroup, wGroup, deterBuffer, biasGroup, groupListSingle},
                {share_input, y,
                 prologueSharedInputOffset,
                 prologueSharedInputLen, n,
                 prologueBatch,
                 gmmFinalizeRoutingQuantParams_.residualScale},
                {deterBuffer, wScaleGroup, xScaleGroup, biasGroup, logitGroup, row_index,
                 matmulTiling_.baseM, matmulTiling_.baseN},
                gmmParamsSingle};

            GmmKernelDeterministic gmm;
            gmm(singleParams);
            SyncAll();

            // Aggregate this group's workspace data -> yGm
            GMMFRDeterministic::SyncConfig syncConfig{};
            syncConfig.curM = static_cast<uint64_t>(groupM);
            syncConfig.baseN = baseN;

            GMMFRDeterministic::FRDeterministicA5<float, rowIndexType>(
                syncConfig, deterBufferGm, yGm, rowIndexGm, queBind,
                matmulTiling_.usedCoreNum, static_cast<uint32_t>(n), globalRowOffset);

            // Update offsets
            globalRowOffset += static_cast<uint64_t>(groupM);
            xRowOffset += static_cast<uint64_t>(groupM);
            if (singleGroupWeightNZ > 0) {
                wGroupOffset += singleGroupWeightNZ;
            } else {
                wGroupOffset += static_cast<uint64_t>(n) * k;
            }
            isFirstGroup = false;
        }

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