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
    constexpr uint32_t DETER_UB_SIZE = 8 * 1024;
    constexpr uint32_t BUFFER_NUM = 2;

    if (gmmFinalizeRoutingQuantParams_.deterministicFlag == 1) {
        // === Deterministic mode: sliding window multi-round loop ===
        constexpr uint64_t SYS_WORKSPACE_SIZE = 16UL * 1024 * 1024;
        GM_ADDR deterBuffer = workspaceGM + SYS_WORKSPACE_SIZE;

        uint32_t windowSize = gmmFinalizeRoutingQuantParams_.windowSize;
        uint32_t groupNum = gmmFinalizeRoutingQuantParams_.groupNum;
        uint8_t groupListType = gmmFinalizeRoutingQuantParams_.groupListType;
        uint32_t n = static_cast<uint32_t>(matmulTiling_.N);
        uint64_t k = static_cast<uint64_t>(matmulTiling_.Ka);

        // Read groupList for window grouping calculation
        GlobalTensor<int64_t> groupListGm;
        groupListGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list));

        // Prepare GlobalTensors for aggregation
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
        uint64_t nTimes = Ceil(n, DETER_UB_SIZE / sizeof(float));
        uint64_t baseN = Ceil(Ceil(n, nTimes), 128) * 128;

        // Helper: get M value for group[idx]
        auto getGroupM = [&](uint32_t idx) -> uint32_t {
            int64_t val = groupListGm.GetValue(idx);
            if (groupListType == 0) {
                int64_t prev = (idx > 0) ? groupListGm.GetValue(idx - 1) : 0;
                return static_cast<uint32_t>(val - prev);
            }
            return static_cast<uint32_t>(val);
        };

        // NZ format weight offset constant (consistent with Cgmct Kernel UpdateOffset)
        constexpr uint32_t WEIGHT_TILE_N_LARGE = 32;
        constexpr uint32_t WEIGHT_TILE_K_SMALL = 16;
        constexpr uint32_t WEIGHT_TILE_CAPACITY = 512;
        uint64_t singleGroupWeightNZ = CeilDiv(n, WEIGHT_TILE_N_LARGE) *
                                       CeilDiv(k, WEIGHT_TILE_K_SMALL) *
                                       WEIGHT_TILE_CAPACITY;

        // === Sliding window loop ===
        uint32_t globalRowOffset = 0;
        uint32_t groupStart = 0;
        uint64_t xRowOffset = 0;
        uint64_t wGroupOffset = 0;
        bool isFirstRound = true;

        // Must use GmmKernelDeterministic's own GMMTiling type (different kernel instantiations have incompatible nested types)
        using GMMTilingDeterministic = typename GmmKernelDeterministic::GMMTiling;

        while (groupStart < groupNum) {
            // 1. Determine this round's group range [groupStart, groupEnd)
            uint32_t roundM = 0;
            uint32_t groupEnd = groupStart;
            while (groupEnd < groupNum) {
                uint32_t mi = getGroupM(groupEnd);
                if (mi == 0) { groupEnd++; continue; }
                if (roundM + mi > windowSize && roundM > 0) {
                    break;
                }
                roundM += mi;
                groupEnd++;
            }
            if (roundM == 0) {
                groupStart = groupEnd;
                continue;
            }

            uint32_t roundGroupNum = groupEnd - groupStart;

            // 2. Compute preOffsetInit for cumulative mode
            int32_t preOffsetInit = 0;
            if (groupListType == 0 && groupStart > 0) {
                preOffsetInit = static_cast<int32_t>(groupListGm.GetValue(groupStart - 1));
            }

            // 3. Construct offset input pointers
            GM_ADDR xRound = x + xRowOffset * k;
            GM_ADDR wRound = w + wGroupOffset * singleGroupWeightNZ;
            GM_ADDR groupListRound = group_list + groupStart * sizeof(int64_t);
            GM_ADDR wScaleRound = w_scale + wGroupOffset * n * sizeof(weightscaleType);
            GM_ADDR xScaleRound = x_scale + xRowOffset * sizeof(float);
            GM_ADDR biasRound = bias + wGroupOffset * n * sizeof(bfloat16_t);
            GM_ADDR logitRound = logit + xRowOffset * sizeof(float);
            GM_ADDR rowIndexRound = row_index + xRowOffset * sizeof(rowIndexType);

            // 4. Prologue parameter: first round normal init, subsequent rounds skip (batch=0)
            uint32_t prologueBatch = isFirstRound ?
                gmmFinalizeRoutingQuantParams_.batch : 0;
            uint32_t prologueSharedInputOffset = isFirstRound ?
                gmmFinalizeRoutingQuantParams_.sharedInputOffset : 0;
            uint32_t prologueSharedInputLen = isFirstRound ?
                gmmFinalizeRoutingQuantParams_.sharedInputLen : 0;

            // 5. Construct GMMTiling and Kernel parameters
            GMMTilingDeterministic gmmParamsRound{
                roundGroupNum, groupListType,
                matmulTiling_.baseM, matmulTiling_.baseN, matmulTiling_.baseK,
                gmmFinalizeRoutingQuantParams_.hasBias,
                preOffsetInit
            };
            gmmParamsRound.matmulTiling = &matmulTiling_;

            ParamsDeterministic params = {
                {1, 1, 1, 1},
                {xRound, wRound, deterBuffer, biasRound, groupListRound},
                {share_input, y,
                 prologueSharedInputOffset,
                 prologueSharedInputLen, n,
                 prologueBatch,
                 gmmFinalizeRoutingQuantParams_.residualScale},
                {deterBuffer, wScaleRound, xScaleRound, biasRound, logitRound, rowIndexRound,
                 matmulTiling_.baseM, matmulTiling_.baseN},
                gmmParamsRound
            };

            // 6. Run this round's Kernel (Prologue + Cube MMAD + Vector Epilogue)
            GmmKernelDeterministic gmm;
            gmm(params);

            // 7. Window aggregation: workspace -> yGm
            GMMFRDeterministic::SyncConfig syncConfig{};
            syncConfig.curM = roundM;
            syncConfig.lowBoundM = roundM;
            syncConfig.windowSize = roundM;
            syncConfig.baseN = baseN;

            GMMFRDeterministic::FRDeterministicA5<float, rowIndexType>(
                syncConfig, deterBufferGm, yGm, rowIndexGm, queBind,
                matmulTiling_.usedCoreNum, n, globalRowOffset);

            // 8. Update offsets
            globalRowOffset += roundM;
            xRowOffset += roundM;
            wGroupOffset += roundGroupNum;
            groupStart = groupEnd;
            isFirstRound = false;
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