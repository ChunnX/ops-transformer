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
        // Deterministic sliding-window mode (W8A8 INT8 only):
        //   Splits groups into rounds where each round fits in workspace.
        //   Phase 1 (first round): Prologue zeros yGm + writes residual -> yGm
        //   Phase 2: SequentialWrite epilogue writes dequant -> workspace (no race)
        //   Phase 3: FRDeterministicA5 aggregates workspace -> yGm via AtomicAdd
        //   Repeat phases 2-3 for subsequent rounds (prologue skipped).

        constexpr uint64_t SYS_WORKSPACE_SIZE = 16UL * 1024 * 1024;
        GM_ADDR deterBuffer = workspaceGM + SYS_WORKSPACE_SIZE;

        uint32_t windowSize = gmmFinalizeRoutingQuantParams_.windowSize;
        uint32_t totalM = gmmFinalizeRoutingQuantParams_.totalM;
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

        // Prepare global tensors for rowIndex and groupList
        GlobalTensor<rowIndexType> rowIndexGm;
        rowIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ rowIndexType*>(row_index));
        GlobalTensor<int64_t> groupListFullGm;
        groupListFullGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list));

        // helper: get group M for group at local index gi within current round
        // groupListType==0: cumulative mode; groupListType==1: incremental mode
        auto getGroupM = [&](uint32_t gi, uint32_t groupStart, int32_t preOffsetInitForGetGroupM) -> int32_t {
            if (groupListType == 1) {
                return static_cast<int32_t>(groupListFullGm.GetValue(groupStart + gi));
            }
            // cumulative mode
            int32_t val = static_cast<int32_t>(groupListFullGm.GetValue(groupStart + gi));
            if (gi == 0) {
                return val - preOffsetInitForGetGroupM;
            }
            int32_t prev = static_cast<int32_t>(groupListFullGm.GetValue(groupStart + gi - 1));
            return val - prev;
        };

        // Sliding window loop
        using GMMTilingDeterministic = typename GmmKernelDeterministic::GMMTiling;

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

        uint32_t groupStart = 0;
        uint64_t globalRowOffset = 0;
        uint64_t xRowOffset = 0;
        uint64_t wGroupOffset = 0;
        bool isFirstRound = true;

        while (groupStart < groupNum) {
            // Determine round group range [groupStart, groupEnd)
            int32_t preOffsetInitForGetGroupM = 0;
            if (groupListType == 0 && groupStart > 0) {
                preOffsetInitForGetGroupM = static_cast<int32_t>(groupListFullGm.GetValue(groupStart - 1));
            }

            int32_t roundM = 0;
            uint32_t groupEnd = groupStart;
            while (groupEnd < groupNum) {
                int32_t gm = getGroupM(groupEnd - groupStart, groupStart, preOffsetInitForGetGroupM);
                if (roundM + gm > static_cast<int32_t>(windowSize) && roundM > 0) {
                    break;
                }
                roundM += gm;
                groupEnd++;
                // single group exceeds window: let it occupy a round alone
                if (roundM >= static_cast<int32_t>(windowSize)) {
                    break;
                }
            }

            // Handle groups with M=0: skip kernel/aggregation but advance weight offset
            if (roundM == 0) {
                if (singleGroupWeightNZ > 0) {
                    wGroupOffset += static_cast<uint64_t>(groupEnd - groupStart) * singleGroupWeightNZ;
                } else {
                    wGroupOffset += static_cast<uint64_t>(groupEnd - groupStart) * n * k;
                }
                groupStart = groupEnd;

                // A5: empty round must execute same SyncAll count as normal round (4 total)
                SyncAll();  // Match gmm() internal SyncAll<false>()
                SyncAll();  // Match explicit SyncAll() at aggregation boundary

                GMMFRDeterministic::SyncConfig emptyConfig{};
                emptyConfig.curM = 0;
                emptyConfig.baseN = baseN;
                GMMFRDeterministic::FRDeterministicA5<float, rowIndexType>(
                    emptyConfig, deterBufferGm, yGm, rowIndexGm, queBind,
                    matmulTiling_.usedCoreNum, static_cast<uint32_t>(n), globalRowOffset);
                continue;
            }

            uint32_t roundGroupNum = groupEnd - groupStart;

            // preOffsetInit for cumulative mode kernel
            int32_t preOffsetInit = 0;
            if (groupListType == 0 && groupStart > 0) {
                preOffsetInit = static_cast<int32_t>(groupListFullGm.GetValue(groupStart - 1));
            }

            // Construct offset input pointers
            GM_ADDR xRound = x + xRowOffset * k;
            GM_ADDR wRound = w + wGroupOffset;
            // bias: per-group stride N elements (bfloat16), kernel internally does += n per group
            GM_ADDR biasRound = bias + static_cast<uint64_t>(groupStart) * n * sizeof(bfloat16_t);
            GM_ADDR groupListRound = group_list + groupStart * sizeof(int64_t);
            // w_scale is per-channel: stride N per group; kernel reads w_scale[groupIdx * n]
            // Pre-offset so groupIdx=0 in kernel reads from the correct group
            uint64_t wScaleOffsetElements = static_cast<uint64_t>(groupStart) * n;
            GM_ADDR wScaleRound = w_scale + wScaleOffsetElements * sizeof(float);
            // x_scale is per-token (float): kernel internally does += m per group
            GM_ADDR xScaleRound = x_scale + xRowOffset * sizeof(float);
            // logit is per-token (float): kernel internally does += m per group
            GM_ADDR logitRound = logit + xRowOffset * sizeof(float);

            // GMMTiling for this round
            GMMTilingDeterministic gmmParamsDeter{roundGroupNum,
                                                   groupListType,
                                                   matmulTiling_.baseM,
                                                   matmulTiling_.baseN,
                                                   matmulTiling_.baseK,
                                                   gmmFinalizeRoutingQuantParams_.hasBias};
            gmmParamsDeter.matmulTiling = &matmulTiling_;
            gmmParamsDeter.preOffsetInit = preOffsetInit;

            // Prologue params: first round normal, subsequent rounds skip residual init
            uint32_t prologueBatch = isFirstRound ? gmmFinalizeRoutingQuantParams_.batch : 0;
            uint32_t prologueSharedInputOffset = isFirstRound ? gmmFinalizeRoutingQuantParams_.sharedInputOffset : 0;
            uint32_t prologueSharedInputLen = isFirstRound ? gmmFinalizeRoutingQuantParams_.sharedInputLen : 0;

            ParamsDeterministic params = {
                {1, 1, 1, 1},
                {xRound, wRound, deterBuffer, biasRound, groupListRound},
                {share_input, y,
                 prologueSharedInputOffset,
                 prologueSharedInputLen, n,
                 prologueBatch,
                 gmmFinalizeRoutingQuantParams_.residualScale},
                {deterBuffer, wScaleRound, xScaleRound, biasRound, logitRound, row_index,
                 matmulTiling_.baseM, matmulTiling_.baseN},
                gmmParamsDeter};

            GmmKernelDeterministic gmm;
            gmm(params);
            SyncAll();

            // Window aggregation: workspace -> yGm
            GMMFRDeterministic::SyncConfig syncConfig{};
            syncConfig.curM = static_cast<uint64_t>(roundM);
            syncConfig.baseN = baseN;

            GMMFRDeterministic::FRDeterministicA5<float, rowIndexType>(
                syncConfig, deterBufferGm, yGm, rowIndexGm, queBind,
                matmulTiling_.usedCoreNum, static_cast<uint32_t>(n), globalRowOffset);

            // Update offsets
            globalRowOffset += static_cast<uint64_t>(roundM);
            xRowOffset += static_cast<uint64_t>(roundM);
            if (singleGroupWeightNZ > 0) {
                wGroupOffset += static_cast<uint64_t>(roundGroupNum) * singleGroupWeightNZ;
            } else {
                wGroupOffset += static_cast<uint64_t>(roundGroupNum) * n * k;
            }
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