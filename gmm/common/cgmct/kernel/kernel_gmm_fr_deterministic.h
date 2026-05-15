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
 * \file kernel_gmm_fr_deterministic.h
 * \brief Workspace cycling kernel for A5 GMMFR deterministic mode.
 *
 * Based on: common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h
 * Key change: tile-level workspace flow control with baseM granularity.
 *   Each sub-chunk is at most baseM rows (one matmul M-tile).
 *   Before each sub-chunk: if workspace can't fit one more tile, aggregate first.
 *   MTE3 DMA drain before aggregation to avoid hardware data hazard.
 *
 * Why sub-chunk cycling is SyncAll-safe:
 *   SyncAll is a global barrier requiring ALL cores to call it simultaneously.
 *   workspaceBaseRow and windowSizeRows are identical across all cores
 *   (deterministic from tiling data and accumulated identically), so the
 *   "workspace full" decision is uniform -> all cores reach SyncAll together.
 *
 * Init order (v5): Uses raw-offset LocalTensor (aggVecout_) at VECOUT offset 200KB
 *   instead of TPipe-managed TQueBind, to avoid UB memory overlap with the epilogue's
 *   l0cOutUbFloat_ buffer (128KB at offset 0). The epilogue uses raw-offset tensors
 *   starting from UB 0, so TPipe InitBuffer at offset 0 would corrupt epilogue data.
 */

#ifndef KERNEL_GMM_FR_DETERMINISTIC_H
#define KERNEL_GMM_FR_DETERMINISTIC_H
#include "kernel_basic_intf.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/status_utils.h"

#include "../block/block_mmad_builder.h"
#include "../epilogue/block_epilogue_dequant_finalize_routing.h"
#include "../prologue/block_prologue_finalize_routing.h"
#include "./semaphore.h"

#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_gmm_aswt_with_tail_split.h"

#include "../epilogue/block_epilogue_dequant_sequential_write_deterministic.h"

namespace Cgmct {
namespace Gemm {
namespace Kernel {

namespace {
constexpr uint64_t DET_IDX_A_OFFSETS = 0UL;
constexpr uint64_t DET_IDX_B_OFFSETS = 1UL;
constexpr uint64_t DET_IDX_X1SCALE_OFFSETS = 2UL;
constexpr uint64_t DET_IDX_X2SCALE_OFFSETS = 3UL;
constexpr uint64_t DET_IDX_BIAS_OFFSETS = 4UL;
constexpr uint64_t DET_IDX_C_OFFSETS = 5UL;
constexpr uint64_t DET_IDX_LOGIT_OFFSETS = 6UL;
constexpr uint64_t DET_IDX_M_TILEIDXS = 0UL;
constexpr uint64_t DET_IDX_N_TILEIDXS = 1UL;
constexpr uint64_t DET_IDX_M_TAIL_SPLIT_TILEIDXS = 2UL;
constexpr uint64_t DET_IDX_N_TAIL_SPLIT_TILEIDXS = 3UL;
constexpr uint8_t DET_SYNC_AIC_AIV_MODES = 4U;
constexpr uint16_t DET_FLAG_ID_MAXS = 16U;
constexpr uint16_t DET_AIC_SYNC_AIV_FLAGS = 4U;
constexpr uint16_t DET_AIV_SYNC_AIC_FLAGS = 6U;
constexpr uint32_t DET_WEIGHT_TILE_K_SMALL = 16U;
constexpr uint32_t DET_WEIGHT_TILE_N_SMALL = 16U;
constexpr uint32_t DET_WEIGHT_TILE_K_LARGE = 32U;
constexpr uint32_t DET_WEIGHT_TILE_N_LARGE = 32U;
constexpr uint32_t DET_WEIGHT_TILE_CAPACITY = 512U;

constexpr uint32_t DET_DETER_UB_SIZE = 12 * 1024;
constexpr uint32_t DET_BUFFER_NUM = 2;
} // namespace

// Debug checkpoint: write value to debugGm[blockIdx * 16 + cpIdx]
// Host reads debugGm after kernel to see which cores reached which points.
// Set DET_DEBUG_ENABLE to 0 to strip all checkpoints.
#ifndef DET_DEBUG_ENABLE
#define DET_DEBUG_ENABLE 1
#endif
#if DET_DEBUG_ENABLE
#define DET_DEBUG_CP(debugGm, cpIdx, val)                                          \
    do {                                                                            \
        if (debugGm != nullptr) {                                                   \
            auto __dbgGm = reinterpret_cast<__gm__ uint32_t *>(debugGm);            \
            __dbgGm[static_cast<uint64_t>(GetBlockIdx()) * 16 + (cpIdx)] = (val);   \
        }                                                                           \
    } while (0)
#else
#define DET_DEBUG_CP(debugGm, cpIdx, val) ((void)0)
#endif

using namespace AscendC;
template <class ProblemShape_, class BlockMmadBuilder_, class BlockPrologue_,
          class BlockEpilogueDequantFinalizeRouting_, class BlockScheduler_, typename Enable_ = void>
class KernelGmmFRDeterministic {
    static_assert(AscendC::Std::always_false_v<BlockScheduler_>,
                  "KernelGmmFRDeterministic is not implemented for this scheduler");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockPrologue_,
          class BlockEpilogueDequantFinalizeRouting_, class BlockScheduler_>
class KernelGmmFRDeterministic<
    ProblemShape_, BlockMmadBuilder_, BlockPrologue_, BlockEpilogueDequantFinalizeRouting_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockScheduler_, GroupedMatmulAswtWithTailSplitScheduler>>> {
public:
    __aicore__ inline KernelGmmFRDeterministic()
    {
    }

    __aicore__ inline ~KernelGmmFRDeterministic()
    {
    }

    using BlockPrologue = BlockPrologue_;
    using BlockEpilogueDequantFinalizeRouting = BlockEpilogueDequantFinalizeRouting_;
    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    static constexpr bool transA = BlockMmadBuilder::transA;
    static constexpr bool transB = BlockMmadBuilder::transB;
    static constexpr int64_t l1M = BlockMmadBuilder::l1M;
    static constexpr int64_t l1N = BlockMmadBuilder::l1N;
    static constexpr int64_t l1K = BlockMmadBuilder::l1K;
    static constexpr auto formatB = BlockMmadBuilder::formatB;
    // schedulerOp
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmadBuilder::L1TileShape,
                                               typename BlockMmadBuilder::L0TileShape, BlockScheduler, transA,
                                               transB>::SchedulerOp;
    // mmadOp
    using BlockMmadOp = typename BlockMmadBuilder::BlockMmadOp;
    using BlockMmadArguments = typename BlockMmadBuilder::Arguments;
    using BlockPrologueArguments = typename BlockPrologue::Arguments;
    using BlockMmadParams = typename BlockMmadBuilder::Params;
    using BlockPrologueParams = typename BlockPrologue::Params;
    using BlockEpilogueParams = typename BlockEpilogueDequantFinalizeRouting::Params;
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // a, b, x1scale, x2scale, bias, y, logit
    using BaseOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // a, b, x1scale, x2scale, bias, y
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // coordinate
    using CoordClass =
        Coordinate<transA, transB, BlockMmadBuilder::formatA, BlockMmadBuilder::formatB, BlockMmadBuilder::formatC>;

    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<int64_t> groupListGm_;
    // shape
    TupleShape problemShape_{};
    BaseOffset baseOffset_{0, 0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};
    uint64_t preOffset_ = 0;
    BlockMmadOp mmadOp_;

    BlockPrologue prologueOp_;
    BlockEpilogueDequantFinalizeRouting epilogueDequantOp_;
    AscendC::LocalTensor<CType> l0cOutUb_;
    bool isVecSetSyncCom_ = false;

    // Aggregation buffer for FRDeterministicA5: raw-offset LocalTensors placed
    // AFTER the epilogue's UB usage (~198KB) to avoid overlap.
    // UB total = 248KB; epilogue uses ~198KB; free space ~50KB; need 12KB.
    static constexpr uint32_t DET_AGG_UB_BYTES = 12 * 1024;
    static constexpr uint32_t DET_AGG_VECOUT_OFFSET = 200 * 1024;  // after epilogue's outUbPong_
    AscendC::LocalTensor<float> aggVecout_;

    struct GMMTiling {
        uint32_t groupNum;
        uint8_t groupListType;
        int32_t baseM;
        int32_t baseN;
        int32_t baseK;
        uint8_t hasBias;
        const TCubeTiling *__restrict matmulTiling;
        __aicore__ GMMTiling()
        {
        }
        __aicore__ GMMTiling(uint32_t groupNum_, uint8_t groupListType_, int32_t baseM_, int32_t baseN_, int32_t baseK_,
                             uint8_t hasBias_)
            : groupNum(groupNum_), groupListType(groupListType_), baseM(baseM_), baseN(baseN_), baseK(baseK_),
              hasBias(hasBias_)
        {
        }
    };

    // Workspace cycling parameters passed from the caller
    struct CyclingParams {
        GM_ADDR yGmAddr;
        GM_ADDR rowIndexAddr;
        uint32_t coreNum;
        uint32_t n;
        uint64_t baseN;
        uint64_t windowSizeRows;
        GM_ADDR debugGmAddr;  // host-allocated uint32_t[36*16] for checkpoint writes
    };

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        BlockPrologueArguments prologueArgs;
        BlockEpilogueParams epilogueArgs;
        GMMTiling gmmArgs;
        CyclingParams cyclingArgs;
        Arguments() = default;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockPrologueParams prologueParams;
        BlockEpilogueParams epilogueParams;
        GMMTiling gmmParams;
        CyclingParams cyclingParams;
        Params() = default;
    };

    __aicore__ inline void NotifyCube()
    {
        AscendC::CrossCoreSetFlag<DET_SYNC_AIC_AIV_MODES, PIPE_V>(DET_AIV_SYNC_AIC_FLAGS);
    }
    __aicore__ inline void WaitForVector()
    {
        AscendC::CrossCoreWaitFlag<DET_SYNC_AIC_AIV_MODES, PIPE_FIX>(DET_AIV_SYNC_AIC_FLAGS);
        AscendC::CrossCoreWaitFlag<DET_SYNC_AIC_AIV_MODES, PIPE_FIX>(DET_AIV_SYNC_AIC_FLAGS + DET_FLAG_ID_MAXS);
    }
    __aicore__ inline void NotifyVector()
    {
        AscendC::CrossCoreSetFlag<DET_SYNC_AIC_AIV_MODES, PIPE_FIX>(DET_AIC_SYNC_AIV_FLAGS);
        AscendC::CrossCoreSetFlag<DET_SYNC_AIC_AIV_MODES, PIPE_FIX>(DET_AIC_SYNC_AIV_FLAGS + DET_FLAG_ID_MAXS);
    }
    __aicore__ inline void WaitForCube()
    {
        AscendC::CrossCoreWaitFlag<DET_SYNC_AIC_AIV_MODES, PIPE_V>(DET_AIC_SYNC_AIV_FLAGS);
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            if (isVecSetSyncCom_) {
                WaitForVector();
            }
        }
    }

    __aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, uint8_t groupListType)
    {
        int32_t splitValue = 0;
        if (groupListType == 0) {
            int32_t offset = static_cast<int32_t>(groupListGm_.GetValue(groupIdx));
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else {
            splitValue = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
        }
        return splitValue;
    }

    // UpdateGlobalBuffer with cyclingBase parameter.
    // elem[4] = LOGIT_OFFSETS (global, for logit/rowIndex/x1Scale reads -- never reset)
    // elem[5] = cyclingBase (workspace write base row, from workspaceBaseRow)
    __aicore__ inline void UpdateGlobalBuffer(const Params &params, uint64_t cyclingBase)
    {
        if ASCEND_IS_AIC {
            aGlobal_.SetGlobalBuffer((__gm__ AType *)params.mmadParams.aGmAddr + Get<DET_IDX_A_OFFSETS>(baseOffset_));
            bGlobal_.SetGlobalBuffer((__gm__ BType *)params.mmadParams.bGmAddr + Get<DET_IDX_B_OFFSETS>(baseOffset_));
        }
        if ASCEND_IS_AIV {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                0L,
                Get<DET_IDX_BIAS_OFFSETS>(baseOffset_),
                Get<DET_IDX_X2SCALE_OFFSETS>(baseOffset_),
                Get<DET_IDX_X1SCALE_OFFSETS>(baseOffset_),
                Get<DET_IDX_LOGIT_OFFSETS>(baseOffset_),    // 4: logit/rowIndex global offset
                static_cast<int64_t>(cyclingBase)};         // 5: workspace write base (workspaceBaseRow)
            epilogueDequantOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    }

    __aicore__ inline void UpdateOffset(uint32_t groupIdx)
    {
        if (groupIdx == 0) {
            return;
        }
        uint64_t m = Get<MNK_M>(problemShape_);
        uint64_t n = Get<MNK_N>(problemShape_);
        uint64_t k = Get<MNK_K>(problemShape_);
        Get<DET_IDX_A_OFFSETS>(baseOffset_) = Get<DET_IDX_A_OFFSETS>(baseOffset_) + m * k;
        if constexpr (formatB == CubeFormat::NZ) {
            if (transB) {
                Get<DET_IDX_B_OFFSETS>(baseOffset_) = Get<DET_IDX_B_OFFSETS>(baseOffset_) +
                                              CeilDiv(k, DET_WEIGHT_TILE_K_LARGE) * CeilDiv(n, DET_WEIGHT_TILE_N_SMALL) * DET_WEIGHT_TILE_CAPACITY;
            } else {
                Get<DET_IDX_B_OFFSETS>(baseOffset_) = Get<DET_IDX_B_OFFSETS>(baseOffset_) +
                                              CeilDiv(n, DET_WEIGHT_TILE_N_LARGE) * CeilDiv(k, DET_WEIGHT_TILE_K_SMALL) * DET_WEIGHT_TILE_CAPACITY;
            }
        } else {
            Get<DET_IDX_B_OFFSETS>(baseOffset_) = Get<DET_IDX_B_OFFSETS>(baseOffset_) + n * k;
        }
        Get<DET_IDX_BIAS_OFFSETS>(baseOffset_) += n;
        Get<DET_IDX_X1SCALE_OFFSETS>(baseOffset_) += m;
        Get<DET_IDX_X2SCALE_OFFSETS>(baseOffset_) = groupIdx * n;
        // LOGIT_OFFSETS is cumulative global offset, never reset during cycling
        Get<DET_IDX_LOGIT_OFFSETS>(baseOffset_) += m;
    }

    __aicore__ inline bool UpdateGroupParams(const Params &params, uint32_t groupIdx)
    {
        UpdateOffset(groupIdx);
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, params.gmmParams.groupListType);
        Get<MNK_M>(problemShape_) = splitValue;
        if (Get<MNK_M>(problemShape_) == 0) {
            return false;
        }
        return true;
    }

    __aicore__ inline void InitParamsAndTensor(const Params &params)
    {
        Get<MNK_N>(problemShape_) = params.gmmParams.matmulTiling->N;
        Get<MNK_K>(problemShape_) = params.gmmParams.matmulTiling->Ka;
        groupListGm_.SetGlobalBuffer((__gm__ int64_t *)params.mmadParams.groupListGmAddr);
    }

    __aicore__ inline void operator()(const Params &params)
    {
        auto dbgGm = params.cyclingParams.debugGmAddr;
        DET_DEBUG_CP(dbgGm, 0, 1);  // CP0: operator() entry

        if ASCEND_IS_AIV {
            prologueOp_.Init(params.prologueParams);
            prologueOp_();
            DET_DEBUG_CP(dbgGm, 1, 1);  // CP1: prologue done
        }
        if ASCEND_IS_AIC {
            mmadOp_.Init(const_cast<TCubeTiling *__restrict>(params.gmmParams.matmulTiling), GetTPipePtr());
            l0cOutUb_ = epilogueDequantOp_.GetL0c2UbTensor();
            DET_DEBUG_CP(dbgGm, 1, 2);  // CP1: mmadInit done (AIC uses value 2)
        }
        InitParamsAndTensor(params);
        BlockSchedulerOp bs(params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        DET_DEBUG_CP(dbgGm, 2, 1);  // CP2: before first SyncAll
        SyncAll<false>();
        DET_DEBUG_CP(dbgGm, 3, 1);  // CP3: after first SyncAll
        if ASCEND_IS_AIV {
            epilogueDequantOp_.Init(params.epilogueParams);
            // Raw-offset aggregation buffer at VECOUT offset 200KB (after epilogue's ~198KB usage)
            aggVecout_ = AscendC::LocalTensor<float>(
                AscendC::TPosition::VECOUT, DET_AGG_VECOUT_OFFSET,
                DET_AGG_UB_BYTES / sizeof(float));
            DET_DEBUG_CP(dbgGm, 4, 1);  // CP4: epilogue+aggBuf init done
        }

        if constexpr (formatB == CubeFormat::NZ) {
            if constexpr (transB) {
                bs.SetTailAlign(1, MATMUL_MNK_ALIGN);
            } else {
                bs.SetTailAlign(1, MATMUL_MNK_ALIGN_INT8);
            }
        }

        // === Cycling state ===
        uint64_t windowSizeRows = params.cyclingParams.windowSizeRows;
        uint64_t workspaceBaseRow = 0;   // workspace window rows used (reset after aggregation)
        uint64_t globalRowOffset = 0;    // global rows already aggregated (for rowIndex lookup)

        // Prepare aggregation tensors (AIV side)
        GlobalTensor<float> deterBufferGm;
        GlobalTensor<float> yGmAgg;
        GlobalTensor<typename BlockEpilogueDequantFinalizeRouting::DataTypeRowIndex> rowIndexGm;
        if ASCEND_IS_AIV {
            deterBufferGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.epilogueParams.yGMAddr));
            yGmAgg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.cyclingParams.yGmAddr));
            rowIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ typename BlockEpilogueDequantFinalizeRouting::DataTypeRowIndex*>(
                params.cyclingParams.rowIndexAddr));
        }

        // === Main loop: tile-level workspace cycling ===
        // Each sub-chunk is at most baseM rows (one matmul M-tile).
        // Before each sub-chunk: if workspace can't fit one more tile, aggregate first.
        // All cores agree because workspaceBaseRow advances identically on all cores.
        //
        // MTE3 DMA drain: epilogue's VectorSequentialWrite (DataCopyPad) is asynchronous.
        // Before aggregation reads workspace, must drain MTE3 pipeline to ensure
        // all DMA writes have landed. Otherwise hardware data hazard -> hang.
        uint32_t groupNum = params.gmmParams.groupNum;
        int32_t baseM = params.gmmParams.baseM;
        DET_DEBUG_CP(dbgGm, 5, 1);  // CP5: before group loop
        for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
            if (!UpdateGroupParams(params, groupIdx)) {
                continue;
            }

            int64_t groupM = Get<MNK_M>(problemShape_);
            int64_t n = Get<MNK_N>(problemShape_);
            int64_t k = Get<MNK_K>(problemShape_);
            uint64_t groupMProcessed = 0;
            DET_DEBUG_CP(dbgGm, 6, groupIdx + 1);  // CP6: group start (1-based)

            while (groupMProcessed < static_cast<uint64_t>(groupM)) {
                // Proactive check: can workspace fit one more matmul M-tile?
                if (workspaceBaseRow + static_cast<uint64_t>(baseM) > windowSizeRows) {
                    DET_DEBUG_CP(dbgGm, 7, static_cast<uint32_t>(workspaceBaseRow));  // CP7: before aggregation
                    // Drain AIC-AIV pipeline
                    End();
                    isVecSetSyncCom_ = false;
                    // Drain MTE3: ensure all epilogue DMA writes to workspace complete
                    if ASCEND_IS_AIV {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
                    }
                    SyncAll();
                    DET_DEBUG_CP(dbgGm, 8, static_cast<uint32_t>(workspaceBaseRow));  // CP8: after SyncAll pre-agg
                    {
                        GMMFRDeterministic::SyncConfig syncConfig{};
                        syncConfig.curM = workspaceBaseRow;
                        syncConfig.lowBoundM = workspaceBaseRow;
                        syncConfig.windowSize = workspaceBaseRow;
                        syncConfig.baseN = params.cyclingParams.baseN;
                        GMMFRDeterministic::FRDeterministicA5<float,
                            typename BlockEpilogueDequantFinalizeRouting::DataTypeRowIndex>(
                            syncConfig, deterBufferGm, yGmAgg, rowIndexGm, aggVecout_,
                            params.cyclingParams.coreNum, params.cyclingParams.n,
                            globalRowOffset);
                    }
                    SyncAll();
                    DET_DEBUG_CP(dbgGm, 9, static_cast<uint32_t>(globalRowOffset));  // CP9: after aggregation SyncAll
                    globalRowOffset += workspaceBaseRow;
                    workspaceBaseRow = 0;
                    continue;
                }

                // One matmul M-tile per sub-chunk
                uint64_t remaining = static_cast<uint64_t>(groupM) - groupMProcessed;
                uint64_t chunkM = (remaining < static_cast<uint64_t>(baseM))
                                  ? remaining : static_cast<uint64_t>(baseM);

                // Adjust baseOffset_ for sub-chunk position within group.
                // Save/restore to keep baseOffset_ at group start for next UpdateGroupParams.
                int64_t savedA = Get<DET_IDX_A_OFFSETS>(baseOffset_);
                int64_t savedLogit = Get<DET_IDX_LOGIT_OFFSETS>(baseOffset_);
                int64_t savedX1Scale = Get<DET_IDX_X1SCALE_OFFSETS>(baseOffset_);

                Get<DET_IDX_A_OFFSETS>(baseOffset_) += static_cast<int64_t>(groupMProcessed) * k;
                Get<DET_IDX_LOGIT_OFFSETS>(baseOffset_) += static_cast<int64_t>(groupMProcessed);
                Get<DET_IDX_X1SCALE_OFFSETS>(baseOffset_) += static_cast<int64_t>(groupMProcessed);

                // Initialize scheduler and epilogue for this sub-chunk
                TupleShape resProblemShape{static_cast<int64_t>(chunkM), n, k, 0};
                bs.UpdateNextProblem(resProblemShape);
                epilogueDequantOp_.UpdateNextProblem(resProblemShape);
                UpdateGlobalBuffer(params, workspaceBaseRow);
                CoordClass coord(static_cast<int64_t>(chunkM), n, k,
                                 baseM, params.gmmParams.baseN, params.gmmParams.baseK);

                // --- Tile loop for this sub-chunk ---
                BlockCoord tileIdx;
                while (bs.GetTileIdx(tileIdx)) {
                    BlockShape singleShape = bs.GetBlockShape(tileIdx);
                    blockOffset_ = coord.template GetQuantOffset<
                        GroupedMatmul::QuantMode::PERTOKEN_MODE>(
                        Get<DET_IDX_M_TILEIDXS>(tileIdx), Get<DET_IDX_N_TILEIDXS>(tileIdx),
                        Get<DET_IDX_M_TAIL_SPLIT_TILEIDXS>(singleShape),
                        Get<DET_IDX_N_TAIL_SPLIT_TILEIDXS>(singleShape));

                    // AIC: MMAD
                    if ASCEND_IS_AIC {
                        if (isVecSetSyncCom_) {
                            WaitForVector();
                        }
                        AscendC::Std::tuple<int32_t, int32_t, int32_t> mmSingleShape{
                            Get<MNK_M>(singleShape), Get<MNK_N>(singleShape), k};
                        mmadOp_(aGlobal_[Get<DET_IDX_A_OFFSETS>(blockOffset_)],
                                bGlobal_[Get<DET_IDX_B_OFFSETS>(blockOffset_)],
                                l0cOutUb_, mmSingleShape, transA, transB);
                        NotifyVector();
                    }
                    isVecSetSyncCom_ = true;

                    // AIV: Epilogue -> workspace
                    if ASCEND_IS_AIV {
                        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{
                            Get<MNK_M>(singleShape), Get<MNK_N>(singleShape), 0, 0};
                        int64_t y = Get<DET_IDX_C_OFFSETS>(blockOffset_);
                        int64_t groupInnerMOffset = y / n;
                        int64_t nOffset = y - groupInnerMOffset * n;
                        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
                            epilogueOffset{nOffset,
                                Get<DET_IDX_BIAS_OFFSETS>(blockOffset_),
                                Get<DET_IDX_X2SCALE_OFFSETS>(blockOffset_),
                                Get<DET_IDX_X1SCALE_OFFSETS>(blockOffset_),
                                groupInnerMOffset,
                                static_cast<int64_t>(workspaceBaseRow)};
                        WaitForCube();
                        epilogueDequantOp_(epilogueShape, epilogueOffset);
                        NotifyCube();
                    }
                }
                // --- tile loop end ---

                // Restore offsets to group start
                Get<DET_IDX_A_OFFSETS>(baseOffset_) = savedA;
                Get<DET_IDX_LOGIT_OFFSETS>(baseOffset_) = savedLogit;
                Get<DET_IDX_X1SCALE_OFFSETS>(baseOffset_) = savedX1Scale;

                workspaceBaseRow += chunkM;
                groupMProcessed += chunkM;
                DET_DEBUG_CP(dbgGm, 10, static_cast<uint32_t>(groupMProcessed));  // CP10: sub-chunk done
            }
        }

        // === Final aggregation (remaining rows) ===
        DET_DEBUG_CP(dbgGm, 11, static_cast<uint32_t>(workspaceBaseRow));  // CP11: before final agg
        // Consume remaining CrossCore flags (matches original kernel's End() after group loop)
        End();
        isVecSetSyncCom_ = false;

        // Drain MTE3: ensure all epilogue DMA writes to workspace complete
        if ASCEND_IS_AIV {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
        }

        // Final aggregation: all cores must participate (SyncAll is global barrier).
        SyncAll();
        DET_DEBUG_CP(dbgGm, 12, 1);  // CP12: after final SyncAll pre-agg
        {
            GMMFRDeterministic::SyncConfig syncConfig{};
            syncConfig.curM = workspaceBaseRow;
            syncConfig.lowBoundM = workspaceBaseRow;
            syncConfig.windowSize = workspaceBaseRow;
            syncConfig.baseN = params.cyclingParams.baseN;

            GMMFRDeterministic::FRDeterministicA5<float,
                typename BlockEpilogueDequantFinalizeRouting::DataTypeRowIndex>(
                syncConfig, deterBufferGm, yGmAgg, rowIndexGm, aggVecout_,
                params.cyclingParams.coreNum, params.cyclingParams.n,
                globalRowOffset);
        }
        SyncAll();
        DET_DEBUG_CP(dbgGm, 13, 1);  // CP13: kernel complete
    }
};
} // namespace Kernel
} // namespace Gemm
} // namespace Cgmct
#endif