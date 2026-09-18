/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/** A5 C8 fast payload QLI for valid -3/-2/-1 requests. */

#ifndef VLLM_A5_LI_MANAGE_C8_FAST_QLI_H
#define VLLM_A5_LI_MANAGE_C8_FAST_QLI_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fused_quant_lightning_indexer_manage_tiling.h"
#include "fused_quant_lightning_indexer_manage_fast_workspace.h"

// The vendored payload implementation has private namespaces and include
// guards, so it can coexist with the stock QuantLI cold path.
using VllmFastLIC8TilingData = FusedQuantLightningIndexerManageTilingData;
#define A5_MTP_CLASSIFY_ONLY 1
#define A5_MTP_SLOT_OUTPUT_STRIDE 2048U
#include "arch35_payload_fast_c8/fqlim_fast_lightning_indexer_common.h"
#include "arch35_payload_fast_c8/fqlim_fast_lightning_indexer_service_cube.h"
#include "arch35_payload_fast_c8/fqlim_fast_lightning_indexer_service_vector.h"
#undef A5_MTP_SLOT_OUTPUT_STRIDE
#undef A5_MTP_CLASSIFY_ONLY

namespace vllm_a5_li_manage_c8_fast {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 128U;
constexpr uint32_t HEAD_DIM = 128U;
constexpr uint32_t TOPK = 2048U;
constexpr uint32_t ROUTES = 4U;
constexpr int32_t ROW_MODE_DENSE = -3;
constexpr int32_t ROW_MODE_FIRST_FILL = -2;
constexpr int32_t ROW_MODE_SPARSE = -1;
constexpr uint32_t THRESHOLD_STRIDE = 16U;
constexpr uint16_t INACTIVE_THRESHOLD = 0xffffU;

using C8MtpQliType =
    VllmFastQLICommon::QLIType<fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, true, VllmFastQLICommon::LI_LAYOUT::TND,
                               VllmFastQLICommon::LI_LAYOUT::PA_BSND, bfloat16_t, float32_t, float32_t, uint16_t>;

// Keep identity normalization and malformed metadata on the general path.
// First-fill uses dense-direct TopK publication so its old pool is ignored.
// The batch-wide gate prevents mixed QLI writers.
__aicore__ inline bool RequiresColdPath(GM_ADDR actualSeqLengthsQuery, GM_ADDR finalSeqLengthsKv, GM_ADDR offloadedLens,
                                        GM_ADDR numCacheTokens, GM_ADDR requestState, GM_ADDR reqPoolEntries,
                                        GM_ADDR cacheSlotsPool,
                                        const FusedQuantLightningIndexerManageTilingData *tiling)
{
    if (tiling->fastPathEnabled == 0U || tiling->batchSize == 0U || tiling->totalQueryRows == 0U) {
        return true;
    }
    GlobalTensor<int32_t> queryEnds;
    GlobalTensor<int32_t> finalLengths;
    GlobalTensor<int32_t> prefixes;
    GlobalTensor<int32_t> budgets;
    GlobalTensor<int32_t> modes;
    GlobalTensor<int32_t> entries;
    GlobalTensor<int32_t> pool;
    queryEnds.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQuery);
    finalLengths.SetGlobalBuffer((__gm__ int32_t *)finalSeqLengthsKv);
    prefixes.SetGlobalBuffer((__gm__ int32_t *)offloadedLens);
    budgets.SetGlobalBuffer((__gm__ int32_t *)numCacheTokens);
    modes.SetGlobalBuffer((__gm__ int32_t *)requestState);
    entries.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
    pool.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
    int32_t queryStart = 0;
    for (uint32_t batch = 0U; batch < tiling->batchSize; ++batch) {
        const int32_t queryEnd = queryEnds.GetValue(batch);
        const int32_t routeCount = queryEnd - queryStart;
        if (routeCount < 1 || routeCount > static_cast<int32_t>(ROUTES) ||
            queryEnd > static_cast<int32_t>(tiling->totalQueryRows)) {
            return true;
        }
        const int32_t mode = modes.GetValue(batch);
        if (mode != ROW_MODE_DENSE && mode != ROW_MODE_FIRST_FILL && mode != ROW_MODE_SPARSE)
            return true;
        const int32_t finalLen = finalLengths.GetValue(batch);
        if (finalLen < static_cast<int32_t>(TOPK) + routeCount ||
            finalLen > static_cast<int32_t>(tiling->sourceCapacity)) {
            return true;
        }
        const int32_t poolRow = entries.GetValue(batch);
        if (poolRow < 0 || poolRow >= static_cast<int32_t>(tiling->poolSize))
            return true;
        for (uint32_t other = 0; other < batch; ++other)
            if (entries.GetValue(other) == poolRow)
                return true;
        if (mode == ROW_MODE_FIRST_FILL || mode == ROW_MODE_SPARSE) {
            const int32_t prefix = prefixes.GetValue(batch);
            const int32_t budget = budgets.GetValue(batch);
            if (prefix < static_cast<int32_t>(TOPK) || prefix % static_cast<int32_t>(BLOCK_SIZE) != 0 ||
                prefix > finalLen - (routeCount - 1) || finalLen - prefix > 256 ||
                (budget != 8192 && budget != 10240 && budget != 12288) || budget > prefix)
                return true;
            // A direct -3 -> -1 transition still has an identity row. Let
            // the general manager normalize it before fast classification.
            if (mode == ROW_MODE_SPARSE && budget < static_cast<int32_t>(tiling->sourceCapacity) &&
                pool.GetValue(static_cast<uint64_t>(poolRow) * tiling->sourceCapacity +
                              static_cast<uint32_t>(budget)) == budget)
                return true;
        }
        queryStart = queryEnd;
    }
    return queryStart != static_cast<int32_t>(tiling->totalQueryRows);
}

class QuantLiMtpPhase {
public:
    __aicore__ inline QuantLiMtpPhase(TPipe *pipe, const FusedQuantLightningIndexerManageTilingData *tiling)
        : pipe_(pipe),
          tiling_(tiling)
    {}

    __aicore__ inline void Init(GM_ADDR indexWeights, GM_ADDR query, GM_ADDR queryDequantScale,
                                GM_ADDR actualSeqLengthsQuery, GM_ADDR indexKeyCache, GM_ADDR indexKeyDequantScale,
                                GM_ADDR finalSeqLengthsKv, GM_ADDR reqPoolEntries, GM_ADDR cacheSlotsPool,
                                GM_ADDR offloadedLens, GM_ADDR numCacheTokens, GM_ADDR rowModes,
                                GM_ADDR indexBlockTable, GM_ADDR routePairRows, GM_ADDR topkSlots,
                                GM_ADDR topkSourceIds, GM_ADDR routeThresholds, GM_ADDR routeMissCounts,
                                GM_ADDR userWorkspace)
    {
        subBlockIdx_ = GetBlockIdx();
        if ASCEND_IS_AIV {
            aiCoreIdx_ = subBlockIdx_ / 2U;
        } else {
            aiCoreIdx_ = subBlockIdx_;
        }

        queryEndsGm_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQuery);
        finalLengthsGm_.SetGlobalBuffer((__gm__ int32_t *)finalSeqLengthsKv);
        offloadedLensGm_.SetGlobalBuffer((__gm__ int32_t *)offloadedLens);
        cacheTokensGm_.SetGlobalBuffer((__gm__ int32_t *)numCacheTokens);
        rowModesGm_.SetGlobalBuffer((__gm__ int32_t *)rowModes);
        reqPoolEntriesGm_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        cacheSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)indexBlockTable);

        constInfo_.batchSize = tiling_->batchSize;
        constInfo_.tSize = tiling_->totalQueryRows;
        constInfo_.gSize = tiling_->indexHeads;
        constInfo_.qHeadNum = tiling_->indexHeads;
        constInfo_.kHeadNum = 1;
        constInfo_.headDim = HEAD_DIM;
        constInfo_.sparseCount = TOPK;
        constInfo_.kSeqSize = tiling_->fastScoreRowStride;
        constInfo_.qSeqSize = ROUTES;
        constInfo_.kCacheBlockSize = BLOCK_SIZE;
        constInfo_.maxBlockNumPerBatch = tiling_->maxBlockNumPerBatch;
        constInfo_.outputLayout = VllmFastQLICommon::LI_LAYOUT::TND;
        // Both states scan the full final length and clip each route.
        constInfo_.attenMaskFlag = false;
        constInfo_.isAccumSeqS1 = true;
        constInfo_.s1BaseSize = tiling_->fastQueryTileSize;
        constInfo_.mBaseSize = constInfo_.s1BaseSize * tiling_->indexHeads;
        constInfo_.s2BaseSize = BLOCK_SIZE;
        constInfo_.keyStride0 = tiling_->keyStride;
        constInfo_.keyDequantScaleStride0 = tiling_->scaleStride;
        constInfo_.poolSize = tiling_->poolSize;
        // Standard pool rows contain source mappings only; no state column.
        constInfo_.cacheSlotsSize = static_cast<uint64_t>(tiling_->sourceCapacity);
        constInfo_.setL2DisableFlag = false;

        scoreWorkspaceBaseGm_.SetGlobalBuffer((__gm__ uint16_t *)userWorkspace);
        if ASCEND_IS_AIV {
            weightsGm_.SetGlobalBuffer((__gm__ bfloat16_t *)indexWeights);
            queryScaleGm_.SetGlobalBuffer((__gm__ float *)queryDequantScale);
            keyScaleGm_.SetGlobalBuffer((__gm__ float *)indexKeyDequantScale);
            routePairRowsGm_.SetGlobalBuffer((__gm__ int32_t *)routePairRows);
            topkSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)topkSlots);
            topkSourceIdsGm_.SetGlobalBuffer((__gm__ int32_t *)topkSourceIds);
            routeMissCountsGm_.SetGlobalBuffer((__gm__ int32_t *)routeMissCounts);
            routeThresholdsGm_.SetGlobalBuffer((__gm__ uint16_t *)routeThresholds);
            vectorService_.InitParams(constInfo_, tiling_);
            vectorService_.InitVecInputTensor(weightsGm_, queryScaleGm_, keyScaleGm_, routePairRowsGm_, blockTableGm_,
                                              cacheSlotsGm_, topkSlotsGm_, routeMissCountsGm_);
            // Dense publish writes the selected logical IDs to both public
            // TopK rows. The manager separately fills the identity pool.
            vectorService_.InitMtpTopkSourceTensor(topkSourceIdsGm_);
            vectorService_.InitMtpThresholdTensor(routeThresholdsGm_);
            vectorService_.InitBuffers(pipe_);
        } else {
            queryGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)query);
            keyGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)indexKeyCache);
            matmulService_.InitParams(constInfo_);
            matmulService_.InitMm1GlobalTensor(blockTableGm_, keyGm_, queryGm_);
            matmulService_.InitBuffers(pipe_);
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t tile = tiling_->fastQueryTileSize;
        if (tile != 2U && tile != ROUTES) {
            return;
        }
        const uint32_t queryTileCount = (ROUTES + tile - 1U) / tile;
        const uint32_t taskCount = tiling_->batchSize * queryTileCount;
        if ASCEND_IS_AIV {
            vectorService_.AllocEventID();
            CrossCoreSetFlag<VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT);
            CrossCoreSetFlag<VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT + 1U);
        } else {
            matmulService_.AllocEventID();
        }

        uint32_t globalLoop = 0U;
        for (uint32_t task = aiCoreIdx_; task < taskCount; task += tiling_->usedCoreNum) {
            const uint32_t batch = task / queryTileCount;
            uint32_t queryStart = 0U;
            uint32_t queryEnd = 0U;
            uint32_t routeCount = 0U;
            GetQueryRange(batch, queryStart, queryEnd, routeCount);
            const uint32_t gS1 = task % queryTileCount;
            const uint32_t routeBase = gS1 * tile;
            if (routeBase >= routeCount || !TaskNeedsQli(batch, routeBase, tile, routeCount)) {
                continue;
            }
            const uint32_t scanLen = static_cast<uint32_t>(finalLengthsGm_.GetValue(batch));
            const bool sparse = rowModesGm_.GetValue(batch) == ROW_MODE_SPARSE;
            const uint32_t poolRow = !sparse ? 0U : static_cast<uint32_t>(reqPoolEntriesGm_.GetValue(batch));
            const uint32_t budget = !sparse ? 0U : static_cast<uint32_t>(cacheTokensGm_.GetValue(batch));
            const uint32_t activeRoutes = routeBase + tile > routeCount ? routeCount - routeBase : tile;
            const uint32_t loopCount = (scanLen + BLOCK_SIZE - 1U) / BLOCK_SIZE;
            if ASCEND_IS_AIV {
                const uint64_t requestScoreBase =
                    static_cast<uint64_t>(batch) * (tiling_->fastScoreWorkspaceStride / sizeof(uint16_t));
                const uint64_t routeScoreBase = static_cast<uint64_t>(gS1) * tile * tiling_->fastScoreRowStride;
                vectorService_.InitVecWorkspaceTensor(scoreWorkspaceBaseGm_[requestScoreBase + routeScoreBase]);
            }
            for (uint32_t s2 = 0U; s2 < loopCount; ++s2, ++globalLoop) {
                VllmFastQLICommon::RunInfo run{};
                run.loop = globalLoop;
                run.bN2Idx = batch;
                run.bIdx = batch;
                run.n2Idx = 0U;
                run.gS1Idx = gS1;
                run.s2Idx = s2;
                run.s2Start = 0U;
                run.validS2Len = scanLen;
                run.kScaleLoop = s2 / 16U + 1U;
                run.cacheRowIdx = poolRow;
                run.cacheTokenCount = budget;
                run.offloadedLen = !sparse ? 0U : static_cast<uint32_t>(offloadedLensGm_.GetValue(batch));
                run.actS1Size = routeCount;
                run.actS2Size = scanLen;
                run.actS2SizeOrig = scanLen;
                run.actMBaseSize = activeRoutes * tiling_->indexHeads;
                run.actualSingleProcessSInnerSize =
                    scanLen - s2 * BLOCK_SIZE < BLOCK_SIZE ? scanLen - s2 * BLOCK_SIZE : BLOCK_SIZE;
                run.actualSingleProcessSInnerSizeAlign =
                    VllmFastQLICommon::Align(run.actualSingleProcessSInnerSize, 32U);
                run.tensorQueryOffset = static_cast<uint64_t>(queryStart) * tiling_->indexHeads * HEAD_DIM +
                                        static_cast<uint64_t>(gS1) * constInfo_.mBaseSize * HEAD_DIM;
                run.tensorKeyOffset = static_cast<uint64_t>(s2) * BLOCK_SIZE * HEAD_DIM;
                run.tensorKeyScaleOffset = static_cast<uint64_t>(s2) * BLOCK_SIZE;
                run.tensorWeightsOffset = static_cast<uint64_t>(queryStart) * tiling_->weightStride;
                run.tensorQueryScaleOffset = static_cast<uint64_t>(queryStart) * tiling_->indexHeads;
                run.indiceOutOffset = static_cast<uint64_t>(queryStart) * TOPK;
                run.workspaceRow = batch * ROUTES;
                run.causalClip = true;
                run.denseDirectPublish = !sparse;
                run.isFirstS2InnerLoop = s2 == 0U;
                run.isLastS2InnerLoop = s2 + 1U == loopCount;
                run.isAllLoopEnd = false;
                run.isValid = true;

                if ASCEND_IS_AIC {
                    matmulService_.ComputeMm1(run);
                } else {
                    vectorService_.ProcessVec1(run);
                    if (run.isLastS2InnerLoop) {
                        const bool finalTask = !HasLaterActiveTask(task, taskCount, queryTileCount, tile);
                        vectorService_.ProcessTopK(run, sparse && finalTask);
                    }
                }
            }
        }

        if ASCEND_IS_AIV {
            vectorService_.FreeEventID();
        } else {
            matmulService_.FreeEventID();
            CrossCoreWaitFlag<VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT);
            CrossCoreWaitFlag<VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT + 1U);
        }
    }

private:
    __aicore__ inline void GetQueryRange(uint32_t batch, uint32_t &queryStart, uint32_t &queryEnd,
                                         uint32_t &routeCount) const
    {
        const int32_t end = queryEndsGm_.GetValue(batch);
        const int32_t start = batch == 0U ? 0 : queryEndsGm_.GetValue(batch - 1U);
        queryStart = static_cast<uint32_t>(start);
        queryEnd = static_cast<uint32_t>(end);
        routeCount = queryEnd - queryStart;
    }

    __aicore__ inline bool TaskNeedsQli(uint32_t batch, uint32_t routeBase, uint32_t tile, uint32_t routeCount) const
    {
        if (routeBase >= routeCount) {
            return false;
        }
        const int32_t mode = rowModesGm_.GetValue(batch);
        if (mode != ROW_MODE_DENSE && mode != ROW_MODE_FIRST_FILL && mode != ROW_MODE_SPARSE) {
            return false;
        }
        if (mode != ROW_MODE_DENSE)
            return true;
        const uint32_t activeRoutes = routeBase + tile > routeCount ? routeCount - routeBase : tile;
        const uint32_t lastRoute = routeBase + activeRoutes - 1U;
        const uint32_t finalLen = static_cast<uint32_t>(finalLengthsGm_.GetValue(batch));
        const uint32_t visible = finalLen - (routeCount - 1U - lastRoute);
        return visible > TOPK;
    }

    __aicore__ inline bool HasLaterActiveTask(uint32_t task, uint32_t taskCount, uint32_t queryTileCount,
                                              uint32_t tile) const
    {
        for (uint32_t next = task + tiling_->usedCoreNum; next < taskCount; next += tiling_->usedCoreNum) {
            const uint32_t nextBatch = next / queryTileCount;
            uint32_t queryStart = 0U;
            uint32_t queryEnd = 0U;
            uint32_t routeCount = 0U;
            GetQueryRange(nextBatch, queryStart, queryEnd, routeCount);
            const uint32_t routeBase = (next % queryTileCount) * tile;
            if (TaskNeedsQli(nextBatch, routeBase, tile, routeCount)) {
                return true;
            }
        }
        return false;
    }

    TPipe *pipe_;
    const FusedQuantLightningIndexerManageTilingData *tiling_;
    uint32_t subBlockIdx_ = 0U;
    uint32_t aiCoreIdx_ = 0U;
    VllmFastQLICommon::ConstInfo constInfo_{};
    VllmFastQLIKernel::QLIMatmul<C8MtpQliType> matmulService_;
    VllmFastQLIKernel::QLIVector<C8MtpQliType> vectorService_;
    GlobalTensor<fp8_e4m3fn_t> queryGm_;
    GlobalTensor<fp8_e4m3fn_t> keyGm_;
    GlobalTensor<bfloat16_t> weightsGm_;
    GlobalTensor<float> queryScaleGm_;
    GlobalTensor<float> keyScaleGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> queryEndsGm_;
    GlobalTensor<int32_t> finalLengthsGm_;
    GlobalTensor<int32_t> offloadedLensGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    GlobalTensor<int32_t> rowModesGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> cacheSlotsGm_;
    GlobalTensor<int32_t> routePairRowsGm_;
    GlobalTensor<int32_t> topkSlotsGm_;
    GlobalTensor<int32_t> topkSourceIdsGm_;
    GlobalTensor<int32_t> routeMissCountsGm_;
    GlobalTensor<uint16_t> routeThresholdsGm_;
    GlobalTensor<uint16_t> scoreWorkspaceBaseGm_;
};

} // namespace vllm_a5_li_manage_c8_fast

#endif
