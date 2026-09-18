/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * A5 MTP0..3 C8 LightningIndexer with the standardized ABI.
 * request_state=-3 publishes identity slots; request_state=-2 performs the
 * correctness-first full-visible first-fill cache construction; and
 * request_state=-1 performs steady cache management, including normalization
 * of an identity row inherited directly from request_state=-3.
 */

#include "kernel_operator.h"
#include "lim_simt_config.h"
#include "lib/matmul_intf.h"
#include "fused_quant_lightning_indexer_manage_tiling.h"
#include "fqlim_quant_lightning_indexer_common.h"
#include "fqlim_quant_lightning_indexer_service_cube.h"
#include "fqlim_quant_lightning_indexer_service_vector.h"
#include "fused_quant_lightning_indexer_manage_fast_qli.h"
#include "vllm_a5_li_manage_c8_fast_union.h"
#include "vllm_a5_li_manage_c8_fast_victim_vf.h"
#ifndef FQLIM_EXPERIMENTAL_SIMT_SLOT_REPAIR
#define FQLIM_EXPERIMENTAL_SIMT_SLOT_REPAIR 0
#endif
#if FQLIM_EXPERIMENTAL_SIMT_SLOT_REPAIR
#include "vllm_a5_li_manage_c8_fast_slot_repair.h"
#endif

namespace {
using namespace AscendC;
using namespace QLICommon;
using namespace QLIKernel;

template <HardEvent event>
__aicore__ inline void SyncPipes()
{
    const event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

constexpr uint32_t BLOCK_SIZE = 128;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t TOPK = 2048;
constexpr uint32_t OUTPUT_CAPACITY = TOPK;
constexpr uint32_t MISS_CAPACITY = 32768;
constexpr uint32_t MAX_QUERIES_PER_REQUEST = 4;
constexpr uint32_t REQUEST_DONE_EVENT = 6;

using MtpQliType =
    QLIType<fp8_e4m3fn_t, fp8_e4m3fn_t, float, uint16_t, int32_t, true, LI_LAYOUT::TND, LI_LAYOUT::PA_BSND>;

class VllmA5MtpC8QliPhase {
public:
    __aicore__ inline VllmA5MtpC8QliPhase(TPipe *pipe, const FusedQuantLightningIndexerManageTilingData *tiling)
        : pipe_(pipe),
          tiling_(tiling)
    {}

    __aicore__ inline void Init(GM_ADDR indexWeights, GM_ADDR queryDequantScale, GM_ADDR query,
                                GM_ADDR indexKeyDequantScale, GM_ADDR indexKeyCache, GM_ADDR indexBlockTable,
                                GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKey, GM_ADDR requestState,
                                GM_ADDR topkSrcIds, GM_ADDR userWorkspace)
    {
        if ASCEND_IS_AIV {
            subBlockIdx_ = GetBlockIdx();
            aiCoreIdx_ = subBlockIdx_ / 2U;
        } else {
            subBlockIdx_ = GetBlockIdx();
            aiCoreIdx_ = subBlockIdx_;
        }

        actualSeqLengthsQueryGm_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQuery);
        actualSeqLengthsKeyGm_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsKey);
        requestStateGm_.SetGlobalBuffer((__gm__ int32_t *)requestState);
        indexBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)indexBlockTable);

        constInfo_.batchSize = tiling_->batchSize;
        constInfo_.gSize = tiling_->indexHeads;
        constInfo_.qHeadNum = tiling_->indexHeads;
        constInfo_.kHeadNum = 1;
        constInfo_.headDim = HEAD_DIM;
        constInfo_.sparseCount = TOPK;
        constInfo_.kSeqSize = tiling_->sourceCapacity;
        constInfo_.qSeqSize = 1;
        constInfo_.kCacheBlockSize = BLOCK_SIZE;
        constInfo_.maxBlockNumPerBatch = tiling_->maxBlockNumPerBatch;
        constInfo_.outputLayout = LI_LAYOUT::TND;
        constInfo_.attenMaskFlag = false;
        constInfo_.cmpRatio = 1;
        constInfo_.batchSupperFlag = false;
        constInfo_.stride = tiling_->keyStride;
        constInfo_.scaleStride = tiling_->scaleStride;
        constInfo_.mBaseSize = 256;
        constInfo_.s1BaseSize = (constInfo_.mBaseSize + constInfo_.gSize - 1U) / constInfo_.gSize;
        constInfo_.s2BaseSize = BLOCK_SIZE;

        GlobalTensor<uint16_t> scoreWorkspace;
        scoreWorkspace.SetGlobalBuffer(
            (__gm__ uint16_t *)(userWorkspace + static_cast<uint64_t>(aiCoreIdx_) * tiling_->scoreWorkspaceStride));

        if ASCEND_IS_AIV {
            weightsGm_.SetGlobalBuffer((__gm__ bfloat16_t *)indexWeights);
            queryScaleGm_.SetGlobalBuffer((__gm__ float *)queryDequantScale);
            keyScaleGm_.SetGlobalBuffer((__gm__ float *)indexKeyDequantScale);
            topkSrcIdsGm_.SetGlobalBuffer((__gm__ int32_t *)topkSrcIds);
            vectorService_.InitParams(constInfo_);
            vectorService_.InitVecInputTensor(weightsGm_, queryScaleGm_, keyScaleGm_, topkSrcIdsGm_,
                                              indexBlockTableGm_);
            vectorService_.InitVecWorkspaceTensor(scoreWorkspace);
            vectorService_.InitBuffers(pipe_);
        } else {
            queryGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)query);
            keyGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)indexKeyCache);
            matmulService_.InitParams(constInfo_);
            matmulService_.InitMm1GlobalTensor(indexBlockTableGm_, keyGm_, queryGm_);
            matmulService_.InitBuffers(pipe_);
        }
    }

    __aicore__ inline void Process()
    {
        bool hasSelection = false;
        for (uint32_t batch = aiCoreIdx_; batch < tiling_->batchSize; batch += tiling_->usedCoreNum) {
            uint32_t queryStart = 0;
            uint32_t queryEnd = 0;
            if (IsValidSelectionRequest(batch, queryStart, queryEnd)) {
                hasSelection = true;
                break;
            }
        }
        if (!hasSelection) {
            return;
        }

        if ASCEND_IS_AIV {
            vectorService_.AllocEventID();
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(ConstInfo::CROSS_VC_EVENT);
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(ConstInfo::CROSS_VC_EVENT + 1U);
        } else {
            matmulService_.AllocEventID();
        }

        uint32_t globalLoop = 0;
        for (uint32_t batch = aiCoreIdx_; batch < tiling_->batchSize; batch += tiling_->usedCoreNum) {
            uint32_t queryStart = 0;
            uint32_t queryEnd = 0;
            if (!IsValidSelectionRequest(batch, queryStart, queryEnd)) {
                continue;
            }
            for (uint32_t queryRow = queryStart; queryRow < queryEnd; ++queryRow) {
                const uint32_t finalLen = static_cast<uint32_t>(actualSeqLengthsKeyGm_.GetValue(batch));
                const uint32_t candidate = QueryVisibleLen(queryRow, queryEnd, finalLen);
                const uint32_t loopCount = (candidate + BLOCK_SIZE - 1U) / BLOCK_SIZE;
                for (uint32_t s2 = 0; s2 < loopCount; ++s2, ++globalLoop) {
                    RunInfo run{};
                    run.loop = globalLoop;
                    run.bN2Idx = batch;
                    run.bIdx = batch;
                    run.n2Idx = 0;
                    run.gS1Idx = 0;
                    run.s2Idx = s2;
                    run.actS1Size = 1;
                    run.actS2Size = candidate;
                    run.actS2SizeOrig = candidate;
                    run.actMBaseSize = tiling_->indexHeads;
                    run.actualSingleProcessSInnerSize =
                        candidate - s2 * BLOCK_SIZE < BLOCK_SIZE ? candidate - s2 * BLOCK_SIZE : BLOCK_SIZE;
                    run.actualSingleProcessSInnerSizeAlign =
                        QLICommon::Align(run.actualSingleProcessSInnerSize, ConstInfo::BUFFER_SIZE_BYTE_32B);
                    run.tensorQueryOffset = static_cast<uint64_t>(queryRow) * tiling_->indexHeads * HEAD_DIM;
                    run.tensorKeyOffset = static_cast<uint64_t>(s2) * BLOCK_SIZE * HEAD_DIM;
                    run.tensorKeyScaleOffset = static_cast<uint64_t>(s2) * BLOCK_SIZE;
                    run.tensorWeightsOffset = static_cast<uint64_t>(queryRow) * tiling_->weightStride;
                    run.tensorQueryScaleOffset = static_cast<uint64_t>(queryRow) * tiling_->indexHeads;
                    run.indiceOutOffset = static_cast<uint64_t>(queryRow) * OUTPUT_CAPACITY;
                    run.isFirstS2InnerLoop = s2 == 0;
                    run.isLastS2InnerLoop = s2 + 1U == loopCount;
                    run.isAllLoopEnd = false;
                    run.isValid = true;

                    if ASCEND_IS_AIC {
                        matmulService_.ComputeMm1(run);
                    } else {
                        vectorService_.ProcessVec1(run);
                        if (run.isLastS2InnerLoop) {
                            vectorService_.ProcessTopK(run);
                        }
                    }
                }

                // The score workspace is reused for the next query row.
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(REQUEST_DONE_EVENT);
                } else if ((subBlockIdx_ & 1U) == 0U) {
                    CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(REQUEST_DONE_EVENT);
                }
            }
        }

        if ASCEND_IS_AIV {
            vectorService_.FreeEventID();
        } else {
            matmulService_.FreeEventID();
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(ConstInfo::CROSS_VC_EVENT);
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(ConstInfo::CROSS_VC_EVENT + 1U);
        }
    }

private:
    __aicore__ inline bool GetQueryRange(uint32_t batch, uint32_t &queryStart, uint32_t &queryEnd)
    {
        const int32_t end = actualSeqLengthsQueryGm_.GetValue(batch);
        const int32_t start = batch == 0 ? 0 : actualSeqLengthsQueryGm_.GetValue(batch - 1U);
        if (start < 0 || end <= start || end > static_cast<int32_t>(tiling_->totalQueryRows) ||
            end - start > static_cast<int32_t>(MAX_QUERIES_PER_REQUEST)) {
            return false;
        }
        queryStart = static_cast<uint32_t>(start);
        queryEnd = static_cast<uint32_t>(end);
        return true;
    }

    __aicore__ inline bool IsValidSelectionRequest(uint32_t batch, uint32_t &queryStart, uint32_t &queryEnd)
    {
        if (!GetQueryRange(batch, queryStart, queryEnd)) {
            return false;
        }
        const int32_t state = requestStateGm_.GetValue(batch);
        const int32_t finalLen = actualSeqLengthsKeyGm_.GetValue(batch);
        if ((state != -3 && state != -2 && state != -1) || finalLen < static_cast<int32_t>(queryEnd - queryStart) ||
            finalLen > static_cast<int32_t>(tiling_->sourceCapacity)) {
            return false;
        }
        return static_cast<uint32_t>(finalLen) <= tiling_->sourceCapacity;
    }

    __aicore__ inline uint32_t QueryVisibleLen(uint32_t queryRow, uint32_t queryEnd, uint32_t finalLen) const
    {
        const uint32_t laterQueries = queryEnd - 1U - queryRow;
        return finalLen > laterQueries ? finalLen - laterQueries : 0;
    }

private:
    TPipe *pipe_;
    const FusedQuantLightningIndexerManageTilingData *tiling_;
    uint32_t subBlockIdx_ = 0;
    uint32_t aiCoreIdx_ = 0;
    ConstInfo constInfo_{};
    QLIMatmul<MtpQliType> matmulService_;
    QLIVector<MtpQliType> vectorService_;
    GlobalTensor<fp8_e4m3fn_t> queryGm_;
    GlobalTensor<fp8_e4m3fn_t> keyGm_;
    GlobalTensor<bfloat16_t> weightsGm_;
    GlobalTensor<float> queryScaleGm_;
    GlobalTensor<float> keyScaleGm_;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm_;
    GlobalTensor<int32_t> indexBlockTableGm_;
    GlobalTensor<int32_t> actualSeqLengthsKeyGm_;
    GlobalTensor<int32_t> requestStateGm_;
    GlobalTensor<int32_t> topkSrcIdsGm_;
};

class NonOffloadPublisher {
public:
    __aicore__ inline void Init(TPipe *pipe, bool fastDenseAlreadyPublished, GM_ADDR actualSeqLengthsQuery,
                                GM_ADDR actualSeqLengthsKey, GM_ADDR requestState, GM_ADDR reqPoolEntries,
                                GM_ADDR cacheSlotsPool, GM_ADDR topkSrcIds, GM_ADDR topkDstSlots,
                                GM_ADDR topkMissCounts, GM_ADDR missCounts,
                                const FusedQuantLightningIndexerManageTilingData *tiling)
    {
        queryEnds_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQuery);
        keyLengths_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsKey);
        states_.SetGlobalBuffer((__gm__ int32_t *)requestState);
        entries_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        pool_.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
        topkSources_.SetGlobalBuffer((__gm__ int32_t *)topkSrcIds);
        topkDestinations_.SetGlobalBuffer((__gm__ int32_t *)topkDstSlots);
        routeMissCounts_.SetGlobalBuffer((__gm__ int32_t *)topkMissCounts);
        requestMissCounts_.SetGlobalBuffer((__gm__ int32_t *)missCounts);
        pipe_ = pipe;
        fastDenseAlreadyPublished_ = fastDenseAlreadyPublished;
        tiling_ = tiling;
    }

    __aicore__ inline void Process(uint32_t owner, uint32_t owners)
    {
        for (uint32_t batch = owner; batch < tiling_->batchSize; batch += owners) {
            const int32_t state = states_.GetValue(batch);
            if (state == -2 || state == -1) {
                continue;
            }
            int32_t startValue = 0;
            int32_t endValue = 0;
            int32_t row = -1;
            const bool valid = LoadRequest(batch, startValue, endValue, row);

            const uint32_t safeStart = startValue > 0 ? static_cast<uint32_t>(startValue) : 0U;
            const uint32_t safeEnd = endValue > 0 && endValue <= static_cast<int32_t>(tiling_->totalQueryRows) ?
                                         static_cast<uint32_t>(endValue) :
                                         safeStart;
            for (uint32_t route = safeStart; route < safeEnd; ++route) {
                const uint64_t base = static_cast<uint64_t>(route) * TOPK;
                if (!valid) {
                    for (uint32_t position = 0; position < TOPK; ++position) {
                        topkSources_.SetValue(base + position, -1);
                        topkDestinations_.SetValue(base + position, -1);
                    }
                    continue;
                }
                // A5 fast QLI has already published both dense TopK outputs.
                if (fastDenseAlreadyPublished_)
                    continue;
                for (uint32_t position = 0; position < TOPK; ++position) {
                    topkDestinations_.SetValue(base + position, topkSources_.GetValue(base + position));
                }
            }
        }
    }

    // The standard lifecycle requires an identity pool on every -3 call.
    // Port the A3 standard operator's disjoint 4096-token AIV chunks instead
    // of issuing one scalar GM store per source on a single request owner.
    __aicore__ inline void FillIdentityChunks(uint32_t owner, uint32_t owners)
    {
        constexpr uint32_t CHUNK = 4096U;
        if (owner * CHUNK >= tiling_->sourceCapacity)
            return;
        bool initialized = false;
        for (uint32_t batch = 0U; batch < tiling_->batchSize; ++batch) {
            if (states_.GetValue(batch) != -3)
                continue;
            int32_t startValue = 0, endValue = 0, row = -1;
            if (fastDenseAlreadyPublished_) {
                // The batch-wide fast gate has already checked query ranges,
                // pool entries, and active-row uniqueness on every core.
                row = entries_.GetValue(batch);
            } else if (!LoadRequest(batch, startValue, endValue, row)) {
                continue;
            }
            if (!initialized) {
                pipe_->InitBuffer(identityBaseBuf_, CHUNK * sizeof(int32_t));
                pipe_->InitBuffer(identityValuesBuf_, CHUNK * sizeof(int32_t));
                LocalTensor<int32_t> base = identityBaseBuf_.Get<int32_t>();
                Arange<int32_t>(base, 0, 1, static_cast<int32_t>(CHUNK));
                PipeBarrier<PIPE_V>();
                initialized = true;
            }
            LocalTensor<int32_t> base = identityBaseBuf_.Get<int32_t>();
            LocalTensor<int32_t> values = identityValuesBuf_.Get<int32_t>();
            const uint64_t poolBase = static_cast<uint64_t>(row) * tiling_->sourceCapacity;
            for (uint32_t source = owner * CHUNK; source < tiling_->sourceCapacity; source += owners * CHUNK) {
                const uint32_t valid =
                    tiling_->sourceCapacity - source < CHUNK ? tiling_->sourceCapacity - source : CHUNK;
                Adds(values, base, static_cast<int32_t>(source), valid);
                PipeBarrier<PIPE_V>();
                SyncPipes<HardEvent::V_MTE3>();
                DataCopyPad(pool_[poolBase + source], values,
                            {1, static_cast<uint16_t>(valid * sizeof(int32_t)), 0, 0});
                SyncPipes<HardEvent::MTE3_V>();
            }
        }
    }

    // Keep compact count tensors single-writer. Adjacent requests share a GM
    // cache line, so scalar writes from multiple AIVs can otherwise lose the
    // first route count at a core boundary.
    __aicore__ inline void PublishCounts()
    {
        for (uint32_t batch = 0; batch < tiling_->batchSize; ++batch) {
            const int32_t state = states_.GetValue(batch);
            if (state == -2 || state == -1) {
                continue;
            }
            int32_t startValue = 0;
            int32_t endValue = 0;
            int32_t row = -1;
            bool valid = fastDenseAlreadyPublished_;
            if (fastDenseAlreadyPublished_) {
                endValue = queryEnds_.GetValue(batch);
                startValue = batch == 0U ? 0 : queryEnds_.GetValue(batch - 1U);
            } else {
                valid = LoadRequest(batch, startValue, endValue, row);
            }
            const uint32_t safeStart = startValue > 0 ? static_cast<uint32_t>(startValue) : 0U;
            const uint32_t safeEnd = endValue > 0 && endValue <= static_cast<int32_t>(tiling_->totalQueryRows) ?
                                         static_cast<uint32_t>(endValue) :
                                         safeStart;
            for (uint32_t route = safeStart; route < safeEnd; ++route) {
                routeMissCounts_.SetValue(route, valid ? 0 : -1);
            }
            requestMissCounts_.SetValue(batch, valid ? 0 : -1);
        }
    }

    // As in the A3 standard operator, homogeneous non-offload counts can be
    // published with contiguous DMA instead of one scalar GM store per route.
    // This remains single-writer to avoid adjacent-request false sharing.
    __aicore__ inline void PublishFastHomogeneousCounts()
    {
        constexpr uint32_t CHUNK = 256U;
        pipe_->InitBuffer(zeroCountsBuf_, CHUNK * sizeof(int32_t));
        LocalTensor<int32_t> zeros = zeroCountsBuf_.Get<int32_t>();
        Duplicate(zeros, 0, CHUNK);
        PipeBarrier<PIPE_V>();
        for (uint32_t start = 0U; start < tiling_->totalQueryRows; start += CHUNK) {
            const uint32_t valid = tiling_->totalQueryRows - start < CHUNK ? tiling_->totalQueryRows - start : CHUNK;
            SyncPipes<HardEvent::V_MTE3>();
            DataCopyPad(routeMissCounts_[start], zeros, {1, static_cast<uint16_t>(valid * sizeof(int32_t)), 0, 0});
            SyncPipes<HardEvent::MTE3_V>();
        }
        for (uint32_t start = 0U; start < tiling_->batchSize; start += CHUNK) {
            const uint32_t valid = tiling_->batchSize - start < CHUNK ? tiling_->batchSize - start : CHUNK;
            SyncPipes<HardEvent::V_MTE3>();
            DataCopyPad(requestMissCounts_[start], zeros, {1, static_cast<uint16_t>(valid * sizeof(int32_t)), 0, 0});
            SyncPipes<HardEvent::MTE3_V>();
        }
    }

private:
    __aicore__ inline bool LoadRequest(uint32_t batch, int32_t &startValue, int32_t &endValue, int32_t &row)
    {
        endValue = queryEnds_.GetValue(batch);
        startValue = batch == 0U ? 0 : queryEnds_.GetValue(batch - 1U);
        const int32_t state = states_.GetValue(batch);
        row = entries_.GetValue(batch);
        const int32_t keyLength = keyLengths_.GetValue(batch);
        const bool validRange = startValue >= 0 && endValue > startValue &&
                                endValue <= static_cast<int32_t>(tiling_->totalQueryRows) &&
                                endValue - startValue <= static_cast<int32_t>(MAX_QUERIES_PER_REQUEST);
        bool uniqueRow = true;
        if (row >= 0) {
            for (uint32_t other = 0; other < tiling_->batchSize; ++other) {
                if (other == batch) {
                    continue;
                }
                const int32_t otherState = states_.GetValue(other);
                if ((otherState == -3 || otherState == -2 || otherState == -1) && entries_.GetValue(other) == row) {
                    uniqueRow = false;
                    break;
                }
            }
        }
        return validRange && state == -3 && row >= 0 && row < static_cast<int32_t>(tiling_->poolSize) && uniqueRow &&
               keyLength >= endValue - startValue && keyLength <= static_cast<int32_t>(tiling_->sourceCapacity);
    }

    TPipe *pipe_ = nullptr;
    bool fastDenseAlreadyPublished_ = false;
    const FusedQuantLightningIndexerManageTilingData *tiling_ = nullptr;
    TBuf<TPosition::VECCALC> identityBaseBuf_;
    TBuf<TPosition::VECCALC> identityValuesBuf_;
    TBuf<TPosition::VECCALC> zeroCountsBuf_;
    GlobalTensor<int32_t> queryEnds_;
    GlobalTensor<int32_t> keyLengths_;
    GlobalTensor<int32_t> states_;
    GlobalTensor<int32_t> entries_;
    GlobalTensor<int32_t> pool_;
    GlobalTensor<int32_t> topkSources_;
    GlobalTensor<int32_t> topkDestinations_;
    GlobalTensor<int32_t> routeMissCounts_;
    GlobalTensor<int32_t> requestMissCounts_;
};

class CacheManagementPublisher {
public:
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR userWorkspace, GM_ADDR actualSeqLengthsQuery,
                                GM_ADDR actualSeqLengthsKey, GM_ADDR offloadSeqLengthsKey, GM_ADDR numCacheTokens,
                                GM_ADDR requestState, GM_ADDR reqPoolEntries, GM_ADDR cacheSlotsPool,
                                GM_ADDR topkSrcIds, GM_ADDR topkDstSlots, GM_ADDR topkMissCounts, GM_ADDR missSrcIds,
                                GM_ADDR missDstSlots, GM_ADDR missCounts, GM_ADDR fastRoutePairs,
                                GM_ADDR fastRouteCounts, GM_ADDR fastRouteThresholds,
                                const FusedQuantLightningIndexerManageTilingData *tiling)
    {
        pipe_ = pipe;
        tiling_ = tiling;
        queryEnds_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQuery);
        keyLengths_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsKey);
        prefixLengths_.SetGlobalBuffer((__gm__ int32_t *)offloadSeqLengthsKey);
        cacheTokens_.SetGlobalBuffer((__gm__ int32_t *)numCacheTokens);
        states_.SetGlobalBuffer((__gm__ int32_t *)requestState);
        entries_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        pool_.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
        topkSources_.SetGlobalBuffer((__gm__ int32_t *)topkSrcIds);
        topkDestinations_.SetGlobalBuffer((__gm__ int32_t *)topkDstSlots);
        routeMissCounts_.SetGlobalBuffer((__gm__ int32_t *)topkMissCounts);
        missSources_.SetGlobalBuffer((__gm__ int32_t *)missSrcIds);
        missDestinations_.SetGlobalBuffer((__gm__ int32_t *)missDstSlots);
        requestMissCounts_.SetGlobalBuffer((__gm__ int32_t *)missCounts);
        useFastPairs_ = fastRoutePairs != nullptr;
        if (useFastPairs_) {
            fastRoutePairs_.SetGlobalBuffer((__gm__ int32_t *)fastRoutePairs);
            fastRouteCounts_.SetGlobalBuffer((__gm__ int32_t *)fastRouteCounts);
            fastRouteThresholds_.SetGlobalBuffer((__gm__ uint16_t *)fastRouteThresholds);
            fastScoreWorkspace_.SetGlobalBuffer((__gm__ uint16_t *)userWorkspace);
        }
        requestStatus_.SetGlobalBuffer((__gm__ int32_t *)(userWorkspace + tiling_->managementWorkspaceOffset));
        routeStatus_.SetGlobalBuffer((__gm__ int32_t *)(userWorkspace + tiling_->managementWorkspaceOffset +
                                                        static_cast<uint64_t>(tiling_->batchSize) * 32U));
        pipe_->InitBuffer(topkBuf_, TOPK * sizeof(int32_t));
        pipe_->InitBuffer(slotBuf_, TOPK * sizeof(int32_t));
        pipe_->InitBuffer(sourceBuf_, TOPK * sizeof(int32_t));
        pipe_->InitBuffer(cacheChunkBuf_, 2048U * sizeof(int32_t));
        pipe_->InitBuffer(hashBuf_, 16384U * sizeof(uint32_t));
        pipe_->InitBuffer(residentBuf_, 12288U * sizeof(int32_t));
        pipe_->InitBuffer(protectedSlotBuf_, 12288U * sizeof(uint8_t));
        pipe_->InitBuffer(victimBuf_, 8192U * sizeof(int32_t));
        pipe_->InitBuffer(scalarBuf_, 32U);
    }

    __aicore__ inline void Process(uint32_t owner, uint32_t owners)
    {
        for (uint32_t batch = owner; batch < tiling_->batchSize; batch += owners) {
            const int32_t state = states_.GetValue(batch);
            if (state == -2 || (state == -1 && !useFastPairs_)) {
                ProcessRequest(batch);
            }
        }
    }

    __aicore__ inline void PublishCounts()
    {
        for (uint32_t batch = 0; batch < tiling_->batchSize; ++batch) {
            const int32_t state = states_.GetValue(batch);
            if (state != -2 && (state != -1 || useFastPairs_)) {
                continue;
            }
            const int32_t end = queryEnds_.GetValue(batch);
            const int32_t start = batch == 0U ? 0 : queryEnds_.GetValue(batch - 1U);
            if (start >= 0 && end > start && end <= static_cast<int32_t>(tiling_->totalQueryRows)) {
                for (int32_t route = start; route < end; ++route) {
                    routeMissCounts_.SetValue(static_cast<uint32_t>(route),
                                              routeStatus_.GetValue(static_cast<uint32_t>(route) * 8U));
                }
            }
            // A single-request launch has no compact-count false sharing.
            // Its owner already published the count directly; avoid reading
            // a staged scalar from workspace for this degenerate batch.
            if (tiling_->batchSize != 1U)
                requestMissCounts_.SetValue(batch, requestStatus_.GetValue(batch * 8U));
        }
    }

private:
    static constexpr uint32_t HASH_CAPACITY = 16384U;
    static constexpr uint32_t HASH_MASK = HASH_CAPACITY - 1U;
    static constexpr uint32_t TOKEN_MASK = (1U << 18) - 1U;
    static constexpr uint32_t SLOT_SHIFT = 18U;
    static constexpr uint32_t MISS_SLOT = 0x3ffeU;
    static constexpr uint32_t HASH_EMPTY = 0xffffffffU;
    static constexpr uint32_t TAIL_CAPACITY = 256U;

    __aicore__ inline uint32_t MinU32(uint32_t left, uint32_t right) const
    {
        return left < right ? left : right;
    }

    __aicore__ inline bool QueryRange(uint32_t batch, uint32_t &start, uint32_t &end) const
    {
        const int32_t endValue = queryEnds_.GetValue(batch);
        const int32_t startValue = batch == 0U ? 0 : queryEnds_.GetValue(batch - 1U);
        if (startValue < 0 || endValue <= startValue || endValue > static_cast<int32_t>(tiling_->totalQueryRows) ||
            endValue - startValue > static_cast<int32_t>(MAX_QUERIES_PER_REQUEST)) {
            return false;
        }
        start = static_cast<uint32_t>(startValue);
        end = static_cast<uint32_t>(endValue);
        return true;
    }

    __aicore__ inline bool PoolRowIsUnique(uint32_t batch, int32_t row) const
    {
        for (uint32_t other = 0; other < tiling_->batchSize; ++other) {
            if (other == batch) {
                continue;
            }
            const int32_t otherState = states_.GetValue(other);
            if ((otherState == -3 || otherState == -2 || otherState == -1) && entries_.GetValue(other) == row) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool Validate(uint32_t batch, uint32_t queryCount, int32_t row, uint32_t &prefixLength,
                                    uint32_t &finalLength, uint32_t &budget) const
    {
        const int32_t prefix = prefixLengths_.GetValue(batch);
        const int32_t finalValue = keyLengths_.GetValue(batch);
        const int32_t cache = cacheTokens_.GetValue(batch);
        if (row < 0 || row >= static_cast<int32_t>(tiling_->poolSize) || !PoolRowIsUnique(batch, row) ||
            prefix < static_cast<int32_t>(TOPK) || prefix % static_cast<int32_t>(BLOCK_SIZE) != 0 ||
            finalValue < static_cast<int32_t>(queryCount) ||
            finalValue > static_cast<int32_t>(tiling_->sourceCapacity) ||
            prefix > finalValue - static_cast<int32_t>(queryCount - 1U) || finalValue - prefix < 0 ||
            finalValue - prefix > 256 || (cache != 8192 && cache != 10240 && cache != 12288) || cache > prefix ||
            cache > static_cast<int32_t>(MISS_CAPACITY)) {
            return false;
        }
        prefixLength = static_cast<uint32_t>(prefix);
        finalLength = static_cast<uint32_t>(finalValue);
        budget = static_cast<uint32_t>(cache);
        return true;
    }

    __aicore__ inline uint32_t PackHashEntry(uint32_t token, uint32_t slot) const
    {
        return (slot << SLOT_SHIFT) | token;
    }

    __aicore__ inline void InitHash(LocalTensor<uint32_t> hash)
    {
        Duplicate(hash, HASH_EMPTY, HASH_CAPACITY);
        SyncPipes<HardEvent::V_S>();
    }

    __aicore__ inline bool FindHashPosition(LocalTensor<uint32_t> hash, int32_t token, uint32_t &position,
                                            bool &found) const
    {
        position = (static_cast<uint32_t>(token) * 2654435761U) & HASH_MASK;
        for (uint32_t probe = 0; probe < HASH_CAPACITY; ++probe) {
            const uint32_t entry = hash.GetValue(position);
            if (entry != HASH_EMPTY && (entry & TOKEN_MASK) == static_cast<uint32_t>(token)) {
                found = true;
                return true;
            }
            if (entry == HASH_EMPTY) {
                found = false;
                return true;
            }
            position = (position + 1U) & HASH_MASK;
        }
        found = false;
        return false;
    }

    __aicore__ inline void LoadTopk(uint32_t queryRow, LocalTensor<int32_t> topk)
    {
        DataCopyExtParams copy{1, TOPK * sizeof(int32_t), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(topk, topkSources_[static_cast<uint64_t>(queryRow) * TOPK], copy, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
    }

    __aicore__ inline void StoreRange(GlobalTensor<int32_t> destination, LocalTensor<int32_t> source, uint32_t count)
    {
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
        DataCopyPad<int32_t, PaddingMode::Normal>(destination, source, copy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
    }

    __aicore__ inline void FillGlobalRange(GlobalTensor<int32_t> destination, uint32_t count, int32_t value)
    {
        LocalTensor<int32_t> scratch = cacheChunkBuf_.Get<int32_t>();
        Duplicate(scratch, value, 2048U);
        PipeBarrier<PIPE_V>();
        SyncPipes<HardEvent::V_MTE3>();
        for (uint32_t offset = 0; offset < count; offset += 2048U) {
            const uint32_t length = MinU32(2048U, count - offset);
            DataCopyExtParams copy{1, static_cast<uint32_t>(length * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad<int32_t, PaddingMode::Normal>(destination[offset], scratch, copy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        }
    }

    // First-fill owns the complete row. Reuse one immutable -1 block for all
    // disjoint DMA writes and wait only once before installing resident slots.
    __aicore__ inline void ClearFirstFillPool(uint64_t poolBase)
    {
        constexpr uint32_t CHUNK = 2048U;
        LocalTensor<int32_t> invalid = cacheChunkBuf_.Get<int32_t>();
        Duplicate(invalid, -1, CHUNK);
        PipeBarrier<PIPE_V>();
        SyncPipes<HardEvent::V_MTE3>();
        for (uint32_t source = 0; source < tiling_->sourceCapacity; source += CHUNK) {
            const uint32_t count = MinU32(CHUNK, tiling_->sourceCapacity - source);
            DataCopyPad(pool_[poolBase + source], invalid, {1, static_cast<uint16_t>(count * sizeof(int32_t)), 0, 0});
        }
        SyncPipes<HardEvent::MTE3_S>();
    }

    // The resident list is contiguous in UB and destinations are 0..C-1.
    // Publish both lists with DMA instead of two synchronized scalar stores
    // per resident. Keep the proven 2048-element Arange bound.
    __aicore__ inline void PublishFirstFillLists(uint32_t batch, uint32_t budget, LocalTensor<int32_t> residentSources)
    {
        constexpr uint32_t CHUNK = 2048U;
        const uint64_t missBase = static_cast<uint64_t>(batch) * MISS_CAPACITY;
        StoreRange(missSources_[missBase], residentSources, budget);
        LocalTensor<int32_t> destinations = cacheChunkBuf_.Get<int32_t>();
        for (uint32_t slot = 0; slot < budget; slot += CHUNK) {
            const uint32_t count = MinU32(CHUNK, budget - slot);
            Arange<int32_t>(destinations, static_cast<int32_t>(slot), 1, static_cast<int32_t>(CHUNK));
            PipeBarrier<PIPE_V>();
            SyncPipes<HardEvent::V_MTE3>();
            DataCopyPad(missDestinations_[missBase + slot], destinations,
                        {1, static_cast<uint16_t>(count * sizeof(int32_t)), 0, 0});
            SyncPipes<HardEvent::MTE3_V>();
        }
    }

    __aicore__ inline void WriteScalar(GlobalTensor<int32_t> destination, int32_t value)
    {
        LocalTensor<int32_t> scalar = scalarBuf_.Get<int32_t>();
        scalar.SetValue(0, value);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyParams one{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
        DataCopyPad(destination, scalar, one);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void WritePool(uint64_t offset, int32_t value)
    {
        WriteScalar(pool_[offset], value);
    }

    __aicore__ inline void WriteMissPair(uint32_t batch, uint32_t index, int32_t source, int32_t destination)
    {
        const uint64_t offset = static_cast<uint64_t>(batch) * MISS_CAPACITY + index;
        WriteScalar(missSources_[offset], source);
        WriteScalar(missDestinations_[offset], destination);
    }

    __aicore__ inline void WriteRequestStatus(uint32_t batch, int32_t value)
    {
        if (tiling_->batchSize == 1U)
            WriteScalar(requestMissCounts_[batch], value);
        else
            WriteScalar(requestStatus_[batch * 8U], value);
    }

    __aicore__ inline void WriteRouteStatus(uint32_t route, int32_t value)
    {
        WriteScalar(routeStatus_[route * 8U], value);
    }

    __aicore__ inline void MarkError(uint32_t batch, uint32_t queryStart, uint32_t queryEnd)
    {
        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        LocalTensor<int32_t> slots = slotBuf_.Get<int32_t>();
        Duplicate(sources, -1, TOPK);
        Duplicate(slots, -1, TOPK);
        SyncPipes<HardEvent::V_S>();
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            const uint64_t base = static_cast<uint64_t>(route) * TOPK;
            StoreRange(topkSources_[base], sources, TOPK);
            StoreRange(topkDestinations_[base], slots, TOPK);
            WriteRouteStatus(route, -1);
        }
        WriteRequestStatus(batch, -1);
    }

    __aicore__ inline bool BuildUnion(uint32_t queryStart, uint32_t queryEnd, uint32_t prefixLength,
                                      uint32_t finalLength, uint32_t budget, LocalTensor<int32_t> topk,
                                      LocalTensor<uint32_t> hash)
    {
        InitHash(hash);
        uint32_t unionCount = 0;
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            LoadTopk(route, topk);
            const uint32_t visible = finalLength - (queryEnd - 1U - route);
            for (uint32_t index = 0; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                if (token < 0 || static_cast<uint32_t>(token) >= visible) {
                    return false;
                }
                if (static_cast<uint32_t>(token) >= prefixLength) {
                    continue;
                }
                uint32_t position = 0;
                bool found = false;
                if (!FindHashPosition(hash, token, position, found)) {
                    return false;
                }
                if (!found) {
                    if (unionCount >= budget) {
                        return false;
                    }
                    hash.SetValue(position, PackHashEntry(static_cast<uint32_t>(token), MISS_SLOT));
                    ++unionCount;
                }
            }
        }
        return true;
    }

    __aicore__ inline bool CommitFirstFill(uint32_t batch, uint32_t queryStart, uint32_t queryEnd, uint32_t row,
                                           uint32_t prefixLength, uint32_t budget, LocalTensor<int32_t> topk,
                                           LocalTensor<uint32_t> hash)
    {
        const uint64_t poolBase = static_cast<uint64_t>(row) * tiling_->sourceCapacity;
        LocalTensor<int32_t> residentSources = residentBuf_.Get<int32_t>();
        uint32_t resident = 0;
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            LoadTopk(route, topk);
            for (uint32_t index = 0; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                if (static_cast<uint32_t>(token) >= prefixLength) {
                    continue;
                }
                uint32_t position = 0;
                bool found = false;
                if (!FindHashPosition(hash, token, position, found) || !found) {
                    return false;
                }
                if ((hash.GetValue(position) >> SLOT_SHIFT) == MISS_SLOT) {
                    hash.SetValue(position, PackHashEntry(static_cast<uint32_t>(token), resident));
                    residentSources.SetValue(resident, token);
                    ++resident;
                }
            }
        }
        for (uint32_t token = 0; token < prefixLength && resident < budget; ++token) {
            uint32_t position = 0;
            bool found = false;
            if (!FindHashPosition(hash, static_cast<int32_t>(token), position, found)) {
                return false;
            }
            if (found) {
                continue;
            }
            hash.SetValue(position, PackHashEntry(token, resident));
            residentSources.SetValue(resident, static_cast<int32_t>(token));
            ++resident;
        }
        if (resident != budget) {
            return false;
        }

        // Every fallible metadata/union/filler decision is complete.  Only
        // now replace the caller-owned row and publish the C copy pairs.
        ClearFirstFillPool(poolBase);
        for (uint32_t index = 0; index < budget; ++index) {
            const int32_t token = residentSources.GetValue(index);
            WritePool(poolBase + static_cast<uint32_t>(token), static_cast<int32_t>(index));
        }
        PublishFirstFillLists(batch, budget, residentSources);

        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        LocalTensor<int32_t> slots = slotBuf_.Get<int32_t>();
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            LoadTopk(route, topk);
            uint32_t missCount = 0;
            for (uint32_t index = 0; index < TOPK; ++index) {
                if (static_cast<uint32_t>(topk.GetValue(index)) < prefixLength) {
                    ++missCount;
                }
            }
            Duplicate(sources, -1, TOPK);
            Duplicate(slots, -1, TOPK);
            SyncPipes<HardEvent::V_S>();
            uint32_t missCursor = 0;
            uint32_t hitCursor = missCount;
            for (uint32_t index = 0; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                const bool prefix = static_cast<uint32_t>(token) < prefixLength;
                uint32_t slot = budget + static_cast<uint32_t>(token) % TAIL_CAPACITY;
                if (prefix) {
                    uint32_t position = 0;
                    bool found = false;
                    if (!FindHashPosition(hash, token, position, found) || !found) {
                        return false;
                    }
                    slot = hash.GetValue(position) >> SLOT_SHIFT;
                }
                const uint32_t output = prefix ? missCursor++ : hitCursor++;
                sources.SetValue(output, token);
                slots.SetValue(output, static_cast<int32_t>(slot));
            }
            const uint64_t base = static_cast<uint64_t>(route) * TOPK;
            StoreRange(topkSources_[base], sources, TOPK);
            StoreRange(topkDestinations_[base], slots, TOPK);
            WriteRouteStatus(route, static_cast<int32_t>(missCount));
        }
        WriteRequestStatus(batch, static_cast<int32_t>(budget));
        return true;
    }

    __aicore__ inline uint32_t HashVictimScanSeed(uint32_t finalLength) const
    {
        uint32_t value = finalLength;
        value ^= value >> 16U;
        value *= 0x7feb352dU;
        value ^= value >> 15U;
        value *= 0x846ca68bU;
        value ^= value >> 16U;
        return value;
    }

    __aicore__ inline bool ValidateSparsePool(uint64_t poolBase, uint32_t prefixLength, uint32_t budget,
                                              LocalTensor<int32_t> cacheChunk, LocalTensor<uint8_t> protectedSlots)
    {
        Duplicate(protectedSlots, static_cast<uint8_t>(0), budget);
        SyncPipes<HardEvent::V_S>();
        uint32_t residentCount = 0;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        for (uint32_t chunkBase = 0; chunkBase < tiling_->sourceCapacity; chunkBase += 2048U) {
            const uint32_t chunkLength = MinU32(2048U, tiling_->sourceCapacity - chunkBase);
            DataCopyExtParams copy{1, static_cast<uint32_t>(chunkLength * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(cacheChunk, pool_[poolBase + chunkBase], copy, pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
            for (uint32_t offset = 0; offset < chunkLength; ++offset) {
                const uint32_t source = chunkBase + offset;
                const int32_t slot = cacheChunk.GetValue(offset);
                if (source >= prefixLength) {
                    if (slot != -1) {
                        return false;
                    }
                    continue;
                }
                if (slot == -1) {
                    continue;
                }
                if (slot < 0 || slot >= static_cast<int32_t>(budget)) {
                    return false;
                }
                const uint32_t slotU32 = static_cast<uint32_t>(slot);
                if ((protectedSlots.GetValue(slotU32) & 1U) != 0U) {
                    return false;
                }
                protectedSlots.SetValue(slotU32, static_cast<uint8_t>(1U));
                ++residentCount;
            }
        }
        return residentCount == budget;
    }

    __aicore__ inline bool PrepareSteadyPool(uint64_t poolBase, uint32_t prefixLength, uint32_t budget,
                                             LocalTensor<int32_t> cacheChunk, LocalTensor<uint8_t> protectedSlots,
                                             bool &identityRow)
    {
        identityRow = false;
        if (budget >= tiling_->sourceCapacity || pool_.GetValue(poolBase + budget) != static_cast<int32_t>(budget)) {
            return ValidateSparsePool(poolBase, prefixLength, budget, cacheChunk, protectedSlots);
        }

        // The standard lifecycle uses pool[C] == C as the identity-row probe.
        // Validate the complete row before treating it as identity so a
        // corrupted sparse row cannot be partially normalized on failure.
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        for (uint32_t chunkBase = 0; chunkBase < tiling_->sourceCapacity; chunkBase += 2048U) {
            const uint32_t chunkLength = MinU32(2048U, tiling_->sourceCapacity - chunkBase);
            DataCopyExtParams copy{1, static_cast<uint32_t>(chunkLength * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(cacheChunk, pool_[poolBase + chunkBase], copy, pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
            for (uint32_t offset = 0; offset < chunkLength; ++offset) {
                const uint32_t source = chunkBase + offset;
                if (cacheChunk.GetValue(offset) != static_cast<int32_t>(source)) {
                    return false;
                }
            }
        }
        Duplicate(protectedSlots, static_cast<uint8_t>(1), budget);
        SyncPipes<HardEvent::V_S>();
        identityRow = true;
        return true;
    }

    __aicore__ inline bool BuildSteadyUnion(uint32_t queryStart, uint32_t queryEnd, uint64_t poolBase,
                                            uint32_t prefixLength, uint32_t finalLength, uint32_t budget,
                                            LocalTensor<int32_t> topk, LocalTensor<uint32_t> hash,
                                            LocalTensor<uint8_t> protectedSlots, bool identityRow, uint32_t &missCount)
    {
        InitHash(hash);
        missCount = 0;
        uint32_t uniqueHitCount = 0;
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            LoadTopk(route, topk);
            const uint32_t visible = finalLength - (queryEnd - 1U - route);
            for (uint32_t index = 0; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                if (token < 0 || static_cast<uint32_t>(token) >= visible) {
                    return false;
                }
                if (static_cast<uint32_t>(token) >= prefixLength) {
                    continue;
                }
                uint32_t position = 0;
                bool found = false;
                if (!FindHashPosition(hash, token, position, found)) {
                    return false;
                }
                if (found) {
                    continue;
                }
                const int32_t slot = identityRow ? (static_cast<uint32_t>(token) < budget ? token : -1) :
                                                   pool_.GetValue(poolBase + static_cast<uint32_t>(token));
                if (slot >= 0 && slot < static_cast<int32_t>(budget)) {
                    const uint32_t slotU32 = static_cast<uint32_t>(slot);
                    if ((protectedSlots.GetValue(slotU32) & 1U) == 0U) {
                        return false;
                    }
                    hash.SetValue(position, PackHashEntry(static_cast<uint32_t>(token), slotU32));
                    protectedSlots.SetValue(slotU32, static_cast<uint8_t>(protectedSlots.GetValue(slotU32) | 2U));
                    ++uniqueHitCount;
                } else if (slot == -1) {
                    if (missCount >= 8192U) {
                        return false;
                    }
                    hash.SetValue(position, PackHashEntry(static_cast<uint32_t>(token), MISS_SLOT));
                    ++missCount;
                } else {
                    return false;
                }
            }
        }
        return uniqueHitCount + missCount <= budget;
    }

    // Stage 1 of the A5 fast payload publishes a sorted prefix-miss pair
    // list per route. Consume its source words directly instead of deriving
    // a second miss list from the TopK/pool. The general hash and victim
    // planner deliberately remain the correctness baseline for this stage.
    __aicore__ inline bool BuildSteadyUnionFromFastPairs(uint32_t batch, uint32_t queryStart, uint32_t queryEnd,
                                                         uint64_t poolBase, uint32_t prefixLength, uint32_t finalLength,
                                                         uint32_t budget, LocalTensor<int32_t> topk,
                                                         LocalTensor<uint32_t> hash,
                                                         LocalTensor<uint8_t> protectedSlots, uint32_t &missCount)
    {
        InitHash(hash);
        missCount = 0U;
        uint32_t uniqueHitCount = 0U;
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            LoadTopk(route, topk);
            const uint32_t localRoute = route - queryStart;
            const uint32_t workspaceRow = batch * 4U + localRoute;
            const int32_t rawMisses = fastRouteCounts_.GetValue(workspaceRow);
            if (rawMisses < 0 || rawMisses > static_cast<int32_t>(TOPK))
                return false;
            const uint32_t routeMisses = static_cast<uint32_t>(rawMisses);
            const uint32_t visible = finalLength - (queryEnd - 1U - route);
            int32_t previousMiss = -1;
            for (uint32_t index = 0U; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                if (token < 0 || static_cast<uint32_t>(token) >= visible)
                    return false;
                if (index < routeMisses) {
                    const uint64_t pairOffset = static_cast<uint64_t>(workspaceRow) * TOPK * 2U + index * 2U + 1U;
                    const int32_t pairSource = fastRoutePairs_.GetValue(pairOffset);
                    if (token != pairSource || token <= previousMiss || static_cast<uint32_t>(token) >= prefixLength ||
                        pool_.GetValue(poolBase + static_cast<uint32_t>(token)) != -1)
                        return false;
                    previousMiss = token;
                } else if (static_cast<uint32_t>(token) < prefixLength) {
                    const int32_t slot = pool_.GetValue(poolBase + static_cast<uint32_t>(token));
                    if (slot < 0 || slot >= static_cast<int32_t>(budget))
                        return false;
                }
                if (static_cast<uint32_t>(token) >= prefixLength)
                    continue;
                uint32_t position = 0U;
                bool found = false;
                if (!FindHashPosition(hash, token, position, found))
                    return false;
                if (found)
                    continue;
                if (index < routeMisses) {
                    if (missCount >= 8192U)
                        return false;
                    hash.SetValue(position, PackHashEntry(static_cast<uint32_t>(token), MISS_SLOT));
                    ++missCount;
                } else {
                    const uint32_t slot =
                        static_cast<uint32_t>(pool_.GetValue(poolBase + static_cast<uint32_t>(token)));
                    if ((protectedSlots.GetValue(slot) & 1U) == 0U)
                        return false;
                    hash.SetValue(position, PackHashEntry(static_cast<uint32_t>(token), slot));
                    protectedSlots.SetValue(slot, static_cast<uint8_t>(protectedSlots.GetValue(slot) | 2U));
                    ++uniqueHitCount;
                }
            }
        }
        return uniqueHitCount + missCount <= budget;
    }

    // A5's score-threshold VF yields only definitely unselected resident
    // tokens. Selected hits are still checked against protectedSlots before
    // use; threshold ties and any remaining victims go to the exact scan.
    __aicore__ inline bool CollectFastVictims(uint32_t batch, uint32_t queryCount, uint64_t poolBase,
                                              uint32_t prefixLength, uint32_t budget, uint32_t firstChunk,
                                              uint32_t missCount, uint32_t &assigned, uint32_t &missHashPosition,
                                              LocalTensor<int32_t> victimSources, LocalTensor<uint8_t> protectedSlots,
                                              LocalTensor<uint32_t> hash)
    {
        constexpr uint32_t CHUNK = 2048U;
        constexpr uint32_t SCORE_BYTES = CHUNK * sizeof(uint16_t);
        LocalTensor<uint8_t> scratch = residentBuf_.Get<uint8_t>();
        LocalTensor<uint16_t> score0 = scratch.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> score1 = scratch[SCORE_BYTES].ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> score2 = scratch[SCORE_BYTES * 2U].ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> score3 = scratch[SCORE_BYTES * 3U].ReinterpretCast<uint16_t>();
        LocalTensor<int32_t> slots32 = scratch[SCORE_BYTES * 4U].ReinterpretCast<int32_t>();
        LocalTensor<int16_t> slots16 = scratch[SCORE_BYTES * 4U + CHUNK * sizeof(int32_t)].ReinterpretCast<int16_t>();
        LocalTensor<uint32_t> compact =
            scratch[SCORE_BYTES * 4U + CHUNK * (sizeof(int32_t) + sizeof(int16_t))].ReinterpretCast<uint32_t>();
        uint16_t thresholds[4] = {0xffffU, 0xffffU, 0xffffU, 0xffffU};
        for (uint32_t route = 0U; route < queryCount; ++route)
            thresholds[route] = fastRouteThresholds_.GetValue((batch * 4U + route) * 16U);
        const uint32_t chunks = (prefixLength + CHUNK - 1U) / CHUNK;
        const uint64_t scoreBase = static_cast<uint64_t>(batch) * 4U * tiling_->fastScoreRowStride;
        DataCopyPadExtParams<uint16_t> scorePad{false, 0, 0, 0};
        DataCopyPadExtParams<int32_t> slotPad{false, 0, 0, 0};
        for (uint32_t visit = 0U; visit < chunks && assigned < missCount; ++visit) {
            const uint32_t chunkBase = ((firstChunk + visit) % chunks) * CHUNK;
            const uint32_t chunkLength = MinU32(CHUNK, prefixLength - chunkBase);
            const DataCopyExtParams scoreCopy{1, static_cast<uint32_t>(chunkLength * sizeof(uint16_t)), 0, 0, 0};
            DataCopyPad(score0, fastScoreWorkspace_[scoreBase + chunkBase], scoreCopy, scorePad);
            if (queryCount > 1U)
                DataCopyPad(score1, fastScoreWorkspace_[scoreBase + tiling_->fastScoreRowStride + chunkBase], scoreCopy,
                            scorePad);
            else
                Duplicate(score1, static_cast<uint16_t>(0), chunkLength);
            if (queryCount > 2U)
                DataCopyPad(score2, fastScoreWorkspace_[scoreBase + tiling_->fastScoreRowStride * 2U + chunkBase],
                            scoreCopy, scorePad);
            else
                Duplicate(score2, static_cast<uint16_t>(0), chunkLength);
            if (queryCount > 3U)
                DataCopyPad(score3, fastScoreWorkspace_[scoreBase + tiling_->fastScoreRowStride * 3U + chunkBase],
                            scoreCopy, scorePad);
            else
                Duplicate(score3, static_cast<uint16_t>(0), chunkLength);
            DataCopyPad(slots32, pool_[poolBase + chunkBase],
                        {1, static_cast<uint32_t>(chunkLength * sizeof(int32_t)), 0, 0, 0}, slotPad);
            SyncPipes<HardEvent::MTE2_V>();
            Cast(slots16, slots32, RoundMode::CAST_NONE, chunkLength);
            PipeBarrier<PIPE_V>();
            VllmA5LiManageC8FastVictimVF::CompactEligiblePayloads(
                (__ubuf__ uint32_t *)compact.GetPhyAddr(), (__ubuf__ uint16_t *)score0.GetPhyAddr(),
                (__ubuf__ uint16_t *)score1.GetPhyAddr(), (__ubuf__ uint16_t *)score2.GetPhyAddr(),
                (__ubuf__ uint16_t *)score3.GetPhyAddr(), (__ubuf__ uint16_t *)slots16.GetPhyAddr(), thresholds[0],
                thresholds[1], thresholds[2], thresholds[3], static_cast<uint16_t>(budget), chunkBase,
                chunkLength / VllmA5LiManageC8FastVictimVF::B32_LANES);
            const uint32_t compactCount = static_cast<uint32_t>(GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t));
            PipeBarrier<PIPE_V>();
            SyncPipes<HardEvent::V_S>();
            for (uint32_t index = 0U; index < compactCount && assigned < missCount; ++index) {
                const uint32_t payload = compact.GetValue(index);
                const uint32_t slot = payload >> SLOT_SHIFT;
                if (slot >= budget || protectedSlots.GetValue(slot) != 1U)
                    continue;
                while (missHashPosition < HASH_CAPACITY &&
                       (hash.GetValue(missHashPosition) == HASH_EMPTY ||
                        (hash.GetValue(missHashPosition) >> SLOT_SHIFT) != MISS_SLOT))
                    ++missHashPosition;
                if (missHashPosition >= HASH_CAPACITY)
                    return false;
                victimSources.SetValue(assigned, static_cast<int32_t>(payload & TOKEN_MASK));
                protectedSlots.SetValue(slot, static_cast<uint8_t>(1U | 4U));
                hash.SetValue(missHashPosition, PackHashEntry(hash.GetValue(missHashPosition) & TOKEN_MASK, slot));
                ++assigned;
                ++missHashPosition;
            }
        }
        return true;
    }

    __aicore__ inline bool PlanVictims(uint32_t batch, uint32_t queryCount, uint64_t poolBase, uint32_t prefixLength,
                                       uint32_t finalLength, uint32_t budget, uint32_t missCount,
                                       LocalTensor<int32_t> cacheChunk, LocalTensor<int32_t> victimSources,
                                       LocalTensor<uint8_t> protectedSlots, LocalTensor<uint32_t> hash)
    {
        if (missCount == 0U) {
            return true;
        }
        uint32_t assigned = 0;
        uint32_t missHashPosition = 0;
        const uint32_t chunks = (prefixLength + 2047U) / 2048U;
        const uint32_t firstChunk = HashVictimScanSeed(finalLength) % chunks;
        if (useFastPairs_ &&
            !CollectFastVictims(batch, queryCount, poolBase, prefixLength, budget, firstChunk, missCount, assigned,
                                missHashPosition, victimSources, protectedSlots, hash))
            return false;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        for (uint32_t visit = 0; visit < chunks && assigned < missCount; ++visit) {
            const uint32_t chunkBase = ((firstChunk + visit) % chunks) * 2048U;
            const uint32_t chunkLength = MinU32(2048U, prefixLength - chunkBase);
            DataCopyExtParams copy{1, static_cast<uint32_t>(chunkLength * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(cacheChunk, pool_[poolBase + chunkBase], copy, pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
            for (uint32_t offset = 0; offset < chunkLength && assigned < missCount; ++offset) {
                const int32_t slot = cacheChunk.GetValue(offset);
                if (slot < 0 || slot >= static_cast<int32_t>(budget)) {
                    continue;
                }
                const uint32_t slotU32 = static_cast<uint32_t>(slot);
                if (protectedSlots.GetValue(slotU32) != 1U) {
                    continue;
                }
                while (missHashPosition < HASH_CAPACITY &&
                       (hash.GetValue(missHashPosition) == HASH_EMPTY ||
                        (hash.GetValue(missHashPosition) >> SLOT_SHIFT) != MISS_SLOT)) {
                    ++missHashPosition;
                }
                if (missHashPosition >= HASH_CAPACITY) {
                    return false;
                }
                victimSources.SetValue(assigned, static_cast<int32_t>(chunkBase + offset));
                protectedSlots.SetValue(slotU32, static_cast<uint8_t>(1U | 4U));
                hash.SetValue(missHashPosition, PackHashEntry(hash.GetValue(missHashPosition) & TOKEN_MASK, slotU32));
                ++assigned;
                ++missHashPosition;
            }
        }
        return assigned == missCount;
    }

    __aicore__ inline void CommitSteadyUpdates(uint32_t batch, uint64_t poolBase, uint32_t budget, uint32_t missCount,
                                               bool identityRow, LocalTensor<int32_t> victimSources,
                                               LocalTensor<uint8_t> protectedSlots, LocalTensor<uint32_t> hash)
    {
        if (identityRow && budget < tiling_->sourceCapacity) {
            FillGlobalRange(pool_[poolBase + budget], tiling_->sourceCapacity - budget, -1);
        }
        uint32_t cursor = 0;
        for (uint32_t position = 0; position < HASH_CAPACITY && cursor < missCount; ++position) {
            const uint32_t entry = hash.GetValue(position);
            if (entry == HASH_EMPTY) {
                continue;
            }
            const uint32_t slot = entry >> SLOT_SHIFT;
            if ((protectedSlots.GetValue(slot) & 4U) == 0U) {
                continue;
            }
            const int32_t missToken = static_cast<int32_t>(entry & TOKEN_MASK);
            const int32_t victimToken = victimSources.GetValue(cursor);
            WritePool(poolBase + static_cast<uint32_t>(victimToken), -1);
            WritePool(poolBase + static_cast<uint32_t>(missToken), static_cast<int32_t>(slot));
            WriteMissPair(batch, cursor, missToken, static_cast<int32_t>(slot));
            ++cursor;
        }
    }

    __aicore__ inline bool PublishSteadySelections(uint32_t queryStart, uint32_t queryEnd, uint32_t prefixLength,
                                                   uint32_t budget, LocalTensor<int32_t> topk,
                                                   LocalTensor<int32_t> sources, LocalTensor<int32_t> slots,
                                                   LocalTensor<uint8_t> protectedSlots, LocalTensor<uint32_t> hash)
    {
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            LoadTopk(route, topk);
            uint32_t routeMissCount = 0;
            for (uint32_t index = 0; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                if (static_cast<uint32_t>(token) >= prefixLength) {
                    continue;
                }
                uint32_t position = 0;
                bool found = false;
                if (!FindHashPosition(hash, token, position, found) || !found) {
                    return false;
                }
                const uint32_t slot = hash.GetValue(position) >> SLOT_SHIFT;
                if ((protectedSlots.GetValue(slot) & 4U) != 0U) {
                    ++routeMissCount;
                }
            }
            Duplicate(sources, -1, TOPK);
            Duplicate(slots, -1, TOPK);
            SyncPipes<HardEvent::V_S>();
            uint32_t missCursor = 0;
            uint32_t hitCursor = routeMissCount;
            for (uint32_t index = 0; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                uint32_t slot = budget + static_cast<uint32_t>(token) % TAIL_CAPACITY;
                bool isMiss = false;
                if (static_cast<uint32_t>(token) < prefixLength) {
                    uint32_t position = 0;
                    bool found = false;
                    if (!FindHashPosition(hash, token, position, found) || !found) {
                        return false;
                    }
                    slot = hash.GetValue(position) >> SLOT_SHIFT;
                    isMiss = (protectedSlots.GetValue(slot) & 4U) != 0U;
                }
                const uint32_t output = isMiss ? missCursor++ : hitCursor++;
                sources.SetValue(output, token);
                slots.SetValue(output, static_cast<int32_t>(slot));
            }
            const uint64_t base = static_cast<uint64_t>(route) * TOPK;
            StoreRange(topkSources_[base], sources, TOPK);
            StoreRange(topkDestinations_[base], slots, TOPK);
            WriteRouteStatus(route, static_cast<int32_t>(routeMissCount));
        }
        return true;
    }

#if FQLIM_EXPERIMENTAL_SIMT_SLOT_REPAIR
    // A5 Stage 1 already publishes sorted prefix misses followed by hits.
    // Build a sorted request union from its validated pair rows, then repair
    // only the miss destination prefixes with the A5 SIMT binary lookup.
    // Hit slots (including selected tail) remain the payload's old slots.
    __aicore__ inline void PublishFastSteadySelections(uint32_t batch, uint32_t queryStart, uint32_t queryEnd,
                                                       uint32_t missCount, LocalTensor<uint32_t> hash)
    {
        LocalTensor<int32_t> unionSources = victimBuf_.Get<int32_t>();
        LocalTensor<int32_t> unionDestinations = residentBuf_.Get<int32_t>();
        LocalTensor<int32_t> routePairs = unionDestinations[8192U];
        LocalTensor<int32_t> repaired = topkBuf_.Get<int32_t>();
        const uint32_t routeCount = queryEnd - queryStart;
        uint32_t lengths[4] = {0U, 0U, 0U, 0U};
        uint32_t cursors[4] = {0U, 0U, 0U, 0U};
        int32_t nextSources[4] = {-1, -1, -1, -1};
        for (uint32_t route = 0U; route < routeCount; ++route) {
            lengths[route] = static_cast<uint32_t>(fastRouteCounts_.GetValue(batch * 4U + route));
            if (lengths[route] != 0U)
                nextSources[route] =
                    fastRoutePairs_.GetValue((static_cast<uint64_t>(batch) * 4U + route) * TOPK * 2U + 1U);
        }
        uint32_t unionCursor = 0U;
        while (unionCursor < missCount) {
            int32_t source = static_cast<int32_t>(tiling_->sourceCapacity);
            for (uint32_t route = 0U; route < routeCount; ++route)
                if (cursors[route] < lengths[route] && nextSources[route] < source)
                    source = nextSources[route];
            uint32_t position = 0U;
            bool found = false;
            // The pair rows and hash were validated before any pool update.
            FindHashPosition(hash, source, position, found);
            unionSources.SetValue(unionCursor, source);
            unionDestinations.SetValue(unionCursor, static_cast<int32_t>(hash.GetValue(position) >> SLOT_SHIFT));
            ++unionCursor;
            for (uint32_t route = 0U; route < routeCount; ++route) {
                if (cursors[route] >= lengths[route] || nextSources[route] != source)
                    continue;
                ++cursors[route];
                if (cursors[route] < lengths[route])
                    nextSources[route] = fastRoutePairs_.GetValue(
                        (static_cast<uint64_t>(batch) * 4U + route) * TOPK * 2U + cursors[route] * 2U + 1U);
            }
        }
        SyncPipes<HardEvent::S_V>();
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const uint32_t length = lengths[route];
            if (length != 0U) {
                const uint64_t pairOffset = (static_cast<uint64_t>(batch) * 4U + route) * TOPK * 2U;
                DataCopyPad(routePairs, fastRoutePairs_[pairOffset],
                            {1, static_cast<uint32_t>(length * 2U * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
                SyncPipes<HardEvent::MTE2_V>();
                asc_vf_call<VllmA5LiManageC8FastSlotRepair::RepairMissSlots>(
                    dim3(VllmA5LiManageC8FastSlotRepair::THREADS), (__ubuf__ const int32_t *)routePairs.GetPhyAddr(),
                    (__ubuf__ const int32_t *)unionSources.GetPhyAddr(),
                    (__ubuf__ const int32_t *)unionDestinations.GetPhyAddr(), (__ubuf__ int32_t *)repaired.GetPhyAddr(),
                    length, missCount);
                // Match the reference's SIMT -> MTE2 hand-off before the
                // next route can overwrite this UB pair row.
                SyncPipes<HardEvent::V_MTE2>();
                SyncPipes<HardEvent::V_MTE3>();
                DataCopyPad(topkDestinations_[(static_cast<uint64_t>(queryStart) + route) * TOPK], repaired,
                            {1, static_cast<uint16_t>(length * sizeof(int32_t)), 0, 0});
                SyncPipes<HardEvent::MTE3_S>();
            }
            WriteRouteStatus(queryStart + route, static_cast<int32_t>(length));
        }
    }
#endif

    __aicore__ inline bool ProcessSteady(uint32_t batch, uint32_t queryStart, uint32_t queryEnd, uint32_t row,
                                         uint32_t prefixLength, uint32_t finalLength, uint32_t budget,
                                         LocalTensor<int32_t> topk, LocalTensor<uint32_t> hash)
    {
        const uint64_t poolBase = static_cast<uint64_t>(row) * tiling_->sourceCapacity;
        LocalTensor<int32_t> cacheChunk = cacheChunkBuf_.Get<int32_t>();
        LocalTensor<uint8_t> protectedSlots = protectedSlotBuf_.Get<uint8_t>();
        LocalTensor<int32_t> victimSources = victimBuf_.Get<int32_t>();
        bool identityRow = false;
        if (!PrepareSteadyPool(poolBase, prefixLength, budget, cacheChunk, protectedSlots, identityRow)) {
            return false;
        }
        uint32_t missCount = 0;
        const bool unionOk =
            useFastPairs_ ? BuildSteadyUnionFromFastPairs(batch, queryStart, queryEnd, poolBase, prefixLength,
                                                          finalLength, budget, topk, hash, protectedSlots, missCount) :
                            BuildSteadyUnion(queryStart, queryEnd, poolBase, prefixLength, finalLength, budget, topk,
                                             hash, protectedSlots, identityRow, missCount);
        if (!unionOk || !PlanVictims(batch, queryEnd - queryStart, poolBase, prefixLength, finalLength, budget,
                                     missCount, cacheChunk, victimSources, protectedSlots, hash)) {
            return false;
        }

        // Pool validation, complete hit protection and victim assignment all
        // finish before the first caller-owned mapping is changed.
        CommitSteadyUpdates(batch, poolBase, budget, missCount, identityRow, victimSources, protectedSlots, hash);
#if FQLIM_EXPERIMENTAL_SIMT_SLOT_REPAIR
        if (useFastPairs_ && missCount != 0U)
            PublishFastSteadySelections(batch, queryStart, queryEnd, missCount, hash);
        else
#endif
            if (!PublishSteadySelections(queryStart, queryEnd, prefixLength, budget, topk, sourceBuf_.Get<int32_t>(),
                                         slotBuf_.Get<int32_t>(), protectedSlots, hash))
            return false;
        WriteRequestStatus(batch, static_cast<int32_t>(missCount));
        return true;
    }

    __aicore__ inline void ProcessRequest(uint32_t batch)
    {
        uint32_t queryStart = 0;
        uint32_t queryEnd = 0;
        if (!QueryRange(batch, queryStart, queryEnd)) {
            const int32_t endValue = queryEnds_.GetValue(batch);
            const int32_t startValue = batch == 0U ? 0 : queryEnds_.GetValue(batch - 1U);
            const uint32_t safeStart = startValue > 0 ? static_cast<uint32_t>(startValue) : 0U;
            const uint32_t safeEnd = endValue > static_cast<int32_t>(safeStart) &&
                                             endValue <= static_cast<int32_t>(tiling_->totalQueryRows) ?
                                         static_cast<uint32_t>(endValue) :
                                         safeStart;
            MarkError(batch, safeStart, safeEnd);
            return;
        }
        const int32_t rowValue = entries_.GetValue(batch);
        uint32_t prefixLength = 0;
        uint32_t finalLength = 0;
        uint32_t budget = 0;
        if (!Validate(batch, queryEnd - queryStart, rowValue, prefixLength, finalLength, budget)) {
            MarkError(batch, queryStart, queryEnd);
            return;
        }
        LocalTensor<int32_t> topk = topkBuf_.Get<int32_t>();
        LocalTensor<uint32_t> hash = hashBuf_.Get<uint32_t>();
        const bool ok = states_.GetValue(batch) == -2 ?
                            BuildUnion(queryStart, queryEnd, prefixLength, finalLength, budget, topk, hash) &&
                                CommitFirstFill(batch, queryStart, queryEnd, static_cast<uint32_t>(rowValue),
                                                prefixLength, budget, topk, hash) :
                            ProcessSteady(batch, queryStart, queryEnd, static_cast<uint32_t>(rowValue), prefixLength,
                                          finalLength, budget, topk, hash);
        if (!ok) {
            MarkError(batch, queryStart, queryEnd);
        }
    }

    TPipe *pipe_ = nullptr;
    const FusedQuantLightningIndexerManageTilingData *tiling_ = nullptr;
    GlobalTensor<int32_t> queryEnds_;
    GlobalTensor<int32_t> keyLengths_;
    GlobalTensor<int32_t> prefixLengths_;
    GlobalTensor<int32_t> cacheTokens_;
    GlobalTensor<int32_t> states_;
    GlobalTensor<int32_t> entries_;
    GlobalTensor<int32_t> pool_;
    GlobalTensor<int32_t> topkSources_;
    GlobalTensor<int32_t> topkDestinations_;
    GlobalTensor<int32_t> routeMissCounts_;
    GlobalTensor<int32_t> missSources_;
    GlobalTensor<int32_t> missDestinations_;
    GlobalTensor<int32_t> requestMissCounts_;
    GlobalTensor<int32_t> fastRoutePairs_;
    GlobalTensor<int32_t> fastRouteCounts_;
    GlobalTensor<uint16_t> fastRouteThresholds_;
    GlobalTensor<uint16_t> fastScoreWorkspace_;
    bool useFastPairs_ = false;
    GlobalTensor<int32_t> requestStatus_;
    GlobalTensor<int32_t> routeStatus_;
    TBuf<TPosition::VECCALC> topkBuf_;
    TBuf<TPosition::VECCALC> slotBuf_;
    TBuf<TPosition::VECCALC> sourceBuf_;
    TBuf<TPosition::VECCALC> cacheChunkBuf_;
    TBuf<TPosition::VECCALC> hashBuf_;
    TBuf<TPosition::VECCALC> residentBuf_;
    TBuf<TPosition::VECCALC> protectedSlotBuf_;
    TBuf<TPosition::VECCALC> victimBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
};
} // namespace

extern "C" __global__ __aicore__ void fused_quant_lightning_indexer_manage(
    GM_ADDR indexWeights, GM_ADDR queryDequantScale, GM_ADDR query, GM_ADDR indexKeyDequantScale, GM_ADDR indexKeyCache,
    GM_ADDR indexBlockTable, GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKey, GM_ADDR offloadSeqLengthsKey,
    GM_ADDR numCacheTokens, GM_ADDR requestState, GM_ADDR reqPoolEntries, GM_ADDR cacheSlotsPool, GM_ADDR topkSrcIds,
    GM_ADDR topkDstSlots, GM_ADDR topkMissCounts, GM_ADDR missSrcIds, GM_ADDR missDstSlots, GM_ADDR missCounts,
    GM_ADDR cacheSlotsPoolOut, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)cacheSlotsPoolOut;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(FusedQuantLightningIndexerManageTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

    const bool coldPath = vllm_a5_li_manage_c8_fast::RequiresColdPath(
        actualSeqLengthsQuery, actualSeqLengthsKey, offloadSeqLengthsKey, numCacheTokens, requestState, reqPoolEntries,
        cacheSlotsPool, &tilingData);
    if (!coldPath) {
        const uint64_t scoreStride = tilingData.fastScoreWorkspaceStride;
        const uint64_t batchSize = tilingData.batchSize;
        // Keep constexpr workspace arithmetic expanded in device code, as in
        // the A5 reference: host-only constexpr helpers are used by tiling.
        const uint64_t pairOffset = scoreStride * batchSize;
        const uint64_t thresholdOffset = pairOffset + batchSize * 4U * TOPK * 2U * sizeof(int32_t);
        const uint64_t countOffset = thresholdOffset + batchSize * 4U * 16U * sizeof(uint16_t);
        GM_ADDR routePairRows = userWorkspace + pairOffset;
        GM_ADDR routeThresholds = userWorkspace + thresholdOffset;
        GM_ADDR routeMissCounts = userWorkspace + countOffset;

        vllm_a5_li_manage_c8_fast::QuantLiMtpPhase fastQli(&pipe, &tilingData);
        fastQli.Init(indexWeights, query, queryDequantScale, actualSeqLengthsQuery, indexKeyCache, indexKeyDequantScale,
                     actualSeqLengthsKey, reqPoolEntries, cacheSlotsPool, offloadSeqLengthsKey, numCacheTokens,
                     requestState, indexBlockTable, routePairRows, topkDstSlots, topkSrcIds, routeThresholds,
                     routeMissCounts, userWorkspace);
        fastQli.Process();

        if ASCEND_IS_AIV {
            AscendC::SyncAll();
            GlobalTensor<int32_t> states;
            states.SetGlobalBuffer((__gm__ int32_t *)requestState);
            bool needsNonOffload = false;
            bool needsFirstFill = false;
            bool needsSparse = false;
            for (uint32_t batch = 0U; batch < tilingData.batchSize; ++batch) {
                const int32_t state = states.GetValue(batch);
                needsNonOffload |= state != -2 && state != -1;
                needsFirstFill |= state == -2;
                needsSparse |= state == -1;
            }
            NonOffloadPublisher nonOffload;
            CacheManagementPublisher cacheManagement;
            vllm_a5_li_manage_c8_fast::OrderedMissUnion fastUnion;
            if ((GetBlockIdx() & 1U) == 0U) {
                if (needsNonOffload) {
                    pipe.Reset();
                    nonOffload.Init(&pipe, true, actualSeqLengthsQuery, actualSeqLengthsKey, requestState,
                                    reqPoolEntries, cacheSlotsPool, topkSrcIds, topkDstSlots, topkMissCounts,
                                    missCounts, &tilingData);
                    nonOffload.FillIdentityChunks(GetBlockIdx() / 2U, tilingData.usedCoreNum);
                }
                if (needsFirstFill || needsSparse)
                    pipe.Reset();
                // Only -2 needs the standard first-fill manager. Pure -1
                // batches proceed from QLI directly to A5 fast union.
                if (needsFirstFill) {
                    cacheManagement.Init(&pipe, userWorkspace, actualSeqLengthsQuery, actualSeqLengthsKey,
                                         offloadSeqLengthsKey, numCacheTokens, requestState, reqPoolEntries,
                                         cacheSlotsPool, topkSrcIds, topkDstSlots, topkMissCounts, missSrcIds,
                                         missDstSlots, missCounts, routePairRows, routeMissCounts, routeThresholds,
                                         &tilingData);
                    cacheManagement.Process(GetBlockIdx() / 2U, tilingData.usedCoreNum);
                    if (needsSparse)
                        pipe.Reset();
                }
                if (needsSparse) {
                    fastUnion.Init(
                        routePairRows, routeThresholds, routeMissCounts, userWorkspace, actualSeqLengthsQuery,
                        offloadSeqLengthsKey, actualSeqLengthsKey, requestState, reqPoolEntries, cacheSlotsPool,
                        numCacheTokens, missSrcIds, missDstSlots, missCounts, topkDstSlots, topkSrcIds, topkMissCounts,
                        userWorkspace + tilingData.managementWorkspaceOffset,
                        userWorkspace + tilingData.managementWorkspaceOffset + batchSize * 32U,
                        tilingData.sourceCapacity, MISS_CAPACITY, tilingData.fastScoreRowStride, tilingData.batchSize,
                        tilingData.poolSize, tilingData.totalQueryRows, tilingData.sourceCapacity, &pipe);
                    fastUnion.Process(GetBlockIdx() / 2U, tilingData.usedCoreNum);
                }
            }
            AscendC::SyncAll();
            if (GetBlockIdx() == 0U) {
                if (needsNonOffload) {
                    if (!needsFirstFill && !needsSparse) {
                        nonOffload.PublishFastHomogeneousCounts();
                    } else {
                        nonOffload.PublishCounts();
                    }
                }
                if (needsFirstFill)
                    cacheManagement.PublishCounts();
                if (needsSparse)
                    fastUnion.PublishCounts();
            }
        }
        return;
    }

    VllmA5MtpC8QliPhase qli(&pipe, &tilingData);
    qli.Init(indexWeights, queryDequantScale, query, indexKeyDequantScale, indexKeyCache, indexBlockTable,
             actualSeqLengthsQuery, actualSeqLengthsKey, requestState, topkSrcIds, userWorkspace);
    qli.Process();

    if ASCEND_IS_AIV {
        AscendC::SyncAll();
        NonOffloadPublisher nonOffload;
        CacheManagementPublisher cacheManagement;
        if ((GetBlockIdx() & 1U) == 0U) {
            pipe.Reset();
            nonOffload.Init(&pipe, false, actualSeqLengthsQuery, actualSeqLengthsKey, requestState, reqPoolEntries,
                            cacheSlotsPool, topkSrcIds, topkDstSlots, topkMissCounts, missCounts, &tilingData);
            nonOffload.Process(GetBlockIdx() / 2U, tilingData.usedCoreNum);
            nonOffload.FillIdentityChunks(GetBlockIdx() / 2U, tilingData.usedCoreNum);
            pipe.Reset();
            cacheManagement.Init(&pipe, userWorkspace, actualSeqLengthsQuery, actualSeqLengthsKey, offloadSeqLengthsKey,
                                 numCacheTokens, requestState, reqPoolEntries, cacheSlotsPool, topkSrcIds, topkDstSlots,
                                 topkMissCounts, missSrcIds, missDstSlots, missCounts, nullptr, nullptr, nullptr,
                                 &tilingData);
            cacheManagement.Process(GetBlockIdx() / 2U, tilingData.usedCoreNum);
        }
        AscendC::SyncAll();
        if (GetBlockIdx() == 0U) {
            nonOffload.PublishCounts();
            cacheManagement.PublishCounts();
        }
    }
}
