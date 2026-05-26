/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_cut_origin_t.h
 * \brief
 */
#ifndef MOE_V3_CUT_ORIGIN_T_H
#define MOE_V3_CUT_ORIGIN_T_H

#include "moe_v3_common.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3CutOriginT {
public:
    __aicore__ inline MoeV3CutOriginT(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR expertTokens,
        GM_ADDR expandedScale, GM_ADDR workspace, const MoeInitRoutingV3TilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void FilterAndCountChunked();
    __aicore__ inline void WriteExpertCountToWorkspace();
    __aicore__ inline void ComputeGlobalOffset();
    __aicore__ inline void WriteExpertTokens();
    __aicore__ inline void GatherAndWriteChunked();

private:
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;
    static constexpr int64_t VECTOR_FILTER_THRESHOLD = 64;

    TPipe* pipe_;
    TBuf<TPosition::VECCALC> buf_;

    // GM tensors
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int64_t> expertTokensGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<int32_t> pairsWorkspaceGm_;
    GlobalTensor<int32_t> expertCountWorkspaceGm_;

    // Tiling params
    int64_t blockIdx_;
    int64_t n_;
    int64_t k_;
    int64_t cols_;
    int64_t totalLength_;
    int64_t expertStart_;
    int64_t expertEnd_;
    int64_t actualExpertNum_;
    int64_t rowIdxType_;
    int64_t isInputScale_;
    int64_t filterNeedCoreNum_;
    int64_t coreNum_;
    int64_t expertTokensNumFlag_;
    int64_t filterChunkSize_;
    int64_t activeNum_;

    // Per-core distribution
    int64_t coreTokenStart_;
    int64_t coreTokenEnd_;
    int64_t coreTokenNum_;
    int64_t coreFlatStart_;
    int64_t coreEntries_;

    // Cols loop params (from gatherOutComputeParamsOp)
    int64_t colsLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t perLoopColsAligned_;

    // Derived constants
    int64_t expertCountStride_;
    int64_t colsAligned_;
    int64_t scaleSlotSize_;
    int64_t pairsPerCore_;
    int64_t maxChunks_;
    int64_t chunkAligned_;
    bool useVectorFilter_;
    int64_t maskBytes_;

    // Persistent UB offsets (survive all phases)
    int64_t expertCountLocalOffset_;
    int64_t persistentSize_;

    // Phase A result
    int64_t totalPairCount_;

    // Short path: gathered data stays in UB (maxChunks_==1 && totalPairCount_ < 2048)
    bool shortPath_;
    int64_t gatheredIdxLocalOffset_;
    int64_t gatheredExpertLocalOffset_;

    // Phase B UB offsets (stored for WriteExpertTokens access)
    int64_t allCoreExpertCountLocalOffset_;
    int64_t totalCountLocalOffset_;
    int64_t prefixSumLocalOffset_;
    int64_t expertTokensLocalOffset_;

    int64_t totalBufSize_;
};

// ========================== Init ==========================
template <typename T>
__aicore__ inline void MoeV3CutOriginT<T>::Init(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR expertTokens,
    GM_ADDR expandedScale, GM_ADDR workspace, const MoeInitRoutingV3TilingData* tiling, TPipe* pipe)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();

    // Parse tiling params
    n_ = tiling->n;
    k_ = tiling->k;
    cols_ = tiling->cols;
    totalLength_ = n_ * k_;
    expertStart_ = tiling->expertStart;
    expertEnd_ = tiling->expertEnd;
    actualExpertNum_ = tiling->actualExpertNum;
    rowIdxType_ = tiling->rowIdxType;
    isInputScale_ = tiling->isInputScale;
    filterNeedCoreNum_ = tiling->gatherOutComputeParamsOp.needCoreNum;
    coreNum_ = tiling->coreNum;
    expertTokensNumFlag_ = tiling->expertTokensNumFlag;
    activeNum_ = tiling->activeNum;

    filterChunkSize_ = FILTER_CHUNK_SIZE;

    // Per-core token distribution
    int64_t filterPerCoreTokens = tiling->gatherOutComputeParamsOp.perCoreIndicesElements;
    coreTokenStart_ = blockIdx_ * filterPerCoreTokens;
    coreTokenEnd_ = Min(coreTokenStart_ + filterPerCoreTokens, n_);
    if (blockIdx_ == filterNeedCoreNum_ - 1) {
        coreTokenEnd_ = n_;
    }
    coreTokenNum_ = coreTokenEnd_ - coreTokenStart_;
    coreFlatStart_ = coreTokenStart_ * k_;
    coreEntries_ = coreTokenNum_ * k_;

    // Derived constants
    expertCountStride_ = OpsBaseCeilAlign(actualExpertNum_, static_cast<int64_t>(8));
    colsAligned_ =
        OpsBaseCeilAlign(cols_ * static_cast<int64_t>(sizeof(T)), BLOCK_BYTES) / static_cast<int64_t>(sizeof(T));
    scaleSlotSize_ = BLOCK_BYTES / static_cast<int64_t>(sizeof(float)); // 8
    chunkAligned_ = OpsBaseCeilAlign(filterChunkSize_, static_cast<int64_t>(8));
    maxChunks_ = Ceil(coreEntries_, filterChunkSize_);

    // Cols loop params: when cols is too large for UB, split into multiple loops
    colsLoops_ = tiling->gatherOutComputeParamsOp.colsLoops;
    perLoopCols_ = tiling->gatherOutComputeParamsOp.perLoopCols;
    lastLoopCols_ = tiling->gatherOutComputeParamsOp.lastLoopCols;
    perLoopColsAligned_ =
        OpsBaseCeilAlign(perLoopCols_ * static_cast<int64_t>(sizeof(T)), BLOCK_BYTES) / static_cast<int64_t>(sizeof(T));

    // Workspace stride: consistent across all cores (use max possible per-core entries)
    int64_t lastCoreTokens = n_ - (filterNeedCoreNum_ - 1) * filterPerCoreTokens;
    int64_t maxCoreTokens = Max(filterPerCoreTokens, lastCoreTokens);
    int64_t maxCoreEntries = maxCoreTokens * k_;
    pairsPerCore_ = OpsBaseCeilAlign(maxCoreEntries, static_cast<int64_t>(8)) * 2;

    // Vector filter decision
    maskBytes_ = OpsBaseCeilAlign(Ceil(chunkAligned_, static_cast<int64_t>(8)), static_cast<int64_t>(sizeof(int8_t)));

    // Setup GlobalTensors
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expertIdx);
    expandedXGm_.SetGlobalBuffer((__gm__ T*)expandedX);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx);
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        expertTokensGm_.SetGlobalBuffer((__gm__ int64_t*)expertTokens);
    }
    if (isInputScale_) {
        scaleGm_.SetGlobalBuffer((__gm__ float*)scale);
        expandedScaleGm_.SetGlobalBuffer((__gm__ float*)expandedScale);
    }

    // InitGlobalMemory: GATHER mode pre-fill expandedRowIdx with -1
    if (rowIdxType_ == GATHER) {
        if (blockIdx_ < filterNeedCoreNum_) {
            GlobalTensor<int32_t> expandedRowIdxGmTmp = expandedRowIdxGm_[filterPerCoreTokens * k_ * blockIdx_];
            if (blockIdx_ == filterNeedCoreNum_ - 1) {
                InitGlobalMemory(expandedRowIdxGmTmp, lastCoreTokens * k_, -1);
            } else {
                InitGlobalMemory(expandedRowIdxGmTmp, maxCoreEntries, -1);
            }
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
        SyncAll();
    }

    // Workspace layout: pairs + expertCount (2 regions only)
    int64_t wsOffset = 0;
    pairsWorkspaceGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + wsOffset);
    wsOffset += filterNeedCoreNum_ * pairsPerCore_;
    expertCountWorkspaceGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + wsOffset);

    // ===== UB layout: persistent region + max(phaseA, phaseB, phaseC) =====

    // Short path: maxChunks_==1 means all pairs fit in one chunk
    shortPath_ = (maxChunks_ == 1);

    // Persistent region
    int64_t offset = 0;
    expertCountLocalOffset_ = offset;
    offset += OpsBaseCeilAlign(expertCountStride_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    if (shortPath_) {
        // Gathered data survives from Phase A to Phase C
        gatheredIdxLocalOffset_ = offset;
        offset += OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
        gatheredExpertLocalOffset_ = offset;
        offset += OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    }
    persistentSize_ = offset;

    // Phase A shared size (from persistentSize_)
    int64_t phaseASize = 0;
    phaseASize +=
        OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES); // expertIdxLocal
    phaseASize +=
        OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(float)), BLOCK_BYTES); // expertIdxFp32Local
    phaseASize += maskBytes_ * 3; // compareMask0/1 + gatherMask
    phaseASize +=
        OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES); // flatIdxBufferLocal
    if (!shortPath_) {
        phaseASize += OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES) *
                      2; // gatheredExpert + gatheredIdx
    }

    // Phase B shared offsets (absolute from buf_ start, stored for WriteExpertTokens)
    int64_t phaseBSize = 0;
    allCoreExpertCountLocalOffset_ = persistentSize_;
    phaseBSize +=
        OpsBaseCeilAlign(filterNeedCoreNum_ * expertCountStride_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    totalCountLocalOffset_ = persistentSize_ + phaseBSize;
    phaseBSize += OpsBaseCeilAlign(expertCountStride_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    prefixSumLocalOffset_ = persistentSize_ + phaseBSize;
    phaseBSize += OpsBaseCeilAlign(expertCountStride_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    expertTokensLocalOffset_ = persistentSize_ + phaseBSize;
    phaseBSize += OpsBaseCeilAlign(actualExpertNum_ * static_cast<int64_t>(sizeof(int64_t)), BLOCK_BYTES);

    // Phase C shared size (from persistentSize_)
    // use double buffer: 2x xBuf + 2x scaleBuf
    int64_t phaseCSize = 0;
    phaseCSize += OpsBaseCeilAlign(chunkAligned_ * 2 * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    phaseCSize += BLOCK_BYTES; // idxBuf: 32B, separate from xBufLocal
    phaseCSize += OpsBaseCeilAlign(perLoopColsAligned_ * static_cast<int64_t>(sizeof(T)), BLOCK_BYTES) * 2;
    if (isInputScale_) {
        phaseCSize += OpsBaseCeilAlign(scaleSlotSize_ * static_cast<int64_t>(sizeof(float)), BLOCK_BYTES) * 2;
    }

    totalBufSize_ = persistentSize_ + Max(phaseASize, Max(phaseBSize, phaseCSize));
    pipe_->InitBuffer(buf_, totalBufSize_);
}

// ========================== Process ==========================
template <typename T>
__aicore__ inline void MoeV3CutOriginT<T>::Process()
{
    if (blockIdx_ >= filterNeedCoreNum_) {
        SyncAll();
        return;
    }

    // ======= Phase A: Chunked filter + count (all cores parallel) =======
    FilterAndCountChunked();
    WriteExpertCountToWorkspace();

    SyncAll();

    // // // // ======= Phase B: Global reduce + offset (all cores parallel) =======
    ComputeGlobalOffset();
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        WriteExpertTokens();
    }

    // // // // ======= Phase C: Chunked gather + on-demand x DMA (all cores parallel) =======
    GatherAndWriteChunked();
}

// ========================== FilterAndCountChunked ==========================
template <typename T>
__aicore__ inline void MoeV3CutOriginT<T>::FilterAndCountChunked()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    // Vectorized initialization
    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    // PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    int64_t pairCursor = 0;

    for (int64_t chunkIdx = 0; chunkIdx < maxChunks_; chunkIdx++) {
        int64_t chunkStart = chunkIdx * filterChunkSize_;
        int64_t chunkLength = Min(filterChunkSize_, coreEntries_ - chunkStart);
        int64_t gmFlatOffset = coreFlatStart_ + chunkStart;

        // Phase A shared layout (computed locally per-chunk, from persistentSize_)
        int64_t off = persistentSize_;
        int64_t expertIdxOff = off;
        off += OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

        // Load expertIdx chunk from GM

        LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxOff / sizeof(int32_t)];
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(chunkLength * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(expertIdxLocal, expertIdxGm_[gmFlatOffset], copyParams, padParams);

        int64_t filteredInChunk = 0;

        // === Vector filter path ===
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

        int64_t alignedLen = Ceil(chunkLength, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;

        // Sub-layout for vector buffers (temporary only)
        int64_t expertIdxFp32Off = off;
        off += OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(float)), BLOCK_BYTES);
        int64_t compareMask0Off = off;
        off += maskBytes_;
        int64_t compareMask1Off = off;
        off += maskBytes_;
        int64_t gatherMaskOff = off;
        off += maskBytes_;
        int64_t flatIdxBufferOff = off;
        off += OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

        // Gathered data: fixed offset in persistent region (shortPath) or dynamic (normal)
        int64_t gatheredExpertOff, gatheredIdxOff;
        if (shortPath_) {
            gatheredExpertOff = gatheredExpertLocalOffset_;
            gatheredIdxOff = gatheredIdxLocalOffset_;
        } else {
            gatheredExpertOff = off;
            off += OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
            gatheredIdxOff = off;
            off += OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
        }

        // Cast int32 -> fp32 for CompareScalar
        LocalTensor<float> expertIdxFp32Local = buf_.Get<float>()[expertIdxFp32Off / sizeof(float)];
        Cast(expertIdxFp32Local, expertIdxLocal, RoundMode::CAST_ROUND, alignedLen);
        PipeBarrier<PIPE_V>();

        // Range filter: [expertStart_, expertEnd_)
        LocalTensor<uint8_t> compareMask0 = buf_.Get<uint8_t>()[compareMask0Off];
        LocalTensor<uint8_t> compareMask1 = buf_.Get<uint8_t>()[compareMask1Off];
        LocalTensor<uint8_t> gatherMaskLocal = buf_.Get<uint8_t>()[gatherMaskOff];

        CompareScalar(compareMask0, expertIdxFp32Local, static_cast<float>(expertStart_), CMPMODE::GE, alignedLen);
        PipeBarrier<PIPE_V>();
        CompareScalar(compareMask1, expertIdxFp32Local, static_cast<float>(expertEnd_), CMPMODE::LT, alignedLen);
        PipeBarrier<PIPE_V>();
        And(gatherMaskLocal.ReinterpretCast<uint16_t>(), compareMask0.ReinterpretCast<uint16_t>(),
            compareMask1.ReinterpretCast<uint16_t>(), Ceil(alignedLen, MASK_STRIDE) * MASK_STRIDE / DST_REP_STRIDE / 2);
        PipeBarrier<PIPE_V>();

        // GatherMask: compress expert IDs
        LocalTensor<int32_t> gatheredExpertLocal = buf_.Get<int32_t>()[gatheredExpertOff / sizeof(int32_t)];
        uint64_t rsvdCnt = 0;
        GatherMaskParams gatherMaskParams;
        gatherMaskParams.repeatTimes = 1;
        gatherMaskParams.src0BlockStride = 1;
        gatherMaskParams.src0RepeatStride = DST_REP_STRIDE;
        gatherMaskParams.src1RepeatStride = DST_REP_STRIDE;
        GatherMask(
            gatheredExpertLocal, expertIdxLocal, gatherMaskLocal.ReinterpretCast<uint32_t>(), true,
            static_cast<uint32_t>(chunkLength), gatherMaskParams, rsvdCnt);
        PipeBarrier<PIPE_V>();
        filteredInChunk = static_cast<int64_t>(rsvdCnt);

        if (filteredInChunk > 0) {
            // GatherMask: compress flat indices
            LocalTensor<int32_t> flatIdxBufferLocal = buf_.Get<int32_t>()[flatIdxBufferOff / sizeof(int32_t)];
            ArithProgression<int32_t>(
                flatIdxBufferLocal, static_cast<int32_t>(gmFlatOffset), 1, static_cast<int32_t>(chunkLength));
            PipeBarrier<PIPE_V>();

            LocalTensor<int32_t> gatheredIdxLocal = buf_.Get<int32_t>()[gatheredIdxOff / sizeof(int32_t)];
            uint64_t idxRsvdCnt = 0;
            GatherMask(
                gatheredIdxLocal, flatIdxBufferLocal, gatherMaskLocal.ReinterpretCast<uint32_t>(), true,
                static_cast<uint32_t>(chunkLength), gatherMaskParams, idxRsvdCnt);
            PipeBarrier<PIPE_V>();

            // Vector: subtract expertStart_ in-place -> gatheredExpertLocal becomes expertOffset
            Adds(
                gatheredExpertLocal, gatheredExpertLocal, static_cast<int32_t>(-expertStart_),
                static_cast<int32_t>(filteredInChunk));

            // V -> S sync before scalar count accumulation
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

            // Scalar: accumulate expert counts using gatheredExpertLocal (now expertOffset)
            for (int64_t j = 0; j < filteredInChunk; j++) {
                int32_t expertOffset = gatheredExpertLocal.GetValue(j);
                int32_t curCount = expertCountLocal.GetValue(expertOffset);
                expertCountLocal.SetValue(expertOffset, curCount + 1);
            }

            if (!shortPath_ || filteredInChunk >= 2048) {
                // V -> MTE3 sync: gathered buffers ready for DMA
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

                // Write gatheredIdxLocal and gatheredExpertLocal to workspace (separated layout)
                DataCopyExtParams copyParams{
                    static_cast<uint16_t>(1), static_cast<uint32_t>(filteredInChunk * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairCursor], gatheredIdxLocal, copyParams);
                DataCopyPad(
                    pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairsPerCore_ / 2 + pairCursor], gatheredExpertLocal,
                    copyParams);
            }
        }

        pairCursor += filteredInChunk;
    }
    if (!shortPath_ || pairCursor >= 2048) {
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }

    totalPairCount_ = pairCursor;
}

// ========================== WriteExpertCountToWorkspace ==========================
template <typename T>
__aicore__ inline void MoeV3CutOriginT<T>::WriteExpertCountToWorkspace()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(expertCountStride_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(expertCountWorkspaceGm_[blockIdx_ * expertCountStride_], expertCountLocal, copyParams);
}

// ========================== ComputeGlobalOffset ==========================
template <typename T>
__aicore__ inline void MoeV3CutOriginT<T>::ComputeGlobalOffset()
{
    // Invalidate cache to see workspace writes from all cores
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expertCountWorkspaceGm_);

    LocalTensor<int32_t> allCoreExpertCountLocal =
        buf_.Get<int32_t>()[allCoreExpertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> totalCountLocal = buf_.Get<int32_t>()[totalCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> prefixSumLocal = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    // Load all cores' expert counts from workspace
    int64_t totalWsElements = filterNeedCoreNum_ * expertCountStride_;
    DataCopyExtParams loadParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(totalWsElements * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(allCoreExpertCountLocal, expertCountWorkspaceGm_[0], loadParams, padParams);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    // Vectorized cross-core sum -> totalCountLocal
    Duplicate(totalCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    for (int64_t c = 0; c < filterNeedCoreNum_; c++) {
        Add(totalCountLocal, totalCountLocal, allCoreExpertCountLocal[c * expertCountStride_], expertCountStride_);
        PipeBarrier<PIPE_V>();
    }

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // Vectorized prefix sum: sum of counts from cores [0, blockIdx_)
    Duplicate(prefixSumLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    for (int64_t c = 0; c < blockIdx_; c++) {
        Add(prefixSumLocal, prefixSumLocal, allCoreExpertCountLocal[c * expertCountStride_], expertCountStride_);
        PipeBarrier<PIPE_V>();
    }

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // Scalar: compute per-expert global offset -> store in expertCountLocal
    int64_t cumulativeSum = 0;
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        int32_t totalForExpert = totalCountLocal.GetValue(e);
        int32_t prefixForExpert = prefixSumLocal.GetValue(e);

        // expertCountLocal[e] = global write start position for this core's expert e entries
        expertCountLocal.SetValue(e, static_cast<int32_t>(cumulativeSum) + prefixForExpert);
        cumulativeSum += totalForExpert;
    }
}

// ========================== WriteExpertTokens ==========================
template <typename T>
__aicore__ inline void MoeV3CutOriginT<T>::WriteExpertTokens()
{
    if (blockIdx_ == 0 && expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        LocalTensor<int32_t> totalCountLocal = buf_.Get<int32_t>()[totalCountLocalOffset_ / sizeof(int32_t)];
        LocalTensor<int64_t> expertTokensLocal = buf_.Get<int64_t>()[expertTokensLocalOffset_ / sizeof(int64_t)];

        Cast(expertTokensLocal, totalCountLocal, RoundMode::CAST_NONE, static_cast<int32_t>(actualExpertNum_));
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(actualExpertNum_ * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);
    }
}

// ========================== GatherAndWriteChunked ==========================
template <typename T>
__aicore__ inline void MoeV3CutOriginT<T>::GatherAndWriteChunked()
{
    if (totalPairCount_ <= 0) {
        return;
    }

    bool useWorkspace = !(shortPath_ && totalPairCount_ < 2048);
    if (useWorkspace) {
        // Invalidate workspace cache to see pairs written in Phase A
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(pairsWorkspaceGm_);
    }

    // Phase C shared layout (from persistentSize_) — double buffer for xBuf/scaleBuf
    int64_t off = persistentSize_;
    int64_t pairsOff = off;
    int64_t pairsBufCapacity = chunkAligned_;
    if (useWorkspace) {
        off += OpsBaseCeilAlign(pairsBufCapacity * 2 * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    }
    int64_t idxBufOff = off;
    off += BLOCK_BYTES;
    int64_t xBufSlotSize = OpsBaseCeilAlign(perLoopColsAligned_ * static_cast<int64_t>(sizeof(T)), BLOCK_BYTES);
    int64_t xBufOff0 = off;
    off += xBufSlotSize;
    int64_t xBufOff1 = off;
    off += xBufSlotSize;
    int64_t scaleBufSlotSize = OpsBaseCeilAlign(scaleSlotSize_ * static_cast<int64_t>(sizeof(float)), BLOCK_BYTES);
    int64_t scaleBufOff0 = off;
    int64_t scaleBufOff1 = off + scaleBufSlotSize;

    // Persistent buffer
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    // Reusable DMA params
    DataCopyPadExtParams<T> xPadParams{false, 0, 0, 0};
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    DataCopyExtParams scaleCopyInParams{0, 0, 0, 0, 0};
    DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0};
    DataCopyExtParams scaleCopyOutParams{0, 0, 0, 0, 0};
    if (isInputScale_) {
        scaleCopyInParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        scaleCopyOutParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    }

    // Persistent event IDs for double buffer pipeline
    event_t evMte2Done0 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_MTE3));
    event_t evMte2Done1 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_MTE3));
    event_t evMte3Done0 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
    event_t evMte3Done1 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));

    // Flat iteration over all pairs, loaded in UB-sized batches
    int64_t pairCursor = 0;
    while (pairCursor < totalPairCount_) {
        int64_t batchSize = Min(pairsBufCapacity, totalPairCount_ - pairCursor);
        batchSize = Min(batchSize, static_cast<int64_t>(2048));

        // Load pairs batch: from workspace (normal) or UB persistent region (shortPath)
        LocalTensor<int32_t> gatheredIdxLocal;
        LocalTensor<int32_t> gatheredExpertLocal;
        if (useWorkspace) {
            gatheredIdxLocal = buf_.Get<int32_t>()[pairsOff / sizeof(int32_t)];
            gatheredExpertLocal = buf_.Get<int32_t>()[pairsOff / sizeof(int32_t) + chunkAligned_];
            DataCopyPadExtParams<int32_t> intPadParams{false, 0, 0, 0};
            DataCopyExtParams pairLoadParams{
                static_cast<uint16_t>(1), static_cast<uint32_t>(batchSize * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(
                gatheredIdxLocal, pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairCursor], pairLoadParams,
                intPadParams);
            DataCopyPad(
                gatheredExpertLocal, pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairsPerCore_ / 2 + pairCursor],
                pairLoadParams, intPadParams);
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        } else {
            gatheredIdxLocal = buf_.Get<int32_t>()[gatheredIdxLocalOffset_ / sizeof(int32_t)];
            gatheredExpertLocal = buf_.Get<int32_t>()[gatheredExpertLocalOffset_ / sizeof(int32_t)];
        }

        LocalTensor<int32_t> idxBuf = buf_.Get<int32_t>()[idxBufOff / sizeof(int32_t)];

        // Pre-compute newPos/srcRow for all pairs in this batch
        // (scalar work, no DMA dependency)
        int32_t newPosArr[2048];
        int32_t srcRowArr[2048];
        int32_t origFlatIdxArr[2048];
        for (int64_t i = 0; i < batchSize; i++) {
            int32_t origFlatIdx = gatheredIdxLocal.GetValue(i);
            int32_t expertOffset = gatheredExpertLocal.GetValue(i);
            int32_t newPos = expertCountLocal.GetValue(expertOffset);
            expertCountLocal.SetValue(expertOffset, newPos + 1);
            newPosArr[i] = newPos;
            srcRowArr[i] = origFlatIdx / static_cast<int32_t>(k_);
            origFlatIdxArr[i] = origFlatIdx;
        }

        // Ping-pong buffers
        event_t evMte2[2] = {evMte2Done0, evMte2Done1};
        event_t evMte3[2] = {evMte3Done0, evMte3Done1};
        LocalTensor<T> xBuf0 = buf_.Get<T>()[xBufOff0 / sizeof(T)];
        LocalTensor<T> xBuf1 = buf_.Get<T>()[xBufOff1 / sizeof(T)];
        LocalTensor<float> scBuf0, scBuf1;
        if (isInputScale_) {
            scBuf0 = buf_.Get<float>()[scaleBufOff0 / sizeof(float)];
            scBuf1 = buf_.Get<float>()[scaleBufOff1 / sizeof(float)];
        }

        if (colsLoops_ == 1) {
            // === Double buffer path: MTE2/MTE3 overlap across pairs ===
            DataCopyExtParams xCp{static_cast<uint16_t>(1), static_cast<uint32_t>(lastLoopCols_ * sizeof(T)), 0, 0, 0};

            // Prologue: prefetch pair 0 -> buf[0]
            if (batchSize >= 1) {
                DataCopyPad(xBuf0, xGm_[static_cast<int64_t>(srcRowArr[0]) * cols_], xCp, xPadParams);
                if (isInputScale_) {
                    DataCopyPad(scBuf0, scaleGm_[srcRowArr[0]], scaleCopyInParams, scalePadParams);
                }
                SetFlag<HardEvent::MTE2_MTE3>(evMte2[0]);
            }
            // Prologue: prefetch pair 1 -> buf[1]
            if (batchSize >= 2) {
                DataCopyPad(xBuf1, xGm_[static_cast<int64_t>(srcRowArr[1]) * cols_], xCp, xPadParams);
                if (isInputScale_) {
                    DataCopyPad(scBuf1, scaleGm_[srcRowArr[1]], scaleCopyInParams, scalePadParams);
                }
                SetFlag<HardEvent::MTE2_MTE3>(evMte2[1]);
            }

            // Main loop: store pair i from buf[bc], prefetch pair i+2 into buf[bc]
            for (int64_t i = 0; i < batchSize; i++) {
                int32_t bc = static_cast<int32_t>(i & 1);
                int32_t newPos = newPosArr[i];
                LocalTensor<T>& curXBuf = (bc == 0) ? xBuf0 : xBuf1;

                int32_t origFlatIdx = origFlatIdxArr[i];
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                Duplicate(idxBuf, static_cast<int32_t>(0), 8);
                PipeBarrier<PIPE_V>();
                if (rowIdxType_ == GATHER) {
                    Adds(idxBuf, idxBuf, static_cast<int32_t>(newPos), 8);
                } else {
                    Adds(idxBuf, idxBuf, static_cast<int32_t>(origFlatIdx), 8);
                }
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

                WaitFlag<HardEvent::MTE2_MTE3>(evMte2[bc]);
                if (newPos < activeNum_) {
                    DataCopyPad(expandedXGm_[static_cast<int64_t>(newPos) * cols_], curXBuf, xCp);
                    if (isInputScale_) {
                        LocalTensor<float>& curScBuf = (bc == 0) ? scBuf0 : scBuf1;
                        DataCopyPad(expandedScaleGm_[newPos], curScBuf, scaleCopyOutParams);
                    }
                }
                if (rowIdxType_ == GATHER) {
                    DataCopyPad(expandedRowIdxGm_[origFlatIdx], idxBuf, idxCopyParams);
                } else {
                    DataCopyPad(expandedRowIdxGm_[newPos], idxBuf, idxCopyParams);
                }
                SetFlag<HardEvent::MTE3_MTE2>(evMte3[bc]);

                if (i + 2 < batchSize) {
                    WaitFlag<HardEvent::MTE3_MTE2>(evMte3[bc]);
                    DataCopyPad(curXBuf, xGm_[static_cast<int64_t>(srcRowArr[i + 2]) * cols_], xCp, xPadParams);
                    if (isInputScale_) {
                        LocalTensor<float>& curScBuf = (bc == 0) ? scBuf0 : scBuf1;
                        DataCopyPad(curScBuf, scaleGm_[srcRowArr[i + 2]], scaleCopyInParams, scalePadParams);
                    }
                    SetFlag<HardEvent::MTE2_MTE3>(evMte2[bc]);
                }
            }

            // Epilogue: drain remaining MTE3 flags
            if (batchSize >= 1) {
                WaitFlag<HardEvent::MTE3_MTE2>(evMte3[(batchSize - 1) & 1]);
            }
            if (batchSize >= 2) {
                WaitFlag<HardEvent::MTE3_MTE2>(evMte3[1 - ((batchSize - 1) & 1)]);
            }
        } else {
            // === Serial fallback for colsLoops_ > 1 ===
            for (int64_t i = 0; i < batchSize; i++) {
                int32_t newPos = newPosArr[i];
                int32_t srcRow = srcRowArr[i];
                int32_t origFlatIdx = origFlatIdxArr[i];

                if (rowIdxType_ == GATHER) {
                    idxBuf.SetValue(0, newPos);
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    DataCopyPad(expandedRowIdxGm_[origFlatIdx], idxBuf, idxCopyParams);
                } else {
                    idxBuf.SetValue(0, origFlatIdx);
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    DataCopyPad(expandedRowIdxGm_[newPos], idxBuf, idxCopyParams);
                }
                if (newPos >= activeNum_) {
                    continue;
                }
                if (isInputScale_) {
                    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
                    DataCopyPad(scBuf0, scaleGm_[srcRow], scaleCopyInParams, scalePadParams);
                    SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
                    DataCopyPad(expandedScaleGm_[newPos], scBuf0, scaleCopyOutParams);
                }

                int64_t curLoopCols = perLoopCols_;
                for (int64_t cl = 0; cl < colsLoops_; cl++) {
                    if (cl == colsLoops_ - 1) {
                        curLoopCols = lastLoopCols_;
                    }
                    int64_t colsOff = cl * perLoopCols_;
                    DataCopyExtParams xIn{
                        static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};
                    DataCopyExtParams xOut{
                        static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};

                    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
                    DataCopyPad(xBuf0, xGm_[static_cast<int64_t>(srcRow) * cols_ + colsOff], xIn, xPadParams);
                    SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
                    DataCopyPad(expandedXGm_[static_cast<int64_t>(newPos) * cols_ + colsOff], xBuf0, xOut);
                }
            }
        }

        pairCursor += batchSize;
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_CUT_ORIGIN_T_H