/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_cut_origin_t.h
 * \brief arch35 counting-sort kernel (DropLess + unquant only).
 *        Pure sort: produces sortedExpertIdx and expandedRowIdx in standard workspace layout,
 *        then downstream stages (ExpertTokensCount, RowIdxGather, MoeGatherOut) handle the rest.
 */
#ifndef MOE_V3_CUT_ORIGIN_T_H
#define MOE_V3_CUT_ORIGIN_T_H

#include "moe_v3_common.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t FILTER_CHUNK_SIZE = 4096;

template <typename T>
__aicore__ inline T OpsBaseCeilDiv(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T OpsBaseCeilAlign(T a, T b)
{
    return (a + b - 1) / b * b;
}

class MoeV3CutOriginT {
public:
    __aicore__ inline MoeV3CutOriginT(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void FilterAndCountChunked();
    __aicore__ inline void WriteExpertCountToWorkspace();
    __aicore__ inline void ComputeGlobalOffset();
    __aicore__ inline void WriteSortedOutput();
    __aicore__ inline void FillSortedExpertIdxSentinel();

private:
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;

    TPipe *pipe_;
    TBuf<TPosition::VECCALC> buf_;

    // GM tensors
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> pairsWorkspaceGm_;
    GlobalTensor<int32_t> expertCountWorkspaceGm_;
    // Standard workspace outputs (same layout as MoeSortOneCore)
    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<int32_t> rowIdxWorkspaceGm_;
    // Raw workspace base (for initializing expertCountTemp for downstream)
    __gm__ int32_t *workspaceBase_;

    // Tiling params
    int64_t blockIdx_;
    int64_t n_;
    int64_t k_;
    int64_t totalLength_;
    int64_t expertStart_;
    int64_t expertEnd_;
    int64_t actualExpertNum_;
    int64_t rowIdxType_;
    int64_t ep_;
    int64_t filterNeedCoreNum_;
    int64_t coreNum_;
    int64_t filterChunkSize_;

    // Per-core distribution
    int64_t coreTokenStart_;
    int64_t coreTokenEnd_;
    int64_t coreTokenNum_;
    int64_t coreFlatStart_;
    int64_t coreEntries_;

    // Derived constants
    int64_t expertCountStride_;
    int64_t pairsPerCore_;
    int64_t maxChunks_;
    int64_t chunkAligned_;
    int64_t maskBytes_;

    // Standard workspace layout offsets (in int32_t elements)
    int64_t sortedExpertIdxWsOffset_;
    int64_t rowIdxWsOffset_;
    int64_t pairsWsOffset_;
    int64_t expertCountWsOffset_;

    // Persistent UB offsets (survive all phases)
    int64_t expertCountLocalOffset_;
    int64_t persistentSize_;

    // Phase A result
    int64_t totalPairCount_;

    // Short path: gathered data stays in UB (maxChunks_==1 && totalPairCount_ < 2048)
    bool shortPath_;
    int64_t gatheredIdxLocalOffset_;
    int64_t gatheredExpertLocalOffset_;

    // Phase B UB offsets
    int64_t oneCoreExpertCountLocalOffset_;
    int64_t prefixSumLocalOffset_;
    int64_t totalCountLocalOffset_;

    // Global total pair count (computed in ComputeGlobalOffset, used by FillSortedExpertIdxSentinel)
    int64_t globalTotal_;

    int64_t totalBufSize_;
};

// ========================== Init ==========================
__aicore__ inline void
MoeV3CutOriginT::Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR workspace,
                      const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();

    // Parse tiling params
    n_ = tiling->n;
    k_ = tiling->k;
    totalLength_ = n_ * k_;
    expertStart_ = tiling->expertStart;
    expertEnd_ = tiling->expertEnd;
    actualExpertNum_ = tiling->actualExpertNum;
    rowIdxType_ = tiling->rowIdxType;
    filterNeedCoreNum_ = tiling->gatherOutComputeParamsOp.needCoreNum;
    coreNum_ = tiling->coreNum;
    ep_ = (expertStart_ == 0 && expertEnd_ == tiling->expertNum) ? 0 : 1;

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
    chunkAligned_ = OpsBaseCeilAlign(filterChunkSize_, static_cast<int64_t>(8));
    maxChunks_ = Ceil(coreEntries_, filterChunkSize_);

    // Workspace stride: consistent across all cores (use max possible per-core entries)
    int64_t lastCoreTokens = n_ - (filterNeedCoreNum_ - 1) * filterPerCoreTokens;
    int64_t maxCoreTokens = Max(filterPerCoreTokens, lastCoreTokens);
    int64_t maxCoreEntries = maxCoreTokens * k_;
    pairsPerCore_ = OpsBaseCeilAlign(maxCoreEntries, static_cast<int64_t>(8)) * 2;

    // Vector filter decision
    maskBytes_ = OpsBaseCeilAlign(Ceil(chunkAligned_, static_cast<int64_t>(8)), static_cast<int64_t>(sizeof(int8_t)));

    // Setup GlobalTensors
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx);
    workspaceBase_ = (__gm__ int32_t *)workspace;

    // Standard workspace layout (same as MoeSortOneCore):
    //   [0 .. Align(n*k)-1]                     sortedExpertIdx
    //   [Align(n*k) .. Align(n*k)*2-1]          rowIdxWorkspace (GATHER mode)
    //   [Align(n*k)*2 ..]                       pairs (counting-sort internal)
    //   [after pairs]                           per-core expertCount (counting-sort internal)
    int64_t totalLengthAligned = Align(totalLength_, sizeof(int32_t));
    sortedExpertIdxWsOffset_ = 0;
    rowIdxWsOffset_ = totalLengthAligned;
    pairsWsOffset_ = totalLengthAligned * 2;
    expertCountWsOffset_ = pairsWsOffset_ + filterNeedCoreNum_ * pairsPerCore_;

    sortedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + sortedExpertIdxWsOffset_);
    if (rowIdxType_ == GATHER) {
        rowIdxWorkspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + rowIdxWsOffset_);
    }
    pairsWorkspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + pairsWsOffset_);
    expertCountWorkspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + expertCountWsOffset_);

    // Short path: maxChunks_==1 means all pairs fit in one chunk
    shortPath_ = (coreEntries_ <= 2048) && (ep_ == 1);

    // Persistent UB region
    int64_t ubOff = 0;
    expertCountLocalOffset_ = ubOff;
    ubOff += OpsBaseCeilAlign(expertCountStride_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    if (shortPath_) {
        gatheredIdxLocalOffset_ = ubOff;
        ubOff += OpsBaseCeilAlign(2048 * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
        gatheredExpertLocalOffset_ = ubOff;
        ubOff += OpsBaseCeilAlign(2048 * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    }
    persistentSize_ = ubOff;

    // Phase A shared size
    int64_t phaseASize = 0;
    phaseASize +=
        OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES); // expertIdxLocal
    phaseASize +=
        OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(float)), BLOCK_BYTES); // expertIdxFp32Local
    phaseASize += maskBytes_ * 3;                                                           // compareMask0/1 + gatherMask
    phaseASize +=
        OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES); // flatIdxBufferLocal
    if (!shortPath_) {
        phaseASize += OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES) *
                      2; // gatheredExpert + gatheredIdx
    }

    // Phase B: batched loading (totalCount + prefixSum + batchBuf)
    int64_t expertCountAlign =
        OpsBaseCeilAlign(expertCountStride_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    int64_t fixedOverhead = persistentSize_ + 2 * expertCountAlign;
    int64_t maxBatchBufSize = static_cast<int64_t>(196608) - fixedOverhead - static_cast<int64_t>(1024);
    if (maxBatchBufSize < expertCountAlign) {
        maxBatchBufSize = expertCountAlign;
    }
    int64_t allCoresBufSize =
        OpsBaseCeilAlign(filterNeedCoreNum_ * expertCountStride_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    int64_t batchBufSize = Min(allCoresBufSize, maxBatchBufSize);
    int64_t phaseBSize = 2 * expertCountAlign + batchBufSize;
    totalCountLocalOffset_ = persistentSize_;
    prefixSumLocalOffset_ = persistentSize_ + expertCountAlign;
    oneCoreExpertCountLocalOffset_ = persistentSize_ + 2 * expertCountAlign;

    // Phase C: pairs buffer + idxBuf (much smaller without x/scale/quant buffers)
    int64_t phaseCSize = 0;
    if (!shortPath_) {
        phaseCSize += OpsBaseCeilAlign(2048 * 2 * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    }
    phaseCSize += BLOCK_BYTES; // idxBuf: 32B

    totalBufSize_ = persistentSize_ + Max(phaseASize, Max(phaseBSize, phaseCSize));
    pipe_->InitBuffer(buf_, totalBufSize_);
}

// ========================== Process ==========================
__aicore__ inline void MoeV3CutOriginT::Process()
{
    if (blockIdx_ >= filterNeedCoreNum_) {
        // Match SyncAll count of main path: Phase A + Phase C + expertCountTemp init
        SyncAll();
        SyncAll();
        SyncAll();
        return;
    }

    // ======= Phase A: Chunked filter + count (all cores parallel) =======
    FilterAndCountChunked();
    WriteExpertCountToWorkspace();

    SyncAll();

    // ======= Phase B: Global reduce + offset (all cores parallel) =======
    ComputeGlobalOffset();

    // ======= Phase C: Write sortedExpertIdx + expandedRowIdx (all cores parallel) =======
    WriteSortedOutput();

    SyncAll();

    // ======= Fill sentinel for entries beyond globalTotal_ (ep_==1 only) =======
    if (ep_ == 1) {
        FillSortedExpertIdxSentinel();
    }

    // ======= Initialize expertCountTemp for downstream ExpertTokensCount =======
    // ExpertTokensCount uses SetAtomicAdd on workspace[Align(n*k)*2], which overlaps
    // with our pairs workspace. After pairs are consumed, zero this region so
    // atomic add starts from 0.
    if (blockIdx_ == 0) {
        GlobalTensor<int32_t> expertCountTempGm;
        expertCountTempGm.SetGlobalBuffer(workspaceBase_ + Align(totalLength_, sizeof(int32_t)) * 2,
                                          actualExpertNum_);
        InitGlobalMemory(expertCountTempGm, actualExpertNum_, 0);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
    SyncAll();
}

// ========================== FilterAndCountChunked ==========================
__aicore__ inline void MoeV3CutOriginT::FilterAndCountChunked()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    int64_t pairCursor = 0;

    for (int64_t chunkIdx = 0; chunkIdx < maxChunks_; chunkIdx++) {
        int64_t chunkStart = chunkIdx * filterChunkSize_;
        int64_t chunkLength = Min(filterChunkSize_, coreEntries_ - chunkStart);
        int64_t gmFlatOffset = coreFlatStart_ + chunkStart;

        int64_t filteredInChunk = 0;

        if (ep_ == 0) {
            // ===== No filtering: all entries valid =====
            int64_t off = persistentSize_;
            int64_t expertIdxOff = off;
            off += OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
            int64_t flatIdxOff = off;
            off += OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
            int64_t expertIdxCopyOff = off;

            LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxOff / sizeof(int32_t)];
            LocalTensor<int32_t> flatIdxLocal = buf_.Get<int32_t>()[flatIdxOff / sizeof(int32_t)];
            LocalTensor<int32_t> expertIdxCopyLocal = buf_.Get<int32_t>()[expertIdxCopyOff / sizeof(int32_t)];

            if (chunkIdx > 0) {
                SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            }

            DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(chunkLength * sizeof(int32_t)),
                                         0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(expertIdxLocal, expertIdxGm_[gmFlatOffset], copyParams, padParams);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

            Adds(expertIdxCopyLocal, expertIdxLocal, static_cast<int32_t>(0), static_cast<int32_t>(chunkLength));
            ArithProgression<int32_t>(flatIdxLocal, static_cast<int32_t>(gmFlatOffset), 1,
                                      static_cast<int32_t>(chunkLength));
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

            for (int64_t j = 0; j < chunkLength; j++) {
                int32_t expertVal = expertIdxCopyLocal.GetValue(j);
                int32_t curCount = expertCountLocal.GetValue(expertVal);
                expertCountLocal.SetValue(expertVal, curCount + 1);
            }

            filteredInChunk = chunkLength;

            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyExtParams wpCopyParams{static_cast<uint16_t>(1),
                                           static_cast<uint32_t>(filteredInChunk * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairCursor], flatIdxLocal, wpCopyParams);
            DataCopyPad(pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairsPerCore_ / 2 + pairCursor], expertIdxLocal,
                        wpCopyParams);
        } else {
            // ===== Vector filter path: range [expertStart_, expertEnd_) =====
            int64_t off = persistentSize_;
            int64_t expertIdxOff = off;
            off += OpsBaseCeilAlign(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

            LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxOff / sizeof(int32_t)];
            DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(chunkLength * sizeof(int32_t)),
                                         0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(expertIdxLocal, expertIdxGm_[gmFlatOffset], copyParams, padParams);

            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

            int64_t alignedLen = Ceil(chunkLength, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;

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

            LocalTensor<float> expertIdxFp32Local = buf_.Get<float>()[expertIdxFp32Off / sizeof(float)];
            Cast(expertIdxFp32Local, expertIdxLocal, RoundMode::CAST_ROUND, alignedLen);
            PipeBarrier<PIPE_V>();

            LocalTensor<uint8_t> compareMask0 = buf_.Get<uint8_t>()[compareMask0Off];
            LocalTensor<uint8_t> compareMask1 = buf_.Get<uint8_t>()[compareMask1Off];
            LocalTensor<uint8_t> gatherMaskLocal = buf_.Get<uint8_t>()[gatherMaskOff];

            CompareScalar(compareMask0, expertIdxFp32Local, static_cast<float>(expertStart_), CMPMODE::GE, alignedLen);
            PipeBarrier<PIPE_V>();
            CompareScalar(compareMask1, expertIdxFp32Local, static_cast<float>(expertEnd_), CMPMODE::LT, alignedLen);
            PipeBarrier<PIPE_V>();
            And(gatherMaskLocal.ReinterpretCast<uint16_t>(), compareMask0.ReinterpretCast<uint16_t>(),
                compareMask1.ReinterpretCast<uint16_t>(),
                Ceil(alignedLen, MASK_STRIDE) * MASK_STRIDE / DST_REP_STRIDE / 2);
            PipeBarrier<PIPE_V>();

            LocalTensor<int32_t> gatheredExpertLocal = buf_.Get<int32_t>()[gatheredExpertOff / sizeof(int32_t)];
            uint64_t rsvdCnt = 0;
            GatherMaskParams gatherMaskParams;
            gatherMaskParams.repeatTimes = 1;
            gatherMaskParams.src0BlockStride = 1;
            gatherMaskParams.src0RepeatStride = DST_REP_STRIDE;
            gatherMaskParams.src1RepeatStride = DST_REP_STRIDE;
            GatherMask(gatheredExpertLocal, expertIdxLocal, gatherMaskLocal.ReinterpretCast<uint32_t>(), true,
                       static_cast<uint32_t>(chunkLength), gatherMaskParams, rsvdCnt);
            PipeBarrier<PIPE_V>();
            filteredInChunk = static_cast<int64_t>(rsvdCnt);

            if (filteredInChunk > 0) {
                LocalTensor<int32_t> flatIdxBufferLocal = buf_.Get<int32_t>()[flatIdxBufferOff / sizeof(int32_t)];
                ArithProgression<int32_t>(flatIdxBufferLocal, static_cast<int32_t>(gmFlatOffset), 1,
                                          static_cast<int32_t>(chunkLength));
                PipeBarrier<PIPE_V>();

                LocalTensor<int32_t> gatheredIdxLocal = buf_.Get<int32_t>()[gatheredIdxOff / sizeof(int32_t)];
                uint64_t idxRsvdCnt = 0;
                GatherMask(gatheredIdxLocal, flatIdxBufferLocal, gatherMaskLocal.ReinterpretCast<uint32_t>(), true,
                           static_cast<uint32_t>(chunkLength), gatherMaskParams, idxRsvdCnt);
                PipeBarrier<PIPE_V>();

                // Subtract expertStart_ -> gatheredExpertLocal becomes expertOffset
                Adds(gatheredExpertLocal, gatheredExpertLocal, static_cast<int32_t>(-expertStart_),
                     static_cast<int32_t>(filteredInChunk));

                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

                for (int64_t j = 0; j < filteredInChunk; j++) {
                    int32_t expertOffset = gatheredExpertLocal.GetValue(j);
                    int32_t curCount = expertCountLocal.GetValue(expertOffset);
                    expertCountLocal.SetValue(expertOffset, curCount + 1);
                }

                if (!shortPath_ || filteredInChunk >= 2048) {
                    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

                    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                                 static_cast<uint32_t>(filteredInChunk * sizeof(int32_t)), 0, 0, 0};
                    DataCopyPad(pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairCursor], gatheredIdxLocal,
                                copyParams);
                    DataCopyPad(pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairsPerCore_ / 2 + pairCursor],
                                gatheredExpertLocal, copyParams);
                }
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
__aicore__ inline void MoeV3CutOriginT::WriteExpertCountToWorkspace()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(expertCountStride_ * sizeof(int32_t)),
                                 0, 0, 0};
    DataCopyPad(expertCountWorkspaceGm_[blockIdx_ * expertCountStride_], expertCountLocal, copyParams);
}

// ========================== ComputeGlobalOffset ==========================
__aicore__ inline void MoeV3CutOriginT::ComputeGlobalOffset()
{
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expertCountWorkspaceGm_);

    int64_t expertCountAlign =
        OpsBaseCeilAlign(expertCountStride_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    int64_t availableForBatch = (totalBufSize_ - persistentSize_) - 2 * expertCountAlign;
    int64_t batchCores = availableForBatch / expertCountAlign;
    if (batchCores > filterNeedCoreNum_) {
        batchCores = filterNeedCoreNum_;
    }

    LocalTensor<int32_t> batchBufLocal = buf_.Get<int32_t>()[oneCoreExpertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> totalCountLocal = buf_.Get<int32_t>()[totalCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> prefixSumLocal = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    Duplicate(totalCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    Duplicate(prefixSumLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();

    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    for (int64_t batchStart = 0; batchStart < filterNeedCoreNum_; batchStart += batchCores) {
        int64_t batchEnd = Min(batchStart + batchCores, filterNeedCoreNum_);
        int64_t curBatchSize = batchEnd - batchStart;

        int64_t batchElements = curBatchSize * expertCountStride_;
        DataCopyExtParams loadParams{static_cast<uint16_t>(1), static_cast<uint32_t>(batchElements * sizeof(int32_t)),
                                     0, 0, 0};
        DataCopyPad(batchBufLocal, expertCountWorkspaceGm_[batchStart * expertCountStride_], loadParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        for (int64_t c = 0; c < curBatchSize; c++) {
            Add(totalCountLocal, totalCountLocal, batchBufLocal[c * expertCountStride_], expertCountStride_);
            PipeBarrier<PIPE_V>();
        }

        for (int64_t c = 0; c < curBatchSize; c++) {
            int64_t globalCoreIdx = batchStart + c;
            if (globalCoreIdx < blockIdx_) {
                Add(prefixSumLocal, prefixSumLocal, batchBufLocal[c * expertCountStride_], expertCountStride_);
                PipeBarrier<PIPE_V>();
            }
        }

        if (batchStart + batchCores < filterNeedCoreNum_) {
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        }
    }

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // Scalar: compute per-expert global offset -> store in expertCountLocal
    int64_t cumulativeSum = 0;
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        int32_t totalForExpert = totalCountLocal.GetValue(e);
        int32_t prefixForExpert = prefixSumLocal.GetValue(e);

        expertCountLocal.SetValue(e, static_cast<int32_t>(cumulativeSum) + prefixForExpert);
        cumulativeSum += totalForExpert;
    }

    // Store global total for FillSortedExpertIdxSentinel
    globalTotal_ = cumulativeSum;
}

// ========================== WriteSortedOutput ==========================
__aicore__ inline void MoeV3CutOriginT::WriteSortedOutput()
{
    if (totalPairCount_ <= 0) {
        return;
    }

    bool useWorkspace = !(shortPath_ && totalPairCount_ < 2048);
    if (useWorkspace) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(pairsWorkspaceGm_);
    }

    // Phase C layout: pairsBuf (if useWorkspace) + idxBuf
    int64_t off = persistentSize_;
    int64_t pairsOff = off;
    int64_t pairsBufCapacity = 2048;
    if (useWorkspace) {
        off += OpsBaseCeilAlign(pairsBufCapacity * 2 * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    }
    int64_t idxBufOff = off;
    off += BLOCK_BYTES;

    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    int64_t pairCursor = 0;
    while (pairCursor < totalPairCount_) {
        int64_t batchSize = Min(pairsBufCapacity, totalPairCount_ - pairCursor);
        batchSize = Min(batchSize, static_cast<int64_t>(2048));

        // Load pairs batch
        LocalTensor<int32_t> gatheredIdxLocal;
        LocalTensor<int32_t> gatheredExpertLocal;
        if (useWorkspace) {
            gatheredIdxLocal = buf_.Get<int32_t>()[pairsOff / sizeof(int32_t)];
            gatheredExpertLocal = buf_.Get<int32_t>()[pairsOff / sizeof(int32_t) + 2048];
            DataCopyPadExtParams<int32_t> intPadParams{false, 0, 0, 0};
            DataCopyExtParams pairLoadParams{static_cast<uint16_t>(1),
                                             static_cast<uint32_t>(batchSize * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(gatheredIdxLocal, pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairCursor], pairLoadParams,
                        intPadParams);
            DataCopyPad(gatheredExpertLocal,
                        pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairsPerCore_ / 2 + pairCursor], pairLoadParams,
                        intPadParams);
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        } else {
            gatheredIdxLocal = buf_.Get<int32_t>()[gatheredIdxLocalOffset_ / sizeof(int32_t)];
            gatheredExpertLocal = buf_.Get<int32_t>()[gatheredExpertLocalOffset_ / sizeof(int32_t)];
        }

        LocalTensor<int32_t> idxBuf = buf_.Get<int32_t>()[idxBufOff / sizeof(int32_t)];

        // For each pair: compute newPos, write sortedExpertIdx[newPos] and rowIdx[newPos]
        for (int64_t i = 0; i < batchSize; i++) {
            int32_t origFlatIdx = gatheredIdxLocal.GetValue(i);
            int32_t expertOffset = gatheredExpertLocal.GetValue(i);
            int32_t newPos = expertCountLocal.GetValue(expertOffset);
            expertCountLocal.SetValue(expertOffset, newPos + 1);

            // Write sortedExpertIdx[newPos] = expert ID
            // ep_==0: expert ID = expertOffset (original)
            // ep_==1: expert ID = expertOffset + expertStart_
            int32_t expertId = (ep_ == 1) ? (expertOffset + static_cast<int32_t>(expertStart_)) : expertOffset;
            idxBuf.SetValue(0, expertId);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(sortedExpertIdxGm_[newPos], idxBuf, idxCopyParams);

            // Write rowIdx:
            //   SCATTER: expandedRowIdx[newPos] = origFlatIdx (to output)
            //   GATHER:  rowIdxWorkspace[newPos] = origFlatIdx (to workspace, RowIdxGather inverts later)
            idxBuf.SetValue(0, origFlatIdx);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            if (rowIdxType_ == GATHER) {
                DataCopyPad(rowIdxWorkspaceGm_[newPos], idxBuf, idxCopyParams);
            } else {
                DataCopyPad(expandedRowIdxGm_[newPos], idxBuf, idxCopyParams);
            }
        }

        pairCursor += batchSize;
    }
}

// ========================== FillSortedExpertIdxSentinel ==========================
// For ep_==1: fill sortedExpertIdx entries beyond globalTotal_ with a value >= expertEnd
// so that ExpertTokensCount breaks correctly when it encounters them.
__aicore__ inline void MoeV3CutOriginT::FillSortedExpertIdxSentinel()
{
    if (blockIdx_ != 0) {
        return;
    }

    if (globalTotal_ >= totalLength_) {
        return;
    }

    int32_t sentinel = static_cast<int32_t>(expertEnd_);
    LocalTensor<int32_t> sentinelLocal = buf_.Get<int32_t>()[persistentSize_ / sizeof(int32_t)];
    int64_t sentinelBufElems = (totalBufSize_ - persistentSize_) / static_cast<int64_t>(sizeof(int32_t));
    Duplicate(sentinelLocal, sentinel, static_cast<int32_t>(sentinelBufElems));
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

    int64_t fillOffset = globalTotal_;
    int64_t fillRemaining = totalLength_ - globalTotal_;
    while (fillRemaining > 0) {
        int64_t chunk = Min(fillRemaining, sentinelBufElems);
        DataCopyExtParams fillParams{static_cast<uint16_t>(1), static_cast<uint32_t>(chunk * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(sortedExpertIdxGm_[fillOffset], sentinelLocal, fillParams);
        fillOffset += chunk;
        fillRemaining -= chunk;
        if (fillRemaining > 0) {
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_CUT_ORIGIN_T_H
