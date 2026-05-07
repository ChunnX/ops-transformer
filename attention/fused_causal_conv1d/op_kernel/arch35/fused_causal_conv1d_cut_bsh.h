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
 * \file fused_causal_conv1d_cut_bsh.h
 */

#ifndef FUSED_CAUSAL_CONV1D_CUT_BSH_H
#define FUSED_CAUSAL_CONV1D_CUT_BSH_H

#include "kernel_operator.h"
#include "./vf/compute.h"
#include "fused_causal_conv1d_cut_bsh_struct.h"

namespace FusedCausalConv1dCutBSHNs {

using namespace AscendC;

constexpr int32_t  BUFFER_NUM   = 2;
constexpr uint32_t ALIGN_BYTES  = 32;

template <typename T>
class FusedCausalConv1dCutBSH {
public:
    __aicore__ inline FusedCausalConv1dCutBSH() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR numAcceptedToken,
                                GM_ADDR numComputedTokens, GM_ADDR blockIdxFirst,
                                GM_ADDR blockIdxLast, GM_ADDR initialStateIdx,
                                GM_ADDR y, GM_ADDR workspace,
                                const FusedCausalConv1dCutBSHTilingData* tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseBSHTilingData(const FusedCausalConv1dCutBSHTilingData* tiling);
    __aicore__ inline void LoadMetaData();
    __aicore__ inline uint32_t FindBatchIdx(uint64_t globalSeqIdx);

    __aicore__ inline void ProcessMainCompute(uint64_t bsStart, uint64_t dimStart,
        uint32_t loopNumBS, uint32_t ubFactorBS, uint32_t ubTailFactorBS,
        uint32_t loopNumDim, uint32_t ubFactorDim, uint32_t ubTailFactorDim);
    __aicore__ inline void ProcessUBBlock(uint32_t bsStart, uint32_t bsSize, uint32_t dimStart, uint32_t dimSize,
        uint32_t iStart, uint32_t& curBatchIdx, uint32_t& curSequenceIdx);

    __aicore__ inline uint16_t ProcessTokensNeedCache(LocalTensor<T>& xLocal, LocalTensor<T>& yLocal,
        LocalTensor<T>& weightLocal, uint32_t i, uint32_t N, uint32_t dimSize, uint32_t dimBlocks,
        uint32_t dimStart, uint32_t cacheSkipBlocks, uint32_t curBatchLen, uint32_t curSequenceIdx,
        int64_t readCIdx, int64_t writeCIdx, uint32_t curBatchIdx, bool& reachBatchEnd);
    __aicore__ inline uint16_t ProcessTokensNoCache(LocalTensor<T>& xLocal, LocalTensor<T>& yLocal,
        LocalTensor<T>& weightLocal, uint32_t i, uint32_t N, uint32_t dimSize, uint32_t dimBlocks,
        uint32_t dimStart, uint32_t cacheSkipBlocks, uint32_t curBatchLen, uint32_t curSequenceIdx,
        int64_t writeCIdx, uint32_t curBatchIdx, bool& reachBatchEnd);

    __aicore__ inline void WriteCacheShortBatch(LocalTensor<T>& cacheLocal, LocalTensor<T>& xLocal,
        uint32_t i, uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart, uint32_t cacheSkipBlocks,
        uint32_t curBatchLen, int64_t writeCIdx, uint32_t curBatchIdx, uint32_t cachedStateLen);
    __aicore__ inline void WriteCacheLongBatch(LocalTensor<T>& xLocal, uint32_t i, uint16_t step,
        uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart, uint32_t cacheSkipBlocks,
        uint32_t curBatchLen, int64_t writeCIdx, uint32_t curBatchIdx, uint32_t cachedStateLen);

    __aicore__ inline void WriteDeferredCacheToStates();
    __aicore__ inline void FillApcBlockCaches();

    // APC 辅助：根据 batchIdx + blockIdx 标量读取 2D cache_indices[batchIdx, blockIdx]
    __aicore__ inline int64_t GetApcCacheLine(uint32_t batchIdx, uint32_t blockIdx) {
        return static_cast<int64_t>(cacheIndices2DGM_.GetValue(
            static_cast<uint64_t>(batchIdx) * maxNumBlocks_ + blockIdx));
    }

    // 辅助：read/write cIdx 解析（APC vs 1D vs None）
    __aicore__ inline void ResolveCacheIndices(uint32_t curBatchIdx, int64_t& readCIdx, int64_t& writeCIdx) {
        if (apcEnabled_) {
            int32_t initIdx = initialStateIdxLocal_.GetValue(curBatchIdx);
            int32_t lastIdx = blockIdxLastLocal_.GetValue(curBatchIdx);
            readCIdx  = GetApcCacheLine(curBatchIdx, static_cast<uint32_t>(initIdx));
            writeCIdx = GetApcCacheLine(curBatchIdx, static_cast<uint32_t>(lastIdx));
        } else if (hasCacheIndices_) {
            readCIdx  = static_cast<int64_t>(cacheIdxLocal_.GetValue(curBatchIdx));
            writeCIdx = readCIdx;
        } else {
            // cache_indices 为空，直接用 batch_idx 作为 cache line（对齐 golden）
            readCIdx  = static_cast<int64_t>(curBatchIdx);
            writeCIdx = readCIdx;
        }
    }

    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt) {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    static __aicore__ inline uint32_t AlignUp(uint32_t n, uint32_t align) {
        return (n + align - 1) / align * align;
    }

private:
    // -------------------------- Tiling 参数 --------------------------
    uint32_t dim_;
    uint32_t kernelWidth_;
    uint32_t batchSize_;
    uint32_t stateLen_;             // = cacheStride0_ / cacheStride1_

    uint32_t loopNumBS_;
    uint32_t loopNumDim_;
    uint32_t ubFactorBS_;
    uint32_t ubTailFactorBS_;
    uint32_t ubFactorDim_;
    uint32_t ubTailFactorDim_;

    uint32_t tailBlockloopNumBS_;
    uint32_t tailBlockloopNumDim_;
    uint32_t tailBlockubFactorBS_;
    uint32_t tailBlockubTailFactorBS_;
    uint32_t tailBlockubFactorDim_;
    uint32_t tailBlockubTailFactorDim_;

    uint32_t dimCoreNum_;
    uint32_t dimRemainderCores_;
    uint32_t dimBlockFactor_;
    uint32_t dimBlockTailFactor_;

    uint32_t bsRemainderCores_;
    uint32_t bsBlockFactor_;
    uint32_t bsBlockTailFactor_;

    uint32_t realCoreNum_;

    uint32_t xStride_;
    uint32_t cacheStride0_;
    uint32_t cacheStride1_;
    uint32_t residualConnection_;

    int64_t  padSlotId_;

    // ===== v2.0 新增 =====
    bool     apcEnabled_;
    uint32_t blockSize_;
    uint32_t maxNumBlocks_;
    uint32_t convMode_;
    bool     hasAcceptTokenNum_;
    bool     hasNumComputedTokens_;   // num_computed_tokens 是否可用（convMode!=0 || APC）
    bool     hasCacheIndices_;        // cache_indices 是否提供（为false时用 batch_idx 作为 cache line）
    bool     inplace_;

    // 运行时核索引
    uint32_t bsIdx_;
    uint32_t dimIdx_;

    // -------------------------- Pipeline & GM --------------------------
    TPipe pipe_;

    GlobalTensor<T>       xGM_;
    GlobalTensor<T>       weightGM_;
    GlobalTensor<T>       cacheStatesGM_;
    GlobalTensor<int32_t> cacheIndicesGM_;        // 1D 模式
    GlobalTensor<int32_t> cacheIndices2DGM_;      // 2D APC 模式
    GlobalTensor<int32_t> seqStartIndexGM_;
    GlobalTensor<int32_t> numComputedTokensGM_;
    GlobalTensor<int32_t> acceptTokenGM_;
    GlobalTensor<int32_t> blockIdxFirstGM_;
    GlobalTensor<int32_t> blockIdxLastGM_;
    GlobalTensor<int32_t> initialStateIdxGM_;
    GlobalTensor<T>       yGM_;

    // -------------------------- Queue（v2.0：4 Queue × 双 buffer = 8 槽位）--------------------------
    // xQueue:     TQueBind<VECIN, VECOUT>, BUF=2  (MTE2 搬入 + MTE3 写回 cache)
    // cacheQueue: TQueBind<VECIN, VECOUT>, BUF=2  (MTE2 搬入 + MTE3 写回 cache)
    // yQueue:     VECOUT, BUF=2
    // weightQueue: VECIN, BUF=2                   (双 buffer 支持 MTE2 与 Vector 流水并行)
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> cacheQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TQue<QuePosition::VECIN,  1> weightQueue_;

    // -------------------------- TBuf 元数据 --------------------------
    TBuf<TPosition::VECCALC> seqStartBuf_;
    TBuf<TPosition::VECCALC> cacheIdxBuf_;
    TBuf<TPosition::VECCALC> numComputedBuf_;
    TBuf<TPosition::VECCALC> acceptTokenBuf_;
    TBuf<TPosition::VECCALC> blockIdxFirstBuf_;
    TBuf<TPosition::VECCALC> blockIdxLastBuf_;
    TBuf<TPosition::VECCALC> initialStateIdxBuf_;

    // 持久持有的 LocalTensor
    LocalTensor<int32_t> seqStartLocal_;
    LocalTensor<int32_t> cacheIdxLocal_;
    LocalTensor<int32_t> numComputedLocal_;
    LocalTensor<int32_t> acceptTokenLocal_;
    LocalTensor<int32_t> blockIdxFirstLocal_;
    LocalTensor<int32_t> blockIdxLastLocal_;
    LocalTensor<int32_t> initialStateIdxLocal_;

    // -------------------------- 跨核 batch 跟踪 --------------------------
    uint32_t firstBatchIdx_;
    int64_t  firstBatchWriteCIdx_;
    bool     firstBatchComplete_;
    bool     firstBatchNeedsDeferredWrite_;
    uint32_t lastBatchIdx_;     // 当前核处理的最后一个 batch idx（用于 FillApcBlockCaches）
};

// ============================================================================
// ParseBSHTilingData
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::ParseBSHTilingData(const FusedCausalConv1dCutBSHTilingData* tiling)
{
    dim_                       = tiling->dim;
    kernelWidth_               = tiling->kernelWidth;
    batchSize_                 = tiling->batch;

    loopNumBS_                 = tiling->loopNumBS;
    loopNumDim_                = tiling->loopNumDim;
    ubFactorBS_                = tiling->ubFactorBS;
    ubTailFactorBS_            = tiling->ubTailFactorBS;
    ubFactorDim_               = tiling->ubFactorDim;
    ubTailFactorDim_           = tiling->ubTailFactorDim;

    tailBlockloopNumBS_        = tiling->tailBlockloopNumBS;
    tailBlockloopNumDim_       = tiling->tailBlockloopNumDim;
    tailBlockubFactorBS_       = tiling->tailBlockubFactorBS;
    tailBlockubTailFactorBS_   = tiling->tailBlockubTailFactorBS;
    tailBlockubFactorDim_      = tiling->tailBlockubFactorDim;
    tailBlockubTailFactorDim_  = tiling->tailBlockubTailFactorDim;

    dimCoreNum_                = tiling->dimCoreNum;
    dimRemainderCores_         = tiling->dimRemainderCores;
    dimBlockFactor_            = tiling->dimBlockFactor;
    dimBlockTailFactor_        = tiling->dimBlockTailFactor;

    bsRemainderCores_          = tiling->bsRemainderCores;
    bsBlockFactor_             = tiling->bsBlockFactor;
    bsBlockTailFactor_         = tiling->bsBlockTailFactor;

    realCoreNum_               = tiling->realCoreNum;

    xStride_                   = tiling->xStride;
    cacheStride0_              = tiling->cacheStride0;
    cacheStride1_              = tiling->cacheStride1;
    residualConnection_        = tiling->residualConnection;

    padSlotId_                 = tiling->padSlotId;

    // v2.0 新增
    apcEnabled_                = (tiling->apcEnabled != 0);
    blockSize_                 = static_cast<uint32_t>(tiling->blockSize);
    maxNumBlocks_              = static_cast<uint32_t>(tiling->maxNumBlocks);
    convMode_                  = static_cast<uint32_t>(tiling->convMode);
    hasAcceptTokenNum_         = (tiling->hasAcceptTokenNum != 0);
    hasNumComputedTokens_      = (tiling->hasNumComputedTokens != 0);
    hasCacheIndices_           = (tiling->hasCacheIndices != 0);
    inplace_                   = (tiling->inplace != 0);

    // 派生：state_len（cache 第二维大小）
    stateLen_ = (cacheStride1_ != 0) ? (cacheStride0_ / cacheStride1_) : (kernelWidth_ - 1);
}

// ============================================================================
// Init — v2.0：12 GM_ADDR 入参；inplace 复用 x；TBuf 元数据；4 Queue × 双 buffer（8 槽）
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates,
    GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR numAcceptedToken,
    GM_ADDR numComputedTokens, GM_ADDR blockIdxFirst, GM_ADDR blockIdxLast,
    GM_ADDR initialStateIdx, GM_ADDR y, GM_ADDR workspace,
    const FusedCausalConv1dCutBSHTilingData* tiling)
{
    (void)workspace;
    ParseBSHTilingData(tiling);

    uint32_t blockIdx = GetBlockIdx();
    uint32_t dimCN = (dimCoreNum_ == 0) ? 1 : dimCoreNum_;
    bsIdx_  = blockIdx / dimCN;
    dimIdx_ = blockIdx - bsIdx_ * dimCN;

    bool isBsTailCore  = (bsIdx_  >= bsRemainderCores_);
    bool isDimTailCore = (dimIdx_ >= dimRemainderCores_);
    uint32_t maxUbBS  = isBsTailCore  ? tailBlockubFactorBS_  : ubFactorBS_;
    uint32_t maxUbDim = isDimTailCore ? tailBlockubFactorDim_ : ubFactorDim_;

    // ---------- GM 绑定 ----------
    uint64_t cuSeqLen = tiling->cuSeqLen;

    // inplace：y 复用 x 的 GM；非 inplace：y 独立 GM（步长 = dim_）
    uint64_t yStride = inplace_ ? static_cast<uint64_t>(xStride_) : static_cast<uint64_t>(dim_);
    yGM_.SetGlobalBuffer((__gm__ T*)(inplace_ ? x : y), cuSeqLen * yStride);

    xGM_.SetGlobalBuffer((__gm__ T*)x, cuSeqLen * static_cast<uint64_t>(xStride_));
    weightGM_.SetGlobalBuffer((__gm__ T*)weight, static_cast<uint64_t>(kernelWidth_) * dim_);
    cacheStatesGM_.SetGlobalBuffer((__gm__ T*)convStates);
    seqStartIndexGM_.SetGlobalBuffer((__gm__ int32_t*)queryStartLoc, batchSize_ + 1);

    if (apcEnabled_) {
        cacheIndices2DGM_.SetGlobalBuffer((__gm__ int32_t*)cacheIndices,
            static_cast<uint64_t>(batchSize_) * maxNumBlocks_);
        if (initialStateIdx != nullptr) {
            initialStateIdxGM_.SetGlobalBuffer((__gm__ int32_t*)initialStateIdx, batchSize_);
        }
        if (blockIdxFirst != nullptr) {
            blockIdxFirstGM_.SetGlobalBuffer((__gm__ int32_t*)blockIdxFirst, batchSize_);
        }
        if (blockIdxLast != nullptr) {
            blockIdxLastGM_.SetGlobalBuffer((__gm__ int32_t*)blockIdxLast, batchSize_);
        }
    } else {
        if (hasCacheIndices_) {
            cacheIndicesGM_.SetGlobalBuffer((__gm__ int32_t*)cacheIndices, batchSize_);
        }
    }

    // hasNumComputedTokens_ 已由 ParseBSHTilingData 从 tiling 字段读取
    if (hasNumComputedTokens_ && numComputedTokens != nullptr) {
        numComputedTokensGM_.SetGlobalBuffer((__gm__ int32_t*)numComputedTokens, batchSize_);
    }
    if (hasAcceptTokenNum_ && numAcceptedToken != nullptr) {
        acceptTokenGM_.SetGlobalBuffer((__gm__ int32_t*)numAcceptedToken, batchSize_);
    }

    // ---------- Queue 初始化（4 Queue × 双 buffer = 8 槽位） ----------
    uint32_t K = kernelWidth_;
    uint32_t weightBufBytes = AlignUp(K * maxUbDim * sizeof(T), ALIGN_BYTES);
    // cacheBufBytes 按 stateLen_ 分配：cachedStateLen 最大值为 stateLen_，
    // 正向加载时需要容纳 conv_states[readCIdx][0:stateLen_] 的全部行
    uint32_t cacheBufBytes  = AlignUp(stateLen_ * maxUbDim * sizeof(T), ALIGN_BYTES);
    uint32_t startLocBytes  = AlignUp((batchSize_ + 1) * sizeof(int32_t), ALIGN_BYTES);
    uint32_t xBufBytes      = AlignUp(maxUbBS * maxUbDim * sizeof(T), ALIGN_BYTES);
    uint32_t yBufBytes      = xBufBytes;

    pipe_.InitBuffer(xQueue_,        BUFFER_NUM,  xBufBytes);
    pipe_.InitBuffer(yQueue_,        BUFFER_NUM,  yBufBytes);
    pipe_.InitBuffer(cacheQueue_,    BUFFER_NUM,  cacheBufBytes);
    pipe_.InitBuffer(weightQueue_,   BUFFER_NUM,  weightBufBytes);

    // ---------- TBuf 初始化（不占 Queue 槽位） ----------
    pipe_.InitBuffer(seqStartBuf_, startLocBytes);
    uint32_t metaBytes = AlignUp(batchSize_ * sizeof(int32_t), ALIGN_BYTES);
    if (hasCacheIndices_ && !apcEnabled_) {
        pipe_.InitBuffer(cacheIdxBuf_, metaBytes);  // 1D 模式承载 cache_indices
    }
    if (hasNumComputedTokens_) {
        pipe_.InitBuffer(numComputedBuf_, metaBytes);
    }
    if (hasAcceptTokenNum_) {
        pipe_.InitBuffer(acceptTokenBuf_, metaBytes);
    }
    if (apcEnabled_) {
        pipe_.InitBuffer(blockIdxFirstBuf_,    metaBytes);
        pipe_.InitBuffer(blockIdxLastBuf_,     metaBytes);
        pipe_.InitBuffer(initialStateIdxBuf_,  metaBytes);
    }

    // 跨核跟踪初值
    firstBatchIdx_                 = 0;
    firstBatchWriteCIdx_           = 0;
    firstBatchComplete_            = true;
    firstBatchNeedsDeferredWrite_  = false;
    lastBatchIdx_                  = 0;
}

// ============================================================================
// LoadMetaData — v2.0：seqStart 走 TBuf（一次性搬入后持久持有，避免占用 Queue 槽位）；其余元数据走 TBuf
//   - 1D：cacheIdxBuf 承载 cache_indices
//   - APC：blockIdxFirst / blockIdxLast / initialStateIdx 三个 TBuf
//   - convMode==1：numComputedBuf
//   - hasAcceptTokenNum：acceptTokenBuf
// 元数据加载后通过 PipeBarrier<PIPE_ALL> 确保后续 scalar GetValue 可见
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::LoadMetaData()
{
    // seqStartIndex 经 TBuf 搬入并持久持有（由 Queue 改为 TBuf，避免占用 buffer 槽位）
    seqStartLocal_ = seqStartBuf_.Get<int32_t>();
    {
        DataCopyExtParams cpParams{1, static_cast<uint16_t>((batchSize_ + 1) * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(seqStartLocal_, seqStartIndexGM_[0], cpParams, padParams);
    }

    DataCopyExtParams metaCp{1, static_cast<uint16_t>(batchSize_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> metaPad{false, 0, 0, 0};

    // 1D cache_indices（APC 模式下不使用 cacheIdxLocal_；cache_indices 为空时也不加载）
    if (hasCacheIndices_ && !apcEnabled_) {
        cacheIdxLocal_ = cacheIdxBuf_.Get<int32_t>();
        DataCopyPad(cacheIdxLocal_, cacheIndicesGM_[0], metaCp, metaPad);
    }

    // APC：三组下标
    if (apcEnabled_) {
        blockIdxFirstLocal_   = blockIdxFirstBuf_.Get<int32_t>();
        blockIdxLastLocal_    = blockIdxLastBuf_.Get<int32_t>();
        initialStateIdxLocal_ = initialStateIdxBuf_.Get<int32_t>();
        DataCopyPad(blockIdxFirstLocal_,   blockIdxFirstGM_[0],   metaCp, metaPad);
        DataCopyPad(blockIdxLastLocal_,    blockIdxLastGM_[0],    metaCp, metaPad);
        DataCopyPad(initialStateIdxLocal_, initialStateIdxGM_[0], metaCp, metaPad);
    }

    // num_computed_tokens 决定 cache 是否零初始化（golden 不区分 convMode）
    if (hasNumComputedTokens_) {
        numComputedLocal_ = numComputedBuf_.Get<int32_t>();
        DataCopyPad(numComputedLocal_, numComputedTokensGM_[0], metaCp, metaPad);
    }

    // MTP：num_accepted_tokens 决定 cache 读取偏移
    if (hasAcceptTokenNum_) {
        acceptTokenLocal_ = acceptTokenBuf_.Get<int32_t>();
        DataCopyPad(acceptTokenLocal_, acceptTokenGM_[0], metaCp, metaPad);
    }

    // 等待全部 MTE2 完成且对 scalar 可见
    PipeBarrier<PIPE_ALL>();
}

// ============================================================================
// FindBatchIdx：二分查找 globalSeqIdx 所属 batch（保证 [start[i], start[i+1]) 包含）
// ============================================================================
template <typename T>
__aicore__ inline uint32_t FusedCausalConv1dCutBSH<T>::FindBatchIdx(uint64_t globalSeqIdx)
{
    uint32_t lo = 0;
    uint32_t hi = batchSize_ - 1;
    while (lo < hi) {
        uint32_t mid = (lo + hi + 1) / 2;
        if ((uint64_t)seqStartLocal_.GetValue(mid) <= globalSeqIdx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// ============================================================================
// ProcessUBBlock — v2.0：拆分 xQueue/yQueue/cacheQueue；APC 时 readCIdx≠writeCIdx
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::ProcessUBBlock(uint32_t bsStart, uint32_t bsSize,
    uint32_t dimStart, uint32_t dimSize, uint32_t iStart, uint32_t& curBatchIdx, uint32_t& curSequenceIdx)
{
    uint32_t K = kernelWidth_;
    uint32_t N = bsSize;

    uint32_t dimBlocks        = dimSize * sizeof(T) / ALIGN_BYTES;
    uint32_t weightSkipBlocks = (dim_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t xSkipBlocks      = (xStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t cacheSkipBlocks  = (cacheStride1_ - dimSize) * sizeof(T) / ALIGN_BYTES;

    uint64_t batchStart  = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
    uint64_t batchEnd    = (uint64_t)seqStartLocal_.GetValue(curBatchIdx + 1);
    uint32_t curBatchLen = (uint32_t)(batchEnd - batchStart);

    // ---- 加载 weight ----
    LocalTensor<T> weightLocal = weightQueue_.AllocTensor<T>();
    {
        DataCopyExtParams wcp{static_cast<uint16_t>(K), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              weightSkipBlocks * ALIGN_BYTES, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(weightLocal, weightGM_[dimStart], wcp, padParams);
    }
    weightQueue_.EnQue(weightLocal);
    weightLocal = weightQueue_.DeQue<T>();

    // ---- 加载 x ----
    LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
    {
        DataCopyExtParams xcp{static_cast<uint16_t>(bsSize), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              xSkipBlocks * ALIGN_BYTES, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGM_[bsStart * xStride_ + dimStart], xcp, padParams);
    }
    xQueue_.EnQue(xLocal);
    xLocal = xQueue_.DeQue<T>();

    // ---- 申请 yLocal（独立 yQueue，每个 UB 块一份） ----
    LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();

    uint32_t i = iStart;
    while (i < N) {
        // 解析 read/write cache 行（APC 与 1D 分支）
        int64_t readCIdx, writeCIdx;
        ResolveCacheIndices(curBatchIdx, readCIdx, writeCIdx);

        // padSlot 跳过：使用 writeCIdx 判定（写无效则该 batch 视为无效）
        if (writeCIdx == padSlotId_) {
            uint32_t remainInBatch = curBatchLen - curSequenceIdx;
            uint16_t skip = (N - i < remainInBatch) ? (N - i) : remainInBatch;
            i += skip;
            curSequenceIdx += skip;
            bool reachEnd = (skip == remainInBatch);
            if (reachEnd && curBatchIdx + 1 < batchSize_) {
                curBatchIdx++;
                batchStart   = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
                batchEnd     = (uint64_t)seqStartLocal_.GetValue(curBatchIdx + 1);
                curBatchLen  = (uint32_t)(batchEnd - batchStart);
                curSequenceIdx = 0;
            }
            continue;
        }

        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        uint16_t step = 0;
        bool reachBatchEnd = false;

        if (curSequenceIdx < K - 1) {
            step = ProcessTokensNeedCache(
                xLocal, yLocal, weightLocal, i, N, dimSize, dimBlocks,
                dimStart, cacheSkipBlocks, curBatchLen, curSequenceIdx,
                readCIdx, writeCIdx, curBatchIdx, reachBatchEnd);
        } else {
            step = ProcessTokensNoCache(
                xLocal, yLocal, weightLocal, i, N, dimSize, dimBlocks,
                dimStart, cacheSkipBlocks, curBatchLen, curSequenceIdx,
                writeCIdx, curBatchIdx, reachBatchEnd);
        }

        i += step;
        curSequenceIdx += step;

        if (reachBatchEnd && curBatchIdx + 1 < batchSize_) {
            curBatchIdx++;
            batchStart   = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
            batchEnd     = (uint64_t)seqStartLocal_.GetValue(curBatchIdx + 1);
            curBatchLen  = (uint32_t)(batchEnd - batchStart);
            curSequenceIdx = 0;
        }
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    weightQueue_.FreeTensor(weightLocal);
    xQueue_.FreeTensor(xLocal);
    yQueue_.FreeTensor(yLocal);
}

// ============================================================================
// ProcessTokensNeedCache — v2.0：
//   - convMode==1 时按 numComputedTokens[batch]==0 决定 cache 零初始化
//   - MTP：cache 读取固定从尾部 K-1 行（与 write 位置一致）
//   - 输出走 yLocal（独立 yQueue），yGM 写出步长 = inplace ? xStride : dim
// ============================================================================
template <typename T>
__aicore__ inline uint16_t FusedCausalConv1dCutBSH<T>::ProcessTokensNeedCache(LocalTensor<T>& xLocal,
    LocalTensor<T>& yLocal, LocalTensor<T>& weightLocal, uint32_t i, uint32_t N,
    uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart, uint32_t cacheSkipBlocks,
    uint32_t curBatchLen, uint32_t curSequenceIdx,
    int64_t readCIdx, int64_t writeCIdx, uint32_t curBatchIdx, bool& reachBatchEnd)
{
    uint32_t K = kernelWidth_;

    uint16_t step = K - 1 - curSequenceIdx;
    if (step > N - i) step = N - i;
    if (step > curBatchLen - curSequenceIdx) step = curBatchLen - curSequenceIdx;
    reachBatchEnd = (step == curBatchLen - curSequenceIdx);

    // useZeroCache：num_computed_tokens[curBatchIdx]==0 → 强制零初始化（golden 不区分 convMode）
    bool useZeroCache = false;
    if (hasNumComputedTokens_) {
        useZeroCache = (numComputedLocal_.GetValue(curBatchIdx) == 0);
    }

    // cache 读取行偏移 + 有效 cache 行数钳制
    // golden: offset = accepted-1 (MTP) 或 stateLen-(K-1) (无MTP)
    // 注：useZeroCache 时 golden 的 offset=0
    uint32_t mtpOffset = 0;
    uint32_t effectiveState = K - 1;  // 可读取的有效 cache 行数（默认 K-1）
    if (useZeroCache) {
        mtpOffset = 0;
    } else if (hasAcceptTokenNum_) {
        int32_t accepted = acceptTokenLocal_.GetValue(curBatchIdx);
        if (accepted > 0) {
            mtpOffset = static_cast<uint32_t>(accepted - 1);
            // 钳制：当 accepted-1 + K-1 > stateLen_ 时减少有效行数，防止 GM OOB 读取
            if (mtpOffset >= stateLen_) {
                effectiveState = 0;
            } else if (mtpOffset + (K - 1) > stateLen_) {
                effectiveState = stateLen_ - mtpOffset;
            }
        }
    } else {
        // 无 MTP：golden offset = stateLen-(K-1)
        mtpOffset = stateLen_ - (K - 1);
    }

    // 计算 cachedStateLen：golden padded_input 中 cached_state 的行数
    // golden: offset = accepted-1 (MTP) 或 stateLen-(K-1) (无MTP)
    //         cached_state = conv_states[readCIdx][:offset+K-1]
    uint32_t cachedStateLen;
    if (useZeroCache) {
        cachedStateLen = K - 1;
    } else if (hasAcceptTokenNum_) {
        // MTP：cachedStateLen = mtpOffset + (K-1)，与 golden offset+width-1 对齐
        cachedStateLen = mtpOffset + (K - 1);
        if (cachedStateLen > stateLen_) cachedStateLen = stateLen_;
    } else {
        // 无 MTP：offset = stateLen-(K-1)，cachedStateLen = stateLen
        cachedStateLen = stateLen_;
    }

    // ---- 加载 / 初始化 cacheLocal ----
    // 正向加载：cacheLocal[0:cachedStateLen] = conv_states[readCIdx][0:cachedStateLen]
    // 卷积计算时 stateSlice 从 cacheLocal[(mtpOffset+seqPos)*dimSize] 读取
    LocalTensor<T> cacheLocal = cacheQueue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (!useZeroCache && cachedStateLen > 0) {
        uint64_t cacheGmOffset = (uint64_t)readCIdx * cacheStride0_ + dimStart;
        DataCopyExtParams ccp{static_cast<uint16_t>(cachedStateLen), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              cacheSkipBlocks * ALIGN_BYTES, 0, 0};
        DataCopyPad(cacheLocal, cacheStatesGM_[cacheGmOffset], ccp, padParams);
    } else {
        Duplicate(cacheLocal, (T)0, cachedStateLen * dimSize);
    }
    cacheQueue_.EnQue(cacheLocal);
    cacheLocal = cacheQueue_.DeQue<T>();

    // 短 batch + reachEnd 的 WriteCacheShortBatch 已挪到 conv 循环之后，与 Plan A 前缀写一起复用 V_MTE3 同步。

    // ---- 卷积计算与输出 ----
    uint32_t ySkipBlocks = (inplace_ ? (xStride_ - dimSize) : (dim_ - dimSize)) * sizeof(T) / ALIGN_BYTES;
    uint64_t yRowStride  = inplace_ ? xStride_ : dim_;
    uint64_t yGmOffset   = ((uint64_t)seqStartLocal_.GetValue(curBatchIdx) + curSequenceIdx) * yRowStride + dimStart;
    DataCopyExtParams ycp{step, static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                          0, ySkipBlocks * ALIGN_BYTES, 0};

    // ---- 盘古v2 conv_mode=1 零填充 ----
    // golden: last_reset_idx = K-1 - num_computed_tokens[batch], clamp [0, seq_len]
    //         result[:last_reset_idx] = 0
    // 注：conv_mode=1 时始终需要零填充，不仅限于 useZeroCache
    uint32_t lastResetIdx = 0;
    if (convMode_ == 1 && hasNumComputedTokens_) {
        int32_t nct = numComputedLocal_.GetValue(curBatchIdx);
        int32_t lri = static_cast<int32_t>(K - 1) - nct;
        if (lri > 0) {
            lastResetIdx = static_cast<uint32_t>(lri);
            // lastResetIdx 是全局 seq 位置，需要减去 curSequenceIdx 得到本块内偏移
            if (lastResetIdx > curSequenceIdx) {
                lastResetIdx -= curSequenceIdx;
            } else {
                lastResetIdx = 0;
            }
            if (lastResetIdx > step) lastResetIdx = step;
        }
    }

    if (convMode_ == 1 && useZeroCache) {
        // num_computed_tokens==0：lastResetIdx=K-1=step，全部清零
        if (residualConnection_) {
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            // 修复：xLocal 索引应为 (i + j) 而非 (i - curSequenceIdx + j)
            // (i - curSequenceIdx + j) 给出的是 batch 内第 j 个 token，而非当前 seqPos 对应的 token
            // 当 curSequenceIdx > 0 时（batch 跨 UB 块），前者指向错误位置
            for (uint32_t j = 0; j < step; j++) {
                Adds(yLocal[j * dimSize], xLocal[(i + j) * dimSize], (T)0, dimSize);
            }
        } else {
            Duplicate(yLocal, (T)0, step * dimSize);
        }
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(yGM_[yGmOffset], yLocal, ycp);
    } else {
        for (uint32_t j = 0; j < step; j++) {
            uint32_t seqPos = curSequenceIdx + j;
            // MTP OOB 修正：当 effectiveState < K-1 时，减少 state 行并增加 x 行，
            // 同时偏移 xSlice 起点，与 golden 的 Python slicing 截断行为一致
            uint32_t stateSLen = (seqPos < effectiveState) ? (effectiveState - seqPos) : 0;
            uint32_t xSLen = K - stateSLen;
            uint32_t xOffset = (seqPos >= effectiveState) ? (seqPos - effectiveState) : 0;
            LocalTensor<T> xSlice = xLocal[(i - curSequenceIdx + xOffset) * dimSize];
            LocalTensor<T> stateSlice = cacheLocal[((stateSLen > 0) ? (mtpOffset + seqPos) : 0) * dimSize];
            LocalTensor<T> ySlice     = yLocal[j * dimSize];
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            Conv1dNeedState(xSlice, weightLocal, stateSlice, ySlice, stateSLen, xSLen, dimSize, residualConnection_);
        }
        // conv_mode=1 零填充：将前 lastResetIdx 个 token 的卷积结果置零
        // golden: result[:last_reset_idx] = 0; out = result + seq_x if residual else result
        // 当 residualConnection_ 时，输出应为 0 + x = x（而非 0）
        if (lastResetIdx > 0) {
            for (uint32_t j = 0; j < lastResetIdx; j++) {
                if (residualConnection_) {
                    // 输出 = 0 + x[curSequenceIdx+j]，即复制当前 token 的 x 到 y
                    // 修复：xLocal 索引应为 (i + j) 而非 (i - curSequenceIdx + j)
                    Adds(yLocal[j * dimSize], xLocal[(i + j) * dimSize], (T)0, dimSize);
                } else {
                    Duplicate(yLocal[j * dimSize], (T)0, dimSize);
                }
            }
        }
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(yGM_[yGmOffset], yLocal, ycp);
    }

    // Plan A (方案1)：在 conv 循环 + yLocal DataCopyPad 之后再写 cacheLocal 前缀。
    // 此时 cacheLocal 已被 V 流水消费完，复用上面的 V_MTE3 同步即可保证 MTE3 安全读取，
    // 不依赖 MTE2_MTE3 事件。
    // 正向加载后 cacheLocal[0:cachedStateLen] 连续存储，写回时用 cachedStateLen 替代 K-1
    if (curBatchLen < K && reachBatchEnd) {
        WriteCacheShortBatch(cacheLocal, xLocal, i - curSequenceIdx, dimSize, dimBlocks, dimStart,
                             cacheSkipBlocks, curBatchLen, writeCIdx, curBatchIdx, cachedStateLen);
    }
    else if (curSequenceIdx == 0 && curBatchLen >= K) {
        uint32_t cFromCachePA, cFromXPA;
        if (curBatchLen >= stateLen_) {
            cFromCachePA = 0;
            cFromXPA = stateLen_;
        } else if (curBatchLen + cachedStateLen >= stateLen_) {
            cFromCachePA = stateLen_ - curBatchLen;
            cFromXPA = curBatchLen;
        } else {
            cFromCachePA = cachedStateLen;
            cFromXPA = curBatchLen;
        }
        bool needDeferredPA = (curBatchIdx == firstBatchIdx_ && !firstBatchComplete_);
        if (needDeferredPA) {
            firstBatchNeedsDeferredWrite_ = true;
        } else if (cFromCachePA > 0) {
            uint32_t tailRowOffPA = stateLen_ - cFromCachePA - cFromXPA;
            // 正向加载：cacheLocal 末尾 cFromCachePA 行的起始行号 = cachedStateLen - cFromCachePA
            uint32_t cacheRowStartPA = cachedStateLen - cFromCachePA;
            uint64_t csPrefixOff = (uint64_t)writeCIdx * cacheStride0_
                                   + (uint64_t)tailRowOffPA * cacheStride1_ + dimStart;
            DataCopyExtParams wcpPA{static_cast<uint16_t>(cFromCachePA),
                                    static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                                    0, cacheSkipBlocks * ALIGN_BYTES, 0};
            DataCopyPad(cacheStatesGM_[csPrefixOff], cacheLocal[cacheRowStartPA * dimSize], wcpPA);
        }
    }
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    cacheQueue_.FreeTensor(cacheLocal);
    return step;
}

// ============================================================================
// ProcessTokensNoCache — v2.0：独立 yLocal 输出，writeCIdx 写回
// ============================================================================
template <typename T>
__aicore__ inline uint16_t FusedCausalConv1dCutBSH<T>::ProcessTokensNoCache(LocalTensor<T>& xLocal,
    LocalTensor<T>& yLocal, LocalTensor<T>& weightLocal, uint32_t i, uint32_t N,
    uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart, uint32_t cacheSkipBlocks,
    uint32_t curBatchLen, uint32_t curSequenceIdx,
    int64_t writeCIdx, uint32_t curBatchIdx, bool& reachBatchEnd)
{
    uint32_t K = kernelWidth_;

    uint32_t remainInBatch = curBatchLen - curSequenceIdx;
    uint16_t step = (N - i < remainInBatch) ? (N - i) : remainInBatch;
    reachBatchEnd = (step == remainInBatch);

    // 计算 cachedStateLen：与 ProcessTokensNeedCache 保持一致
    bool useZeroCacheNC = false;
    if (hasNumComputedTokens_) {
        useZeroCacheNC = (numComputedLocal_.GetValue(curBatchIdx) == 0);
    }
    uint32_t mtpOffsetNC = 0;
    if (useZeroCacheNC) {
        mtpOffsetNC = 0;
    } else if (hasAcceptTokenNum_) {
        int32_t accepted = acceptTokenLocal_.GetValue(curBatchIdx);
        if (accepted > 0) mtpOffsetNC = static_cast<uint32_t>(accepted - 1);
    } else {
        // 无 MTP：golden offset = stateLen-(K-1)
        mtpOffsetNC = stateLen_ - (K - 1);
    }
    uint32_t cachedStateLenNC;
    if (useZeroCacheNC) {
        cachedStateLenNC = K - 1;
    } else if (hasAcceptTokenNum_) {
        cachedStateLenNC = mtpOffsetNC + (K - 1);
        if (cachedStateLenNC > stateLen_) cachedStateLenNC = stateLen_;
    } else {
        cachedStateLenNC = stateLen_;
    }

    if (reachBatchEnd) {
        // V_MTE3 同步：确保 xLocal（DeQue→VECIN）对 MTE3 可见
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        WriteCacheLongBatch(xLocal, i, step, dimSize, dimBlocks, dimStart,
                            cacheSkipBlocks, curBatchLen, writeCIdx, curBatchIdx, cachedStateLenNC);
    }

    uint32_t xStartIdx = i - (K - 1);
    LocalTensor<T> xSlice = xLocal[xStartIdx * dimSize];
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    Conv1dNoNeedState(xSlice, weightLocal, yLocal, step, dimSize, residualConnection_);

    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    uint32_t ySkipBlocks = (inplace_ ? (xStride_ - dimSize) : (dim_ - dimSize)) * sizeof(T) / ALIGN_BYTES;
    uint64_t yRowStride  = inplace_ ? xStride_ : dim_;
    uint64_t yGmOffset = ((uint64_t)seqStartLocal_.GetValue(curBatchIdx) + curSequenceIdx) * yRowStride + dimStart;
    DataCopyExtParams ycp{step, static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                          0, ySkipBlocks * ALIGN_BYTES, 0};
    DataCopyPad(yGM_[yGmOffset], yLocal, ycp);

    return step;
}

// ============================================================================
// WriteCacheShortBatch — Plan A：golden 语义
//   padded_input = [cacheLocal (cachedStateLen)] + [xLocal seq_x (curBatchLen)]
//   cache_len = min(stateLen, cachedStateLen+curBatchLen)
//   conv_states[-cache_len:] = padded_input[-cache_len:]
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::WriteCacheShortBatch(LocalTensor<T>& cacheLocal,
    LocalTensor<T>& xLocal, uint32_t i, uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart,
    uint32_t cacheSkipBlocks, uint32_t curBatchLen, int64_t writeCIdx, uint32_t curBatchIdx,
    uint32_t cachedStateLen)
{
    bool needDeferredWrite = (curBatchIdx == firstBatchIdx_ && !firstBatchComplete_);
    if (needDeferredWrite) {
        firstBatchNeedsDeferredWrite_ = true;
        return;
    }

    uint32_t cFromCache, cFromX;
    if (curBatchLen >= stateLen_) {
        cFromCache = 0;
        cFromX = stateLen_;
    } else if (curBatchLen + cachedStateLen >= stateLen_) {
        cFromCache = stateLen_ - curBatchLen;
        cFromX = curBatchLen;
    } else {
        cFromCache = cachedStateLen;
        cFromX = curBatchLen;
    }
    uint32_t tailRowOff = stateLen_ - cFromCache - cFromX;

    uint64_t csBase = (uint64_t)writeCIdx * cacheStride0_
                      + (uint64_t)tailRowOff * cacheStride1_ + dimStart;

    // 调用点已位于 conv 循环之后，外层已 SetWaitFlag<V_MTE3>，此处无需再次同步。
    if (cFromCache > 0) {
        // 正向加载：cacheLocal 末尾 cFromCache 行的起始行号 = cachedStateLen - cFromCache
        uint32_t cacheRowStart = cachedStateLen - cFromCache;
        DataCopyExtParams wcp{static_cast<uint16_t>(cFromCache),
                              static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              0, cacheSkipBlocks * ALIGN_BYTES, 0};
        DataCopyPad(cacheStatesGM_[csBase], cacheLocal[cacheRowStart * dimSize], wcp);
    }
    if (cFromX > 0) {
        // xLocal 末尾 cFromX 行（短 batch 场景下 cFromX==curBatchLen）
        uint32_t xRowStart = i + (curBatchLen - cFromX);
        DataCopyExtParams wcp2{static_cast<uint16_t>(cFromX),
                               static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                               0, cacheSkipBlocks * ALIGN_BYTES, 0};
        DataCopyPad(cacheStatesGM_[csBase + (uint64_t)cFromCache * cacheStride1_],
                    xLocal[xRowStart * dimSize], wcp2);
    }
}

// ============================================================================
// WriteCacheLongBatch — Plan A：仅写 xLocal 提供的 cFromX 行
//   cacheLocal 提供的 cFromCache 前缀已由 ProcessTokensNeedCache 写出。
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::WriteCacheLongBatch(LocalTensor<T>& xLocal,
    uint32_t i, uint16_t step, uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart,
    uint32_t cacheSkipBlocks, uint32_t curBatchLen, int64_t writeCIdx, uint32_t curBatchIdx,
    uint32_t cachedStateLen)
{
    bool needDeferredWrite = (curBatchIdx == firstBatchIdx_ && !firstBatchComplete_);
    if (needDeferredWrite) {
        firstBatchNeedsDeferredWrite_ = true;
        return;
    }

    uint32_t cFromCache, cFromX;
    if (curBatchLen >= stateLen_) {
        cFromCache = 0;
        cFromX = stateLen_;
    } else if (curBatchLen + cachedStateLen >= stateLen_) {
        cFromCache = stateLen_ - curBatchLen;
        cFromX = curBatchLen;
    } else {
        cFromCache = cachedStateLen;
        cFromX = curBatchLen;
    }

    if (cFromX == 0) return;

    // xLocal 末尾 cFromX 行：起点 = i + step - cFromX
    uint32_t xCacheStart = i + step - cFromX;
    uint32_t tailRowOff = stateLen_ - cFromX;
    uint64_t csOffset = (uint64_t)writeCIdx * cacheStride0_
                        + (uint64_t)tailRowOff * cacheStride1_ + dimStart;
    DataCopyExtParams wcp{static_cast<uint16_t>(cFromX),
                          static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                          0, cacheSkipBlocks * ALIGN_BYTES, 0};
    DataCopyPad(cacheStatesGM_[csOffset], xLocal[xCacheStart * dimSize], wcp);
}

// ============================================================================
// WriteDeferredCacheToStates — Plan A：与 in-pipeline 公式一致
//   cFromCache + cFromX 行写到 conv_states 尾部，源自旧 cache 末尾 + 当前 x 末尾。
//   对 firstBatch 非完成核（!firstBatchComplete_）延后到 SyncAll 之后执行。
//   正向加载语义：cached_state = conv_states[readCIdx][0:cachedStateLen]
//   padded_input[-cache_len:] 中来自 cached_state 的部分 = cached_state[cachedStateLen - cFromCache:]
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::WriteDeferredCacheToStates()
{
    if (!firstBatchNeedsDeferredWrite_ || firstBatchWriteCIdx_ == padSlotId_) {
        return;
    }

    uint32_t K = kernelWidth_;

    uint32_t dimStart, dimSize;
    if (dimIdx_ < dimRemainderCores_) {
        dimStart = dimIdx_ * dimBlockFactor_;
        dimSize  = dimBlockFactor_;
    } else {
        dimStart = dimRemainderCores_ * dimBlockFactor_ + (dimIdx_ - dimRemainderCores_) * dimBlockTailFactor_;
        dimSize  = dimBlockTailFactor_;
    }

    uint64_t batchStart  = (uint64_t)seqStartLocal_.GetValue(firstBatchIdx_);
    uint64_t batchEnd    = (uint64_t)seqStartLocal_.GetValue(firstBatchIdx_ + 1);
    uint32_t curBatchLen = (uint32_t)(batchEnd - batchStart);

    uint32_t rowBlocks = dimSize * sizeof(T) / ALIGN_BYTES;
    uint32_t cacheSkip = (cacheStride1_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t xSkip     = (xStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;

    int64_t writeCIdx = firstBatchWriteCIdx_;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    bool useZeroCache = false;
    if (hasNumComputedTokens_) {
        useZeroCache = (numComputedLocal_.GetValue(firstBatchIdx_) == 0);
    }

    // MTP 偏移与 cachedStateLen 计算：与 ProcessTokensNeedCache 保持一致
    // 注：useZeroCache 时 mtpOffset 视为 0；无 MTP 时 offset = stateLen-(K-1)
    uint32_t mtpOffsetD = 0;
    if (useZeroCache) {
        mtpOffsetD = 0;
    } else if (hasAcceptTokenNum_) {
        int32_t accepted = acceptTokenLocal_.GetValue(firstBatchIdx_);
        if (accepted > 0) mtpOffsetD = static_cast<uint32_t>(accepted - 1);
    } else {
        // 无 MTP：golden offset = stateLen-(K-1)
        mtpOffsetD = stateLen_ - (K - 1);
    }
    uint32_t cachedStateLenD;
    if (useZeroCache) {
        cachedStateLenD = K - 1;
    } else if (hasAcceptTokenNum_) {
        cachedStateLenD = mtpOffsetD + (K - 1);
        if (cachedStateLenD > stateLen_) cachedStateLenD = stateLen_;
    } else {
        cachedStateLenD = stateLen_;
    }

    uint32_t cFromCache, cFromX;
    if (curBatchLen >= stateLen_) {
        cFromCache = 0;
        cFromX = stateLen_;
    } else if (curBatchLen + cachedStateLenD >= stateLen_) {
        cFromCache = stateLen_ - curBatchLen;
        cFromX = curBatchLen;
    } else {
        cFromCache = cachedStateLenD;
        cFromX = curBatchLen;
    }
    uint32_t totalRows = cFromCache + cFromX;
    if (totalRows == 0) return;

    LocalTensor<T> tmpBuf = xQueue_.AllocTensor<T>();

    if (cFromCache > 0) {
        if (!useZeroCache) {
            int64_t readCIdx, dummyWrite;
            ResolveCacheIndices(firstBatchIdx_, readCIdx, dummyWrite);
            // 正向加载语义：cached_state = conv_states[readCIdx][0:cachedStateLenD]
            // padded_input[-cache_len:] 中来自 cached_state 的部分起始于 cachedStateLenD - cFromCache
            uint32_t srcRowOff = cachedStateLenD - cFromCache;
            uint64_t cacheSrcOffset = (uint64_t)readCIdx * cacheStride0_
                                      + (uint64_t)srcRowOff * cacheStride1_ + dimStart;
            DataCopyExtParams rcp{static_cast<uint16_t>(cFromCache),
                                  static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                                  cacheSkip * ALIGN_BYTES, 0, 0};
            DataCopyPad(tmpBuf, cacheStatesGM_[cacheSrcOffset], rcp, padParams);
        } else {
            Duplicate(tmpBuf, (T)0, cFromCache * dimSize);
        }
    }
    if (cFromX > 0) {
        // x 末尾 cFromX 行
        uint64_t xSrcOffset = (batchEnd - cFromX) * xStride_ + dimStart;
        DataCopyExtParams rcp{static_cast<uint16_t>(cFromX),
                              static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                              xSkip * ALIGN_BYTES, 0, 0};
        DataCopyPad(tmpBuf[cFromCache * dimSize], xGM_[xSrcOffset], rcp, padParams);
    }

    xQueue_.EnQue(tmpBuf);
    tmpBuf = xQueue_.DeQue<T>();

    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

    uint32_t tailRowOff = stateLen_ - totalRows;
    uint64_t csDstOffset = (uint64_t)writeCIdx * cacheStride0_
                           + (uint64_t)tailRowOff * cacheStride1_ + dimStart;
    DataCopyExtParams wcp{static_cast<uint16_t>(totalRows),
                          static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                          0, cacheSkip * ALIGN_BYTES, 0};
    DataCopyPad(cacheStatesGM_[csDstOffset], tmpBuf, wcp);

    xQueue_.FreeTensor(tmpBuf);
}

// ============================================================================
// FillApcBlockCaches — v2.0 §8 APC 专属：
//   仅 dimIdx_==0 的核执行（按 batch 维独立写入，避免重复）
//   对每个本核覆盖的 batch（firstBatchIdx_ .. lastBatchIdx_），按 blockSize 分段
//   将 padded_input 中对应区间的 token 写到 cache_indices[batch, blockId] 指向的 cache 行（尾对齐）
//   范围：blockIdxFirst[batch] .. (blockIdxLast[batch] - 1)
//   注意：lastBlk 由主计算路径（WriteCacheShort/LongBatch + WriteDeferredCacheToStates）
//         负责写入，包含旧 cache 前缀 + x 后缀的完整逻辑，此处必须跳过，
//         否则 FillApcBlockCaches 用纯 x 数据覆盖会导致 cache 前缀丢失。
//   每段写到对应 cache 行尾部（stateLen - chunkLen 偏移）
//
//   与 golden 对齐：boundary_idx = last_full_block_token_index - (n_block_to_fill-chunk-1)*B_size
//   数据来源为 padded_input = [cached_state] + [seq_x]，boundary_idx 为 padded_input 内索引。
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::FillApcBlockCaches()
{
    if (!apcEnabled_) return;
    if (blockSize_ == 0) return;

    uint32_t K    = kernelWidth_;
    uint32_t rows = K - 1;

    // 各 dim 核写各自的 dim 分片（与 WriteDeferredCacheToStates 一致）
    uint32_t dimStart, dimSize;
    if (dimIdx_ < dimRemainderCores_) {
        dimStart = dimIdx_ * dimBlockFactor_;
        dimSize  = dimBlockFactor_;
    } else {
        dimStart = dimRemainderCores_ * dimBlockFactor_ + (dimIdx_ - dimRemainderCores_) * dimBlockTailFactor_;
        dimSize  = dimBlockTailFactor_;
    }

    uint32_t rowBlocks = dimSize * sizeof(T) / ALIGN_BYTES;
    uint32_t cacheSkip = (cacheStride1_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t xSkip     = (xStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    for (uint32_t b = firstBatchIdx_; b <= lastBatchIdx_; b++) {
        if (b >= batchSize_) break;

        int32_t firstBlk = blockIdxFirstLocal_.GetValue(b);
        int32_t lastBlk  = blockIdxLastLocal_.GetValue(b);
        if (lastBlk <= firstBlk) continue;

        // 多 BS 核防重叠：每个 batch 只由第一个覆盖它的 BS 核写入 APC cache
        if (b == firstBatchIdx_ && bsIdx_ != 0) continue;

        uint32_t seqLen = static_cast<uint32_t>(
            seqStartLocal_.GetValue(b + 1) - seqStartLocal_.GetValue(b));

        // ---- 计算 offset (MTP) 与 cached_state 长度 ----
        // 当 num_computed_tokens[b]==0 时，golden 使用零初始化 cache + offset=0，
        // 此处必须与 ProcessTokensNeedCache 的 useZeroCache 逻辑保持一致
        bool useZeroCacheForBatch = false;
        if (hasNumComputedTokens_) {
            useZeroCacheForBatch = (numComputedLocal_.GetValue(b) == 0);
        }
        uint32_t offset = 0;
        if (!useZeroCacheForBatch) {
            if (hasAcceptTokenNum_) {
                int32_t accepted = acceptTokenLocal_.GetValue(b);
                if (accepted > 0) offset = static_cast<uint32_t>(accepted - 1);
            } else {
                offset = stateLen_ - (K - 1);
            }
        }
        uint32_t cachedStateLen = offset + (K - 1);

        // ---- 计算 last_full_block_token_index（与 golden 对齐）----
        int32_t numComputed = 0;
        if (hasNumComputedTokens_) {
            numComputed = numComputedLocal_.GetValue(b);
        }

        uint32_t seqCompletedOffsetToken = (numComputed >= 0) ?
            static_cast<uint32_t>(numComputed % static_cast<int32_t>(blockSize_)) : 0;
        uint32_t seqCompletedOffset = blockSize_ - seqCompletedOffsetToken;

        int32_t seqEndOffsetRaw = static_cast<int32_t>(seqLen) - static_cast<int32_t>(seqCompletedOffset);
        uint32_t seqEndOffset = static_cast<uint32_t>(
            (seqEndOffsetRaw % static_cast<int32_t>(blockSize_) + static_cast<int32_t>(blockSize_))
            % static_cast<int32_t>(blockSize_));

        int32_t lastFullBlockTokenIndexRaw = static_cast<int32_t>(seqLen) - static_cast<int32_t>(seqEndOffset);
        if (seqEndOffset == 0 && seqLen >= seqCompletedOffset) {
            lastFullBlockTokenIndexRaw -= static_cast<int32_t>(blockSize_);
        }

        int32_t nBlockToFill = lastBlk - firstBlk;

        // ---- 遍历 [firstBlk, lastBlk) 范围内的 block ----
        for (int32_t chunk = 0; chunk < nBlockToFill; chunk++) {
            int32_t blkId = firstBlk + chunk;
            int64_t cLine = GetApcCacheLine(b, static_cast<uint32_t>(blkId));
            if (cLine == padSlotId_) continue;

            int32_t boundaryIdxRaw = lastFullBlockTokenIndexRaw -
                (nBlockToFill - chunk - 1) * static_cast<int32_t>(blockSize_);
            if (boundaryIdxRaw <= 0) continue;
            uint32_t boundaryIdx = static_cast<uint32_t>(boundaryIdxRaw);

            // boundaryIdx 索引的是 post-offset padded_input: [cache(K-1), x(seqLen)]
            // 因此 cache/x 边界在 K-1，而非 cachedStateLen(=offset+K-1)
            uint64_t srcOffset;
            bool fromCache;
            if (boundaryIdx < (K - 1)) {
                fromCache = true;
                srcOffset = offset + boundaryIdx;   // 映射回 pre-offset cache 行号
            } else {
                fromCache = false;
                srcOffset = boundaryIdx - (K - 1);  // 直接为 x 行索引
            }

            // chunkLen = K-1，防越界截断
            uint32_t chunkLen = rows;
            uint32_t maxAvail = (K - 1) + seqLen - boundaryIdx;
            if (chunkLen > maxAvail) chunkLen = maxAvail;
            if (chunkLen == 0) continue;

            LocalTensor<T> tmpBuf = xQueue_.AllocTensor<T>();

            if (fromCache) {
                // boundaryIdx < K-1：数据从 post-offset padded_input[boundaryIdx] 开始，
                // 可能跨越 cache/x 边界（cache 占 [0, K-2]，x 占 [K-1, ...]）
                uint32_t cacheRows = (K - 1) - boundaryIdx;
                if (cacheRows > chunkLen) cacheRows = chunkLen;
                uint32_t xRows = chunkLen - cacheRows;

                if (useZeroCacheForBatch) {
                    // num_computed_tokens==0：cached_state 应为全零，与 golden 零初始化路径对齐
                    if (xRows == 0) {
                        // 纯 cache 行，全零
                        Duplicate(tmpBuf, (T)0, chunkLen * dimSize);
                    } else {
                        // cacheRows 行零 + xRows 行 x
                        // 先将 cacheRows 行置零（Vector），再从 x 读取 xRows 行（MTE2）
                        Duplicate(tmpBuf, (T)0, cacheRows * dimSize);
                        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
                        uint64_t batchStart = (uint64_t)seqStartLocal_.GetValue(b);
                        uint64_t xSrcOff = batchStart * xStride_ + dimStart;
                        DataCopyExtParams xrcp{static_cast<uint16_t>(xRows),
                                               static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                                               xSkip * ALIGN_BYTES, 0, 0};
                        DataCopyPad(tmpBuf[cacheRows * dimSize], xGM_[xSrcOff], xrcp, padParams);
                    }
                } else {
                    // 从 cached_state (cacheStatesGM_) 读取 cache 部分
                    int32_t initIdx = initialStateIdxLocal_.GetValue(b);
                    int64_t readCIdx = GetApcCacheLine(b, static_cast<uint32_t>(initIdx));
                    if (cacheRows > 0) {
                        uint64_t cacheSrcOffset = (uint64_t)readCIdx * cacheStride0_
                                                  + (uint64_t)srcOffset * cacheStride1_ + dimStart;
                        DataCopyExtParams crcp{static_cast<uint16_t>(cacheRows),
                                               static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                                               cacheSkip * ALIGN_BYTES, 0, 0};
                        DataCopyPad(tmpBuf, cacheStatesGM_[cacheSrcOffset], crcp, padParams);
                    }
                    // 从 seq_x (xGM_) 读取 x 部分
                    if (xRows > 0) {
                        uint64_t batchStart = (uint64_t)seqStartLocal_.GetValue(b);
                        uint64_t xSrcOff = batchStart * xStride_ + dimStart;
                        DataCopyExtParams xrcp{static_cast<uint16_t>(xRows),
                                               static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                                               xSkip * ALIGN_BYTES, 0, 0};
                        DataCopyPad(tmpBuf[cacheRows * dimSize], xGM_[xSrcOff], xrcp, padParams);
                    }
                }
            } else {
                // 从 seq_x (xGM_) 读取
                uint64_t batchStart = (uint64_t)seqStartLocal_.GetValue(b);
                uint64_t xSrcOffset = (batchStart + srcOffset) * xStride_ + dimStart;
                DataCopyExtParams rcp{static_cast<uint16_t>(chunkLen),
                                      static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                                      xSkip * ALIGN_BYTES, 0, 0};
                DataCopyPad(tmpBuf, xGM_[xSrcOffset], rcp, padParams);
            }

            xQueue_.EnQue(tmpBuf);
            tmpBuf = xQueue_.DeQue<T>();

            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

            uint32_t tailRowOff = (stateLen_ > chunkLen) ? (stateLen_ - chunkLen) : 0;
            uint64_t csDst = (uint64_t)cLine * cacheStride0_
                             + (uint64_t)tailRowOff * cacheStride1_ + dimStart;
            DataCopyExtParams wcp{static_cast<uint16_t>(chunkLen),
                                  static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                                  0, cacheSkip * ALIGN_BYTES, 0};
            DataCopyPad(cacheStatesGM_[csDst], tmpBuf, wcp);

            xQueue_.FreeTensor(tmpBuf);
        }
    }
}

// ============================================================================
// ProcessMainCompute — v2.0：跟踪 firstBatchWriteCIdx_（APC 用 writeCIdx）与 lastBatchIdx_
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::ProcessMainCompute(uint64_t bsStart, uint64_t dimStart,
    uint32_t loopNumBS, uint32_t ubFactorBS, uint32_t ubTailFactorBS,
    uint32_t loopNumDim, uint32_t ubFactorDim, uint32_t ubTailFactorDim)
{
    uint32_t K    = kernelWidth_;
    uint16_t step = (ubFactorBS > K - 1) ? (ubFactorBS - (K - 1)) : 1;

    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    uint32_t curBatchIdx = FindBatchIdx(bsStart);

    bool isBsFirstCore = (bsIdx_ == 0);
    uint32_t iStart = isBsFirstCore ? 0 : (K - 1);
    firstBatchNeedsDeferredWrite_ = false;

    uint64_t curBsOff = 0;

    for (uint32_t bsLoop = 0; bsLoop < loopNumBS; bsLoop++) {
        uint32_t curBS = (bsLoop == loopNumBS - 1) ? ubTailFactorBS : ubFactorBS;
        uint64_t curBsStart = bsStart + curBsOff;
        uint64_t actualStart = curBsStart + iStart;

        while (curBatchIdx + 1 < batchSize_ &&
               actualStart >= (uint64_t)seqStartLocal_.GetValue(curBatchIdx + 1)) {
            curBatchIdx++;
        }

        uint64_t batchStart = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
        uint32_t curSequenceIdx = (uint32_t)(actualStart - batchStart);

        if (bsLoop == 0) {
            firstBatchIdx_ = curBatchIdx;
            int64_t readC, writeC;
            ResolveCacheIndices(curBatchIdx, readC, writeC);
            firstBatchWriteCIdx_ = writeC;
            firstBatchComplete_  = isBsFirstCore || (curSequenceIdx == 0);
        }

        uint32_t dimOff = (uint32_t)dimStart;
        uint32_t batchIdxForDim = curBatchIdx;
        uint32_t seqIdxForDim   = curSequenceIdx;
        for (uint32_t dimLoop = 0; dimLoop < loopNumDim; dimLoop++) {
            uint32_t curDim = (dimLoop == loopNumDim - 1) ? ubTailFactorDim : ubFactorDim;
            batchIdxForDim = curBatchIdx;
            seqIdxForDim   = curSequenceIdx;
            ProcessUBBlock((uint32_t)curBsStart, curBS, dimOff, curDim, iStart, batchIdxForDim, seqIdxForDim);
            dimOff += curDim;
        }
        curBatchIdx = batchIdxForDim;
        lastBatchIdx_ = curBatchIdx;

        curBsOff += step;
        iStart = K - 1;
    }
}

// ============================================================================
// Process — v2.0：SyncAll → WriteDeferred → FillApc
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::Process()
{
    uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= realCoreNum_) {
        return;
    }

    LoadMetaData();

    uint32_t K = kernelWidth_;
    uint64_t bsStart;
    if (bsIdx_ < bsRemainderCores_) {
        uint64_t effectiveStep = bsBlockFactor_ - (K - 1);
        bsStart = (uint64_t)bsIdx_ * effectiveStep;
    } else {
        uint64_t effectiveStepMain = bsBlockFactor_ - (K - 1);
        uint64_t effectiveStepTail = bsBlockTailFactor_ - (K - 1);
        bsStart = (uint64_t)bsRemainderCores_ * effectiveStepMain +
                  (uint64_t)(bsIdx_ - bsRemainderCores_) * effectiveStepTail;
    }

    uint64_t dimStart;
    if (dimIdx_ < dimRemainderCores_) {
        dimStart = (uint64_t)dimIdx_ * dimBlockFactor_;
    } else {
        dimStart = (uint64_t)dimRemainderCores_ * dimBlockFactor_ +
                   (uint64_t)(dimIdx_ - dimRemainderCores_) * dimBlockTailFactor_;
    }

    bool isBsTailCore  = (bsIdx_  >= bsRemainderCores_);
    bool isDimTailCore = (dimIdx_ >= dimRemainderCores_);

    uint32_t loopBS  = isBsTailCore  ? tailBlockloopNumBS_      : loopNumBS_;
    uint32_t factBS  = isBsTailCore  ? tailBlockubFactorBS_     : ubFactorBS_;
    uint32_t tailBS  = isBsTailCore  ? tailBlockubTailFactorBS_ : ubTailFactorBS_;

    uint32_t loopDim = isDimTailCore ? tailBlockloopNumDim_      : loopNumDim_;
    uint32_t factDim = isDimTailCore ? tailBlockubFactorDim_     : ubFactorDim_;
    uint32_t tailDim = isDimTailCore ? tailBlockubTailFactorDim_ : ubTailFactorDim_;

    ProcessMainCompute(bsStart, dimStart, loopBS, factBS, tailBS, loopDim, factDim, tailDim);

    SyncAll();

    WriteDeferredCacheToStates();
    FillApcBlockCaches();
}

} // namespace FusedCausalConv1dCutBSHNs

#endif // FUSED_CAUSAL_CONV1D_CUT_BSH_H
