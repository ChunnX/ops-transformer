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

// ============================================================================
// 常量
// ============================================================================
constexpr int32_t BUFFER_NUM   = 2;              // 双缓冲
constexpr uint32_t ALIGN_BYTES = 32;             // DataCopy 32 字节对齐单元
// ============================================================================
// FusedCausalConv1dCutBSH Kernel 类
//
// 流程概述：
//   Init  → 解析 TilingData，绑定 GM，初始化 UB 队列/缓冲
//   Process →
//     1. LoadMetaData：一次性加载 seqStartIndex、cacheIndices、hasInitialState
//     2. 加载核内切分参数
//     3. ProcessMainCompute：双层循环（BS × Dim），每次调用 ProcessUBBlock
//     4. SyncAll（全核同步）
//     5. WriteDeferredCacheToStates：从 GM 原始数据重建 cache 并写回 cacheStates
// ============================================================================
template <typename T>
class FusedCausalConv1dCutBSH {
public:
    __aicore__ inline FusedCausalConv1dCutBSH() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR initialStateMode, GM_ADDR y, GM_ADDR workspace,
                                const FusedCausalConv1dCutBSHTilingData* tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseBSHTilingData(const FusedCausalConv1dCutBSHTilingData* tiling);
    __aicore__ inline void ProcessMainCompute(uint64_t bsStart, uint64_t dimStart,
        uint32_t loopNumBS, uint32_t ubFactorBS, uint32_t ubTailFactorBS,
        uint32_t loopNumDim, uint32_t ubFactorDim, uint32_t ubTailFactorDim);
    __aicore__ inline void ProcessUBBlock(uint32_t bsStart, uint32_t bsSize, uint32_t dimStart, uint32_t dimSize,
        uint32_t iStart, uint32_t& curBatchIdx, uint32_t& curSequenceIdx);
    __aicore__ inline uint16_t ProcessTokensNeedCache(LocalTensor<T>& xLocal, LocalTensor<T>& weightLocal,
        uint32_t i, uint32_t N, uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart,
        uint32_t cacheSkipBlocks, uint32_t ySkipBlocks, uint64_t batchStart, uint32_t curBatchLen,
        uint32_t curSequenceIdx, int32_t hasInitState, int64_t cIdx, uint32_t curBatchIdx, bool& reachBatchEnd);
    __aicore__ inline uint16_t ProcessTokensNoCache(LocalTensor<T>& xLocal, LocalTensor<T>& weightLocal,
        uint32_t i, uint32_t N, uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart,
        uint32_t cacheSkipBlocks, uint32_t ySkipBlocks, uint64_t batchStart, uint32_t curBatchLen,
        uint32_t curSequenceIdx, int64_t cIdx, uint32_t curBatchIdx, bool& reachBatchEnd);
    __aicore__ inline void WriteCacheShortBatch(LocalTensor<T>& cacheLocal, LocalTensor<T>& xLocal,
        uint32_t i, uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart, uint32_t cacheSkipBlocks,
        uint32_t curBatchLen, int64_t cIdx, uint32_t curBatchIdx);
    __aicore__ inline void WriteCacheLongBatch(LocalTensor<T>& xLocal, uint32_t i, uint16_t step,
        uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart, uint32_t cacheSkipBlocks,
        int64_t cIdx, uint32_t curBatchIdx);

    // 全核同步后，从 GM 原始数据重建 cache 并写回 cacheStates（延迟写回）
    __aicore__ inline void WriteDeferredCacheToStates();

    // 通过二分查找确定 globalSeqIdx 所属的 batch（0-indexed）
    __aicore__ inline uint32_t FindBatchIdx(uint64_t globalSeqIdx);

    // 一次性加载元数据到 UB
    __aicore__ inline void LoadMetaData();

    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    // 辅助：向上对齐到 align 字节
    static __aicore__ inline uint32_t AlignUp(uint32_t n, uint32_t align) {
        return (n + align - 1) / align * align;
    }

private:
    // -------------------------------------------------------------------------
    // Tiling 参数
    // -------------------------------------------------------------------------
    uint32_t dim_;
    uint32_t kernelWidth_;       // K
    uint32_t batchSize_;

    // 核内切分参数（主核）
    uint32_t loopNumBS_;
    uint32_t loopNumDim_;
    uint32_t ubFactorBS_;
    uint32_t ubTailFactorBS_;
    uint32_t ubFactorDim_;
    uint32_t ubTailFactorDim_;

    // 核内切分参数（尾核）
    uint32_t tailBlockloopNumBS_;
    uint32_t tailBlockloopNumDim_;
    uint32_t tailBlockubFactorBS_;
    uint32_t tailBlockubTailFactorBS_;
    uint32_t tailBlockubFactorDim_;
    uint32_t tailBlockubTailFactorDim_;

    // dim 方向核间切分
    uint32_t dimCoreNum_;            // dim 方向总核数
    uint32_t dimRemainderCores_;     // dim 方向前多少个核是主核
    uint32_t dimBlockFactor_;        // dim 方向主核处理的大小
    uint32_t dimBlockTailFactor_;    // dim 方向尾核处理的大小

    // BS 方向核间切分
    uint32_t bsRemainderCores_;      // BS 方向前多少个核是主核
    uint32_t bsBlockFactor_;         // BS 方向主核处理的长度（含 overlap）
    uint32_t bsBlockTailFactor_;     // BS 方向尾核处理的长度

    uint32_t realCoreNum_;

    // stride（跨 sequence 的步长）
    uint32_t xStride_;           // x 的行 stride（>= dim_）
    uint32_t cacheStride0_;      // cacheStates 的 batch 维 stride（元素数）
    uint32_t cacheStride1_;      // cacheStates 的 sequence 维 stride（元素数）
    uint32_t residualConnection_; // 是否加残差：0-不需要，1-需要

    int64_t padSlotId_;          // 无效 batch 标记值

    // 运行时辅助：当前核的二维索引
    uint32_t bsIdx_;             // BS 方向的核索引
    uint32_t dimIdx_;            // Dim 方向的核索引
    bool     initialStateModeNull_;  // initialStateMode 是否为空指针（None），为空时按 hasInitState=2 处理

    // -------------------------------------------------------------------------
    // Pipeline & Global Tensors
    // -------------------------------------------------------------------------
    TPipe pipe_;

    GlobalTensor<T>       xGM_;
    GlobalTensor<T>       weightGM_;
    GlobalTensor<T>       cacheStatesGM_;
    GlobalTensor<int32_t> cacheIndicesGM_;
    GlobalTensor<int32_t> seqStartIndexGM_;
    GlobalTensor<int32_t> hasInitialStateGM_;   // 0: 用0填充cache计算, 1: 使用cache, 2: 前K-1个置0
    GlobalTensor<T>       yGM_;


    // -------------------------------------------------------------------------
    // UB 队列 & 缓冲
    //
    //   weightInQueue  : K × maxUbDim × sizeof(T)，BUF_NUM=1
    //   cacheQueue     : (K-1) × maxUbDim × sizeof(T)，BUF_NUM=1（TQueBind 可 VECIN/VECOUT）
    //   startLocInQueue: (batch+1) × sizeof(int32_t)，BUF_NUM=1
    //   indicesInQueue : batch × sizeof(int32_t)，BUF_NUM=1
    //   hasInitInQueue : batch × sizeof(int32_t)，BUF_NUM=1
    //   xQueue         : maxUbBS × maxUbDim × sizeof(T)，BUF_NUM=2（双缓冲，y 复用）
    // -------------------------------------------------------------------------
    TQue<QuePosition::VECIN, 1>        weightInQueue_;
    TQue<QuePosition::VECIN, 1>        cacheQueue_;
    TQue<QuePosition::VECIN, 1>        startLocInQueue_;
    TQue<QuePosition::VECIN, 1>        indicesInQueue_;
    TQue<QuePosition::VECIN, 1>        hasInitInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> xQueue_;  // x 搬入(VECIN)，y 搬出(VECOUT)

    // 从队列 DeQue 后持久持有的 meta 数据（在 Process 期间一直有效）
    LocalTensor<int32_t> seqStartLocal_;
    LocalTensor<int32_t> cacheIdxLocal_;
    LocalTensor<int32_t> hasInitLocal_;

    // -------------------------------------------------------------------------
    // 第一个 batch 跟踪（用于延迟写回 cache）
    // 只有当第一个 batch 不完整（首个 token 在前一个核）时才需要延迟写回
    // -------------------------------------------------------------------------
    uint32_t firstBatchIdx_;        // 当前核的第一个 batch 索引
    int64_t  firstBatchCIdx_;       // 第一个 batch 对应的 cacheIndices 值
    bool     firstBatchComplete_;   // 第一个 batch 是否完整（首个 token 在当前核）
    bool     firstBatchNeedsDeferredWrite_;// 是否需要在 SyncAll 后延迟写回 cache
};

// ============================================================================
// ParseBSHTilingData：解析 TilingData 到成员变量
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::ParseBSHTilingData(const FusedCausalConv1dCutBSHTilingData* tiling)
{
    dim_                       = tiling->dim;
    kernelWidth_               = tiling->kernelWidth;
    batchSize_                 = tiling->batch;

    // 核内切分参数（主核）
    loopNumBS_                 = tiling->loopNumBS;
    loopNumDim_                = tiling->loopNumDim;
    ubFactorBS_                = tiling->ubFactorBS;
    ubTailFactorBS_            = tiling->ubTailFactorBS;
    ubFactorDim_               = tiling->ubFactorDim;
    ubTailFactorDim_           = tiling->ubTailFactorDim;

    // 核内切分参数（尾核）
    tailBlockloopNumBS_        = tiling->tailBlockloopNumBS;
    tailBlockloopNumDim_       = tiling->tailBlockloopNumDim;
    tailBlockubFactorBS_       = tiling->tailBlockubFactorBS;
    tailBlockubTailFactorBS_   = tiling->tailBlockubTailFactorBS;
    tailBlockubFactorDim_      = tiling->tailBlockubFactorDim;
    tailBlockubTailFactorDim_  = tiling->tailBlockubTailFactorDim;

    // dim 方向核间切分
    dimCoreNum_                = tiling->dimCoreNum;
    dimRemainderCores_         = tiling->dimRemainderCores;
    dimBlockFactor_            = tiling->dimBlockFactor;
    dimBlockTailFactor_        = tiling->dimBlockTailFactor;

    // BS 方向核间切分
    bsRemainderCores_          = tiling->bsRemainderCores;
    bsBlockFactor_             = tiling->bsBlockFactor;
    bsBlockTailFactor_         = tiling->bsBlockTailFactor;

    realCoreNum_               = tiling->realCoreNum;

    // stride（x 和 cacheStates 非连续存储）
    xStride_                   = tiling->xStride;
    cacheStride0_              = tiling->cacheStride0;
    cacheStride1_              = tiling->cacheStride1;
    residualConnection_        = tiling->residualConnection;

    // 有效 batch 范围
    padSlotId_                 = tiling->padSlotId;
}

// ============================================================================
// Init
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates,
    GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode, GM_ADDR y, GM_ADDR workspace,
    const FusedCausalConv1dCutBSHTilingData* tiling)
{
    ParseBSHTilingData(tiling);

    // 计算当前核的二维索引
    uint32_t blockIdx = GetBlockIdx();
    bsIdx_  = blockIdx / dimCoreNum_;                    // BS 方向的核索引
    dimIdx_ = blockIdx - bsIdx_ * dimCoreNum_;           // Dim 方向的核索引

    initialStateModeNull_ = (initialStateMode == nullptr);  // 检查是否为空指针

    // 根据当前核的实际类型计算 UB 因子（用于 buffer 分配，仅 Init 内使用）
    bool isBsTailCore  = (bsIdx_ >= bsRemainderCores_);
    bool isDimTailCore = (dimIdx_ >= dimRemainderCores_);
    uint32_t maxUbBS  = isBsTailCore  ? tailBlockubFactorBS_  : ubFactorBS_;
    uint32_t maxUbDim = isDimTailCore ? tailBlockubFactorDim_ : ubFactorDim_;

    // --- 绑定 GM ---
    // xGM_ 从 x 起始地址开始，bsStart 就是全局 token 位置
    xGM_.SetGlobalBuffer((__gm__ T*)x, (uint64_t)tiling->cuSeqLen * xStride_);
    weightGM_.SetGlobalBuffer((__gm__ T*)weight, kernelWidth_ * dim_);
    cacheStatesGM_.SetGlobalBuffer((__gm__ T*)convStates);
    cacheIndicesGM_.SetGlobalBuffer((__gm__ int32_t*)cacheIndices, batchSize_);
    seqStartIndexGM_.SetGlobalBuffer((__gm__ int32_t*)queryStartLoc, batchSize_ + 1);
    if (!initialStateModeNull_) {
        hasInitialStateGM_.SetGlobalBuffer((__gm__ int32_t*)initialStateMode, batchSize_);
    }
    yGM_.SetGlobalBuffer((__gm__ T*)y, (uint64_t)tiling->cuSeqLen * dim_);

    // --- 计算 buffer 字节大小并对齐到 32 字节 ---
    uint32_t K = kernelWidth_;

    uint32_t weightBufBytes  = AlignUp(K * maxUbDim * sizeof(T),       ALIGN_BYTES);
    uint32_t cacheBufBytes   = AlignUp((K - 1) * maxUbDim * sizeof(T), ALIGN_BYTES);
    uint32_t startLocBytes   = AlignUp((batchSize_ + 1) * sizeof(int32_t), ALIGN_BYTES);
    uint32_t indicesBytes    = AlignUp(batchSize_ * sizeof(int32_t),       ALIGN_BYTES);
    uint32_t hasInitBytes    = AlignUp(batchSize_ * sizeof(int32_t),       ALIGN_BYTES);
    uint32_t xBufBytes       = AlignUp(maxUbBS * maxUbDim * sizeof(T),  ALIGN_BYTES);

    // --- 初始化 UB 队列 / 缓冲 ---
    pipe_.InitBuffer(weightInQueue_,   1,           weightBufBytes);
    pipe_.InitBuffer(cacheQueue_,      1,           cacheBufBytes);
    pipe_.InitBuffer(startLocInQueue_, 1,           startLocBytes);
    pipe_.InitBuffer(indicesInQueue_,  1,           indicesBytes);
    pipe_.InitBuffer(hasInitInQueue_,  1,           hasInitBytes);
    pipe_.InitBuffer(xQueue_,          BUFFER_NUM,  xBufBytes);
}

// ============================================================================
// LoadMetaData：一次性加载 seqStartIndex、cacheIndices、hasInitialState
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::LoadMetaData()
{
    // seqStartIndex (queryStartLoc)
    {
        LocalTensor<int32_t> tmp = startLocInQueue_.AllocTensor<int32_t>();
        DataCopyExtParams cpParams{1, static_cast<uint16_t>((batchSize_ + 1) * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(tmp, seqStartIndexGM_[0], cpParams, padParams);
        startLocInQueue_.EnQue(tmp);
        seqStartLocal_ = startLocInQueue_.DeQue<int32_t>();
    }
    // cacheIndices
    {
        LocalTensor<int32_t> tmp = indicesInQueue_.AllocTensor<int32_t>();
        DataCopyExtParams cpParams{1, static_cast<uint16_t>(batchSize_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(tmp, cacheIndicesGM_[0], cpParams, padParams);
        indicesInQueue_.EnQue(tmp);
        cacheIdxLocal_ = indicesInQueue_.DeQue<int32_t>();
    }
    // hasInitialState (initialStateMode)
    // 如果 initialStateMode 为空（None），则填充 2（按 hasInitState=2 的逻辑处理）
    {
        LocalTensor<int32_t> tmp = hasInitInQueue_.AllocTensor<int32_t>();
        if (initialStateModeNull_) {
            // initialStateMode 为 None，填充 2
            Duplicate(tmp, (int32_t)2, batchSize_);
        } else {
            // 从 GM 加载
            DataCopyExtParams cpParams{1, static_cast<uint16_t>(batchSize_ * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(tmp, hasInitialStateGM_[0], cpParams, padParams);
        }
        hasInitInQueue_.EnQue(tmp);
        hasInitLocal_ = hasInitInQueue_.DeQue<int32_t>();
    }
}

// ============================================================================
// FindBatchIdx：二分查找 globalSeqIdx 所在 batch
// 保证：seqStartLocal_[result] <= globalSeqIdx < seqStartLocal_[result+1]
// 搜索范围为 [0, batchSize_)
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
// ProcessUBBlock - 处理单个 UB 块
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::ProcessUBBlock(uint32_t bsStart, uint32_t bsSize,
    uint32_t dimStart, uint32_t dimSize, uint32_t iStart, uint32_t& curBatchIdx, uint32_t& curSequenceIdx)
{
    uint32_t K = kernelWidth_;
    uint32_t N = bsSize;

    // 计算各种 stride 参数
    uint32_t dimBlocks  = dimSize * sizeof(T) / ALIGN_BYTES;
    uint32_t weightSkipBlocks = (dim_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t ySkipBlocks = weightSkipBlocks;
    uint32_t xSkipBlocks = (xStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t cacheSkipBlocks = (cacheStride1_ - dimSize) * sizeof(T) / ALIGN_BYTES;

    // 计算当前 batch 信息
    uint64_t batchStart  = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
    uint64_t batchEnd    = (uint64_t)seqStartLocal_.GetValue(curBatchIdx + 1);
    uint32_t curBatchLen = (uint32_t)(batchEnd - batchStart);

    // 加载 weight
    LocalTensor<T> weightLocal = weightInQueue_.AllocTensor<T>();
    {
        DataCopyExtParams wcp{static_cast<uint16_t>(K), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              weightSkipBlocks * ALIGN_BYTES, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(weightLocal, weightGM_[dimStart], wcp, padParams);
    }
    weightInQueue_.EnQue(weightLocal);
    weightLocal = weightInQueue_.DeQue<T>();

    // 加载 x
    LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
    {
        DataCopyExtParams xcp{static_cast<uint16_t>(bsSize), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              xSkipBlocks * ALIGN_BYTES, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGM_[bsStart * xStride_ + dimStart], xcp, padParams);
    }
    xQueue_.EnQue(xLocal);
    xLocal = xQueue_.DeQue<T>();

    // 主循环：遍历 UB 块中的 token
    uint32_t i = iStart;
    while (i < N) {
        int64_t cIdx = cacheIdxLocal_.GetValue(curBatchIdx);

        // 跳过 padSlotId 标记的无效 batch
        if (cIdx == padSlotId_) {
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

        int32_t hasInitState = hasInitLocal_.GetValue(curBatchIdx);

        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        uint16_t step = 0;
        bool reachBatchEnd = false;

        if (curSequenceIdx < K - 1) {
            step = ProcessTokensNeedCache(
                xLocal, weightLocal, i, N, dimSize, dimBlocks,
                dimStart, cacheSkipBlocks, ySkipBlocks,
                batchStart, curBatchLen, curSequenceIdx,
                hasInitState, cIdx, curBatchIdx, reachBatchEnd);
        } else {
            step = ProcessTokensNoCache(
                xLocal, weightLocal, i, N, dimSize, dimBlocks,
                dimStart, cacheSkipBlocks, ySkipBlocks,
                batchStart, curBatchLen, curSequenceIdx,
                cIdx, curBatchIdx, reachBatchEnd);
        }
        // 更新索引
        i += step;
        curSequenceIdx += step;

        // 如果到达 batch 末尾，更新 batch 索引
        if (reachBatchEnd && curBatchIdx + 1 < batchSize_) {
            curBatchIdx++;
            batchStart   = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
            batchEnd     = (uint64_t)seqStartLocal_.GetValue(curBatchIdx + 1);
            curBatchLen  = (uint32_t)(batchEnd - batchStart);
            curSequenceIdx = 0;
        }
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    // 释放 buffer
    weightInQueue_.FreeTensor(weightLocal);
    xQueue_.FreeTensor(xLocal);
}

// ============================================================================
// ProcessTokensNeedCache - 处理需要 cache state 的 token（curSequenceIdx < K-1）
// ============================================================================
template <typename T>
__aicore__ inline uint16_t FusedCausalConv1dCutBSH<T>::ProcessTokensNeedCache(LocalTensor<T>& xLocal,
    LocalTensor<T>& weightLocal, uint32_t i, uint32_t N, uint32_t dimSize, uint32_t dimBlocks,
    uint32_t dimStart, uint32_t cacheSkipBlocks, uint32_t ySkipBlocks, uint64_t batchStart,
    uint32_t curBatchLen, uint32_t curSequenceIdx, int32_t hasInitState, int64_t cIdx,
    uint32_t curBatchIdx, bool& reachBatchEnd)
{
    uint32_t K = kernelWidth_;

    // 计算 step
    uint16_t step = K - 1 - curSequenceIdx;
    if (step > N - i) step = N - i;
    if (step > curBatchLen - curSequenceIdx) step = curBatchLen - curSequenceIdx;
    reachBatchEnd = (step == curBatchLen - curSequenceIdx);

    // 加载或初始化 cache state
    LocalTensor<T> cacheLocal = cacheQueue_.AllocTensor<T>();
    if (hasInitState == 1) {
        uint64_t cacheGmOffset = (uint64_t)cIdx * cacheStride0_ + dimStart;
        DataCopyExtParams ccp{static_cast<uint16_t>(K - 1), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              cacheSkipBlocks * ALIGN_BYTES, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(cacheLocal, cacheStatesGM_[cacheGmOffset], ccp, padParams);
    } else {
        Duplicate(cacheLocal, (T)0, (K - 1) * dimSize);
    }
    cacheQueue_.EnQue(cacheLocal);
    cacheLocal = cacheQueue_.DeQue<T>();

    // 等待 cacheLocal 数据就绪
    // hasInitState==1: MTE2 搬运; 否则: Duplicate (PIPE_V)

    // 如果 batch 长度 < K 且到达 batch 末尾，回写 cache
    if (curBatchLen < K && reachBatchEnd) {
        WriteCacheShortBatch(cacheLocal, xLocal, i - curSequenceIdx, dimSize, dimBlocks, dimStart,
                             cacheSkipBlocks, curBatchLen, cIdx, curBatchIdx);
    }

    // 写回 y 的公共参数
    uint64_t yGmOffset = (batchStart + curSequenceIdx) * dim_ + dimStart;
    DataCopyExtParams ycp{step, static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                          0, ySkipBlocks * ALIGN_BYTES, 0};
    // 卷积计算（hasInitState == 2 时跳过）
    if (hasInitState != 2) {
        for (uint32_t j = 0; j < step; j++) {
            uint32_t seqPos = curSequenceIdx + j;
            uint32_t stateSLen = K - 1 - seqPos;
            uint32_t xSLen = seqPos + 1;
            LocalTensor<T> xSlice = xLocal[(i - curSequenceIdx) * dimSize];
            LocalTensor<T> stateSlice = cacheLocal[seqPos * dimSize];
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            Conv1dNeedState(xSlice, weightLocal, stateSlice, stateSlice, stateSLen, xSLen, dimSize, residualConnection_);
            // 每次 VF 计算后同步，确保写入完成
        }
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(yGM_[yGmOffset], cacheLocal[curSequenceIdx * dimSize], ycp);
    } else {
        // hasInitState == 2，跳过卷积
        if (residualConnection_ == 1) {
            // 需要残差连接，直接把 x 搬到 yGM（y = x）
            DataCopyPad(yGM_[yGmOffset], xLocal[i * dimSize], ycp);
        } else {
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            // residualConnection_ == 0，需要把 y 置 0，cacheLocal 已填充 0，直接搬出
            DataCopyPad(yGM_[yGmOffset], cacheLocal[curSequenceIdx * dimSize], ycp);
        }
    }
    cacheQueue_.FreeTensor(cacheLocal);
    return step;
}

// ============================================================================
// ProcessTokensNoCache - 处理不需要 cache 的 token（curSequenceIdx >= K-1）
// ============================================================================
template <typename T>
__aicore__ inline uint16_t FusedCausalConv1dCutBSH<T>::ProcessTokensNoCache(LocalTensor<T>& xLocal,
    LocalTensor<T>& weightLocal, uint32_t i, uint32_t N, uint32_t dimSize, uint32_t dimBlocks,
    uint32_t dimStart, uint32_t cacheSkipBlocks, uint32_t ySkipBlocks, uint64_t batchStart,
    uint32_t curBatchLen, uint32_t curSequenceIdx, int64_t cIdx, uint32_t curBatchIdx, bool& reachBatchEnd)
{
    uint32_t K = kernelWidth_;

    // 计算 step
    uint32_t remainInBatch = curBatchLen - curSequenceIdx;
    uint16_t step = (N - i < remainInBatch) ? (N - i) : remainInBatch;
    reachBatchEnd = (step == remainInBatch);

    // 如果到达 batch 末尾，回写 cache
    if (reachBatchEnd) {
        WriteCacheLongBatch(xLocal, i, step, dimSize, dimBlocks, dimStart,
                            cacheSkipBlocks, cIdx, curBatchIdx);
    }

    // 卷积计算
    uint32_t xStartIdx = i - (K - 1);
    LocalTensor<T> xSlice = xLocal[xStartIdx * dimSize];
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    Conv1dNoNeedState(xSlice, weightLocal, xSlice, step, dimSize, residualConnection_);

    // 写回 y
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    uint64_t yGmOffset = (batchStart + curSequenceIdx) * dim_ + dimStart;
    DataCopyExtParams ycp{step, static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                          0, ySkipBlocks * ALIGN_BYTES, 0};
    DataCopyPad(yGM_[yGmOffset], xLocal[(i - K + 1) * dimSize], ycp);

    return step;
}

// ============================================================================
// WriteCacheShortBatch - 回写 cache（短 batch，长度 < K）
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::WriteCacheShortBatch(LocalTensor<T>& cacheLocal,
    LocalTensor<T>& xLocal, uint32_t i, uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart,
    uint32_t cacheSkipBlocks, uint32_t curBatchLen, int64_t cIdx, uint32_t curBatchIdx)
{
    uint32_t K = kernelWidth_;
    uint32_t cacheRowsToKeep = (K - 1 > curBatchLen) ? (K - 1 - curBatchLen) : 0;

    bool needDeferredWrite = (curBatchIdx == firstBatchIdx_ && !firstBatchComplete_);

    // 等待 cacheLocal 数据就绪（可能由 Duplicate 填充）
    if (needDeferredWrite) {
        // 延迟写回：SyncAll 后再从 GM 原始数据重建 cache
        firstBatchNeedsDeferredWrite_ = true;
        return;
    }

    uint64_t csOffset = (uint64_t)cIdx * cacheStride0_ + dimStart;
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    if (cacheRowsToKeep > 0) {
        DataCopyExtParams wcp{static_cast<uint16_t>(cacheRowsToKeep),
                              static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              0, cacheSkipBlocks * ALIGN_BYTES, 0};
        DataCopyPad(cacheStatesGM_[csOffset], cacheLocal[curBatchLen * dimSize], wcp);
    }
    if (curBatchLen > 0) {
        DataCopyExtParams wcp2{static_cast<uint16_t>(curBatchLen),
                               static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                               0, cacheSkipBlocks * ALIGN_BYTES, 0};
        DataCopyPad(cacheStatesGM_[csOffset + cacheRowsToKeep * cacheStride1_], xLocal[i * dimSize], wcp2);
    }
}

// ============================================================================
// WriteCacheLongBatch - 回写 cache（长 batch，长度 >= K）
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::WriteCacheLongBatch(LocalTensor<T>& xLocal,
    uint32_t i, uint16_t step, uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart,
    uint32_t cacheSkipBlocks, int64_t cIdx, uint32_t curBatchIdx)
{
    uint32_t K = kernelWidth_;

    bool needDeferredWrite = (curBatchIdx == firstBatchIdx_ && !firstBatchComplete_);
    if (needDeferredWrite) {
        // 延迟写回：SyncAll 后再从 GM 原始数据重建 cache
        firstBatchNeedsDeferredWrite_ = true;
        return;
    }

    uint32_t lastK1Start = i + step - (K - 1);
    uint64_t csOffset = (uint64_t)cIdx * cacheStride0_ + dimStart;
    DataCopyExtParams wcp{static_cast<uint16_t>(K - 1),
                          static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                          0, cacheSkipBlocks * ALIGN_BYTES, 0};
    DataCopyPad(cacheStatesGM_[csOffset], xLocal[lastK1Start * dimSize], wcp);
}

// ============================================================================
// ProcessMainCompute：统一的二维切分计算逻辑
// bsStart: 当前核在 BS 方向的起始位置（相对于有效序列）
// dimStart: 当前核在 Dim 方向的起始位置
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::ProcessMainCompute(uint64_t bsStart, uint64_t dimStart,
    uint32_t loopNumBS, uint32_t ubFactorBS, uint32_t ubTailFactorBS,
    uint32_t loopNumDim, uint32_t ubFactorDim, uint32_t ubTailFactorDim)
{
    // 每次 BS 方向前进的步长（由于有 K-1 重叠，每次实际前进 ubFactorBS - (K-1) 行）
    uint32_t K    = kernelWidth_;
    uint16_t step = (ubFactorBS > K - 1) ? (ubFactorBS - (K - 1)) : 1;

    // 只在核开始时调用一次 FindBatchIdx（传入全局位置）
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    uint32_t curBatchIdx = FindBatchIdx(bsStart);

    // iStart：只有 bsIdx_ == 0 的核的第一个 BS 循环从 0 开始，其余从 K-1 开始
    bool isBsFirstCore = (bsIdx_ == 0);
    uint32_t iStart = isBsFirstCore ? 0 : (K - 1);
    firstBatchNeedsDeferredWrite_ = false;

    uint64_t curBsOff = 0;  // 相对于 bsStart 的偏移

    for (uint32_t bsLoop = 0; bsLoop < loopNumBS; bsLoop++) {
        uint32_t curBS = (bsLoop == loopNumBS - 1) ? ubTailFactorBS : ubFactorBS;
        uint64_t curBsStart = bsStart + curBsOff;

        // 计算实际开始处理的全局位置（非重叠部分的第一个 token）
        uint64_t actualStart = curBsStart + iStart;

        // 更新 curBatchIdx 到 actualStart 所在的 batch（大多数情况下不执行，因为上一轮已更新）
        while (curBatchIdx + 1 < batchSize_ &&
               actualStart >= (uint64_t)seqStartLocal_.GetValue(curBatchIdx + 1)) {
            curBatchIdx++;
        }

        // 计算 curSequenceIdx（基于实际开始位置）
        uint64_t batchStart = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
        uint32_t curSequenceIdx = (uint32_t)(actualStart - batchStart);

        // 初始化第一个 batch 的跟踪信息（仅第一次 BS 迭代）
        // 必须在 while 循环推进 curBatchIdx 之后，确保指向实际处理的第一个 batch
        if (bsLoop == 0) {
            firstBatchIdx_      = curBatchIdx;
            firstBatchCIdx_     = cacheIdxLocal_.GetValue(curBatchIdx);
            firstBatchComplete_ = isBsFirstCore || (curSequenceIdx == 0);
        }

        uint32_t dimOff = (uint32_t)dimStart;
        uint32_t batchIdxForDim = curBatchIdx;
        uint32_t seqIdxForDim = curSequenceIdx;
        for (uint32_t dimLoop = 0; dimLoop < loopNumDim; dimLoop++) {
            uint32_t curDim = (dimLoop == loopNumDim - 1) ? ubTailFactorDim : ubFactorDim;
            // dim 循环处理同一块 BS 数据的不同 dim 切片，每次传入相同的初始 batch 状态
            batchIdxForDim = curBatchIdx;
            seqIdxForDim = curSequenceIdx;
            ProcessUBBlock((uint32_t)curBsStart, curBS, dimOff, curDim, iStart, batchIdxForDim, seqIdxForDim);
            dimOff += curDim;
        }
        // dim 循环结束后，从最后一次 ProcessUBBlock 获取更新后的 curBatchIdx
        curBatchIdx = batchIdxForDim;

        // BS 方向前进
        curBsOff += step;
        // 第一次循环后，iStart 始终为 K-1
        iStart = K - 1;
    }
}

// ============================================================================
// WriteDeferredCacheToStates：SyncAll 后，从 GM 原始数据重建 cache 并写回 cacheStates
//
// 当核的第一个 batch 不完整（跨核 split）时，compute 期间跳过了 cache 写回。
// SyncAll 后所有核已完成计算，可安全从 xGM 和 cacheStatesGM 读取原始数据重建 cache：
//   - Long batch (len >= K)：新 cache = x 的最后 K-1 行
//   - Short batch (len < K)：新 cache = [旧 cache 尾部, x 行]
//     （hasInitState==1 时从 GM 读旧 cache，否则填零）
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::WriteDeferredCacheToStates()
{
    if (!firstBatchNeedsDeferredWrite_ || firstBatchCIdx_ == padSlotId_) {
        return;
    }

    uint32_t K    = kernelWidth_;
    uint32_t rows = K - 1;

    // 计算当前核负责的 dim 范围
    uint32_t dimStart, dimSize;
    if (dimIdx_ < dimRemainderCores_) {
        dimStart = dimIdx_ * dimBlockFactor_;
        dimSize  = dimBlockFactor_;
    } else {
        dimStart = dimRemainderCores_ * dimBlockFactor_ + (dimIdx_ - dimRemainderCores_) * dimBlockTailFactor_;
        dimSize  = dimBlockTailFactor_;
    }

    // 计算 batch 信息
    uint64_t batchStart  = (uint64_t)seqStartLocal_.GetValue(firstBatchIdx_);
    uint64_t batchEnd    = (uint64_t)seqStartLocal_.GetValue(firstBatchIdx_ + 1);
    uint32_t curBatchLen = (uint32_t)(batchEnd - batchStart);

    uint32_t rowBlocks  = dimSize * sizeof(T) / ALIGN_BYTES;
    uint32_t cacheSkip  = (cacheStride1_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t xSkip      = (xStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;

    int64_t cIdx = firstBatchCIdx_;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    LocalTensor<T> tmpBuf = xQueue_.AllocTensor<T>();

    if (curBatchLen >= K) {
        // Long batch：从 xGM 读取 batch 最后 K-1 行
        uint64_t xSrcOffset = (batchEnd - rows) * xStride_ + dimStart;
        DataCopyExtParams rcp{static_cast<uint16_t>(rows),
                              static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                              xSkip * ALIGN_BYTES, 0, 0};
        DataCopyPad(tmpBuf, xGM_[xSrcOffset], rcp, padParams);
    } else {
        // Short batch：组装 [旧 cache 尾部, x 行]
        uint32_t cacheRowsToKeep = rows - curBatchLen;
        int32_t hasInit = initialStateModeNull_ ? 2 : hasInitLocal_.GetValue(firstBatchIdx_);

        if (cacheRowsToKeep > 0) {
            if (hasInit == 1) {
                // 从 cacheStatesGM 读取旧 cache 尾部（偏移 curBatchLen 行）
                uint64_t cacheSrcOffset = (uint64_t)cIdx * cacheStride0_ +
                                          (uint64_t)curBatchLen * cacheStride1_ + dimStart;
                DataCopyExtParams rcp{static_cast<uint16_t>(cacheRowsToKeep),
                                      static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                                      cacheSkip * ALIGN_BYTES, 0, 0};
                DataCopyPad(tmpBuf, cacheStatesGM_[cacheSrcOffset], rcp, padParams);
            } else {
                // hasInit == 0 或 2：旧 cache 为零
                Duplicate(tmpBuf, (T)0, cacheRowsToKeep * dimSize);
            }
        }
        if (curBatchLen > 0) {
            // 从 xGM 读取该 batch 的全部 x 行
            uint64_t xSrcOffset = batchStart * xStride_ + dimStart;
            DataCopyExtParams rcp{static_cast<uint16_t>(curBatchLen),
                                  static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                                  xSkip * ALIGN_BYTES, 0, 0};
            DataCopyPad(tmpBuf[cacheRowsToKeep * dimSize], xGM_[xSrcOffset], rcp, padParams);
        }
    }

    xQueue_.EnQue(tmpBuf);
    tmpBuf = xQueue_.DeQue<T>();

    // 写回 cacheStates
    uint64_t csDstOffset = (uint64_t)cIdx * cacheStride0_ + dimStart;
    DataCopyExtParams wcp{static_cast<uint16_t>(rows),
                          static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                          0, cacheSkip * ALIGN_BYTES, 0};
    DataCopyPad(cacheStatesGM_[csDstOffset], tmpBuf, wcp);

    xQueue_.FreeTensor(tmpBuf);
}

// ============================================================================
// Process：主流程
// ============================================================================
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::Process()
{
    uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= realCoreNum_) {
        return;
    }

    // 1. 一次性加载元数据
    LoadMetaData();

    // 2. 计算当前核在 BS 方向的起始位置和处理长度
    uint32_t K = kernelWidth_;
    uint64_t bsStart;
    if (bsIdx_ < bsRemainderCores_) {
        // 主核：前 bsRemainderCores_ 个核
        // 有效步长 = bsBlockFactor_ - (K-1)
        uint64_t effectiveStep = bsBlockFactor_ - (K - 1);
        bsStart = (uint64_t)bsIdx_ * effectiveStep;
    } else {
        // 尾核：后面的核
        // 前面主核贡献的总有效长度
        uint64_t effectiveStepMain = bsBlockFactor_ - (K - 1);
        uint64_t effectiveStepTail = bsBlockTailFactor_ - (K - 1);
        bsStart = (uint64_t)bsRemainderCores_ * effectiveStepMain +
                  (uint64_t)(bsIdx_ - bsRemainderCores_) * effectiveStepTail;
    }

    // 3. 计算当前核在 Dim 方向的起始位置
    uint64_t dimStart;
    if (dimIdx_ < dimRemainderCores_) {
        // 主核
        dimStart = (uint64_t)dimIdx_ * dimBlockFactor_;
    } else {
        // 尾核
        dimStart = (uint64_t)dimRemainderCores_ * dimBlockFactor_ +
                   (uint64_t)(dimIdx_ - dimRemainderCores_) * dimBlockTailFactor_;
    }

    // 4. 选择核内切分参数
    bool isBsTailCore  = (bsIdx_  >= bsRemainderCores_);
    bool isDimTailCore = (dimIdx_ >= dimRemainderCores_);

    // BS 方向：主核用主核参数，尾核用尾核参数
    uint32_t loopBS  = isBsTailCore  ? tailBlockloopNumBS_        : loopNumBS_;
    uint32_t factBS  = isBsTailCore  ? tailBlockubFactorBS_       : ubFactorBS_;
    uint32_t tailBS  = isBsTailCore  ? tailBlockubTailFactorBS_   : ubTailFactorBS_;

    // Dim 方向
    uint32_t loopDim = isDimTailCore ? tailBlockloopNumDim_       : loopNumDim_;
    uint32_t factDim = isDimTailCore ? tailBlockubFactorDim_      : ubFactorDim_;
    uint32_t tailDim = isDimTailCore ? tailBlockubTailFactorDim_  : ubTailFactorDim_;

    // 5. 调用统一的计算逻辑
    ProcessMainCompute(bsStart, dimStart, loopBS, factBS, tailBS, loopDim, factDim, tailDim);

    // 6. 全核同步（保证所有核完成计算后再执行延迟 cache 写回）
    SyncAll();

    // 7. 从 GM 原始数据重建 cache 并写回 cacheStates（延迟写回）
    WriteDeferredCacheToStates();
}

} // namespace FusedCausalConv1dCutBSHNs

#endif // FUSED_CAUSAL_CONV1D_CUT_BSH_H
