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
 * \file fused_causal_conv1d_cut_bh.h
 * \brief FusedCausalConv1dCutBH kernel implementation
 */

#ifndef CAUSAL_CONV1D_CUT_BH_H
#define CAUSAL_CONV1D_CUT_BH_H

#include "kernel_operator.h"
#include "fused_causal_conv1d_cut_bh_struct.h"
#include "vf/fused_causal_conv1d_state_single_tail.h"
#include "vf/fused_causal_conv1d_128dim.h"

using namespace AscendC;

// ========== 常量定义 ==========
constexpr int32_t DOUBLE_BUFFER = 2;    // 双Buffer数量，用于重叠数据搬运和计算
constexpr int32_t ALIGN_BYTES = 32;     // 32字节对齐，用于DataCopyParams的stride和对齐计算
constexpr int32_t MAX_SEQUENCE_LEN = 8; // F4: m最大为7，seqLen最大为m+1=8
constexpr int32_t CAST_REP_SIZE = 64;
constexpr int32_t CONV_WIDTH4 = 4;
constexpr int32_t CONV_STATE_WIDTH4 = CONV_WIDTH4 - 1;

template <typename T>
class FusedCausalConv1dCutBH {
public:
    __aicore__ inline FusedCausalConv1dCutBH(TPipe *pipe) : pipe_(pipe){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR numAcceptedToken, GM_ADDR y,
                                const FusedCausalConv1dCutBHTilingData *tilingData);

    __aicore__ inline void Process();

private:
    // ========== Init子函数 ==========
    __aicore__ inline void InitParams(const FusedCausalConv1dCutBHTilingData *tilingData);
    __aicore__ inline void InitQueues();

    // ========== 三阶段流水线函数 ==========
    __aicore__ inline void CopyInWeightBias();
    __aicore__ inline void CopyInX(uint16_t blockCount, int64_t xOffset);
    __aicore__ inline void
    Compute(int32_t batchLoop, const LocalTensor<T> &weightBiasLocal, const LocalTensor<int32_t> &indicesLocal,
            const LocalTensor<int32_t> &acceptTokenLocal, const LocalTensor<int32_t> &queryStartLocLocal,
            uint16_t yBlockCount, int64_t yOffset);

    __aicore__ inline void UpdateConvStates(const LocalTensor<T> &xLocal, const LocalTensor<T> &convStatesLocal,
                                            int32_t curBatchUbOffset, int64_t writeCacheLine, int32_t curBatchSeq,
                                            int32_t offset);

    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    // ========== Global Memory指针 ==========
    GlobalTensor<T> xGm;                                 // 输入序列 [batch, seqLen, dim]
    GlobalTensor<T> weightGm;                            // 卷积核 [K, dim]
    GlobalTensor<T> biasGm;                              // 偏置 [dim]
    GlobalTensor<T> convStatesGm;                        // 输入cache state [-1, K-1+m, dim]
    GlobalTensor<int32_t> cacheIndicesGm;                // cache索引：[batch]
    GlobalTensor<int32_t> acceptTokenNumGm;              // 接受的token数 [batch]
    GlobalTensor<int32_t> queryStartLocGm;               // query起始位置 [batch+1]
    GlobalTensor<T> yGm;                                 // 输出序列 [batch, seqLen, dim]

    // ========== UB队列（Unified Buffer中的数据缓存） ==========
    TQue<QuePosition::VECIN, 1> cacheQueueIn;
    TQue<QuePosition::VECOUT, 1> cacheQueueOut;
    TQue<QuePosition::VECIN, 1> weightBiasQueue; // weight和bias合并队列（前半段weight，后半段bias）
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECOUT, 1> yQueue;
    TBuf<TPosition::VECCALC> indicesBuf;           // cache索引 1D=[batch]
    TBuf<TPosition::VECCALC> acceptTokenBuf;       // acceptToken（始终）
    TBuf<TPosition::VECCALC> queryStartLocBuf;     // queryStartLoc（2D输入时）
    TPipe *pipe_;                                  // Pipeline管理对象

    // ========== Tiling参数（二维切分：Dim方向 × Batch方向） ==========
    // 核间切分参数
    int64_t dimCoreCnt_;       // dim方向的核数
    int64_t dimMainCoreCnt_;   // dim方向的主核个数
    int64_t mainCoredimLen_;   // 主核处理的Dim大小（128对齐），尾核可能含不足128的余数元素
    int64_t tailCoredimLen_;   // 尾核处理的Dim大小
    int64_t batchMainCoreCnt_; // batch方向的主核个数
    int64_t mainCoreBatchNum_; // 主核batch个数
    int64_t tailCoreBatchNum_; // 尾核batch个数

    // 核内UB切分参数
    int64_t ubMainFactorBS_;  // 主UB循环处理的batch数
    int64_t ubTailFactorBS_;  // 尾UB循环处理的batch数
    int64_t ubMainFactorDim_; // 主UB循环处理的dim数
    int64_t ubTailFactorDim_; // 尾UB循环处理的dim数
    int64_t loopNumBS_;       // Batch方向的循环次数
    int64_t loopNumDim_;      // Dim方向的循环次数

    // ========== Shape参数 ==========
    int64_t batchSize_;  // Batch大小
    int64_t seqLen_;     // 序列长度（m+1）
    int64_t cuSeqLen_;   // 累积序列长度（2D输入用）
    int64_t dim_;        // 特征维度
    int64_t stateLen_;   // cache_state第二维 (K-1+m)
    int64_t xInputMode_; // 输入模式：0=3D, 1=2D
    int64_t hasQueryStartLoc_;

    // ========== 运行时计算参数 ==========
    int32_t blockIdx_;          // 当前核的索引
    int32_t batchIdx_;          // 当前核在Batch维度的索引
    int32_t dimIdx_;            // 当前核在Dim维度的索引
    int32_t firstBatchIdx_;     // 当前核处理的起始batch索引
    int32_t coreBatchNum_;      // 当前核处理的batch数量
    int32_t dimOffset_;         // 当前核处理的Dim起始偏移（元素）
    int32_t coreDimLen_;        // 当前核处理的Dim大小
    int64_t hasAcceptTokenNum_; // 是否提供了acceptTokenNum输入
    int64_t hasBias_;           // 是否提供 bias 输入
    int64_t activationMode_;    // 0: None, 1: SiLU, 2: Swish
    int64_t dimSum_;            // x每个token间的stride
    int64_t cacheLenSum_;       // cache每个token间的stride
    int64_t cacheBatchLenSum_;  // cache每个batch间的stride
    int32_t batchNumInLoop_;
    int32_t dimSizeInLoop_;
    int32_t dimOffsetPerLoop_;
    int8_t padSlotId_;
    int64_t yDim_;
    int64_t maxQueryLen_;
    int64_t hasCacheIndices_;      // cacheIndices 是否有效（F2）
    int32_t xQueueSize_;           // xQueue 单 buffer 大小
    int32_t weightBiasQueueSize_;  // weightBiasQueue 单 buffer 大小（weight部分 + bias部分）
    int32_t weightPartSize_;       // weightBiasQueue 中 weight 部分的元素个数（用于计算bias偏移）
};

template <typename T>
__aicore__ inline void
FusedCausalConv1dCutBH<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR numAcceptedToken, GM_ADDR y,
                                const FusedCausalConv1dCutBHTilingData *tilingData)
{
    InitParams(tilingData);

    // === 设置Global Memory buffers ===
    if (xInputMode_ == 0) {
        xGm.SetGlobalBuffer((__gm__ T *)x, batchSize_ * seqLen_ * dimSum_);
        yGm.SetGlobalBuffer((__gm__ T *)y, batchSize_ * seqLen_ * yDim_);
    } else {
        xGm.SetGlobalBuffer((__gm__ T *)x, cuSeqLen_ * dimSum_);
        yGm.SetGlobalBuffer((__gm__ T *)y, cuSeqLen_ * yDim_);
    }
    weightGm.SetGlobalBuffer((__gm__ T *)weight, CONV_WIDTH4 * dim_);
    if (hasBias_) {
        biasGm.SetGlobalBuffer((__gm__ T *)bias, dim_);
    }
    convStatesGm.SetGlobalBuffer((__gm__ T *)convStates);

    if (hasCacheIndices_) {
        cacheIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cacheIndices, batchSize_);
    }

    if (hasAcceptTokenNum_ == 1) {
        acceptTokenNumGm.SetGlobalBuffer((__gm__ int32_t *)numAcceptedToken, batchSize_);
    }
    if (hasQueryStartLoc_ == 1) {
        queryStartLocGm.SetGlobalBuffer((__gm__ int32_t *)queryStartLoc, batchSize_ + 1);
    }

    InitQueues();
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::InitParams(const FusedCausalConv1dCutBHTilingData *tilingData)
{
    // === 核间切分参数（二维：Dim方向 × Batch方向） ===
    dimCoreCnt_ = tilingData->dimCoreCnt;
    dimMainCoreCnt_ = tilingData->dimMainCoreCnt;
    mainCoredimLen_ = tilingData->mainCoredimLen;
    tailCoredimLen_ = tilingData->tailCoredimLen;
    batchMainCoreCnt_ = tilingData->batchMainCoreCnt;
    mainCoreBatchNum_ = tilingData->mainCoreBatchNum;
    tailCoreBatchNum_ = tilingData->tailCoreBatchNum;

    // === shape参数 ===
    batchSize_ = tilingData->batchSize;
    seqLen_ = tilingData->seqLen;
    cuSeqLen_ = tilingData->cuSeqLen;
    dim_ = tilingData->dim;
    stateLen_ = tilingData->stateLen;
    hasAcceptTokenNum_ = tilingData->hasAcceptTokenNum;
    hasBias_ = tilingData->hasBias;
    activationMode_ = tilingData->activationMode;
    xInputMode_ = tilingData->xInputMode;
    hasQueryStartLoc_ = tilingData->hasQueryStartLoc;
    padSlotId_ = tilingData->padSlotId;
    hasCacheIndices_ = tilingData->hasCacheIndices;
    maxQueryLen_ = tilingData->maxQueryLen;

    // === stride参数 ===
    dimSum_ = tilingData->xStride;
    cacheLenSum_ = tilingData->cacheStride1;
    cacheBatchLenSum_ = tilingData->cacheStride0;
    yDim_ = dim_;

    // === 当前核的参数计算 ===
    blockIdx_ = GetBlockIdx();
    batchIdx_ = blockIdx_ / dimCoreCnt_;           // Batch方向索引
    dimIdx_ = blockIdx_ - batchIdx_ * dimCoreCnt_; // Dim方向索引

    // dim 方向：多主核/多尾核策略
    //   dimIdx_ < dimMainCoreCnt_  → 大核（每核 mainCoredimLen_ 元素）
    //   dimIdx_ >= dimMainCoreCnt_ → 小核（每核 tailCoredimLen_ 元素）
    //   dimIdx_ == dimCoreCnt_-1   → 最后一个 dim 核，额外处理 dimRemainderElems 余数元素
    bool isLastDimCore = (dimIdx_ == dimCoreCnt_ - 1);
    if (dimIdx_ < dimMainCoreCnt_) {
        // 大核
        loopNumDim_ = tilingData->loopNumDim;
        coreDimLen_ = mainCoredimLen_;
        ubMainFactorDim_ = tilingData->ubMainFactorDim;
        ubTailFactorDim_ = tilingData->ubTailFactorDim;
        dimOffset_ = dimIdx_ * mainCoredimLen_;
    } else {
        // 小核：偏移 = 所有大核占用的 dim 量 + 当前小核在小核中的序号 × tailCoredimLen_
        int64_t tailIdx = dimIdx_ - dimMainCoreCnt_;
        loopNumDim_ = tilingData->tailBlockloopNumDim;
        coreDimLen_ = tailCoredimLen_;
        ubMainFactorDim_ = tilingData->tailBlockubFactorDim;
        ubTailFactorDim_ = tilingData->tailBlockubTailFactorDim;
        dimOffset_ = dimMainCoreCnt_ * mainCoredimLen_ + tailIdx * tailCoredimLen_;
    }
    // 最后一个 dim 核：在普通 dim 大小基础上追加余数元素，使用专用 UB 参数
    if (isLastDimCore && tilingData->dimRemainderElems > 0) {
        coreDimLen_ += tilingData->dimRemainderElems;
        loopNumDim_ = tilingData->lastCoreloopNumDim;
        ubMainFactorDim_ = tilingData->lastCoreubMainFactorDim;
        ubTailFactorDim_ = tilingData->lastCoreubTailFactorDim;
    }

    // batch 方向：多主核/多尾核策略
    //   batchIdx_ < batchMainCoreCnt_  → 大核（每核 mainCoreBatchNum_ 个 batch）
    //   batchIdx_ >= batchMainCoreCnt_ → 小核（每核 tailCoreBatchNum_ 个 batch）
    if (batchIdx_ < batchMainCoreCnt_) {
        loopNumBS_ = tilingData->loopNumBS;
        ubMainFactorBS_ = tilingData->ubMainFactorBS;
        ubTailFactorBS_ = tilingData->ubTailFactorBS;
        coreBatchNum_ = mainCoreBatchNum_;
        firstBatchIdx_ = batchIdx_ * mainCoreBatchNum_;
    } else {
        int64_t batchTailIdx = batchIdx_ - batchMainCoreCnt_;
        loopNumBS_ = tilingData->tailBlockloopNumBS;
        ubMainFactorBS_ = tilingData->tailBlockubFactorBS;
        ubTailFactorBS_ = tilingData->tailBlockubTailFactorBS;
        coreBatchNum_ = tailCoreBatchNum_;
        firstBatchIdx_ = batchMainCoreCnt_ * mainCoreBatchNum_ + batchTailIdx * tailCoreBatchNum_;
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::InitQueues()
{
    xQueueSize_ = ubMainFactorBS_ * seqLen_ * ubMainFactorDim_ * sizeof(T);
    if (xInputMode_ == 1) {
        xQueueSize_ = ubMainFactorBS_ * maxQueryLen_ * ubMainFactorDim_ * sizeof(T);
    }
    int32_t weightQueueSize = CONV_WIDTH4 * ubMainFactorDim_ * sizeof(T);
    int32_t biasQueueSize = ubMainFactorDim_ * sizeof(T);
    weightPartSize_ = weightQueueSize / sizeof(T);
    weightBiasQueueSize_ = weightQueueSize + biasQueueSize;
    pipe_->InitBuffer(weightBiasQueue, 1, weightBiasQueueSize_);
    pipe_->InitBuffer(xQueue, 1, xQueueSize_);
    pipe_->InitBuffer(yQueue, 1, xQueueSize_);

    // cacheQueue: 存储cache state（stateLen行）
    int32_t cacheQueueSize = stateLen_ * ubMainFactorDim_ * sizeof(T);
    pipe_->InitBuffer(cacheQueueIn, DOUBLE_BUFFER, cacheQueueSize);
    pipe_->InitBuffer(cacheQueueOut, DOUBLE_BUFFER, cacheQueueSize);

    // acceptTokenBuf: 始终分配
    int32_t batchBufSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    pipe_->InitBuffer(acceptTokenBuf, batchBufSize);

    // indicesBuf：有cacheIndices时分配 1D [batch] 大小
    if (hasCacheIndices_) {
        int32_t indicesBufSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        pipe_->InitBuffer(indicesBuf, indicesBufSize);
    }

    // queryStartLocBuf: 仅2D输入时分配
    if (hasQueryStartLoc_ == 1) {
        int32_t queryStartLocQueueSize =
            ((batchSize_ + 1) * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        pipe_->InitBuffer(queryStartLocBuf, queryStartLocQueueSize);
    }

}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::Process()
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    uint32_t indicesBlockLen = batchSize_ * sizeof(int32_t);
    DataCopyParams batchCopyParams;
    batchCopyParams.blockCount = 1;
    batchCopyParams.blockLen = indicesBlockLen;
    batchCopyParams.srcStride = 0;
    batchCopyParams.dstStride = 0;

    // === acceptTokenBuf：始终分配，无输入时填充1 ===
    LocalTensor<int32_t> acceptTokenLocal = acceptTokenBuf.Get<int32_t>();
    Duplicate(acceptTokenLocal, static_cast<int32_t>(1), batchSize_);
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2); // 确保Duplicate(V)完成后再做MTE2
    if (hasAcceptTokenNum_ == 1) {
        DataCopyPad(acceptTokenLocal, acceptTokenNumGm, batchCopyParams, padParams);
    }

    LocalTensor<int32_t> indicesLocal;
    if (hasCacheIndices_) {
        // 有cacheIndices：一次性将全部 batch 的 1D 索引从 GM 搬入 UB
        indicesLocal = indicesBuf.Get<int32_t>();
        DataCopyPad(indicesLocal, cacheIndicesGm, batchCopyParams, padParams);
    }

    // 2D输入时搬运 queryStartLoc
    LocalTensor<int32_t> queryStartLocLocal;
    if (hasQueryStartLoc_ == 1) {
        queryStartLocLocal = queryStartLocBuf.Get<int32_t>();
        uint32_t queryStartLocBlockLen = (batchSize_ + 1) * sizeof(int32_t);
        DataCopyParams queryStartLocCopyParams;
        queryStartLocCopyParams.blockCount = 1;
        queryStartLocCopyParams.blockLen = queryStartLocBlockLen;
        queryStartLocCopyParams.srcStride = 0;
        queryStartLocCopyParams.dstStride = 0;
        DataCopyPad(queryStartLocLocal, queryStartLocGm, queryStartLocCopyParams, padParams);
    }

    // 等待所有常驻数据预搬运完成
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    // === 主双重循环（Dim方向 × BS方向）===
    // 同一个 dim tile 的 weight/bias 对所有 batch 复用，避免 batchLoop 内重复搬运。
    for (int32_t dimLoop = 0; dimLoop < loopNumDim_; dimLoop++) {
        dimSizeInLoop_ = (dimLoop == loopNumDim_ - 1) ? ubTailFactorDim_ : ubMainFactorDim_;
        dimOffsetPerLoop_ = dimLoop * ubMainFactorDim_;
        CopyInWeightBias();
        LocalTensor<T> weightBiasLocal = weightBiasQueue.DeQue<T>();

        for (int32_t batchLoop = 0; batchLoop < loopNumBS_; batchLoop++) {
            batchNumInLoop_ = (batchLoop == loopNumBS_ - 1) ? ubTailFactorBS_ : ubMainFactorBS_;
            int32_t firstBatchIdxInLoop = firstBatchIdx_ + batchLoop * ubMainFactorBS_;
            uint16_t blockCount = batchNumInLoop_ * seqLen_;
            int64_t startSeqIdx = firstBatchIdxInLoop * seqLen_;
            int64_t yOffset = firstBatchIdxInLoop * seqLen_ * yDim_ + dimOffset_ + dimOffsetPerLoop_;
            if (hasQueryStartLoc_ == 1) {
                blockCount = queryStartLocLocal.GetValue(firstBatchIdxInLoop + batchNumInLoop_) -
                             queryStartLocLocal.GetValue(firstBatchIdxInLoop);
                startSeqIdx = queryStartLocLocal.GetValue(firstBatchIdxInLoop) - queryStartLocLocal.GetValue(0);
                yOffset = startSeqIdx * yDim_ + dimOffset_ + dimOffsetPerLoop_;
            }
            int64_t xOffset = startSeqIdx * dimSum_ + dimOffset_ + dimOffsetPerLoop_;
            CopyInX(blockCount, xOffset);
            Compute(batchLoop, weightBiasLocal, indicesLocal, acceptTokenLocal, queryStartLocLocal, blockCount, yOffset);
        }
        weightBiasQueue.FreeTensor(weightBiasLocal);
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::CopyInWeightBias()
{
    LocalTensor<T> weightBiasLocal = weightBiasQueue.AllocTensor<T>();
    LocalTensor<T> weightLocal = weightBiasLocal[0];
    LocalTensor<T> biasLocal = weightBiasLocal[weightPartSize_];
    DataCopyPadParams padParams{false, 0, 0, 0};

    // === 拷贝weight数据：[4, dimSizeInLoop] ===
    int32_t weightOffset = dimOffset_ + dimOffsetPerLoop_;
    DataCopyParams weightCopyParams;
    weightCopyParams.blockCount = CONV_WIDTH4;
    weightCopyParams.blockLen = dimSizeInLoop_ * sizeof(T);
    weightCopyParams.srcStride = (dim_ - dimSizeInLoop_) * sizeof(T);
    weightCopyParams.dstStride = 0;
    DataCopyPad(weightLocal, weightGm[weightOffset], weightCopyParams, padParams);

    if (hasBias_) {
        DataCopyParams biasCopyParams;
        biasCopyParams.blockCount = 1;
        biasCopyParams.blockLen = dimSizeInLoop_ * sizeof(T);
        biasCopyParams.srcStride = 0;
        biasCopyParams.dstStride = 0;
        DataCopyPad(biasLocal, biasGm[weightOffset], biasCopyParams, padParams);
    }

    weightBiasQueue.EnQue(weightBiasLocal);
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::CopyInX(uint16_t blockCount, int64_t xOffset)
{
    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();

    // === 拷贝x数据：[blockCount, dimSizeInLoop] ===
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = dimSizeInLoop_ * sizeof(T);
    dataCopyParams.srcStride = (dimSum_ - dimSizeInLoop_) * sizeof(T);
    dataCopyParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[xOffset], dataCopyParams, padParams);

    xQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::Compute(
    int32_t batchLoop, const LocalTensor<T> &weightBiasLocal, const LocalTensor<int32_t> &indicesLocal,
    const LocalTensor<int32_t> &acceptTokenLocal, const LocalTensor<int32_t> &queryStartLocLocal,
    uint16_t yBlockCount, int64_t yOffset)
{
    LocalTensor<T> xLocal = xQueue.DeQue<T>();
    LocalTensor<T> weightLocal = weightBiasLocal[0];
    LocalTensor<T> biasLocal = weightBiasLocal[weightPartSize_];
    LocalTensor<T> yLocal = yQueue.AllocTensor<T>();

    int32_t batchNumPerLoop = batchLoop * ubMainFactorBS_;
    int64_t yUbOffset = 0;
    DataCopyPadParams padParams{false, 0, 0, 0};

    DataCopyParams cacheCopyParams;
    // blockCount 在 batch 循环内按 offset 动态设置
    cacheCopyParams.blockLen = dimSizeInLoop_ * sizeof(T);
    cacheCopyParams.srcStride = (cacheLenSum_ - dimSizeInLoop_) * sizeof(T);
    cacheCopyParams.dstStride = 0;

    for (int32_t b = 0; b < batchNumInLoop_; b++) {
        int32_t curBatchIdx = firstBatchIdx_ + batchNumPerLoop + b;
        LocalTensor<T> convStatesLocalIn = cacheQueueIn.AllocTensor<T>();
        // cache_indices 存在时使用物理 slot，否则 batch id 即 slot。
        int64_t readCacheLine = curBatchIdx;
        int64_t writeCacheLine = curBatchIdx;
        if (hasCacheIndices_) {
            readCacheLine = indicesLocal.GetValue(curBatchIdx);
            writeCacheLine = indicesLocal.GetValue(curBatchIdx);
        }

        // ========== 计算当前 batch 的序列长度和 UB 偏移 ==========
        int32_t curBatchSeq = seqLen_;
        int32_t curBatchUbOffset = b * curBatchSeq * dimSizeInLoop_;
        if (hasQueryStartLoc_ == 1) {
            curBatchSeq = queryStartLocLocal.GetValue(curBatchIdx + 1) - queryStartLocLocal.GetValue(curBatchIdx);
            curBatchUbOffset =
                (queryStartLocLocal.GetValue(curBatchIdx) - queryStartLocLocal.GetValue(curBatchIdx - b)) *
                dimSizeInLoop_;
        }

        // pad_slot_id 仅在 cache_indices 映射到 padding slot 时生效。
        if (readCacheLine == padSlotId_) {
            cacheQueueIn.FreeTensor(convStatesLocalIn);
            yUbOffset += curBatchSeq * dimSizeInLoop_;
            continue;
        }

        // num_accepted_tokens 存在时表示 speculative/MTP 接受 token 数；
        // 普通 decode 没有该输入时，读取 state 末尾 W-1 行。
        int32_t acceptToken = acceptTokenLocal.GetValue(curBatchIdx);
        int32_t offset = hasAcceptTokenNum_ ? (acceptToken - 1) : (stateLen_ - CONV_STATE_WIDTH4);
        if (offset < 0) {
            offset = 0;
        }

        // 搬入 [0, offset + W - 1) 行。计算只用 offset 开始的 W-1 行，更新 state 需要前缀历史。
        cacheCopyParams.blockCount = offset + CONV_STATE_WIDTH4;

        int32_t convStatesGmOffset = readCacheLine * cacheBatchLenSum_ + dimOffset_ + dimOffsetPerLoop_;
        DataCopyPad(convStatesLocalIn, convStatesGm[convStatesGmOffset], cacheCopyParams, padParams);
        cacheQueueIn.EnQue<T>(convStatesLocalIn);
        convStatesLocalIn = cacheQueueIn.DeQue<T>();

        LocalTensor<T> convStatesLocalOut = cacheQueueOut.AllocTensor<T>();
        int32_t cacheRowsCopied = offset + CONV_STATE_WIDTH4;
        DataCopy(convStatesLocalOut, convStatesLocalIn, cacheRowsCopied * dimSizeInLoop_);
        cacheQueueOut.EnQue<T>(convStatesLocalOut);

        // 更新 running state：保留必要旧历史并追加本次 accepted/new token。
        convStatesLocalOut = cacheQueueOut.DeQue<T>();
        UpdateConvStates(xLocal, convStatesLocalOut, curBatchUbOffset, writeCacheLine, curBatchSeq, offset);
        cacheQueueOut.FreeTensor<T>(convStatesLocalOut);

        // ========== 卷积计算（情况A：seq[0, K-2]含cache；情况B：seq[K-1, end]纯x）==========
        uint32_t convStateLen = (curBatchSeq < CONV_STATE_WIDTH4) ? curBatchSeq : CONV_STATE_WIDTH4;
        LocalTensor<T> xSlice = xLocal[curBatchUbOffset];
        if (convStateLen > 0) {
            LocalTensor<T> y1Slice = yLocal[yUbOffset];
            LocalTensor<T> convStatesSlice = convStatesLocalIn[offset * dimSizeInLoop_];
            uint16_t dimRem = dimSizeInLoop_ - dimSizeInLoop_ / CAST_REP_SIZE * CAST_REP_SIZE;
            if (dimRem == 0) {
                Conv1dNeedStateBH(xSlice, weightLocal, biasLocal, convStatesSlice, y1Slice, convStateLen,
                                  dimSizeInLoop_, hasBias_, activationMode_);
            } else {
                Conv1dNeedStateSingleTailBH(xSlice, weightLocal, biasLocal, convStatesSlice, y1Slice, convStateLen,
                                            dimSizeInLoop_, hasBias_, activationMode_);
            }
        }
        cacheQueueIn.FreeTensor<T>(convStatesLocalIn);

        int16_t blockCount = curBatchSeq - convStateLen;
        if (blockCount > 0) {
            LocalTensor<T> y2Slice = yLocal[yUbOffset + convStateLen * dimSizeInLoop_];
            uint16_t dimRem = dimSizeInLoop_ - dimSizeInLoop_ / CAST_REP_SIZE * CAST_REP_SIZE;
            for (int16_t blockIdx = 0; blockIdx < blockCount; ++blockIdx) {
                uint32_t xWindowOffset = static_cast<uint32_t>(blockIdx) * dimSizeInLoop_;
                uint32_t yOffset = static_cast<uint32_t>(blockIdx) * dimSizeInLoop_;
                if (dimRem == 0) {
                    Conv1dPureXW4(xSlice, weightLocal, biasLocal, y2Slice, xWindowOffset, yOffset, dimSizeInLoop_,
                                  hasBias_, activationMode_);
                } else {
                    Conv1dPureXW4SingleTail(xSlice, weightLocal, biasLocal, y2Slice, xWindowOffset, yOffset,
                                            dimSizeInLoop_, hasBias_, activationMode_);
                }
            }
        }
        yUbOffset += curBatchSeq * dimSizeInLoop_;
    }

    // === 写出 yLocal ===
    yQueue.EnQue<T>(yLocal);
    yLocal = yQueue.DeQue<T>();

    DataCopyParams yParams;
    yParams.blockCount = yBlockCount;
    yParams.blockLen = dimSizeInLoop_ * sizeof(T);
    yParams.srcStride = 0;
    yParams.dstStride = (yDim_ - dimSizeInLoop_) * sizeof(T);
    // 写 y GM
    DataCopyPad(yGm[yOffset], yLocal, yParams);
    xQueue.FreeTensor(xLocal);
    yQueue.FreeTensor(yLocal);
}

// ========== F5: UpdateConvStates（尾对齐机制重构）==========
template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::UpdateConvStates(const LocalTensor<T> &xLocal,
                                                                   const LocalTensor<T> &convStatesLocal,
                                                                   int32_t curBatchUbOffset, int64_t writeCacheLine,
                                                                   int32_t curBatchSeq, int32_t offset)
{
    // padded_input 总长 = (offset + K-1) + seq_len
    int64_t paddedLen = offset + CONV_STATE_WIDTH4 + curBatchSeq;
    // 实际写入行数（尾对齐：最多写 stateLen_ 行）
    int64_t cacheLen = paddedLen < stateLen_ ? paddedLen : stateLen_;
    int64_t xRows = cacheLen < (int64_t)curBatchSeq ? cacheLen : (int64_t)curBatchSeq;
    int64_t cacheRows = cacheLen - xRows; // 可能为0

    // 写入 GM 的起始偏移（尾对齐：从 stateLen_ - cacheLen 行开始写）
    int64_t dstBaseGmOffset =
        writeCacheLine * cacheBatchLenSum_ + (stateLen_ - cacheLen) * cacheLenSum_ + dimOffset_ + dimOffsetPerLoop_;
    uint32_t blockLen = dimSizeInLoop_ * sizeof(T);
    uint32_t dstStride = (cacheLenSum_ - dimSizeInLoop_) * sizeof(T);

    // Step1：拷贝旧 cache 末尾 cacheRows 行（跳过头部 (K-1-cacheRows) 行）
    if (cacheRows > 0) {
        int32_t srcCacheSkip = offset + CONV_STATE_WIDTH4 - cacheRows;
        DataCopyParams cacheToGmParams;
        cacheToGmParams.blockCount = cacheRows;
        cacheToGmParams.blockLen = blockLen;
        cacheToGmParams.srcStride = 0;
        cacheToGmParams.dstStride = dstStride;
        DataCopyPad(convStatesGm[dstBaseGmOffset], convStatesLocal[srcCacheSkip * dimSizeInLoop_], cacheToGmParams);
    }

    // Step2：拷贝 xLocal 末尾 xRows 行
    int32_t xStartRowInUb = curBatchUbOffset + (curBatchSeq - xRows) * dimSizeInLoop_;
    int64_t xDstGmOffset = dstBaseGmOffset + cacheRows * cacheLenSum_;
    DataCopyParams xToGmParams;
    xToGmParams.blockCount = xRows;
    xToGmParams.blockLen = blockLen;
    xToGmParams.srcStride = 0;
    xToGmParams.dstStride = dstStride;
    DataCopyPad(convStatesGm[xDstGmOffset], xLocal[xStartRowInUb], xToGmParams);
}

#endif // CAUSAL_CONV1D_CUT_BH_H
