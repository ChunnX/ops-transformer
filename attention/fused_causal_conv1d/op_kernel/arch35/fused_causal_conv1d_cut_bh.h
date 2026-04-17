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
 *
 * 本文件实现了fused_causal_conv1d_cut_bh算子的核函数，用于执行因果一维卷积并更新缓存状态。
 *
 * 算子功能：
 * 1. 对输入序列执行因果1D卷积（每个特征通道独立）
 * 2. 根据acceptTokenNum动态使用历史cache state和当前输入x
 * 3. 计算完成后自动更新cache state，确保后续推理能正确延续上下文
 * 4. 保证输出满足因果性约束（输出位置j只能看到输入位置0到j）
 *
 * 主要特点：
 * - 支持FP16/BF16数据类型
 * - 使用双Buffer机制提高数据搬运和计算的并行度
 * - 支持核间切分（只切batch维度）和核内切分（BS和Dim方向）
 * - 针对序列位置<K-1和>=K-1采用不同的计算策略
 */

#ifndef CAUSAL_CONV1D_CUT_BH_H
#define CAUSAL_CONV1D_CUT_BH_H

#include "kernel_operator.h"
#include "fused_causal_conv1d_cut_bh_struct.h"
#include "vf/compute.h"

using namespace AscendC;

// ========== 常量定义 ==========
constexpr int32_t BUFFER_NUM = 2;    // 双Buffer数量，用于重叠数据搬运和计算
constexpr int32_t ALIGN_BYTES = 32;  // 32字节对齐，用于DataCopyParams的stride和对齐计算
constexpr int32_t MAX_SEQUENCE_LEN = 6;

// TilingData结构定义在Host侧：op_host/fused_causal_conv1d_cut_bh_tiling_arch35.h
// 核间切分策略：二维切分（Dim方向 × Batch方向）
// - Dim方向：256B对齐分割，mainCoredimLen_ * (dimCoreCnt-1) + dimTailSize = dim
// - Batch方向：按照有效batch范围分配，支持过滤无效batch

// ========== 核函数主类 ==========
/**
 * @brief FusedCausalConv1dCutBH核函数实现类
 * @tparam T 数据类型（half或bfloat16）
 *
 * 该类实现了因果1D卷积的完整计算流程：
 * 1. Init(): 初始化Global Memory指针、分配UB队列和TBuf
 * 2. Process(): 执行双重循环（BS方向×Dim方向），每次迭代处理一个UB块
 * 3. CopyIn/Compute/CopyOut: 三阶段流水线处理
 *
 * 内存布局：
 * - xQueue: 存储输入x数据 [batchNum, seqLen, currentDim]
 * - weightQueue: 存储卷积核 [K, currentDim]
 * - cacheQueueIn: 存储历史cache state [K-1+m, currentDim]
 * - cacheQueueOut: 存储历史cache state [K-1+m, currentDim]
 * - indicesQueue: 存储cache索引 [batchNum]
 * - acceptTokenQueue: 存储接受的token数 [batchNum]
 * - yQueue: 存储输出y数据
 */
template <typename T>
class FusedCausalConv1dCutBH {
public:
    __aicore__ inline FusedCausalConv1dCutBH(TPipe *pipe) : pipe_(pipe){};
    /**
     * @brief 初始化函数，设置所有Global Memory指针并分配UB资源
     * @param x 输入序列 [batch, m+1, dim]
     * @param weight 卷积核 [K, dim]
     * @param convStates 输入的cache state [-1, K-1+m, dim]
     * @param cacheIndices cache索引 [batch]，指定每个batch对应的cache state位置
     * @param numAcceptedToken 接受的token数量 [batch]，可选输入
     * @param queryStartLoc query起始位置 [batch+1]
     * @param y 输出序列 [batch, m+1, dim]
     * @param outputconvStates 输出的cache state，原地更新
     * @param tilingData tiling参数结构体指针
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR queryStartLoc,
        GM_ADDR cacheIndices, GM_ADDR numAcceptedToken, GM_ADDR y, const FusedCausalConv1dCutBHTilingData *tilingData);

    /**
     * @brief 主处理函数，执行双重循环处理所有数据
     *
     * 处理流程：
     * for loopBS in [0, loopNumBS):     # BS方向循环
     *     for j in [0, loopNumDim): # Dim方向循环
     *         CopyIn(loopBS, j)          # 从GM搬入数据到UB
     *         Compute(loopBS, j)         # 在UB上执行计算
     */
    __aicore__ inline void Process();

private:
    // ========== Init子函数 ==========
    __aicore__ inline void InitParams(const FusedCausalConv1dCutBHTilingData *tilingData);
    __aicore__ inline void InitQueues();

    // ========== 三阶段流水线函数 ==========
    __aicore__ inline void CopyIn(int32_t batchLoop, int32_t dimLoop, uint16_t blockCount, int64_t xOffset);
    __aicore__ inline void Compute(int32_t batchLoop, int32_t dimLoop, const LocalTensor<int32_t> &indicesLocal,
        const LocalTensor<int32_t> &acceptTokenLocal, const LocalTensor<int32_t> &queryStartLocLocal,
        uint16_t yBlockCount, int64_t yOffset);

    __aicore__ inline void UpdateConvStates(const LocalTensor<T> &xLocal, const LocalTensor<T> &convStatesLocal1,
        int32_t acceptToken, int32_t curBatchUbOffset, int64_t convStatesIdx, int32_t curBatchSeq);

    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }
    // ========== Global Memory指针 ==========
    GlobalTensor<T> xGm;                     // 输入序列 [batch, seqLen, dim]
    GlobalTensor<T> weightGm;                // 卷积核 [K, dim]
    GlobalTensor<T> convStatesGm;            // 输入cache state [-1, K-1+m, dim]
    GlobalTensor<int32_t> cacheIndicesGm;    // cache索引 [batch]
    GlobalTensor<int32_t> acceptTokenNumGm;  // 接受的token数 [batch]
    GlobalTensor<int32_t> queryStartLocGm;   // query起始位置 [batch+1]
    GlobalTensor<T> yGm;                     // 输出序列 [batch, seqLen, dim]

    // ========== UB队列（Unified Buffer中的数据缓存） ==========
    TQue<QuePosition::VECIN, 1> cacheQueueIn;
    TQue<QuePosition::VECOUT, 1> cacheQueueOut;
    TQue<QuePosition::VECIN, 1> weightQueue;         // 卷积核队列（单Buffer）
    TQue<QuePosition::VECIN, 1> indicesQueue;        // cache索引队列（单Buffer）
    TQue<QuePosition::VECIN, 1> acceptTokenQueue;    // accept token队列（单Buffer）
    TQue<QuePosition::VECIN, 1> queryStartLocQueue;  // query起始位置队列（单Buffer）
    TQue<QuePosition::VECIN, 1> xQueue;              // 输入x队列（双Buffer）
    TQue<QuePosition::VECOUT, 1> yQueue;

    TPipe *pipe_;  // Pipeline管理对象

    // ========== Tiling参数（二维切分：Dim方向 × Batch方向） ==========
    // 核间切分参数
    int64_t dimCoreCnt_;      // dim方向的核数
    int64_t dimMainCoreCnt_;  // dim方向的主核个数
    int64_t mainCoredimLen_;  // 主核处理的Dim大小（128对齐
    int64_t tailCoredimLen_;  // 主核处理的Dim大小

    int64_t batchMainCoreCnt_;  // batch方向的主核个数
    int64_t mainCoreBatchNum_;  // 主核bacth个数
    int64_t tailCoreBatchNum_;  // 尾核batch个数

    // 核内UB切分参数
    int64_t ubMainFactorBS_;   // 主UB循环处理的batch数
    int64_t ubTailFactorBS_;   // 尾UB循环处理的batch数
    int64_t ubMainFactorDim_;  // 主UB循环处理的dim数
    int64_t ubTailFactorDim_;  // 尾UB循环处理的dim数

    int64_t loopNumBS_;   // Batch方向的循环次数
    int64_t loopNumDim_;  // Dim方向的循环次数

    // ========== Shape参数 ==========
    int64_t batchSize_;   // Batch大小
    int64_t seqLen_;      // 序列长度（m+1）
    int64_t cuSeqLen_;    // 累积序列长度（2D输入用）
    int64_t dim_;         // 特征维度
    int64_t kernelSize_;  // 卷积核大小（K=3）
    int64_t cacheLen_;    // cache_state第二维
    int64_t xInputMode_;  // 输入模式：0=3D, 1=2D

    // ========== 运行时计算参数 ==========
    int32_t blockIdx_;           // 当前核的索引
    int32_t batchIdx_;           // 当前核在Batch维度的索引
    int32_t dimIdx_;             // 当前核在Dim维度的索引
    int32_t firstBatchIdx_;      // 当前核处理的起始batch索引
    int32_t coreBatchNum_;       // 当前核处理的batch数量
    int32_t dimOffset_;          // 当前核处理的Dim起始偏移（元素）
    int32_t coreDimLen_;         // 当前核处理的Dim大小
    int64_t hasAcceptTokenNum_;  // 是否提供了acceptTokenNum输入
    int64_t dimSum_;             // x每个token间的stride
    int64_t cacheLenSum_;        // cache每个token间的stride
    int64_t cacheBatchLenSum_;   // cache每个bacth间的stride
    int32_t batchNumInLoop_;
    int32_t dimSizeInLoop_;
    int32_t dimOffsetPerLoop_;
    int64_t isResidualConnection_;
    int8_t padSlotId_;
};

// ==================== 函数实现 ====================

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates,
    GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR numAcceptedToken, GM_ADDR y,
    const FusedCausalConv1dCutBHTilingData *tilingData)
{
    InitParams(tilingData);

    // === 设置Global Memory buffers ===
    if (xInputMode_ == 0) {
        xGm.SetGlobalBuffer((__gm__ T *)x, batchSize_ * seqLen_ * dimSum_);
        yGm.SetGlobalBuffer((__gm__ T *)y, batchSize_ * seqLen_ * dim_);
    } else {
        xGm.SetGlobalBuffer((__gm__ T *)x, cuSeqLen_ * dimSum_);
        yGm.SetGlobalBuffer((__gm__ T *)y, cuSeqLen_ * dim_);
    }
    weightGm.SetGlobalBuffer((__gm__ T *)weight, kernelSize_ * dim_);
    convStatesGm.SetGlobalBuffer((__gm__ T *)convStates);
    cacheIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cacheIndices, batchSize_);
    if (hasAcceptTokenNum_ == 1) {
        acceptTokenNumGm.SetGlobalBuffer((__gm__ int32_t *)numAcceptedToken, batchSize_);
    }
    if (xInputMode_ == 1) {
        queryStartLocGm.SetGlobalBuffer((__gm__ int32_t *)queryStartLoc, batchSize_ + 1);
    }

    InitQueues();
}

// ==================== Init子函数实现 ====================

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
    kernelSize_ = tilingData->kernelSize;
    cacheLen_ = tilingData->stateLen;
    hasAcceptTokenNum_ = tilingData->hasAcceptTokenNum;
    xInputMode_ = tilingData->xInputMode;
    isResidualConnection_ = tilingData->residualConnection;
    padSlotId_ = tilingData->padSlotId;

    // === stride参数 ===
    dimSum_ = tilingData->xStride;
    cacheLenSum_ = tilingData->cacheStride1;
    cacheBatchLenSum_ = tilingData->cacheStride0;

    // === 当前核的参数计算 ===
    blockIdx_ = GetBlockIdx();
    batchIdx_ = blockIdx_ / dimCoreCnt_;            // Batch方向索引
    dimIdx_ = blockIdx_ - batchIdx_ * dimCoreCnt_;  // Dim方向索引
    if (dimIdx_ < dimMainCoreCnt_) {
        loopNumDim_ = tilingData->loopNumDim;
        coreDimLen_ = mainCoredimLen_;
        ubMainFactorDim_ = tilingData->ubMainFactorDim;
        ubTailFactorDim_ = tilingData->ubTailFactorDim;
        dimOffset_ = dimIdx_ * mainCoredimLen_;
    } else {
        loopNumDim_ = tilingData->tailBlockloopNumDim;
        coreDimLen_ = tailCoredimLen_;
        ubMainFactorDim_ = tilingData->tailBlockubFactorDim;
        ubTailFactorDim_ = tilingData->tailBlockubTailFactorDim;
        dimOffset_ = dimMainCoreCnt_ * mainCoredimLen_ + (dimIdx_ - dimMainCoreCnt_) * tailCoredimLen_;
    }
    if (batchIdx_ < batchMainCoreCnt_) {
        loopNumBS_ = tilingData->loopNumBS;
        ubMainFactorBS_ = tilingData->ubMainFactorBS;
        ubTailFactorBS_ = tilingData->ubTailFactorBS;
        coreBatchNum_ = mainCoreBatchNum_;
        firstBatchIdx_ = batchIdx_ * mainCoreBatchNum_;
    } else {
        loopNumBS_ = tilingData->tailBlockloopNumBS;
        ubMainFactorBS_ = tilingData->tailBlockubFactorBS;
        ubTailFactorBS_ = tilingData->tailBlockubTailFactorBS;
        coreBatchNum_ = tailCoreBatchNum_;
        firstBatchIdx_ = batchMainCoreCnt_ * mainCoreBatchNum_ + (batchIdx_ - batchMainCoreCnt_) * tailCoreBatchNum_;
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::InitQueues()
{
    // xQueue: 存储输入x数据（y复用此buffer）
    int32_t xQueueSize = ubMainFactorBS_ * seqLen_ * ubMainFactorDim_ * sizeof(T);
    if (xInputMode_ == 1) {
        xQueueSize = ubMainFactorBS_ * MAX_SEQUENCE_LEN * ubMainFactorDim_ * sizeof(T);
    }
    pipe_->InitBuffer(xQueue, BUFFER_NUM, xQueueSize);
    pipe_->InitBuffer(yQueue, BUFFER_NUM, xQueueSize);

    // cacheQueue: 存储cache state
    int32_t cacheQueueSize = cacheLen_ * ubMainFactorDim_ * sizeof(T);
    pipe_->InitBuffer(cacheQueueIn, BUFFER_NUM, cacheQueueSize);
    pipe_->InitBuffer(cacheQueueOut, BUFFER_NUM, cacheQueueSize);

    // weightQueue: 存储卷积核 [K, ubDimSize]
    int32_t weightQueueSize = kernelSize_ * ubMainFactorDim_ * sizeof(T);
    pipe_->InitBuffer(weightQueue, 1, weightQueueSize);

    // indicesQueue: 存储cache索引
    int32_t indicesQueueSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    pipe_->InitBuffer(indicesQueue, 1, indicesQueueSize);

    // acceptTokenQueue: 存储accept token数量
    int32_t acceptTokenQueueSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    pipe_->InitBuffer(acceptTokenQueue, 1, acceptTokenQueueSize);

    // queryStartLocQueue: 只有二维TH的时候，存储query起始位置
    if (xInputMode_) {
        int32_t queryStartLocQueueSize =
            ((batchSize_ + 1) * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        pipe_->InitBuffer(queryStartLocQueue, 1, queryStartLocQueueSize);
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::Process()
{
    // 预先搬运cache indices、accept token numbers和query start loc到UB
    // 这些是常驻数据，整个Process过程中都需要访问
    LocalTensor<int32_t> indicesLocal = indicesQueue.AllocTensor<int32_t>();
    LocalTensor<int32_t> acceptTokenLocal = acceptTokenQueue.AllocTensor<int32_t>();
    LocalTensor<int32_t> queryStartLocLocal;

    DataCopyPadParams padParams{false, 0, 0, 0};
    uint32_t indicesBlockLen = batchSize_ * sizeof(int32_t);
    DataCopyParams indicesCopyParams;
    indicesCopyParams.blockCount = 1;
    indicesCopyParams.blockLen = indicesBlockLen;
    indicesCopyParams.srcStride = 0;
    indicesCopyParams.dstStride = 0;
    DataCopyPad(indicesLocal, cacheIndicesGm, indicesCopyParams, padParams);

    Duplicate(acceptTokenLocal, static_cast<int32_t>(1), batchSize_);
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    if (hasAcceptTokenNum_ == 1) {
        DataCopyPad(acceptTokenLocal, acceptTokenNumGm, indicesCopyParams, padParams);
    }

    // 拷贝query start loc：[batch+1]
    if (xInputMode_ == 1) {
        queryStartLocLocal = queryStartLocQueue.AllocTensor<int32_t>();
        uint32_t queryStartLocBlockLen = (batchSize_ + 1) * sizeof(int32_t);
        DataCopyParams queryStartLocCopyParams;
        queryStartLocCopyParams.blockCount = 1;
        queryStartLocCopyParams.blockLen = queryStartLocBlockLen;
        queryStartLocCopyParams.srcStride = 0;
        queryStartLocCopyParams.dstStride = 0;
        DataCopyPad(queryStartLocLocal, queryStartLocGm, queryStartLocCopyParams, padParams);
    }

    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    for (int32_t batchLoop = 0; batchLoop < loopNumBS_; batchLoop++) {
        for (int32_t dimLoop = 0; dimLoop < loopNumDim_; dimLoop++) {
            batchNumInLoop_ = (batchLoop == loopNumBS_ - 1) ? ubTailFactorBS_ : ubMainFactorBS_;
            dimSizeInLoop_ = (dimLoop == loopNumDim_ - 1) ? ubTailFactorDim_ : ubMainFactorDim_;
            dimOffsetPerLoop_ = dimLoop * ubMainFactorDim_;
            int32_t firstBatchIdxInLoop = firstBatchIdx_ + batchLoop * ubMainFactorBS_;
            uint16_t blockCount = batchNumInLoop_ * seqLen_;
            int64_t startSeqIdx = firstBatchIdxInLoop * seqLen_;
            int64_t yOffset = firstBatchIdxInLoop * seqLen_ * dim_ + dimOffset_ + dimOffsetPerLoop_;
            if (xInputMode_ == 1) {
                blockCount = queryStartLocLocal.GetValue(firstBatchIdxInLoop + batchNumInLoop_) -
                             queryStartLocLocal.GetValue(firstBatchIdxInLoop);
                startSeqIdx = queryStartLocLocal.GetValue(firstBatchIdxInLoop) - queryStartLocLocal.GetValue(0);
                yOffset = startSeqIdx * dim_ + dimOffset_ + dimOffsetPerLoop_;
            }
            int64_t xOffset = startSeqIdx * dimSum_ + dimOffset_ + dimOffsetPerLoop_;

            // printf("blockCount %d, xOffset %d", blockCount, xOffset);
            CopyIn(batchLoop, dimLoop, blockCount, xOffset);
            Compute(batchLoop, dimLoop, indicesLocal, acceptTokenLocal, queryStartLocLocal, blockCount, yOffset);
        }
    }

    // 释放indices、acceptToken和queryStartLoc tensors
    indicesQueue.FreeTensor(indicesLocal);
    acceptTokenQueue.FreeTensor(acceptTokenLocal);
    if (xInputMode_ == 1) {
        queryStartLocQueue.FreeTensor(queryStartLocLocal);
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::CopyIn(
    int32_t batchLoop, int32_t dimLoop, uint16_t blockCount, int64_t xOffset)
{
    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    LocalTensor<T> weightLocal = weightQueue.AllocTensor<T>();
    // === 计算当前循环处理的batch数和dim大小 ===

    // === 拷贝x数据：[blockCount, dimSizeInLoop] ===
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;  // 一次性搬运batchNumInLoop * seqLen行
    dataCopyParams.blockLen = dimSizeInLoop_ * sizeof(T);
    dataCopyParams.srcStride = (dimSum_ - dimSizeInLoop_) * sizeof(T);
    dataCopyParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[xOffset], dataCopyParams, padParams);

    // === 拷贝weight数据：[K, dimSizeInLoop] ===
    // weight的全局偏移 = dimOffset_ + dimOffsetPerLoop_
    int32_t weightOffset = dimOffset_ + dimOffsetPerLoop_;
    DataCopyParams weightCopyParams;
    weightCopyParams.blockCount = kernelSize_;
    weightCopyParams.blockLen = dimSizeInLoop_ * sizeof(T);
    weightCopyParams.srcStride = (dim_ - dimSizeInLoop_) * sizeof(T);
    weightCopyParams.dstStride = 0;
    DataCopyPad(weightLocal, weightGm[weightOffset], weightCopyParams, padParams);

    // === 将tensors入队 ===
    xQueue.EnQue(xLocal);
    weightQueue.EnQue(weightLocal);
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::Compute(int32_t batchLoop, int32_t dimLoop,
    const LocalTensor<int32_t> &indicesLocal, const LocalTensor<int32_t> &acceptTokenLocal,
    const LocalTensor<int32_t> &queryStartLocLocal, uint16_t yBlockCount, int64_t yOffset)
{
    LocalTensor<T> xLocal = xQueue.DeQue<T>();
    LocalTensor<T> weightLocal = weightQueue.DeQue<T>();
    LocalTensor<T> yLocal = yQueue.AllocTensor<T>();

    int32_t batchNumPerLoop = batchLoop * ubMainFactorBS_;
    int64_t yUbOffset = 0;
    DataCopyPadParams padParams{false, 0, 0, 0};
    for (int32_t b = 0; b < batchNumInLoop_; b++) {
        LocalTensor<T> convStatesLocalIn = cacheQueueIn.AllocTensor<T>();
        int32_t curBatchIdx = firstBatchIdx_ + batchNumPerLoop + b;
        // 当前state的idx
        int64_t convStatesIdx = indicesLocal.GetValue(curBatchIdx);
        if (convStatesIdx == padSlotId_) {
            continue;
        }
        int32_t acceptToken = acceptTokenLocal.GetValue(curBatchIdx);
        int32_t convStatesGmOffset = convStatesIdx * cacheBatchLenSum_ + dimOffset_ + dimOffsetPerLoop_;

        // === 拷贝convStates数据：[, dimSizeInLoop] ===
        DataCopyParams cacheCopyParams;
        cacheCopyParams.blockCount = cacheLen_;
        cacheCopyParams.blockLen = dimSizeInLoop_ * sizeof(T);
        cacheCopyParams.srcStride = (cacheLenSum_ - dimSizeInLoop_) * sizeof(T);
        cacheCopyParams.dstStride = 0;
        DataCopyPad(
            convStatesLocalIn, convStatesGm[convStatesGmOffset], cacheCopyParams, padParams);  // convStates GM->UB
        cacheQueueIn.EnQue<T>(convStatesLocalIn);
        convStatesLocalIn = cacheQueueIn.DeQue<T>();

        LocalTensor<T> convStatesLocalOut = cacheQueueOut.AllocTensor<T>();
        DataCopy(convStatesLocalOut, convStatesLocalIn, cacheLen_ * dimSizeInLoop_);
        cacheQueueOut.EnQue<T>(convStatesLocalOut);

        //=== 更新cachestate ===
        int32_t curBatchSeq = seqLen_;
        int32_t curBatchUbOffset = b * curBatchSeq * dimSizeInLoop_;
        if (xInputMode_ == 1) {
            curBatchSeq = queryStartLocLocal.GetValue(curBatchIdx + 1) - queryStartLocLocal.GetValue(curBatchIdx);
            curBatchUbOffset =
                (queryStartLocLocal.GetValue(curBatchIdx) - queryStartLocLocal.GetValue(curBatchIdx - b)) *
                dimSizeInLoop_;
        }
        convStatesLocalOut = cacheQueueOut.DeQue<T>();
        UpdateConvStates(xLocal, convStatesLocalOut, acceptToken, curBatchUbOffset, convStatesIdx, curBatchSeq);
        cacheQueueOut.FreeTensor<T>(convStatesLocalOut);
        // 情况A：序列位置 j ∈ [0, K-2]，更新到yub
        LocalTensor<T> xSlice = xLocal[curBatchUbOffset];
        LocalTensor<T> stateSlice = convStatesLocalIn[(acceptToken - 1) * dimSizeInLoop_];
        LocalTensor<T> y1Slice = yLocal[yUbOffset];
        Conv1dNeedStateBH(
            xSlice, weightLocal, stateSlice, y1Slice, kernelSize_ - 1, dimSizeInLoop_, isResidualConnection_);
        cacheQueueIn.FreeTensor<T>(convStatesLocalIn);

        // 情况B：序列位置 j ∈ [K-1, curBatchSeq-1]，只使用x数据
        int16_t blockCount = curBatchSeq - kernelSize_ + 1;
        if (blockCount > 0) {
            uint8_t xSLen = static_cast<uint8_t>(blockCount);
            LocalTensor<T> y2Slice = yLocal[yUbOffset + (kernelSize_ - 1) * dimSizeInLoop_];
            Conv1dNoNeedState(xSlice, weightLocal, y2Slice, xSLen, dimSizeInLoop_, isResidualConnection_);
        }
        yUbOffset += curBatchSeq * dimSizeInLoop_;
    }
    yQueue.EnQue<T>(yLocal);
    yLocal = yQueue.DeQue<T>();
    DataCopyParams yParams;
    yParams.blockCount = yBlockCount;
    yParams.blockLen = dimSizeInLoop_ * sizeof(T);
    yParams.srcStride = 0;
    yParams.dstStride = (dim_ - dimSizeInLoop_) * sizeof(T);
    DataCopyPad(yGm[yOffset], yLocal, yParams);
    // === 6. 释放输入tensors ===
    xQueue.FreeTensor(xLocal);
    weightQueue.FreeTensor(weightLocal);
    yQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBH<T>::UpdateConvStates(const LocalTensor<T> &xLocal,
    const LocalTensor<T> &convStatesLocal1, int32_t acceptToken, int32_t curBatchUbOffset, int64_t convStatesIdx,
    int32_t curBatchSeq)
{
    int64_t convStatesGmOffset = convStatesIdx * cacheBatchLenSum_ + dimOffset_ + dimOffsetPerLoop_;
    uint32_t blockLen = dimSizeInLoop_ * sizeof(T);
    uint32_t dstStrideBytes = (cacheLenSum_ - dimSizeInLoop_) * sizeof(T);
    // === 步骤1：拷贝旧cache state的后cacheLen - seqLen_行（如果需要） ===
    int32_t convStatesNeedRow = kernelSize_ - 2;
    int32_t xBlockCount = curBatchSeq;
    int32_t xUbOffset = curBatchUbOffset;
    int64_t xToCacheOffset = convStatesGmOffset + convStatesNeedRow * cacheLenSum_;
    if (curBatchSeq + kernelSize_ - 2 > cacheLen_) {
        convStatesNeedRow = cacheLen_ - curBatchSeq;
        xBlockCount = cacheLen_;
        xUbOffset = curBatchUbOffset + (curBatchSeq - xBlockCount) * dimSizeInLoop_;
        xToCacheOffset = convStatesGmOffset;
    }
    if (convStatesNeedRow > 0) {
        int32_t srcCacheOffset = (acceptToken)*dimSizeInLoop_;
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = convStatesNeedRow;
        dataCopyParams.blockLen = blockLen;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = dstStrideBytes;
        DataCopyPad(convStatesGm[convStatesGmOffset], convStatesLocal1[srcCacheOffset], dataCopyParams);
    }
    // // === 步骤2：拷贝x的所有行到cache state ===
    DataCopyParams xToCacheCopyParams;
    xToCacheCopyParams.blockCount = xBlockCount;
    xToCacheCopyParams.blockLen = blockLen;
    xToCacheCopyParams.srcStride = 0;
    xToCacheCopyParams.dstStride = dstStrideBytes;
    DataCopyPad(convStatesGm[xToCacheOffset], xLocal[xUbOffset], xToCacheCopyParams);
}

#endif  // CAUSAL_CONV1D_CUT_BH_H
