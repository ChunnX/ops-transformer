/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_KL_LOSS_H
#define LIGHTNING_INDEXER_KL_LOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lightning_indexer_kl_loss_tiling_data.h"
#include "lightning_indexer_kl_loss_tiling_key.h"

namespace NsLightningIndexerKlLoss {

using namespace AscendC;

constexpr uint32_t HALF_BLOCK_SIZE = 16;
constexpr uint32_t HALF_VECTOR_SIZE = 128;
constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
constexpr uint32_t FLOAT_VECTOR_SIZE = 64;

template <typename T>
class LightningIndexerKlLoss {
public:
    __aicore__ inline LightningIndexerKlLoss(){};
    __aicore__ inline void Init(GM_ADDR targetScore, GM_ADDR indexProbs, GM_ADDR loss,
                                GM_ADDR workspace, const LightningIndexerKlLossTilingData* tilingData);
    __aicore__ inline void InitWorkspaceDet(GM_ADDR workspace);
    __aicore__ inline void InitHalfBufs();
    __aicore__ inline void Process(bool isHalf);
    __aicore__ inline void WriteBackNonDet();
    __aicore__ inline void WriteBackDet();

private:
    // 纯 MTE2 搬运
    __aicore__ inline void CopyInFloat(uint32_t tileElems, uint32_t rowOffset, uint32_t curTileLen);
    __aicore__ inline void CopyInHalf(uint32_t tileElems, uint32_t rowOffset, uint32_t curTileLen);
    // 纯 Vector 计算（含 setflag/waitflag）
    __aicore__ inline void Compute(uint32_t tileElems, uint32_t curTileLen, bool isHalf);
    // 纯 Scalar 累加
    __aicore__ inline void AccumTile();

    TPipe pipe;
    TBuf<TPosition::VECCALC> tmpBuf_;
    __gm__ float* perCoreSum_;

    GlobalTensor<T> inputGMY;
    GlobalTensor<T> inputGMIndexProbs;
    GlobalTensor<float> outputGMLoss;

    // UB buffer LocalTensor（使用 TBuf 分配，Init 中初始化）
    LocalTensor<float> ubTargetScoreIn;
    LocalTensor<float> ubIndexProbsIn;
    LocalTensor<float> ubReduceSum;
    LocalTensor<float> ubLogP;
    LocalTensor<float> ubLogY;
    LocalTensor<float> ubOut;
    LocalTensor<float> ubTmp;
    // half 版本：指向 float buffer 后半段的 half 原始数据
    LocalTensor<T> ubHalfTarget;
    LocalTensor<T> ubHalfIndexProbs;

    // Y 和 y 独立流水线 EventID
    event_t eventMte2YToV_;  // MTE2 → Vector: Y 搬运完成
    event_t eventMte2yToV_;  // MTE2 → Vector: y 搬运完成
    event_t eventVToMte2Y_;  // Vector → MTE2: Y 的 UB 已释放
    event_t eventVToMte2y_;  // Vector → MTE2: y 的 UB 已释放
    // PIPE_V → PIPE_S, PIPE_S → PIPE_V
    event_t eventVToS_;
    event_t eventSToV_;

    int64_t totalLength_;
    int64_t K_;
    int64_t KAligned_;
    int64_t halfKAligned_;
    int64_t formerNum_;
    int64_t formerTileNum_;
    int64_t tailTileNum_;
    int64_t tileLength_;
    float eps_;
    int64_t coreNum_;
    int64_t blockIdx_;
    int64_t totalTileNum_;
    int64_t blocksPerCore_;
    float tileResult_;
};

template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::Init(GM_ADDR targetScore, GM_ADDR indexProbs,
    GM_ADDR loss, GM_ADDR workspace, const LightningIndexerKlLossTilingData* tilingData)
{
    totalLength_ = tilingData->totalLength;
    K_ = tilingData->K;
    KAligned_ = tilingData->KAligned;
    halfKAligned_ = (K_ + HALF_BLOCK_SIZE - 1) / HALF_BLOCK_SIZE * HALF_BLOCK_SIZE;
    formerNum_ = tilingData->formerNum;
    formerTileNum_ = tilingData->formerTileNum;
    tailTileNum_ = tilingData->tailTileNum;
    tileLength_ = tilingData->tileLength;
    eps_ = tilingData->eps;
    coreNum_ = tilingData->coreNum;

    inputGMY.SetGlobalBuffer((__gm__ T*)targetScore, totalLength_ * K_);
    inputGMIndexProbs.SetGlobalBuffer((__gm__ T*)indexProbs, totalLength_ * K_);
    outputGMLoss.SetGlobalBuffer((__gm__ float*)loss, 1);

    // 分配 UB buffer：7 块连续 LocalTensor，总大小 = tileLen_ * (5 * KAligned_ + 8) + 8
    uint32_t tileElemsAligned = tileLength_ * KAligned_;
    int64_t totalUbElems = tileLength_ * (5 * KAligned_ + 8) + 8;

    pipe.InitBuffer(tmpBuf_, totalUbElems * sizeof(float));

    uint32_t ubOffset = 0;

    ubTargetScoreIn = tmpBuf_.GetWithOffset<float>(tileElemsAligned, ubOffset);
    ubOffset += tileElemsAligned * sizeof(float);

    ubIndexProbsIn = tmpBuf_.GetWithOffset<float>(tileElemsAligned, ubOffset);
    ubOffset += tileElemsAligned * sizeof(float);

    ubReduceSum = tmpBuf_.GetWithOffset<float>(tileElemsAligned, ubOffset);
    ubOffset += tileElemsAligned * sizeof(float);

    ubLogP = tmpBuf_.GetWithOffset<float>(tileElemsAligned, ubOffset);
    ubOffset += tileElemsAligned * sizeof(float);

    ubLogY = tmpBuf_.GetWithOffset<float>(tileElemsAligned, ubOffset);
    ubOffset += tileElemsAligned * sizeof(float);

    // ubOut: 8 个元素对齐
    uint32_t ubOutElems = 8;
    ubOut = tmpBuf_.GetWithOffset<float>(ubOutElems, ubOffset);
    ubOffset += ubOutElems * sizeof(float);

    // ubTmp 复用 ubIndexProbsIn 位置（Log_Y 读取完后才写 ubTmp，无冲突）
    ubTmp = tmpBuf_.GetWithOffset<float>(tileElemsAligned, static_cast<uint32_t>(tileElemsAligned * sizeof(float)));

    // 分配流水线同步 EventID
    eventMte2YToV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    eventMte2yToV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    eventVToMte2Y_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    eventVToMte2y_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    eventVToS_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    eventSToV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    // 预置所有 flag，使第一轮各 WaitFlag 直接通过
    SetFlag<HardEvent::V_MTE2>(eventVToMte2Y_);
    SetFlag<HardEvent::V_MTE2>(eventVToMte2y_);
    SetFlag<HardEvent::S_V>(eventSToV_);
    SetFlag<HardEvent::V_S>(eventVToS_);

    // 预计算循环控制参数
    blockIdx_ = GetBlockIdx();
    blocksPerCore_ = (blockIdx_ < formerNum_) ? formerTileNum_ : tailTileNum_;
}

/* ---------- InitWorkspaceDet: 初始化 deterministic workspace ---------- */
template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::InitWorkspaceDet(GM_ADDR workspace)
{
    __gm__ float* user = GetUserWorkspace(workspaceGM);
    perCoreSum = user + blockIdx;
}

/* ---------- InitHalfBufs: half 版本 ubHalf 指向 float buffer 后半段 ---------- */
template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::InitHalfBufs()
{
    uint32_t tileElemsAligned = tileLength_ * KAligned_;
    uint32_t halfDword = tileElemsAligned * sizeof(half);
    ubHalfTarget = tmpBuf_.GetWithOffset<T>(tileElemsAligned, halfDword);
    ubHalfIndexProbs = tmpBuf_.GetWithOffset<T>(tileElemsAligned,
        tileElemsAligned * sizeof(float) + halfDword);
}

/* ---------- CopyInFloat: fp32 版本，DataCopyPad 补 0 对齐 ---------- */
template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::CopyInFloat(uint32_t tileElems, uint32_t rowOffset, uint32_t curTileLen)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = curTileLen;
    copyParams.blockLen = K_ * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = KAligned_ - K_;
    padParams.paddingValue = 0;

    WaitFlag<HardEvent::V_MTE2>(eventVToMte2Y_);
    DataCopyPad(ubTargetScoreIn, inputGMY[rowOffset * K_], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(eventMte2YToV_);

    WaitFlag<HardEvent::V_MTE2>(eventVToMte2y_);
    DataCopyPad(ubIndexProbsIn, inputGMIndexProbs[rowOffset * K_], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(eventMte2yToV_);
}

/* ---------- CopyInHalf: half 版本，DataCopy 到 ubHalf 后半段 buffer ---------- */
template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::CopyInHalf(uint32_t tileElems, uint32_t rowOffset, uint32_t curTileLen)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = curTileLen;
    copyParams.blockLen = K_ * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = (HALF_BLOCK_SIZE - K_ % HALF_BLOCK_SIZE) % HALF_BLOCK_SIZE;
    padParams.paddingValue = 0;

    WaitFlag<HardEvent::V_MTE2>(eventVToMte2Y_);
    DataCopyPad(ubHalfTarget, inputGMY[rowOffset * K_], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(eventMte2YToV_);

    WaitFlag<HardEvent::V_MTE2>(eventVToMte2y_);
    DataCopyPad(ubHalfIndexProbs, inputGMIndexProbs[rowOffset * K_], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(eventMte2yToV_);
}

/* ---------- Compute: Vector 全部计算，含 setflag/waitflag ---------- */
template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::Compute(uint32_t tileElems, uint32_t curTileLen, bool isHalf)
{
    uint32_t repeat, remain;
    WaitFlag<HardEvent::MTE2_V>(eventMte2YToV_);
    if (isHalf) {
        repeat = KAligned_ / HALF_VECTOR_SIZE;
        remain = KAligned_ % HALF_VECTOR_SIZE;
        UnaryRepeatParams repeatParams(kAligned_ / FLOAT_BLOCK_SIZE, halfKAligned_ / HALF_BLOCK_SIZE, 1, 1);
        for (int i = 0; i < repeat; ++i) {
            Cast(ubTargetScoreIn[i * HALF_VECTOR_SIZE], ubHalfTarget[i * HALF_VECTOR_SIZE], RoundMode::CAST_NONE, HALF_VECTOR_SIZE, curTileLen, repeatParams);
        }
        if (remain) {
            Cast(ubTargetScoreIn[repeat * HALF_VECTOR_SIZE], ubHalfTarget[repeat * HALF_VECTOR_SIZE], RoundMode::CAST_NONE, remain, curTileLen, repeatParams);
        }
    }
    Adds(ubLogY, ubIndexProbsIn, eps_, tileElems);
    SetFlag<HardEvent::V_MTE2>(eventVToMte2Y_);
    Ln(ubLogY, ubLogY, tileElems);

    WaitFlag<HardEvent::MTE2_V>(eventMte2yToV_);
    if (isHalf) {
        UnaryRepeatParams repeatParams(kAligned_ / FLOAT_BLOCK_SIZE, halfKAligned_ / HALF_BLOCK_SIZE, 1, 1);
        for (int i = 0; i < repeat; ++i) {
            Cast(ubTargetScoreIn[i * HALF_VECTOR_SIZE], ubHalfTarget[i * HALF_VECTOR_SIZE], RoundMode::CAST_NONE, HALF_VECTOR_SIZE, curTileLen, repeatParams);
        }
        if (remain) {
            Cast(ubTargetScoreIn[repeat * HALF_VECTOR_SIZE], ubHalfTarget[repeat * HALF_VECTOR_SIZE], RoundMode::CAST_NONE, remain, curTileLen, repeatParams);
        }
    }
    // CalcLogP: log(clamp_min(y/sum(y), eps)) * y
    for (int i = 0; i < curTileLen; ++i) {
        ReduceSum(ubReduceSum[i * FLOAT_BLOCK_SIZE], ubTargetScoreIn[i * KAligned_], ubTmp, K_);
    }
    Adds(ubReduceSum, ubReduceSum, eps_, 1, curTileLen, {1, 1, 1, 1});
    VectorPadding(ubReduceSum, ubReduceSum, 0, true, FLOAT_BLOCK_SIZE, curTileLen, {1, 1, 1, 1});
    repeat = KAligned_ / FLOAT_VECTOR_SIZE;
    remain = KAligned_ % FLOAT_VECTOR_SIZE;
    for (int i = 0; i < repeat; ++i) {
        Div(ubLogP[i * FLOAT_VECTOR_SIZE], 
            ubTargetScoreIn[i * FLOAT_VECTOR_SIZE], 
            ubReduceSum[i * FLOAT_VECTOR_SIZE], 
            FLOAT_VECTOR_SIZE, 
            curTileLen, 
            {1, 1, 0, KAligned_ / FLOAT_BLOCK_SIZE, KAligned_ / FLOAT_BLOCK_SIZE, FLOAT_BLOCK_SIZE});
    }
    if (remain) {
        Div(ubLogP[repeat * FLOAT_VECTOR_SIZE], 
            ubTargetScoreIn[repeat * FLOAT_VECTOR_SIZE], 
            ubReduceSum[repeat * FLOAT_VECTOR_SIZE], 
            remain, 
            curTileLen, 
            {1, 1, 0, KAligned_ / FLOAT_BLOCK_SIZE, KAligned_ / FLOAT_BLOCK_SIZE, FLOAT_BLOCK_SIZE});
    }
    
    ClampMin(ubLogP, ubLogP, eps_, tileElems);
    Ln(ubLogP, ubLogP, tileElems);
    Sub(ubLogP, ubLogP, ubLogY, tileElems);
    Mul(ubLogP, ubLogP, ubTargetScoreIn, tileElems);
    SetFlag<HardEvent::V_MTE2>(eventVToMte2y_);

    WaitFlag<HardEvent::S_V>(eventSToV_);
    ReduceSum(ubOut, ubLogP, tileElems, tileElems);
    SetFlag<HardEvent::V_S>(eventVToS_);
}

/* ---------- AccumTile: Scalar 单元累加当前 tile 结果 ---------- */
template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::AccumTile()
{
    // 等待 Vector 完成 ReduceSum，ubOut 已就绪
    WaitFlag<HardEvent::V_S>(eventVToS_);

    // ubOut[0] 包含本 tile 的标量和
    tileResult_ += ubOut.GetValue(0);

    // 通知 Vector ubOut 已读取，可安全写入下一轮
    SetFlag<HardEvent::S_V>(eventSToV_);
}

/* ---------- Process: 主循环，isHalf 区分 half/fp32 分支 ---------- */
template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::Process(bool isHalf)
{
    uint32_t tileId, rowOffset, curTileLen, tileElems;
    tileId = 0;
    for (int64_t b = 0; b < blocksPerCore_; b++) {
        tileId += blockIdx_;
        rowOffset = tileId * tileLength_;
        curTileLen = (rowOffset + tileLength_ <= totalLength_)
                         ? tileLength_
                         : (totalLength_ - rowOffset);
        tileElems = curTileLen * KAligned_;

        if (isHalf) {
            CopyInHalf(tileElems, rowOffset, curTileLen);
        } else {
            CopyInFloat(tileElems, rowOffset, curTileLen);
        }

        Compute(tileElems, curTileLen, isHalf);
        AccumTile();
    }
}

/* ---------- WriteBackNonDet: 非确定性结果写回 ---------- */
template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::WriteBackNonDet()
{
    AtomicAdd(outputGMLoss, tileResult_);
}

/* ---------- WriteBackDet: 确定性结果写回 ---------- */
template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::WriteBackDet()
{
    *perCoreSum = tileResult_;
}

} // namespace NsLightningIndexerKlLoss
#endif // LIGHTNING_INDEXER_KL_LOSS_H
