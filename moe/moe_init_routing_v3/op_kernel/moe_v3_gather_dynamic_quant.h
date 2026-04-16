/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_gather_dynamic_quant.h
 * \brief
 */
#ifndef MOE_V3_GATHER_DYNAMIC_QUANT_H
#define MOE_V3_GATHER_DYNAMIC_QUANT_H

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;
constexpr int64_t GATHER_OUT_DYNAMIC_QUANT_BUFFER_NUM = 2;

template <typename T>
class MoeGatherOutDynamicQuant {
public:
    __aicore__ inline MoeGatherOutDynamicQuant(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR quantSmooth, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR expandedScale, GM_ADDR sortedExpertIdx,
                                const MoeInitRoutingV3TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyOutXQuantEH(int64_t progress);
    __aicore__ inline void Compute(LocalTensor<float> &smoothLocal);
    __aicore__ inline void CopyOutPartialXQuantEH(int64_t progress);
    __aicore__ inline float ComputeMax(LocalTensor<float> &inLocal, LocalTensor<float> &tempLocal,
                                       LocalTensor<float> &scaleLocal, int32_t srcIdx, int32_t expertIdx, int64_t j);
    __aicore__ inline void ComputeScale(LocalTensor<float> &inLocal, LocalTensor<float> &tempLocal, float scaleTemp,
                                        int64_t dstIndex, int64_t j);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> inputXInQueue_;
    TQue<QuePosition::VECIN, 1> smoothInQueue_;
    TQue<QuePosition::VECIN, 1> expandRowIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> calcQueue_;
    TQue<QuePosition::VECOUT, 1> inputXOutQueue_;
    TQue<QuePosition::VECOUT, 1> scaleOutQueue_;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> quantSmoothGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<float> quantTempGm_;
    GlobalTensor<int32_t> expandedExpertIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    const MoeV3GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;
    int64_t totalLength_;
    int64_t perCoreRow_;
    int64_t currentLoopRows_;
    int64_t currentLoopRowsAlign_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t colsTileLength_;
    int64_t perLoopCols_;
    int64_t perLoopColsAlign_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t isInputScale_;
    int64_t expertStart_;

    int64_t indicesOffset_;
    int64_t rowIdxType_ = 0;
};

template <typename T>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::CopyInExpandedExpertIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    DataCopyPad(indicesLocal[currentLoopRowsAlign_], expandedExpertIdxGm_[indicesOffset_], dataCopyParams,
                dataCopyPadParams);
    expandRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::Compute(LocalTensor<float> &smoothLocal)
{
    LocalTensor<float> inLocal = inputXInQueue_.DeQue<float>();

    LocalTensor<float> tempLocal = calcQueue_.AllocTensor<float>();
    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();
    LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();

    if constexpr (!IsSameType<T, float>::value) {
        Cast(inLocal, inLocal.ReinterpretCast<T>()[perLoopColsAlign_], RoundMode::CAST_NONE, cols_);
        PipeBarrier<PIPE_V>();
    }

    if (isInputScale_) {
        Mul(inLocal, inLocal, smoothLocal, cols_);
        PipeBarrier<PIPE_V>();
    }

    Abs(tempLocal, inLocal, cols_);
    PipeBarrier<PIPE_V>();

    ReduceMax(scaleLocal, tempLocal, tempLocal, cols_); // get max value and index [0,1]

    float scaleValue = scaleLocal.GetValue(0) / 127.0f;

    Duplicate<float>(scaleLocal, scaleValue, 8);
    Duplicate<float>(tempLocal, scaleValue, cols_);
    PipeBarrier<PIPE_V>();

    Div(tempLocal, inLocal, tempLocal, cols_);
    PipeBarrier<PIPE_V>();

    Cast(tempLocal.ReinterpretCast<half>(), tempLocal, RoundMode::CAST_TRUNC, cols_); // fp32->fp16
    PipeBarrier<PIPE_V>();

    Cast(outLocal, tempLocal.ReinterpretCast<half>(), RoundMode::CAST_ROUND, cols_); // fp16->int8

    calcQueue_.FreeTensor(tempLocal);
    inputXOutQueue_.EnQue(outLocal);
    scaleOutQueue_.EnQue(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::CopyOutXQuantEH(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams smoothParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(int8_t)), 0, 0, 0};

    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();
    int32_t lastExpertIdx = -1;
    for (int64_t i = 0; i < currentLoopRows_; i++) {
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        int64_t rowOffset = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign_ + i) - expertStart_;
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, inputXGm_[srcIdx / k_ * cols_], copyInParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(inLocal[perLoopColsAlign_], inputXGm_[srcIdx / k_ * cols_], copyInParams, {false, 0, 0, 0});
        }
        inputXInQueue_.EnQue<T>(inLocal);

        if (isInputScale_ && expertIdx != lastExpertIdx) {
            DataCopyPad(smoothLocal, quantSmoothGm_[expertIdx * cols_], smoothParams, {false, 0, 0, 0});
            smoothInQueue_.EnQue(smoothLocal);
            smoothLocal = smoothInQueue_.DeQue<float>();
            lastExpertIdx = expertIdx;
        }
        Compute(smoothLocal);
        inputXInQueue_.FreeTensor(inLocal);
        LocalTensor<float> scaleLocal = scaleOutQueue_.DeQue<float>();
        DataCopyPad(expandedScaleGm_[(rowOffset + i)], scaleLocal, {1, 4, 0, 0, 0});
        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        DataCopyPad(expandedXGm_[(rowOffset + i) * cols_], outLocal, copyOutParams);

        inputXOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(scaleLocal);
    }

    smoothInQueue_.FreeTensor(smoothLocal);
    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline float
MoeGatherOutDynamicQuant<T>::ComputeMax(LocalTensor<float> &inLocal, LocalTensor<float> &tempLocal,
                                        LocalTensor<float> &scaleLocal, int32_t srcIdx, int32_t expertIdx, int64_t j)
{
    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();

    DataCopyExtParams intriParamsT{1, static_cast<uint32_t>(colsTileLength_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams intriParamsFp32{1, static_cast<uint32_t>(colsTileLength_ * sizeof(float)), 0, 0, 0};

    if constexpr (!IsSameType<T, float>::value) {
        DataCopyPad(inLocal.ReinterpretCast<T>()[perLoopColsAlign_], inputXGm_[srcIdx * cols_ + j * perLoopCols_],
                    intriParamsT, {false, 0, 0, 0});
    } else {
        DataCopyPad(inLocal, inputXGm_[srcIdx * cols_ + j * perLoopCols_], intriParamsT, {false, 0, 0, 0});
    }

    inputXInQueue_.EnQue<float>(inLocal);
    inLocal = inputXInQueue_.DeQue<float>();

    if (isInputScale_) {
        DataCopyPad(smoothLocal, quantSmoothGm_[expertIdx * cols_ + j * perLoopCols_], intriParamsFp32,
                    {false, 0, 0, 0});
        smoothInQueue_.EnQue(smoothLocal);
        smoothLocal = smoothInQueue_.DeQue<float>();
    }

    if constexpr (!IsSameType<T, float>::value) {
        Cast(inLocal, inLocal.ReinterpretCast<T>()[perLoopColsAlign_], RoundMode::CAST_NONE, colsTileLength_);
        PipeBarrier<PIPE_V>();
    }

    if (isInputScale_) {
        Mul(inLocal, inLocal, smoothLocal, colsTileLength_);
        PipeBarrier<PIPE_V>();
    }

    Abs(tempLocal, inLocal, colsTileLength_);
    PipeBarrier<PIPE_V>();

    ReduceMax(scaleLocal[8], tempLocal, tempLocal, colsTileLength_);

    DataCopyPad(quantTempGm_[j * perLoopCols_], inLocal, intriParamsFp32);
    smoothInQueue_.FreeTensor(smoothLocal);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    return scaleLocal.GetValue(8);
}

template <typename T>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::ComputeScale(LocalTensor<float> &inLocal,
                                                                 LocalTensor<float> &tempLocal, float scaleTemp,
                                                                 int64_t dstIndex, int64_t j)
{
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(int8_t)), 0, 0, 0};

    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();

    DataCopyPad(inLocal, quantTempGm_[j * perLoopCols_], copyInParams, {false, 0, 0, 0});
    inputXInQueue_.EnQue<float>(inLocal);
    inLocal = inputXInQueue_.DeQue<float>();

    Duplicate<float>(tempLocal, scaleTemp, colsTileLength_);
    PipeBarrier<PIPE_V>();

    Div(tempLocal, inLocal, tempLocal, colsTileLength_);
    PipeBarrier<PIPE_V>();

    Cast(tempLocal.ReinterpretCast<half>(), tempLocal, RoundMode::CAST_TRUNC, colsTileLength_);
    PipeBarrier<PIPE_V>();

    Cast(outLocal, tempLocal.ReinterpretCast<half>(), RoundMode::CAST_ROUND, colsTileLength_);

    inputXOutQueue_.EnQue(outLocal);
    outLocal = inputXOutQueue_.DeQue<int8_t>();
    DataCopyPad(expandedXGm_[dstIndex * cols_ + j * perLoopCols_], outLocal, copyOutParams);

    inputXOutQueue_.FreeTensor(outLocal);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::CopyOutPartialXQuantEH(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    for (int64_t i = 0; i < currentLoopRows_; i++) {
        int64_t rowOffset = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign_ + i) - expertStart_;

        LocalTensor<float> inLocal = inputXInQueue_.AllocTensor<float>();
        LocalTensor<float> tempLocal = calcQueue_.AllocTensor<float>();
        LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();

        uint32_t tmp = 0xFF7FFFFF;
        float reduceMax = *((float *)&tmp);
        for (int64_t j = 0; j < colLoops_; j++) {
            colsTileLength_ = perLoopCols_;
            if (j == colLoops_ - 1) {
                colsTileLength_ = lastLoopCols_;
            }
            float tileMax = ComputeMax(inLocal, tempLocal, scaleLocal, srcIdx / k_, expertIdx, j);
            reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
        }

        float scaleTemp = reduceMax / 127.0f;
        Duplicate<float>(scaleLocal, scaleTemp, 8);
        scaleOutQueue_.EnQue(scaleLocal);
        scaleLocal = scaleOutQueue_.DeQue<float>();

        DataCopyPad(expandedScaleGm_[(rowOffset + i)], scaleLocal, {1, 4, 0, 0, 0});

        for (int64_t j = 0; j < colLoops_; j++) {
            colsTileLength_ = perLoopCols_;
            if (j == colLoops_ - 1) {
                colsTileLength_ = lastLoopCols_;
            }
            ComputeScale(inLocal, tempLocal, scaleTemp, rowOffset + i, j);
        }
        inputXInQueue_.FreeTensor(inLocal);
        calcQueue_.FreeTensor(tempLocal);
        scaleOutQueue_.FreeTensor(scaleLocal);
    }
    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::Init(GM_ADDR inputX, GM_ADDR quantSmooth, GM_ADDR sortedExpertIdx,
                                                         GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                                         GM_ADDR expandedScale,
                                                         const MoeInitRoutingV3TilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;
    totalLength_ = n_ * k_;
    isInputScale_ = tilingData->isInputScale;
    expertStart_ = tilingData->expertStart;
    rowIdxType_ = tilingData->rowIdxType;

    // core split
    int64_t actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)),
                                        1);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        expertTotalCountGm_);
    int64_t expertTotalCount_ = expertTotalCountGm_.GetValue(0);
    perCoreRow_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreRow_);
    int64_t lastCoreIndicesElements = expertTotalCount_ - (needCoreNum_ - 1) * perCoreRow_;

    // inner core split
    int64_t originPerLoopElements;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements;
        originPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    } else {
        coreRows_ = perCoreRow_;
        originPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;
    }
    perLoopRows_ = Min(coreRows_, originPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;

    // cols split
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;

    perLoopColsAlign_ = Align(perLoopCols_, sizeof(T));

    inputXGm_.SetGlobalBuffer((__gm__ T *)inputX);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);

    expandedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + blockIdx_ * perCoreRow_,
                                         Align(coreRows_, sizeof(int32_t)));

    if (rowIdxType_ == SCATTER) {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreRow_,
                                          Align(perCoreRow_, sizeof(int32_t)));
    } else {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + Align(n_ * k_, sizeof(int32_t)) +
                                              blockIdx_ * perCoreRow_,
                                          Align(perCoreRow_, sizeof(int32_t)));
    }

    if (isInputScale_) {
        quantSmoothGm_.SetGlobalBuffer((__gm__ float *)quantSmooth);
    }
    expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);

    if (colLoops_ > 1) {
        // cols非全载 smooth*x结果临时存储
        quantTempGm_.SetGlobalBuffer((__gm__ float *)sortedExpertIdx + Align(totalLength_, sizeof(int32_t)) * 2 +
                                         Align(actualExpertNum_, sizeof(int32_t)) + Align(1, sizeof(int32_t)) +
                                         blockIdx_ * cols_,
                                     cols_ * sizeof(float));
    }

    currentLoopRowsAlign_ = Align(perLoopRows_, sizeof(int32_t));

    int64_t perLoopColsAlignBytes = AlignBytes(this->perLoopCols_, sizeof(T));
    perLoopColsAlignBytes =
        Max(int64_t(perLoopColsAlignBytes * sizeof(float) / sizeof(T)), int64_t(BLOCK_BYTES + BLOCK_BYTES));
    pipe_->InitBuffer(expandRowIdxInQueue_, GATHER_OUT_DYNAMIC_QUANT_BUFFER_NUM,
                      2 * AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(inputXInQueue_, GATHER_OUT_DYNAMIC_QUANT_BUFFER_NUM, perLoopColsAlignBytes); // percols * 2 * 4
    pipe_->InitBuffer(smoothInQueue_, GATHER_OUT_DYNAMIC_QUANT_BUFFER_NUM,
                      AlignBytes(perLoopCols_, sizeof(float)));                      // percols * 2 * 4
    pipe_->InitBuffer(calcQueue_, 1, AlignBytes(perLoopCols_, sizeof(float)));       // percols * 1 * 4
    pipe_->InitBuffer(inputXOutQueue_, 1, AlignBytes(perLoopCols_, sizeof(int8_t))); // percols * 1
    pipe_->InitBuffer(scaleOutQueue_, 1, BLOCK_BYTES + BLOCK_BYTES);                 // 32 + 32
}

template <typename T>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        if (colLoops_ > 1) {
            // 一行无法全载，需要workspace
            for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
                CopyInExpandedExpertIdx(loop);
                CopyOutPartialXQuantEH(loop);
            }
            currentLoopRows_ = lastLoopRows_;
            CopyInExpandedExpertIdx(rowLoops_ - 1);
            CopyOutPartialXQuantEH(rowLoops_ - 1);
        } else {
            // 一行可以全载
            for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
                CopyInExpandedExpertIdx(loop);
                CopyOutXQuantEH(loop);
            }
            currentLoopRows_ = lastLoopRows_;
            CopyInExpandedExpertIdx(rowLoops_ - 1);
            CopyOutXQuantEH(rowLoops_ - 1);
        }
    }
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_DYNAMIC_QUANT_H