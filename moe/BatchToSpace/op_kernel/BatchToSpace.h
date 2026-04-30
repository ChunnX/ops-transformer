/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file BatchToSpace.h
 * \brief BatchToSpace kernel implementation
 */
#ifndef BATCHTOSPACE_H
#define BATCHTOSPACE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "BatchToSpace_tiling_data.h"
#include "BatchToSpace_tiling_key.h"

namespace NsBatchtospace {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t AXIS_COUNT = 4;
constexpr uint32_t INDEX_N = 0;
constexpr uint32_t INDEX_H = 1;
constexpr uint32_t INDEX_W = 2;
constexpr uint32_t INDEX_C = 3;

template <typename T>
class Batchtospace {
public:
    __aicore__ inline Batchtospace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const BatchtospaceTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint64_t blockIdx, uint64_t loopIdx);
    __aicore__ inline void CopyOut(uint64_t blockIdx, uint64_t loopIdx);
    __aicore__ inline uint64_t ComputeOutOffset(uint64_t blockIdx, uint64_t loopIdx, uint32_t ubAxisOffset);
    __aicore__ inline uint64_t ComputeInOffset(uint64_t outOffset);
    __aicore__ inline uint64_t GetUbFactorCount();
    __aicore__ inline uint32_t GetCurrentUbFactor(uint64_t loopIdx);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> queueIn;
    TQue<QuePosition::VECOUT, BUFFER_NUM> queueOut;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t inShape_[AXIS_COUNT] = {0};
    int64_t outShape_[AXIS_COUNT] = {0};
    int64_t blockSize_ = 0;
    int64_t crops_[2][2] = {{0}};
    uint64_t totalCount_ = 0;
    uint64_t perCoreCount_ = 0;
    uint8_t ubAxis_ = 0;
    uint32_t ubFactor_ = 0;
    uint32_t bufferSize_ = 0;
    
    uint64_t inStride_[AXIS_COUNT] = {0};
    uint64_t outStride_[AXIS_COUNT] = {0};
    uint32_t dtypeSize_ = 0;
};

template <typename T>
__aicore__ inline void Batchtospace<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const BatchtospaceTilingData* tilingData)
{
    for (uint32_t i = 0; i < AXIS_COUNT; i++) {
        inShape_[i] = tilingData->inShape[i];
        outShape_[i] = tilingData->outShape[i];
    }
    blockSize_ = tilingData->blockSize;
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 2; j++) {
            crops_[i][j] = tilingData->crops[i][j];
        }
    }
    totalCount_ = tilingData->totalCount;
    perCoreCount_ = tilingData->perCoreCount;
    ubAxis_ = tilingData->ubAxis;
    ubFactor_ = tilingData->ubFactor;
    bufferSize_ = tilingData->bufferSize;
    
    dtypeSize_ = sizeof(T);
    
    inStride_[INDEX_C] = 1;
    inStride_[INDEX_W] = inShape_[INDEX_C];
    inStride_[INDEX_H] = inShape_[INDEX_W] * inStride_[INDEX_W];
    inStride_[INDEX_N] = inShape_[INDEX_H] * inStride_[INDEX_H];
    
    outStride_[INDEX_C] = 1;
    outStride_[INDEX_W] = outShape_[INDEX_C];
    outStride_[INDEX_H] = outShape_[INDEX_W] * outStride_[INDEX_W];
    outStride_[INDEX_N] = outShape_[INDEX_H] * outStride_[INDEX_H];
    
    inputGM.SetGlobalBuffer((__gm__ T*)x, inShape_[INDEX_N] * inStride_[INDEX_N]);
    outputGM.SetGlobalBuffer((__gm__ T*)y, outShape_[INDEX_N] * outStride_[INDEX_N]);
    
    pipe.InitBuffer(queueIn, BUFFER_NUM, bufferSize_);
    pipe.InitBuffer(queueOut, BUFFER_NUM, bufferSize_);
}

template <typename T>
__aicore__ inline uint64_t Batchtospace<T>::GetUbFactorCount()
{
    return (outShape_[ubAxis_] + ubFactor_ - 1) / ubFactor_;
}

template <typename T>
__aicore__ inline uint32_t Batchtospace<T>::GetCurrentUbFactor(uint64_t loopIdx)
{
    uint64_t ubFactorCount = GetUbFactorCount();
    if (loopIdx % ubFactorCount == ubFactorCount - 1) {
        uint32_t tailFactor = outShape_[ubAxis_] - (ubFactorCount - 1) * ubFactor_;
        return tailFactor > 0 ? tailFactor : ubFactor_;
    }
    return ubFactor_;
}

template <typename T>
__aicore__ inline uint64_t Batchtospace<T>::ComputeOutOffset(
    uint64_t blockIdx, uint64_t loopIdx, uint32_t ubAxisOffset)
{
    uint64_t baseIdx = blockIdx * perCoreCount_ + loopIdx;
    
    uint64_t ubFactorCount = GetUbFactorCount();
    uint64_t outerIdx = baseIdx / ubFactorCount;
    uint64_t ubAxisIdx = baseIdx % ubFactorCount;
    
    uint64_t offset = 0;
    uint64_t multiplier = 1;
    
    for (int32_t axis = AXIS_COUNT - 1; axis >= 0; axis--) {
        if (axis == static_cast<int32_t>(ubAxis_)) {
            offset += (ubAxisIdx * ubFactor_ + ubAxisOffset) * outStride_[axis];
        } else {
            uint64_t axisIdx = outerIdx % outShape_[axis];
            outerIdx /= outShape_[axis];
            offset += axisIdx * outStride_[axis];
        }
        multiplier *= outShape_[axis];
    }
    
    return offset;
}

template <typename T>
__aicore__ inline uint64_t Batchtospace<T>::ComputeInOffset(uint64_t outOffset)
{
    uint64_t outN = outOffset / outStride_[INDEX_N];
    uint64_t outH = (outOffset % outStride_[INDEX_N]) / outStride_[INDEX_H];
    uint64_t outW = (outOffset % outStride_[INDEX_H]) / outStride_[INDEX_W];
    uint64_t outC = outOffset % outStride_[INDEX_W];
    
    outH += crops_[0][0];
    outW += crops_[1][0];
    
    uint64_t blockSizeSquare = blockSize_ * blockSize_;
    uint64_t blockH = outH / blockSize_;
    uint64_t blockW = outW / blockSize_;
    uint64_t innerH = outH % blockSize_;
    uint64_t innerW = outW % blockSize_;
    
    uint64_t batchBlockIdx = blockH * blockSize_ + blockW;
    uint64_t inN = outN * blockSizeSquare + batchBlockIdx;
    uint64_t inH = innerH;
    uint64_t inW = innerW;
    
    uint64_t inOffset = inN * inStride_[INDEX_N] + inH * inStride_[INDEX_H] + 
                        inW * inStride_[INDEX_W] + outC;
    
    return inOffset;
}

template <typename T>
__aicore__ inline void Batchtospace<T>::CopyIn(uint64_t blockIdx, uint64_t loopIdx)
{
    LocalTensor<T> inputLocal = queueIn.AllocTensor<T>();
    
    uint32_t currentUbFactor = GetCurrentUbFactor(loopIdx);
    uint64_t totalCopyElements = currentUbFactor;
    
    for (uint32_t axis = ubAxis_ + 1; axis < AXIS_COUNT; axis++) {
        totalCopyElements *= outShape_[axis];
    }
    
    uint64_t ubFactorCount = GetUbFactorCount();
    uint64_t ubAxisLoopIdx = loopIdx % ubFactorCount;
    
    for (uint32_t offset = 0; offset < currentUbFactor; offset++) {
        uint64_t outOffset = ComputeOutOffset(blockIdx, loopIdx, offset);
        uint64_t inOffset = ComputeInOffset(outOffset);
        
        uint64_t innerCopyElements = 1;
        for (uint32_t axis = ubAxis_ + 1; axis < AXIS_COUNT; axis++) {
            innerCopyElements *= outShape_[axis];
        }
        
        if (innerCopyElements > 0) {
            DataCopy(inputLocal[offset * innerCopyElements], inputGM[inOffset], innerCopyElements);
        }
    }
    
    queueIn.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Batchtospace<T>::CopyOut(uint64_t blockIdx, uint64_t loopIdx)
{
    LocalTensor<T> inputLocal = queueIn.DeQue<T>();
    
    uint32_t currentUbFactor = GetCurrentUbFactor(loopIdx);
    uint64_t ubFactorCount = GetUbFactorCount();
    uint64_t ubAxisLoopIdx = loopIdx % ubFactorCount;
    
    for (uint32_t offset = 0; offset < currentUbFactor; offset++) {
        uint64_t outOffset = ComputeOutOffset(blockIdx, loopIdx, offset);
        
        uint64_t innerCopyElements = 1;
        for (uint32_t axis = ubAxis_ + 1; axis < AXIS_COUNT; axis++) {
            innerCopyElements *= outShape_[axis];
        }
        
        if (innerCopyElements > 0) {
            DataCopy(outputGM[outOffset], inputLocal[offset * innerCopyElements], innerCopyElements);
        }
    }
    
    queueIn.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Batchtospace<T>::Process()
{
    uint64_t blockIdx = GetBlockIdx();
    uint64_t beginIdx = blockIdx * perCoreCount_;
    uint64_t endIdx = (blockIdx + 1) * perCoreCount_;
    if (endIdx > totalCount_) {
        endIdx = totalCount_;
    }
    
    for (uint64_t idx = beginIdx; idx < endIdx; idx++) {
        CopyIn(blockIdx, idx - beginIdx);
        CopyOut(blockIdx, idx - beginIdx);
    }
}

} // namespace NsBatchtospace
#endif // BATCHTOSPACE_H