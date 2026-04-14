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
 * \file upsample_nearest3d.h
 * \brief
 */
#ifndef UPSAMPLE_NEAREST3D_H
#define UPSAMPLE_NEAREST3D_H

#include <type_traits>
#include "upsample_nearest3d_struct.h"
#include "kernel_operator.h"

namespace UpsampleNearest3d {
using namespace AscendC;

constexpr int8_t D_INDEX = 0;
constexpr int8_t H_INDEX = 1;
constexpr int8_t W_INDEX = 2;
constexpr int32_t BUFFER_NUM = 2;

constexpr uint32_t BYTE_BLOCK = 32;
constexpr float BEST_PERFORMANCE_SCALE = 100.0f;
constexpr float ZERO_FLOAT = 0.0f;
constexpr float ONE_FLOAT = 1.0f;
constexpr float EXACT_VALUE = 0.5f;
constexpr int32_t NUM_PER_REP_FP32 = 64;
constexpr int32_t NUM_PER_REP_FP16 = 128;

template <typename T>
class UpsampleNearest3dND {
public:
    TPipe pipe;

    __aicore__ inline UpsampleNearest3dND(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, bool isNearestExact, GM_ADDR workspace, const UpsampleNearest3dTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1>
    __aicore__ inline T1 Min(T1 a, T1 b)
    {
        return a < b ? a : b;
    };
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 p, T2 q)
    {
        if (q == 0) {
            return p;
        }
        return (p + q - 1) / q;
    };
    template <typename T1>
    __aicore__ inline T1 Max(T1 m, T1 n)
    {
        return m > n ? m : n;
    };
    __aicore__ inline void ParseTilingData(const UpsampleNearest3dTilingData* tilingData);
    __aicore__ inline void GatherData(int64_t slideIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void CopyIn(int64_t inputOffset, DataCopyExtParams copyParams);
    __aicore__ inline void CopyOut(int64_t outputOffset, DataCopyExtParams copyParams);
    __aicore__ inline void ComputeAndCopyOut(
        uint32_t dataCount, uint32_t srcDataLength, uint32_t blockCount, int64_t outputOffset);
    __aicore__ inline void GetRangeW(int64_t slideIndex);
    __aicore__ inline void GetRangeH(int64_t slideIndex);
    __aicore__ inline void GetRangeD(int64_t slideIndex);
    __aicore__ inline void CalculateSrcIndexTensor(
        int64_t index, int64_t length, int8_t direction, LocalTensor<float> srcIndexTensor);
    __aicore__ inline void CalculateGatherOffsetW();
    __aicore__ inline void ComputeView1DSmallW();
    __aicore__ inline void DoComputeView1DSmallW(
        int64_t row, int64_t batches, int64_t repeat, int64_t batchesEachRepeat);
    __aicore__ inline void DoComputeGatherOffset(int64_t batchesEachRepeat, int64_t numPerRep, int64_t repeatTimes);

private:
    TBuf<QuePosition::VECCALC> srcIndexQueueW;
    TBuf<QuePosition::VECCALC> srcIndexQueueH;
    TBuf<QuePosition::VECCALC> srcIndexQueueD;
    TBuf<QuePosition::VECCALC> srcOffsetQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;

    LocalTensor<float> srcIndexTensorW;
    LocalTensor<float> srcIndexTensorH;
    LocalTensor<float> srcIndexTensorD;
    LocalTensor<int32_t> srcOffsetTensor;
    LocalTensor<uint32_t> gatherOffsetTensor;

    int64_t blockIdx = 0;
    bool isExact = false;
    bool isView1DAndSmallW = false;
    int64_t batches = 0;
    int64_t inputShapes[3] = {0};
    int64_t outputShapes[3] = {0};
    float scales[3] = {ZERO_FLOAT};

    int64_t slideSizeW = 0;
    int64_t tensorSizeD = 0;
    int64_t tensorSizeH = 0;
    int64_t tensorSizeW = 0;

    int64_t needCoreNum = 0;
    int64_t batchNum = 0;
    int64_t slideNumD = 0;
    int64_t slideNumH = 0;
    int64_t eachCoreSlideNum = 0;
    int64_t remainder = 0;
    int64_t tailStartSlideNum = 0;
    int64_t groupCoreNum = 0;
    int64_t inputRow = 0;
    int64_t tailAvergingRow = 0;

    int64_t lastStartW = -1;
    int64_t startW = 0;
    int64_t endW = 0;
    int64_t dataCount = 0;
    int64_t srcStartW = 0;
    int64_t srcEndW = 0;
    int64_t srcDataCount = 0;
    int64_t srcDataLength = 0;
    uint32_t srcBlockLen = 0;
    uint32_t srcStride = 0;

    int64_t indexH = 0;
    int64_t srcIndexH = 0;
    int64_t heightCount = 0;

    int64_t indexD = 0;
    int64_t srcIndexD = 0;
    int64_t depthCount = 0;

    int32_t numPerRep = 0;
};

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::Init(
    GM_ADDR x, GM_ADDR y, bool isNearestExact, GM_ADDR workspace, const UpsampleNearest3dTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    isExact = isNearestExact;
    ParseTilingData(tilingData);

    pipe.InitBuffer(srcIndexQueueW, slideSizeW * sizeof(float));
    pipe.InitBuffer(srcIndexQueueH, CeilA2B(tensorSizeH * sizeof(float), BYTE_BLOCK) * BYTE_BLOCK);
    pipe.InitBuffer(srcIndexQueueD, CeilA2B(tensorSizeD * sizeof(float), BYTE_BLOCK) * BYTE_BLOCK);
    pipe.InitBuffer(srcOffsetQueue, slideSizeW * sizeof(int32_t));
    pipe.InitBuffer(inQueue, BUFFER_NUM, CeilA2B(tensorSizeW * sizeof(T), BYTE_BLOCK) * BYTE_BLOCK);
    pipe.InitBuffer(outQueue, BUFFER_NUM, slideSizeW * sizeof(T));

    inTensorsGM.SetGlobalBuffer((__gm__ T*)x);
    outTensorsGM.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::Process()
{
    if (blockIdx >= needCoreNum) {
        return;
    }
    if (isView1DAndSmallW) {
        ComputeView1DSmallW();
    } else {
        srcOffsetTensor = srcOffsetQueue.AllocTensor<int32_t>();
        srcIndexTensorW = srcIndexQueueW.AllocTensor<float>();
        srcIndexTensorH = srcIndexQueueH.AllocTensor<float>();
        srcIndexTensorD = srcIndexQueueD.AllocTensor<float>();
        lastStartW = -1;

        int64_t slideStart = blockIdx * eachCoreSlideNum;
        int64_t slideEnd = slideStart + eachCoreSlideNum;
        // 计算批量分组的数据
        if (slideStart < slideEnd) {
            for (int64_t slideIdx = slideStart; slideIdx < slideEnd; slideIdx++) {
                GatherData(slideIdx, 0, inputRow);
            }
        }

        int64_t groupIndex = blockIdx / groupCoreNum;
        if (groupIndex < remainder) {
            // 处理尾块部分数据
            int64_t slideIdx = tailStartSlideNum + groupIndex;
            int64_t blockIdxInGroup = blockIdx % groupCoreNum;
            int64_t tailRowStart = blockIdxInGroup * tailAvergingRow;
            int64_t tailRowEnd = Min(tailRowStart + tailAvergingRow, inputRow);
            GatherData(slideIdx, tailRowStart, tailRowEnd);
        }

        srcIndexQueueD.FreeTensor(srcIndexTensorD);
        srcIndexQueueH.FreeTensor(srcIndexTensorH);
        srcIndexQueueW.FreeTensor(srcIndexTensorW);
        srcOffsetQueue.FreeTensor(srcOffsetTensor);
    }
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::ComputeView1DSmallW()
{
    if constexpr (std::is_same<T, float>::value) {
        numPerRep = NUM_PER_REP_FP32;
    } else {
        numPerRep = NUM_PER_REP_FP16;
    }

    srcIndexTensorW = srcIndexQueueW.Get<float>();
    srcOffsetTensor = srcOffsetQueue.Get<int32_t>();
    int64_t ow = outputShapes[W_INDEX];
    int64_t iw = inputShapes[W_INDEX];
    CalculateSrcIndexTensor(0, ow, W_INDEX, srcIndexTensorW);
    int64_t batchesEachRepeat = numPerRep / ow;                   // 每个repeat处理的batch数
    int64_t repeatTimes = slideSizeW / numPerRep;                 // 每行计算处理的repeat数
    int64_t batchesEachCompute = repeatTimes * batchesEachRepeat; // 每行计算可处理的batch数
    int64_t rowNum = tailAvergingRow;
    if (blockIdx == (needCoreNum - 1)) {
        rowNum = inputRow - tailAvergingRow * (needCoreNum - 1);  //获取尾核的batch数
    }
    int64_t times = rowNum / batchesEachCompute;                  // 整行的计算次数
    int64_t tailComputeBatches = rowNum % batchesEachCompute;     // 尾行的batch数
    int64_t tailRepeat = tailComputeBatches / batchesEachRepeat;  // 尾行的repeat数
    int64_t tailBatches = tailComputeBatches % batchesEachRepeat; // 尾行的尾块的batch数

    DoComputeGatherOffset(batchesEachRepeat, numPerRep, repeatTimes);
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    // 整行处理
    for (int64_t i = 0; i < times; i++) {
        DoComputeView1DSmallW(i, batchesEachCompute, repeatTimes, batchesEachRepeat);
    }
    // 尾行前n-1个整repeat处理
    if (tailComputeBatches != 0) {
        DoComputeView1DSmallW(times, batchesEachCompute, tailRepeat, batchesEachRepeat);
    }
    // 尾行的尾repeat处理
    {
        int64_t baseOffset = blockIdx * tailAvergingRow + times * batchesEachCompute  + tailRepeat * batchesEachRepeat;
        int64_t inOffset = baseOffset * iw;
        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(tailBatches * iw * sizeof(T)), 0, 0, 0};
        CopyIn(inOffset, copyInParams);
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        Gather(dstLocal, srcLocal, gatherOffsetTensor, static_cast<uint32_t>(0), tailBatches * ow);
        outQueue.EnQue(dstLocal);
        inQueue.FreeTensor(srcLocal);
        int64_t outOffset = baseOffset * ow;
        DataCopyExtParams copyOutParams{static_cast<uint16_t>(1), static_cast<uint32_t>(tailBatches * ow * sizeof(T)), 0, 0, 0};
        CopyOut(outOffset, copyOutParams);
    }
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::DoComputeGatherOffset(
    int64_t batchesEachRepeat, int64_t numPerRep, int64_t repeatTimes)
{
    int64_t ow = outputShapes[W_INDEX];
    int64_t iw = inputShapes[W_INDEX];
    Duplicate(srcOffsetTensor, static_cast<int32_t>(0), slideSizeW);
    for (int64_t i = 0; i < batchesEachRepeat; i++) {
        for (int64_t j = 0; j < ow; j++) {
            srcOffsetTensor.SetValue(i * ow + j, static_cast<int32_t>(srcIndexTensorW.GetValue(j)) + i * iw);
        }
    }
    for (int64_t i = 1; i < repeatTimes; i++) {
        Adds(srcOffsetTensor[i * numPerRep], srcOffsetTensor[(i - 1) * numPerRep],static_cast<int32_t>(batchesEachRepeat * iw), numPerRep);
        PipeBarrier<PIPE_V>();
    }
    Muls(srcOffsetTensor, srcOffsetTensor, static_cast<int32_t>(sizeof(T)), slideSizeW);
    PipeBarrier<PIPE_V>();
    gatherOffsetTensor = srcOffsetTensor.ReinterpretCast<uint32_t>();
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::DoComputeView1DSmallW(
    int64_t row, int64_t batches, int64_t repeat, int64_t batchesEachRepeat)
{
    int64_t ow = outputShapes[W_INDEX];
    int64_t iw = inputShapes[W_INDEX];
    int64_t inOffset = blockIdx * tailAvergingRow * iw + row * batches * iw;
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(repeat * batchesEachRepeat * iw * sizeof(T)), 0, 0, 0};
    CopyIn(inOffset, copyInParams);
    LocalTensor<T> srcLocal = inQueue.DeQue<T>();
    uint64_t mask = batchesEachRepeat * ow;
    LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
    Gather(dstLocal, srcLocal, gatherOffsetTensor, 0, mask, static_cast<uint8_t>(repeat), 8);
    PipeBarrier<PIPE_V>();
    outQueue.EnQue(dstLocal);
    inQueue.FreeTensor(srcLocal);
    int64_t outOffset = blockIdx * tailAvergingRow * ow + row * batches * ow;
    uint32_t srcStride = (numPerRep - batchesEachRepeat * ow) / (BYTE_BLOCK / sizeof(T));
    DataCopyExtParams copyOutParams{static_cast<uint16_t>(repeat), static_cast<uint32_t>(batchesEachRepeat * ow * sizeof(T)), srcStride, 0, 0};
    CopyOut(outOffset, copyOutParams);
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::GatherData(int64_t slideIndex, int64_t rowStart, int64_t rowEnd)
{
    GetRangeH(slideIndex);
    GetRangeD(slideIndex);
    if (depthCount == 0 || heightCount == 0) {
        return;
    }

    GetRangeW(slideIndex);
    int64_t j = 0;
    while (startW < endW) {
        if (scales[W_INDEX] > BEST_PERFORMANCE_SCALE) {
            srcStartW = static_cast<int64_t>(srcIndexTensorW.GetValue(j));
            j++;
        }
        int64_t inputOffsetsInBatch =
            srcIndexD * inputShapes[H_INDEX] * inputShapes[W_INDEX] + srcIndexH * inputShapes[W_INDEX] + srcStartW;
        int64_t outputOffsetInBatch =
            indexD * outputShapes[H_INDEX] * outputShapes[W_INDEX] + indexH * outputShapes[W_INDEX] + startW;
        for (int64_t batchIndex = rowStart; batchIndex < rowEnd; batchIndex += batchNum) {
            int64_t inputOffset =
                batchIndex * inputShapes[D_INDEX] * inputShapes[H_INDEX] * inputShapes[W_INDEX] + inputOffsetsInBatch;
            int64_t outputOffset = batchIndex * outputShapes[D_INDEX] * outputShapes[H_INDEX] * outputShapes[W_INDEX] +
                                   outputOffsetInBatch;
            uint16_t blockCount = Min(batchNum, rowEnd - batchIndex);
            DataCopyExtParams copyInParams{blockCount, srcBlockLen, srcStride, 0, 0};
            CopyIn(inputOffset, copyInParams);
            ComputeAndCopyOut(dataCount, srcDataLength, blockCount, outputOffset);
        }
        startW += dataCount;
    }
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::CopyIn(int64_t inputOffset, DataCopyExtParams copyParams)
{
    LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(srcLocal, inTensorsGM[inputOffset], copyParams, padParams);
    inQueue.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::CopyOut(int64_t outputOffset, DataCopyExtParams copyParams)
{
    LocalTensor<T> dstLocal = outQueue.DeQue<T>();
    DataCopyPad(outTensorsGM[outputOffset], dstLocal, copyParams);
    outQueue.FreeTensor(dstLocal);
    event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID1);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID1);
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::ComputeAndCopyOut(
    uint32_t dataCount, uint32_t srcDataLength, uint32_t blockCount, int64_t outputOffset)
{
    LocalTensor<T> srcLocal = inQueue.DeQue<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    for (int64_t i = 0; i < blockCount; i++) {
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        Gather(dstLocal, srcLocal, gatherOffsetTensor, static_cast<uint32_t>(i * srcDataLength), dataCount);
        outQueue.EnQue(dstLocal);

        dstLocal = outQueue.DeQue<T>();
        for (int64_t j = 0; j < depthCount; j++) {
            int64_t offset = outputOffset + j * outputShapes[H_INDEX] * outputShapes[W_INDEX];
            for (int64_t k = 0; k < heightCount; k++) {
                DataCopyPad(outTensorsGM[offset + k * outputShapes[W_INDEX]], dstLocal, copyParams);
            }
        }
        outQueue.FreeTensor(dstLocal);
        outputOffset += outputShapes[D_INDEX] * outputShapes[H_INDEX] * outputShapes[W_INDEX];
    }
    inQueue.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::GetRangeW(int64_t slideIndex)
{
    startW = (slideIndex / (slideNumH * slideNumD)) * slideSizeW;
    if (lastStartW != startW) {
        lastStartW = startW;
        endW = Min(startW + slideSizeW, outputShapes[W_INDEX]);
        dataCount = endW - startW;
        CalculateSrcIndexTensor(startW, dataCount, W_INDEX, srcIndexTensorW);
        srcEndW = static_cast<int64_t>(srcIndexTensorW.GetValue(dataCount - 1)) + 1;
        srcStartW = static_cast<int64_t>(srcIndexTensorW.GetValue(0));
        if (scales[W_INDEX] > BEST_PERFORMANCE_SCALE) {
            dataCount = 1;
            srcDataCount = 1;
        } else {
            srcDataCount = srcEndW - srcStartW;
        }
        CalculateGatherOffsetW();
        srcDataLength = CeilA2B(srcDataCount * sizeof(T), BYTE_BLOCK) * BYTE_BLOCK;
        // 搬入数据时可以一次性搬运多个batch
        batchNum = CeilA2B(tensorSizeW * sizeof(T), BYTE_BLOCK) * BYTE_BLOCK / srcDataLength;

        srcBlockLen = srcDataCount * sizeof(T);
        srcStride = (inputShapes[D_INDEX] * inputShapes[H_INDEX] * inputShapes[W_INDEX] - srcDataCount) * sizeof(T);
    }
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::GetRangeH(int64_t slideIndex)
{
    if (ONE_FLOAT <= scales[H_INDEX]) {
        indexH = slideIndex % slideNumH;
        CalculateSrcIndexTensor(indexH, 1, H_INDEX, srcIndexTensorH);
        srcIndexH = static_cast<int64_t>(srcIndexTensorH.GetValue(0));
        heightCount = 1;
        return;
    }
    srcIndexH = slideIndex % slideNumH;
    indexH = Max(static_cast<int64_t>(0), static_cast<int64_t>((float)srcIndexH / scales[H_INDEX] - 2));
    int64_t length = Min(tensorSizeH, outputShapes[H_INDEX] - indexH);
    CalculateSrcIndexTensor(indexH, length, H_INDEX, srcIndexTensorH);
    heightCount = 0;
    for (int64_t j = 0; j < length; j++) {
        int64_t srcIndex = static_cast<int64_t>(srcIndexTensorH.GetValue(j));
        if (srcIndexH == srcIndex) {
            heightCount = 1;
            indexH += j;
            break;
        }
    }
    if (heightCount == 0) {
        return;
    }

    int64_t lastIndexH = Max(indexH, static_cast<int64_t>((float)(srcIndexH + 1) / scales[H_INDEX] - 2));
    lastIndexH = Min(lastIndexH, outputShapes[H_INDEX] - 1);
    length = Min(tensorSizeH, outputShapes[H_INDEX] - lastIndexH);
    CalculateSrcIndexTensor(lastIndexH, length, H_INDEX, srcIndexTensorH);
    for (int64_t j = 0; j < length; j++) {
        int64_t srcIndex = static_cast<int64_t>(srcIndexTensorH.GetValue(j));
        if (srcIndexH == srcIndex) {
            lastIndexH++;
        }
    }
    heightCount = lastIndexH - indexH;
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::GetRangeD(int64_t slideIndex)
{
    if (ONE_FLOAT <= scales[D_INDEX]) {
        indexD = (slideIndex % (slideNumD * slideNumH)) / slideNumH;
        CalculateSrcIndexTensor(indexD, 1, D_INDEX, srcIndexTensorD);
        srcIndexD = static_cast<int64_t>(srcIndexTensorD.GetValue(0));
        depthCount = 1;
        return;
    }
    srcIndexD = (slideIndex % (slideNumD * slideNumH)) / slideNumH;
    indexD = Max(static_cast<int64_t>(0), static_cast<int64_t>((float)srcIndexD / scales[D_INDEX] - 2));
    int64_t length = Min(tensorSizeD, outputShapes[D_INDEX] - indexD);
    CalculateSrcIndexTensor(indexD, length, D_INDEX, srcIndexTensorD);
    depthCount = 0;
    for (int64_t j = 0; j < length; j++) {
        int64_t srcIndex = static_cast<int64_t>(srcIndexTensorD.GetValue(j));
        if (srcIndexD == srcIndex) {
            depthCount = 1;
            indexD += j;
            break;
        }
    }
    if (depthCount == 0) {
        return;
    }

    int64_t lastIndexD = Max(indexD, static_cast<int64_t>((float)(srcIndexD + 1) / scales[D_INDEX] - 2));
    lastIndexD = Min(lastIndexD, outputShapes[D_INDEX] - 1);
    length = Min(tensorSizeD, outputShapes[D_INDEX] - lastIndexD);
    CalculateSrcIndexTensor(lastIndexD, length, D_INDEX, srcIndexTensorD);
    for (int64_t j = 0; j < length; j++) {
        int64_t srcIndex = static_cast<int64_t>(srcIndexTensorD.GetValue(j));
        if (srcIndexD == srcIndex) {
            lastIndexD++;
        }
    }
    depthCount = lastIndexD - indexD;
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::CalculateSrcIndexTensor(
    int64_t index, int64_t length, int8_t direction, LocalTensor<float> srcIndexTensor)
{
    ArithProgression(srcIndexTensor, static_cast<float>(index), ONE_FLOAT, length);
    PipeBarrier<PIPE_V>();
    if (isExact) {
        Adds(srcIndexTensor, srcIndexTensor, EXACT_VALUE, length);
        PipeBarrier<PIPE_V>();
    }
    Muls(srcIndexTensor, srcIndexTensor, scales[direction], length);
    PipeBarrier<PIPE_V>();
    Cast(srcIndexTensor, srcIndexTensor, RoundMode::CAST_FLOOR, length);
    PipeBarrier<PIPE_V>();
    Mins(srcIndexTensor, srcIndexTensor, static_cast<float>(inputShapes[direction] - 1), length);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::CalculateGatherOffsetW()
{
    Cast(srcOffsetTensor, srcIndexTensorW, RoundMode::CAST_FLOOR, dataCount);
    PipeBarrier<PIPE_V>();
    Adds(srcOffsetTensor, srcOffsetTensor, static_cast<int32_t>(-srcStartW), dataCount);
    PipeBarrier<PIPE_V>();
    Muls(srcOffsetTensor, srcOffsetTensor, static_cast<int32_t>(sizeof(T)), dataCount);
    PipeBarrier<PIPE_V>();
    gatherOffsetTensor = srcOffsetTensor.ReinterpretCast<uint32_t>();
}

template <typename T>
__aicore__ inline void UpsampleNearest3dND<T>::ParseTilingData(const UpsampleNearest3dTilingData* tilingData)
{
    batches = tilingData->batches;
    for (int8_t i = 0; i < 3; i++) {
        inputShapes[i] = tilingData->inputShapes[i];
        outputShapes[i] = tilingData->outputShapes[i];
    }

    scales[D_INDEX] = tilingData->scaleD;
    scales[H_INDEX] = tilingData->scaleH;
    scales[W_INDEX] = tilingData->scaleW;
    slideSizeW = tilingData->slideSizeW;
    tensorSizeD = tilingData->tensorSizeD;
    tensorSizeH = tilingData->tensorSizeH;
    tensorSizeW = tilingData->tensorSizeW;

    slideNumD = tilingData->slideNumD;
    slideNumH = tilingData->slideNumH;
    eachCoreSlideNum = tilingData->eachCoreSlideNum;
    remainder = tilingData->remainder;
    tailStartSlideNum = tilingData->tailStartSlideNum;
    groupCoreNum = tilingData->groupCoreNum;
    inputRow = tilingData->inputRow;
    tailAvergingRow = tilingData->tailAvergingRow;
    needCoreNum = tilingData->needCoreNum;
    isView1DAndSmallW = tilingData->isView1DAndSmallW;
}

template <int D_T_X, int D_T_Y>
__aicore__ inline void UpsampleNearest3dKernelImpl(
    __gm__ uint8_t* x, __gm__ uint8_t* y, bool isNearestExact, const UpsampleNearest3dTilingData* tilingData)
{
    GM_ADDR userWS = nullptr;
    if constexpr (D_T_X == UPSAMPLE_NEAREST3D_TPL_FP16 && D_T_Y == UPSAMPLE_NEAREST3D_TPL_FP16) {
        UpsampleNearest3d::UpsampleNearest3dND<half> op;
        op.Init(x, y, isNearestExact, userWS, tilingData);
        op.Process();
    } else if constexpr (D_T_X == UPSAMPLE_NEAREST3D_TPL_FP32 && D_T_Y == UPSAMPLE_NEAREST3D_TPL_FP32) {
        UpsampleNearest3d::UpsampleNearest3dND<float> op;
        op.Init(x, y, isNearestExact, userWS, tilingData);
        op.Process();
    } else if constexpr (D_T_X == UPSAMPLE_NEAREST3D_TPL_BF16 && D_T_Y == UPSAMPLE_NEAREST3D_TPL_BF16) {
        UpsampleNearest3d::UpsampleNearest3dND<bfloat16_t> op;
        op.Init(x, y, isNearestExact, userWS, tilingData);
        op.Process();
    }
}
} // namespace UpsampleNearest3d
#endif // UPSAMPLE_NEAREST3D
