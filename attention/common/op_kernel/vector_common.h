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
 * \file vector_common.h
 * \brief
 */
#ifndef VECTOR_COMMON_H
#define VECTOR_COMMON_H

#include "kernel_operator.h"

using namespace AttentionCommon;
using namespace AscendC;
using AscendC::LocalTensor;

namespace fa_base_vector {

// BLOCK和REPEAT的字节数
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
// BLOCK和REPEAT的FP32元素数
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
// repeat stride不能超过256
constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;
constexpr int64_t HALF_NUM = 2;
constexpr int64_t STRIDE_LENGTH = 8;
constexpr int64_t MAX_VALID_LENGTH = 1024;

__aicore__ inline void VecMulMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec mul by row
    // dstUb[i, j] = src0Ub[j] * src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, actualColumnCount] dstUb:[dealRowCount, columnCount]
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
        uint32_t loopCount = actualColumnCount / mask;
        uint32_t remainCount = actualColumnCount % mask;
        uint32_t offset = 0;
        for (int i = 0; i < loopCount; i++) {
            // offset = i * mask
            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            // offset = loopCount * mask
            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < dealRowCount; i++) {
            Mul(dstUb[offset], src0Ub, src1Ub[offset], actualColumnCount);
            offset += columnCount;
        }
    }
}

__aicore__ inline void MatDivVec(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec mul by row
    // dstUb[i, j] = src1Ub[i, j] / src0Ub[j],
    // src0Ub:[dealRowCount, actualColumnCount] src1Ub:[1, columnCount] dstUb:[dealRowCount, columnCount]
    // restraint: dealRowCount < 256
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src1RepStride = 0;
        repeatParams.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
        uint32_t loopCount = actualColumnCount / mask;
        uint32_t remainCount = actualColumnCount % mask;
        uint32_t offset = 0;
        for (int i = 0; i < loopCount; i++) {
            // offset = i * mask
            Div(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            // offset = loopCount * mask
            Div(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, actualColumnCount);
            offset += columnCount;
        }
    }
}

__aicore__ inline void VecMulBlkMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec mul by row
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, 8] dstUb:[dealRowCount, columnCount]
    BinaryRepeatParams repeatParams;
    uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t loopCount = actualColumnCount / mask;
    uint32_t remainCount = actualColumnCount % mask;
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        // [1, columnCount] * [dealRowCount, 8]
        repeatParams.src0BlkStride = 1;
        repeatParams.src0RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.src1RepStride = 1;
        repeatParams.dstBlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < loopCount; i++) {
            Mul(dstUb[offset], src0Ub[offset], src1Ub, mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            Mul(dstUb[offset], src0Ub[offset], src1Ub, remainCount, dealRowCount, repeatParams);
        }
    } else {
        // [1, columnCount] * [1, 8]
        repeatParams.src0BlkStride = 1;
        repeatParams.src0RepStride = STRIDE_LENGTH;
        repeatParams.src1BlkStride = 0;
        repeatParams.src1RepStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.dstRepStride = STRIDE_LENGTH;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Mul(dstUb[i * columnCount], src0Ub, src1Ub[i * FP32_BLOCK_ELEMENT_NUM], mask, loopCount, repeatParams);
            if (remainCount > 0) {
                Mul(dstUb[i * columnCount + loopCount * mask], src0Ub[loopCount * mask],
                    src1Ub[i * FP32_BLOCK_ELEMENT_NUM], remainCount, 1, repeatParams);
            }
        }
    }
}

__aicore__ inline void VecAddMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec add by row
    // dstUb[i, j] = src0Ub[j] + src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, columnCount] dstUb:[dealRowCount, columnCount]
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
        uint32_t loopCount = actualColumnCount / mask;
        uint32_t remainCount = actualColumnCount % mask;

        uint64_t offset = 0;
        for (int i = 0; i < loopCount; i++) {
            Add(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            Add(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < dealRowCount; i++) {
            Add(dstUb[offset], src0Ub, src1Ub[offset], actualColumnCount);
            offset += columnCount;
        }
    }
}

__aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain, 1,
                    repeatParams);
            }
        }
    }
}

__aicore__ inline void RowSum(LocalTensor<float> &dstUb, LocalTensor<float> srcUb, uint32_t dealRowCount,
                              uint32_t columnCount, uint32_t actualColumnCount)
{
    // sum by row, 按行求和
    // dstUb[i] = sum(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockCount = actualColumnCount / dtypeMask;
    uint32_t remain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / (BYTE_BLOCK / sizeof(float));
    repeatParamsMax.src1RepStride = columnCount / (BYTE_BLOCK / sizeof(float));
    repeatParamsMax.dstRepStride = columnCount / (BYTE_BLOCK / sizeof(float));
    if (blockCount > 0 && remain > 0) {
        Add(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, dealRowCount, repeatParamsMax);
        AscendC::PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopCount = blockCount / HALF_NUM; loopCount > 0; loopCount = blockCount / HALF_NUM) {
        blockCount = (blockCount + 1) / HALF_NUM;
        for (uint32_t j = 0; j < loopCount; j++) {
            Add(srcUb[j * dtypeMask], srcUb[j * dtypeMask], srcUb[(j + blockCount) * dtypeMask], dtypeMask,
                dealRowCount, repeatParamsMax);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    WholeReduceSum(dstUb, srcUb, (actualColumnCount < dtypeMask) ? actualColumnCount : dtypeMask, dealRowCount, 1, 1,
                   columnCount / (BYTE_BLOCK / sizeof(float)));
}

__aicore__ inline uint32_t GetMinPowerTwo(uint32_t cap)
{
    uint32_t i = 1;
    while (i < cap) {
        i = i << 1;
    }
    return i;
}

__aicore__ inline void RowSumForLongColumnCount(LocalTensor<float> &dstUb, LocalTensor<float> srcUb,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // sum by row, 按行求和
    // dstUb[i] = sum(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    // columnCount要求32元素对齐
    uint32_t newColumnCount = columnCount;
    uint32_t newActualColumnCount = actualColumnCount;
    if (columnCount >= REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) {
        uint32_t split = GetMinPowerTwo(actualColumnCount);
        split = split >> 1;

        // deal tail
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Add(srcUb[offset], srcUb[offset], srcUb[offset + split], actualColumnCount - split);
            offset += columnCount;
        }
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t validLen = split;
        while (validLen > MAX_VALID_LENGTH) {
            uint32_t copyLen = validLen / 2;

            offset = 0;
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Add(srcUb[offset], srcUb[offset], srcUb[offset + copyLen], copyLen);
                offset += columnCount;
            }
            AscendC::PipeBarrier<PIPE_V>();

            validLen = copyLen;
        }

        for (uint32_t i = 0; i < dealRowCount; i++) {
            DataCopy(srcUb[i * validLen], srcUb[i * columnCount], validLen);
            AscendC::PipeBarrier<PIPE_V>();
        }

        newColumnCount = validLen;
        newActualColumnCount = validLen;
    }

    RowSum(dstUb, srcUb, dealRowCount, newColumnCount, newActualColumnCount);
}

__aicore__ inline void RowMax(LocalTensor<float> &dstUb, LocalTensor<float> &srcUb, uint32_t dealRowCount,
                              uint32_t columnCount, uint32_t actualColumnCount)
{
    // max by row, 按行求最大值
    // dstUb[i] = max(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockCount = actualColumnCount / dtypeMask;
    uint32_t remain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    if (blockCount > 0 && remain > 0) {
        Max(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, dealRowCount, repeatParamsMax);
        AscendC::PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopCount = blockCount / HALF_NUM; loopCount > 0; loopCount = blockCount / HALF_NUM) {
        blockCount = (blockCount + 1) / HALF_NUM;
        for (uint32_t j = 0; j < loopCount; j++) {
            Max(srcUb[j * dtypeMask], srcUb[j * dtypeMask], srcUb[(j + blockCount) * dtypeMask], dtypeMask,
                dealRowCount, repeatParamsMax);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    WholeReduceMax(dstUb, srcUb, (actualColumnCount < dtypeMask) ? actualColumnCount : dtypeMask, dealRowCount, 1, 1,
                   columnCount / FP32_BLOCK_ELEMENT_NUM, ReduceOrder::ORDER_ONLY_VALUE);
}

__aicore__ inline void RowMaxForLongColumnCount(LocalTensor<float> &dstUb, LocalTensor<float> srcUb,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // max by row, 按行求最大值
    // dstUb[i] = max(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t newColumnCount = columnCount;
    uint32_t newActualColumnCount = actualColumnCount;
    if (columnCount >= REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) {
        uint32_t split = GetMinPowerTwo(actualColumnCount);
        split = split >> 1;

        // deal tail
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Max(srcUb[offset], srcUb[offset], srcUb[offset + split], actualColumnCount - split);
            offset += columnCount;
        }
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t validLen = split;
        while (validLen > MAX_VALID_LENGTH) {
            uint32_t copyLen = validLen / 2;

            offset = 0;
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Max(srcUb[offset], srcUb[offset], srcUb[offset + copyLen], copyLen);
                offset += columnCount;
            }
            AscendC::PipeBarrier<PIPE_V>();

            validLen = copyLen;
        }

        for (uint32_t i = 0; i < dealRowCount; i++) {
            DataCopy(srcUb[i * validLen], srcUb[i * columnCount], validLen);
            AscendC::PipeBarrier<PIPE_V>();
        }

        newColumnCount = validLen;
        newActualColumnCount = validLen;
    }

    RowMax(dstUb, srcUb, dealRowCount, newColumnCount, newActualColumnCount);
}

__aicore__ inline void MatDivsVec(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 1;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 0;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Div(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsDiv);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount, repeatParamsDiv);
    }
}

__aicore__ inline void RowSub(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsSub;
    repeatParamsSub.src0BlkStride = 1;
    repeatParamsSub.src1BlkStride = 1;
    repeatParamsSub.dstBlkStride = 1;
    repeatParamsSub.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsSub.src1RepStride = 0;
    repeatParamsSub.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Sub(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsSub);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Sub(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount, repeatParamsSub);
    }
}

__aicore__ inline void ColMax(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = 0;
    repeatParamsMax.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Max(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsMax);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Max(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount, repeatParamsMax);
    }
}

__aicore__ inline void ColAdd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsAdd;
    repeatParamsAdd.src0BlkStride = 1;
    repeatParamsAdd.src1BlkStride = 1;
    repeatParamsAdd.dstBlkStride = 1;
    repeatParamsAdd.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsAdd.src1RepStride = 0;
    repeatParamsAdd.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Add(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsAdd);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Add(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount, repeatParamsAdd);
    }
}

static constexpr uint64_t headDim = 512ULL;
template <FIA_LAYOUT LAYOUT_T, typename OUT_T>
__aicore__ inline void Bmm2DataCopyOutNBSDMTiling(LocalTensor<OUT_T> &attenOutUb, const FusedTransposeInfo &transInfo,
                                                  const AttentionCommon::ConstInfo &constInfo,
                                                  GlobalTensor<uint64_t> &actualSeqLengthsGmQ,
                                                  GlobalTensor<OUT_T> &attentionOutGm)
{
    uint32_t tSize = constInfo.batchSize * constInfo.qSeqSize;
    uint32_t tBase = transInfo.bIdx * constInfo.qSeqSize;
    if constexpr (LAYOUT_T == FIA_LAYOUT::TND) {
        tSize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1);
        tBase = transInfo.bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(transInfo.bIdx - 1);
    }

    uint32_t s1Idx = transInfo.s1StartIdx;
    uint32_t attenOutUbOffset = 0;
    for (int i = 0; i < transInfo.s1Count; i++) {
        uint32_t gIdx = 0; // 中间块
        uint32_t gCountOneS1 = constInfo.gSize;
        if (i == 0) { // 首块
            gIdx = transInfo.gStartIdx;
            gCountOneS1 = (constInfo.gSize - transInfo.gStartIdx) < transInfo.gCount ?
                              (constInfo.gSize - transInfo.gStartIdx) :
                              transInfo.gCount;  // min(info.gSize - gStartIdx, gCount);
        } else if (i == transInfo.s1Count - 1) { // 尾块
            gIdx = 0;
            gCountOneS1 = transInfo.gEndIdx + 1;
        }
        uint64_t attenOutOffset = transInfo.n2Idx * constInfo.gSize * tSize * headDim + // N2轴的偏移
                                  gIdx * tSize * headDim +                              // G轴的偏移
                                  tBase * headDim +                                     // B轴的偏移
                                  s1Idx * headDim;                                      // S1轴的偏移
        bool dstStrideFlag = ((tSize - 1) * headDim * sizeof(OUT_T) / 32U) > UINT16_MAX ? 1 : 0;
        if (dstStrideFlag) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = gCountOneS1;
            dataCopyParams.blockLen = headDim * sizeof(OUT_T);                // 一个D的大小
            dataCopyParams.srcStride = 0;                                     // 连读
            dataCopyParams.dstStride = (tSize - 1) * headDim * sizeof(OUT_T); // 跳写
            DataCopyPad(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        } else {
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = gCountOneS1;
            dataCopyParams.blockLen = headDim * sizeof(OUT_T) / 32U;                // 一个D的大小
            dataCopyParams.srcStride = 0;                                           // 连读
            dataCopyParams.dstStride = (tSize - 1) * headDim * sizeof(OUT_T) / 32U; // 跳写
            DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        }
        s1Idx++;
        attenOutUbOffset += gCountOneS1 * headDim;
    }
}

template <typename OUT_T>
__aicore__ inline void Bmm2DataCopyOutNBSDGTiling(LocalTensor<OUT_T> &attenOutUb, const FusedTransposeInfo &transInfo,
                                                  const AttentionCommon::ConstInfo &constInfo, GlobalTensor<OUT_T> &attentionOutGm)
{
    bool hasHeadBlock = transInfo.s1StartIdx != 0;
    bool hasTailBlock = (transInfo.s1EndIdx + 1) != constInfo.qSeqSize;
    uint32_t attenOutUbOffset = 0;
    if (hasHeadBlock) { // 头块单独一条DataCopy指令
        DataCopyParams dataCopyParamsHead;
        dataCopyParamsHead.blockCount = 1;
        dataCopyParamsHead.blockLen = (constInfo.qSeqSize - transInfo.s1StartIdx) * headDim * sizeof(OUT_T) / 32U;
        dataCopyParamsHead.srcStride = 0;
        dataCopyParamsHead.dstStride = 0; // blockCount = 1 无所谓跳写
        uint64_t attenOutOffset =
            transInfo.n2Idx * constInfo.gSize * constInfo.batchSize * constInfo.qSeqSize * headDim + // N2轴的偏移
            transInfo.gStartIdx * constInfo.batchSize * constInfo.qSeqSize * headDim +               // G轴的偏移
            transInfo.bIdx * constInfo.qSeqSize * headDim +                                          // B轴的偏移
            transInfo.s1StartIdx * headDim;                                                          // S1轴的偏移
        DataCopy(attentionOutGm[attenOutOffset], attenOutUb, dataCopyParamsHead);
        attenOutUbOffset += (constInfo.qSeqSize - transInfo.s1StartIdx) * headDim;
    }
    // 中间块DataCopy指令
    uint64_t attenOutOffset =
        transInfo.n2Idx * constInfo.gSize * constInfo.batchSize * constInfo.qSeqSize * headDim + // N2轴的偏移
        (transInfo.gStartIdx + static_cast<uint32_t>(hasHeadBlock)) * constInfo.batchSize * constInfo.qSeqSize *
            headDim +                                  // G轴的偏移
        transInfo.bIdx * constInfo.qSeqSize * headDim; // B轴的偏移
    bool dstStrideFlag =
        ((constInfo.batchSize * constInfo.qSeqSize - constInfo.qSeqSize) * headDim * sizeof(OUT_T) / 32U) > UINT16_MAX ?
            1 :
            0;
    if (dstStrideFlag) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount =
            transInfo.gCount - static_cast<uint32_t>(hasHeadBlock) - static_cast<uint32_t>(hasTailBlock); // 处理多少个G
        dataCopyParams.blockLen = constInfo.qSeqSize * headDim * sizeof(OUT_T); // 一个S1*D的大小
        dataCopyParams.srcStride = 0;                                           // 连读
        dataCopyParams.dstStride =
            (constInfo.batchSize * constInfo.qSeqSize - constInfo.qSeqSize) * headDim * sizeof(OUT_T); // 跳写
        DataCopyPad(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        attenOutUbOffset += dataCopyParams.blockCount * (constInfo.qSeqSize * headDim);
    } else {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount =
            transInfo.gCount - static_cast<uint32_t>(hasHeadBlock) - static_cast<uint32_t>(hasTailBlock); // 处理多少个G
        dataCopyParams.blockLen = constInfo.qSeqSize * headDim * sizeof(OUT_T) / 32U; // 一个S1*D的大小
        dataCopyParams.srcStride = 0;                                                 // 连读
        dataCopyParams.dstStride =
            (constInfo.batchSize * constInfo.qSeqSize - constInfo.qSeqSize) * headDim * sizeof(OUT_T) / 32U; // 跳写
        DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        attenOutUbOffset += dataCopyParams.blockCount * (constInfo.qSeqSize * headDim);
    }
    if (hasTailBlock) { // 尾块单独一条DataCopy指令
        DataCopyParams dataCopyParamsTail;
        dataCopyParamsTail.blockCount = 1;
        dataCopyParamsTail.blockLen = (transInfo.s1EndIdx + 1) * headDim * sizeof(OUT_T) / 32U;
        dataCopyParamsTail.srcStride = 0;
        dataCopyParamsTail.dstStride = 0; // blockCount = 1 无所谓跳写
        uint64_t attenOutOffset =
            transInfo.n2Idx * constInfo.gSize * constInfo.batchSize * constInfo.qSeqSize * headDim + // N2轴的偏移
            (transInfo.gStartIdx + transInfo.gCount - 1) * constInfo.batchSize * constInfo.qSeqSize *
                headDim +                                  // G轴的偏移
            transInfo.bIdx * constInfo.qSeqSize * headDim; // B轴的偏移
        DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParamsTail);
    }
}
} // namespace fa_base_vector
#endif