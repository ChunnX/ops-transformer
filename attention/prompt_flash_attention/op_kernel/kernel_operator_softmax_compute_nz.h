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
 * \file kernel_operator_softmax_compute_nz.h
 * \brief
 */
#ifndef KERNEL_OPERATOR_SOFTMAX_COMPUTE_NZ_H
#define KERNEL_OPERATOR_SOFTMAX_COMPUTE_NZ_H


#include "kernel_tensor.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#include "adv_api/activation/softmax.h"
#include "adv_api/activation/softmaxflashv2.h"
#endif
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#pragma begin_pipe(V)
namespace AscendC {
constexpr bool SOFTMAX_EXP_FAST = true;
__aicore__ inline void ReduceMaxLastNZImplPFA(const LocalTensor<half>& dst, const LocalTensor<half>& src,
    const LocalTensor<half>& tmpBuffer, uint64_t mask[2], const ReduceLastND& reduceParam)
{
    const uint32_t splitNZBlockCount = reduceParam.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitOffset = reduceParam.dstM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitCount = reduceParam.originalSrcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t lastBlockMaskLen = reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT : SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(0, splitCount);
    Muls<half, false>(dst, src, 1.0, MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    for (int j = 1; j < splitNZBlockCount; j++) {
        Max<half, false>(dst, dst, src[splitOffset * j], MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
    }
    SetVectorMask<half, MaskMode::COUNTER>(0, reduceParam.srcM * 16);   // 16: FLOAT_NUM_PER_BLK
    BlockReduceMax<half, false>(dst, dst, 1, MASK_PLACEHOLDER, 1, 1, 8);    // 8: DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE
    SetMaskNorm();
    ResetMask();

    PipeBarrier<PIPE_V>();

    uint8_t repeat = reduceParam.srcM / 16;
    for (uint8_t i = 0; i < repeat; i++) {
        Muls<half, false>(tmpBuffer[i * 128 * 2], dst[i * 16], 1.0, MASK_PLACEHOLDER, 2, { 1, 0, DEFAULT_REPEAT_STRIDE, 0 });   // 2: FLOAT_REPEAT_SIZE  128: BLOCK_SIZE  16: is the dst copy factor
    }
    PipeBarrier<PIPE_V>();
    uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE];
    for (int32_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstList[i] = (uint64_t)dst[i * 16].GetPhyAddr();   // 16: FLOAT_NUM_PER_BLK
        srcList[i] = (uint64_t)tmpBuffer[i * 16].GetPhyAddr();   // 16: FLOAT_NUM_PER_BLK
    }
    TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = repeat;
    if (transDataParams.repeatTimes > 1) {
        transDataParams.dstRepStride = B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE;
        transDataParams.srcRepStride = B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE;
    }
    TransDataTo5HD<half>(dstList, srcList, transDataParams);
}

__aicore__ inline void ReduceSumLastNZImplPFA(const LocalTensor<half>& dst, const LocalTensor<half>& src,
    const LocalTensor<half>& tmpBuffer, uint64_t mask[2], const ReduceLastND& reduceParam)
{
    const uint32_t splitNZBlockCount = reduceParam.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitOffset = reduceParam.dstM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitCount = reduceParam.originalSrcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t lastBlockMaskLen = reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT != 0 ?
        reduceParam.originalSrcK % SOFTMAX_SHAPE_NZ_BASIC_COUNT :
        SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(0, splitCount);
    Muls<half, false>(dst, src, 1.0, MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    for (int j = 1; j < splitNZBlockCount; j++) {
        Add<half, false>(dst, dst, src[splitOffset * j], MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
    }

    SetVectorMask<half, MaskMode::COUNTER>(0, reduceParam.srcM * 16);    // 16: FLOAT_NUM_PER_BLK
    BlockReduceSum<half, false>(dst, dst, reduceParam.srcM / 8, MASK_PLACEHOLDER, 1, 1, 8);     // 8: Align 8 elements
    SetMaskNorm();
    ResetMask();

    PipeBarrier<PIPE_V>();
    uint8_t repeat = reduceParam.srcM / 16;
    for (uint8_t i = 0; i < repeat; i++) {
        Muls<half, false>(tmpBuffer[i * 128 * 2], dst[i * 16], 1.0, MASK_PLACEHOLDER, 2, { 1, 0, DEFAULT_REPEAT_STRIDE, 0 });     // 2: FLOAT_REPEAT_SIZE  128: BLOCK_SIZE  16: is the dst copy factor
    }
    PipeBarrier<PIPE_V>();
    uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE];
    for (int32_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstList[i] = (uint64_t)dst[i * 16].GetPhyAddr();  // 16: FLOAT_NUM_PER_BLK
        srcList[i] = (uint64_t)tmpBuffer[i * 16].GetPhyAddr();  // 16: FLOAT_NUM_PER_BLK
    }
    TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = repeat;
    if (transDataParams.repeatTimes > 1) {
        transDataParams.dstRepStride = B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE;
        transDataParams.srcRepStride = B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE;
    }
    TransDataTo5HD<half>(dstList, srcList, transDataParams);
}

__aicore__ inline void CreateSpecialFormatMaskPFA(uint64_t& lowMask, const uint32_t& maskLen, const uint32_t& nzBlockCount)
{
    ASSERT(maskLen <= SOFTMAX_SHAPE_NZ_BASIC_COUNT);
    ASSERT(nzBlockCount <= B32_BYTE_SIZE);
    ASSERT(nzBlockCount >= 1);
    uint64_t defaultMask = 0xFFFF >> (SOFTMAX_SHAPE_NZ_BASIC_COUNT - maskLen); // logic shift right
    lowMask = defaultMask;

    for(uint32_t i = 0; i < nzBlockCount - 1; i++) {
        lowMask = lowMask << SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        lowMask = lowMask | defaultMask;
    }
}

__aicore__ inline void ReduceMaxBlockNZImplPFA(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const ReduceLastND& reduceParam)
{
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_NUM_PER_BLK);

    Max<float, false>(dst, src, src[FLOAT_NUM_PER_BLK], 1, 1,
        { B16_BYTE_SIZE, B16_BYTE_SIZE, B16_BYTE_SIZE, DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE,
        DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE, DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE});
    PipeBarrier<PIPE_V>();
    BlockReduceMax<float, false>(dst, dst, reduceParam.srcM / FLOAT_NUM_PER_BLK, MASK_PLACEHOLDER,
        DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE, B16_BYTE_SIZE, DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE);
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void ContinusColumnBrcbImplPFA(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal,
    const uint32_t& repeat, const uint32_t& brcbCount)
{
    float scalarList[SCALAR_STACK_DEPTH] = {0};
    SetVectorMask<float>(brcbCount);
    const uint32_t rangeM = repeat / SCALAR_STACK_DEPTH;
    const uint32_t tailM = repeat % SCALAR_STACK_DEPTH;
    for(uint32_t i = 0; i < rangeM; i++) {
        for(uint32_t j = 0; j < SCALAR_STACK_DEPTH; j ++) {
            scalarList[j] = srcLocal.GetValue(i * brcbCount * SCALAR_STACK_DEPTH + j);
        }
        for (uint32_t k = 0; k < SCALAR_STACK_DEPTH; k ++) {
            Duplicate<float, false>(dstLocal[i * brcbCount * SCALAR_STACK_DEPTH + k * brcbCount], scalarList[k],
                MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        }
    }
    if (tailM != 0) {
        for (uint32_t j = 0; j < tailM; j++) {
            scalarList[j] = srcLocal.GetValue(rangeM * brcbCount * SCALAR_STACK_DEPTH + j);
        }
        for (uint32_t k = 0; k < tailM; k++) {
            Duplicate<float, false>(dstLocal[rangeM * brcbCount * SCALAR_STACK_DEPTH + k * brcbCount], scalarList[k],
                MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        }
    }
}

__aicore__ inline void BinaryComputeWithSpecialMaskPFA(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, uint64_t mask[2], const uint32_t& lastBlockMaskLen, const uint32_t& splitCount,
    void (*func)(const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t*,
    const uint8_t, const BinaryRepeatParams&))
{
    uint32_t repeat = splitCount / FLOAT_REPEAT_SIZE;
    uint32_t tail = splitCount % FLOAT_REPEAT_SIZE;

    uint32_t repeatRange = repeat / MAX_REPEAT_TIMES;
    uint32_t repeatTail = repeat % MAX_REPEAT_TIMES;
    const auto offsetCount = MAX_REPEAT_TIMES * FLOAT_REPEAT_SIZE;
    uint32_t src0Offset = 0;
    uint32_t src1Offset = 0;
    uint32_t dstOffset = 0;

    for(uint32_t i = 0; i < repeatRange; i++) {
        func(dst[i * offsetCount], src0[i * offsetCount], src1[i * offsetCount], mask, MAX_REPEAT_TIMES,
            {1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    }
    if (repeatTail != 0) {
        func(dst[repeatRange * offsetCount], src0[repeatRange * offsetCount], src1[repeatRange * offsetCount], mask,
            repeatTail, {1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    }
    if (tail != 0) {
        uint64_t tailMask[2] = { 0, 0 };
        CreateSpecialFormatMaskPFA(tailMask[0], lastBlockMaskLen, tail / SOFTMAX_SHAPE_NZ_BASIC_COUNT);
        func(dst[repeat * FLOAT_REPEAT_SIZE], src0[repeat * FLOAT_REPEAT_SIZE], src1[repeat * FLOAT_REPEAT_SIZE],
            tailMask, 1, {1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    }
}

__aicore__ inline void BroadCastNZImplPFA(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t srcM)
{
    uint8_t repeat = srcM / DEFAULT_REPEAT_STRIDE;
    for (uint8_t i = 0; i < repeat; i++) {
        Muls<float, false>(dst[i * B16_BYTE_SIZE * FLOAT_REPEAT_SIZE], src[i * B16_BYTE_SIZE * FLOAT_REPEAT_SIZE], 1.0,
            MASK_PLACEHOLDER, B16_BYTE_SIZE, { 1, 0, DEFAULT_REPEAT_STRIDE, 0 });
    }
    PipeBarrier<PIPE_V>();
    uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE];
    for (int32_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstList[i] = (uint64_t)dst[i * FLOAT_NUM_PER_BLK].GetPhyAddr();
        srcList[i] = (uint64_t)src[i * FLOAT_NUM_PER_BLK].GetPhyAddr();
    }
    TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = repeat;
    if (transDataParams.repeatTimes > 1) {
        transDataParams.dstRepStride = B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE;
        transDataParams.srcRepStride = B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE;
    }
    TransDataTo5HD<float>(dstList, srcList, transDataParams);
}

__aicore__ inline void UnaryComputeWithSpecialMaskPFA(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    uint64_t mask[2], const uint32_t& lastBlockMaskLen, const uint32_t& splitCount,
    void (*func)(const LocalTensor<float>&, const LocalTensor<float>&, uint64_t*, const uint8_t,
    const UnaryRepeatParams&))
{
   uint32_t repeat = splitCount / FLOAT_REPEAT_SIZE;
    uint32_t tail = splitCount % FLOAT_REPEAT_SIZE;

    uint32_t repeatRange = repeat / MAX_REPEAT_TIMES;
    uint32_t repeatTail = repeat % MAX_REPEAT_TIMES;
    const auto offsetCount = MAX_REPEAT_TIMES * FLOAT_REPEAT_SIZE;
    uint32_t dstOffset = 0;
    uint32_t src0Offset = 0;
    uint32_t src1Offset = 0;

    for(uint32_t i = 0; i < repeatRange; i++) {
        func(dst[i * offsetCount], src[i * offsetCount], mask, MAX_REPEAT_TIMES,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    if (repeatTail != 0) {
        func(dst[repeatRange * offsetCount], src[repeatRange * offsetCount], mask, repeatTail,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    if (tail != 0) {
        uint64_t tailMask[2] = { 0, 0 };
        CreateSpecialFormatMaskPFA(tailMask[0], lastBlockMaskLen, tail / SOFTMAX_SHAPE_NZ_BASIC_COUNT);
        func(dst[repeat * FLOAT_REPEAT_SIZE], src[repeat * FLOAT_REPEAT_SIZE], tailMask, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
}

template <typename T>
__aicore__  inline void ExpFast(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (1) {
        int32_t N = 6;
        int32_t temp = 1 << N;
        float a = 1.0f / temp / temp / 2.0f;
        float b = temp;
        float c = 0.5f;
        Maxs<T, false>(dst, src, -64.0f, MASK_PLACEHOLDER, repeatTimes, repeatParams);
        PipeBarrier<PIPE_V>();
        Adds<T, false>(dst, dst, b, MASK_PLACEHOLDER, repeatTimes, repeatParams);
        PipeBarrier<PIPE_V>();
        Mul<T, false>(dst, dst, dst, MASK_PLACEHOLDER, repeatTimes,
            { (uint8_t)repeatParams.dstBlkStride, (uint8_t)repeatParams.srcBlkStride,
            (uint8_t)repeatParams.srcBlkStride, (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride,
            (uint8_t)repeatParams.srcRepStride });
        PipeBarrier<PIPE_V>();
        Muls<T, false>(dst, dst, a, MASK_PLACEHOLDER, repeatTimes, repeatParams);
        PipeBarrier<PIPE_V>();
        Adds<T, false>(dst, dst, c, MASK_PLACEHOLDER, repeatTimes, repeatParams);
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < N; ++i) {
            Mul<T, false>(dst, dst, dst, MASK_PLACEHOLDER, repeatTimes,
                { (uint8_t)repeatParams.dstBlkStride, (uint8_t)repeatParams.srcBlkStride,
                (uint8_t)repeatParams.srcBlkStride, (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride,
                (uint8_t)repeatParams.srcRepStride} );
            PipeBarrier<PIPE_V>();
        }
    } else {
        // 2^10 / 0.69314718
        half a = 1477.319723;
        // 2^10 * (15 - 0.043677448)
        half b = 15315.27429;
        Maxs<T, false>(dst, src, -64.0f, MASK_PLACEHOLDER, repeatTimes, repeatParams);
        PipeBarrier<PIPE_V>();
        Muls<T, false>(dst, dst, a, MASK_PLACEHOLDER, repeatTimes, repeatParams);
        PipeBarrier<PIPE_V>();
        Adds<T, false>(dst, dst, b, MASK_PLACEHOLDER, repeatTimes, repeatParams);
        PipeBarrier<PIPE_V>();
        LocalTensor<int16_t> tmp = dst.template ReinterpretCast<int16_t>();
        Cast<int16_t, T, false>(tmp, dst, RoundMode::CAST_RINT, MASK_PLACEHOLDER, repeatTimes, repeatParams);
    }
}

__aicore__ inline LastAxisShapeND GetLastAxisOriginShapeNDPFA(const ShapeInfo& srcShapeInfo)
{
    uint32_t calculateSize = 1;
    LastAxisShapeND ndinfo;
    for (uint32_t i = 0; i < srcShapeInfo.originalShapeDim; i++) {
        calculateSize *= srcShapeInfo.originalShape[i];
    }
    ASSERT(srcShapeInfo.originalShapeDim > 0);
    ndinfo.k = srcShapeInfo.originalShape[srcShapeInfo.originalShapeDim - 1];
    ASSERT(ndinfo.k > 0);
    ndinfo.m = calculateSize / ndinfo.k;
    return ndinfo;
}

} // namespace AscendC
#pragma end_pipe
#endif // kernel_operator_softmax_compute_nz.h