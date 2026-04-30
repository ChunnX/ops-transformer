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
 * \file ai_infra_kv_rms_norm_rope_cache_helper.h
 * \brief
 */

#ifndef _AI_INFRA_KV_RMS_NORM_ROPE_CACHE_HELPER_H_
#define _AI_INFRA_KV_RMS_NORM_ROPE_CACHE_HELPER_H_

//#include "reduce_common.h"
#include "kernel_operator.h"

namespace AiInfraKvRmsNormRopeCache {
using namespace AscendC;

constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t NUM_EIGHT = 8;
constexpr int64_t NUM_SIXTEEN = 16;

constexpr uint32_t FLOAT_BLOCK_ELEM = 8;
constexpr int32_t ROW_FACTOR = 128;
constexpr uint32_t ELEM_PER_BLK_FP16 = 16;
constexpr float DYNAMIC_QUANT_DIVIDEND = 127.0;
constexpr float DYNAMIC_QUANT_EPSINON = 1e-10;

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {
};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {
};

__aicore__ inline float SafeDiv(float a, float b)
{
    if (b < DYNAMIC_QUANT_EPSINON && b > -DYNAMIC_QUANT_EPSINON) {
        b = DYNAMIC_QUANT_EPSINON;
    }
    return a / b;
}

__aicore__ inline int64_t SafeCeil(int64_t a, int64_t b){
    if(b == 0){
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t Align(int64_t a, int64_t b){
    if(b == 0){
        return 0;
    }
    return SafeCeil(a, b) * b;
}

__aicore__ inline uint32_t ROUND_UP32(uint32_t x)
{
    return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
}

__aicore__ inline uint32_t TWO_NUMS_MIN(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}

__aicore__ inline uint32_t TWO_NUMS_MAX(uint32_t x, uint32_t y)
{
    return x > y ? x : y;
}

__aicore__ inline int32_t findPowerTwoFloor(int32_t n)
{
    // find max power of 2 no more than n (32 bit)
    n |= n >> 1; // Set the first digit of n's binary to 1
    n |= n >> NUM_TWO;
    n |= n >> NUM_FOUR;
    n |= n >> NUM_EIGHT;
    n |= n >> NUM_SIXTEEN;
    return (n + 1) >> 1;
}

__aicore__ inline int32_t findPowerTwoCeil(int32_t n)
{
    // find min power of 2 just more than n (32 bit)
    if (n <= 1) return 1;
    n--;

    n |= n >> 1; // Set the first digit of n's binary to 1
    n |= n >> NUM_TWO;
    n |= n >> NUM_FOUR;
    n |= n >> NUM_EIGHT;
    n |= n >> NUM_SIXTEEN;
    return (n + 1);
}

template <typename T, template <typename U> typename R, template <typename U> typename S>
__aicore__ inline void DataCopyEx(
    const R<T>& dst, const S<T>& src, const uint32_t len, const uint32_t count = 1, const bool ubAligned = false)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = count;
    copyParams.blockLen = len * sizeof(T);
    if constexpr (is_same<R<T>, AscendC::LocalTensor<T>>::value) {
        copyParams.srcStride = 0;
        copyParams.dstStride = (ubAligned) ? 1 : 0;
        DataCopyPad(dst, src, copyParams, {});
    } else {
        copyParams.srcStride = (ubAligned) ? 1 : 0;
        copyParams.dstStride = 0;
        DataCopyPad(dst, src, copyParams);
    }
}

/*
 * support count in (0, 255 * 64)
 * very very slow
 */
__aicore__ inline float ReduceMaxFP32(const LocalTensor<float>& src_local, int32_t count)
{
    float value = 0.0;
    ReduceMax(src_local, src_local, src_local, count);
    value = src_local.GetValue(0);
    return value;
}

__aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& srcLocal, int32_t count, int32_t tileSize = ROW_FACTOR)
{
    int32_t tilePower = findPowerTwoCeil(tileSize);
    uint64_t repsFp32 = count >> tilePower;       // 
    uint64_t offsetsFp32 = repsFp32 << tilePower; // 
    uint64_t remsFp32 = count & ((1 << tilePower) - 1);     // tail length

    if (likely(repsFp32 > 1)) {
        // 8 is rep stride
        Max(srcLocal, srcLocal[MultiAddRNDQ::ELEM_PER_REP_FP32], srcLocal, MultiAddRNDQ::ELEM_PER_REP_FP32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? MultiAddRNDQ::ELEM_PER_REP_FP32 : count;
    // 8 is rep stride
    WholeReduceMax(srcLocal, srcLocal, mask, 1, 8, 1, 8);
    PipeBarrier<PIPE_V>();
}

/*
 * only support count <= 255 * 64 = 16320
 * very fast
 */
__aicore__ inline float ReduceSumFP32(const LocalTensor<float>& src_local, int32_t count)
{
    float value = 0.0;
    ReduceSum(src_local, src_local, src_local, count);
    value = src_local.GetValue(0);
    return value;
}

__aicore__ inline void ReduceSumInplace(const LocalTensor<float>& srcLocal, int32_t count)
{
    int32_t tilePower = findPowerTwoCeil(tileSize);
    uint64_t repsFp32 = count >> tilePower;       // 
    uint64_t offsetsFp32 = repsFp32 << tilePower; // 
    uint64_t remsFp32 = count & ((1 << tilePower) - 1);     // tail length

    if (likely(repsFp32 > 1)) {
        // 8 is rep stride
        Add(srcLocal, srcLocal[MultiAddRNDQ::ELEM_PER_REP_FP32], srcLocal, MultiAddRNDQ::ELEM_PER_REP_FP32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0)) {
        Add(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? MultiAddRNDQ::ELEM_PER_REP_FP32 : count;
    // 8 is rep stride
    WholeReduceSum(srcLocal, srcLocal, mask, 1, 8, 1, 8);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void DivScalarFP32(
    LocalTensor<float>& dstTensor, LocalTensor<float>& dividendTensor, LocalTensor<float>& tmpTensor,
    float divisorScalar, uint32_t count)
{
    int32_t tilePower = findPowerTwoCeil(tileSize);           // 向上取整到2的幂的指数
    uint64_t tileSizeAligned = 1ULL << tilePower;             // 对齐后的tile大小
    uint64_t tileMask = tileSizeAligned - 1;                  // 用于取余的掩码
    uint64_t tileMaskFloor = ~tileMask;                       // 用于向下取整的掩码

    uint64_t repsFp32 = count >> tilePower;                   // 完整tile个数
    uint64_t offsetsFp32 = count & tileMaskFloor;             // 对齐后的偏移量
    uint64_t remsFp32 = count & tileMask;                     // 尾部剩余长度

    Duplicate(tmpTensor, divisorScalar, FLOAT_BLOCK_ELEM); // FLOAT_BLOCK_ELEM);
    PipeBarrier<PIPE_V>();
    Div(dstTensor, dividendTensor, tmpTensor, MultiAddRNDQ::ELEM_PER_REP_FP32, repsFp32, {1, 1, 0, 8, 8, 0});
    if ((remsFp32 > 0)) {
        Div(dstTensor[offsetsFp32], dividendTensor[offsetsFp32], tmpTensor, remsFp32, 1, {1, 1, 0, 8, 8, 0});
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void RoundFloat2Int8(LocalTensor<int8_t>& dstTensor, LocalTensor<float>& srcTensor, int32_t size)
{
    Cast(srcTensor.ReinterpretCast<int32_t>(), srcTensor, RoundMode::CAST_RINT, size);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(srcTensor.ReinterpretCast<half>(), srcTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, size);
    PipeBarrier<PIPE_V>();
    Cast(dstTensor, srcTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, size);
    PipeBarrier<PIPE_V>();
}
} // namespace AiInfraKvRmsNormRopeCache
#endif // _AI_INFRA_KV_RMS_NORM_ROPE_CACHE_HELPER_H_
