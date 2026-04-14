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
 * \file upsample_linear_common.h
 * \brief
 */
#ifndef UPSAMPLE_LINEAR_COMMON_H
#define UPSAMPLE_LINEAR_COMMON_H
#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t ADDR_ALIGN_SIZE = 128;
constexpr uint8_t SYNC_MODE2 = 2;
constexpr uint8_t VEC_FLAG_ID_0 = 0;
constexpr uint8_t VEC_FLAG_ID_1 = 1;
constexpr uint8_t VEC_FLAG_ID_2 = 2;
constexpr uint8_t VEC_FLAG_ID_3 = 3;
__aicore__ inline int64_t ROUND_UP(const int64_t x, const int64_t block_number)
{
    if (block_number > 0) {
        return (x + block_number - 1) / block_number * block_number;
    }
    return 0;
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilA2B(const T1 a, const T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
};

template <typename T1>
__aicore__ inline T1 weightCalculate(T1 a)
{
    if (a < 0) {
        a = -1 * a;
    }
    if (a < (float)1.0) {
        return (float)1.0 - a;
    }
    return 0.0;
};

template <typename T1>
__aicore__ inline T1 Min(const T1 a, const T1 b)
{
    return a < b ? a : b;
};

template <typename T1>
__aicore__ inline T1 Max(const T1 a, const T1 b)
{
    return a > b ? a : b;
};

__aicore__ inline float getCenterValue(const int64_t dstIdx, const float scale, const bool align_corners)
{
    if (align_corners) {
        return scale * dstIdx;
    } else {
        float rel = scale * (dstIdx + (float)0.5) - (float)0.5;
        return Max(rel, (float)0.0);
    }
};

__aicore__ inline float getLambda(const float i_rel_idx, const int64_t i_min)
{
    float i_lambda = Min(Max(static_cast<float>(i_rel_idx - i_min), (float)0.0), (float)1.0);
    return i_lambda;
};

__aicore__ inline bool FloatEqual(const float a, const float b)
{
    const float closeTo0 = 1e-6;
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
};

__aicore__ inline void calculateSingleCoreK(
    int64_t loopIndex, int64_t length, int64_t& xMin, int64_t& singleCoreK, 
    float scale_w, bool align_corners, int64_t wIn)
{
    singleCoreK = 0;
    xMin = getCenterValue(loopIndex, scale_w, align_corners);
    int64_t xMax = getCenterValue(loopIndex + length - 1, scale_w, align_corners);
    int64_t xMaxNext = Min(xMax + (int64_t)2, wIn);
    int64_t xMaxSize = Min(Max(xMaxNext - xMax, static_cast<int64_t>(0)), static_cast<int64_t>(2));
    singleCoreK = Max(xMax - xMin + xMaxSize, (int64_t)1);
    if ((singleCoreK + xMin) > wIn) {
        singleCoreK = wIn - xMin;
    }
}

__aicore__ inline void calculateRadioTensorW(
    int64_t loopIndex, int64_t length, LocalTensor<float> radioTensor,
    int64_t xMin, int64_t singleCoreK, float scale_w, bool align_corners, 
    int64_t wIn, int64_t slide_size_w)
{
    for (int64_t i = 0; i < length; i++) {
        float i_rel_idx = getCenterValue(i + loopIndex, scale_w, align_corners);
        int64_t i_min = Min(static_cast<int64_t>(i_rel_idx), wIn - 1);
        int64_t i_max = Min(i_min + (int64_t)1, wIn - 1);
        int64_t yIndexOffset = i_min - xMin;
        int64_t indexMin = yIndexOffset * slide_size_w + i;
        float i_lambda_1 = 0;
        float i_lambda_0 = 0;
        int64_t indexMax = 0;
        if (i_min == i_max) {
            radioTensor.SetValue(indexMin, 1);
        } else {
            i_lambda_1 = getLambda(i_rel_idx, i_min);
            i_lambda_0 = 1 - i_lambda_1;
            radioTensor.SetValue(indexMin, i_lambda_0);
            indexMax = (1 + yIndexOffset) * slide_size_w + i;
            radioTensor.SetValue(indexMax, i_lambda_1);
        }
    }
}

#endif  // UPSAMPLE_LINEAR_COMMON_H