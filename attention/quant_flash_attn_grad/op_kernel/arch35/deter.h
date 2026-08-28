/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON‑INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deter.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_DETER_H_
#define FLASH_ATTENTION_SCORE_GRAD_DETER_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_flash_attention_score_grad_common.h"

namespace commondef {
constexpr int64_t DETER_PREFIX_NUM = 132;
constexpr int64_t DETER_PREFIX_THRESHOLD = 128;
constexpr int64_t M_BASE_64 = 64;
constexpr int64_t N_BASE_128 = 128;

__aicore__ inline int64_t Square(int64_t num) { return num * num; }

__aicore__ inline int64_t Gcd(int64_t a, int64_t b)
{
    int64_t r;
    while (b > 0) {
        r = a % b;
        a = b;
        b = r;
    }
    return a;
}

__aicore__ inline int64_t AbsCeil(int64_t num1, int64_t num2)
{
    bool isNegative = (num1 < 0) || (num2 < 0);
    int64_t result = (abs(num1) + abs(num2) - 1) / abs(num2);
    return isNegative ? -result : result;
}

struct CoordinateInfo {
    int64_t batchId = 0;
    int64_t n2Idx = 0;
    int64_t gIdx = 0;
    int64_t s1Idx = 0;
    int64_t s2Idx = 0;

    int64_t s1Outer = 0;
    int64_t s2Outer = 0;
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    int64_t sparseMode = NO_MASK;

    int64_t mOffset = 0;
    int64_t nOffset = 0;

    int64_t m = 0;
    int64_t n = 0;
    int64_t p = 0;
    int64_t q = 0;
};

__aicore__ inline void InitCoordinateInfo(int64_t s1Outer, int64_t s2Outer, int64_t mOffset, int64_t nOffset,
                                          CoordinateInfo &coordinateInfo)
{
    coordinateInfo.s1Outer = s1Outer;
    coordinateInfo.s2Outer = s2Outer;
    coordinateInfo.mOffset = mOffset;
    coordinateInfo.nOffset = nOffset;
}

__aicore__ inline void CalDenseIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r, CoordinateInfo &coordinate)
{
    k = Min(k, b * m);
    if (j > k) {
        coordinate.batchId = -1;
        return;
    }

    int64_t p = (Ceil<int64_t>(r, m) - 1) * k + j;
    // Determine w and y
    int64_t w = p % b;
    w = (w != 0) ? w : b;
    int64_t y = Ceil<int64_t>(p, b);

    int64_t y1 = y % m;
    y1 = (y1 != 0) ? y1 : m;
    int64_t r1 = r % m;
    r1 = (r1 != 0) ? r1 : m;

    // Calculate x
    int64_t x = y1 + r1 - 1;
    if (x > m) {
        x -= m;
    }

    // Check if all values are within the valid ranges
    if (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n) {
        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    } else {
        coordinate.batchId = -1;
    }
    return;
}

__aicore__ inline void CalDenseIndexForSingleN(int64_t k, int64_t m, int64_t b, int64_t j, int64_t r, int64_t R, CoordinateInfo &coordinate)
{
    int64_t n = 1;
    int64_t ID = (j - 1) * R + r;
    int64_t num = m * n;

    int64_t delta1 = (ID - 1) / num + 1;
    int64_t delta = ID % num;
    delta = delta == 0 ? num : delta;

    int64_t g = Gcd(m, R);
    int64_t t1 = R / g;
    int64_t t2 = m / g;

    int64_t x = ((delta - 1) % m) + 1;
    int64_t y = (delta - 1) / m + 1;

    if (t1 < n) {
        int64_t n1 = (n % t1);
        n1 = n1 == 0 ? t1 : n1;
        if (y <= n - n1) {
            int64_t delta_adj = Ceil<int64_t>(y, t1);
            delta += delta_adj;
            if (delta > delta_adj * t2 * R) {
                delta -= t2 * R;
            }
            x = ((delta - 1) % m) + 1;
            y = (delta - 1) / m + 1;
        }
    }

    coordinate.batchId = delta1;
    coordinate.s1Idx = x;
    coordinate.s2Idx = y;
}

}

#endif // _FLASH_ATTENTION_SCORE_GRAD_DETER_H_