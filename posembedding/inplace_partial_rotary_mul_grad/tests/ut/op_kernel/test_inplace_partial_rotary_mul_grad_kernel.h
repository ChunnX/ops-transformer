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
 * \file test_inplace_partial_rotary_mul_grad_kernel.h
 * \brief Tiling data struct definitions and reference helpers for InplacePartialRotaryMulGrad kernel UT
 */
#ifndef TEST_INPLACE_PARTIAL_ROTARY_MUL_GRAD_KERNEL_H
#define TEST_INPLACE_PARTIAL_ROTARY_MUL_GRAD_KERNEL_H

#include <cstdint>
#include <vector>

// Tiling data struct — mirrors the base tiling data layout exactly
struct InplacePartialRotaryMulGradUTTilingData {
    int64_t b;
    int64_t s;
    int64_t d;
    int64_t n;
    int64_t blockNumB;
    int64_t blockFactorB;
    int64_t blockNumS;
    int64_t blockFactorS;
    int64_t ubFactorS;
    int64_t ubFactorB;
    int64_t ubLoopNumN;
    int64_t ubFactorN;
    int64_t ubTailFactorN;
    int64_t usedCoreNum;
    int64_t rotaryMode;
    int64_t sliceStart;
    int64_t sliceEnd;
    int64_t sliceLength;
    int64_t dSplitCoef;
};

// Tiling data struct for AB template
struct InplacePartialRotaryMulGradABUTTilingData {
    int64_t b;
    int64_t s;
    int64_t d;
    int64_t n;
    int64_t dAlign;
    int64_t dSplitCoef;
    int64_t blockNumBS;
    int64_t blockFactorBS;
    int64_t blockTailBS;
    int64_t blockNumN;
    int64_t blockFactorN;
    int64_t blockTailN;
    int64_t ubFactorBS;
    int64_t ubFactorN;
    int64_t usedCoreNum;
    int64_t rotaryMode;
    int64_t sliceStart;
    int64_t sliceEnd;
    int64_t sliceLength;
};

// Tiling keys
constexpr uint64_t TILING_KEY_ABA = 201;
constexpr uint64_t TILING_KEY_BA = 202;
constexpr uint64_t TILING_KEY_BAB = 203;
constexpr uint64_t TILING_KEY_AB = 204;
constexpr uint64_t TILING_KEY_A = 205;
constexpr uint64_t TILING_KEY_B = 206;
constexpr uint64_t TILING_KEY_EMPTY = 403;

/// BSND layout reference (BAB template)
inline void InterleaveGradRefBSND(const std::vector<float> &dy, const std::vector<float> &cos,
                                  const std::vector<float> &sin, std::vector<float> &dx, int64_t b, int64_t s,
                                  int64_t n, int64_t d, int64_t sliceStart, int64_t sliceEnd)
{
    int64_t cosD = sliceEnd - sliceStart;
    for (int64_t bi = 0; bi < b; bi++) {
        for (int64_t si = 0; si < s; si++) {
            int64_t cosBase = si * cosD;
            for (int64_t ni = 0; ni < n; ni++) {
                int64_t groupOffset = ((bi * s + si) * n + ni) * d;
                for (int64_t di = 0; di < d; di++) {
                    dx[groupOffset + di] = dy[groupOffset + di];
                }
                for (int64_t k = 0; k < cosD / 2; k++) {
                    int64_t idx0 = groupOffset + sliceStart + 2 * k;
                    int64_t idx1 = groupOffset + sliceStart + 2 * k + 1;
                    int64_t c0 = cosBase + 2 * k;
                    int64_t c1 = cosBase + 2 * k + 1;
                    dx[idx0] = cos[c0] * dy[idx0] + sin[c1] * dy[idx1];
                    dx[idx1] = cos[c1] * dy[idx1] - sin[c0] * dy[idx0];
                }
            }
        }
    }
}

/// SBND layout reference (AB template)
inline void InterleaveGradRefSBND(const std::vector<float> &dy, const std::vector<float> &cos,
                                  const std::vector<float> &sin, std::vector<float> &dx, int64_t s, int64_t b,
                                  int64_t n, int64_t d, int64_t sliceStart, int64_t sliceEnd)
{
    int64_t cosD = sliceEnd - sliceStart;
    for (int64_t si = 0; si < s; si++) {
        int64_t cosBase = si * cosD;
        for (int64_t bi = 0; bi < b; bi++) {
            for (int64_t ni = 0; ni < n; ni++) {
                int64_t groupOffset = ((si * b + bi) * n + ni) * d;
                for (int64_t di = 0; di < d; di++) {
                    dx[groupOffset + di] = dy[groupOffset + di];
                }
                for (int64_t k = 0; k < cosD / 2; k++) {
                    int64_t idx0 = groupOffset + sliceStart + 2 * k;
                    int64_t idx1 = groupOffset + sliceStart + 2 * k + 1;
                    int64_t c0 = cosBase + 2 * k;
                    int64_t c1 = cosBase + 2 * k + 1;
                    dx[idx0] = cos[c0] * dy[idx0] + sin[c1] * dy[idx1];
                    dx[idx1] = cos[c1] * dy[idx1] - sin[c0] * dy[idx0];
                }
            }
        }
    }
}

/// BNSD layout reference (ABA&BA template)
inline void InterleaveGradRefBNSD(const std::vector<float> &dy, const std::vector<float> &cos,
                                  const std::vector<float> &sin, std::vector<float> &dx, int64_t b, int64_t n,
                                  int64_t s, int64_t d, int64_t sliceStart, int64_t sliceEnd, bool isBroadCast)
{
    int64_t cosD = sliceEnd - sliceStart;
    for (int64_t bi = 0; bi < b; bi++) {
        for (int64_t ni = 0; ni < n; ni++) {
            for (int64_t si = 0; si < s; si++) {
                int64_t groupOffset = (bi * n * s + ni * s + si) * d;
                int64_t cosBase;
                if (isBroadCast) {
                    cosBase = si * cosD; // cos broadcast on B
                } else {
                    cosBase = bi * s * cosD + si * cosD;
                }
                for (int64_t di = 0; di < d; di++) {
                    dx[groupOffset + di] = dy[groupOffset + di];
                }
                for (int64_t k = 0; k < cosD / 2; k++) {
                    int64_t idx0 = groupOffset + sliceStart + 2 * k;
                    int64_t idx1 = groupOffset + sliceStart + 2 * k + 1;
                    int64_t c0 = cosBase + 2 * k;
                    int64_t c1 = cosBase + 2 * k + 1;
                    dx[idx0] = cos[c0] * dy[idx0] + sin[c1] * dy[idx1];
                    dx[idx1] = cos[c1] * dy[idx1] - sin[c0] * dy[idx0];
                }
            }
        }
    }
}

/// NO_BROADCAST / BROADCAST_BSN reference (A&B template)
inline void InterleaveGradRefAB(const std::vector<float> &dy, const std::vector<float> &cos,
                                const std::vector<float> &sin, std::vector<float> &dx, int64_t b, int64_t n, int64_t s,
                                int64_t d, int64_t sliceStart, int64_t sliceEnd, bool isBroadCast)
{
    int64_t cosD = sliceEnd - sliceStart;
    int64_t total = b * n * s;
    for (int64_t bi = 0; bi < total; bi++) {
        int64_t groupOffset = bi * d;
        int64_t cosBase = isBroadCast ? 0 : bi * cosD;
        for (int64_t di = 0; di < d; di++) {
            dx[groupOffset + di] = dy[groupOffset + di];
        }
        for (int64_t k = 0; k < cosD / 2; k++) {
            int64_t idx0 = groupOffset + sliceStart + 2 * k;
            int64_t idx1 = groupOffset + sliceStart + 2 * k + 1;
            int64_t c0 = cosBase + 2 * k;
            int64_t c1 = cosBase + 2 * k + 1;
            dx[idx0] = cos[c0] * dy[idx0] + sin[c1] * dy[idx1];
            dx[idx1] = cos[c1] * dy[idx1] - sin[c0] * dy[idx0];
        }
    }
}

#endif // TEST_INPLACE_PARTIAL_ROTARY_MUL_GRAD_KERNEL_H
