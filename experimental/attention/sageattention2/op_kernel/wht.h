/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#if defined(__CCE_AICORE__) || defined(__BISHENG__) || defined(ENABLE_ASCEND_C) || defined(__DAV_C310__) || defined(__DAV_C350__) || (defined(__has_include) && __has_include("kernel_operator.h") && !defined(SAGE_STANDALONE_TEST))
#include "common.h"
#define SAGE_WHT_USE_ASCENDC 1
#else
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#define SAGE_WHT_USE_ASCENDC 0
#endif

namespace Sage {

// =================================================================================================
// Fast Walsh-Hadamard Transform (FWHT) for Incoherent Outlier Suppression in SageAttention2
// =================================================================================================
// Mathematical properties:
// 1. Orthogonality: H_D * H_D^T = I_D, where H_D is symmetric (H_D = H_D^T).
// 2. Attention Score Invariance:
//    Q_tilde = Q * H_D, K_tilde = K * H_D ==> Q_tilde * K_tilde^T = Q * H_D * H_D^T * K^T = Q * K^T.
// 3. Peak Outlier Suppression:
//    An isolated spike M in channel c is spread across all D channels to magnitude M / sqrt(D).
//    For D = 64:  peak suppression ratio = sqrt(64)  = 8.0x  (> 3.0x threshold).
//    For D = 128: peak suppression ratio = sqrt(128) = 11.31x (> 3.0x threshold).
// 4. Butterfly Complexity:
//    O(D * log2(D)) additions and subtractions. Zero floating-point multiplication in butterfly.
//    D = 64:  6 stages * 64 additions/subtractions = 384 ops per row.
//    D = 128: 7 stages * 128 additions/subtractions = 896 ops per row.
// =================================================================================================

template <uint32_t D>
struct FastWalshHadamardTransform {
    static_assert(D == 64 || D == 128, "FastWalshHadamardTransform only supports head_dim D=64 and D=128");

    static constexpr uint32_t DIM = D;
    static constexpr uint32_t STAGES = (D == 64) ? 6 : 7;
    // Exact mathematical scaling factor: 1 / sqrt(D) in full double precision
    // 1 / sqrt(64) = 0.125 (exact 2^-3); 1 / sqrt(128) = 1 / (8 * sqrt(2)) ~ 0.08838834764831844621517441865392
    static constexpr double INV_SQRT_D = (D == 64) ? 0.125 : 0.08838834764831844621517441865392;
    static constexpr float INV_SQRT_D_F32 = (D == 64) ? 0.125f : 0.088388347648318446f;

    FastWalshHadamardTransform() = default;

    template <typename T>
    inline FastWalshHadamardTransform(T *x, bool normalize = true)
    {
        Transform(x, normalize);
    }

    template <typename T>
    inline void operator()(T *x, bool normalize = true) const
    {
        Transform(x, normalize);
    }

    // ---------------------------------------------------------------------------------------------
    // In-place single-row FWHT butterfly algorithm (Portable C++)
    // ---------------------------------------------------------------------------------------------
    template <typename T>
    static inline void Transform(T *x, bool normalize = true)
    {
        // Butterfly execution: stage h = 1, 2, 4, ..., D/2
        #pragma unroll
        for (uint32_t h = 1; h < D; h <<= 1) {
            const uint32_t step = h << 1;
            for (uint32_t i = 0; i < D; i += step) {
                #pragma unroll 4
                for (uint32_t j = i; j < i + h; ++j) {
                    T u = x[j];
                    T v = x[j + h];
                    x[j] = static_cast<T>(u + v);
                    x[j + h] = static_cast<T>(u - v);
                }
            }
        }

        if (normalize) {
            const T scale = static_cast<T>(INV_SQRT_D);
            #pragma unroll 8
            for (uint32_t i = 0; i < D; ++i) {
                x[i] = static_cast<T>(x[i] * scale);
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // In-place multi-row FWHT with 2-row instruction interleaving (Portable C++)
    // Dual-row interleaving hides scalar/vector ALU RAW latency bubbles.
    // ---------------------------------------------------------------------------------------------
    template <typename T>
    static inline void TransformRows(T *data, uint32_t numRows, uint32_t stride = D, bool normalize = true)
    {
        uint32_t row = 0;
        for (; row + 1 < numRows; row += 2) {
            T *ra = data + row * stride;
            T *rb = data + (row + 1) * stride;

            #pragma unroll
            for (uint32_t h = 1; h < D; h <<= 1) {
                const uint32_t step = h << 1;
                for (uint32_t i = 0; i < D; i += step) {
                    #pragma unroll 2
                    for (uint32_t j = i; j < i + h; ++j) {
                        T ua = ra[j];
                        T va = ra[j + h];
                        T ub = rb[j];
                        T vb = rb[j + h];

                        // Interleaved Add and Sub across row_a and row_b
                        T sa = static_cast<T>(ua + va);
                        T sb = static_cast<T>(ub + vb);
                        T da = static_cast<T>(ua - va);
                        T db = static_cast<T>(ub - vb);

                        ra[j] = sa;
                        ra[j + h] = da;
                        rb[j] = sb;
                        rb[j + h] = db;
                    }
                }
            }

            if (normalize) {
                const T scale = static_cast<T>(INV_SQRT_D);
                #pragma unroll 8
                for (uint32_t i = 0; i < D; ++i) {
                    ra[i] = static_cast<T>(ra[i] * scale);
                    rb[i] = static_cast<T>(rb[i] * scale);
                }
            }
        }

        // Tail row if numRows is odd
        if (row < numRows) {
            Transform(data + row * stride, normalize);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Reference matrix generation (Sylvester Hadamard Matrix H_D) for validation & test suites
    // ---------------------------------------------------------------------------------------------
    template <typename MatrixT = double>
    static inline void GenerateMatrix(MatrixT *H, bool normalize = true)
    {
        // Start with H_1 = [1]
        std::vector<MatrixT> current = {1.0};
        uint32_t curSize = 1;

        while (curSize < D) {
            uint32_t nextSize = curSize * 2;
            std::vector<MatrixT> next(nextSize * nextSize, 0.0);
            for (uint32_t r = 0; r < curSize; ++r) {
                for (uint32_t c = 0; c < curSize; ++c) {
                    MatrixT val = current[r * curSize + c];
                    next[r * nextSize + c] = val;                          // Top-left: H
                    next[r * nextSize + (c + curSize)] = val;              // Top-right: H
                    next[(r + curSize) * nextSize + c] = val;              // Bottom-left: H
                    next[(r + curSize) * nextSize + (c + curSize)] = -val; // Bottom-right: -H
                }
            }
            current = std::move(next);
            curSize = nextSize;
        }

        const MatrixT scale = normalize ? static_cast<MatrixT>(INV_SQRT_D) : MatrixT(1.0);
        for (uint32_t i = 0; i < D * D; ++i) {
            H[i] = current[i] * scale;
        }
    }

#if SAGE_WHT_USE_ASCENDC
    // ---------------------------------------------------------------------------------------------
    // AscendC MicroAPI SIMD Vector implementation for AIV cores
    // Operates on Unified Buffer (UB) data buffers with vector instruction interleaving.
    // ---------------------------------------------------------------------------------------------
    template <typename T>
    __simd_vf__ static inline void TransformRowsVF(__ubuf__ T *data, uint32_t numRows, uint32_t stride = D,
                                                   bool normalize = true)
    {
        using namespace AscendC::MicroAPI;
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        uint32_t row = 0;
        for (; row + 1 < numRows; row += 2) {
            __ubuf__ T *ra = data + row * stride;
            __ubuf__ T *rb = data + (row + 1) * stride;

            #pragma unroll
            for (uint32_t h = 1; h < D; h <<= 1) {
                // When D == 128 and h == 64: the two halves are exact 64-element blocks (256 bytes)
                // mapped directly into RegTensor<float> vectors for single-instruction SIMD execution.
                if constexpr (D == 128 && AscendC::IsSameType<T, float>::value) {
                    if (h == 64) {
                        auto full = CreateMask<float, MaskPattern::ALL>();
                        RegTensor<float> lo_a, hi_a, lo_b, hi_b;
                        LoadAlign(lo_a, ra);
                        LoadAlign(hi_a, ra + 64);
                        LoadAlign(lo_b, rb);
                        LoadAlign(hi_b, rb + 64);

                        RegTensor<float> sa, sb, da, db;
                        // Interleaved Add and Sub eliminating RAW pipeline dependency bubbles
                        Add(sa, lo_a, hi_a, full);
                        Add(sb, lo_b, hi_b, full);
                        Sub(da, lo_a, hi_a, full);
                        Sub(db, lo_b, hi_b, full);

                        if (normalize) {
                            Muls(sa, sa, INV_SQRT_D, full);
                            Muls(sb, sb, INV_SQRT_D, full);
                            Muls(da, da, INV_SQRT_D, full);
                            Muls(db, db, INV_SQRT_D, full);
                        }

                        StoreAlign(ra, sa, full);
                        StoreAlign(ra + 64, da, full);
                        StoreAlign(rb, sb, full);
                        StoreAlign(rb + 64, db, full);
                        continue;
                    }
                }

                const uint32_t step = h << 1;
                for (uint32_t i = 0; i < D; i += step) {
                    #pragma unroll 2
                    for (uint32_t j = i; j < i + h; ++j) {
                        T ua = ra[j];
                        T va = ra[j + h];
                        T ub = rb[j];
                        T vb = rb[j + h];

                        T sa = static_cast<T>(ua + va);
                        T sb = static_cast<T>(ub + vb);
                        T da = static_cast<T>(ua - va);
                        T db = static_cast<T>(ub - vb);

                        ra[j] = sa;
                        ra[j + h] = da;
                        rb[j] = sb;
                        rb[j + h] = db;
                    }
                }
            }

            if (normalize) {
                if constexpr (AscendC::IsSameType<T, float>::value) {
                    auto full = CreateMask<float, MaskPattern::ALL>();
                    RegTensor<float> va0, vb0;
                    LoadAlign(va0, ra);
                    LoadAlign(vb0, rb);
                    Muls(va0, va0, INV_SQRT_D, full);
                    Muls(vb0, vb0, INV_SQRT_D, full);
                    StoreAlign(ra, va0, full);
                    StoreAlign(rb, vb0, full);
                    if constexpr (D == 128) {
                        RegTensor<float> va1, vb1;
                        LoadAlign(va1, ra + 64);
                        LoadAlign(vb1, rb + 64);
                        Muls(va1, va1, INV_SQRT_D, full);
                        Muls(vb1, vb1, INV_SQRT_D, full);
                        StoreAlign(ra + 64, va1, full);
                        StoreAlign(rb + 64, vb1, full);
                    }
                } else {
                    for (uint32_t i = 0; i < D; ++i) {
                        ra[i] = static_cast<T>(ra[i] * static_cast<T>(INV_SQRT_D));
                        rb[i] = static_cast<T>(rb[i] * static_cast<T>(INV_SQRT_D));
                    }
                }
            }
        }

        if (row < numRows) {
            Transform(data + row * stride, normalize);
        }

        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
#endif
};

// =================================================================================================
// Convenient Free Function Wrappers
// =================================================================================================
template <uint32_t D, typename T>
inline void FastWalshHadamardTransformRows(T *data, uint32_t numRows, uint32_t stride = D, bool normalize = true)
{
    FastWalshHadamardTransform<D>::TransformRows(data, numRows, stride, normalize);
}

} // namespace Sage
