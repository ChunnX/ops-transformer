/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../common/constants.h"
#include "../common/types.h"
#include "../common/math.h"
#include "../hardware/memory.h"
#include "../hardware/sync.h"
#include "../hardware/layout.h"

namespace sage::compute {

#if defined(__CCE__)
using AscendC::RoundMode;
using AscendC::CMPMODE;

// ============================================================================
// Ultra-Low Register Residency Online Softmax Microkernels (Peak <= 6 Live Regs)
// ============================================================================

// Scale a pair of NZ score blocks across group halves by Q and K factors,
// apply causal mask (if enabled), and accumulate row maximums with <= 6 live registers
// using interleaved VALU and VLD/VST 4-way scheduling.
template <bool MASK>
__simd_callee__ inline __attribute__((always_inline)) void ScaleMaskAndMaxWideNZPair(
    __ubuf__ half *scores0, __ubuf__ half *scores1,
    AscendC::MicroAPI::RegTensor<half> &q0, AscendC::MicroAPI::RegTensor<half> &q1,
    __ubuf__ half *khalf0, __ubuf__ half *khalf1,
    uint16_t col, uint16_t row,
    int32_t causalLimit0, int32_t causalLimit1,
    AscendC::MicroAPI::RegTensor<int16_t> &laneOffset,
    AscendC::MicroAPI::RegTensor<half> &negative,
    AscendC::MicroAPI::RegTensor<half> &m0, AscendC::MicroAPI::RegTensor<half> &m1,
    AscendC::MicroAPI::MaskReg &h) {
    using namespace AscendC::MicroAPI;

    // Interleaved VALU/VLD/VST 4-way schedule: peak live registers <= 6
    RegTensor<half> k, c, x;

    // Slot 0: scores0, sub0 (khalf0, q0)
    LoadAlign<half, LoadDist::DIST_BLK>(k, khalf0 + col * 16);
    Mul(c, q0, k, h);
    LoadAlign(x, scores0 + col * 64 * 16 + row * 16 + 0);
    Mul(x, x, c, h);
    if constexpr (MASK) {
        int32_t t0 = causalLimit0 + row - col * 16;
        MaskReg valid0;
        CompareScalar<int16_t, CMPMODE::LE>(valid0, laneOffset, int16_t(t0), h);
        Select(x, x, negative, valid0);
    }
    Max(m0, m0, x, h);
    StoreAlign(scores0 + col * 64 * 16 + row * 16 + 0, x, h);

    // Slot 1: scores0, sub1 (khalf0, q1)
    Mul(c, q1, k, h);
    LoadAlign(x, scores0 + col * 64 * 16 + row * 16 + 128);
    Mul(x, x, c, h);
    if constexpr (MASK) {
        int32_t t1 = causalLimit0 + row - col * 16 + 8;
        MaskReg valid1;
        CompareScalar<int16_t, CMPMODE::LE>(valid1, laneOffset, int16_t(t1), h);
        Select(x, x, negative, valid1);
    }
    Max(m1, m1, x, h);
    StoreAlign(scores0 + col * 64 * 16 + row * 16 + 128, x, h);

    // Slot 2: scores1, sub0 (khalf1, q0)
    LoadAlign<half, LoadDist::DIST_BLK>(k, khalf1 + col * 16);
    Mul(c, q0, k, h);
    LoadAlign(x, scores1 + col * 64 * 16 + row * 16 + 0);
    Mul(x, x, c, h);
    if constexpr (MASK) {
        int32_t t2 = causalLimit1 + row - col * 16;
        MaskReg valid2;
        CompareScalar<int16_t, CMPMODE::LE>(valid2, laneOffset, int16_t(t2), h);
        Select(x, x, negative, valid2);
    }
    Max(m0, m0, x, h);
    StoreAlign(scores1 + col * 64 * 16 + row * 16 + 0, x, h);

    // Slot 3: scores1, sub1 (khalf1, q1)
    Mul(c, q1, k, h);
    LoadAlign(x, scores1 + col * 64 * 16 + row * 16 + 128);
    Mul(x, x, c, h);
    if constexpr (MASK) {
        int32_t t3 = causalLimit1 + row - col * 16 + 8;
        MaskReg valid3;
        CompareScalar<int16_t, CMPMODE::LE>(valid3, laneOffset, int16_t(t3), h);
        Select(x, x, negative, valid3);
    }
    Max(m1, m1, x, h);
    StoreAlign(scores1 + col * 64 * 16 + row * 16 + 128, x, h);
}

// Compute probabilities, row sums, and pack into NZ FP8 layout with <= 4 live registers
__simd_callee__ inline __attribute__((always_inline)) void ComputeProbAndPackNZSingle(
    __ubuf__ float8_e4m3_t *out,
    AscendC::MicroAPI::RegTensor<half> &x0,
    AscendC::MicroAPI::RegTensor<half> &x1,
    AscendC::MicroAPI::RegTensor<half> &m,
    AscendC::MicroAPI::RegTensor<float> &rowSum,
    uint32_t outOffset) {
    using namespace AscendC::MicroAPI;
    static constexpr CastTrait cast0{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast1{RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast2{RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast3{RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    auto f = CreateMask<float, MaskPattern::ALL>();
    auto h = CreateMask<half, MaskPattern::ALL>();
    auto b = CreateMask<uint8_t, MaskPattern::ALL>();

    RegTensor<float> p;
    RegTensor<float8_e4m3_t> pk0, pk_tmp;

    // Sub-lane 0
    ExpSub<float, half, RegLayout::ZERO>(p, x0, m, h);
    Cast<float8_e4m3_t, float, cast0>(pk0, p, f);
    Add(rowSum, rowSum, p, f);

    // Sub-lane 1
    ExpSub<float, half, RegLayout::ONE>(p, x0, m, h);
    Cast<float8_e4m3_t, float, cast2>(pk_tmp, p, f);
    Or((RegTensor<uint8_t> &)pk0, (RegTensor<uint8_t> &)pk0, (RegTensor<uint8_t> &)pk_tmp, b);
    Add(rowSum, rowSum, p, f);

    // Sub-lane 2
    ExpSub<float, half, RegLayout::ZERO>(p, x1, m, h);
    Cast<float8_e4m3_t, float, cast1>(pk_tmp, p, f);
    Or((RegTensor<uint8_t> &)pk0, (RegTensor<uint8_t> &)pk0, (RegTensor<uint8_t> &)pk_tmp, b);
    Add(rowSum, rowSum, p, f);

    // Sub-lane 3
    ExpSub<float, half, RegLayout::ONE>(p, x1, m, h);
    Cast<float8_e4m3_t, float, cast3>(pk_tmp, p, f);
    Or((RegTensor<uint8_t> &)pk0, (RegTensor<uint8_t> &)pk0, (RegTensor<uint8_t> &)pk_tmp, b);
    Add(rowSum, rowSum, p, f);

    StoreAlign((__ubuf__ uint8_t *)out + outOffset, (RegTensor<uint8_t> &)pk0, b);
}

// Update row maxima with FP8 shift and compute exponential rescaling ratio alpha = exp(oldMax - newMax)
__simd_callee__ inline AscendC::MicroAPI::RegTensor<float> UpdateSoftmaxRescaleFactors(
    __ubuf__ half *maxima, __ubuf__ half *blockMax, __ubuf__ float *alpha,
    AscendC::MicroAPI::MaskReg &f, AscendC::MicroAPI::MaskReg &h) {
    using namespace AscendC::MicroAPI;
    RegTensor<half> oldMax, newMax;
    RegTensor<float> ratio;
    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(oldMax, maxima);
    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(newMax, blockMax);
    Adds(newMax, newMax, half(LN_448), h);
    Max(newMax, newMax, oldMax, h);
    ExpSub<float, half, RegLayout::ZERO>(ratio, oldMax, newMax, h);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(maxima, newMax, f);
    StoreAlign(alpha, ratio, f);
    return ratio;
}

// Mask trailing elements for sequence padding and causal attention (non-wide)
template <bool CAUSAL = false, class T>
__simd_callee__ inline void MaskScoresTail(AscendC::MicroAPI::RegTensor<T> &x,
                                           AscendC::MicroAPI::RegTensor<T> &negative, uint32_t pos, uint32_t at,
                                           uint32_t sk, uint32_t sq) {
    using namespace AscendC::MicroAPI;
    uint32_t valid = at < sk ? (sk - at < 128 ? sk - at : 128) : 0;
    if constexpr (CAUSAL) {
        uint32_t causalLimit = pos >= at ? pos - at + 1 : 0;
        if (causalLimit < valid) valid = causalLimit;
    }
    if (pos >= sq) valid = 0;
    auto mask = UpdateMask<T>(valid);
    Select(x, x, negative, mask);
}

// ============================================================================
// SimdSoftmax Engine Class
// ============================================================================

class SimdSoftmax {
public:
    template <bool MASK>
    __simd_callee__ static inline void SoftmaxWideMaxCore(__ubuf__ half *scores, __ubuf__ half *scoresNext,
                                                          __ubuf__ half *maxima, __ubuf__ float *alpha,
                                                          __ubuf__ half *qhalf, __ubuf__ half *khalf,
                                                          int32_t queryInGroup = 0) {
        using namespace AscendC::MicroAPI;
        auto h = CreateMask<half, MaskPattern::ALL>();
        auto f = CreateMask<float, MaskPattern::ALL>();

        RegTensor<int16_t> laneOffset;
        RegTensor<half> negative;
        if constexpr (MASK) {
            RegTensor<int16_t> lane, rowIndex, colIndex, lowBits;
            Arange(lane, int16_t(0));
            ShiftRights(rowIndex, lane, int16_t(4), h);
            Duplicate(lowBits, int16_t(15));
            And(colIndex, lane, lowBits, h);
            Sub(laneOffset, colIndex, rowIndex, h);
            Duplicate(negative, half(-__builtin_inff()));
        }
        auto blockMax = (__ubuf__ half *)alpha;
        UnalignReg maxStore;
        for (uint16_t row = 0; row < 64; row += 16) {
            RegTensor<half> q0, m0;
            LoadAlign<half, LoadDist::DIST_E2B_B16>(q0, qhalf + row + 0);
            Duplicate(m0, half(SAGE_FP16_MIN));
            RegTensor<half> q1, m1;
            LoadAlign<half, LoadDist::DIST_E2B_B16>(q1, qhalf + row + 8);
            Duplicate(m1, half(SAGE_FP16_MIN));
            for (uint16_t col = 0; col < 16; ++col) {
                ScaleMaskAndMaxWideNZPair<MASK>(scores, scoresNext, q0, q1, khalf + 0, khalf + 256,
                                                col, row, queryInGroup, queryInGroup - 256,
                                                laneOffset, negative, m0, m1, h);
            }
            ReduceDataBlock<ReduceType::MAX>(m0, m0, h);
            ReduceDataBlock<ReduceType::MAX>(m1, m1, h);
            StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m0, maxStore, 8);
            StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m1, maxStore, 8);
        }
        StoreUnAlignPost(blockMax, maxStore, 0);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        UpdateSoftmaxRescaleFactors(maxima, (__ubuf__ half *)alpha, alpha, f, h);
    }

    template <bool FIRST>
    __simd_callee__ static inline void SoftmaxWideProbCore(__ubuf__ half *scores, __ubuf__ half *maxima,
                                                           __ubuf__ float8_e4m3_t *out,
                                                           __ubuf__ float *partialSums,
                                                           __ubuf__ float *scratch = nullptr) {
        using namespace AscendC::MicroAPI;
        if constexpr (FIRST) {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        auto f = CreateMask<float, MaskPattern::ALL>();
        auto sumMask = CreateMask<float, MaskPattern::VL8>();
        for (uint16_t row = 0; row < 64; row += 8) {
            RegTensor<half> m;
            RegTensor<float> rowSum;
            LoadAlign<half, LoadDist::DIST_E2B_B16>(m, maxima + row);
            if constexpr (FIRST)
                Duplicate(rowSum, 0.0f);
            else
                LoadAlign(rowSum, partialSums + row * 8);
            for (uint16_t col = 0; col < 8; col += 2) {
                RegTensor<half> x0, x1;
                LoadAlign(x0, scores + (col + 0) * 64 * 32 + row * 16);
                LoadAlign(x1, scores + (col + 0) * 64 * 32 + 64 * 16 + row * 16);

                RegTensor<half> next0, next1;
                LoadAlign(next0, scores + (col + 1) * 64 * 32 + row * 16);
                LoadAlign(next1, scores + (col + 1) * 64 * 32 + 64 * 16 + row * 16);

                ComputeProbAndPackNZSingle(out, x0, x1, m, rowSum, (col + 0) * 64 * 64 + row * 32);
                ComputeProbAndPackNZSingle(out, next0, next1, m, rowSum, (col + 1) * 64 * 64 + row * 32);
            }
            if constexpr (FIRST) {
                StoreAlign(partialSums + row * 8, rowSum, f);
            } else {
                ReduceDataBlock<ReduceType::SUM>(rowSum, rowSum, f);
                StoreAlign(scratch + row, rowSum, sumMask);
            }
        }
    }

    template <bool MASK>
    __simd_vf__ static inline void SoftmaxWideFirstVF(__ubuf__ half *scores, __ubuf__ half *scoresNext,
                                                      __ubuf__ half *maxima, __ubuf__ float *alpha,
                                                      __ubuf__ float8_e4m3_t *out, __ubuf__ half *qhalf,
                                                      __ubuf__ half *khalf, __ubuf__ float *partialSums,
                                                      int32_t queryInGroup = 0) {
        SoftmaxWideMaxCore<MASK>(scores, scoresNext, maxima, alpha, qhalf, khalf, queryInGroup);
        SoftmaxWideProbCore<true>(scores, maxima, out, partialSums);
    }

    __simd_vf__ static inline void SoftmaxWideSecondVF(__ubuf__ half *scores, __ubuf__ half *maxima,
                                                       __ubuf__ float *alpha, __ubuf__ float8_e4m3_t *out,
                                                       __ubuf__ float *partialSums) {
        SoftmaxWideProbCore<false>(scores, maxima, out, partialSums, alpha + 64);
    }

    template <uint32_t ROWS, uint32_t COLS, bool FULL, uint32_t NZ_PAD_ROWS, bool CAUSAL = false>
    __simd_vf__ static inline void SoftmaxHalfVF(__ubuf__ half *scores, __ubuf__ half *maxima, __ubuf__ float *denom,
                                                 __ubuf__ float *alpha, __ubuf__ float8_e4m3_t *out, __ubuf__ half *qhalf,
                                                 __ubuf__ half *khalf, __ubuf__ float *scratch, uint32_t qstart, uint32_t sq,
                                                 uint32_t start, uint32_t sk) {
        static_assert(ROWS == 64 && COLS == 256, "half softmax requires a 64x256 AIV tile");
        using namespace AscendC::MicroAPI;
        static constexpr CastTrait toFp8{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        auto h = CreateMask<half, MaskPattern::ALL>();
        auto blockMax = (__ubuf__ half *)alpha;
        UnalignReg maxStore;
        RegTensor<half> kh0, kh1, negative;
        LoadAlign(kh0, khalf);
        LoadAlign(kh1, khalf + 128);
        Duplicate(negative, half(-__builtin_inff()));

        for (uint16_t row = 0; row < ROWS; row += 4) {
            const uint16_t r0 = row, r1 = row + 1, r2 = row + 2, r3 = row + 3;
            RegTensor<half> qh0, qh1, qh2, qh3;
            LoadAlign<half, LoadDist::DIST_BRC_B16>(qh0, qhalf + r0);
            LoadAlign<half, LoadDist::DIST_BRC_B16>(qh1, qhalf + r1);
            LoadAlign<half, LoadDist::DIST_BRC_B16>(qh2, qhalf + r2);
            LoadAlign<half, LoadDist::DIST_BRC_B16>(qh3, qhalf + r3);

            RegTensor<half> x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3;
            LoadAlign(x0_0, scores + r0 * COLS + 0);
            LoadAlign(x1_0, scores + r0 * COLS + 128);
            LoadAlign(x0_1, scores + r1 * COLS + 0);
            LoadAlign(x1_1, scores + r1 * COLS + 128);
            LoadAlign(x0_2, scores + r2 * COLS + 0);
            LoadAlign(x1_2, scores + r2 * COLS + 128);
            LoadAlign(x0_3, scores + r3 * COLS + 0);
            LoadAlign(x1_3, scores + r3 * COLS + 128);

            RegTensor<half> c0_0, c0_1, c0_2, c0_3;
            RegTensor<half> c1_0, c1_1, c1_2, c1_3;
            Mul(c0_0, qh0, kh0, h);
            Mul(c0_1, qh1, kh0, h);
            Mul(c0_2, qh2, kh0, h);
            Mul(c0_3, qh3, kh0, h);
            Mul(c1_0, qh0, kh1, h);
            Mul(c1_1, qh1, kh1, h);
            Mul(c1_2, qh2, kh1, h);
            Mul(c1_3, qh3, kh1, h);

            Mul(x0_0, x0_0, c0_0, h);
            Mul(x0_1, x0_1, c0_1, h);
            Mul(x0_2, x0_2, c0_2, h);
            Mul(x0_3, x0_3, c0_3, h);
            Mul(x1_0, x1_0, c1_0, h);
            Mul(x1_1, x1_1, c1_1, h);
            Mul(x1_2, x1_2, c1_2, h);
            Mul(x1_3, x1_3, c1_3, h);

            if constexpr (!FULL) {
                const uint32_t pos0 = qstart + r0, pos1 = qstart + r1, pos2 = qstart + r2, pos3 = qstart + r3;
                MaskScoresTail<CAUSAL>(x0_0, negative, pos0, start + 0, sk, sq);
                MaskScoresTail<CAUSAL>(x0_1, negative, pos1, start + 0, sk, sq);
                MaskScoresTail<CAUSAL>(x0_2, negative, pos2, start + 0, sk, sq);
                MaskScoresTail<CAUSAL>(x0_3, negative, pos3, start + 0, sk, sq);
                MaskScoresTail<CAUSAL>(x1_0, negative, pos0, start + 128, sk, sq);
                MaskScoresTail<CAUSAL>(x1_1, negative, pos1, start + 128, sk, sq);
                MaskScoresTail<CAUSAL>(x1_2, negative, pos2, start + 128, sk, sq);
                MaskScoresTail<CAUSAL>(x1_3, negative, pos3, start + 128, sk, sq);
            }

            StoreAlign(scores + r0 * COLS + 0, x0_0, h);
            StoreAlign(scores + r1 * COLS + 0, x0_1, h);
            StoreAlign(scores + r2 * COLS + 0, x0_2, h);
            StoreAlign(scores + r3 * COLS + 0, x0_3, h);
            StoreAlign(scores + r0 * COLS + 128, x1_0, h);
            StoreAlign(scores + r1 * COLS + 128, x1_1, h);
            StoreAlign(scores + r2 * COLS + 128, x1_2, h);
            StoreAlign(scores + r3 * COLS + 128, x1_3, h);

            RegTensor<half> rowMax0, rowMax1, rowMax2, rowMax3;
            Max(rowMax0, x0_0, x1_0, h);
            Max(rowMax1, x0_1, x1_1, h);
            Max(rowMax2, x0_2, x1_2, h);
            Max(rowMax3, x0_3, x1_3, h);

            RegTensor<half> m0, m1, m2, m3;
            ReduceMax(m0, rowMax0, h);
            ReduceMax(m1, rowMax1, h);
            ReduceMax(m2, rowMax2, h);
            ReduceMax(m3, rowMax3, h);

            StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m0, maxStore, 1);
            StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m1, maxStore, 1);
            StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m2, maxStore, 1);
            StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m3, maxStore, 1);
        }
        StoreUnAlignPost(blockMax, maxStore, 0);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        auto f = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> oldDen;
        LoadAlign(oldDen, denom);
        RegTensor<float> ratio = UpdateSoftmaxRescaleFactors(maxima, (__ubuf__ half *)alpha, alpha, f, h);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        uint32_t halfChunk = 32;
        MaskReg first32 = UpdateMask<float>(halfChunk);
        MaskReg last32;
        Not(last32, first32, f);
        auto blockSum = scratch;
        UnalignReg sumStore;

        for (uint16_t row = 0; row < ROWS; row += 2) {
            const uint16_t row_a = row, row_b = row + 1;
            RegTensor<half> m_a, m_b;
            RegTensor<half> x0_a, x1_a, x2_a, x3_a;
            RegTensor<half> x0_b, x1_b, x2_b, x3_b;

            LoadAlign<half, LoadDist::DIST_BRC_B16>(m_a, maxima + row_a);
            LoadAlign<half, LoadDist::DIST_BRC_B16>(m_b, maxima + row_b);

            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x0_a, scores + row_a * COLS + 0);
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x1_a, scores + row_a * COLS + 64);
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x2_a, scores + row_a * COLS + 128);
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x3_a, scores + row_a * COLS + 192);

            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x0_b, scores + row_b * COLS + 0);
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x1_b, scores + row_b * COLS + 64);
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x2_b, scores + row_b * COLS + 128);
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x3_b, scores + row_b * COLS + 192);

            RegTensor<float> p0_a, p0_b, p1_a, p1_b, p2_a, p2_b, p3_a, p3_b;
            ExpSub<float, half, RegLayout::ZERO>(p0_a, x0_a, m_a, h);
            ExpSub<float, half, RegLayout::ZERO>(p0_b, x0_b, m_b, h);
            ExpSub<float, half, RegLayout::ZERO>(p1_a, x1_a, m_a, h);
            ExpSub<float, half, RegLayout::ZERO>(p1_b, x1_b, m_b, h);
            ExpSub<float, half, RegLayout::ZERO>(p2_a, x2_a, m_a, h);
            ExpSub<float, half, RegLayout::ZERO>(p2_b, x2_b, m_b, h);
            ExpSub<float, half, RegLayout::ZERO>(p3_a, x3_a, m_a, h);
            ExpSub<float, half, RegLayout::ZERO>(p3_b, x3_b, m_b, h);

            RegTensor<float8_e4m3_t> packed0_a, packed1_a, packed2_a, packed3_a;
            RegTensor<float8_e4m3_t> packed0_b, packed1_b, packed2_b, packed3_b;

            Cast<float8_e4m3_t, float, toFp8>(packed0_a, p0_a, f);
            Cast<float8_e4m3_t, float, toFp8>(packed0_b, p0_b, f);
            Cast<float8_e4m3_t, float, toFp8>(packed1_a, p1_a, f);
            Cast<float8_e4m3_t, float, toFp8>(packed1_b, p1_b, f);

            RegTensor<float> s01_a, s01_b, s23_a, s23_b;
            Add(s01_a, p0_a, p1_a, f);
            Add(s01_b, p0_b, p1_b, f);
            Add(s23_a, p2_a, p3_a, f);
            Add(s23_b, p2_b, p3_b, f);

            Cast<float8_e4m3_t, float, toFp8>(packed2_a, p2_a, f);
            Cast<float8_e4m3_t, float, toFp8>(packed2_b, p2_b, f);
            Cast<float8_e4m3_t, float, toFp8>(packed3_a, p3_a, f);
            Cast<float8_e4m3_t, float, toFp8>(packed3_b, p3_b, f);

            RegTensor<float> rowSum_a, rowSum_b;
            Add(rowSum_a, s01_a, s23_a, f);
            Add(rowSum_b, s01_b, s23_b, f);

            constexpr uint32_t nzStride = (ROWS + NZ_PAD_ROWS) * 32;
            StoreNZPlanePair<nzStride>(out, 0, row_a, packed0_a, first32, last32);
            StoreNZPlanePair<nzStride>(out, 0, row_b, packed0_b, first32, last32);
            StoreNZPlanePair<nzStride>(out, 2, row_a, packed1_a, first32, last32);
            StoreNZPlanePair<nzStride>(out, 2, row_b, packed1_b, first32, last32);
            StoreNZPlanePair<nzStride>(out, 4, row_a, packed2_a, first32, last32);
            StoreNZPlanePair<nzStride>(out, 4, row_b, packed2_b, first32, last32);
            StoreNZPlanePair<nzStride>(out, 6, row_a, packed3_a, first32, last32);
            StoreNZPlanePair<nzStride>(out, 6, row_b, packed3_b, first32, last32);

            RegTensor<float> sum_a, sum_b;
            ReduceSum(sum_a, rowSum_a, f);
            ReduceSum(sum_b, rowSum_b, f);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(blockSum, sum_a, sumStore, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(blockSum, sum_b, sumStore, 1);
        }
        StoreUnAlignPost(blockSum, sumStore, 0);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> sum;
        LoadAlign(sum, scratch);
        MulDstAdd(oldDen, ratio, sum, f);
        StoreAlign(denom, oldDen, f);
    }
};

// ============================================================================
// In-Kernel Fused LSE Calculation Microkernel
// ============================================================================
// The softmax denominator carries the ln(448) full-scale shift, which cancels
// exactly against the -ln(448) offset tracked in the maxima register, so
// ln(denom) + max is the true natural-log LSE of the smoothed scores. The
// public contract (upstream core.py, applied in python _ops.py) divides this
// kernel value by log2(e) again and adds the smooth_k correction, so the value
// written to GM must be that natural-domain LSE scaled by log2(e). The multiply
// therefore happens AFTER the max add: scaling earlier would skew the max by
// log2(e); omitting it would double-convert with the Python layer. Numeric
// test [7] pins the whole chain to 1e-5 against a double-precision reference.
__aicore__ inline void StoreTileLse(
    AscendC::GlobalTensor<float> &lg, AscendC::LocalTensor<float> denom,
    AscendC::LocalTensor<half> maxima, AscendC::LocalTensor<float> scratch,
    uint32_t first, uint32_t validRows, uint32_t b, uint32_t h, const SageParams &p)
{
    using namespace AscendC;
    Sync<HardEvent::MTE3_S>();
    Sync<HardEvent::V_S>();
    if (!validRows) return;

    auto lse = scratch;
    auto maxF = scratch[HALF];
    DataCopy(lse, denom, HALF);
    Sync<HardEvent::MTE2_V>();
    Ln(lse, lse, HALF);
    PipeBarrier<PIPE_V>();
    Cast(maxF, maxima, RoundMode::CAST_NONE, HALF);
    PipeBarrier<PIPE_V>();
    Add(lse, lse, maxF, HALF);
    PipeBarrier<PIPE_V>();
    Muls(lse, lse, LOG2_E, HALF);
    PipeBarrier<PIPE_V>();
    Sync<HardEvent::V_MTE3>();
    DataCopyExtParams cp{1, validRows * uint32_t(sizeof(float)), 0, 0, 0};
    DataCopyPad(lg[(uint64_t(b) * p.hq + h) * p.sq + first], lse, cp);
    Sync<HardEvent::MTE3_S>();
}

#endif // __CCE__

} // namespace sage::compute
