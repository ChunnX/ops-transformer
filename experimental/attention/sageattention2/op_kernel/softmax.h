/*
 * Adapted from ops-transformer attention/common/op_kernel/arch35/vf/
 * vf_basic_block_unaligned64_update.h and vf_mul_sel_softmaxflashv2_cast_nz.h.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the repository root.
 */
#pragma once
#include "common.h"
namespace Sage {

// Mask trailing elements for sequence padding and causal attention.
template <class T>
__simd_callee__ inline void MaskScoresTail(AscendC::MicroAPI::RegTensor<T> &x,
                                           AscendC::MicroAPI::RegTensor<T> &negative, uint32_t pos, uint32_t at,
                                           uint32_t sk, uint32_t sq, uint32_t causal)
{
    using namespace AscendC::MicroAPI;
    uint32_t valid = at < sk ? Min(uint32_t(128), sk - at) : 0;
    if (causal) {
        valid = Min(valid, pos >= at ? pos - at + 1 : 0);
    }
    if (pos >= sq) {
        valid = 0;
    }
    auto mask = UpdateMask<T>(valid);
    Select(x, x, negative, mask);
}

// Update row maxima with FP8 shift and compute the exponential rescaling ratio alpha = exp(oldMax - newMax).
__simd_callee__ inline AscendC::MicroAPI::RegTensor<float> UpdateSoftmaxRescaleFactors(
    __ubuf__ half *maxima, __ubuf__ half *blockMax, __ubuf__ float *alpha, AscendC::MicroAPI::MaskReg &f,
    AscendC::MicroAPI::MaskReg &h)
{
    using namespace AscendC::MicroAPI;
    RegTensor<half> oldMax, newMax;
    RegTensor<float> ratio;
    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(oldMax, maxima);
    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(newMax, blockMax);
    // Keep the FP16 shifted maximum consistent across probabilities, alpha and LSE.
    Adds(newMax, newMax, SOFTMAX_FP8_SHIFT, h);
    Max(newMax, newMax, oldMax, h);
    ExpSub<float, half, RegLayout::ZERO>(ratio, oldMax, newMax, h);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(maxima, newMax, f);
    StoreAlign(alpha, ratio, f);
    return ratio;
}

// Store an 8x8 FP8 tile fragment across two 32-element NZ planes.
template <uint32_t STRIDE>
__simd_callee__ inline void StoreNZPlanePair(__ubuf__ float8_e4m3_t *out, uint32_t plane, uint32_t row,
                                             AscendC::MicroAPI::RegTensor<float8_e4m3_t> &packed,
                                             AscendC::MicroAPI::MaskReg &first32,
                                             AscendC::MicroAPI::MaskReg &last32)
{
    using namespace AscendC::MicroAPI;
    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + plane * STRIDE + row * 32, packed, first32);
    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + (plane + 1) * STRIDE + row * 32 - 32, packed, last32);
}

// Production consumes prebalanced FP16 scales. Scores/max remain FP16.
template <uint32_t ROWS = 64, uint32_t COLS = 256, bool FULL = false, bool NZ_OUTPUT = false, uint32_t NZ_PAD_ROWS = 0>
__simd_vf__ inline void SoftmaxHalfVF(__ubuf__ half *scores, __ubuf__ half *maxima, __ubuf__ float *denom,
                                      __ubuf__ float *alpha, __ubuf__ float8_e4m3_t *out, __ubuf__ half *qhalf,
                                      __ubuf__ half *khalf, __ubuf__ float *scratch, uint32_t qstart, uint32_t sq,
                                      uint32_t start, uint32_t sk, uint32_t causal)
{
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

    // Phase 1: 4-row unrolled and interleaved score scaling and row max reduction
    for (uint16_t row = 0; row < ROWS; row += 4) {
        const uint16_t r0 = row;
        const uint16_t r1 = row + 1;
        const uint16_t r2 = row + 2;
        const uint16_t r3 = row + 3;

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

        // 8 independent Muls guarantee maximum instruction interleaving distance (>= 7 instructions)
        Mul(x0_0, x0_0, c0_0, h);
        Mul(x0_1, x0_1, c0_1, h);
        Mul(x0_2, x0_2, c0_2, h);
        Mul(x0_3, x0_3, c0_3, h);
        Mul(x1_0, x1_0, c1_0, h);
        Mul(x1_1, x1_1, c1_1, h);
        Mul(x1_2, x1_2, c1_2, h);
        Mul(x1_3, x1_3, c1_3, h);

        if constexpr (!FULL) {
            const uint32_t pos0 = qstart + r0;
            const uint32_t pos1 = qstart + r1;
            const uint32_t pos2 = qstart + r2;
            const uint32_t pos3 = qstart + r3;
            MaskScoresTail(x0_0, negative, pos0, start + 0, sk, sq, causal);
            MaskScoresTail(x0_1, negative, pos1, start + 0, sk, sq, causal);
            MaskScoresTail(x0_2, negative, pos2, start + 0, sk, sq, causal);
            MaskScoresTail(x0_3, negative, pos3, start + 0, sk, sq, causal);
            MaskScoresTail(x1_0, negative, pos0, start + 128, sk, sq, causal);
            MaskScoresTail(x1_1, negative, pos1, start + 128, sk, sq, causal);
            MaskScoresTail(x1_2, negative, pos2, start + 128, sk, sq, causal);
            MaskScoresTail(x1_3, negative, pos3, start + 128, sk, sq, causal);
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
    RegTensor<float> ratio = UpdateSoftmaxRescaleFactors(maxima, blockMax, alpha, f, h);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg first32, last32;
    if constexpr (NZ_OUTPUT) {
        uint32_t halfChunk = 32;
        first32 = UpdateMask<float>(halfChunk);
        Not(last32, first32, f);
    }
    auto blockSum = scratch;
    UnalignReg sumStore;

    // Phase 2: 2-row interleaved pipeline. 8 ExpSubs in flight completely hide
    // the 12-14 cycle transcendental pipeline latency while capping vector register
    // usage at ~28 RegTensors (well below the 64 HW register limit to prevent UB spills).
    for (uint16_t row = 0; row < ROWS; row += 2) {
        const uint16_t row_a = row;
        const uint16_t row_b = row + 1;

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

        RegTensor<float> p0_a, p1_a, p2_a, p3_a;
        RegTensor<float> p0_b, p1_b, p2_b, p3_b;

        // 8 ExpSubs issued concurrently across row_a and row_b
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

        if constexpr (NZ_OUTPUT) {
            constexpr uint32_t stride = (ROWS + NZ_PAD_ROWS) * 32;
            StoreNZPlanePair<stride>(out, 0, row_a, packed0_a, first32, last32);
            StoreNZPlanePair<stride>(out, 0, row_b, packed0_b, first32, last32);
            StoreNZPlanePair<stride>(out, 2, row_a, packed1_a, first32, last32);
            StoreNZPlanePair<stride>(out, 2, row_b, packed1_b, first32, last32);
            StoreNZPlanePair<stride>(out, 4, row_a, packed2_a, first32, last32);
            StoreNZPlanePair<stride>(out, 4, row_b, packed2_b, first32, last32);
            StoreNZPlanePair<stride>(out, 6, row_a, packed3_a, first32, last32);
            StoreNZPlanePair<stride>(out, 6, row_b, packed3_b, first32, last32);
        } else {
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row_a * COLS + 0, packed0_a, f);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row_b * COLS + 0, packed0_b, f);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row_a * COLS + 64, packed1_a, f);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row_b * COLS + 64, packed1_b, f);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row_a * COLS + 128, packed2_a, f);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row_b * COLS + 128, packed2_b, f);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row_a * COLS + 192, packed3_a, f);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row_b * COLS + 192, packed3_b, f);
        }

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

} // namespace Sage
