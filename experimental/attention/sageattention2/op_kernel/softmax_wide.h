/*
 * NZ block processing adapted from attention/common/op_kernel/arch35/vf/
 * vf_basic_block_fullquant_unaligned256_update.h.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the repository root.
 */
#pragma once
#include "common.h"
#include "softmax.h"
namespace Sage {

// Scale a pair of NZ score blocks across group halves by Q and K factors, apply optional causal mask, and accumulate block maximums.
// Interleaves 4 rows across both blocks to ensure >= 3 independent instructions between Mul and consumers.
template <bool MASK>
__simd_callee__ inline void ScaleMaskAndMaxWideNZPair(
    __ubuf__ half *scores0, __ubuf__ half *scores1,
    AscendC::MicroAPI::RegTensor<half> &q0, AscendC::MicroAPI::RegTensor<half> &q1,
    __ubuf__ half *khalf0, __ubuf__ half *khalf1,
    uint16_t col, uint16_t row,
    int32_t causalLimit0, int32_t causalLimit1,
    AscendC::MicroAPI::RegTensor<int16_t> &laneOffset,
    AscendC::MicroAPI::RegTensor<half> &negative,
    AscendC::MicroAPI::RegTensor<half> &m0, AscendC::MicroAPI::RegTensor<half> &m1,
    AscendC::MicroAPI::MaskReg &h)
{
    using namespace AscendC::MicroAPI;
    RegTensor<half> k0, k1;
    LoadAlign<half, LoadDist::DIST_BLK>(k0, khalf0 + col * 16);
    LoadAlign<half, LoadDist::DIST_BLK>(k1, khalf1 + col * 16);

    RegTensor<half> x0_0, x1_0, x0_1, x1_1;
    LoadAlign(x0_0, scores0 + col * 64 * 16 + row * 16 + 0);
    LoadAlign(x1_0, scores0 + col * 64 * 16 + row * 16 + 128);
    LoadAlign(x0_1, scores1 + col * 64 * 16 + row * 16 + 0);
    LoadAlign(x1_1, scores1 + col * 64 * 16 + row * 16 + 128);

    RegTensor<half> c0_0, c1_0, c0_1, c1_1;
    // 4-coeff interleaved multiplication completely eliminates RAW latency bubbles
    Mul(c0_0, q0, k0, h);
    Mul(c1_0, q1, k0, h);
    Mul(c0_1, q0, k1, h);
    Mul(c1_1, q1, k1, h);

    Mul(x0_0, x0_0, c0_0, h);
    Mul(x1_0, x1_0, c1_0, h);
    Mul(x0_1, x0_1, c0_1, h);
    Mul(x1_1, x1_1, c1_1, h);

    if constexpr (MASK) {
        MaskReg valid0_0, valid1_0, valid0_1, valid1_1;
        CompareScalar<int16_t, CMPMODE::LE>(valid0_0, laneOffset, int16_t(causalLimit0 + row - col * 16), h);
        CompareScalar<int16_t, CMPMODE::LE>(valid1_0, laneOffset, int16_t(causalLimit0 + row + 8 - col * 16), h);
        CompareScalar<int16_t, CMPMODE::LE>(valid0_1, laneOffset, int16_t(causalLimit1 + row - col * 16), h);
        CompareScalar<int16_t, CMPMODE::LE>(valid1_1, laneOffset, int16_t(causalLimit1 + row + 8 - col * 16), h);
        Select(x0_0, x0_0, negative, valid0_0);
        Select(x1_0, x1_0, negative, valid1_0);
        Select(x0_1, x0_1, negative, valid0_1);
        Select(x1_1, x1_1, negative, valid1_1);
    }

    Max(m0, m0, x0_0, h);
    Max(m1, m1, x1_0, h);
    Max(m0, m0, x0_1, h);
    Max(m1, m1, x1_1, h);

    StoreAlign(scores0 + col * 64 * 16 + row * 16 + 0, x0_0, h);
    StoreAlign(scores0 + col * 64 * 16 + row * 16 + 128, x1_0, h);
    StoreAlign(scores1 + col * 64 * 16 + row * 16 + 0, x0_1, h);
    StoreAlign(scores1 + col * 64 * 16 + row * 16 + 128, x1_1, h);
}

// Compute probabilities from scores, tree-accumulate row sums, cast to FP8, and pack into NZ output format.
// Interleaves two consecutive NZ blocks (8 ExpSubs in flight) to fully hide transcendental instruction latency.
__simd_callee__ inline void ComputeProbAndPackNZPair(
    __ubuf__ float8_e4m3_t *out,
    AscendC::MicroAPI::RegTensor<half> &x0_0, AscendC::MicroAPI::RegTensor<half> &x1_0,
    AscendC::MicroAPI::RegTensor<half> &x0_1, AscendC::MicroAPI::RegTensor<half> &x1_1,
    AscendC::MicroAPI::RegTensor<half> &m, AscendC::MicroAPI::RegTensor<float> &rowSum,
    uint32_t outOffset0, uint32_t outOffset1)
{
    using namespace AscendC::MicroAPI;
    static constexpr CastTrait cast0{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast1{RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast2{RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast3{RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    auto f = CreateMask<float, MaskPattern::ALL>();
    auto h = CreateMask<half, MaskPattern::ALL>();
    auto b = CreateMask<uint8_t, MaskPattern::ALL>();

    RegTensor<float> p0_0, p1_0, p2_0, p3_0;
    RegTensor<float> p0_1, p1_1, p2_1, p3_1;

    // 8 ExpSubs in flight across the pair of NZ blocks to fully hide exponential pipeline latency
    ExpSub<float, half, RegLayout::ZERO>(p0_0, x0_0, m, h);
    ExpSub<float, half, RegLayout::ONE>(p1_0, x0_0, m, h);
    ExpSub<float, half, RegLayout::ZERO>(p2_0, x1_0, m, h);
    ExpSub<float, half, RegLayout::ONE>(p3_0, x1_0, m, h);

    ExpSub<float, half, RegLayout::ZERO>(p0_1, x0_1, m, h);
    ExpSub<float, half, RegLayout::ONE>(p1_1, x0_1, m, h);
    ExpSub<float, half, RegLayout::ZERO>(p2_1, x1_1, m, h);
    ExpSub<float, half, RegLayout::ONE>(p3_1, x1_1, m, h);

    Mins(p0_0, p0_0, 448.0f, f);
    Mins(p1_0, p1_0, 448.0f, f);
    Mins(p2_0, p2_0, 448.0f, f);
    Mins(p3_0, p3_0, 448.0f, f);
    Mins(p0_1, p0_1, 448.0f, f);
    Mins(p1_1, p1_1, 448.0f, f);
    Mins(p2_1, p2_1, 448.0f, f);
    Mins(p3_1, p3_1, 448.0f, f);

    RegTensor<float> pair0_0, pair1_0, pair0_1, pair1_1;
    Add(pair0_0, p0_0, p1_0, f);
    Add(pair0_1, p0_1, p1_1, f);
    Add(pair1_0, p2_0, p3_0, f);
    Add(pair1_1, p2_1, p3_1, f);

    RegTensor<float8_e4m3_t> packed0_0, packed1_0, packed2_0, packed3_0;
    RegTensor<float8_e4m3_t> packed0_1, packed1_1, packed2_1, packed3_1;

    Cast<float8_e4m3_t, float, cast0>(packed0_0, p0_0, f);
    Cast<float8_e4m3_t, float, cast0>(packed0_1, p0_1, f);
    Cast<float8_e4m3_t, float, cast2>(packed1_0, p1_0, f);
    Cast<float8_e4m3_t, float, cast2>(packed1_1, p1_1, f);

    Add(pair0_0, pair0_0, pair1_0, f);
    Add(pair0_1, pair0_1, pair1_1, f);

    Cast<float8_e4m3_t, float, cast1>(packed2_0, p2_0, f);
    Cast<float8_e4m3_t, float, cast1>(packed2_1, p2_1, f);
    Cast<float8_e4m3_t, float, cast3>(packed3_0, p3_0, f);
    Cast<float8_e4m3_t, float, cast3>(packed3_1, p3_1, f);

    RegTensor<uint8_t> merge0_0, merge1_0, merged_0;
    RegTensor<uint8_t> merge0_1, merge1_1, merged_1;

    Or(merge0_0, (RegTensor<uint8_t> &)packed0_0, (RegTensor<uint8_t> &)packed1_0, b);
    Or(merge0_1, (RegTensor<uint8_t> &)packed0_1, (RegTensor<uint8_t> &)packed1_1, b);
    Or(merge1_0, (RegTensor<uint8_t> &)packed2_0, (RegTensor<uint8_t> &)packed3_0, b);
    Or(merge1_1, (RegTensor<uint8_t> &)packed2_1, (RegTensor<uint8_t> &)packed3_1, b);

    Add(rowSum, rowSum, pair0_0, f);
    Add(rowSum, rowSum, pair0_1, f);

    Or(merged_0, merge0_0, merge1_0, b);
    Or(merged_1, merge0_1, merge1_1, b);

    StoreAlign((__ubuf__ uint8_t *)out + outOffset0, merged_0, b);
    StoreAlign((__ubuf__ uint8_t *)out + outOffset1, merged_1, b);
}

// NZ scores retain eight rows in each FP16 register. ReduceDataBlock produces
// eight row statistics; FP8 P is packed into full 8x32 NZ output blocks.
template <bool MASK>
__simd_callee__ inline void SoftmaxWideMaxCore(__ubuf__ half *scores, __ubuf__ half *scoresNext, __ubuf__ half *maxima,
                                               __ubuf__ float *alpha, __ubuf__ half *qhalf, __ubuf__ half *khalf,
                                               int32_t queryInGroup = 0)
{
    using namespace AscendC::MicroAPI;
    // QK_READY and the scale MTE2_V event make input scores/scales visible.
    // The fence below the reduction also orders the previous maxima store.
    auto h = CreateMask<half, MaskPattern::ALL>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    // NZ lane i belongs to row i/16 and column i%16. Use coordinates
    // relative to this N512 group so the signed 16-bit comparison is exact.
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
        Duplicate(m0, half(FP16_MIN));
        RegTensor<half> q1, m1;
        LoadAlign<half, LoadDist::DIST_E2B_B16>(q1, qhalf + row + 8);
        Duplicate(m1, half(FP16_MIN));
        for (uint16_t col = 0; col < 16; ++col) {
            ScaleMaskAndMaxWideNZPair<MASK>(scores, scoresNext, q0, q1, khalf + 0, khalf + 256,
                                           col, row, queryInGroup, queryInGroup - 256,
                                           laneOffset, negative, m0, m1, h);
        }
        ReduceDataBlock<ReduceType::MAX>(m0, m0, h);
        StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m0, maxStore, 8);
        ReduceDataBlock<ReduceType::MAX>(m1, m1, h);
        StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m1, maxStore, 8);
    }
    StoreUnAlignPost(blockMax, maxStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    // Keep the FP16 shifted maximum consistent across probabilities, alpha and LSE.
    UpdateSoftmaxRescaleFactors(maxima, (__ubuf__ half *)alpha, alpha, f, h);
}

template <bool FIRST>
__simd_callee__ inline void SoftmaxWideProbCore(__ubuf__ half *scores, __ubuf__ half *maxima,
                                                __ubuf__ float8_e4m3_t *out,
                                                __ubuf__ float *partialSums,
                                                __ubuf__ float *scratch = nullptr)
{
    using namespace AscendC::MicroAPI;
    // FIRST reads MaxCore's maxima; SECOND reads FIRST's partial lane sums.
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
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
            RegTensor<half> x0, x1, next0, next1;
            LoadAlign(x0, scores + (col + 0) * 64 * 32 + row * 16);
            LoadAlign(x1, scores + (col + 0) * 64 * 32 + 64 * 16 + row * 16);
            LoadAlign(next0, scores + (col + 1) * 64 * 32 + row * 16);
            LoadAlign(next1, scores + (col + 1) * 64 * 32 + 64 * 16 + row * 16);

            ComputeProbAndPackNZPair(out, x0, x1, next0, next1, m, rowSum,
                                    (col + 0) * 64 * 64 + row * 32,
                                    (col + 1) * 64 * 64 + row * 32);
        }
        if constexpr (FIRST) {
            StoreAlign(partialSums + row * 8, rowSum, f);
        } else {
            ReduceDataBlock<ReduceType::SUM>(rowSum, rowSum, f);
            StoreAlign(scratch + row, rowSum, sumMask);
        }
    }
}

template <bool MASK = false>
__simd_vf__ inline void SoftmaxWideFirstVF(__ubuf__ half *scores, __ubuf__ half *scoresNext, __ubuf__ half *maxima,
                                           __ubuf__ float *alpha, __ubuf__ float8_e4m3_t *out, __ubuf__ half *qhalf,
                                           __ubuf__ half *khalf, __ubuf__ float *partialSums,
                                           int32_t queryInGroup = 0)
{
    SoftmaxWideMaxCore<MASK>(scores, scoresNext, maxima, alpha, qhalf, khalf, queryInGroup);
    SoftmaxWideProbCore<true>(scores, maxima, out, partialSums);
}

// The second 256-token half publishes its block sum in the unused upper half
// of the alpha slot (alpha + 64), mirroring the pair protocol.
__simd_vf__ inline void SoftmaxWideSecondVF(__ubuf__ half *scores, __ubuf__ half *maxima,
                                            __ubuf__ float *alpha, __ubuf__ float8_e4m3_t *out,
                                            __ubuf__ float *partialSums)
{
    SoftmaxWideProbCore<false>(scores, maxima, out, partialSums, alpha + 64);
}

} // namespace Sage
