/*
 * NZ block processing adapted from attention/common/op_kernel/arch35/vf/
 * vf_basic_block_fullquant_unaligned256_update.h.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the repository root.
 */
#pragma once
#include "common.h"
namespace Sage {
// NZ scores retain eight rows in each FP16 register. ReduceDataBlock produces
// eight row statistics; FP8 P is packed into full 8x32 NZ output blocks.
template <bool MASK = false>
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
        Duplicate(m0, half(-65504.0f));
        RegTensor<half> q1, m1;
        LoadAlign<half, LoadDist::DIST_E2B_B16>(q1, qhalf + row + 8);
        Duplicate(m1, half(-65504.0f));
        for (uint16_t col = 0; col < 16; ++col) {
            {
                RegTensor<half> k, x0, x1, c0, c1;
                LoadAlign<half, LoadDist::DIST_BLK>(k, khalf + 0 + col * 16);
                LoadAlign(x0, scores + col * 64 * 16 + row * 16 + 0);
                Mul(c0, q0, k, h);
                LoadAlign(x1, scores + col * 64 * 16 + row * 16 + 128);
                Mul(c1, q1, k, h);
                Mul(x0, x0, c0, h);
                if constexpr (MASK) {
                    MaskReg valid;
                    CompareScalar<int16_t, CMPMODE::LE>(valid, laneOffset, int16_t(queryInGroup + row - col * 16), h);
                    Select(x0, x0, negative, valid);
                }
                Max(m0, m0, x0, h);
                StoreAlign(scores + col * 64 * 16 + row * 16 + 0, x0, h);
                Mul(x1, x1, c1, h);
                if constexpr (MASK) {
                    MaskReg valid;
                    CompareScalar<int16_t, CMPMODE::LE>(valid, laneOffset, int16_t(queryInGroup + row + 8 - col * 16),
                                                        h);
                    Select(x1, x1, negative, valid);
                }
                Max(m1, m1, x1, h);
                StoreAlign(scores + col * 64 * 16 + row * 16 + 128, x1, h);
            }
            {
                RegTensor<half> k, x0, x1, c0, c1;
                LoadAlign<half, LoadDist::DIST_BLK>(k, khalf + 256 + col * 16);
                LoadAlign(x0, scoresNext + col * 64 * 16 + row * 16 + 0);
                Mul(c0, q0, k, h);
                LoadAlign(x1, scoresNext + col * 64 * 16 + row * 16 + 128);
                Mul(c1, q1, k, h);
                Mul(x0, x0, c0, h);
                if constexpr (MASK) {
                    MaskReg valid;
                    CompareScalar<int16_t, CMPMODE::LE>(valid, laneOffset, int16_t(queryInGroup + row - col * 16 - 256),
                                                        h);
                    Select(x0, x0, negative, valid);
                }
                Max(m0, m0, x0, h);
                StoreAlign(scoresNext + col * 64 * 16 + row * 16 + 0, x0, h);
                Mul(x1, x1, c1, h);
                if constexpr (MASK) {
                    MaskReg valid;
                    CompareScalar<int16_t, CMPMODE::LE>(valid, laneOffset,
                                                        int16_t(queryInGroup + row + 8 - col * 16 - 256), h);
                    Select(x1, x1, negative, valid);
                }
                Max(m1, m1, x1, h);
                StoreAlign(scoresNext + col * 64 * 16 + row * 16 + 128, x1, h);
            }
        }
        ReduceDataBlock<ReduceType::MAX>(m0, m0, h);
        StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m0, maxStore, 8);
        ReduceDataBlock<ReduceType::MAX>(m1, m1, h);
        StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m1, maxStore, 8);
    }
    StoreUnAlignPost(blockMax, maxStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    RegTensor<half> oldMax, newMax;
    RegTensor<float> ratio;
    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(oldMax, maxima);
    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(newMax, (__ubuf__ half *)alpha);
    // Keep the FP16 shifted maximum consistent across probabilities, alpha and LSE.
    Adds(newMax, newMax, half(-6.104793232414985f), h);
    Max(newMax, newMax, oldMax, h);
    ExpSub<float, half, RegLayout::ZERO>(ratio, oldMax, newMax, h);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(maxima, newMax, f);
    StoreAlign(alpha, ratio, f);
}

template <bool FIRST, bool UPDATE_DEN = true>
__simd_callee__ inline void SoftmaxWideProbCore(__ubuf__ half *scores, __ubuf__ half *maxima, __ubuf__ float *denom,
                                                __ubuf__ float *alpha, __ubuf__ float8_e4m3_t *out,
                                                __ubuf__ float *scratch, __ubuf__ float *partialSums)
{
    using namespace AscendC::MicroAPI;
    // FIRST reads MaxCore's maxima; SECOND reads FIRST's partial lane sums.
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait cast0{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast1{RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast2{RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast3{RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    auto f = CreateMask<float, MaskPattern::ALL>();
    auto h = CreateMask<half, MaskPattern::ALL>();
    auto b = CreateMask<uint8_t, MaskPattern::ALL>();
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
            // Load the next NZ pair before computing the current pair so its
            // probability instructions can overlap the current FP8 packing.
            RegTensor<half> next0, next1;
            LoadAlign(next0, scores + (col + 1) * 64 * 32 + row * 16);
            LoadAlign(next1, scores + (col + 1) * 64 * 32 + 64 * 16 + row * 16);
            {
                RegTensor<half> x0, x1;
                RegTensor<float> p0, p1, p2, p3, pair0, pair1;
                RegTensor<float8_e4m3_t> packed0, packed1, packed2, packed3;
                RegTensor<uint8_t> merge0, merge1, merged;
                LoadAlign(x0, scores + (col + 0) * 64 * 32 + row * 16);
                LoadAlign(x1, scores + (col + 0) * 64 * 32 + 64 * 16 + row * 16);
                ExpSub<float, half, RegLayout::ZERO>(p0, x0, m, h);
                ExpSub<float, half, RegLayout::ONE>(p1, x0, m, h);
                ExpSub<float, half, RegLayout::ZERO>(p2, x1, m, h);
                ExpSub<float, half, RegLayout::ONE>(p3, x1, m, h);
                Add(pair0, p0, p1, f);
                Add(pair1, p2, p3, f);
                Add(pair0, pair0, pair1, f);
                Add(rowSum, rowSum, pair0, f);
                Cast<float8_e4m3_t, float, cast0>(packed0, p0, f);
                Cast<float8_e4m3_t, float, cast2>(packed1, p1, f);
                Cast<float8_e4m3_t, float, cast1>(packed2, p2, f);
                Cast<float8_e4m3_t, float, cast3>(packed3, p3, f);
                Or(merge0, (RegTensor<uint8_t> &)packed0, (RegTensor<uint8_t> &)packed1, b);
                Or(merge1, (RegTensor<uint8_t> &)packed2, (RegTensor<uint8_t> &)packed3, b);
                Or(merged, merge0, merge1, b);
                // Overwrite only the first FP16 stripe just loaded for these 8
                // rows. Other rows and column blocks remain disjoint. MTE3
                // skips the unused second stripe when copying packed P.
                StoreAlign((__ubuf__ uint8_t *)out + (col + 0) * 64 * 64 + row * 32, merged, b);
            }
            {
                RegTensor<float> p0, p1, p2, p3, pair0, pair1;
                RegTensor<float8_e4m3_t> packed0, packed1, packed2, packed3;
                RegTensor<uint8_t> merge0, merge1, merged;
                ExpSub<float, half, RegLayout::ZERO>(p0, next0, m, h);
                ExpSub<float, half, RegLayout::ONE>(p1, next0, m, h);
                ExpSub<float, half, RegLayout::ZERO>(p2, next1, m, h);
                ExpSub<float, half, RegLayout::ONE>(p3, next1, m, h);
                Add(pair0, p0, p1, f);
                Add(pair1, p2, p3, f);
                Add(pair0, pair0, pair1, f);
                Add(rowSum, rowSum, pair0, f);
                Cast<float8_e4m3_t, float, cast0>(packed0, p0, f);
                Cast<float8_e4m3_t, float, cast2>(packed1, p1, f);
                Cast<float8_e4m3_t, float, cast1>(packed2, p2, f);
                Cast<float8_e4m3_t, float, cast3>(packed3, p3, f);
                Or(merge0, (RegTensor<uint8_t> &)packed0, (RegTensor<uint8_t> &)packed1, b);
                Or(merge1, (RegTensor<uint8_t> &)packed2, (RegTensor<uint8_t> &)packed3, b);
                Or(merged, merge0, merge1, b);
                // Overwrite only the first FP16 stripe just loaded for these 8
                // rows. Other rows and column blocks remain disjoint. MTE3
                // skips the unused second stripe when copying packed P.
                StoreAlign((__ubuf__ uint8_t *)out + (col + 1) * 64 * 64 + row * 32, merged, b);
            }
        }
        if constexpr (FIRST) {
            StoreAlign(partialSums + row * 8, rowSum, f);
        } else {
            ReduceDataBlock<ReduceType::SUM>(rowSum, rowSum, f);
            StoreAlign(scratch + row, rowSum, sumMask);
        }
    }
    if constexpr (!FIRST && UPDATE_DEN) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> oldDen, ratio;
        LoadAlign(oldDen, denom);
        LoadAlign(ratio, alpha);
        RegTensor<float> sum;
        LoadAlign(sum, scratch);
        MulDstAdd(oldDen, ratio, sum, f);
        StoreAlign(denom, oldDen, f);
    }
}
template <bool MASK = false>
__simd_vf__ inline void SoftmaxWideFirstVF(__ubuf__ half *scores, __ubuf__ half *scoresNext, __ubuf__ half *maxima,
                                           __ubuf__ float *denom, __ubuf__ float *alpha, __ubuf__ float8_e4m3_t *out,
                                           __ubuf__ half *qhalf, __ubuf__ half *khalf, __ubuf__ float *scratch,
                                           __ubuf__ float *partialSums, int32_t queryInGroup = 0)
{
    SoftmaxWideMaxCore<MASK>(scores, scoresNext, maxima, alpha, qhalf, khalf, queryInGroup);
    SoftmaxWideProbCore<true>(scores, maxima, denom, alpha, out, scratch, partialSums);
}
__simd_vf__ inline void SoftmaxWideSecondVF(__ubuf__ half *scores, __ubuf__ half *maxima, __ubuf__ float *denom,
                                            __ubuf__ float *alpha, __ubuf__ float8_e4m3_t *out, __ubuf__ float *scratch,
                                            __ubuf__ float *partialSums)
{
    SoftmaxWideProbCore<false, false>(scores, maxima, denom, alpha, out, alpha + 64, partialSums);
}
} // namespace Sage
