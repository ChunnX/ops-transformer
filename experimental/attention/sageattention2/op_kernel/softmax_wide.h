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

// Scale an NZ score block by Q and K factors, apply optional causal mask, and accumulate block maximums.
template <bool MASK>
__simd_callee__ inline void ScaleMaskAndMaxWideNZBlock(
    __ubuf__ half *tileScores, AscendC::MicroAPI::RegTensor<half> &q0,
    AscendC::MicroAPI::RegTensor<half> &q1, __ubuf__ half *khalfPtr, uint16_t col, uint16_t row,
    int32_t causalLimitBase, AscendC::MicroAPI::RegTensor<int16_t> &laneOffset,
    AscendC::MicroAPI::RegTensor<half> &negative, AscendC::MicroAPI::RegTensor<half> &m0,
    AscendC::MicroAPI::RegTensor<half> &m1, AscendC::MicroAPI::MaskReg &h)
{
    using namespace AscendC::MicroAPI;
    RegTensor<half> k, x0, x1, c0, c1;
    LoadAlign<half, LoadDist::DIST_BLK>(k, khalfPtr + col * 16);
    LoadAlign(x0, tileScores + col * 64 * 16 + row * 16 + 0);
    LoadAlign(x1, tileScores + col * 64 * 16 + row * 16 + 128);
    // Interleave coeff multiplications to completely eliminate pipeline stall bubbles
    Mul(c0, q0, k, h);
    Mul(c1, q1, k, h);
    Mul(x0, x0, c0, h);
    Mul(x1, x1, c1, h);
    if constexpr (MASK) {
        MaskReg valid0, valid1;
        CompareScalar<int16_t, CMPMODE::LE>(valid0, laneOffset, int16_t(causalLimitBase + row - col * 16), h);
        CompareScalar<int16_t, CMPMODE::LE>(valid1, laneOffset, int16_t(causalLimitBase + row + 8 - col * 16), h);
        Select(x0, x0, negative, valid0);
        Select(x1, x1, negative, valid1);
    }
    Max(m0, m0, x0, h);
    Max(m1, m1, x1, h);
    StoreAlign(tileScores + col * 64 * 16 + row * 16 + 0, x0, h);
    StoreAlign(tileScores + col * 64 * 16 + row * 16 + 128, x1, h);
}

// Compute probabilities from scores, tree-accumulate row sums, cast to FP8, and pack into NZ output format.
__simd_callee__ inline void ComputeProbAndPackNZBlock(__ubuf__ float8_e4m3_t *out,
                                                      AscendC::MicroAPI::RegTensor<half> &x0,
                                                      AscendC::MicroAPI::RegTensor<half> &x1,
                                                      AscendC::MicroAPI::RegTensor<half> &m,
                                                      AscendC::MicroAPI::RegTensor<float> &rowSum, uint32_t outOffset)
{
    using namespace AscendC::MicroAPI;
    static constexpr CastTrait cast0{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast1{RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast2{RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast3{RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    auto f = CreateMask<float, MaskPattern::ALL>();
    auto h = CreateMask<half, MaskPattern::ALL>();
    auto b = CreateMask<uint8_t, MaskPattern::ALL>();

    RegTensor<float> p0, p1, p2, p3, pair0, pair1;
    RegTensor<float8_e4m3_t> packed0, packed1, packed2, packed3;
    RegTensor<uint8_t> merge0, merge1, merged;

    ExpSub<float, half, RegLayout::ZERO>(p0, x0, m, h);
    ExpSub<float, half, RegLayout::ONE>(p1, x0, m, h);
    ExpSub<float, half, RegLayout::ZERO>(p2, x1, m, h);
    ExpSub<float, half, RegLayout::ONE>(p3, x1, m, h);

    // Tree reduction interleaved with Cast and Or to eliminate RAW pipeline latency bubbles
    Add(pair0, p0, p1, f);
    Add(pair1, p2, p3, f);
    Cast<float8_e4m3_t, float, cast0>(packed0, p0, f);
    Cast<float8_e4m3_t, float, cast2>(packed1, p1, f);
    Add(pair0, pair0, pair1, f);
    Cast<float8_e4m3_t, float, cast1>(packed2, p2, f);
    Cast<float8_e4m3_t, float, cast3>(packed3, p3, f);
    Or(merge0, (RegTensor<uint8_t> &)packed0, (RegTensor<uint8_t> &)packed1, b);
    Or(merge1, (RegTensor<uint8_t> &)packed2, (RegTensor<uint8_t> &)packed3, b);
    Add(rowSum, rowSum, pair0, f);
    Or(merged, merge0, merge1, b);

    // Overwrite only the first FP16 stripe just loaded for these 8
    // rows. Other rows and column blocks remain disjoint. MTE3
    // skips the unused second stripe when copying packed P.
    StoreAlign((__ubuf__ uint8_t *)out + outOffset, merged, b);
}

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
            ScaleMaskAndMaxWideNZBlock<MASK>(scores, q0, q1, khalf + 0, col, row, queryInGroup, laneOffset, negative,
                                             m0, m1, h);
            ScaleMaskAndMaxWideNZBlock<MASK>(scoresNext, q0, q1, khalf + 256, col, row, queryInGroup - 256, laneOffset,
                                             negative, m0, m1, h);
        }
        ReduceDataBlock<ReduceType::MAX>(m0, m0, h);
        StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m0, maxStore, 8);
        ReduceDataBlock<ReduceType::MAX>(m1, m1, h);
        StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m1, maxStore, 8);
    }
    StoreUnAlignPost(blockMax, maxStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    // Keep the FP16 shifted maximum consistent across probabilities, alpha and LSE.
    UpdateSoftmaxRescaleFactors(maxima, blockMax, alpha, f, h);
}

template <bool FIRST, bool UPDATE_DEN = true>
__simd_callee__ inline void SoftmaxWideProbCore(__ubuf__ half *scores, __ubuf__ half *maxima, __ubuf__ float *denom,
                                                __ubuf__ float *alpha, __ubuf__ float8_e4m3_t *out,
                                                __ubuf__ float *scratch, __ubuf__ float *partialSums)
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
            // Load the next NZ pair before computing the current pair so its
            // probability instructions can overlap the current FP8 packing.
            RegTensor<half> next0, next1;
            LoadAlign(next0, scores + (col + 1) * 64 * 32 + row * 16);
            LoadAlign(next1, scores + (col + 1) * 64 * 32 + 64 * 16 + row * 16);

            RegTensor<half> x0, x1;
            LoadAlign(x0, scores + (col + 0) * 64 * 32 + row * 16);
            LoadAlign(x1, scores + (col + 0) * 64 * 32 + 64 * 16 + row * 16);
            ComputeProbAndPackNZBlock(out, x0, x1, m, rowSum, (col + 0) * 64 * 64 + row * 32);
            ComputeProbAndPackNZBlock(out, next0, next1, m, rowSum, (col + 1) * 64 * 64 + row * 32);
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
