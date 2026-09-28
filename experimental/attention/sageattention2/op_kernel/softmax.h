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
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait toFp8{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    auto f = CreateMask<float, MaskPattern::ALL>();
    auto h = CreateMask<half, MaskPattern::ALL>();
    auto blockMax = (__ubuf__ half *)alpha;
    UnalignReg maxStore;
    RegTensor<half> kh0, kh1, negative;
    LoadAlign(kh0, khalf);
    LoadAlign(kh1, khalf + 128);
    Duplicate(negative, half(-__builtin_inff()));
    for (uint16_t row = 0; row < ROWS; row += 2) {
        const uint16_t row_a = row;
        const uint16_t row_b = row + 1;

        RegTensor<half> qh_a, qh_b;
        LoadAlign<half, LoadDist::DIST_BRC_B16>(qh_a, qhalf + row_a);
        LoadAlign<half, LoadDist::DIST_BRC_B16>(qh_b, qhalf + row_b);

        RegTensor<half> x0_a, x1_a, x0_b, x1_b;
        LoadAlign(x0_a, scores + row_a * COLS + 0);
        LoadAlign(x1_a, scores + row_a * COLS + 128);
        LoadAlign(x0_b, scores + row_b * COLS + 0);
        LoadAlign(x1_b, scores + row_b * COLS + 128);

        RegTensor<half> coeff0_a, coeff0_b, coeff1_a, coeff1_b;
        // 2-row instruction interleaving: compute row_b coefficients to hide row_a latency
        Mul(coeff0_a, qh_a, kh0, h);
        Mul(coeff0_b, qh_b, kh0, h);
        Mul(coeff1_a, qh_a, kh1, h);
        Mul(coeff1_b, qh_b, kh1, h);

        // Score scaling interleaved across row_a and row_b: completely eliminates RAW bubbles
        Mul(x0_a, x0_a, coeff0_a, h);
        Mul(x0_b, x0_b, coeff0_b, h);
        Mul(x1_a, x1_a, coeff1_a, h);
        Mul(x1_b, x1_b, coeff1_b, h);

        if constexpr (!FULL) {
            const uint32_t pos_a = qstart + row_a;
            const uint32_t pos_b = qstart + row_b;
            MaskScoresTail(x0_a, negative, pos_a, start + 0, sk, sq, causal);
            MaskScoresTail(x0_b, negative, pos_b, start + 0, sk, sq, causal);
            MaskScoresTail(x1_a, negative, pos_a, start + 128, sk, sq, causal);
            MaskScoresTail(x1_b, negative, pos_b, start + 128, sk, sq, causal);
        }

        StoreAlign(scores + row_a * COLS + 0, x0_a, h);
        StoreAlign(scores + row_b * COLS + 0, x0_b, h);
        StoreAlign(scores + row_a * COLS + 128, x1_a, h);
        StoreAlign(scores + row_b * COLS + 128, x1_b, h);

        RegTensor<half> rowMax_a, rowMax_b, m_a, m_b;
        Max(rowMax_a, x0_a, x1_a, h);
        Max(rowMax_b, x0_b, x1_b, h);

        ReduceMax(m_a, rowMax_a, h);
        ReduceMax(m_b, rowMax_b, h);

        StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m_a, maxStore, 1);
        StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m_b, maxStore, 1);
    }
    StoreUnAlignPost(blockMax, maxStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
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
    for (uint16_t row = 0; row < ROWS; ++row) {
        // Independent fragments expose load/ExpSub/Cast overlap. Keep the
        // original sum order, and pack directly into the (optionally padded) NZ planes.
        RegTensor<half> m, x0, x1, x2, x3;
        RegTensor<float> p0, p1, p2, p3, s01, s23, rowSum, sum;
        RegTensor<float8_e4m3_t> packed0, packed1, packed2, packed3;
        LoadAlign<half, LoadDist::DIST_BRC_B16>(m, maxima + row);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x0, scores + row * COLS + 0);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x1, scores + row * COLS + 64);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x2, scores + row * COLS + 128);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x3, scores + row * COLS + 192);
        ExpSub<float, half, RegLayout::ZERO>(p0, x0, m, h);
        ExpSub<float, half, RegLayout::ZERO>(p1, x1, m, h);
        ExpSub<float, half, RegLayout::ZERO>(p2, x2, m, h);
        ExpSub<float, half, RegLayout::ZERO>(p3, x3, m, h);
        // Tree reduction interleaved with Cast to completely eliminate RAW pipeline latency bubbles
        Add(s01, p0, p1, f);
        Add(s23, p2, p3, f);
        Cast<float8_e4m3_t, float, toFp8>(packed0, p0, f);
        Cast<float8_e4m3_t, float, toFp8>(packed1, p1, f);
        Add(rowSum, s01, s23, f);
        Cast<float8_e4m3_t, float, toFp8>(packed2, p2, f);
        Cast<float8_e4m3_t, float, toFp8>(packed3, p3, f);
        if constexpr (NZ_OUTPUT) {
            constexpr uint32_t stride = (ROWS + NZ_PAD_ROWS) * 32;
            StoreNZPlanePair<stride>(out, 0, row, packed0, first32, last32);
            StoreNZPlanePair<stride>(out, 2, row, packed1, first32, last32);
            StoreNZPlanePair<stride>(out, 4, row, packed2, first32, last32);
            StoreNZPlanePair<stride>(out, 6, row, packed3, first32, last32);
        } else {
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row * COLS + 0, packed0, f);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row * COLS + 64, packed1, f);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row * COLS + 128, packed2, f);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row * COLS + 192, packed3, f);
        }
        ReduceSum(sum, rowSum, f);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(blockSum, sum, sumStore, 1);
    }
    StoreUnAlignPost(blockSum, sumStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    RegTensor<float> sum;
    LoadAlign(sum, scratch);
    MulDstAdd(oldDen, ratio, sum, f);
    StoreAlign(denom, oldDen, f);
}

} // namespace Sage
