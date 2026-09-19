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
// Production consumes prebalanced FP16 scales (PREPARED=true). Block-local
// preparation is retained for isolated diagnostics. Scores/max remain FP16.
template <uint32_t ROWS = 64, uint32_t COLS = 256, bool FULL = false, bool NZ_OUTPUT = false, bool PREPARED = false,
          uint32_t NZ_PAD_ROWS = 0>
__simd_vf__ inline void SoftmaxHalfVF(__ubuf__ half *scores, __ubuf__ float *ks, __ubuf__ float *qs,
                                      __ubuf__ half *maxima, __ubuf__ float *denom, __ubuf__ float *alpha,
                                      __ubuf__ float8_e4m3_t *out, __ubuf__ half *qhalf, __ubuf__ half *khalf,
                                      __ubuf__ float *scratch, uint32_t qstart, uint32_t sq, uint32_t start,
                                      uint32_t sk, uint32_t causal, float scale)
{
    static_assert(ROWS == 64 && COLS == 256, "half softmax requires a 64x256 AIV tile");
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait toHalf{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toFp8{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    auto f = CreateMask<float, MaskPattern::ALL>();
    auto h = CreateMask<half, MaskPattern::ALL>();
    if constexpr (!PREPARED) {
        uint32_t nr = qstart < sq ? Min(uint32_t(ROWS), sq - qstart) : 0;
        auto qrMask = UpdateMask<float>(nr);
        RegTensor<float> qv, kv, qmax, kmax, tmp, zero, one;
        Duplicate(zero, 0.0f);
        Duplicate(one, 1.0f);
        LoadAlign(qv, qs);
        Select(tmp, qv, zero, qrMask);
        ReduceMax(qmax, tmp, f);
        Duplicate(kmax, 0.0f);
        for (uint16_t col = 0; col < COLS; col += 64) {
            uint32_t at = start + col, valid = at < sk ? Min(uint32_t(64), sk - at) : 0;
            auto mask = UpdateMask<float>(valid);
            LoadAlign(kv, ks + col);
            Select(kv, kv, zero, mask);
            Max(kmax, kmax, kv, f);
        }
        ReduceMax(kmax, kmax, f);
        Duplicate(qmax, qmax, f);
        Duplicate(kmax, kmax, f);
        // Empty AIV halves have no valid scores; use finite harmless denominators.
        Maxs(qmax, qmax, 1.0e-30f, f);
        Maxs(kmax, kmax, 1.0e-30f, f);
        RegTensor<float> rootQ, rootK, common, qfactor, kfactor;
        Sqrt(rootQ, qmax, f);
        Sqrt(rootK, kmax, f);
        Mul(common, rootQ, rootK, f);
        Duplicate(tmp, scale);
        Abs(tmp, tmp, f);
        Muls(tmp, tmp, 2048.0f, f);
        Sqrt(tmp, tmp, f);
        Mul(common, common, tmp, f);
        Div(qfactor, common, qmax, f);
        Div(kfactor, common, kmax, f);
        Duplicate(tmp, scale);
        MaskReg negScale;
        Compare<float, CMPMODE::LT>(negScale, tmp, zero, f);
        Muls(tmp, qfactor, -1.0f, f);
        Select(qfactor, tmp, qfactor, negScale);
        RegTensor<half> packed;
        Mul(qv, qv, qfactor, f);
        Cast<half, float, toHalf>(packed, qv, f);
        StoreAlign<half, StoreDist::DIST_PACK_B32>(qhalf, packed, f);
        for (uint16_t col = 0; col < COLS; col += 64) {
            LoadAlign(kv, ks + col);
            Mul(kv, kv, kfactor, f);
            Cast<half, float, toHalf>(packed, kv, f);
            StoreAlign<half, StoreDist::DIST_PACK_B32>(khalf + col, packed, f);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    auto blockMax = (__ubuf__ half *)alpha;
    UnalignReg maxStore;
    RegTensor<half> kh0, kh1, negative;
    LoadAlign(kh0, khalf);
    LoadAlign(kh1, khalf + 128);
    Duplicate(negative, half(-__builtin_inff()));
    for (uint16_t row = 0; row < ROWS; ++row) {
        RegTensor<half> qh, x, coeff, rowMax, m;
        LoadAlign<half, LoadDist::DIST_BRC_B16>(qh, qhalf + row);

#pragma unroll
        for (uint16_t col = 0; col < COLS; col += 128) {
            LoadAlign(x, scores + row * COLS + col);
            if (col == 0)
                Mul(coeff, qh, kh0, h);
            else
                Mul(coeff, qh, kh1, h);
            Mul(x, x, coeff, h);
            if constexpr (!FULL) {
                uint32_t at = start + col, pos = qstart + row, valid = at < sk ? Min(uint32_t(128), sk - at) : 0;
                if (causal)
                    valid = Min(valid, pos >= at ? pos - at + 1 : 0);
                if (pos >= sq)
                    valid = 0;
                auto mask = UpdateMask<half>(valid);
                Select(x, x, negative, mask);
            }
            StoreAlign(scores + row * COLS + col, x, h);
            // Seed from the first fragment; oldMax still protects fully masked rows.
            if (col == 0)
                rowMax = x;
            else
                Max(rowMax, rowMax, x, h);
        }
        ReduceMax(m, rowMax, h);
        StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m, maxStore, 1);
    }
    StoreUnAlignPost(blockMax, maxStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    RegTensor<half> oldMax, newMax;
    RegTensor<float> ratio, oldDen;
    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(oldMax, maxima);
    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(newMax, (__ubuf__ half *)alpha);
    LoadAlign(oldDen, denom);
    // Keep the FP16 shifted maximum consistent across probabilities, alpha and LSE.
    Adds(newMax, newMax, half(-6.104793232414985f), h);
    Max(newMax, newMax, oldMax, h);
    ExpSub<float, half, RegLayout::ZERO>(ratio, oldMax, newMax, h);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(maxima, newMax, f);
    StoreAlign(alpha, ratio, f);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    uint32_t halfChunk = 32;
    auto first32 = UpdateMask<float>(halfChunk);
    MaskReg last32;
    Not(last32, first32, f);
    auto blockSum = scratch;
    UnalignReg sumStore;
    for (uint16_t row = 0; row < ROWS; ++row) {
        // Independent fragments expose load/ExpSub/Cast overlap. Keep the
        // original sum order, and pack directly into the (optionally padded) NZ planes.
        RegTensor<half> m, x0, x1, x2, x3;
        RegTensor<float> p0, p1, p2, p3, rowSum, sum;
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
        Add(rowSum, p0, p1, f);
        Add(rowSum, rowSum, p2, f);
        Add(rowSum, rowSum, p3, f);
        Cast<float8_e4m3_t, float, toFp8>(packed0, p0, f);
        Cast<float8_e4m3_t, float, toFp8>(packed1, p1, f);
        Cast<float8_e4m3_t, float, toFp8>(packed2, p2, f);
        Cast<float8_e4m3_t, float, toFp8>(packed3, p3, f);
        if constexpr (NZ_OUTPUT) {
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 0 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32,
                                                                 packed0, first32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 1 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32 - 32,
                                                                 packed0, last32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 2 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32,
                                                                 packed1, first32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 3 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32 - 32,
                                                                 packed1, last32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 4 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32,
                                                                 packed2, first32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 5 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32 - 32,
                                                                 packed2, last32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 6 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32,
                                                                 packed3, first32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 7 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32 - 32,
                                                                 packed3, last32);
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
