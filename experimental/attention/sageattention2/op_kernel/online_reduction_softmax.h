#pragma once

#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"
#include "math_approximations.h"
#include "ring_buffer_layout.h"
#include "catlass_fractal_packer.h"
#include "credit_handshake.h"

namespace sage::microkernel {

#if defined(__CCE__)

namespace vapi = AscendC::MicroAPI;

// ============================================================================
// 1. Wide NZ Score Scaling, Masking & Peak Tracker Pipeline
// ============================================================================

// BiSheng backend limit: __simd_callee__ helpers must be free functions, never
// struct members (a struct-member callee trips "VAG's non-zero args must be
// hoisted" during AIV lowering). Keep every helper at namespace scope.
template <bool MASK>
__simd_callee__ __attribute__((always_inline)) inline void ProcessSlotWideNZ(
        __ubuf__ half *scores_base,
        uint32_t element_offset,
        vapi::RegTensor<half> &q_factor,
        vapi::RegTensor<half> &k_elem,
        int32_t causal_threshold,
        vapi::RegTensor<int16_t> &lane_offset,
        vapi::RegTensor<half> &neg_inf,
        vapi::RegTensor<half> &row_max,
        vapi::MaskReg &mask_h) noexcept {

        vapi::RegTensor<half> c_scale, score_val;
        vapi::Mul(c_scale, q_factor, k_elem, mask_h);
        vapi::LoadAlign(score_val, scores_base + element_offset);
        vapi::Mul(score_val, score_val, c_scale, mask_h);

        if constexpr (MASK) {
            vapi::MaskReg is_valid;
            vapi::CompareScalar<int16_t, AscendC::CMPMODE::LE>(
                is_valid, lane_offset, static_cast<int16_t>(causal_threshold), mask_h);
            vapi::Select(score_val, score_val, neg_inf, is_valid);
        }

    vapi::Max(row_max, row_max, score_val, mask_h);
    vapi::StoreAlign(scores_base + element_offset, score_val, mask_h);
}

template <bool MASK>
__simd_callee__ __attribute__((always_inline)) inline void ScaleMaskAndMaxWideNZPair(
    __ubuf__ half *scores0, __ubuf__ half *scores1,
    vapi::RegTensor<half> &q0, vapi::RegTensor<half> &q1,
    __ubuf__ half *khalf0, __ubuf__ half *khalf1,
    uint16_t col, uint16_t row,
    int32_t causalLimit0, int32_t causalLimit1,
    vapi::RegTensor<int16_t> &laneOffset,
    vapi::RegTensor<half> &negative,
    vapi::RegTensor<half> &m0, vapi::RegTensor<half> &m1,
    vapi::MaskReg &h) {

    vapi::RegTensor<half> k_reg;
    const uint32_t col_stride = static_cast<uint32_t>(col) * 64 * 16 + static_cast<uint32_t>(row) * 16;
    const int32_t base_causal0 = causalLimit0 + row - static_cast<int32_t>(col) * 16;
    const int32_t base_causal1 = causalLimit1 + row - static_cast<int32_t>(col) * 16;

    // Slot 0 & 1 on scores0
    vapi::LoadAlign<half, vapi::LoadDist::DIST_BLK>(k_reg, khalf0 + col * 16);
    ProcessSlotWideNZ<MASK>(scores0, col_stride + 0, q0, k_reg, base_causal0 + 0, laneOffset, negative, m0, h);
    ProcessSlotWideNZ<MASK>(scores0, col_stride + 128, q1, k_reg, base_causal0 + 8, laneOffset, negative, m1, h);

    // Slot 2 & 3 on scores1
    vapi::LoadAlign<half, vapi::LoadDist::DIST_BLK>(k_reg, khalf1 + col * 16);
    ProcessSlotWideNZ<MASK>(scores1, col_stride + 0, q0, k_reg, base_causal1 + 0, laneOffset, negative, m0, h);
    ProcessSlotWideNZ<MASK>(scores1, col_stride + 128, q1, k_reg, base_causal1 + 8, laneOffset, negative, m1, h);
}

// ============================================================================
// 2. Homologous E4M3 Quantization & Packing Engine (Task 3.4 Conformance)
// ============================================================================

template <vapi::RegLayout IN_LAYOUT, vapi::RegLayout OUT_LAYOUT>
__simd_callee__ __attribute__((always_inline)) inline void FoldSubLaneHomologous(
        vapi::RegTensor<float> &row_sum_acc,
        vapi::RegTensor<float8_e4m3_t> &pk_dest,
        vapi::RegTensor<half> &score_chunk,
        vapi::RegTensor<half> &max_chunk,
        vapi::MaskReg &mask_f32,
        vapi::MaskReg &mask_f16,
        vapi::MaskReg &mask_byte,
        bool is_initial) noexcept {

        static constexpr vapi::CastTrait kCastSatRint{OUT_LAYOUT, vapi::SatMode::SAT, vapi::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr vapi::CastTrait kCastUnpack{OUT_LAYOUT, vapi::SatMode::UNKNOWN, vapi::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

        vapi::RegTensor<float> p_raw;
        vapi::ExpSub<float, half, IN_LAYOUT>(p_raw, score_chunk, max_chunk, mask_f16);

        vapi::RegTensor<float8_e4m3_t> pk_current;
        vapi::Cast<float8_e4m3_t, float, kCastSatRint>(pk_current, p_raw, mask_f32);

        if (is_initial) {
            pk_dest = pk_current;
        } else {
            auto &dest_u8 = (vapi::RegTensor<uint8_t> &)pk_dest;
            auto &curr_u8 = (vapi::RegTensor<uint8_t> &)pk_current;
            vapi::Or(dest_u8, dest_u8, curr_u8, mask_byte);
        }

    // Task 3.4: Dequantize saturated/rounded E4M3 back to FP32 for homologous normalizer sum
    vapi::Cast<float, float8_e4m3_t, kCastUnpack>(p_raw, pk_current, mask_f32);
    vapi::Add(row_sum_acc, row_sum_acc, p_raw, mask_f32);
}

__simd_callee__ __attribute__((always_inline)) inline void ComputeProbAndPackNZSingle(
    __ubuf__ float8_e4m3_t *out,
    vapi::RegTensor<half> &x0,
    vapi::RegTensor<half> &x1,
    vapi::RegTensor<half> &m,
    vapi::RegTensor<float> &rowSum,
    uint32_t outOffset) {

    auto mask_f = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
    auto mask_h = vapi::CreateMask<half, vapi::MaskPattern::ALL>();
    auto mask_b = vapi::CreateMask<uint8_t, vapi::MaskPattern::ALL>();

    vapi::RegTensor<float8_e4m3_t> packed_result;

    FoldSubLaneHomologous<vapi::RegLayout::ZERO, vapi::RegLayout::ZERO>(
        rowSum, packed_result, x0, m, mask_f, mask_h, mask_b, true);
    FoldSubLaneHomologous<vapi::RegLayout::ONE,  vapi::RegLayout::TWO >(
        rowSum, packed_result, x0, m, mask_f, mask_h, mask_b, false);
    FoldSubLaneHomologous<vapi::RegLayout::ZERO, vapi::RegLayout::ONE >(
        rowSum, packed_result, x1, m, mask_f, mask_h, mask_b, false);
    FoldSubLaneHomologous<vapi::RegLayout::ONE,  vapi::RegLayout::THREE>(
        rowSum, packed_result, x1, m, mask_f, mask_h, mask_b, false);

    auto &res_u8 = (vapi::RegTensor<uint8_t> &)packed_result;
    vapi::StoreAlign((__ubuf__ uint8_t *)out + outOffset, res_u8, mask_b);
}

// ============================================================================
// 3. Online Softmax Rescaling & LN_448 Exponent Tracking
// ============================================================================

__simd_callee__ __attribute__((always_inline)) inline vapi::RegTensor<float> UpdateSoftmaxRescaleFactors(
    __ubuf__ half *maxima, __ubuf__ half *blockMax, __ubuf__ float *alpha,
    vapi::MaskReg &f_mask, vapi::MaskReg &h_mask) noexcept {

    vapi::RegTensor<half> old_peak, new_peak;
    vapi::LoadAlign<half, vapi::LoadDist::DIST_UNPACK_B16>(old_peak, maxima);
    vapi::LoadAlign<half, vapi::LoadDist::DIST_UNPACK_B16>(new_peak, blockMax);

    // Crucial: downward shift by LN_448 guarantees exp(score - max) lands in (0, 448] filling E4M3 range
    vapi::Adds(new_peak, new_peak, half(-sage::core::LN_448), h_mask);
    vapi::Max(new_peak, new_peak, old_peak, h_mask);

    vapi::RegTensor<float> ratio_val;
    vapi::ExpSub<float, half, vapi::RegLayout::ZERO>(ratio_val, old_peak, new_peak, h_mask);

    vapi::StoreAlign<half, vapi::StoreDist::DIST_PACK_B32>(maxima, new_peak, f_mask);
    vapi::StoreAlign(alpha, ratio_val, f_mask);
    return ratio_val;
}

// ============================================================================
// 4. Non-Wide Causal and Sequence Boundary Masking
// ============================================================================

template <bool CAUSAL = false, class T>
__simd_callee__ __attribute__((always_inline)) inline void MaskScoresTail(vapi::RegTensor<T> &x,
                                           vapi::RegTensor<T> &negative, uint32_t pos, uint32_t at,
                                           uint32_t sk, uint32_t sq) {
    uint32_t valid_elements = (at < sk) ? ((sk - at < 128) ? (sk - at) : 128) : 0;
    if constexpr (CAUSAL) {
        const uint32_t causal_bound = (pos >= at) ? (pos - at + 1) : 0;
        if (causal_bound < valid_elements) {
            valid_elements = causal_bound;
        }
    }
    if (pos >= sq) {
        valid_elements = 0;
    }
    auto active_mask = vapi::UpdateMask<T>(valid_elements);
    vapi::Select(x, x, negative, active_mask);
}

// ============================================================================
// 5. Wide-Path Softmax Kernels
// ============================================================================

    template <bool MASK>
    __simd_callee__ __attribute__((always_inline)) inline void SoftmaxWideMaxCore(__ubuf__ half *scores, __ubuf__ half *scoresNext,
                                                          __ubuf__ half *maxima, __ubuf__ float *alpha,
                                                          __ubuf__ half *qhalf, __ubuf__ half *khalf,
                                                          int32_t queryInGroup = 0) {
        auto h = vapi::CreateMask<half, vapi::MaskPattern::ALL>();
        auto f = vapi::CreateMask<float, vapi::MaskPattern::ALL>();

        vapi::RegTensor<int16_t> laneOffset;
        vapi::RegTensor<half> negative;
        if constexpr (MASK) {
            vapi::RegTensor<int16_t> lane, rowIndex, colIndex, lowBits;
            vapi::Arange(lane, int16_t(0));
            vapi::ShiftRights(rowIndex, lane, int16_t(4), h);
            vapi::Duplicate(lowBits, int16_t(15));
            vapi::And(colIndex, lane, lowBits, h);
            vapi::Sub(laneOffset, colIndex, rowIndex, h);
            vapi::Duplicate(negative, half(-__builtin_inff()));
        }
        auto blockMax = (__ubuf__ half *)alpha;
        vapi::UnalignReg maxStore;
        for (uint16_t row = 0; row < 64; row += 16) {
            vapi::RegTensor<half> q0, m0;
            vapi::LoadAlign<half, vapi::LoadDist::DIST_E2B_B16>(q0, qhalf + row + 0);
            vapi::Duplicate(m0, half(SAGE_FP16_MIN));
            vapi::RegTensor<half> q1, m1;
            vapi::LoadAlign<half, vapi::LoadDist::DIST_E2B_B16>(q1, qhalf + row + 8);
            vapi::Duplicate(m1, half(SAGE_FP16_MIN));
            for (uint16_t col = 0; col < 16; ++col) {
                ScaleMaskAndMaxWideNZPair<MASK>(scores, scoresNext, q0, q1, khalf + 0, khalf + 256,
                                                col, row, queryInGroup, queryInGroup - 256,
                                                laneOffset, negative, m0, m1, h);
            }
            vapi::ReduceDataBlock<vapi::ReduceType::MAX>(m0, m0, h);
            vapi::ReduceDataBlock<vapi::ReduceType::MAX>(m1, m1, h);
            vapi::StoreUnAlign<half, vapi::PostLiteral::POST_MODE_UPDATE>(blockMax, m0, maxStore, 8);
            vapi::StoreUnAlign<half, vapi::PostLiteral::POST_MODE_UPDATE>(blockMax, m1, maxStore, 8);
        }
        vapi::StoreUnAlignPost(blockMax, maxStore, 0);
        vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();
        UpdateSoftmaxRescaleFactors(maxima, (__ubuf__ half *)alpha, alpha, f, h);
    }

    template <bool FIRST>
    __simd_callee__ __attribute__((always_inline)) inline void SoftmaxWideProbCore(__ubuf__ half *scores, __ubuf__ half *maxima,
                                                           __ubuf__ float8_e4m3_t *out,
                                                           __ubuf__ float *partialSums,
                                                           __ubuf__ float *scratch = nullptr) {
        if constexpr (FIRST) {
            vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();
        }
        auto f = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
        auto sumMask = vapi::CreateMask<float, vapi::MaskPattern::VL8>();
        for (uint16_t row = 0; row < 64; row += 8) {
            vapi::RegTensor<half> m;
            vapi::RegTensor<float> rowSum;
            vapi::LoadAlign<half, vapi::LoadDist::DIST_E2B_B16>(m, maxima + row);
            if constexpr (FIRST) {
                vapi::Duplicate(rowSum, 0.0f);
            } else {
                vapi::LoadAlign(rowSum, partialSums + row * 8);
            }
            for (uint16_t col = 0; col < 8; col += 2) {
                vapi::RegTensor<half> x0, x1, next0, next1;
                vapi::LoadAlign(x0, scores + (col + 0) * 64 * 32 + row * 16);
                vapi::LoadAlign(x1, scores + (col + 0) * 64 * 32 + 64 * 16 + row * 16);
                vapi::LoadAlign(next0, scores + (col + 1) * 64 * 32 + row * 16);
                vapi::LoadAlign(next1, scores + (col + 1) * 64 * 32 + 64 * 16 + row * 16);

                ComputeProbAndPackNZSingle(out, x0, x1, m, rowSum, (col + 0) * 64 * 64 + row * 32);
                ComputeProbAndPackNZSingle(out, next0, next1, m, rowSum, (col + 1) * 64 * 64 + row * 32);
            }
            if constexpr (FIRST) {
                vapi::StoreAlign(partialSums + row * 8, rowSum, f);
            } else {
                vapi::ReduceDataBlock<vapi::ReduceType::SUM>(rowSum, rowSum, f);
                vapi::StoreAlign(scratch + row, rowSum, sumMask);
            }
        }
    }

    template <bool MASK>
    __simd_vf__ inline void SoftmaxWideFirstVF(__ubuf__ half *scores, __ubuf__ half *scoresNext,
                                                      __ubuf__ half *maxima, __ubuf__ float *alpha,
                                                      __ubuf__ float8_e4m3_t *out, __ubuf__ half *qhalf,
                                                      __ubuf__ half *khalf, __ubuf__ float *partialSums,
                                                      int32_t queryInGroup = 0) {
        SoftmaxWideMaxCore<MASK>(scores, scoresNext, maxima, alpha, qhalf, khalf, queryInGroup);
        SoftmaxWideProbCore<true>(scores, maxima, out, partialSums);
    }

    __simd_vf__ inline void SoftmaxWideSecondVF(__ubuf__ half *scores, __ubuf__ half *maxima,
                                                       __ubuf__ float *alpha, __ubuf__ float8_e4m3_t *out,
                                                       __ubuf__ float *partialSums) {
        SoftmaxWideProbCore<false>(scores, maxima, out, partialSums, alpha + 64);
    }

    template <uint32_t ROWS, uint32_t COLS, bool FULL, uint32_t NZ_PAD_ROWS, bool CAUSAL = false>
    __simd_vf__ inline void SoftmaxHalfVF(__ubuf__ half *scores, __ubuf__ half *maxima, __ubuf__ float *denom,
                                                 __ubuf__ float *alpha, __ubuf__ float8_e4m3_t *out, __ubuf__ half *qhalf,
                                                 __ubuf__ half *khalf, __ubuf__ float *scratch, uint32_t qstart, uint32_t sq,
                                                 uint32_t start, uint32_t sk) {
        static_assert(ROWS == 64 && COLS == 256, "half softmax requires a 64x256 AIV tile");
        static constexpr vapi::CastTrait toFp8{vapi::RegLayout::ZERO, vapi::SatMode::SAT, vapi::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr vapi::CastTrait fromFp8{vapi::RegLayout::ZERO, vapi::SatMode::UNKNOWN, vapi::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

        auto h = vapi::CreateMask<half, vapi::MaskPattern::ALL>();
        auto blockMax = (__ubuf__ half *)alpha;
        vapi::UnalignReg maxStore;

        vapi::RegTensor<half> kh0, kh1, negative;
        vapi::LoadAlign(kh0, khalf);
        vapi::LoadAlign(kh1, khalf + 128);
        vapi::Duplicate(negative, half(-__builtin_inff()));

        for (uint16_t row = 0; row < ROWS; row += 4) {
            const uint16_t r0 = row, r1 = row + 1, r2 = row + 2, r3 = row + 3;
            vapi::RegTensor<half> qh0, qh1, qh2, qh3;
            vapi::LoadAlign<half, vapi::LoadDist::DIST_BRC_B16>(qh0, qhalf + r0);
            vapi::LoadAlign<half, vapi::LoadDist::DIST_BRC_B16>(qh1, qhalf + r1);
            vapi::LoadAlign<half, vapi::LoadDist::DIST_BRC_B16>(qh2, qhalf + r2);
            vapi::LoadAlign<half, vapi::LoadDist::DIST_BRC_B16>(qh3, qhalf + r3);

            vapi::RegTensor<half> x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3;
            vapi::LoadAlign(x0_0, scores + r0 * COLS + 0);
            vapi::LoadAlign(x1_0, scores + r0 * COLS + 128);
            vapi::LoadAlign(x0_1, scores + r1 * COLS + 0);
            vapi::LoadAlign(x1_1, scores + r1 * COLS + 128);
            vapi::LoadAlign(x0_2, scores + r2 * COLS + 0);
            vapi::LoadAlign(x1_2, scores + r2 * COLS + 128);
            vapi::LoadAlign(x0_3, scores + r3 * COLS + 0);
            vapi::LoadAlign(x1_3, scores + r3 * COLS + 128);

            vapi::RegTensor<half> c0_0, c0_1, c0_2, c0_3;
            vapi::RegTensor<half> c1_0, c1_1, c1_2, c1_3;
            vapi::Mul(c0_0, qh0, kh0, h);
            vapi::Mul(c0_1, qh1, kh0, h);
            vapi::Mul(c0_2, qh2, kh0, h);
            vapi::Mul(c0_3, qh3, kh0, h);
            vapi::Mul(c1_0, qh0, kh1, h);
            vapi::Mul(c1_1, qh1, kh1, h);
            vapi::Mul(c1_2, qh2, kh1, h);
            vapi::Mul(c1_3, qh3, kh1, h);

            vapi::Mul(x0_0, x0_0, c0_0, h);
            vapi::Mul(x0_1, x0_1, c0_0, h);
            vapi::Mul(x0_2, x0_2, c0_2, h);
            vapi::Mul(x0_3, x0_3, c0_3, h);
            vapi::Mul(x1_0, x1_0, c1_0, h);
            vapi::Mul(x1_1, x1_1, c1_1, h);
            vapi::Mul(x1_2, x1_2, c1_2, h);
            vapi::Mul(x1_3, x1_3, c1_3, h);

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

            vapi::StoreAlign(scores + r0 * COLS + 0, x0_0, h);
            vapi::StoreAlign(scores + r1 * COLS + 0, x0_1, h);
            vapi::StoreAlign(scores + r2 * COLS + 0, x0_2, h);
            vapi::StoreAlign(scores + r3 * COLS + 0, x0_3, h);
            vapi::StoreAlign(scores + r0 * COLS + 128, x1_0, h);
            vapi::StoreAlign(scores + r1 * COLS + 128, x1_1, h);
            vapi::StoreAlign(scores + r2 * COLS + 128, x1_2, h);
            vapi::StoreAlign(scores + r3 * COLS + 128, x1_3, h);

            vapi::RegTensor<half> rowMax0, rowMax1, rowMax2, rowMax3;
            vapi::Max(rowMax0, x0_0, x1_0, h);
            vapi::Max(rowMax1, x0_1, x1_1, h);
            vapi::Max(rowMax2, x0_2, x1_2, h);
            vapi::Max(rowMax3, x0_3, x1_3, h);

            vapi::RegTensor<half> m0, m1, m2, m3;
            vapi::ReduceMax(m0, rowMax0, h);
            vapi::ReduceMax(m1, rowMax1, h);
            vapi::ReduceMax(m2, rowMax2, h);
            vapi::ReduceMax(m3, rowMax3, h);

            vapi::StoreUnAlign<half, vapi::PostLiteral::POST_MODE_UPDATE>(blockMax, m0, maxStore, 1);
            vapi::StoreUnAlign<half, vapi::PostLiteral::POST_MODE_UPDATE>(blockMax, m1, maxStore, 1);
            vapi::StoreUnAlign<half, vapi::PostLiteral::POST_MODE_UPDATE>(blockMax, m2, maxStore, 1);
            vapi::StoreUnAlign<half, vapi::PostLiteral::POST_MODE_UPDATE>(blockMax, m3, maxStore, 1);
        }
        vapi::StoreUnAlignPost(blockMax, maxStore, 0);
        vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();

        auto f = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
        vapi::RegTensor<float> oldDen;
        vapi::LoadAlign(oldDen, denom);
        vapi::RegTensor<float> ratio = UpdateSoftmaxRescaleFactors(maxima, (__ubuf__ half *)alpha, alpha, f, h);
        vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();

        uint32_t halfChunk = 32;
        vapi::MaskReg first32 = vapi::UpdateMask<float>(halfChunk);
        vapi::MaskReg last32;
        vapi::Not(last32, first32, f);
        auto blockSum = scratch;
        vapi::UnalignReg sumStore;

        for (uint16_t row = 0; row < ROWS; row += 2) {
            const uint16_t row_a = row, row_b = row + 1;
            vapi::RegTensor<half> m_a, m_b;
            vapi::RegTensor<half> x0_a, x1_a, x2_a, x3_a;
            vapi::RegTensor<half> x0_b, x1_b, x2_b, x3_b;

            vapi::LoadAlign<half, vapi::LoadDist::DIST_BRC_B16>(m_a, maxima + row_a);
            vapi::LoadAlign<half, vapi::LoadDist::DIST_BRC_B16>(m_b, maxima + row_b);

            vapi::LoadAlign<half, vapi::LoadDist::DIST_UNPACK_B16>(x0_a, scores + row_a * COLS + 0);
            vapi::LoadAlign<half, vapi::LoadDist::DIST_UNPACK_B16>(x1_a, scores + row_a * COLS + 64);
            vapi::LoadAlign<half, vapi::LoadDist::DIST_UNPACK_B16>(x2_a, scores + row_a * COLS + 128);
            vapi::LoadAlign<half, vapi::LoadDist::DIST_UNPACK_B16>(x3_a, scores + row_a * COLS + 192);

            vapi::LoadAlign<half, vapi::LoadDist::DIST_UNPACK_B16>(x0_b, scores + row_b * COLS + 0);
            vapi::LoadAlign<half, vapi::LoadDist::DIST_UNPACK_B16>(x1_b, scores + row_b * COLS + 64);
            vapi::LoadAlign<half, vapi::LoadDist::DIST_UNPACK_B16>(x2_b, scores + row_b * COLS + 128);
            vapi::LoadAlign<half, vapi::LoadDist::DIST_UNPACK_B16>(x3_b, scores + row_b * COLS + 192);

            vapi::RegTensor<float> p0_a, p0_b, p1_a, p1_b, p2_a, p2_b, p3_a, p3_b;
            vapi::ExpSub<float, half, vapi::RegLayout::ZERO>(p0_a, x0_a, m_a, h);
            vapi::ExpSub<float, half, vapi::RegLayout::ZERO>(p0_b, x0_b, m_b, h);
            vapi::ExpSub<float, half, vapi::RegLayout::ZERO>(p1_a, x1_a, m_a, h);
            vapi::ExpSub<float, half, vapi::RegLayout::ZERO>(p1_b, x1_b, m_b, h);
            vapi::ExpSub<float, half, vapi::RegLayout::ZERO>(p2_a, x2_a, m_a, h);
            vapi::ExpSub<float, half, vapi::RegLayout::ZERO>(p2_b, x2_b, m_b, h);
            vapi::ExpSub<float, half, vapi::RegLayout::ZERO>(p3_a, x3_a, m_a, h);
            vapi::ExpSub<float, half, vapi::RegLayout::ZERO>(p3_b, x3_b, m_b, h);

            vapi::RegTensor<float8_e4m3_t> packed0_a, packed1_a, packed2_a, packed3_a;
            vapi::RegTensor<float8_e4m3_t> packed0_b, packed1_b, packed2_b, packed3_b;

            vapi::Cast<float8_e4m3_t, float, toFp8>(packed0_a, p0_a, f);
            vapi::Cast<float8_e4m3_t, float, toFp8>(packed0_b, p0_b, f);
            vapi::Cast<float8_e4m3_t, float, toFp8>(packed1_a, p1_a, f);
            vapi::Cast<float8_e4m3_t, float, toFp8>(packed1_b, p1_b, f);
            vapi::Cast<float8_e4m3_t, float, toFp8>(packed2_a, p2_a, f);
            vapi::Cast<float8_e4m3_t, float, toFp8>(packed2_b, p2_b, f);
            vapi::Cast<float8_e4m3_t, float, toFp8>(packed3_a, p3_a, f);
            vapi::Cast<float8_e4m3_t, float, toFp8>(packed3_b, p3_b, f);

            // Task 3.4: Homologous dequantization before row sum accumulation
            vapi::Cast<float, float8_e4m3_t, fromFp8>(p0_a, packed0_a, f);
            vapi::Cast<float, float8_e4m3_t, fromFp8>(p0_b, packed0_b, f);
            vapi::Cast<float, float8_e4m3_t, fromFp8>(p1_a, packed1_a, f);
            vapi::Cast<float, float8_e4m3_t, fromFp8>(p1_b, packed1_b, f);
            vapi::Cast<float, float8_e4m3_t, fromFp8>(p2_a, packed2_a, f);
            vapi::Cast<float, float8_e4m3_t, fromFp8>(p2_b, packed2_b, f);
            vapi::Cast<float, float8_e4m3_t, fromFp8>(p3_a, packed3_a, f);
            vapi::Cast<float, float8_e4m3_t, fromFp8>(p3_b, packed3_b, f);

            vapi::RegTensor<float> s01_a, s01_b, s23_a, s23_b;
            vapi::Add(s01_a, p0_a, p1_a, f);
            vapi::Add(s01_b, p0_b, p1_b, f);
            vapi::Add(s23_a, p2_a, p3_a, f);
            vapi::Add(s23_b, p2_b, p3_b, f);

            vapi::RegTensor<float> rowSum_a, rowSum_b;
            vapi::Add(rowSum_a, s01_a, s23_a, f);
            vapi::Add(rowSum_b, s01_b, s23_b, f);

            constexpr uint32_t nzStride = (ROWS + NZ_PAD_ROWS) * 32;
            StoreNZPlanePair<nzStride>(out, 0, row_a, packed0_a, first32, last32);
            StoreNZPlanePair<nzStride>(out, 0, row_b, packed0_b, first32, last32);
            StoreNZPlanePair<nzStride>(out, 2, row_a, packed1_a, first32, last32);
            StoreNZPlanePair<nzStride>(out, 2, row_b, packed1_b, first32, last32);
            StoreNZPlanePair<nzStride>(out, 4, row_a, packed2_a, first32, last32);
            StoreNZPlanePair<nzStride>(out, 4, row_b, packed2_b, first32, last32);
            StoreNZPlanePair<nzStride>(out, 6, row_a, packed3_a, first32, last32);
            StoreNZPlanePair<nzStride>(out, 6, row_b, packed3_b, first32, last32);

            vapi::RegTensor<float> sum_a, sum_b;
            vapi::ReduceSum(sum_a, rowSum_a, f);
            vapi::ReduceSum(sum_b, rowSum_b, f);
            vapi::StoreUnAlign<float, vapi::PostLiteral::POST_MODE_UPDATE>(blockSum, sum_a, sumStore, 1);
            vapi::StoreUnAlign<float, vapi::PostLiteral::POST_MODE_UPDATE>(blockSum, sum_b, sumStore, 1);
        }
        vapi::StoreUnAlignPost(blockSum, sumStore, 0);
        vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();
        vapi::RegTensor<float> sum;
        vapi::LoadAlign(sum, scratch);
        vapi::MulDstAdd(oldDen, ratio, sum, f);
        vapi::StoreAlign(denom, oldDen, f);
    }

// ============================================================================
// 6. In-Kernel Fused LSE Calculation Microkernel
// ----------------------------------------------------------------------------
// [Contract P0.3 - Architectural Contract for LSE Output & Smooth-K Restoration]:
// StoreTileLse writes the natural-log LSE of the *smoothed* scores to Global Memory (lg):
//   lse_kernel = ln(denom) + maxF = ln( sum_j exp(s_smooth(i, j)) )
// The whole online-softmax state machine (running max, denominator, alpha rescale,
// the ln(448) probability shift) lives in the natural domain end to end, so this
// function is the only place the LSE is formed and it needs no base conversion.
//
// 1. Mathematical Grounding:
//    In smooth-K attention, centered key vectors (k_j - mean_k) yield centered scores:
//      s_smooth(i, j) = q_i * (k_j - mean_k) * sm_scale
//    The true mathematical attention score is:
//      s_raw(i, j) = q_i * k_j * sm_scale = s_smooth(i, j) + (q_i * mean_k) * sm_scale
//    where delta_i = (q_i * mean_k) is a per-row constant across all j.
//    Therefore:
//      sum_j exp(s_raw(i, j)) = exp(delta_i * sm_scale) * sum_j exp(s_smooth(i, j))
//      ln(sum_j exp(s_raw(i, j))) = ln(sum_j exp(s_smooth(i, j))) + delta_i * sm_scale
//
// 2. Python Host Layer Reconstruction (sageattention2._ops):
//    The public contract restores the second term only:
//      lse_final = lse_kernel + lse_correction * sm_scale
//    where lse_correction = q @ mean_k.T (delta_i, batched host GEMM) and, for GQA,
//    the repeat_interleave mapping kh = h / (Hq / Hkv) is applied in
//    _compute_lse_correction. smooth_k=False leaves lse_kernel untouched, so the
//    aclnn entry and the PyTorch entry agree exactly on the value the kernel defines.
//
// 3. Architectural Anti-Misuse Guidelines:
//    - RULE 1: NEVER apply a log base conversion in either path; a spurious
//      multiply/divide by 1.44269504 would make the two interfaces diverge.
//    - RULE 2: NEVER add lse_correction inside the kernel. Doing so would force
//      an extra O(S) GM staging buffer plus a per-tile Q@mean_k reduction, and the
//      Q tile is not resident any more when the LSE is retired.
//    - RULE 3: Keep the emission site shared - both the Standard path
//      (single_pass_engine.h) and the Wide Streaming path (decoupled_tri_queue.h)
//      call this one function, so their LSE semantics cannot drift apart.
// Conformance reference: numeric_error_test.cpp [7] & [18].
// ============================================================================

__aicore__ inline void StoreTileLse(
    AscendC::GlobalTensor<float> &lg, AscendC::LocalTensor<float> denom,
    AscendC::LocalTensor<half> maxima, AscendC::LocalTensor<float> scratch,
    uint32_t first, uint32_t validRows, uint32_t b, uint32_t h, const SageParams &p)
{
    memory::Sync<AscendC::HardEvent::MTE3_S>();
    memory::Sync<AscendC::HardEvent::V_S>();
    if (!validRows) {
        return;
    }

    AscendC::LocalTensor<float> lse = scratch;
    AscendC::LocalTensor<float> maxF = scratch[HALF];
    AscendC::DataCopy(lse, denom, HALF);
    memory::Sync<AscendC::HardEvent::MTE2_V>();
    AscendC::Ln(lse, lse, HALF);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(maxF, maxima, AscendC::RoundMode::CAST_NONE, HALF);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(lse, lse, maxF, HALF);
    AscendC::PipeBarrier<PIPE_V>();
    memory::Sync<AscendC::HardEvent::V_MTE3>();

    AscendC::DataCopyExtParams cp{1, validRows * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    const uint64_t target_idx = (static_cast<uint64_t>(b) * p.hq + h) * p.sq + first;
    AscendC::DataCopyPad(lg[target_idx], lse, cp);
    memory::Sync<AscendC::HardEvent::MTE3_S>();
}

// ============================================================================
// 7. Output Accumulation & Rescaling Pipeline (OutAccumulator Core)
// ============================================================================
// [FlashMLA / AMLA 3-State Online Softmax & Atlas 950PR Microarchitectural Adaptation]
//
// 1. FlashMLA / AMLA Three-State Online Softmax State Machine:
//    Online Softmax processes key-value sequence blocks sequentially without
//    storing the full attention score matrix in GM/UB. For each block k:
//
//    State 1 (FIRST / Cold Start Initialization):
//      - Running max: m^{(0)} = max(S^{(0)}) - ln(448)
//      - Denominator: d^{(0)} = sum_j exp(S^{(0)}_{:,j} - m^{(0)})
//      - Accumulator: O^{(0)} = P^{(0)} * V^{(0)}
//      - Rescale factor: alpha^{(0)} = 1.0f (no previous accumulation to rescale).
//
//    State 2 (Stationary Peak / Fast-Forward Path, alpha == 1.0f):
//      - Condition: max(S^{(k)}) - ln(448) <= m^{(k-1)} (current chunk peak does
//        not exceed running maximum).
//      - Running max: m^{(k)} = m^{(k-1)}
//      - Rescale factor: alpha^{(k)} = exp(m^{(k-1)} - m^{(k)}) = exp(0) = 1.0f
//      - Denominator update: d^{(k)} = d^{(k-1)} + sum_j exp(S^{(k)}_{:,j} - m^{(k)})
//      - Accumulator update: O^{(k)} = O^{(k-1)} + P^{(k)} * V^{(k)}
//      - Microarchitecture optimization (AMLA): Bypasses the broadcast load of alpha and
//        avoids the vector multiply-add (MulDstAdd) instruction. Uses direct vector
//        addition (vapi::Add) in Vector Core, eliminating multiplier pipeline latency.
//
//    State 3 (Dynamic Rescale Transition, alpha < 1.0f):
//      - Condition: max(S^{(k)}) - ln(448) > m^{(k-1)} (new peak detected).
//      - Running max: m^{(k)} = max(S^{(k)}) - ln(448)
//      - Rescale factor: alpha^{(k)} = exp(m^{(k-1)} - m^{(k)}) < 1.0f
//      - Denominator update: d^{(k)} = alpha^{(k)} * d^{(k-1)} + sum_j exp(...)
//      - Accumulator update: O^{(k)} = alpha^{(k)} * O^{(k-1)} + P^{(k)} * V^{(k)}
//      - Execution: Vector Core uses MulDstAdd to scale prior accumulated O and
//        add the newly materialized PV partial tile in FP32 precision.
//
// 2. Prefill (M=64, N=256) vs. Decoding (M=1) Architectural Adaptation Boundary:
//    - Prefill Dense Tile (M=64, N=256):
//      * Compute Intensity: Compute-bound on Cube Core (64x128x256 MMAD).
//      * Storage: L0C capacity (256 KB) is dedicated to current tile MMAD. The
//        running O accumulator (64x128x4B = 32 KB) is maintained on Vector Core UB.
//      * Vector Parallelism: 64 queries are processed concurrently across 2 AIVs
//        (32 queries per AIV).
//      * AMLA Fast Path: For each row, when alpha[row] == 1.0f (no peak expansion),
//        the vector multiply-add is bypassed in favor of direct vector addition
//        (vapi::Add), avoiding the broadcast load of alpha and eliminating multiplier latency.
//    - Decoding Stream (M=1):
//      * Compute Intensity: Memory/bandwidth-bound on streaming KV cache.
//      * Scalar Control: Single query row allows scalar thresholding in SS buffer.

// AMLA fast bypass row chunk accumulator helper:
// When AMLA_BYPASS is true (local maximum unchanged, alpha == 1.0f), performs direct additive
// accumulation (vapi::Add), bypassing vector multiplier latency and redundant broadcast loads.
template <bool AMLA_BYPASS = false>
__simd_callee__ __attribute__((always_inline)) inline void FoldRowChunk(
    vapi::RegTensor<float> &accum,
    vapi::RegTensor<float> &partial,
    vapi::RegTensor<float> &alphaReg,
    vapi::MaskReg &mask) noexcept {
    if constexpr (AMLA_BYPASS) {
        vapi::Add(accum, accum, partial, mask);
    } else {
        vapi::MulDstAdd(accum, alphaReg, partial, mask);
    }
}

// Common output normalization, mean-V residual compensation, and FP16/BF16 cast helper.
template <class T>
__simd_callee__ __attribute__((always_inline)) inline void NormalizeCastChunk(
    __ubuf__ T *outPtr,
    vapi::RegTensor<float> &x,
    vapi::RegTensor<float> &ratio,
    vapi::RegTensor<float> &mv,
    vapi::MaskReg &mask) noexcept {
    static constexpr vapi::CastTrait cast{
        vapi::RegLayout::ZERO, vapi::SatMode::SAT, vapi::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    vapi::Mul(x, x, ratio, mask);
    vapi::Add(x, x, mv, mask);
    vapi::RegTensor<T> p;
    vapi::Cast<T, float, cast>(p, x, mask);
    vapi::StoreAlign<T, vapi::StoreDist::DIST_PACK_B32>(outPtr, p, mask);
}

template <uint32_t DP, bool AMLA_BYPASS = false>
__simd_callee__ __attribute__((always_inline)) inline void FoldRowSingle(
    __ubuf__ float *accum,
    __ubuf__ float *partial,
    __ubuf__ float *alpha,
    uint16_t row,
    vapi::MaskReg &mask) noexcept {
    vapi::RegTensor<float> a;
    if constexpr (!AMLA_BYPASS) {
        vapi::LoadAlign<float, vapi::LoadDist::DIST_BRC_B32>(a, alpha + row);
    }
    if constexpr (DP == 128) {
        vapi::RegTensor<float> x0, x1, y0, y1;
        vapi::LoadAlign(x0, accum + row * DP);
        vapi::LoadAlign(x1, accum + row * DP + 64);
        vapi::LoadAlign(y0, partial + row * DP);
        vapi::LoadAlign(y1, partial + row * DP + 64);

        FoldRowChunk<AMLA_BYPASS>(x0, y0, a, mask);
        FoldRowChunk<AMLA_BYPASS>(x1, y1, a, mask);

        vapi::StoreAlign(accum + row * DP, x0, mask);
        vapi::StoreAlign(accum + row * DP + 64, x1, mask);
    } else {
        vapi::RegTensor<float> x, y;
        vapi::LoadAlign(x, accum + row * DP);
        vapi::LoadAlign(y, partial + row * DP);

        FoldRowChunk<AMLA_BYPASS>(x, y, a, mask);

        vapi::StoreAlign(accum + row * DP, x, mask);
    }
}

template <uint32_t DP, bool AMLA_BYPASS0 = false, bool AMLA_BYPASS1 = false>
__simd_callee__ __attribute__((always_inline)) inline void FoldRowPair(
    __ubuf__ float *accum,
    __ubuf__ float *partial0,
    __ubuf__ float *partial1,
    __ubuf__ float *alpha0,
    __ubuf__ float *alpha1,
    uint16_t row,
    vapi::MaskReg &mask) noexcept {
    vapi::RegTensor<float> a0, a1;
    if constexpr (!AMLA_BYPASS0) {
        vapi::LoadAlign<float, vapi::LoadDist::DIST_BRC_B32>(a0, alpha0 + row);
    }
    if constexpr (!AMLA_BYPASS1) {
        vapi::LoadAlign<float, vapi::LoadDist::DIST_BRC_B32>(a1, alpha1 + row);
    }

    if constexpr (DP == 128) {
        vapi::RegTensor<float> x0, x1, y0, y1, z0, z1;
        vapi::LoadAlign(x0, accum + row * DP);
        vapi::LoadAlign(x1, accum + row * DP + 64);
        vapi::LoadAlign(y0, partial0 + row * DP);
        vapi::LoadAlign(y1, partial0 + row * DP + 64);

        FoldRowChunk<AMLA_BYPASS0>(x0, y0, a0, mask);
        FoldRowChunk<AMLA_BYPASS0>(x1, y1, a0, mask);

        vapi::LoadAlign(z0, partial1 + row * DP);
        vapi::LoadAlign(z1, partial1 + row * DP + 64);

        FoldRowChunk<AMLA_BYPASS1>(x0, z0, a1, mask);
        FoldRowChunk<AMLA_BYPASS1>(x1, z1, a1, mask);

        vapi::StoreAlign(accum + row * DP, x0, mask);
        vapi::StoreAlign(accum + row * DP + 64, x1, mask);
    } else {
        vapi::RegTensor<float> x, y, z;
        vapi::LoadAlign(x, accum + row * DP);
        vapi::LoadAlign(y, partial0 + row * DP);

        FoldRowChunk<AMLA_BYPASS0>(x, y, a0, mask);

        vapi::LoadAlign(z, partial1 + row * DP);

        FoldRowChunk<AMLA_BYPASS1>(x, z, a1, mask);

        vapi::StoreAlign(accum + row * DP, x, mask);
    }
}

template <uint32_t DP, bool AMLA_BYPASS = false>
__simd_callee__ __attribute__((always_inline)) inline void AccumulateFp32Core(
    __ubuf__ float *accum, __ubuf__ float *partial, __ubuf__ float *alpha, vapi::MaskReg &f) noexcept {
    if constexpr (DP == 128) {
        for (uint16_t row = 0; row < 64; row += 2) {
            FoldRowSingle<DP, AMLA_BYPASS>(accum, partial, alpha, row + 0, f);
            FoldRowSingle<DP, AMLA_BYPASS>(accum, partial, alpha, row + 1, f);
        }
    } else {
        for (uint16_t row = 0; row < 64; row += 4) {
            FoldRowSingle<DP, AMLA_BYPASS>(accum, partial, alpha, row + 0, f);
            FoldRowSingle<DP, AMLA_BYPASS>(accum, partial, alpha, row + 1, f);
            FoldRowSingle<DP, AMLA_BYPASS>(accum, partial, alpha, row + 2, f);
            FoldRowSingle<DP, AMLA_BYPASS>(accum, partial, alpha, row + 3, f);
        }
    }
}

template <uint32_t DP, bool AMLA_BYPASS = false>
__simd_vf__ inline void AccumulateFp32VF(__ubuf__ float *accum, __ubuf__ float *partial,
                                         __ubuf__ float *alpha) {
    vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();
    auto f = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
    AccumulateFp32Core<DP, AMLA_BYPASS>(accum, partial, alpha, f);
}

template <uint32_t DP, bool AMLA_BYPASS0 = false, bool AMLA_BYPASS1 = false>
__simd_callee__ __attribute__((always_inline)) inline void AccumulateFp32PairCore(
    __ubuf__ float *accum, __ubuf__ float *partial0, __ubuf__ float *partial1,
    __ubuf__ float *alpha0, __ubuf__ float *alpha1, vapi::MaskReg &f) noexcept {
    if constexpr (DP == 128) {
        for (uint16_t row = 0; row < 64; row += 2) {
            FoldRowPair<DP, AMLA_BYPASS0, AMLA_BYPASS1>(accum, partial0, partial1, alpha0, alpha1, row + 0, f);
            FoldRowPair<DP, AMLA_BYPASS0, AMLA_BYPASS1>(accum, partial0, partial1, alpha0, alpha1, row + 1, f);
        }
    } else {
        for (uint16_t row = 0; row < 64; row += 4) {
            FoldRowPair<DP, AMLA_BYPASS0, AMLA_BYPASS1>(accum, partial0, partial1, alpha0, alpha1, row + 0, f);
            FoldRowPair<DP, AMLA_BYPASS0, AMLA_BYPASS1>(accum, partial0, partial1, alpha0, alpha1, row + 1, f);
            FoldRowPair<DP, AMLA_BYPASS0, AMLA_BYPASS1>(accum, partial0, partial1, alpha0, alpha1, row + 2, f);
            FoldRowPair<DP, AMLA_BYPASS0, AMLA_BYPASS1>(accum, partial0, partial1, alpha0, alpha1, row + 3, f);
        }
    }
}

template <uint32_t DP, bool AMLA_BYPASS0 = false, bool AMLA_BYPASS1 = false>
__simd_vf__ inline void AccumulateFp32PairVF(__ubuf__ float *accum, __ubuf__ float *partial0,
                                             __ubuf__ float *partial1, __ubuf__ float *alpha0,
                                             __ubuf__ float *alpha1) {
    vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();
    auto f = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
    AccumulateFp32PairCore<DP, AMLA_BYPASS0, AMLA_BYPASS1>(accum, partial0, partial1, alpha0, alpha1, f);
}

// Round per-block PV partial to FP16 for fp32+fp16 accumulation
__simd_vf__ inline void RoundPartialFp16VF(__ubuf__ float *partial, uint32_t count) {
    vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();
    auto f = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
    static constexpr vapi::CastTrait narrow{vapi::RegLayout::ZERO, vapi::SatMode::NO_SAT, vapi::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr vapi::CastTrait widen{vapi::RegLayout::ZERO, vapi::SatMode::UNKNOWN, vapi::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    for (uint32_t i = 0; i < count; i += 128) {
        vapi::RegTensor<float> x0, x1;
        vapi::RegTensor<half> packed0, packed1;
        vapi::LoadAlign(x0, partial + i);
        vapi::LoadAlign(x1, partial + i + 64);
        vapi::Cast<half, float, narrow>(packed0, x0, f);
        vapi::Cast<half, float, narrow>(packed1, x1, f);
        vapi::Cast<float, half, widen>(x0, packed0, f);
        vapi::Cast<float, half, widen>(x1, packed1, f);
        vapi::StoreAlign(partial + i, x0, f);
        vapi::StoreAlign(partial + i + 64, x1, f);
    }
    vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();
}

// Normalize and cast output (standard path)
template <class T, uint32_t DP>
__simd_vf__ inline void NormalizeCastVF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *denom,
                                        __ubuf__ float *scratch, __ubuf__ float *meanV) {
    vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();
    auto f = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
    vapi::RegTensor<float> den, inv;
    vapi::LoadAlign(den, denom);
    core::FastNormalizerReciprocal(inv, den, f);
    vapi::StoreAlign(scratch, inv, f);
    vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();

    if constexpr (DP == 128) {
        vapi::RegTensor<float> mv0, mv1;
        vapi::LoadAlign(mv0, meanV);
        vapi::LoadAlign(mv1, meanV + 64);
        for (uint16_t row = 0; row < 64; ++row) {
            vapi::RegTensor<float> ratio, x0, x1;
            vapi::LoadAlign<float, vapi::LoadDist::DIST_BRC_B32>(ratio, scratch + row);
            vapi::LoadAlign(x0, accum + row * DP);
            vapi::LoadAlign(x1, accum + row * DP + 64);
            NormalizeCastChunk(out + row * DP, x0, ratio, mv0, f);
            NormalizeCastChunk(out + row * DP + 64, x1, ratio, mv1, f);
        }
    } else {
        vapi::RegTensor<float> mv;
        vapi::LoadAlign(mv, meanV);
        for (uint16_t row = 0; row < 64; ++row) {
            vapi::RegTensor<float> ratio, x;
            vapi::LoadAlign<float, vapi::LoadDist::DIST_BRC_B32>(ratio, scratch + row);
            vapi::LoadAlign(x, accum + row * DP);
            NormalizeCastChunk(out + row * DP, x, ratio, mv, f);
        }
    }
}

// ============================================================================
// 8. Wide Path Output & Denominator Folding Pipeline
// ============================================================================

template <bool FIRST>
__simd_callee__ __attribute__((always_inline)) inline void WideFoldDenomSingle(
    vapi::RegTensor<float> &denAccum,
    __ubuf__ float *alphaBuf,
    __ubuf__ float *denomBuf,
    vapi::MaskReg &mask) noexcept {
    vapi::RegTensor<float> sumChunk;
    vapi::LoadAlign(sumChunk, alphaBuf + 64);
    if constexpr (FIRST) {
        denAccum = sumChunk;
    } else {
        vapi::RegTensor<float> ratioVal;
        vapi::LoadAlign(denAccum, denomBuf);
        vapi::LoadAlign(ratioVal, alphaBuf);
        vapi::MulDstAdd(denAccum, ratioVal, sumChunk, mask);
    }
    vapi::StoreAlign(denomBuf, denAccum, mask);
}

template <bool FIRST>
__simd_callee__ __attribute__((always_inline)) inline void WideFoldDenomPair(
    vapi::RegTensor<float> &denAccum,
    __ubuf__ float *alpha0Buf,
    __ubuf__ float *alpha1Buf,
    __ubuf__ float *denomBuf,
    vapi::MaskReg &mask) noexcept {
    vapi::RegTensor<float> sum0;
    vapi::LoadAlign(sum0, alpha0Buf + 64);
    if constexpr (FIRST) {
        denAccum = sum0;
    } else {
        vapi::RegTensor<float> a0;
        vapi::LoadAlign(denAccum, denomBuf);
        vapi::LoadAlign(a0, alpha0Buf);
        vapi::MulDstAdd(denAccum, a0, sum0, mask);
    }
    vapi::RegTensor<float> a1, sum1;
    vapi::LoadAlign(a1, alpha1Buf);
    vapi::LoadAlign(sum1, alpha1Buf + 64);
    vapi::MulDstAdd(denAccum, a1, sum1, mask);
    vapi::StoreAlign(denomBuf, denAccum, mask);
}

template <class T, uint32_t DP, bool FIRST, bool AMLA_BYPASS = false>
__simd_vf__ inline void FinalizeWideFp32VF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial,
                                           __ubuf__ float *alpha, __ubuf__ float *denom, __ubuf__ float *scratch,
                                           __ubuf__ float *meanV) {
    static_assert(DP == 128, "FinalizeWideFp32VF is specialized for DP == 128");
    vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();
    auto f = vapi::CreateMask<float, vapi::MaskPattern::ALL>();

    vapi::RegTensor<float> den, inv;
    WideFoldDenomSingle<FIRST>(den, alpha, denom, f);
    core::FastNormalizerReciprocal(inv, den, f);
    vapi::StoreAlign(scratch, inv, f);
    vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();

    vapi::RegTensor<float> mv0, mv1;
    vapi::LoadAlign(mv0, meanV);
    vapi::LoadAlign(mv1, meanV + 64);
    for (uint16_t row = 0; row < 64; ++row) {
        vapi::RegTensor<float> x0, x1;
        if constexpr (FIRST) {
            vapi::LoadAlign(x0, partial + row * DP);
            vapi::LoadAlign(x1, partial + row * DP + 64);
        } else {
            vapi::LoadAlign(x0, accum + row * DP);
            vapi::LoadAlign(x1, accum + row * DP + 64);
            vapi::RegTensor<float> y0, y1, a;
            vapi::LoadAlign(y0, partial + row * DP);
            vapi::LoadAlign(y1, partial + row * DP + 64);
            if constexpr (!AMLA_BYPASS) {
                vapi::LoadAlign<float, vapi::LoadDist::DIST_BRC_B32>(a, alpha + row);
            }
            FoldRowChunk<AMLA_BYPASS>(x0, y0, a, f);
            FoldRowChunk<AMLA_BYPASS>(x1, y1, a, f);
        }
        vapi::RegTensor<float> ratio;
        vapi::LoadAlign<float, vapi::LoadDist::DIST_BRC_B32>(ratio, scratch + row);
        NormalizeCastChunk(out + row * DP, x0, ratio, mv0, f);
        NormalizeCastChunk(out + row * DP + 64, x1, ratio, mv1, f);
    }
}

template <uint32_t DP>
__simd_vf__ inline void InitializeWideFp32VF(__ubuf__ float *accum, __ubuf__ float *partial,
                                             __ubuf__ float *alpha, __ubuf__ float *denom) {
    static_assert(DP == 128, "InitializeWideFp32VF is specialized for DP == 128");
    vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();
    auto f = vapi::CreateMask<float, vapi::MaskPattern::ALL>();

    vapi::RegTensor<float> den;
    WideFoldDenomSingle<true>(den, alpha, denom, f);

    for (uint16_t row = 0; row < 64; ++row) {
        vapi::RegTensor<float> x0, x1;
        vapi::LoadAlign(x0, partial + row * DP);
        vapi::LoadAlign(x1, partial + row * DP + 64);
        vapi::StoreAlign(accum + row * DP, x0, f);
        vapi::StoreAlign(accum + row * DP + 64, x1, f);
    }
}

template <uint32_t DP>
__simd_vf__ inline void AccumulateWideFp32VF(__ubuf__ float *accum, __ubuf__ float *partial,
                                             __ubuf__ float *alpha, __ubuf__ float *denom) {
    static_assert(DP == 128, "AccumulateWideFp32VF is specialized for DP == 128");
    vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();
    auto f = vapi::CreateMask<float, vapi::MaskPattern::ALL>();

    vapi::RegTensor<float> den;
    WideFoldDenomSingle<false>(den, alpha, denom, f);
    AccumulateFp32Core<DP>(accum, partial, alpha, f);
}

template <class T, uint32_t DP, bool FIRST, bool LAST, bool AMLA_BYPASS0 = false, bool AMLA_BYPASS1 = false>
__simd_vf__ inline void UpdateWidePairVF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial0,
                                         __ubuf__ float *partial1, __ubuf__ float *alpha0, __ubuf__ float *alpha1,
                                         __ubuf__ float *denom, __ubuf__ float *scratch, __ubuf__ float *meanV) {
    static_assert(DP == 128, "UpdateWidePairVF is specialized for DP == 128");
    vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();
    auto f = vapi::CreateMask<float, vapi::MaskPattern::ALL>();

    vapi::RegTensor<float> den;
    WideFoldDenomPair<FIRST>(den, alpha0, alpha1, denom, f);

    if constexpr (LAST) {
        vapi::RegTensor<float> inv;
        core::FastNormalizerReciprocal(inv, den, f);
        vapi::StoreAlign(scratch, inv, f);
        vapi::LocalMemBar<vapi::MemType::VEC_STORE, vapi::MemType::VEC_LOAD>();
    }
    vapi::RegTensor<float> mv0, mv1;
    if constexpr (LAST) {
        vapi::LoadAlign(mv0, meanV);
        vapi::LoadAlign(mv1, meanV + 64);
    }
    for (uint16_t row = 0; row < 64; ++row) {
        vapi::RegTensor<float> x0, x1;
        if constexpr (FIRST) {
            vapi::LoadAlign(x0, partial0 + row * DP);
            vapi::LoadAlign(x1, partial0 + row * DP + 64);
        } else {
            vapi::LoadAlign(x0, accum + row * DP);
            vapi::LoadAlign(x1, accum + row * DP + 64);
            vapi::RegTensor<float> y0, y1, a0;
            vapi::LoadAlign(y0, partial0 + row * DP);
            vapi::LoadAlign(y1, partial0 + row * DP + 64);
            if constexpr (!AMLA_BYPASS0) {
                vapi::LoadAlign<float, vapi::LoadDist::DIST_BRC_B32>(a0, alpha0 + row);
            }
            FoldRowChunk<AMLA_BYPASS0>(x0, y0, a0, f);
            FoldRowChunk<AMLA_BYPASS0>(x1, y1, a0, f);
        }

        vapi::RegTensor<float> z0, z1, a1;
        vapi::LoadAlign(z0, partial1 + row * DP);
        vapi::LoadAlign(z1, partial1 + row * DP + 64);
        if constexpr (!AMLA_BYPASS1) {
            vapi::LoadAlign<float, vapi::LoadDist::DIST_BRC_B32>(a1, alpha1 + row);
        }
        FoldRowChunk<AMLA_BYPASS1>(x0, z0, a1, f);
        FoldRowChunk<AMLA_BYPASS1>(x1, z1, a1, f);

        if constexpr (LAST) {
            vapi::RegTensor<float> ratio;
            vapi::LoadAlign<float, vapi::LoadDist::DIST_BRC_B32>(ratio, scratch + row);
            NormalizeCastChunk(out + row * DP, x0, ratio, mv0, f);
            NormalizeCastChunk(out + row * DP + 64, x1, ratio, mv1, f);
        } else {
            vapi::StoreAlign(accum + row * DP, x0, f);
            vapi::StoreAlign(accum + row * DP + 64, x1, f);
        }
    }
}

class OutAccumulator {
public:
    template <uint32_t DP>
    __aicore__ static inline void AccumulateFp32(__ubuf__ float *accum, __ubuf__ float *partial,
                                                  __ubuf__ float *alpha) {
        AccumulateFp32VF<DP>(accum, partial, alpha);
    }

    template <uint32_t DP>
    __aicore__ static inline void AccumulateFp32Pair(__ubuf__ float *accum, __ubuf__ float *partial0,
                                                      __ubuf__ float *partial1, __ubuf__ float *alpha0,
                                                      __ubuf__ float *alpha1) {
        AccumulateFp32PairVF<DP>(accum, partial0, partial1, alpha0, alpha1);
    }

    // Two-level PV accumulation policy shared by both datapaths. `fp32+fp16` (the default
    // acceptance config) folds every block partial through an FP16 round-trip before the FP32
    // running accumulator absorbs it; the FP32 policies fold the L0C result as-is, because the
    // 950PR cube accumulator is already FP32, so `fp32` and `fp32+fp32` are numerically identical here.
    __aicore__ static inline void FoldPartial(__ubuf__ float *partial, const SageParams &p) {
        if (p.accum == SagePvAccum::FP32_PLUS_FP16) {
            RoundPartialFp16VF(partial, HALF * p.dp);
        }
    }

    template <class T, uint32_t DP>
    __aicore__ static inline void NormalizeCast(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *denom,
                                                 __ubuf__ float *scratch, __ubuf__ float *meanV) {
        NormalizeCastVF<T, DP>(out, accum, denom, scratch, meanV);
    }

    template <class T, uint32_t DP, bool FIRST>
    __aicore__ static inline void FinalizeWideFp32(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial,
                                                    __ubuf__ float *alpha, __ubuf__ float *denom, __ubuf__ float *scratch,
                                                    __ubuf__ float *meanV) {
        FinalizeWideFp32VF<T, DP, FIRST>(out, accum, partial, alpha, denom, scratch, meanV);
    }

    // Single-block wide-path accumulator primitives backing RetireSingleStep:
    // persist the running state without emitting output. `InitializeWideFp32`
    // absorbs the first block (denom := blockSum, accum := partial);
    // `AccumulateWideFp32` folds every later block exactly like the FIRST=false
    // leg of FinalizeWideFp32VF, minus the normalizer/output stage.
    template <uint32_t DP>
    __aicore__ static inline void InitializeWideFp32(__ubuf__ float *accum, __ubuf__ float *partial,
                                                     __ubuf__ float *alpha, __ubuf__ float *denom) {
        InitializeWideFp32VF<DP>(accum, partial, alpha, denom);
    }

    template <uint32_t DP>
    __aicore__ static inline void AccumulateWideFp32(__ubuf__ float *accum, __ubuf__ float *partial,
                                                     __ubuf__ float *alpha, __ubuf__ float *denom) {
        AccumulateWideFp32VF<DP>(accum, partial, alpha, denom);
    }

    template <class T, uint32_t DP, bool FIRST, bool LAST>
    __aicore__ static inline void UpdateWidePair(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial0,
                                                  __ubuf__ float *partial1, __ubuf__ float *alpha0, __ubuf__ float *alpha1,
                                                  __ubuf__ float *denom, __ubuf__ float *scratch, __ubuf__ float *meanV) {
        UpdateWidePairVF<T, DP, FIRST, LAST>(out, accum, partial0, partial1, alpha0, alpha1, denom, scratch, meanV);
    }
};

#endif // __CCE__

} // namespace sage::microkernel

namespace sage {
#if defined(__CCE__)
using microkernel::StoreTileLse;
using microkernel::OutAccumulator;
#endif
}
