#pragma once

#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"
#include "math_approximations.h"
#include "ring_buffer_layout.h"
#include "catlass_fractal_packer.h"
#include "credit_handshake.h"
#include "mxfp8_e4m3_encoder.h"

namespace sage::microkernel {

#if defined(__CCE__)

// Channel-wise Column Sum Reduction for Sequence Statistics
template <class T, uint32_t DP>
__simd_vf__ inline void StatsSumImpl(__ubuf__ T *rows, __ubuf__ float *state, uint32_t count) {
    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    auto maskF32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    static constexpr AscendC::MicroAPI::CastTrait unpackToF32{
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    for (uint32_t colOffset = 0; colOffset < DP; colOffset += 64) {
        AscendC::MicroAPI::RegTensor<float> totalAcc;
        AscendC::MicroAPI::LoadAlign(totalAcc, state + colOffset);
        uint32_t rIdx = 0;
        for (; rIdx + 1 < count; rIdx += 2) {
            if constexpr (AscendC::IsSameType<T, float>::value) {
                AscendC::MicroAPI::RegTensor<float> elem0, elem1;
                AscendC::MicroAPI::LoadAlign(elem0, rows + static_cast<uint64_t>(rIdx) * DP + colOffset);
                AscendC::MicroAPI::LoadAlign(elem1, rows + static_cast<uint64_t>(rIdx + 1) * DP + colOffset);
                AscendC::MicroAPI::Add(totalAcc, totalAcc, elem0, maskF32);
                AscendC::MicroAPI::Add(totalAcc, totalAcc, elem1, maskF32);
            } else {
                AscendC::MicroAPI::RegTensor<T> raw0, raw1;
                AscendC::MicroAPI::RegTensor<float> fVal0, fVal1;
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    raw0, rows + static_cast<uint64_t>(rIdx) * DP + colOffset);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    raw1, rows + static_cast<uint64_t>(rIdx + 1) * DP + colOffset);
                AscendC::MicroAPI::Cast<float, T, unpackToF32>(fVal0, raw0, maskF32);
                AscendC::MicroAPI::Cast<float, T, unpackToF32>(fVal1, raw1, maskF32);
                AscendC::MicroAPI::Add(totalAcc, totalAcc, fVal0, maskF32);
                AscendC::MicroAPI::Add(totalAcc, totalAcc, fVal1, maskF32);
            }
        }
        if (rIdx < count) {
            if constexpr (AscendC::IsSameType<T, float>::value) {
                AscendC::MicroAPI::RegTensor<float> elemLast;
                AscendC::MicroAPI::LoadAlign(elemLast, rows + static_cast<uint64_t>(rIdx) * DP + colOffset);
                AscendC::MicroAPI::Add(totalAcc, totalAcc, elemLast, maskF32);
            } else {
                AscendC::MicroAPI::RegTensor<T> rawLast;
                AscendC::MicroAPI::RegTensor<float> fValLast;
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    rawLast, rows + static_cast<uint64_t>(rIdx) * DP + colOffset);
                AscendC::MicroAPI::Cast<float, T, unpackToF32>(fValLast, rawLast, maskF32);
                AscendC::MicroAPI::Add(totalAcc, totalAcc, fValLast, maskF32);
            }
        }
        AscendC::MicroAPI::StoreAlign(state + colOffset, totalAcc, maskF32);
    }
}

template <class T>
__aicore__ inline void StatsSumVF(AscendC::LocalTensor<T> rows, AscendC::LocalTensor<float> state,
                                  uint32_t count, uint32_t dp) {
    auto inputPtr = reinterpret_cast<__ubuf__ T *>(rows.GetPhyAddr());
    auto sumPtr = reinterpret_cast<__ubuf__ float *>(state.GetPhyAddr());
    if (dp == 128) {
        StatsSumImpl<T, 128>(inputPtr, sumPtr, count);
    } else {
        StatsSumImpl<T, 64>(inputPtr, sumPtr, count);
    }
}

// Subtracts channel means from rows in Unified Buffer
template <class T, uint32_t DP>
__simd_vf__ inline void CenterVImpl(__ubuf__ T *rows, __ubuf__ float *mean, uint32_t count) {
    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    auto maskF32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    static constexpr AscendC::MicroAPI::CastTrait castToFloat{
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castToTarget{
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    for (uint32_t c = 0; c < DP; c += 64) {
        AscendC::MicroAPI::RegTensor<float> meanCol;
        AscendC::MicroAPI::LoadAlign(meanCol, mean + c);
        uint32_t rIdx = 0;
        for (; rIdx + 1 < count; rIdx += 2) {
            AscendC::MicroAPI::RegTensor<T> raw0, raw1;
            AscendC::MicroAPI::RegTensor<float> valF32_0, valF32_1;
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                raw0, rows + static_cast<uint64_t>(rIdx) * DP + c);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                raw1, rows + static_cast<uint64_t>(rIdx + 1) * DP + c);
            AscendC::MicroAPI::Cast<float, T, castToFloat>(valF32_0, raw0, maskF32);
            AscendC::MicroAPI::Cast<float, T, castToFloat>(valF32_1, raw1, maskF32);
            AscendC::MicroAPI::Sub(valF32_0, valF32_0, meanCol, maskF32);
            AscendC::MicroAPI::Sub(valF32_1, valF32_1, meanCol, maskF32);
            AscendC::MicroAPI::Cast<T, float, castToTarget>(raw0, valF32_0, maskF32);
            AscendC::MicroAPI::Cast<T, float, castToTarget>(raw1, valF32_1, maskF32);
            AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                rows + static_cast<uint64_t>(rIdx) * DP + c, raw0, maskF32);
            AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                rows + static_cast<uint64_t>(rIdx + 1) * DP + c, raw1, maskF32);
        }
        if (rIdx < count) {
            AscendC::MicroAPI::RegTensor<T> rawLast;
            AscendC::MicroAPI::RegTensor<float> valF32Last;
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                rawLast, rows + static_cast<uint64_t>(rIdx) * DP + c);
            AscendC::MicroAPI::Cast<float, T, castToFloat>(valF32Last, rawLast, maskF32);
            AscendC::MicroAPI::Sub(valF32Last, valF32Last, meanCol, maskF32);
            AscendC::MicroAPI::Cast<T, float, castToTarget>(rawLast, valF32Last, maskF32);
            AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                rows + static_cast<uint64_t>(rIdx) * DP + c, rawLast, maskF32);
        }
    }
}

template <class T>
__aicore__ inline void CenterVRowsVF(__ubuf__ T *rows, __ubuf__ float *mean, uint32_t dp, uint32_t count) {
    if (dp == 128) {
        CenterVImpl<T, 128>(rows, mean, count);
    } else {
        CenterVImpl<T, 64>(rows, mean, count);
    }
}

// Harmonizes group-shared INT8 scale across token groups (4 for per_thread, 32 for per_warp)
__simd_callee__ __attribute__((always_inline)) inline void BroadcastGroupMaxima(__ubuf__ float *maxima, uint32_t group) {
    if (group <= 1) return;
    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    for (uint32_t offset = 0; offset < 64; offset += group) {
        float lead = maxima[offset];
        for (uint32_t i = 1; i < group; ++i) {
            float v = maxima[offset + i];
            if (v > lead) lead = v;
        }
        for (uint32_t i = 0; i < group; ++i) {
            maxima[offset + i] = lead;
        }
    }
    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
}

// Converts float scales to half precision with precomputed balancing factor
__simd_callee__ __attribute__((always_inline)) inline void PackFixedScaleBlock(__ubuf__ float *scales, __ubuf__ half *halfScales,
                                                float factor, uint32_t valid) {
    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    auto mask = AscendC::MicroAPI::UpdateMask<float>(valid);
    AscendC::MicroAPI::RegTensor<float> s0, factorReg;
    AscendC::MicroAPI::RegTensor<half> h0;
    AscendC::MicroAPI::Duplicate(factorReg, factor);
    AscendC::MicroAPI::LoadAlign(s0, scales);
    AscendC::MicroAPI::Mul(s0, s0, factorReg, mask);
    static constexpr AscendC::MicroAPI::CastTrait castToHalf{
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    AscendC::MicroAPI::Cast<half, float, castToHalf>(h0, s0, mask);
    AscendC::MicroAPI::StoreAlign<half, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(halfScales, h0, mask);
}

// High-performance INT8 Symmetric Quantization Implementation
template <uint32_t DP, bool SMOOTH>
__simd_vf__ inline void QuantRowsImpl(__ubuf__ float *x, [[maybe_unused]] __ubuf__ float *mean,
                                      __ubuf__ float *scales, __ubuf__ int8_t *out,
                                      __ubuf__ half *halfScales, float factor,
                                      __ubuf__ float *inverseScales, uint32_t valid, uint32_t group) {
    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    auto full = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::RegTensor<float> mean0, mean1;
    if constexpr (SMOOTH) {
        AscendC::MicroAPI::LoadAlign(mean0, mean);
        if constexpr (DP == 128) {
            AscendC::MicroAPI::LoadAlign(mean1, mean + 64);
        }
    }

    auto dstScale = scales;
    AscendC::MicroAPI::UnalignReg packState;

    if constexpr (DP == 128) {
        for (uint16_t r = 0; r < 64; r += 2) {
            AscendC::MicroAPI::RegTensor<float> v0_lo, v1_lo, v0_hi, v1_hi;
            AscendC::MicroAPI::LoadAlign(v0_lo, x + static_cast<uint32_t>(r) * DP);
            AscendC::MicroAPI::LoadAlign(v1_lo, x + static_cast<uint32_t>(r + 1) * DP);
            AscendC::MicroAPI::LoadAlign(v0_hi, x + static_cast<uint32_t>(r) * DP + 64);
            AscendC::MicroAPI::LoadAlign(v1_hi, x + static_cast<uint32_t>(r + 1) * DP + 64);

            if constexpr (SMOOTH) {
                AscendC::MicroAPI::Sub(v0_lo, v0_lo, mean0, full);
                AscendC::MicroAPI::Sub(v1_lo, v1_lo, mean0, full);
                AscendC::MicroAPI::Sub(v0_hi, v0_hi, mean1, full);
                AscendC::MicroAPI::Sub(v1_hi, v1_hi, mean1, full);
                AscendC::MicroAPI::StoreAlign(x + static_cast<uint32_t>(r) * DP, v0_lo, full);
                AscendC::MicroAPI::StoreAlign(x + static_cast<uint32_t>(r) * DP + 64, v0_hi, full);
                AscendC::MicroAPI::StoreAlign(x + static_cast<uint32_t>(r + 1) * DP, v1_lo, full);
                AscendC::MicroAPI::StoreAlign(x + static_cast<uint32_t>(r + 1) * DP + 64, v1_hi, full);
            }

            AscendC::MicroAPI::Abs(v0_lo, v0_lo, full);
            AscendC::MicroAPI::Abs(v1_lo, v1_lo, full);
            AscendC::MicroAPI::Abs(v0_hi, v0_hi, full);
            AscendC::MicroAPI::Abs(v1_hi, v1_hi, full);

            AscendC::MicroAPI::RegTensor<float> peak0, peak1;
            AscendC::MicroAPI::Max(peak0, v0_lo, v0_hi, full);
            AscendC::MicroAPI::Max(peak1, v1_lo, v1_hi, full);
            AscendC::MicroAPI::ReduceMax(peak0, peak0, full);
            AscendC::MicroAPI::ReduceMax(peak1, peak1, full);

            AscendC::MicroAPI::StoreUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dstScale, peak0, packState, 1);
            AscendC::MicroAPI::StoreUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dstScale, peak1, packState, 1);
        }
    } else {
        for (uint16_t r = 0; r < 64; r += 2) {
            AscendC::MicroAPI::RegTensor<float> v0, v1;
            AscendC::MicroAPI::LoadAlign(v0, x + static_cast<uint32_t>(r) * DP);
            AscendC::MicroAPI::LoadAlign(v1, x + static_cast<uint32_t>(r + 1) * DP);
            if constexpr (SMOOTH) {
                AscendC::MicroAPI::Sub(v0, v0, mean0, full);
                AscendC::MicroAPI::Sub(v1, v1, mean0, full);
                AscendC::MicroAPI::StoreAlign(x + static_cast<uint32_t>(r) * DP, v0, full);
                AscendC::MicroAPI::StoreAlign(x + static_cast<uint32_t>(r + 1) * DP, v1, full);
            }
            AscendC::MicroAPI::Abs(v0, v0, full);
            AscendC::MicroAPI::Abs(v1, v1, full);
            AscendC::MicroAPI::ReduceMax(v0, v0, full);
            AscendC::MicroAPI::ReduceMax(v1, v1, full);
            AscendC::MicroAPI::StoreUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dstScale, v0, packState, 1);
            AscendC::MicroAPI::StoreUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dstScale, v1, packState, 1);
        }
    }

    AscendC::MicroAPI::StoreUnAlignPost(dstScale, packState, 0);
    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    BroadcastGroupMaxima(scales, group);

    AscendC::MicroAPI::RegTensor<float> peakVec, zeroFloor, defaultUnit, invDivisor;
    AscendC::MicroAPI::Duplicate(zeroFloor, 0.0f);
    AscendC::MicroAPI::Duplicate(defaultUnit, 1.0f);
    AscendC::MicroAPI::Duplicate(invDivisor, sage::core::INV_INT8_MAX);

    AscendC::MicroAPI::LoadAlign(peakVec, scales);
    AscendC::MicroAPI::Mul(peakVec, peakVec, invDivisor, full);

    AscendC::MicroAPI::MaskReg isPositive;
    AscendC::MicroAPI::Compare<float, AscendC::CMPMODE::GT>(isPositive, peakVec, zeroFloor, full);
    AscendC::MicroAPI::Select(peakVec, peakVec, defaultUnit, isPositive);
    AscendC::MicroAPI::StoreAlign(scales, peakVec, full);

    AscendC::MicroAPI::RegTensor<float> invScaleVec;
    // Qualify our kernel artifact: AscendC::MicroAPI also declares FastReciprocal
    // overloads on 9.2, which make the 3-arg MicroAPI-qualified call ambiguous.
    sage::core::FastReciprocal(invScaleVec, peakVec, full);
    AscendC::MicroAPI::StoreAlign(inverseScales, invScaleVec, full);
    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    static constexpr AscendC::MicroAPI::CastTrait castToInt32{
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    for (uint16_t r = 0; r < 64; r += 2) {
        AscendC::MicroAPI::RegTensor<float> inv0, inv1;
        AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(inv0, inverseScales + r);
        AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(inv1, inverseScales + r + 1);

        for (uint16_t c = 0; c < DP; c += 64) {
            AscendC::MicroAPI::RegTensor<float> val0, val1;
            AscendC::MicroAPI::LoadAlign(val0, x + static_cast<uint32_t>(r) * DP + c);
            AscendC::MicroAPI::LoadAlign(val1, x + static_cast<uint32_t>(r + 1) * DP + c);

            AscendC::MicroAPI::Mul(val0, val0, inv0, full);
            AscendC::MicroAPI::Mul(val1, val1, inv1, full);

            AscendC::MicroAPI::RegTensor<int32_t> i0, i1;
            AscendC::MicroAPI::Cast<int32_t, float, castToInt32>(i0, val0, full);
            AscendC::MicroAPI::Cast<int32_t, float, castToInt32>(i1, val1, full);

            AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                out + static_cast<uint32_t>(r) * DP + c, (AscendC::MicroAPI::RegTensor<int8_t> &)i0, full);
            AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                out + static_cast<uint32_t>(r + 1) * DP + c, (AscendC::MicroAPI::RegTensor<int8_t> &)i1, full);
        }
    }

    PackFixedScaleBlock(scales, halfScales, factor, valid);
}

__aicore__ inline void QuantRowsVF(__ubuf__ float *x, __ubuf__ float *mean, __ubuf__ float *scales,
                                   __ubuf__ int8_t *out, __ubuf__ half *halfScales, float factor,
                                   __ubuf__ float *inverseScales, uint32_t dp,
                                   uint32_t smooth, uint32_t valid, uint32_t group) {
    if (dp == 128) {
        if (smooth) {
            QuantRowsImpl<128, true>(x, mean, scales, out, halfScales, factor, inverseScales, valid, group);
        } else {
            QuantRowsImpl<128, false>(x, mean, scales, out, halfScales, factor, inverseScales, valid, group);
        }
    } else {
        if (smooth) {
            QuantRowsImpl<64, true>(x, mean, scales, out, halfScales, factor, inverseScales, valid, group);
        } else {
            QuantRowsImpl<64, false>(x, mean, scales, out, halfScales, factor, inverseScales, valid, group);
        }
    }
}

// Global tensor bundle for FastQuant operations
template <class T>
struct QuantGlobalTensors {
    AscendC::GlobalTensor<T> qg;
    AscendC::GlobalTensor<T> kg;
    AscendC::GlobalTensor<T> vg;
    AscendC::GlobalTensor<uint16_t> mxg;
    AscendC::GlobalTensor<float> mg;
    AscendC::GlobalTensor<float> vmg;
    AscendC::GlobalTensor<half> qsg;
    AscendC::GlobalTensor<half> ksg;
    AscendC::GlobalTensor<int8_t> qo;
    AscendC::GlobalTensor<int8_t> ko;
    AscendC::GlobalTensor<float8_e4m3_t> vo;

    __aicore__ inline void BindBatchAddresses(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, GM_ADDR mxScale,
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks)
    {
        qg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q));
        kg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k));
        vg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(v));
        qg.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        vg.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // K is read twice (ProcessStats mean pass, then Quantize) — keep its
        // lines as L2 last-victim so the second pass hits L2 instead of GM.
        kg.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_LAST);

        mxg.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(mxScale));
        mg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mean));
        vmg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(vMean));
        qsg.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(qs));
        ksg.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ks));
        qo.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(q8));
        ko.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(k8));
        vo.SetGlobalBuffer(reinterpret_cast<__gm__ float8_e4m3_t *>(v8));
    }
};

// Double-buffered Local Buffer Allocation for Quantization Pipeline
template <class T>
struct DoubleBufferAllocation {
    AscendC::LocalTensor<T> inBuf[2];
    AscendC::LocalTensor<T> vInBuf[2];
    AscendC::LocalTensor<float> xBuf[2];
    AscendC::LocalTensor<float> muBuf[2];
    AscendC::LocalTensor<float> vMeanBuf[2];
    AscendC::LocalTensor<float> vsBuf[2];
    AscendC::LocalTensor<int8_t> outInt8[2];
    AscendC::LocalTensor<float8_e4m3_t> outFp8[2];
    AscendC::LocalTensor<float> scaleBuf[2];
    AscendC::LocalTensor<half> outHalfScale[2];
    AscendC::LocalTensor<float> invBuf[2];

    __aicore__ inline void AllocateBuffers(memory::FixedResource &r) {
        constexpr uint32_t CHUNK_BYTES = 96 * 1024;
        constexpr uint32_t V_IN_OFF = 16384, X_F32_OFF = 32768;
        constexpr uint32_t OUT_INT8_OFF = 65536, OUT_FP8_OFF = 73728;
        constexpr uint32_t MU_OFF = 81920, VS_OFF = 82432, SCALE_OFF = 82944, HALF_OFF = 83200;
        constexpr uint32_t INV_OFF = 83456, VMEAN_OFF = 83712;

        for (uint32_t slot = 0; slot < 2; ++slot) {
            const uint32_t base = slot * CHUNK_BYTES;
            inBuf[slot] = r.ubBuf.template GetBufferByByte<T>(base);
            vInBuf[slot] = r.ubBuf.template GetBufferByByte<T>(base + V_IN_OFF);
            xBuf[slot] = r.ubBuf.template GetBufferByByte<float>(base + X_F32_OFF);
            muBuf[slot] = r.ubBuf.template GetBufferByByte<float>(base + MU_OFF);
            vMeanBuf[slot] = r.ubBuf.template GetBufferByByte<float>(base + VMEAN_OFF);
            vsBuf[slot] = r.ubBuf.template GetBufferByByte<float>(base + VS_OFF);
            outInt8[slot] = r.ubBuf.template GetBufferByByte<int8_t>(base + OUT_INT8_OFF);
            outFp8[slot] = r.ubBuf.template GetBufferByByte<float8_e4m3_t>(base + OUT_FP8_OFF);
            scaleBuf[slot] = r.ubBuf.template GetBufferByByte<float>(base + SCALE_OFF);
            outHalfScale[slot] = r.ubBuf.template GetBufferByByte<half>(base + HALF_OFF);
            invBuf[slot] = r.ubBuf.template GetBufferByByte<float>(base + INV_OFF);
        }
    }
};

struct BatchQuantSchedule {
    uint64_t totalSteps = 0;
    uint64_t activeQ = 0;
    uint64_t activeK = 0;
    uint32_t coreId = 0;
    uint32_t totalCores = 0;

    __aicore__ inline void Init(const SageParams &p, bool wideMode) {
        totalCores = 2U * p.cores;
        coreId = AscendC::GetBlockIdx() % totalCores;
        const uint64_t numQ = (static_cast<uint64_t>(p.b) * p.hq * p.qp) / 64ULL;
        const uint64_t numK = (static_cast<uint64_t>(p.b) * p.hk * p.kp) / 64ULL;
        activeQ = wideMode ? 0ULL : numQ;
        activeK = numK;
        const uint64_t totalTiles = activeQ + activeK;
        totalSteps = (coreId < totalTiles) ? ((totalTiles - coreId + totalCores - 1ULL) / totalCores) : 0ULL;
    }
};

struct QuantTileTask {
    uint64_t linearIdx = 0;
    uint32_t batchIdx = 0;
    uint32_t headIdx = 0;
    uint32_t tokenOffset = 0;
    uint32_t validRows = 0;
    bool isQuery = false;
};

__aicore__ inline QuantTileTask ResolveQuantTileTask(uint64_t stepIdx, const BatchQuantSchedule &sched, const SageParams &p) {
    const uint64_t taskIdx = sched.coreId + stepIdx * sched.totalCores;
    const bool isKeyVal = taskIdx < sched.activeK;
    const bool isQuery = !isKeyVal;
    const uint64_t tileId = isKeyVal ? taskIdx : (taskIdx - sched.activeK);
    const uint64_t linIdx = tileId * 64ULL;
    const uint32_t seqStride = isQuery ? p.qp : p.kp;
    const uint32_t numHeads = isQuery ? p.hq : p.hk;
    const uint32_t seqBound = isQuery ? p.sq : p.sk;
    const uint32_t tokPos = linIdx % seqStride;
    const uint32_t hIdx = (linIdx / seqStride) % numHeads;
    const uint32_t bIdx = static_cast<uint32_t>(linIdx / seqStride / numHeads);
    // The sequence stride is padded to TILE_M, so a 64-row tile can straddle the real
    // sequence end (sq/sk not a multiple of TILE_ROWS). Clamping to the live row count keeps
    // the GM fetch inside this head and lets the tail zero-fill below cover the rest.
    const uint32_t validRows = (tokPos < seqBound) ? Min(QUANT_TILE_ROWS, seqBound - tokPos) : 0U;
    return QuantTileTask{linIdx, bIdx, hIdx, tokPos, validRows, isQuery};
}

// Pipelined Fast Quantization and Key/Value Preprocessing Engine
template <class T, bool WIDE>
class FastQuant {
public:
    __aicore__ static inline void ProcessStats(GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean,
                                               const SageParams &p) {
        if (p.smoothK) {
            Stats<false>(k, mean, p);
        }
        if (p.smoothV) {
            Stats<true>(v, vMean, p);
        }
    }

    template <bool IS_V>
    __aicore__ static void Stats(GM_ADDR src, GM_ADDR mean, const SageParams &p) {
        const uint64_t *const strides = IS_V ? p.vs : p.ks;
        const bool centerActive = IS_V ? bool(p.smoothV) : bool(p.smoothK);
        if (!centerActive) return;

        const uint32_t workers = 2 * p.cores;
        const uint64_t totalHeads = uint64_t(p.b) * p.hk;
        memory::FixedResource r;
        AscendC::GlobalTensor<T> inGm;
        inGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src));
        AscendC::GlobalTensor<float> meanGm;
        meanGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mean));

        constexpr uint32_t BATCH_TOKENS = 128, MAX_DP = 128;
        constexpr uint32_t SLOT_BYTES = BATCH_TOKENS * MAX_DP * 2U;
        constexpr uint32_t F32_BYTES = BATCH_TOKENS * MAX_DP * 4U;
        constexpr uint32_t RAW_OFFSET = 0, F32_OFFSET = 2 * SLOT_BYTES;
        constexpr uint32_t SUM_OFFSET = F32_OFFSET + F32_BYTES;

        auto rawBuf = r.ubBuf.template GetBufferByByte<T>(RAW_OFFSET);
        auto f32Buf = r.ubBuf.template GetBufferByByte<float>(F32_OFFSET);
        auto sumBuf = r.ubBuf.template GetBufferByByte<float>(SUM_OFFSET);
        const uint32_t coreRank = AscendC::GetBlockIdx() % workers;
        const float invLength = 1.0f / static_cast<float>(p.sk);

        for (uint64_t headIdx = coreRank; headIdx < totalHeads; headIdx += workers) {
            const uint32_t batchNum = headIdx / p.hk, headNum = headIdx % p.hk;
            AscendC::Duplicate(sumBuf, 0.0f, p.dp);
            if (p.d == p.dp) {
                AscendC::LocalTensor<T> pingPong[2] = {
                    r.ubBuf.template GetBufferByByte<T>(0),
                    r.ubBuf.template GetBufferByByte<T>(SLOT_BYTES)
                };
                const uint32_t batchLimit = (p.sk + BATCH_TOKENS - 1U) / BATCH_TOKENS;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
                for (uint32_t iter = 0; iter <= batchLimit; ++iter) {
                    if (iter < batchLimit) {
                        const uint32_t currSlot = iter & 1U;
                        const uint32_t fromRow = iter * BATCH_TOKENS;
                        const uint32_t rowSpan = Min(BATCH_TOKENS, p.sk - fromRow);
                        const event_t inEv = static_cast<event_t>(1U + currSlot);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inEv);
                        AscendC::DataCopy(pingPong[currSlot], inGm[Offset(strides, batchNum, headNum, fromRow)], rowSpan * p.dp);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inEv);
                    }
                    if (iter > 0) {
                        const uint32_t doneSlot = (iter - 1U) & 1U;
                        const uint32_t doneRows = Min(BATCH_TOKENS, p.sk - (iter - 1U) * BATCH_TOKENS);
                        const event_t doneEv = static_cast<event_t>(1U + doneSlot);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(doneEv);
                        StatsSumVF(pingPong[doneSlot], sumBuf, doneRows, p.dp);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(doneEv);
                    }
                }
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
            } else {
                for (uint32_t row = 0; row < p.sk; ++row) {
                    ReadRow(rawBuf, inGm, Offset(strides, batchNum, headNum, row), p.d, p.dp);
                    AscendC::Cast(f32Buf, rawBuf, AscendC::RoundMode::CAST_NONE, p.dp);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(sumBuf, sumBuf, f32Buf, p.dp);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(sumBuf, sumBuf, invLength, p.dp);
            WriteRow(meanGm, headIdx * p.dp, sumBuf, p.dp);
        }
    }

    __aicore__ static void Quantize(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, GM_ADDR mxScale,
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, const SageParams &p)
    {
        if (p.d == p.dp) {
            ExecuteBatchQuant(q, k, v, mean, vMean, mxScale, q8, k8, v8, qs, ks, p);
        } else {
            ExecuteFallbackQuant(q, k, v, mean, vMean, mxScale, q8, k8, v8, qs, ks, p);
        }
    }

private:
    __aicore__ static void ExecuteBatchQuant(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, GM_ADDR mxScale,
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, const SageParams &p)
    {
        memory::FixedResource r;
        QuantGlobalTensors<T> gt;
        gt.BindBatchAddresses(q, k, v, mean, vMean, mxScale, q8, k8, v8, qs, ks);

        DoubleBufferAllocation<T> dba;
        dba.AllocateBuffers(r);

        BatchQuantSchedule sched;
        sched.Init(p, WIDE);
        if (sched.totalSteps == 0) return;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);

        QuantTileTask pipeSlot[2];

        for (uint64_t step = 0; step <= sched.totalSteps; ++step) {
            if (step < sched.totalSteps) {
                const uint32_t s = step & 1;
                pipeSlot[s] = ResolveQuantTileTask(step, sched, p);
                const QuantTileTask &cur = pipeSlot[s];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(1 + s));
                if (cur.validRows) {
                    AscendC::DataCopy(dba.inBuf[s],
                        (cur.isQuery ? gt.qg : gt.kg)[Offset(cur.isQuery ? p.qs : p.ks, cur.batchIdx, cur.headIdx, cur.tokenOffset)],
                        cur.validRows * p.dp);
                }
                if (!cur.isQuery) {
                    if (cur.validRows) {
                        AscendC::DataCopy(dba.vInBuf[s], gt.vg[Offset(p.vs, cur.batchIdx, cur.headIdx, cur.tokenOffset)], cur.validRows * p.dp);
                    }
                    AscendC::DataCopy(dba.muBuf[s], gt.mg[(uint64_t(cur.batchIdx) * p.hk + cur.headIdx) * p.dp], p.dp);
                    if (p.smoothV) {
                        AscendC::DataCopy(dba.vMeanBuf[s], gt.vmg[(uint64_t(cur.batchIdx) * p.hk + cur.headIdx) * p.dp], p.dp);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(1 + s));
            }
            if (step == 0) continue;
            const uint32_t prevS = (step - 1) & 1;
            const event_t ev = static_cast<event_t>(1 + prevS);
            const QuantTileTask &desc = pipeSlot[prevS];

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev);
            if (step > 2) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ev);
            }

            if (desc.validRows < 64) {
                AscendC::Duplicate(dba.inBuf[prevS][desc.validRows * p.dp], static_cast<T>(0), (64 - desc.validRows) * p.dp);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(dba.xBuf[prevS], dba.inBuf[prevS], AscendC::RoundMode::CAST_NONE, 64 * p.dp);
            AscendC::PipeBarrier<PIPE_V>();

            QuantRowsVF(reinterpret_cast<__ubuf__ float *>(dba.xBuf[prevS].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(dba.muBuf[prevS].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(dba.scaleBuf[prevS].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ int8_t *>(dba.outInt8[prevS].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ half *>(dba.outHalfScale[prevS].GetPhyAddr()),
                        desc.isQuery ? p.qFactor : p.kFactor,
                        reinterpret_cast<__ubuf__ float *>(dba.invBuf[prevS].GetPhyAddr()),
                        p.dp, !desc.isQuery && p.smoothK && desc.validRows,
                        desc.validRows, desc.isQuery ? p.qg : p.kg);

            if (!desc.isQuery) {
                AscendC::PipeBarrier<PIPE_V>();
                if (desc.validRows < 64) {
                    AscendC::Duplicate(dba.vInBuf[prevS][desc.validRows * p.dp], static_cast<T>(0), (64 - desc.validRows) * p.dp);
                }
                AscendC::PipeBarrier<PIPE_V>();
                if (p.smoothV && desc.validRows) {
                    CenterVRowsVF<T>(reinterpret_cast<__ubuf__ T *>(dba.vInBuf[prevS].GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float *>(dba.vMeanBuf[prevS].GetPhyAddr()),
                                     p.dp, desc.validRows);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                QuantVRowsMxVF<T, WIDE>(reinterpret_cast<__ubuf__ T *>(dba.vInBuf[prevS].GetPhyAddr()),
                                        reinterpret_cast<__ubuf__ T *>(dba.xBuf[prevS].GetPhyAddr()),
                                        reinterpret_cast<__ubuf__ uint16_t *>(dba.vsBuf[prevS].GetPhyAddr()),
                                        reinterpret_cast<__ubuf__ float8_e4m3_t *>(dba.outFp8[prevS].GetPhyAddr()), p.dp);
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ev);
            memory::Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy((desc.isQuery ? gt.qo : gt.ko)[desc.linearIdx * p.dp], dba.outInt8[prevS], 64 * p.dp);
            AscendC::DataCopy((desc.isQuery ? gt.qsg : gt.ksg)[desc.linearIdx], dba.outHalfScale[prevS], 64);
            if (!desc.isQuery) {
                AscendC::DataCopy(gt.vo[desc.linearIdx * p.dp], dba.outFp8[prevS], 64 * p.dp);
                AscendC::DataCopy(gt.mxg[desc.linearIdx / 64 * p.dp], dba.vsBuf[prevS].template ReinterpretCast<uint16_t>(), p.dp);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ev);
        }

        for (uint32_t s = 0; s < Min(uint64_t(2), sched.totalSteps); ++s) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<event_t>(1 + s));
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
    }

    __aicore__ static void ExecuteFallbackQuant(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, GM_ADDR mxScale,
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, const SageParams &p)
    {
        memory::FixedResource r;
        QuantGlobalTensors<T> gt;
        gt.BindBatchAddresses(q, k, v, mean, vMean, mxScale, q8, k8, v8, qs, ks);

        ExecuteFallbackQKTiles(r, gt, p);
        ExecuteFallbackVTiles(r, gt, p);
    }

    __aicore__ static void ExecuteFallbackQKTiles(
        memory::FixedResource &r, QuantGlobalTensors<T> &gt, const SageParams &p)
    {
        constexpr uint32_t X_OFF = 16384, MU_OFF = 49152, SCALE_OFF = 49664, INV_OFF = 49920;
        constexpr uint32_t HALF_OFF = 50176, OUT_OFF = 50304;

        auto inRaw = r.ubBuf.template GetBufferByByte<T>(0);
        auto inF32 = r.ubBuf.template GetBufferByByte<float>(X_OFF);
        auto muBuf = r.ubBuf.template GetBufferByByte<float>(MU_OFF);
        auto scaleBuf = r.ubBuf.template GetBufferByByte<float>(SCALE_OFF);
        auto invBuf = r.ubBuf.template GetBufferByByte<float>(INV_OFF);
        auto halfScaleBuf = r.ubBuf.template GetBufferByByte<half>(HALF_OFF);
        auto outBuf = r.ubBuf.template GetBufferByByte<int8_t>(OUT_OFF);

        const uint64_t nq = WIDE ? 0 : (uint64_t(p.b) * p.hq * p.qp);
        const uint64_t nk = uint64_t(p.b) * p.hk * p.kp;

        for (uint64_t tile = AscendC::GetBlockIdx(); tile < (nq + nk) / 64; tile += 2 * p.cores) {
            const bool isKv = tile < nk / 64;
            const bool isQ = !isKv;
            const uint64_t linIdx = (isKv ? tile : tile - nk / 64) * 64;
            const uint32_t seqLen = isQ ? p.qp : p.kp, numH = isQ ? p.hq : p.hk;
            const uint32_t tokIdx = linIdx % seqLen, h = (linIdx / seqLen) % numH, b = linIdx / seqLen / numH;
            const uint32_t seqBound = isQ ? p.sq : p.sk;
            const uint32_t valid = tokIdx < seqBound ? Min(uint32_t(64), seqBound - tokIdx) : 0;

            for (uint32_t row = 0; row < valid; ++row) {
                ReadRow(inRaw[row * p.dp], isQ ? gt.qg : gt.kg, Offset(isQ ? p.qs : p.ks, b, h, tokIdx + row), p.d, p.dp);
            }
            if (valid < 64) {
                AscendC::Duplicate(inRaw[valid * p.dp], static_cast<T>(0), (64 - valid) * p.dp);
            }
            AscendC::Cast(inF32, inRaw, AscendC::RoundMode::CAST_NONE, 64 * p.dp);
            AscendC::PipeBarrier<PIPE_V>();

            if (!isQ && p.smoothK && valid) {
                AscendC::DataCopy(muBuf, gt.mg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                memory::Sync<AscendC::HardEvent::MTE2_V>();
            }
            QuantRowsVF(reinterpret_cast<__ubuf__ float *>(inF32.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(muBuf.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(scaleBuf.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ int8_t *>(outBuf.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ half *>(halfScaleBuf.GetPhyAddr()),
                        isQ ? p.qFactor : p.kFactor,
                        reinterpret_cast<__ubuf__ float *>(invBuf.GetPhyAddr()),
                        p.dp, !isQ && p.smoothK && valid, valid,
                        isQ ? p.qg : p.kg);
            WriteRow(isQ ? gt.qo : gt.ko, linIdx * p.dp, outBuf, 64 * p.dp);
            WriteRow(isQ ? gt.qsg : gt.ksg, linIdx, halfScaleBuf, 64);
        }
    }

    __aicore__ static void ExecuteFallbackVTiles(
        memory::FixedResource &r, QuantGlobalTensors<T> &gt, const SageParams &p)
    {
        constexpr uint32_t VIN_OFF = 65536, VMAX_OFF = 81920, VOUT_OFF = 82944, MX_OFF = 91136;
        constexpr uint32_t VMEAN_OFF = 92160;

        AscendC::PipeBarrier<PIPE_ALL>();
        auto vInRaw = r.ubBuf.template GetBufferByByte<T>(VIN_OFF);
        auto vPeak = r.ubBuf.template GetBufferByByte<T>(VMAX_OFF);
        auto vOutBuf = r.ubBuf.template GetBufferByByte<float8_e4m3_t>(VOUT_OFF);
        auto vScale16 = r.ubBuf.template GetBufferByByte<uint16_t>(MX_OFF);
        auto vMeanF32 = r.ubBuf.template GetBufferByByte<float>(VMEAN_OFF);

        const uint64_t nk = uint64_t(p.b) * p.hk * p.kp;

        for (uint64_t tile = AscendC::GetBlockIdx(); tile < nk / 64; tile += 2 * p.cores) {
            const uint64_t linIdx = tile * 64;
            const uint32_t tokIdx = linIdx % p.kp, h = (linIdx / p.kp) % p.hk, b = linIdx / p.kp / p.hk;
            const uint32_t valid = tokIdx < p.sk ? Min(uint32_t(64), p.sk - tokIdx) : 0;
            if (p.d == p.dp && valid) {
                memory::Sync<AscendC::HardEvent::V_MTE2>();
                AscendC::DataCopy(vInRaw, gt.vg[Offset(p.vs, b, h, tokIdx)], valid * p.dp);
                memory::Sync<AscendC::HardEvent::MTE2_V>();
            } else {
                for (uint32_t row = 0; row < valid; ++row) {
                    ReadRow(vInRaw[row * p.dp], gt.vg, Offset(p.vs, b, h, tokIdx + row), p.d, p.dp);
                }
            }
            if (valid < 64) {
                AscendC::Duplicate(vInRaw[valid * p.dp], static_cast<T>(0), (64 - valid) * p.dp);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (p.smoothV && valid) {
                AscendC::DataCopy(vMeanF32, gt.vmg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                memory::Sync<AscendC::HardEvent::MTE2_V>();
                CenterVRowsVF<T>(reinterpret_cast<__ubuf__ T *>(vInRaw.GetPhyAddr()),
                                 reinterpret_cast<__ubuf__ float *>(vMeanF32.GetPhyAddr()), p.dp, valid);
                AscendC::PipeBarrier<PIPE_V>();
            }
            QuantVRowsMxVF<T, WIDE>(reinterpret_cast<__ubuf__ T *>(vInRaw.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ T *>(vPeak.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ uint16_t *>(vScale16.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ float8_e4m3_t *>(vOutBuf.GetPhyAddr()), p.dp);
            WriteRow(gt.vo, linIdx * p.dp, vOutBuf, 64 * p.dp);
            WriteRow(gt.mxg, linIdx / 64 * p.dp, vScale16, p.dp);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

#endif // __CCE__

} // namespace sage::microkernel
