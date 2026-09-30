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
using AscendC::RoundMode;
using AscendC::CMPMODE;
using memory::Sync;
using AscendC::HardEvent;

// Channel-wise Column Sum Reduction for Sequence Statistics
template <class T, uint32_t DP>
__simd_vf__ inline void StatsSumImpl(__ubuf__ T *rows, __ubuf__ float *state, uint32_t count) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto maskF32 = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait unpackToF32{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint32_t c = 0; c < DP; c += 64) {
        RegTensor<float> totalAcc;
        LoadAlign(totalAcc, state + c);
        for (uint32_t r = 0; r < count; ++r) {
            if constexpr (AscendC::IsSameType<T, float>::value) {
                RegTensor<float> elem;
                LoadAlign(elem, rows + r * DP + c);
                Add(totalAcc, totalAcc, elem, maskF32);
            } else {
                RegTensor<T> raw;
                RegTensor<float> fVal;
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, rows + r * DP + c);
                Cast<float, T, unpackToF32>(fVal, raw, maskF32);
                Add(totalAcc, totalAcc, fVal, maskF32);
            }
        }
        StoreAlign(state + c, totalAcc, maskF32);
    }
}

template <class T>
__aicore__ inline void StatsSumVF(AscendC::LocalTensor<T> rows, AscendC::LocalTensor<float> state,
                                  uint32_t count, uint32_t dp) {
    auto inputPtr = reinterpret_cast<__ubuf__ T *>(rows.GetPhyAddr());
    auto sumPtr = reinterpret_cast<__ubuf__ float *>(state.GetPhyAddr());
    if (dp == 128)
        StatsSumImpl<T, 128>(inputPtr, sumPtr, count);
    else
        StatsSumImpl<T, 64>(inputPtr, sumPtr, count);
}

// Subtracts channel means from rows in Unified Buffer
template <class T, uint32_t DP>
__simd_vf__ inline void CenterVImpl(__ubuf__ T *rows, __ubuf__ float *mean, uint32_t count) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto maskF32 = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait castToFloat{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr CastTrait castToTarget{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    for (uint32_t c = 0; c < DP; c += 64) {
        RegTensor<float> meanCol;
        LoadAlign(meanCol, mean + c);
        for (uint32_t r = 0; r < count; ++r) {
            RegTensor<T> rawIn;
            RegTensor<float> valF32;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(rawIn, rows + r * DP + c);
            Cast<float, T, castToFloat>(valF32, rawIn, maskF32);
            Sub(valF32, valF32, meanCol, maskF32);
            Cast<T, float, castToTarget>(rawIn, valF32, maskF32);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(rows + r * DP + c, rawIn, maskF32);
        }
    }
}

template <class T>
__aicore__ inline void CenterVRowsVF(__ubuf__ T *rows, __ubuf__ float *mean, uint32_t dp, uint32_t count) {
    if (dp == 128)
        CenterVImpl<T, 128>(rows, mean, count);
    else
        CenterVImpl<T, 64>(rows, mean, count);
}

// Harmonizes group-shared INT8 scale across token groups (4 for per_thread, 32 for per_warp)
__simd_callee__ inline void BroadcastGroupMaxima(__ubuf__ float *maxima, uint32_t group) {
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> curPeak, neighbor;
    LoadAlign(curPeak, maxima);

    // Tree reduction across powers of two within the group
    for (uint32_t span = 1; span < group; span <<= 1) {
        StoreAlign(maxima, curPeak, full);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(neighbor, maxima + span);
        Max(curPeak, curPeak, neighbor, full);
    }
    StoreAlign(maxima, curPeak, full);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // Broadcast the group maximum to all members in the group
    auto grpMask = UpdateMask<float>(group);
    for (uint32_t offset = 0; offset < 64; offset += group) {
        RegTensor<float> leadVal;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(leadVal, maxima + offset);
        StoreAlign(maxima + offset, leadVal, grpMask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

// Converts float scales to half precision with precomputed balancing factor
__simd_callee__ inline void PackFixedScaleBlock(__ubuf__ float *scales, __ubuf__ half *halfScales,
                                                float factor, uint32_t valid) {
    using namespace AscendC::MicroAPI;
    auto mask = UpdateMask<float>(valid);
    RegTensor<float> s0, s1, factorReg;
    RegTensor<half> h0, h1;
    Duplicate(factorReg, factor);
    LoadAlign(s0, scales);
    LoadAlign(s1, scales + 64);
    Mul(s0, s0, factorReg, mask);
    Mul(s1, s1, factorReg, mask);
    static constexpr CastTrait castToHalf{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    Cast<half, float, castToHalf>(h0, s0, mask);
    Cast<half, float, castToHalf>(h1, s1, mask);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(halfScales, h0, mask);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(halfScales + 64, h1, mask);
}

// High-performance INT8 Symmetric Quantization Implementation
template <uint32_t DP, bool SMOOTH>
__simd_vf__ inline void QuantRowsImpl(__ubuf__ float *x, [[maybe_unused]] __ubuf__ float *mean,
                                      __ubuf__ float *scales, __ubuf__ int8_t *out,
                                      __ubuf__ half *halfScales, float factor,
                                      __ubuf__ float *inverseScales, uint32_t valid, uint32_t group) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto full = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> mean0, mean1;
    if constexpr (SMOOTH) {
        LoadAlign(mean0, mean);
        if constexpr (DP == 128) LoadAlign(mean1, mean + 64);
    }

    auto dstScale = scales;
    UnalignReg packState;

    // Pass 1: Mean centering (if smooth) and per-row peak finding
    if constexpr (DP == 128) {
        for (uint16_t r = 0; r < 64; r += 2) {
            RegTensor<float> v0_lo, v1_lo, v0_hi, v1_hi;
            LoadAlign(v0_lo, x + r * DP);
            LoadAlign(v1_lo, x + (r + 1) * DP);
            LoadAlign(v0_hi, x + r * DP + 64);
            LoadAlign(v1_hi, x + (r + 1) * DP + 64);

            if constexpr (SMOOTH) {
                Sub(v0_lo, v0_lo, mean0, full);
                Sub(v1_lo, v1_lo, mean0, full);
                Sub(v0_hi, v0_hi, mean1, full);
                Sub(v1_hi, v1_hi, mean1, full);
                StoreAlign(x + r * DP, v0_lo, full);
                StoreAlign(x + (r + 1) * DP, v1_lo, full);
                StoreAlign(x + r * DP + 64, v0_hi, full);
                StoreAlign(x + (r + 1) * DP + 64, v1_hi, full);
            }

            RegTensor<float> abs0_lo, abs1_lo, abs0_hi, abs1_hi;
            Abs(abs0_lo, v0_lo, full);
            Abs(abs1_lo, v1_lo, full);
            Abs(abs0_hi, v0_hi, full);
            Abs(abs1_hi, v1_hi, full);

            RegTensor<float> peak0, peak1;
            Max(peak0, abs0_lo, abs0_hi, full);
            Max(peak1, abs1_lo, abs1_hi, full);
            ReduceDataBlock<ReduceType::MAX>(peak0, peak0, full);
            ReduceDataBlock<ReduceType::MAX>(peak1, peak1, full);

            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dstScale, peak0, packState, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dstScale, peak1, packState, 1);
        }
    } else {
        for (uint16_t r = 0; r < 64; r += 4) {
            RegTensor<float> vec[4], absVec[4];
            for (int k = 0; k < 4; ++k) {
                LoadAlign(vec[k], x + (r + k) * DP);
                if constexpr (SMOOTH) {
                    Sub(vec[k], vec[k], mean0, full);
                    StoreAlign(x + (r + k) * DP, vec[k], full);
                }
                Abs(absVec[k], vec[k], full);
                ReduceDataBlock<ReduceType::MAX>(absVec[k], absVec[k], full);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dstScale, absVec[k], packState, 1);
            }
        }
    }

    StoreUnAlignPost(dstScale, packState, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // Pass 2: Group scale alignment & FastReciprocal inversion
    BroadcastGroupMaxima(scales, group);

    RegTensor<float> peakVec, zeroFloor, defaultUnit, invDivisor;
    Duplicate(zeroFloor, 0.0f);
    Duplicate(defaultUnit, 1.0f);
    Duplicate(invDivisor, INV_INT8_MAX);

    LoadAlign(peakVec, scales);
    Mul(peakVec, peakVec, invDivisor, full);

    MaskReg isPositive;
    Compare<float, CMPMODE::GT>(isPositive, peakVec, zeroFloor, full);
    Select(peakVec, peakVec, defaultUnit, isPositive);
    StoreAlign(scales, peakVec, full);

    RegTensor<float> invScaleVec;
    FastReciprocal(invScaleVec, peakVec, full);
    StoreAlign(inverseScales, invScaleVec, full);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // Pass 3: Scale multiplication and INT8 cast with RNE saturation
    static constexpr CastTrait castToInt32{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    if constexpr (DP == 128) {
        for (uint16_t r = 0; r < 64; r += 2) {
            RegTensor<float> inv0, inv1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(inv0, inverseScales + r);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(inv1, inverseScales + r + 1);

            RegTensor<float> val0_lo, val0_hi, val1_lo, val1_hi;
            LoadAlign(val0_lo, x + r * DP);
            LoadAlign(val0_hi, x + r * DP + 64);
            LoadAlign(val1_lo, x + (r + 1) * DP);
            LoadAlign(val1_hi, x + (r + 1) * DP + 64);

            Mul(val0_lo, val0_lo, inv0, full);
            Mul(val0_hi, val0_hi, inv0, full);
            Mul(val1_lo, val1_lo, inv1, full);
            Mul(val1_hi, val1_hi, inv1, full);

            RegTensor<int32_t> i0_lo, i0_hi, i1_lo, i1_hi;
            Cast<int32_t, float, castToInt32>(i0_lo, val0_lo, full);
            Cast<int32_t, float, castToInt32>(i0_hi, val0_hi, full);
            Cast<int32_t, float, castToInt32>(i1_lo, val1_lo, full);
            Cast<int32_t, float, castToInt32>(i1_hi, val1_hi, full);

            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + r * DP, reinterpret_cast<RegTensor<int8_t> &>(i0_lo), full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + r * DP + 64, reinterpret_cast<RegTensor<int8_t> &>(i0_hi), full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (r + 1) * DP, reinterpret_cast<RegTensor<int8_t> &>(i1_lo), full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (r + 1) * DP + 64, reinterpret_cast<RegTensor<int8_t> &>(i1_hi), full);
        }
    } else {
        for (uint16_t r = 0; r < 64; r += 4) {
            for (int k = 0; k < 4; ++k) {
                RegTensor<float> invK, valK;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(invK, inverseScales + r + k);
                LoadAlign(valK, x + (r + k) * DP);
                Mul(valK, valK, invK, full);
                RegTensor<int32_t> intK;
                Cast<int32_t, float, castToInt32>(intK, valK, full);
                StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (r + k) * DP,
                    reinterpret_cast<RegTensor<int8_t> &>(intK), full);
            }
        }
    }

    PackFixedScaleBlock(scales, halfScales, factor, valid);
}

__aicore__ inline void QuantRowsVF(__ubuf__ float *x, __ubuf__ float *mean, __ubuf__ float *scales,
                                   __ubuf__ int8_t *out, __ubuf__ half *halfScales, float factor,
                                   __ubuf__ float *inverseScales, uint32_t dp,
                                   uint32_t smooth, uint32_t valid, uint32_t group) {
    if (dp == 128) {
        if (smooth)
            QuantRowsImpl<128, true>(x, mean, scales, out, halfScales, factor, inverseScales, valid, group);
        else
            QuantRowsImpl<128, false>(x, mean, scales, out, halfScales, factor, inverseScales, valid, group);
    } else {
        if (smooth)
            QuantRowsImpl<64, true>(x, mean, scales, out, halfScales, factor, inverseScales, valid, group);
        else
            QuantRowsImpl<64, false>(x, mean, scales, out, halfScales, factor, inverseScales, valid, group);
    }
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
        using namespace AscendC;
        const uint64_t *const strides = IS_V ? p.vs : p.ks;
        const bool centerActive = IS_V ? bool(p.smoothV) : bool(p.smoothK);
        if (!centerActive) return;

        const uint32_t workers = 2 * p.cores;
        const uint64_t totalHeads = uint64_t(p.b) * p.hk;
        memory::FixedResource r;
        GlobalTensor<T> inGm;
        inGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src));
        GlobalTensor<float> meanGm;
        meanGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mean));

        constexpr uint32_t BATCH_TOKENS = 128, MAX_DP = 128;
        constexpr uint32_t SLOT_BYTES = BATCH_TOKENS * MAX_DP * 2U;
        constexpr uint32_t F32_BYTES = BATCH_TOKENS * MAX_DP * 4U;
        constexpr uint32_t RAW_OFFSET = 0, F32_OFFSET = 2 * SLOT_BYTES;
        constexpr uint32_t SUM_OFFSET = F32_OFFSET + F32_BYTES;

        auto rawBuf = r.ubBuf.template GetBufferByByte<T>(RAW_OFFSET);
        auto f32Buf = r.ubBuf.template GetBufferByByte<float>(F32_OFFSET);
        auto sumBuf = r.ubBuf.template GetBufferByByte<float>(SUM_OFFSET);
        const uint32_t coreRank = GetBlockIdx() % workers;
        const float invLength = 1.0f / float(p.sk);

        for (uint64_t headIdx = coreRank; headIdx < totalHeads; headIdx += workers) {
            const uint32_t batchNum = headIdx / p.hk, headNum = headIdx % p.hk;
            Duplicate(sumBuf, 0.0f, p.dp);
            if (p.d == p.dp) {
                LocalTensor<T> pingPong[2] = {
                    r.ubBuf.template GetBufferByByte<T>(0),
                    r.ubBuf.template GetBufferByByte<T>(SLOT_BYTES)
                };
                const uint32_t numBatches = (p.sk + BATCH_TOKENS - 1) / BATCH_TOKENS;
                SetFlag<HardEvent::V_MTE2>(1);
                SetFlag<HardEvent::V_MTE2>(2);
                for (uint32_t bIdx = 0; bIdx <= numBatches; ++bIdx) {
                    if (bIdx < numBatches) {
                        const uint32_t s = bIdx & 1, startRow = bIdx * BATCH_TOKENS;
                        const uint32_t rows = Min(BATCH_TOKENS, p.sk - startRow);
                        const event_t ev = static_cast<event_t>(1 + s);
                        WaitFlag<HardEvent::V_MTE2>(ev);
                        DataCopy(pingPong[s], inGm[Offset(strides, batchNum, headNum, startRow)], rows * p.dp);
                        SetFlag<HardEvent::MTE2_V>(ev);
                    }
                    if (bIdx == 0) continue;
                    const uint32_t prevS = (bIdx - 1) & 1;
                    const uint32_t prevRows = Min(BATCH_TOKENS, p.sk - (bIdx - 1) * BATCH_TOKENS);
                    const event_t prevEv = static_cast<event_t>(1 + prevS);
                    WaitFlag<HardEvent::MTE2_V>(prevEv);
                    StatsSumVF(pingPong[prevS], sumBuf, prevRows, p.dp);
                    SetFlag<HardEvent::V_MTE2>(prevEv);
                }
                WaitFlag<HardEvent::V_MTE2>(1);
                WaitFlag<HardEvent::V_MTE2>(2);
            } else {
                for (uint32_t row = 0; row < p.sk; ++row) {
                    ReadRow(rawBuf, inGm, Offset(strides, batchNum, headNum, row), p.d, p.dp);
                    Cast(f32Buf, rawBuf, RoundMode::CAST_NONE, p.dp);
                    PipeBarrier<PIPE_V>();
                    Add(sumBuf, sumBuf, f32Buf, p.dp);
                }
            }
            PipeBarrier<PIPE_V>();
            Muls(sumBuf, sumBuf, invLength, p.dp);
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
        using namespace AscendC;
        memory::FixedResource r;
        GlobalTensor<T> qg, kg, vg;
        qg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q));
        kg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k));
        vg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(v));
        qg.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        vg.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        GlobalTensor<uint16_t> mxg;
        mxg.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(mxScale));
        GlobalTensor<float> mg, vmg;
        mg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mean));
        vmg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(vMean));
        GlobalTensor<half> qsg, ksg;
        qsg.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(qs));
        ksg.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ks));
        GlobalTensor<int8_t> qo, ko;
        qo.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(q8));
        ko.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(k8));
        GlobalTensor<float8_e4m3_t> vo;
        vo.SetGlobalBuffer(reinterpret_cast<__gm__ float8_e4m3_t *>(v8));

        constexpr uint32_t CHUNK_BYTES = 96 * 1024;
        constexpr uint32_t V_IN_OFF = 16384, X_F32_OFF = 32768;
        constexpr uint32_t OUT_INT8_OFF = 65536, OUT_FP8_OFF = 73728;
        constexpr uint32_t MU_OFF = 81920, VS_OFF = 82432, SCALE_OFF = 82944, HALF_OFF = 83200;
        constexpr uint32_t INV_OFF = 83456, VMEAN_OFF = 83712;

        LocalTensor<T> inBuf[2], vInBuf[2];
        LocalTensor<float> xBuf[2], muBuf[2], vMeanBuf[2], vsBuf[2], scaleBuf[2], invBuf[2];
        LocalTensor<int8_t> outInt8[2];
        LocalTensor<float8_e4m3_t> outFp8[2];
        LocalTensor<half> outHalfScale[2];

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

        const uint64_t numQ = uint64_t(p.b) * p.hq * p.qp / 64;
        const uint64_t numK = uint64_t(p.b) * p.hk * p.kp / 64;
        const uint32_t totalCores = 2 * p.cores;
        const uint32_t coreId = GetBlockIdx() % totalCores;
        const uint64_t activeQ = WIDE ? 0 : numQ, activeK = numK;
        if (coreId >= activeQ + activeK) return;
        const uint64_t totalSteps = (activeQ + activeK - coreId + totalCores - 1) / totalCores;

        SetFlag<HardEvent::V_MTE2>(1);
        SetFlag<HardEvent::V_MTE2>(2);

        struct PipelineDesc {
            uint64_t linearIdx;
            bool isQuery;
            uint32_t validRows;
        } pipeSlot[2];

        for (uint64_t step = 0; step <= totalSteps; ++step) {
            if (step < totalSteps) {
                const uint32_t s = step & 1;
                const uint64_t taskIdx = coreId + step * totalCores;
                const bool isKeyVal = taskIdx < activeK;
                const bool isQuery = !isKeyVal;
                const uint64_t linearIdx = (isKeyVal ? taskIdx : (taskIdx - activeK)) * 64;
                const uint32_t seqStride = isQuery ? p.qp : p.kp;
                const uint32_t numHeads = isQuery ? p.hq : p.hk;
                const uint32_t tokPos = linearIdx % seqStride;
                const uint32_t hIdx = (linearIdx / seqStride) % numHeads;
                const uint32_t bIdx = linearIdx / seqStride / numHeads;
                const uint32_t validRows = tokPos < (isQuery ? p.sq : p.sk) ? 64 : 0;
                pipeSlot[s] = {linearIdx, isQuery, validRows};

                WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(1 + s));
                if (validRows)
                    DataCopy(inBuf[s], (isQuery ? qg : kg)[Offset(isQuery ? p.qs : p.ks, bIdx, hIdx, tokPos)], 64 * p.dp);
                if (!isQuery) {
                    if (validRows)
                        DataCopy(vInBuf[s], vg[Offset(p.vs, bIdx, hIdx, tokPos)], 64 * p.dp);
                    DataCopy(muBuf[s], mg[(uint64_t(bIdx) * p.hk + hIdx) * p.dp], p.dp);
                    if (p.smoothV)
                        DataCopy(vMeanBuf[s], vmg[(uint64_t(bIdx) * p.hk + hIdx) * p.dp], p.dp);
                }
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(1 + s));
            }
            if (step == 0) continue;
            const uint32_t prevS = (step - 1) & 1;
            const event_t ev = static_cast<event_t>(1 + prevS);
            const PipelineDesc desc = pipeSlot[prevS];

            WaitFlag<HardEvent::MTE2_V>(ev);
            if (step > 2)
                WaitFlag<HardEvent::MTE3_V>(ev);

            if (desc.validRows < 64) {
                Duplicate(inBuf[prevS][desc.validRows * p.dp], T(0), (64 - desc.validRows) * p.dp);
                PipeBarrier<PIPE_V>();
            }
            Cast(xBuf[prevS], inBuf[prevS], RoundMode::CAST_NONE, 64 * p.dp);
            PipeBarrier<PIPE_V>();

            QuantRowsVF(reinterpret_cast<__ubuf__ float *>(xBuf[prevS].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(muBuf[prevS].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(scaleBuf[prevS].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ int8_t *>(outInt8[prevS].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ half *>(outHalfScale[prevS].GetPhyAddr()),
                        desc.isQuery ? p.qFactor : p.kFactor,
                        reinterpret_cast<__ubuf__ float *>(invBuf[prevS].GetPhyAddr()),
                        p.dp, !desc.isQuery && p.smoothK && desc.validRows,
                        desc.validRows, desc.isQuery ? p.qg : p.kg);

            if (!desc.isQuery) {
                PipeBarrier<PIPE_V>();
                if (desc.validRows < 64)
                    Duplicate(vInBuf[prevS][desc.validRows * p.dp], T(0), (64 - desc.validRows) * p.dp);
                PipeBarrier<PIPE_V>();
                if (p.smoothV && desc.validRows) {
                    CenterVRowsVF<T>(reinterpret_cast<__ubuf__ T *>(vInBuf[prevS].GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float *>(vMeanBuf[prevS].GetPhyAddr()),
                                     p.dp, desc.validRows);
                    PipeBarrier<PIPE_V>();
                }
                QuantVRowsMxVF<T, WIDE>(reinterpret_cast<__ubuf__ T *>(vInBuf[prevS].GetPhyAddr()),
                                        reinterpret_cast<__ubuf__ T *>(xBuf[prevS].GetPhyAddr()),
                                        reinterpret_cast<__ubuf__ uint16_t *>(vsBuf[prevS].GetPhyAddr()),
                                        reinterpret_cast<__ubuf__ float8_e4m3_t *>(outFp8[prevS].GetPhyAddr()), p.dp);
            }

            SetFlag<HardEvent::V_MTE2>(ev);
            Sync<HardEvent::V_MTE3>();
            DataCopy((desc.isQuery ? qo : ko)[desc.linearIdx * p.dp], outInt8[prevS], 64 * p.dp);
            DataCopy((desc.isQuery ? qsg : ksg)[desc.linearIdx], outHalfScale[prevS], 64);
            if (!desc.isQuery) {
                DataCopy(vo[desc.linearIdx * p.dp], outFp8[prevS], 64 * p.dp);
                DataCopy(mxg[desc.linearIdx / 64 * p.dp], vsBuf[prevS].template ReinterpretCast<uint16_t>(), p.dp);
            }
            SetFlag<HardEvent::MTE3_V>(ev);
        }

        for (uint32_t s = 0; s < Min(uint64_t(2), totalSteps); ++s)
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(1 + s));
        WaitFlag<HardEvent::V_MTE2>(1);
        WaitFlag<HardEvent::V_MTE2>(2);
    }

    __aicore__ static void ExecuteFallbackQuant(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, GM_ADDR mxScale,
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, const SageParams &p)
    {
        using namespace AscendC;
        memory::FixedResource r;
        GlobalTensor<T> qg, kg, vg;
        qg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q));
        kg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k));
        vg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(v));
        GlobalTensor<float> mg, vmg;
        mg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mean));
        vmg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(vMean));
        GlobalTensor<half> qsg, ksg;
        qsg.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(qs));
        ksg.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ks));
        GlobalTensor<int8_t> qo, ko;
        qo.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(q8));
        ko.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(k8));
        GlobalTensor<float8_e4m3_t> vo;
        vo.SetGlobalBuffer(reinterpret_cast<__gm__ float8_e4m3_t *>(v8));

        constexpr uint32_t X_OFF = 16384, MU_OFF = 49152, SCALE_OFF = 49664, INV_OFF = 49920;
        constexpr uint32_t HALF_OFF = 50176, OUT_OFF = 50304;
        constexpr uint32_t VIN_OFF = 65536, VMAX_OFF = 81920, VOUT_OFF = 82944, MX_OFF = 91136;
        constexpr uint32_t VMEAN_OFF = 92160;

        auto inRaw = r.ubBuf.template GetBufferByByte<T>(0);
        auto inF32 = r.ubBuf.template GetBufferByByte<float>(X_OFF);
        auto muBuf = r.ubBuf.template GetBufferByByte<float>(MU_OFF);
        auto scaleBuf = r.ubBuf.template GetBufferByByte<float>(SCALE_OFF);
        auto invBuf = r.ubBuf.template GetBufferByByte<float>(INV_OFF);
        auto halfScaleBuf = r.ubBuf.template GetBufferByByte<half>(HALF_OFF);
        auto outBuf = r.ubBuf.template GetBufferByByte<int8_t>(OUT_OFF);

        const uint64_t nq = WIDE ? 0 : (uint64_t(p.b) * p.hq * p.qp);
        const uint64_t nk = uint64_t(p.b) * p.hk * p.kp;

        for (uint64_t tile = GetBlockIdx(); tile < (nq + nk) / 64; tile += 2 * p.cores) {
            const bool isKv = tile < nk / 64;
            const bool isQ = !isKv;
            const uint64_t linIdx = (isKv ? tile : tile - nk / 64) * 64;
            const uint32_t seqLen = isQ ? p.qp : p.kp, numH = isQ ? p.hq : p.hk;
            const uint32_t tokIdx = linIdx % seqLen, h = (linIdx / seqLen) % numH, b = linIdx / seqLen / numH;
            const uint32_t seqBound = isQ ? p.sq : p.sk;
            const uint32_t valid = tokIdx < seqBound ? Min(uint32_t(64), seqBound - tokIdx) : 0;

            for (uint32_t row = 0; row < valid; ++row)
                ReadRow(inRaw[row * p.dp], isQ ? qg : kg, Offset(isQ ? p.qs : p.ks, b, h, tokIdx + row), p.d, p.dp);
            if (valid < 64)
                Duplicate(inRaw[valid * p.dp], T(0), (64 - valid) * p.dp);
            Cast(inF32, inRaw, RoundMode::CAST_NONE, 64 * p.dp);
            PipeBarrier<PIPE_V>();

            if (!isQ && p.smoothK && valid) {
                DataCopy(muBuf, mg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                Sync<HardEvent::MTE2_V>();
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
            WriteRow(isQ ? qo : ko, linIdx * p.dp, outBuf, 64 * p.dp);
            WriteRow(isQ ? qsg : ksg, linIdx, halfScaleBuf, 64);
        }

        PipeBarrier<PIPE_ALL>();
        GlobalTensor<uint16_t> mxg;
        mxg.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(mxScale));
        auto vInRaw = r.ubBuf.template GetBufferByByte<T>(VIN_OFF);
        auto vPeak = r.ubBuf.template GetBufferByByte<T>(VMAX_OFF);
        auto vOutBuf = r.ubBuf.template GetBufferByByte<float8_e4m3_t>(VOUT_OFF);
        auto vScale16 = r.ubBuf.template GetBufferByByte<uint16_t>(MX_OFF);
        auto vMeanF32 = r.ubBuf.template GetBufferByByte<float>(VMEAN_OFF);

        for (uint64_t tile = GetBlockIdx(); tile < nk / 64; tile += 2 * p.cores) {
            const uint64_t linIdx = tile * 64;
            const uint32_t tokIdx = linIdx % p.kp, h = (linIdx / p.kp) % p.hk, b = linIdx / p.kp / p.hk;
            const uint32_t valid = tokIdx < p.sk ? Min(uint32_t(64), p.sk - tokIdx) : 0;
            if (p.d == p.dp && valid) {
                Sync<HardEvent::V_MTE2>();
                DataCopy(vInRaw, vg[Offset(p.vs, b, h, tokIdx)], valid * p.dp);
                Sync<HardEvent::MTE2_V>();
            } else {
                for (uint32_t row = 0; row < valid; ++row)
                    ReadRow(vInRaw[row * p.dp], vg, Offset(p.vs, b, h, tokIdx + row), p.d, p.dp);
            }
            if (valid < 64)
                Duplicate(vInRaw[valid * p.dp], T(0), (64 - valid) * p.dp);
            PipeBarrier<PIPE_V>();
            if (p.smoothV && valid) {
                DataCopy(vMeanF32, vmg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                Sync<HardEvent::MTE2_V>();
                CenterVRowsVF<T>(reinterpret_cast<__ubuf__ T *>(vInRaw.GetPhyAddr()),
                                 reinterpret_cast<__ubuf__ float *>(vMeanF32.GetPhyAddr()), p.dp, valid);
                PipeBarrier<PIPE_V>();
            }
            QuantVRowsMxVF<T, WIDE>(reinterpret_cast<__ubuf__ T *>(vInRaw.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ T *>(vPeak.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ uint16_t *>(vScale16.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ float8_e4m3_t *>(vOutBuf.GetPhyAddr()), p.dp);
            WriteRow(vo, linIdx * p.dp, vOutBuf, 64 * p.dp);
            WriteRow(mxg, linIdx / 64 * p.dp, vScale16, p.dp);
        }
        PipeBarrier<PIPE_ALL>();
    }
};

#endif // __CCE__

} // namespace sage::microkernel

namespace sage {
using microkernel::BroadcastGroupMaxima;
using microkernel::PackFixedScaleBlock;
using microkernel::QuantRowsImpl;
using microkernel::QuantRowsVF;
#if defined(__CCE__)
using microkernel::FastQuant;
#endif
}
