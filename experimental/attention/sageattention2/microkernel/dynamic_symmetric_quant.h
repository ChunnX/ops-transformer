#pragma once

#include "../core/platform_traits.h"
#include "../core/numerical_types.h"
#include "../core/math_approximations.h"
#include "../memory/ring_buffer_layout.h"
#include "../memory/catlass_fractal_packer.h"
#include "../orchestration/credit_handshake.h"
#include "mxfp8_e4m3_encoder.h"

namespace sage::microkernel {

#if defined(__CCE__)
using AscendC::RoundMode;
using AscendC::CMPMODE;
using memory::Sync;
using AscendC::HardEvent;

// Channel-wise Reduction Microkernel (Stats Sum / Mean)

template <class T, uint32_t DP>
__simd_callee__ inline void LoadLaneF32(AscendC::MicroAPI::RegTensor<float> &x, __ubuf__ T *addr,
                                        uint32_t row, uint32_t col) {
    using namespace AscendC::MicroAPI;
    auto mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<T> raw;
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, addr + row * DP + col);
    Cast<float, T, cast>(x, raw, mask);
}

template <class T, uint32_t DP>
__simd_vf__ inline void StatsSumImpl(__ubuf__ T *rows, __ubuf__ float *state, uint32_t count) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto mask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t col = 0; col < DP; col += 64) {
        RegTensor<float> acc0, acc1, acc2, acc3;
        LoadAlign(acc0, state + col);
        Duplicate(acc1, 0.0f);
        Duplicate(acc2, 0.0f);
        Duplicate(acc3, 0.0f);
        uint32_t row = 0;
        for (; row + 3 < count; row += 4) {
            RegTensor<float> x0, x1, x2, x3;
            RegTensor<T> raw0, raw1, raw2, raw3;
            static constexpr CastTrait cast{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw0, rows + (row + 0) * DP + col);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw1, rows + (row + 1) * DP + col);
            Cast<float, T, cast>(x0, raw0, mask);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw2, rows + (row + 2) * DP + col);
            Cast<float, T, cast>(x1, raw1, mask);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw3, rows + (row + 3) * DP + col);
            Cast<float, T, cast>(x2, raw2, mask);
            Add(acc0, acc0, x0, mask);
            Cast<float, T, cast>(x3, raw3, mask);
            Add(acc1, acc1, x1, mask);
            Add(acc2, acc2, x2, mask);
            Add(acc3, acc3, x3, mask);
        }
        for (; row + 1 < count; row += 2) {
            RegTensor<float> x0, x1;
            LoadLaneF32<T, DP>(x0, rows, row, col);
            LoadLaneF32<T, DP>(x1, rows, row + 1, col);
            Add(acc0, acc0, x0, mask);
            Add(acc1, acc1, x1, mask);
        }
        if (row < count) {
            RegTensor<float> x_tail;
            LoadLaneF32<T, DP>(x_tail, rows, row, col);
            Add(acc0, acc0, x_tail, mask);
        }
        Add(acc0, acc0, acc1, mask);
        Add(acc2, acc2, acc3, mask);
        Add(acc0, acc0, acc2, mask);
        StoreAlign(state + col, acc0, mask);
    }
}

template <class T>
__aicore__ inline void StatsSumVF(AscendC::LocalTensor<T> rows, AscendC::LocalTensor<float> state,
                                  uint32_t count, uint32_t dp) {
    auto input = (__ubuf__ T *)rows.GetPhyAddr();
    auto sum = (__ubuf__ float *)state.GetPhyAddr();
    if (dp == 128)
        StatsSumImpl<T, 128>(input, sum, count);
    else
        StatsSumImpl<T, 64>(input, sum, count);
}

// Smooth-V Mean Centering Microkernel

template <class T, uint32_t DP>
__simd_vf__ inline void CenterVImpl(__ubuf__ T *rows, __ubuf__ float *mean, uint32_t count) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto mask = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toF32{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr CastTrait toT{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint32_t col = 0; col < DP; col += 64) {
        RegTensor<float> m;
        LoadAlign(m, mean + col);
        for (uint32_t row = 0; row < count; ++row) {
            RegTensor<T> raw;
            RegTensor<float> val;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, rows + row * DP + col);
            Cast<float, T, toF32>(val, raw, mask);
            Sub(val, val, m, mask);
            Cast<T, float, toT>(raw, val, mask);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(rows + row * DP + col, raw, mask);
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

// Symmetrical INT8 Quantization Microkernel Primitives

// Broadcast grouped INT8 scale across token groups (4 for per_thread, 32 for per_warp)
__simd_callee__ inline void BroadcastGroupMaxima(__ubuf__ float *maxima, uint32_t group) {
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> m, shifted;
    LoadAlign(m, maxima);
    for (uint32_t step = 1; step < group; step <<= 1) {
        StoreAlign(maxima, m, full);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(shifted, maxima + step);
        Max(m, m, shifted, full);
    }
    StoreAlign(maxima, m, full);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto keep = UpdateMask<float>(group);
    for (uint32_t start = 0; start < 64; start += group) {
        RegTensor<float> value;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(value, maxima + start);
        StoreAlign(maxima + start, value, keep);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

// Convert FP32 scales to half precision with pre-fused dequant factor
__simd_callee__ inline void PackFixedScaleBlock(__ubuf__ float *scales, __ubuf__ half *halfScales,
                                                float factor, uint32_t valid) {
    using namespace AscendC::MicroAPI;
    auto mask = UpdateMask<float>(valid);
    RegTensor<float> r0, r1, rf;
    RegTensor<half> h0, h1;
    Duplicate(rf, factor);
    LoadAlign(r0, scales);
    LoadAlign(r1, scales + 64);
    Mul(r0, r0, rf, mask);
    Mul(r1, r1, rf, mask);
    static constexpr CastTrait trait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    Cast<half, float, trait>(h0, r0, mask);
    Cast<half, float, trait>(h1, r1, mask);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(halfScales, h0, mask);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(halfScales + 64, h1, mask);
}

// Symmetrical INT8 Quantization microkernel with 4-way unrolling to saturate Vector ALU
template <uint32_t DP, bool SMOOTH>
__simd_vf__ inline void QuantRowsImpl(__ubuf__ float *x, [[maybe_unused]] __ubuf__ float *mean,
                                      __ubuf__ float *scales, __ubuf__ int8_t *out,
                                      __ubuf__ half *halfScales, float factor,
                                      __ubuf__ float *inverseScales, uint32_t valid, uint32_t group) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> ma, mb, zero, one, invDivisor;
    Duplicate(zero, 0.0f);
    Duplicate(one, 1.0f);
    Duplicate(invDivisor, INV_INT8_MAX);

    if constexpr (SMOOTH) {
        LoadAlign(ma, mean);
        if constexpr (DP == 128) LoadAlign(mb, mean + 64);
    }

    auto dst = scales;
    UnalignReg compact;

    if constexpr (DP == 128) {
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> a0, a1, ab0, ab1, m0, m1;
            LoadAlign(a0, x + row * DP);
            LoadAlign(a1, x + (row + 1) * DP);
            if constexpr (SMOOTH) {
                Sub(a0, a0, ma, full);
                Sub(a1, a1, ma, full);
            }
            Abs(ab0, a0, full);
            Abs(ab1, a1, full);
            if constexpr (SMOOTH) {
                StoreAlign(x + row * DP, a0, full);
                StoreAlign(x + (row + 1) * DP, a1, full);
            }
            RegTensor<float> b0, b1, bb0, bb1;
            LoadAlign(b0, x + row * DP + 64);
            LoadAlign(b1, x + (row + 1) * DP + 64);
            if constexpr (SMOOTH) {
                Sub(b0, b0, mb, full);
                Sub(b1, b1, mb, full);
            }
            Abs(bb0, b0, full);
            Abs(bb1, b1, full);
            if constexpr (SMOOTH) {
                StoreAlign(x + row * DP + 64, b0, full);
                StoreAlign(x + (row + 1) * DP + 64, b1, full);
            }
            Max(m0, ab0, bb0, full);
            Max(m1, ab1, bb1, full);
            ReduceDataBlock<ReduceType::MAX>(m0, m0, full);
            ReduceDataBlock<ReduceType::MAX>(m1, m1, full);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, m0, compact, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, m1, compact, 1);
        }
    } else {
        for (uint16_t row = 0; row < 64; row += 4) {
            RegTensor<float> a0, a1, a2, a3;
            RegTensor<float> ab0, ab1, ab2, ab3;
            LoadAlign(a0, x + (row + 0) * DP);
            LoadAlign(a1, x + (row + 1) * DP);
            LoadAlign(a2, x + (row + 2) * DP);
            LoadAlign(a3, x + (row + 3) * DP);
            if constexpr (SMOOTH) {
                Sub(a0, a0, ma, full);
                Sub(a1, a1, ma, full);
                Sub(a2, a2, ma, full);
                Sub(a3, a3, ma, full);
            }
            Abs(ab0, a0, full);
            Abs(ab1, a1, full);
            Abs(ab2, a2, full);
            Abs(ab3, a3, full);
            if constexpr (SMOOTH) {
                StoreAlign(x + (row + 0) * DP, a0, full);
                StoreAlign(x + (row + 1) * DP, a1, full);
                StoreAlign(x + (row + 2) * DP, a2, full);
                StoreAlign(x + (row + 3) * DP, a3, full);
            }
            ReduceDataBlock<ReduceType::MAX>(ab0, ab0, full);
            ReduceDataBlock<ReduceType::MAX>(ab1, ab1, full);
            ReduceDataBlock<ReduceType::MAX>(ab2, ab2, full);
            ReduceDataBlock<ReduceType::MAX>(ab3, ab3, full);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, ab0, compact, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, ab1, compact, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, ab2, compact, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, ab3, compact, 1);
        }
    }

    StoreUnAlignPost(dst, compact, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    BroadcastGroupMaxima(scales, group);

    RegTensor<float> m;
    LoadAlign(m, scales);
    Mul(m, m, invDivisor, full);
    MaskReg positive;
    Compare<float, CMPMODE::GT>(positive, m, zero, full);
    Select(m, m, one, positive);
    StoreAlign(scales, m, full);

    // Two-stage Taylor-Newton reciprocal
    RegTensor<float> inverse;
    FastReciprocal(inverse, m, full);
    StoreAlign(inverseScales, inverse, full);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    static constexpr CastTrait trait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    if constexpr (DP == 128) {
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> m0, m1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(m0, inverseScales + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(m1, inverseScales + row + 1);

            RegTensor<float> a0_0, a0_1, a1_0, a1_1;
            RegTensor<int32_t> p0_0, p0_1, p1_0, p1_1;
            LoadAlign(a0_0, x + row * DP);
            LoadAlign(a0_1, x + row * DP + 64);
            LoadAlign(a1_0, x + (row + 1) * DP);
            LoadAlign(a1_1, x + (row + 1) * DP + 64);

            Mul(a0_0, a0_0, m0, full);
            Mul(a0_1, a0_1, m0, full);
            Mul(a1_0, a1_0, m1, full);
            Mul(a1_1, a1_1, m1, full);

            Cast<int32_t, float, trait>(p0_0, a0_0, full);
            Cast<int32_t, float, trait>(p0_1, a0_1, full);
            Cast<int32_t, float, trait>(p1_0, a1_0, full);
            Cast<int32_t, float, trait>(p1_1, a1_1, full);

            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + row * DP, (RegTensor<int8_t> &)p0_0, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + row * DP + 64, (RegTensor<int8_t> &)p0_1, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (row + 1) * DP, (RegTensor<int8_t> &)p1_0, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (row + 1) * DP + 64, (RegTensor<int8_t> &)p1_1, full);
        }
    } else {
        for (uint16_t row = 0; row < 64; row += 4) {
            RegTensor<float> m0, m1, m2, m3;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(m0, inverseScales + row + 0);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(m1, inverseScales + row + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(m2, inverseScales + row + 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(m3, inverseScales + row + 3);

            RegTensor<float> a0, a1, a2, a3;
            RegTensor<int32_t> p0, p1, p2, p3;
            LoadAlign(a0, x + (row + 0) * DP);
            LoadAlign(a1, x + (row + 1) * DP);
            LoadAlign(a2, x + (row + 2) * DP);
            LoadAlign(a3, x + (row + 3) * DP);

            Mul(a0, a0, m0, full);
            Mul(a1, a1, m1, full);
            Mul(a2, a2, m2, full);
            Mul(a3, a3, m3, full);

            Cast<int32_t, float, trait>(p0, a0, full);
            Cast<int32_t, float, trait>(p1, a1, full);
            Cast<int32_t, float, trait>(p2, a2, full);
            Cast<int32_t, float, trait>(p3, a3, full);

            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (row + 0) * DP, (RegTensor<int8_t> &)p0, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (row + 1) * DP, (RegTensor<int8_t> &)p1, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (row + 2) * DP, (RegTensor<int8_t> &)p2, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (row + 3) * DP, (RegTensor<int8_t> &)p3, full);
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

// QuantKernel Pipeline Wrapper

template <class T, bool WIDE>
class QuantKernel {
public:
    struct QuantTaskInfo {
        uint64_t idx;
        uint32_t valid;
        bool isq;
    };

    __aicore__ static QuantTaskInfo DecodeQuantTask(uint64_t tile, const SageParams &p, uint64_t nq, uint64_t nk) {
        const bool iskv = tile < nk / 64;
        const bool isq = !iskv;
        const uint64_t idx = (iskv ? tile : tile - nk / 64) * 64;
        const uint32_t seq = isq ? p.qp : p.kp;
        const uint32_t token = idx % seq;
        const uint32_t limit = isq ? p.sq : p.sk;
        const uint32_t valid = token < limit ? (uint32_t(64) < (limit - token) ? uint32_t(64) : (limit - token)) : 0;
        return {idx, valid, isq};
    }

    __aicore__ static void Execute(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, GM_ADDR mxScale,
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, const SageParams &p)
    {
        using namespace AscendC;
        memory::FixedResource r;
        GlobalTensor<T> qg, kg, vg;
        qg.SetGlobalBuffer((__gm__ T *)q);
        kg.SetGlobalBuffer((__gm__ T *)k);
        vg.SetGlobalBuffer((__gm__ T *)v);
        GlobalTensor<float> mg, vmg;
        mg.SetGlobalBuffer((__gm__ float *)mean);
        vmg.SetGlobalBuffer((__gm__ float *)vMean);
        GlobalTensor<half> qsg, ksg;
        qsg.SetGlobalBuffer((__gm__ half *)qs);
        ksg.SetGlobalBuffer((__gm__ half *)ks);
        GlobalTensor<int8_t> qo, ko;
        qo.SetGlobalBuffer((__gm__ int8_t *)q8);
        ko.SetGlobalBuffer((__gm__ int8_t *)k8);
        GlobalTensor<float8_e4m3_t> vo;
        vo.SetGlobalBuffer((__gm__ float8_e4m3_t *)v8);
        GlobalTensor<uint16_t> mxg;
        mxg.SetGlobalBuffer((__gm__ uint16_t *)mxScale);

        const uint64_t nq = WIDE ? 0 : (uint64_t(p.b) * p.hq * p.qp);
        const uint64_t nk = uint64_t(p.b) * p.hk * p.kp;
        const uint64_t total = (nq + nk) / 64;
        const uint64_t step = 2 * p.cores;

        constexpr uint32_t IN_SLOT = 16384, X_SLOT = 16384, V_IN_SLOT = 16384;
        LocalTensor<T> input[2] = {r.ubBuf.template GetBufferByByte<T>(0),
                                  r.ubBuf.template GetBufferByByte<T>(IN_SLOT)};
        LocalTensor<float> x[2] = {r.ubBuf.template GetBufferByByte<float>(32768),
                                  r.ubBuf.template GetBufferByByte<float>(32768 + X_SLOT)};
        LocalTensor<float> mu[2] = {r.ubBuf.template GetBufferByByte<float>(65536),
                                   r.ubBuf.template GetBufferByByte<float>(65536 + 512)};
        LocalTensor<float> vmean[2] = {r.ubBuf.template GetBufferByByte<float>(66560),
                                      r.ubBuf.template GetBufferByByte<float>(66560 + 512)};
        LocalTensor<float> scales[2] = {r.ubBuf.template GetBufferByByte<float>(67584),
                                       r.ubBuf.template GetBufferByByte<float>(67584 + 256)};
        LocalTensor<float> inverseScales[2] = {r.ubBuf.template GetBufferByByte<float>(68096),
                                              r.ubBuf.template GetBufferByByte<float>(68096 + 256)};
        LocalTensor<half> halfScales[2] = {r.ubBuf.template GetBufferByByte<half>(68608),
                                          r.ubBuf.template GetBufferByByte<half>(68608 + 128)};
        LocalTensor<int8_t> out[2] = {r.ubBuf.template GetBufferByByte<int8_t>(68864),
                                     r.ubBuf.template GetBufferByByte<int8_t>(68864 + 8192)};
        LocalTensor<T> vInput[2] = {r.ubBuf.template GetBufferByByte<T>(85248),
                                   r.ubBuf.template GetBufferByByte<T>(85248 + V_IN_SLOT)};
        LocalTensor<float8_e4m3_t> fp8[2] = {r.ubBuf.template GetBufferByByte<float8_e4m3_t>(117992),
                                            r.ubBuf.template GetBufferByByte<float8_e4m3_t>(117992 + 8192)};
        LocalTensor<T> vs[2] = {r.ubBuf.template GetBufferByByte<T>(134376),
                               r.ubBuf.template GetBufferByByte<T>(134376 + 256)};

        SetFlag<HardEvent::V_MTE2>(1);
        SetFlag<HardEvent::V_MTE2>(2);
        SetFlag<HardEvent::MTE3_V>(1);
        SetFlag<HardEvent::MTE3_V>(2);

        uint64_t iterations = 0;
        for (uint64_t tile = GetBlockIdx(); tile < total; tile += step, ++iterations) {
            const uint32_t slot = static_cast<uint32_t>(iterations & 1);
            const auto event = static_cast<event_t>(1 + slot);
            const QuantTaskInfo info = DecodeQuantTask(tile, p, nq, nk);
            const uint32_t seq = info.isq ? p.qp : p.kp;
            const uint32_t heads = info.isq ? p.hq : p.hk;
            const uint32_t token = info.idx % seq;
            const uint32_t h = (info.idx / seq) % heads;
            const uint32_t b = info.idx / seq / heads;

            WaitFlag<HardEvent::V_MTE2>(event);
            if (p.d == p.dp && info.valid) {
                Sync<HardEvent::V_MTE2>();
                DataCopy(input[slot], (info.isq ? qg : kg)[Offset(info.isq ? p.qs : p.ks, b, h, token)], info.valid * p.dp);
                Sync<HardEvent::MTE2_V>();
            } else {
                for (uint32_t row = 0; row < info.valid; ++row) {
                    ReadRow(input[slot][row * p.dp], info.isq ? qg : kg,
                            Offset(info.isq ? p.qs : p.ks, b, h, token + row), p.d, p.dp);
                }
            }
            if (!info.isq) {
                if (p.d == p.dp && info.valid) {
                    Sync<HardEvent::V_MTE2>();
                    DataCopy(vInput[slot], vg[Offset(p.vs, b, h, token)], info.valid * p.dp);
                    Sync<HardEvent::MTE2_V>();
                } else {
                    for (uint32_t row = 0; row < info.valid; ++row) {
                        ReadRow(vInput[slot][row * p.dp], vg, Offset(p.vs, b, h, token + row), p.d, p.dp);
                    }
                }
            }
            if (!info.isq && p.smoothK && info.valid) {
                Sync<HardEvent::V_MTE2>();
                DataCopy(mu[slot], mg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                Sync<HardEvent::MTE2_V>();
            }
            if (!info.isq && p.smoothV && info.valid) {
                Sync<HardEvent::V_MTE2>();
                DataCopy(vmean[slot], vmg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                Sync<HardEvent::MTE2_V>();
            }

            WaitFlag<HardEvent::MTE3_V>(event);
            if (info.valid < 64)
                Duplicate(input[slot][info.valid * p.dp], T(0), (64 - info.valid) * p.dp);
            PipeBarrier<PIPE_V>();
            Cast(x[slot], input[slot], RoundMode::CAST_NONE, 64 * p.dp);
            PipeBarrier<PIPE_V>();
            QuantRowsVF((__ubuf__ float *)x[slot].GetPhyAddr(), (__ubuf__ float *)mu[slot].GetPhyAddr(),
                        (__ubuf__ float *)scales[slot].GetPhyAddr(), (__ubuf__ int8_t *)out[slot].GetPhyAddr(),
                        (__ubuf__ half *)halfScales[slot].GetPhyAddr(), info.isq ? p.qFactor : p.kFactor,
                        (__ubuf__ float *)inverseScales[slot].GetPhyAddr(), p.dp, !info.isq && p.smoothK && info.valid,
                        info.valid, info.isq ? p.qg : p.kg);
            if (!info.isq) {
                PipeBarrier<PIPE_V>();
                if (info.valid < 64)
                    Duplicate(vInput[slot][info.valid * p.dp], T(0), (64 - info.valid) * p.dp);
                PipeBarrier<PIPE_V>();
                if (p.smoothV && info.valid) {
                    CenterVRowsVF<T>((__ubuf__ T *)vInput[slot].GetPhyAddr(),
                                     (__ubuf__ float *)vmean[slot].GetPhyAddr(), p.dp, info.valid);
                    PipeBarrier<PIPE_V>();
                }
                QuantVRowsMxVF<T, WIDE>((__ubuf__ T *)vInput[slot].GetPhyAddr(),
                                        (__ubuf__ T *)x[slot].GetPhyAddr(),
                                        (__ubuf__ uint16_t *)vs[slot].GetPhyAddr(),
                                        (__ubuf__ float8_e4m3_t *)fp8[slot].GetPhyAddr(), p.dp);
            }
            SetFlag<HardEvent::V_MTE2>(event);
            Sync<HardEvent::V_MTE3>();
            DataCopy((info.isq ? qo : ko)[info.idx * p.dp], out[slot], 64 * p.dp);
            DataCopy((info.isq ? qsg : ksg)[info.idx], halfScales[slot], 64);
            if (!info.isq) {
                DataCopy(vo[info.idx * p.dp], fp8[slot], 64 * p.dp);
                DataCopy(mxg[info.idx / 64 * p.dp], vs[slot].template ReinterpretCast<uint16_t>(), p.dp);
            }
            SetFlag<HardEvent::MTE3_V>(event);
        }
        for (uint32_t slot = 0; slot < Min(uint64_t(2), iterations); ++slot)
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(1 + slot));
        WaitFlag<HardEvent::V_MTE2>(1);
        WaitFlag<HardEvent::V_MTE2>(2);
    }
};

// Sequence Mean Accumulator Pipeline

template <class T>
class StatsKernel {
public:
    __aicore__ static void Execute(GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, const SageParams &p) {
        using namespace AscendC;
        memory::FixedResource r;
        GlobalTensor<T> kg, vg;
        kg.SetGlobalBuffer((__gm__ T *)k);
        vg.SetGlobalBuffer((__gm__ T *)v);
        GlobalTensor<float> mg, vmg;
        mg.SetGlobalBuffer((__gm__ float *)mean);
        vmg.SetGlobalBuffer((__gm__ float *)vMean);

        auto rows = r.ubBuf.template GetBufferByByte<T>(0);
        auto sum = r.ubBuf.template GetBufferByByte<float>(QUANT_TILE_ROWS * p.dp * sizeof(T));
        auto invSk = 1.0f / float(p.sk);

        for (uint64_t idx = GetBlockIdx(); idx < uint64_t(p.b) * p.hk; idx += 2 * p.cores) {
            const uint32_t h = idx % p.hk, b = idx / p.hk;
            Duplicate(sum, 0.0f, p.dp);
            PipeBarrier<PIPE_V>();

            for (uint32_t t = 0; t < p.sk; t += QUANT_TILE_ROWS) {
                const uint32_t valid = Min(QUANT_TILE_ROWS, p.sk - t);
                for (uint32_t row = 0; row < valid; ++row) {
                    ReadRow(rows[row * p.dp], kg, Offset(p.ks, b, h, t + row), p.d, p.dp);
                }
                PipeBarrier<PIPE_V>();
                StatsSumVF(rows, sum, valid, p.dp);
            }
            PipeBarrier<PIPE_V>();
            Muls(sum, sum, invSk, p.dp);
            Sync<HardEvent::V_MTE3>();
            DataCopy(mg[idx * p.dp], sum, p.dp);

            if (p.smoothV) {
                Duplicate(sum, 0.0f, p.dp);
                PipeBarrier<PIPE_V>();
                for (uint32_t t = 0; t < p.sk; t += QUANT_TILE_ROWS) {
                    const uint32_t valid = Min(QUANT_TILE_ROWS, p.sk - t);
                    for (uint32_t row = 0; row < valid; ++row) {
                        ReadRow(rows[row * p.dp], vg, Offset(p.vs, b, h, t + row), p.d, p.dp);
                    }
                    PipeBarrier<PIPE_V>();
                    StatsSumVF(rows, sum, valid, p.dp);
                }
                PipeBarrier<PIPE_V>();
                Muls(sum, sum, invSk, p.dp);
                Sync<HardEvent::V_MTE3>();
                DataCopy(vmg[idx * p.dp], sum, p.dp);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
};

// FastQuant Engine Class

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
        const bool center = IS_V ? bool(p.smoothV) : bool(p.smoothK);
        if (!center) return;

        const uint32_t workers = 2 * p.cores;
        const uint64_t heads = uint64_t(p.b) * p.hk;
        memory::FixedResource r;
        GlobalTensor<T> kg;
        kg.SetGlobalBuffer((__gm__ T *)src);
        GlobalTensor<float> mg;
        mg.SetGlobalBuffer((__gm__ float *)mean);

        constexpr uint32_t STATS_BATCH_ROWS = 128, STATS_DP_MAX = 128;
        constexpr uint32_t STATS_SLOT_BYTES = STATS_BATCH_ROWS * STATS_DP_MAX * 2U;
        constexpr uint32_t STATS_F32_BYTES = STATS_BATCH_ROWS * STATS_DP_MAX * 4U;
        constexpr uint32_t STATS_RAW_UB = 0, STATS_F32_UB = 2 * STATS_SLOT_BYTES;
        constexpr uint32_t STATS_SUM_UB = STATS_F32_UB + STATS_F32_BYTES;

        auto input = r.ubBuf.template GetBufferByByte<T>(STATS_RAW_UB);
        auto x = r.ubBuf.template GetBufferByByte<float>(STATS_F32_UB);
        auto sum = r.ubBuf.template GetBufferByByte<float>(STATS_SUM_UB);
        const uint32_t rank = GetBlockIdx() % workers;
        const float invSk = 1.0f / float(p.sk);

        for (uint64_t bh = rank; bh < heads; bh += workers) {
            uint32_t b = bh / p.hk, h = bh % p.hk;
            Duplicate(sum, 0.0f, p.dp);
            if (p.d == p.dp) {
                LocalTensor<T> slotBuf[2] = {
                    r.ubBuf.template GetBufferByByte<T>(0),
                    r.ubBuf.template GetBufferByByte<T>(STATS_SLOT_BYTES)
                };
                const uint32_t batches = (p.sk + STATS_BATCH_ROWS - 1) / STATS_BATCH_ROWS;
                SetFlag<HardEvent::V_MTE2>(1);
                SetFlag<HardEvent::V_MTE2>(2);
                for (uint32_t batch = 0; batch <= batches; ++batch) {
                    if (batch < batches) {
                        const uint32_t slot = batch & 1, start = batch * STATS_BATCH_ROWS;
                        const uint32_t rows = Min(STATS_BATCH_ROWS, p.sk - start);
                        const event_t event = static_cast<event_t>(1 + slot);
                        WaitFlag<HardEvent::V_MTE2>(event);
                        DataCopy(slotBuf[slot], kg[Offset(strides, b, h, start)], rows * p.dp);
                        SetFlag<HardEvent::MTE2_V>(event);
                    }
                    if (batch == 0) continue;
                    const uint32_t slot = (batch - 1) & 1;
                    const uint32_t rows = Min(STATS_BATCH_ROWS, p.sk - (batch - 1) * STATS_BATCH_ROWS);
                    const event_t event = static_cast<event_t>(1 + slot);
                    WaitFlag<HardEvent::MTE2_V>(event);
                    StatsSumVF(slotBuf[slot], sum, rows, p.dp);
                    SetFlag<HardEvent::V_MTE2>(event);
                }
                WaitFlag<HardEvent::V_MTE2>(1);
                WaitFlag<HardEvent::V_MTE2>(2);
            } else {
                for (uint32_t row = 0; row < p.sk; ++row) {
                    ReadRow(input, kg, Offset(strides, b, h, row), p.d, p.dp);
                    Cast(x, input, RoundMode::CAST_NONE, p.dp);
                    PipeBarrier<PIPE_V>();
                    Add(sum, sum, x, p.dp);
                }
            }
            PipeBarrier<PIPE_V>();
            Muls(sum, sum, invSk, p.dp);
            WriteRow(mg, bh * p.dp, sum, p.dp);
        }
    }

    __aicore__ static void Quantize(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, GM_ADDR mxScale,
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, const SageParams &p)
    {
        if (p.d == p.dp) {
            QuantBatch(q, k, v, mean, vMean, mxScale, q8, k8, v8, qs, ks, p);
        } else {
            QuantFallback(q, k, v, mean, vMean, mxScale, q8, k8, v8, qs, ks, p);
        }
    }

    __aicore__ static void QuantBatch(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, GM_ADDR mxScale,
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, const SageParams &p)
    {
        using namespace AscendC;
        memory::FixedResource r;
        GlobalTensor<T> qg, kg, vg;
        qg.SetGlobalBuffer((__gm__ T *)q);
        kg.SetGlobalBuffer((__gm__ T *)k);
        vg.SetGlobalBuffer((__gm__ T *)v);
        qg.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        vg.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        GlobalTensor<uint16_t> mxg;
        mxg.SetGlobalBuffer((__gm__ uint16_t *)mxScale);
        GlobalTensor<float> mg, vmg;
        mg.SetGlobalBuffer((__gm__ float *)mean);
        vmg.SetGlobalBuffer((__gm__ float *)vMean);
        GlobalTensor<half> qsg, ksg;
        qsg.SetGlobalBuffer((__gm__ half *)qs);
        ksg.SetGlobalBuffer((__gm__ half *)ks);
        GlobalTensor<int8_t> qo, ko;
        qo.SetGlobalBuffer((__gm__ int8_t *)q8);
        ko.SetGlobalBuffer((__gm__ int8_t *)k8);
        GlobalTensor<float8_e4m3_t> vo;
        vo.SetGlobalBuffer((__gm__ float8_e4m3_t *)v8);

        constexpr uint32_t SLOT_BYTES = 96 * 1024;
        constexpr uint32_t V_INPUT = 16384, FLOAT_INPUT = 32768;
        constexpr uint32_t QK_OUTPUT = 65536, V_OUTPUT = 73728;
        constexpr uint32_t MEAN = 81920, MX_SCALE = 82432, RAW_SCALE = 82944, HALF_SCALE = 83200;
        constexpr uint32_t INVERSE_SCALE = 83456, V_MEAN = 83712;

        LocalTensor<T> input[2], vInput[2];
        LocalTensor<float> x[2], mu[2], vmean[2], vs[2], scales[2], inverseScales[2];
        LocalTensor<int8_t> out[2];
        LocalTensor<float8_e4m3_t> fp8[2];
        LocalTensor<half> halfScales[2];
        for (uint32_t s = 0; s < 2; ++s) {
            const uint32_t b = s * SLOT_BYTES;
            input[s] = r.ubBuf.template GetBufferByByte<T>(b);
            vInput[s] = r.ubBuf.template GetBufferByByte<T>(b + V_INPUT);
            x[s] = r.ubBuf.template GetBufferByByte<float>(b + FLOAT_INPUT);
            mu[s] = r.ubBuf.template GetBufferByByte<float>(b + MEAN);
            vmean[s] = r.ubBuf.template GetBufferByByte<float>(b + V_MEAN);
            vs[s] = r.ubBuf.template GetBufferByByte<float>(b + MX_SCALE);
            out[s] = r.ubBuf.template GetBufferByByte<int8_t>(b + QK_OUTPUT);
            fp8[s] = r.ubBuf.template GetBufferByByte<float8_e4m3_t>(b + V_OUTPUT);
            scales[s] = r.ubBuf.template GetBufferByByte<float>(b + RAW_SCALE);
            halfScales[s] = r.ubBuf.template GetBufferByByte<half>(b + HALF_SCALE);
            inverseScales[s] = r.ubBuf.template GetBufferByByte<float>(b + INVERSE_SCALE);
        }

        const uint64_t nq = uint64_t(p.b) * p.hq * p.qp / 64;
        const uint64_t nk = uint64_t(p.b) * p.hk * p.kp / 64;
        const uint32_t cores = 2 * p.cores;
        const uint32_t core = GetBlockIdx() % cores;
        const uint64_t localNq = WIDE ? 0 : nq, localNk = nk;
        if (core >= localNq + localNk) return;
        const uint64_t iterations = (localNq + localNk - core + cores - 1) / cores;

        SetFlag<HardEvent::V_MTE2>(1);
        SetFlag<HardEvent::V_MTE2>(2);
        struct BatchInfo {
            uint64_t idx;
            bool isq;
            uint32_t valid;
        } pending[2];

        for (uint64_t iteration = 0; iteration <= iterations; ++iteration) {
            if (iteration < iterations) {
                const uint32_t slot = iteration & 1;
                const uint64_t task = core + iteration * cores;
                const bool iskv = task < localNk;
                const bool isq = !iskv;
                const uint64_t idx = (iskv ? task : (task - localNk)) * 64;
                const uint32_t seq = isq ? p.qp : p.kp, heads = isq ? p.hq : p.hk;
                const uint32_t token = idx % seq, h = (idx / seq) % heads, b = idx / seq / heads;
                const uint32_t valid = token < (isq ? p.sq : p.sk) ? 64 : 0;
                pending[slot] = {idx, isq, valid};
                WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(1 + slot));
                if (valid)
                    DataCopy(input[slot], (isq ? qg : kg)[Offset(isq ? p.qs : p.ks, b, h, token)], 64 * p.dp);
                if (!isq) {
                    if (valid)
                        DataCopy(vInput[slot], vg[Offset(p.vs, b, h, token)], 64 * p.dp);
                    DataCopy(mu[slot], mg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                    if (p.smoothV)
                        DataCopy(vmean[slot], vmg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                }
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(1 + slot));
            }
            if (iteration == 0) continue;
            const uint32_t slot = (iteration - 1) & 1;
            const event_t event = static_cast<event_t>(1 + slot);
            const BatchInfo info = pending[slot];
            WaitFlag<HardEvent::MTE2_V>(event);
            if (iteration > 2)
                WaitFlag<HardEvent::MTE3_V>(event);
            if (info.valid < 64) {
                Duplicate(input[slot][info.valid * p.dp], T(0), (64 - info.valid) * p.dp);
                PipeBarrier<PIPE_V>();
            }
            Cast(x[slot], input[slot], RoundMode::CAST_NONE, 64 * p.dp);
            PipeBarrier<PIPE_V>();
            QuantRowsVF((__ubuf__ float *)x[slot].GetPhyAddr(), (__ubuf__ float *)mu[slot].GetPhyAddr(),
                        (__ubuf__ float *)scales[slot].GetPhyAddr(), (__ubuf__ int8_t *)out[slot].GetPhyAddr(),
                        (__ubuf__ half *)halfScales[slot].GetPhyAddr(), info.isq ? p.qFactor : p.kFactor,
                        (__ubuf__ float *)inverseScales[slot].GetPhyAddr(), p.dp, !info.isq && p.smoothK && info.valid,
                        info.valid, info.isq ? p.qg : p.kg);
            if (!info.isq) {
                PipeBarrier<PIPE_V>();
                if (info.valid < 64)
                    Duplicate(vInput[slot][info.valid * p.dp], T(0), (64 - info.valid) * p.dp);
                PipeBarrier<PIPE_V>();
                if (p.smoothV && info.valid) {
                    CenterVRowsVF<T>((__ubuf__ T *)vInput[slot].GetPhyAddr(),
                                     (__ubuf__ float *)vmean[slot].GetPhyAddr(), p.dp, info.valid);
                    PipeBarrier<PIPE_V>();
                }
                QuantVRowsMxVF<T, WIDE>((__ubuf__ T *)vInput[slot].GetPhyAddr(),
                                        (__ubuf__ T *)x[slot].GetPhyAddr(),
                                        (__ubuf__ uint16_t *)vs[slot].GetPhyAddr(),
                                        (__ubuf__ float8_e4m3_t *)fp8[slot].GetPhyAddr(), p.dp);
            }
            SetFlag<HardEvent::V_MTE2>(event);
            Sync<HardEvent::V_MTE3>();
            DataCopy((info.isq ? qo : ko)[info.idx * p.dp], out[slot], 64 * p.dp);
            DataCopy((info.isq ? qsg : ksg)[info.idx], halfScales[slot], 64);
            if (!info.isq) {
                DataCopy(vo[info.idx * p.dp], fp8[slot], 64 * p.dp);
                DataCopy(mxg[info.idx / 64 * p.dp], vs[slot].template ReinterpretCast<uint16_t>(), p.dp);
            }
            SetFlag<HardEvent::MTE3_V>(event);
        }
        for (uint32_t slot = 0; slot < Min(uint64_t(2), iterations); ++slot)
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(1 + slot));
        WaitFlag<HardEvent::V_MTE2>(1);
        WaitFlag<HardEvent::V_MTE2>(2);
    }

    __aicore__ static void QuantFallback(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, GM_ADDR mxScale,
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, const SageParams &p)
    {
        using namespace AscendC;
        memory::FixedResource r;
        GlobalTensor<T> qg, kg, vg;
        qg.SetGlobalBuffer((__gm__ T *)q);
        kg.SetGlobalBuffer((__gm__ T *)k);
        vg.SetGlobalBuffer((__gm__ T *)v);
        GlobalTensor<float> mg, vmg;
        mg.SetGlobalBuffer((__gm__ float *)mean);
        vmg.SetGlobalBuffer((__gm__ float *)vMean);
        GlobalTensor<half> qsg, ksg;
        qsg.SetGlobalBuffer((__gm__ half *)qs);
        ksg.SetGlobalBuffer((__gm__ half *)ks);
        GlobalTensor<int8_t> qo, ko;
        qo.SetGlobalBuffer((__gm__ int8_t *)q8);
        ko.SetGlobalBuffer((__gm__ int8_t *)k8);
        GlobalTensor<float8_e4m3_t> vo;
        vo.SetGlobalBuffer((__gm__ float8_e4m3_t *)v8);

        constexpr uint32_t X = 16384, MU = 49152, RAW_SCALE = 49664, INVERSE_SCALE = 49920;
        constexpr uint32_t HALF_SCALE = 50176, QK_OUTPUT = 50304;
        constexpr uint32_t V_INPUT = 65536, V_MAXIMA = 81920, V_OUTPUT = 82944, MX_SCALE = 91136;
        constexpr uint32_t V_MEAN = 92160;

        auto input = r.ubBuf.template GetBufferByByte<T>(0);
        auto x = r.ubBuf.template GetBufferByByte<float>(X);
        auto mu = r.ubBuf.template GetBufferByByte<float>(MU);
        auto scales = r.ubBuf.template GetBufferByByte<float>(RAW_SCALE);
        auto inverseScales = r.ubBuf.template GetBufferByByte<float>(INVERSE_SCALE);
        auto halfScales = r.ubBuf.template GetBufferByByte<half>(HALF_SCALE);
        auto out = r.ubBuf.template GetBufferByByte<int8_t>(QK_OUTPUT);

        const uint64_t nq = WIDE ? 0 : (uint64_t(p.b) * p.hq * p.qp);
        const uint64_t nk = uint64_t(p.b) * p.hk * p.kp;

        for (uint64_t tile = GetBlockIdx(); tile < (nq + nk) / 64; tile += 2 * p.cores) {
            const bool iskv = tile < nk / 64;
            const bool isq = !iskv;
            const uint64_t idx = (iskv ? tile : tile - nk / 64) * 64;
            const uint32_t seq = isq ? p.qp : p.kp, heads = isq ? p.hq : p.hk;
            const uint32_t token = idx % seq, h = (idx / seq) % heads, b = idx / seq / heads;
            const uint32_t limit = isq ? p.sq : p.sk;
            const uint32_t valid = token < limit ? Min(uint32_t(64), limit - token) : 0;
            for (uint32_t row = 0; row < valid; ++row)
                ReadRow(input[row * p.dp], isq ? qg : kg, Offset(isq ? p.qs : p.ks, b, h, token + row), p.d, p.dp);
            if (valid < 64)
                Duplicate(input[valid * p.dp], T(0), (64 - valid) * p.dp);
            Cast(x, input, RoundMode::CAST_NONE, 64 * p.dp);
            PipeBarrier<PIPE_V>();
            if (!isq && p.smoothK && valid) {
                DataCopy(mu, mg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                Sync<HardEvent::MTE2_V>();
            }
            QuantRowsVF((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)mu.GetPhyAddr(),
                        (__ubuf__ float *)scales.GetPhyAddr(), (__ubuf__ int8_t *)out.GetPhyAddr(),
                        (__ubuf__ half *)halfScales.GetPhyAddr(), isq ? p.qFactor : p.kFactor,
                        (__ubuf__ float *)inverseScales.GetPhyAddr(), p.dp, !isq && p.smoothK && valid, valid,
                        isq ? p.qg : p.kg);
            WriteRow(isq ? qo : ko, idx * p.dp, out, 64 * p.dp);
            WriteRow(isq ? qsg : ksg, idx, halfScales, 64);
        }

        PipeBarrier<PIPE_ALL>();
        GlobalTensor<uint16_t> mxg;
        mxg.SetGlobalBuffer((__gm__ uint16_t *)mxScale);
        auto vInput = r.ubBuf.template GetBufferByByte<T>(V_INPUT);
        auto maxima = r.ubBuf.template GetBufferByByte<T>(V_MAXIMA);
        auto vOut = r.ubBuf.template GetBufferByByte<float8_e4m3_t>(V_OUTPUT);
        auto mx = r.ubBuf.template GetBufferByByte<uint16_t>(MX_SCALE);
        auto vmean = r.ubBuf.template GetBufferByByte<float>(V_MEAN);

        for (uint64_t tile = GetBlockIdx(); tile < nk / 64; tile += 2 * p.cores) {
            const uint64_t idx = tile * 64;
            const uint32_t token = idx % p.kp, h = (idx / p.kp) % p.hk, b = idx / p.kp / p.hk;
            const uint32_t valid = token < p.sk ? Min(uint32_t(64), p.sk - token) : 0;
            if (p.d == p.dp && valid) {
                Sync<HardEvent::V_MTE2>();
                DataCopy(vInput, vg[Offset(p.vs, b, h, token)], valid * p.dp);
                Sync<HardEvent::MTE2_V>();
            } else {
                for (uint32_t row = 0; row < valid; ++row)
                    ReadRow(vInput[row * p.dp], vg, Offset(p.vs, b, h, token + row), p.d, p.dp);
            }
            if (valid < 64)
                Duplicate(vInput[valid * p.dp], T(0), (64 - valid) * p.dp);
            PipeBarrier<PIPE_V>();
            if (p.smoothV && valid) {
                DataCopy(vmean, vmg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                Sync<HardEvent::MTE2_V>();
                CenterVRowsVF<T>((__ubuf__ T *)vInput.GetPhyAddr(), (__ubuf__ float *)vmean.GetPhyAddr(), p.dp, valid);
                PipeBarrier<PIPE_V>();
            }
            QuantVRowsMxVF<T, WIDE>((__ubuf__ T *)vInput.GetPhyAddr(), (__ubuf__ T *)maxima.GetPhyAddr(),
                                    (__ubuf__ uint16_t *)mx.GetPhyAddr(),
                                    (__ubuf__ float8_e4m3_t *)vOut.GetPhyAddr(), p.dp);
            WriteRow(vo, idx * p.dp, vOut, 64 * p.dp);
            WriteRow(mxg, idx / 64 * p.dp, mx, p.dp);
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
using microkernel::QuantKernel;
using microkernel::StatsKernel;
using microkernel::FastQuant;
#endif
}

