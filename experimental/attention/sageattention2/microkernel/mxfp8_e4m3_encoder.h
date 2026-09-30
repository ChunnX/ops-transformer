#pragma once

#include "../core/platform_traits.h"
#include "../core/numerical_types.h"
#include "../core/math_approximations.h"
#include "../memory/ring_buffer_layout.h"
#include "../memory/catlass_fractal_packer.h"
#include "../orchestration/credit_handshake.h"

#if defined(__CCE__)
#include "kernel_operator.h"

namespace sage::microkernel {

using AscendC::RoundMode;
using memory::Sync;
using AscendC::HardEvent;

__aicore__ inline void InitMxP(memory::FixedResource &r) {
    using namespace AscendC;
    InitConstValueParams<int16_t> cp;
    cp.repeatTimes = 1;
    cp.blockNum = MX_P_BLOCK_NUM;
    cp.initValue = static_cast<int16_t>(PV_PARTIAL_E8M0_PAIR);
    cp.dstGap = 0;
    InitConstValue(r.l1Buf.GetBufferByByte<int16_t>(MX_P_L1), cp);
    Sync<HardEvent::MTE2_MTE1>();
    const uint64_t mxDst = reinterpret_cast<uint64_t>(r.l0ABuf.GetBufferByByte<float8_e4m3_t>(L0A_MX_P_OFFSET).GetPhyAddr()) / 16;
    load_cbuf_to_ca_mx(mxDst, (__cbuf__ void *)r.l1Buf.GetBufferByByte<float8_e8m0_t>(MX_P_L1).GetPhyAddr(), 0, 0,
                       MX_P_M_BLOCKS, MX_P_N_BLOCKS, MX_P_N_BLOCKS, MX_P_N_BLOCKS);
    Sync<HardEvent::MTE1_M>();
}

template <class T, bool WIDE>
__simd_vf__ inline void QuantVRowsMxVF(__ubuf__ T *x, __ubuf__ T *maxima, __ubuf__ uint16_t *scales,
                                       __ubuf__ float8_e4m3_t *out, uint32_t dp) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    uint32_t count = dp;
    auto h = UpdateMask<T>(count);
    RegTensor<uint16_t> sign;
    Duplicate(sign, FP16_ABS_MASK);

    for (uint16_t group = 0; group < 2; ++group) {
        RegTensor<T> mx0, mx1, mx2, mx3;
        Duplicate(mx0, T(0));
        Duplicate(mx1, T(0));
        Duplicate(mx2, T(0));
        Duplicate(mx3, T(0));
        __ubuf__ T *grpBase = x + (group * 32) * dp;
        for (uint16_t row = 0; row < 32; row += 4) {
            RegTensor<T> a0, a1, a2, a3;
            LoadAlign(a0, grpBase + (row + 0) * dp);
            LoadAlign(a1, grpBase + (row + 1) * dp);
            LoadAlign(a2, grpBase + (row + 2) * dp);
            LoadAlign(a3, grpBase + (row + 3) * dp);
            And((RegTensor<uint16_t> &)a0, (RegTensor<uint16_t> &)a0, sign, h);
            And((RegTensor<uint16_t> &)a1, (RegTensor<uint16_t> &)a1, sign, h);
            And((RegTensor<uint16_t> &)a2, (RegTensor<uint16_t> &)a2, sign, h);
            And((RegTensor<uint16_t> &)a3, (RegTensor<uint16_t> &)a3, sign, h);
            Max(mx0, mx0, a0, h);
            Max(mx1, mx1, a1, h);
            Max(mx2, mx2, a2, h);
            Max(mx3, mx3, a3, h);
        }
        Max(mx0, mx0, mx1, h);
        Max(mx2, mx2, mx3, h);
        Max(mx0, mx0, mx2, h);
        StoreAlign(maxima + group * dp, mx0, h);
    }

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait toFloat{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<uint32_t> expMask, expBias;
    Duplicate(expMask, EXPONENT_MASK);
    Duplicate(expBias, RECIPROCAL_EXP_BIAS);

    for (uint16_t col = 0; col < dp; col += 64) {
        RegTensor<uint32_t> pair;
        Duplicate(pair, uint32_t(0));
        for (uint16_t group = 0; group < 2; ++group) {
            RegTensor<float> mx, inv;
            RegTensor<T> mx16;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(mx16, maxima + group * dp + col);
            Cast<float, T, toFloat>(mx, mx16, f);
            Muls(mx, mx, INV_MX_FP8_MAX, f);
            Maxs(mx, mx, MIN_POSITIVE_FLOAT, f);

            RegTensor<uint32_t> bits, rounded, exponent, inverseBits;
            Adds(bits, (RegTensor<uint32_t> &)mx, EXPONENT_ROUND_BIAS, f);
            And(rounded, bits, expMask, f);
            ShiftRights(exponent, rounded, int16_t(23), f);
            if (group == 1)
                ShiftLefts(exponent, exponent, int16_t(8), f);
            Or(pair, pair, exponent, f);

            Sub(inverseBits, expBias, rounded, f);
            inv = (RegTensor<float> &)inverseBits;

            __ubuf__ T *xBase0 = x + (group * 32) * dp + col;
            __ubuf__ T *xBase1 = x + (group * 32 + 16) * dp + col;

            if constexpr (WIDE) {
                __ubuf__ float8_e4m3_t *outBase = out + (group * 32) * dp + col;
                for (uint16_t row = 0; row < 16; row += 2) {
                    RegTensor<T> x0, x1, x2, x3;
                    RegTensor<float> a0, a1, a2, a3;
                    RegTensor<float8_e4m3_t> p0, p1, p2, p3;
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x0, xBase0 + row * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x1, xBase1 + row * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x2, xBase0 + (row + 1) * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x3, xBase1 + (row + 1) * dp);
                    Cast<float, T, toFloat>(a0, x0, f);
                    Cast<float, T, toFloat>(a1, x1, f);
                    Cast<float, T, toFloat>(a2, x2, f);
                    Cast<float, T, toFloat>(a3, x3, f);
                    Mul(a0, a0, inv, f);
                    Mul(a1, a1, inv, f);
                    Mul(a2, a2, inv, f);
                    Mul(a3, a3, inv, f);
                    Cast<float8_e4m3_t, float, cast>(p0, a0, f);
                    Cast<float8_e4m3_t, float, cast>(p1, a1, f);
                    Cast<float8_e4m3_t, float, cast>(p2, a2, f);
                    Cast<float8_e4m3_t, float, cast>(p3, a3, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase + (row * 2) * dp, p0, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase + (row * 2 + 1) * dp, p1, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase + ((row + 1) * 2) * dp, p2, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase + ((row + 1) * 2 + 1) * dp, p3, f);
                }
            } else {
                __ubuf__ float8_e4m3_t *outBase0 = out + (group * 32) * dp + col;
                __ubuf__ float8_e4m3_t *outBase1 = out + (group * 32 + 16) * dp + col;
                for (uint16_t row = 0; row < 16; row += 2) {
                    RegTensor<T> x0, x1, x2, x3;
                    RegTensor<float> a0, a1, a2, a3;
                    RegTensor<float8_e4m3_t> p0, p1, p2, p3;
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x0, xBase0 + row * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x1, xBase1 + row * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x2, xBase0 + (row + 1) * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x3, xBase1 + (row + 1) * dp);
                    Cast<float, T, toFloat>(a0, x0, f);
                    Cast<float, T, toFloat>(a1, x1, f);
                    Cast<float, T, toFloat>(a2, x2, f);
                    Cast<float, T, toFloat>(a3, x3, f);
                    Mul(a0, a0, inv, f);
                    Mul(a1, a1, inv, f);
                    Mul(a2, a2, inv, f);
                    Mul(a3, a3, inv, f);
                    Cast<float8_e4m3_t, float, cast>(p0, a0, f);
                    Cast<float8_e4m3_t, float, cast>(p1, a1, f);
                    Cast<float8_e4m3_t, float, cast>(p2, a2, f);
                    Cast<float8_e4m3_t, float, cast>(p3, a3, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase0 + row * dp, p0, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase1 + row * dp, p1, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase0 + (row + 1) * dp, p2, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase1 + (row + 1) * dp, p3, f);
                }
            }
        }
        StoreAlign<uint16_t, StoreDist::DIST_PACK_B32>(scales + col, (RegTensor<uint16_t> &)pair, f);
    }
}

} // namespace sage::microkernel

namespace sage {
using microkernel::InitMxP;
using microkernel::QuantVRowsMxVF;
}

#endif // __CCE__
