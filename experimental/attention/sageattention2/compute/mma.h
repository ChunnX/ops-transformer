/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../common/constants.h"
#include "../common/types.h"
#include "../common/math.h"
#include "../hardware/memory.h"
#include "../hardware/sync.h"
#include "../hardware/layout.h"

#if defined(__CCE__)
#include "kernel_operator.h"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#endif

namespace sage::compute {

#if defined(__CCE__)
using AscendC::RoundMode;

struct MxScaleCache {
    GM_ADDR base = nullptr;
};

// ============================================================================
// Initialize Constant P-Scale (2^-8) into L0A for MXFP8 PV MMA
// ============================================================================
__aicore__ inline void InitMxP(hardware::FixedResource &r) {
    using namespace AscendC;
    InitConstValueParams<int16_t> cp;
    cp.repeatTimes = 1;
    cp.blockNum = hardware::MX_P_BLOCK_NUM;
    cp.initValue = static_cast<int16_t>(PV_PARTIAL_E8M0_PAIR);
    cp.dstGap = 0;
    InitConstValue(r.l1Buf.GetBufferByByte<int16_t>(MX_P_L1), cp);
    Sync<HardEvent::MTE2_MTE1>();
    const uint64_t mxDst = reinterpret_cast<uint64_t>(r.l0ABuf.GetBufferByByte<float8_e4m3_t>(hardware::L0A_MX_P_OFFSET).GetPhyAddr()) / 16;
    load_cbuf_to_ca_mx(mxDst, (__cbuf__ void *)r.l1Buf.GetBufferByByte<float8_e8m0_t>(MX_P_L1).GetPhyAddr(), 0, 0,
                       hardware::MX_P_M_BLOCKS, hardware::MX_P_N_BLOCKS, hardware::MX_P_N_BLOCKS, hardware::MX_P_N_BLOCKS);
    Sync<HardEvent::MTE1_M>();
}

// ============================================================================
// Tile-wise MTE2/MTE1 Prefetch Primitives
// ============================================================================

template <uint32_t BOffset>
__aicore__ inline __attribute__((always_inline)) void PrefetchWideK(hardware::FixedResource &r, GM_ADDR b, uint32_t parity) {
    using namespace AscendC;
    using namespace Catlass;
    using namespace tla;
    using T = int8_t;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<hardware::Arch, T, layout::RowMajor, T, layout::ColumnMajor, float, layout::RowMajor>;
    event_t ev = static_cast<event_t>(4 + parity);
    WaitFlag<HardEvent::MTE1_MTE2>(ev);
    GlobalTensor<T> bg;
    bg.SetGlobalBuffer((__gm__ T *)b);
    auto gb = tla::MakeTensor(bg, tla::MakeLayout<T, layout::ColumnMajor>(128, N), Catlass::Arch::PositionGM{});
    auto lb = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(BOffset + parity * 65536),
                             tla::MakeLayout<T, typename Copy::LayoutTagL1B>(128, N), Catlass::Arch::PositionL1{});
    typename Copy::template CopyGmToL1B<decltype(gb)> copy;
    copy(lb, gb);
    SetFlag<HardEvent::MTE2_MTE1>(ev);
}

template <uint32_t BOffset>
__aicore__ inline __attribute__((always_inline)) void PrefetchWideV(hardware::FixedResource &r, GM_ADDR b, GM_ADDR vScale, uint32_t parity) {
    using namespace AscendC;
    using namespace Catlass;
    using namespace tla;
    using T = float8_e4m3_t;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<hardware::Arch, T, layout::RowMajor, T, layout::RowMajor, float, layout::RowMajor>;
    event_t ev = static_cast<event_t>(6 + (parity & 1));
    WaitFlag<HardEvent::MTE1_MTE2>(ev);
    GlobalTensor<T> bg;
    bg.SetGlobalBuffer((__gm__ T *)b);
    auto gb = tla::MakeTensor(bg, tla::MakeLayout<T, layout::RowMajor>(N, 128), Catlass::Arch::PositionGM{});
    auto lb = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(BOffset + N * 128 + (parity & 1) * 65536),
                             tla::MakeLayout<T, typename Copy::LayoutTagL1B>(N, 128), Catlass::Arch::PositionL1{});
    typename Copy::template CopyGmToL1B<decltype(gb)> copy;
    copy(lb, gb);

    // Tile-wise Co-streaming V-scale slice into ping-pong buffer: net releases 174 KB L1
    if (vScale) {
        uint32_t scaleOffset = (parity & 1) ? V_SCALE_L1_PONG : V_SCALE_L1_PING;
        GlobalTensor<float8_e8m0_t> sg;
        sg.SetGlobalBuffer((__gm__ float8_e8m0_t *)vScale);
        auto sb = tla::MakeTensor(r.l1Buf.template GetBufferByByte<float8_e8m0_t>(scaleOffset),
                                  tla::MakeMxScaleLayout<float8_e8m0_t, layout::nN, true>(N / 32, 128),
                                  tla::MakeCoord(0, 0), Catlass::Arch::PositionL1{});
        auto gs = tla::MakeTensor(sg, tla::MakeMxScaleLayout<float8_e8m0_t, layout::RowMajor, true>(N / 32, 128),
                                  Catlass::Arch::PositionGM{});
        Catlass::Gemm::Tile::TileCopyTla<hardware::Arch, decltype(gs), decltype(sb)> copyScale;
        copyScale(sb, gs);
    }
    SetFlag<HardEvent::MTE2_MTE1>(ev);
}

// ============================================================================
// Split-M GEMM-1 (Q * K^T) on Cube
// ============================================================================

template <uint32_t BOffset>
__aicore__ inline __attribute__((always_inline)) void GemmQkSplit(hardware::FixedResource &r,
                                                                  uint32_t ubOffset, bool loadA, uint32_t kParity) {
    using namespace AscendC;
    using namespace Catlass;
    using namespace tla;
    constexpr uint32_t k = 128, n = N;
    using T = int8_t;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<hardware::Arch, T, layout::RowMajor, T, layout::ColumnMajor, float, layout::RowMajor>;

    WaitFlag<HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
    auto la = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(0),
                              tla::MakeLayout<T, typename Copy::LayoutTagL1A>(M, k), Catlass::Arch::PositionL1{});
    auto lb = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(BOffset + kParity * 65536),
                              tla::MakeLayout<T, typename Copy::LayoutTagL1B>(k, n), Catlass::Arch::PositionL1{});
    if (loadA) {
        EventHandshake::WaitFromBothAiv<PIPE_MTE1>(TokenSignal::Q_READY);
    }
    event_t kEvent = static_cast<event_t>(4 + kParity);
    WaitFlag<HardEvent::MTE2_MTE1>(kEvent);
    WaitFlag<HardEvent::M_MTE1>(EventHandshake::QK_EVENT);
    auto bb = tla::MakeTensor(r.l0BBuf.template GetBufferByByte<T>(0),
                              tla::MakeLayout<T, typename Copy::LayoutTagL0B>(k, n), Catlass::Arch::PositionL0B{});
    typename Copy::CopyL1ToL0B c0b;
    c0b(bb, tla::GetTile(lb, tla::MakeCoord(0, 0), tla::MakeShape(k, n)));

    #pragma unroll
    for (uint32_t sub = 0; sub < 2; ++sub) {
        auto aa = tla::MakeTensor(r.l0ABuf.template GetBufferByByte<T>(sub * HALF * k),
                                  tla::MakeLayout<T, typename Copy::LayoutTagL0A>(HALF, k), Catlass::Arch::PositionL0A{});
        if (loadA) {
            typename Copy::CopyL1ToL0A c0a;
            c0a(aa, tla::GetTile(la, tla::MakeCoord(sub * HALF, 0), tla::MakeShape(HALF, k)));
        }
    }
    SetFlag<HardEvent::MTE1_MTE2>(kEvent);
    SetFlag<HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
    Sync<HardEvent::MTE1_M>();

    #pragma unroll
    for (uint32_t sub = 0; sub < 2; ++sub) {
        event_t ev = static_cast<event_t>(sub ? EventHandshake::QK_SECOND_FIX_EVENT : EventHandshake::QK_EVENT);
        WaitFlag<HardEvent::FIX_M>(ev);
        auto aa = tla::MakeTensor(r.l0ABuf.template GetBufferByByte<T>(sub * HALF * k),
                                  tla::MakeLayout<T, typename Copy::LayoutTagL0A>(HALF, k), Catlass::Arch::PositionL0A{});
        auto cc = tla::MakeTensor(r.l0CBuf.template GetBufferByByte<int32_t>(sub * HALF * n * 4),
                                  tla::MakeLayout<int32_t, layout::L0C>(HALF, n), Catlass::Arch::PositionL0C{});
        Catlass::Gemm::Tile::TileMmadTla<hardware::Arch, T, typename Copy::LayoutTagL1A> mma;
        mma(cc, aa, bb, HALF, n, k, true);
        SetFlag<HardEvent::M_FIX>(ev);
        WaitFlag<HardEvent::M_FIX>(ev);

        static constexpr FixpipeConfig cfg{CO2Layout::NZ, true};
        FixpipeParamsArch3510<CO2Layout::NZ> fp;
        fp.mSize = HALF;
        fp.nSize = n;
        fp.srcStride = HALF;
        fp.dstStride = HALF * 16;
        fp.quantPre = QuantMode_t::DEQF16;
        fp.deqScalar = DEQ_SCALAR_DEQF16;
        fp.dualDstCtl = 0;
        fp.subBlockId = sub != 0;

        auto dst = r.ubBuf.template GetBufferByByte<half>(ubOffset);
        auto src = r.l0CBuf.template GetBufferByByte<int32_t>(sub * HALF * n * 4);
        Fixpipe<half, int32_t, cfg>(dst, src, fp);
        SetFlag<HardEvent::FIX_M>(ev);
        EventHandshake::NotifyAiv<PIPE_FIX>(TokenSignal::QK_READY, sub);
    }
}

// ============================================================================
// Cube GEMM-2 (P * V) with Delayed FP32 Buffering
// ============================================================================

template <class T, class BLayout, bool LocalA = false, uint32_t BOffset = WIDE_B_L1, bool Emit = true,
          bool Initialize = true, bool PreloadedB = false>
__aicore__ inline __attribute__((always_inline)) void Gemm(hardware::FixedResource &r, GM_ADDR a, GM_ADDR b, uint32_t k,
                                                           uint32_t n, uint32_t ubOffset, uint32_t parity = 0,
                                                           bool loadA = true, GM_ADDR mxScale = nullptr,
                                                           MxScaleCache *mxCache = nullptr, uint32_t mxK = 0,
                                                           uint32_t mxStart = 0, uint32_t vParity = 0) {
    using namespace AscendC;
    using namespace Catlass;
    using namespace tla;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<hardware::Arch, T, layout::RowMajor, T, BLayout, float, layout::RowMajor>;
    using Acc = typename Copy::ElementAccumulator;
    using L0T = std::conditional_t<LocalA, AscendC::mx_fp8_e4m3_t, T>;

    constexpr event_t slot = static_cast<event_t>(LocalA ? EventHandshake::PV_EVENT : EventHandshake::QK_EVENT);
    constexpr uint32_t l0Offset = LocalA ? 32768 : 0;
    constexpr uint32_t l0cOffset = LocalA ? 131072 : 0;

    WaitFlag<HardEvent::MTE1_MTE2>(slot);
    auto la = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(LocalA ? (BOffset == WIDE_B_L1 ? 0 : P_L1) + parity * M * N : 0),
                              tla::MakeLayout<T, typename Copy::LayoutTagL1A>(M, k), Catlass::Arch::PositionL1{});
    auto lb = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(BOffset + (LocalA ? N * 128 : 0) + (PreloadedB ? vParity * 65536 : 0)),
                              tla::MakeLayout<T, typename Copy::LayoutTagL1B>(k, n), Catlass::Arch::PositionL1{});

    const bool cached = mxCache && uint64_t(mxK) / 32 * n <= MX_P_L1 - MX_V_L1;
    const uint32_t scaleK = cached ? mxK : k;
    const uint32_t scaleStart = cached ? mxStart : 0;
    GM_ADDR scaleBase = cached ? mxScale - uint64_t(mxStart) / 32 * n : mxScale;
    const uint32_t scaleOffset = (PreloadedB ? ((vParity & 1) ? V_SCALE_L1_PONG : V_SCALE_L1_PING) : MX_V_L1);
    auto sb = tla::MakeTensor(r.l1Buf.GetBufferByByte<float8_e8m0_t>(scaleOffset),
                              tla::MakeMxScaleLayout<float8_e8m0_t, layout::nN, true>(PreloadedB ? k / 32 : scaleK / 32, n),
                              tla::MakeCoord(0, 0), Catlass::Arch::PositionL1{});

    if constexpr (LocalA) {
        if constexpr (!PreloadedB) {
            if (!cached || mxCache->base != scaleBase) {
                GlobalTensor<float8_e8m0_t> sg;
                sg.SetGlobalBuffer((__gm__ float8_e8m0_t *)scaleBase);
                auto gs = tla::MakeTensor(sg, tla::MakeMxScaleLayout<float8_e8m0_t, layout::RowMajor, true>(scaleK / 32, n),
                                          Catlass::Arch::PositionGM{});
                Catlass::Gemm::Tile::TileCopyTla<hardware::Arch, decltype(gs), decltype(sb)> copyScale;
                copyScale(sb, gs);
                if (cached) mxCache->base = scaleBase;
            }
        }
    }

    if constexpr (PreloadedB)
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(6 + (vParity & 1)));
    else
        Sync<HardEvent::MTE2_MTE1>();

    WaitFlag<HardEvent::M_MTE1>(slot);
    auto aa = tla::MakeTensor(r.l0ABuf.template GetBufferByByte<L0T>(l0Offset),
                              tla::MakeLayout<T, typename Copy::LayoutTagL0A>(M, k), Catlass::Arch::PositionL0A{});
    auto bb = tla::MakeTensor(r.l0BBuf.template GetBufferByByte<L0T>(l0Offset),
                              tla::MakeLayout<T, typename Copy::LayoutTagL0B>(k, n), Catlass::Arch::PositionL0B{});
    typename Copy::CopyL1ToL0A c0a;
    typename Copy::CopyL1ToL0B c0b;

    if constexpr (LocalA) {
        auto aaData = tla::MakeTensor(aa.data().template ReinterpretCast<T>(), aa.layout(), Catlass::Arch::PositionL0A{});
        c0a(aaData, tla::GetTile(la, tla::MakeCoord(0, 0), tla::MakeShape(M, k)));
        c0b(bb, tla::GetTile(lb, tla::MakeCoord(0, 0), tla::MakeShape(k, n)),
            tla::GetTile(sb, tla::MakeCoord(PreloadedB ? 0 : scaleStart / 32, 0), tla::MakeShape(k / 32, n)));
    } else {
        if (loadA) c0a(aa, tla::GetTile(la, tla::MakeCoord(0, 0), tla::MakeShape(M, k)));
        c0b(bb, tla::GetTile(lb, tla::MakeCoord(0, 0), tla::MakeShape(k, n)));
    }

    if constexpr (PreloadedB) SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(6 + (vParity & 1)));
    SetFlag<HardEvent::MTE1_MTE2>(slot);
    Sync<HardEvent::MTE1_M>();

    WaitFlag<HardEvent::FIX_M>(slot);
    auto cc = tla::MakeTensor(r.l0CBuf.template GetBufferByByte<Acc>(l0cOffset),
                              tla::MakeLayout<Acc, layout::L0C>(M, n), Catlass::Arch::PositionL0C{});
    Catlass::Gemm::Tile::TileMmadTla<hardware::Arch, T, typename Copy::LayoutTagL1A> mma;
    mma(cc, aa, bb, M, n, k, Initialize);
    SetFlag<HardEvent::M_MTE1>(slot);
    Sync<HardEvent::M_FIX>();

    if constexpr (!Emit) {
        // Retain first PV partial in L0C
    } else if constexpr (std::is_same_v<T, int8_t>) {
        static constexpr FixpipeConfig cfg{CO2Layout::NZ, true};
        FixpipeParamsArch3510<CO2Layout::NZ> fp;
        fp.mSize = M;
        fp.nSize = n;
        fp.srcStride = M;
        fp.dstStride = M * 16;
        fp.quantPre = QuantMode_t::DEQF16;
        fp.deqScalar = DEQ_SCALAR_DEQF16;
        fp.dualDstCtl = 0;
        fp.subBlockId = 0;

        auto dst = r.ubBuf.template GetBufferByByte<half>(ubOffset);
        auto src = r.l0CBuf.template GetBufferByByte<int32_t>(l0cOffset);
        Fixpipe<half, int32_t, cfg>(dst, src, fp);
    } else {
        auto u = tla::MakeTensor(r.ubBuf.template GetBufferByByte<float>(ubOffset),
                                 tla::MakeLayout<float, layout::RowMajor>(M, n), Catlass::Arch::PositionUB{});
        Catlass::Gemm::Tile::CopyL0CToUBTla<hardware::Arch, decltype(cc), decltype(u),
                                            Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M> copy;
        copy(u, cc);
    }
    SetFlag<HardEvent::FIX_M>(slot);
}

// ============================================================================
// Output Accumulators & Normalization
// ============================================================================

template <uint32_t DP>
__simd_vf__ inline void AccumulateFp32VF(__ubuf__ float *accum, __ubuf__ float *partial,
                                         __ubuf__ float *alpha) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();

    if constexpr (DP == 128) {
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> ratio0, ratio1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, alpha + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, alpha + row + 1);

            RegTensor<float> x0_0, x1_0, x0_1, x1_1;
            RegTensor<float> y0_0, y1_0, y0_1, y1_1;

            LoadAlign(x0_0, accum + row * DP);
            LoadAlign(x1_0, accum + row * DP + 64);
            LoadAlign(x0_1, accum + (row + 1) * DP);
            LoadAlign(x1_1, accum + (row + 1) * DP + 64);

            LoadAlign(y0_0, partial + row * DP);
            LoadAlign(y1_0, partial + row * DP + 64);
            LoadAlign(y0_1, partial + (row + 1) * DP);
            LoadAlign(y1_1, partial + (row + 1) * DP + 64);

            MulDstAdd(x0_0, ratio0, y0_0, f);
            MulDstAdd(x1_0, ratio0, y1_0, f);
            MulDstAdd(x0_1, ratio1, y0_1, f);
            MulDstAdd(x1_1, ratio1, y1_1, f);

            StoreAlign(accum + row * DP, x0_0, f);
            StoreAlign(accum + row * DP + 64, x1_0, f);
            StoreAlign(accum + (row + 1) * DP, x0_1, f);
            StoreAlign(accum + (row + 1) * DP + 64, x1_1, f);
        }
    } else {
        for (uint16_t row = 0; row < 64; row += 4) {
            RegTensor<float> ratio0, ratio1, ratio2, ratio3;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, alpha + row + 0);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, alpha + row + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio2, alpha + row + 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio3, alpha + row + 3);

            RegTensor<float> x0, x1, x2, x3;
            RegTensor<float> y0, y1, y2, y3;
            LoadAlign(x0, accum + (row + 0) * DP);
            LoadAlign(x1, accum + (row + 1) * DP);
            LoadAlign(x2, accum + (row + 2) * DP);
            LoadAlign(x3, accum + (row + 3) * DP);

            LoadAlign(y0, partial + (row + 0) * DP);
            LoadAlign(y1, partial + (row + 1) * DP);
            LoadAlign(y2, partial + (row + 2) * DP);
            LoadAlign(y3, partial + (row + 3) * DP);

            MulDstAdd(x0, ratio0, y0, f);
            MulDstAdd(x1, ratio1, y1, f);
            MulDstAdd(x2, ratio2, y2, f);
            MulDstAdd(x3, ratio3, y3, f);

            StoreAlign(accum + (row + 0) * DP, x0, f);
            StoreAlign(accum + (row + 1) * DP, x1, f);
            StoreAlign(accum + (row + 2) * DP, x2, f);
            StoreAlign(accum + (row + 3) * DP, x3, f);
        }
    }
}

template <uint32_t DP>
__simd_vf__ inline void AccumulateFp32PairVF(__ubuf__ float *accum, __ubuf__ float *partial0,
                                             __ubuf__ float *partial1, __ubuf__ float *alpha0,
                                             __ubuf__ float *alpha1) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();

    if constexpr (DP == 128) {
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> a0_0, a0_1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_0, alpha0 + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_1, alpha0 + row + 1);

            RegTensor<float> x0_0, x1_0, x0_1, x1_1;
            RegTensor<float> y0_0, y1_0, y0_1, y1_1;
            LoadAlign(x0_0, accum + row * DP);
            LoadAlign(x1_0, accum + row * DP + 64);
            LoadAlign(x0_1, accum + (row + 1) * DP);
            LoadAlign(x1_1, accum + (row + 1) * DP + 64);

            LoadAlign(y0_0, partial0 + row * DP);
            LoadAlign(y1_0, partial0 + row * DP + 64);
            LoadAlign(y0_1, partial0 + (row + 1) * DP);
            LoadAlign(y1_1, partial0 + (row + 1) * DP + 64);

            MulDstAdd(x0_0, a0_0, y0_0, f);
            MulDstAdd(x1_0, a0_0, y1_0, f);
            MulDstAdd(x0_1, a0_1, y0_1, f);
            MulDstAdd(x1_1, a0_1, y1_1, f);

            RegTensor<float> a1_0, a1_1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_0, alpha1 + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_1, alpha1 + row + 1);

            RegTensor<float> z0_0, z1_0, z0_1, z1_1;
            LoadAlign(z0_0, partial1 + row * DP);
            LoadAlign(z1_0, partial1 + row * DP + 64);
            LoadAlign(z0_1, partial1 + (row + 1) * DP);
            LoadAlign(z1_1, partial1 + (row + 1) * DP + 64);

            MulDstAdd(x0_0, a1_0, z0_0, f);
            MulDstAdd(x1_0, a1_0, z1_0, f);
            MulDstAdd(x0_1, a1_1, z0_1, f);
            MulDstAdd(x1_1, a1_1, z1_1, f);

            StoreAlign(accum + row * DP, x0_0, f);
            StoreAlign(accum + row * DP + 64, x1_0, f);
            StoreAlign(accum + (row + 1) * DP, x0_1, f);
            StoreAlign(accum + (row + 1) * DP + 64, x1_1, f);
        }
    } else {
        for (uint16_t row = 0; row < 64; row += 4) {
            RegTensor<float> a0_0, a0_1, a0_2, a0_3;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_0, alpha0 + row + 0);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_1, alpha0 + row + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_2, alpha0 + row + 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_3, alpha0 + row + 3);

            RegTensor<float> x0, x1, x2, x3;
            RegTensor<float> y0, y1, y2, y3;
            LoadAlign(x0, accum + (row + 0) * DP);
            LoadAlign(x1, accum + (row + 1) * DP);
            LoadAlign(x2, accum + (row + 2) * DP);
            LoadAlign(x3, accum + (row + 3) * DP);

            LoadAlign(y0, partial0 + (row + 0) * DP);
            LoadAlign(y1, partial0 + (row + 1) * DP);
            LoadAlign(y2, partial0 + (row + 2) * DP);
            LoadAlign(y3, partial0 + (row + 3) * DP);

            MulDstAdd(x0, a0_0, y0, f);
            MulDstAdd(x1, a0_1, y1, f);
            MulDstAdd(x2, a0_2, y2, f);
            MulDstAdd(x3, a0_3, y3, f);

            RegTensor<float> a1_0, a1_1, a1_2, a1_3;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_0, alpha1 + row + 0);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_1, alpha1 + row + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_2, alpha1 + row + 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_3, alpha1 + row + 3);

            RegTensor<float> z0, z1, z2, z3;
            LoadAlign(z0, partial1 + (row + 0) * DP);
            LoadAlign(z1, partial1 + (row + 1) * DP);
            LoadAlign(z2, partial1 + (row + 2) * DP);
            LoadAlign(z3, partial1 + (row + 3) * DP);

            MulDstAdd(x0, a1_0, z0, f);
            MulDstAdd(x1, a1_1, z1, f);
            MulDstAdd(x2, a1_2, z2, f);
            MulDstAdd(x3, a1_3, z3, f);

            StoreAlign(accum + (row + 0) * DP, x0, f);
            StoreAlign(accum + (row + 1) * DP, x1, f);
            StoreAlign(accum + (row + 2) * DP, x2, f);
            StoreAlign(accum + (row + 3) * DP, x3, f);
        }
    }
}

// Round per-block PV partial to FP16 for fp32+fp16 accumulation
__simd_vf__ inline void RoundPartialFp16VF(__ubuf__ float *partial, uint32_t count) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait narrow{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait widen{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    for (uint32_t i = 0; i < count; i += 128) {
        RegTensor<float> x0, x1;
        RegTensor<half> packed0, packed1;
        LoadAlign(x0, partial + i);
        LoadAlign(x1, partial + i + 64);
        Cast<half, float, narrow>(packed0, x0, f);
        Cast<half, float, narrow>(packed1, x1, f);
        Cast<float, half, widen>(x0, packed0, f);
        Cast<float, half, widen>(x1, packed1, f);
        StoreAlign(partial + i, x0, f);
        StoreAlign(partial + i + 64, x1, f);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

// Normalize and cast output (standard path)
template <class T, uint32_t DP>
__simd_vf__ inline void NormalizeCastVF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *denom,
                                        __ubuf__ float *scratch, __ubuf__ float *meanV) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> den, inv;
    LoadAlign(den, denom);
    math::FastNormalizerReciprocal(inv, den, f);
    StoreAlign(scratch, inv, f);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    if constexpr (DP == 128) {
        RegTensor<float> mv0, mv1;
        LoadAlign(mv0, meanV);
        LoadAlign(mv1, meanV + 64);
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> ratio0, ratio1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, scratch + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, scratch + row + 1);

            RegTensor<float> x0_0, x1_0, x0_1, x1_1;
            LoadAlign(x0_0, accum + row * DP);
            LoadAlign(x1_0, accum + row * DP + 64);
            LoadAlign(x0_1, accum + (row + 1) * DP);
            LoadAlign(x1_1, accum + (row + 1) * DP + 64);

            Mul(x0_0, x0_0, ratio0, f);
            Mul(x1_0, x1_0, ratio0, f);
            Mul(x0_1, x0_1, ratio1, f);
            Mul(x1_1, x1_1, ratio1, f);

            Add(x0_0, x0_0, mv0, f);
            Add(x1_0, x1_0, mv1, f);
            Add(x0_1, x0_1, mv0, f);
            Add(x1_1, x1_1, mv1, f);

            RegTensor<T> p0_0, p1_0, p0_1, p1_1;
            Cast<T, float, cast>(p0_0, x0_0, f);
            Cast<T, float, cast>(p1_0, x1_0, f);
            Cast<T, float, cast>(p0_1, x0_1, f);
            Cast<T, float, cast>(p1_1, x1_1, f);

            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP, p0_0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP + 64, p1_0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP, p0_1, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP + 64, p1_1, f);
        }
    } else {
        RegTensor<float> mv;
        LoadAlign(mv, meanV);
        for (uint16_t row = 0; row < 64; row += 4) {
            RegTensor<float> ratio0, ratio1, ratio2, ratio3;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, scratch + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, scratch + row + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio2, scratch + row + 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio3, scratch + row + 3);

            RegTensor<float> x0, x1, x2, x3;
            LoadAlign(x0, accum + row * DP);
            LoadAlign(x1, accum + (row + 1) * DP);
            LoadAlign(x2, accum + (row + 2) * DP);
            LoadAlign(x3, accum + (row + 3) * DP);

            Mul(x0, x0, ratio0, f);
            Mul(x1, x1, ratio1, f);
            Mul(x2, x2, ratio2, f);
            Mul(x3, x3, ratio3, f);

            Add(x0, x0, mv, f);
            Add(x1, x1, mv, f);
            Add(x2, x2, mv, f);
            Add(x3, x3, mv, f);

            RegTensor<T> packed0, packed1, packed2, packed3;
            Cast<T, float, cast>(packed0, x0, f);
            Cast<T, float, cast>(packed1, x1, f);
            Cast<T, float, cast>(packed2, x2, f);
            Cast<T, float, cast>(packed3, x3, f);

            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP, packed0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP, packed1, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 2) * DP, packed2, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 3) * DP, packed3, f);
        }
    }
}

template <class T, uint32_t DP, bool FIRST>
__simd_vf__ inline void FinalizeWideFp32VF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial,
                                           __ubuf__ float *alpha, __ubuf__ float *denom, __ubuf__ float *scratch,
                                           __ubuf__ float *meanV) {
    static_assert(DP == 128, "FinalizeWideFp32VF is specialized for DP == 128");
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> den, inv;
    LoadAlign(den, denom);
    math::FastNormalizerReciprocal(inv, den, f);
    StoreAlign(scratch, inv, f);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<float> mv0, mv1;
    LoadAlign(mv0, meanV);
    LoadAlign(mv1, meanV + 64);
    for (uint16_t row = 0; row < 64; row += 2) {
        RegTensor<float> r0, r1;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(r0, scratch + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(r1, scratch + row + 1);

        RegTensor<float> x0_0, x1_0, x0_1, x1_1;
        RegTensor<float> y0_0, y1_0, y0_1, y1_1;
        LoadAlign(x0_0, accum + row * DP);
        LoadAlign(x1_0, accum + row * DP + 64);
        LoadAlign(x0_1, accum + (row + 1) * DP);
        LoadAlign(x1_1, accum + (row + 1) * DP + 64);

        LoadAlign(y0_0, partial + row * DP);
        LoadAlign(y1_0, partial + row * DP + 64);
        LoadAlign(y0_1, partial + (row + 1) * DP);
        LoadAlign(y1_1, partial + (row + 1) * DP + 64);

        if constexpr (!FIRST) {
            RegTensor<float> a0, a1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0, alpha + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1, alpha + row + 1);
            MulDstAdd(y0_0, a0, x0_0, f);
            MulDstAdd(y1_0, a0, x1_0, f);
            MulDstAdd(y0_1, a1, x0_1, f);
            MulDstAdd(y1_1, a1, x1_1, f);
        }

        Mul(y0_0, y0_0, r0, f);
        Mul(y1_0, y1_0, r0, f);
        Mul(y0_1, y0_1, r1, f);
        Mul(y1_1, y1_1, r1, f);

        Add(y0_0, y0_0, mv0, f);
        Add(y1_0, y1_0, mv1, f);
        Add(y0_1, y0_1, mv0, f);
        Add(y1_1, y1_1, mv1, f);

        RegTensor<T> p0_0, p1_0, p0_1, p1_1;
        Cast<T, float, cast>(p0_0, y0_0, f);
        Cast<T, float, cast>(p1_0, y1_0, f);
        Cast<T, float, cast>(p0_1, y0_1, f);
        Cast<T, float, cast>(p1_1, y1_1, f);

        StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP, p0_0, f);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP + 64, p1_0, f);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP, p0_1, f);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP + 64, p1_1, f);
    }
}

template <class T, uint32_t DP, bool FIRST, bool LAST>
__simd_vf__ inline void UpdateWidePairVF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial0,
                                         __ubuf__ float *partial1, __ubuf__ float *alpha0, __ubuf__ float *alpha1,
                                         __ubuf__ float *denom, __ubuf__ float *scratch, __ubuf__ float *meanV) {
    static_assert(DP == 128, "UpdateWidePairVF is specialized for DP == 128");
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    if constexpr (LAST) {
        RegTensor<float> den, inv;
        LoadAlign(den, denom);
        math::FastNormalizerReciprocal(inv, den, f);
        StoreAlign(scratch, inv, f);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<float> mv0, mv1;
    if constexpr (LAST) {
        LoadAlign(mv0, meanV);
        LoadAlign(mv1, meanV + 64);
    }
    for (uint16_t row = 0; row < 64; row += 2) {
        RegTensor<float> a0_0, a0_1;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_0, alpha0 + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_1, alpha0 + row + 1);

        RegTensor<float> a1_0, a1_1;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_0, alpha1 + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_1, alpha1 + row + 1);

        RegTensor<float> y0_0, y1_0, y0_1, y1_1;
        RegTensor<float> z0_0, z1_0, z0_1, z1_1;
        LoadAlign(y0_0, partial0 + row * DP);
        LoadAlign(y1_0, partial0 + row * DP + 64);
        LoadAlign(y0_1, partial0 + (row + 1) * DP);
        LoadAlign(y1_1, partial0 + (row + 1) * DP + 64);

        LoadAlign(z0_0, partial1 + row * DP);
        LoadAlign(z1_0, partial1 + row * DP + 64);
        LoadAlign(z0_1, partial1 + (row + 1) * DP);
        LoadAlign(z1_1, partial1 + (row + 1) * DP + 64);

        if constexpr (FIRST) {
            MulDstAdd(z0_0, a1_0, y0_0, f);
            MulDstAdd(z1_0, a1_0, y1_0, f);
            MulDstAdd(z0_1, a1_1, y0_1, f);
            MulDstAdd(z1_1, a1_1, y1_1, f);
        } else {
            RegTensor<float> x0_0, x1_0, x0_1, x1_1;
            LoadAlign(x0_0, accum + row * DP);
            LoadAlign(x1_0, accum + row * DP + 64);
            LoadAlign(x0_1, accum + (row + 1) * DP);
            LoadAlign(x1_1, accum + (row + 1) * DP + 64);

            MulDstAdd(y0_0, a0_0, x0_0, f);
            MulDstAdd(y1_0, a0_0, x1_0, f);
            MulDstAdd(y0_1, a0_1, x0_1, f);
            MulDstAdd(y1_1, a0_1, x1_1, f);

            MulDstAdd(z0_0, a1_0, y0_0, f);
            MulDstAdd(z1_0, a1_0, y1_0, f);
            MulDstAdd(z0_1, a1_1, y0_1, f);
            MulDstAdd(z1_1, a1_1, y1_1, f);
        }

        if constexpr (LAST) {
            RegTensor<float> r0, r1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(r0, scratch + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(r1, scratch + row + 1);

            Mul(z0_0, z0_0, r0, f);
            Mul(z1_0, z1_0, r0, f);
            Mul(z0_1, z0_1, r1, f);
            Mul(z1_1, z1_1, r1, f);

            Add(z0_0, z0_0, mv0, f);
            Add(z1_0, z1_0, mv1, f);
            Add(z0_1, z0_1, mv0, f);
            Add(z1_1, z1_1, mv1, f);

            RegTensor<T> p0_0, p1_0, p0_1, p1_1;
            Cast<T, float, cast>(p0_0, z0_0, f);
            Cast<T, float, cast>(p1_0, z1_0, f);
            Cast<T, float, cast>(p0_1, z0_1, f);
            Cast<T, float, cast>(p1_1, z1_1, f);

            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP, p0_0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP + 64, p1_0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP, p0_1, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP + 64, p1_1, f);
        } else {
            StoreAlign(accum + row * DP, z0_0, f);
            StoreAlign(accum + row * DP + 64, z1_0, f);
            StoreAlign(accum + (row + 1) * DP, z0_1, f);
            StoreAlign(accum + (row + 1) * DP + 64, z1_1, f);
        }
    }
}

// Clean-Room OutAccumulator Engine wrapping normalization & accumulation
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

    __aicore__ static inline void RoundPartialFp16(__ubuf__ float *partial, uint32_t count) {
        RoundPartialFp16VF(partial, count);
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

    template <class T, uint32_t DP, bool FIRST, bool LAST>
    __aicore__ static inline void UpdateWidePair(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial0,
                                                  __ubuf__ float *partial1, __ubuf__ float *alpha0, __ubuf__ float *alpha1,
                                                  __ubuf__ float *denom, __ubuf__ float *scratch, __ubuf__ float *meanV) {
        UpdateWidePairVF<T, DP, FIRST, LAST>(out, accum, partial0, partial1, alpha0, alpha1, denom, scratch, meanV);
    }
};

#endif // __CCE__

} // namespace sage::compute
