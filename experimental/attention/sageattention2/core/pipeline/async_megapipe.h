/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../types.h"
#include "../fast_math.h"
#include "event_handshake.h"
#include "../scheduler/snake_scheduler.h"
#include "../compute/fast_quant.h"
#include "../compute/simd_softmax.h"
#include "../compute/out_accumulator.h"

#if defined(__CCE__)
#include "../device_resource.h"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#endif

namespace sage::pipeline {

using namespace sage::compute;

#if defined(__CCE__)
using AscendC::RoundMode;

// Static UB layout for Wide streaming pipeline
struct WideVectorLayout {
    static constexpr uint32_t SCORE_UB = 0;
    static constexpr uint32_t PV_UB = 4 * HALF * N * 2;
    static constexpr uint32_t P_PART_BYTES = HALF * N;
    static constexpr uint32_t PV_SLOT_BYTES = 2 * P_PART_BYTES;
    static constexpr uint32_t ACCUM_UB = PV_UB + 2 * PV_SLOT_BYTES;
    static constexpr uint32_t P_UB = ACCUM_UB + HALF * 128 * 4;
    static constexpr uint32_t KS_UB = P_UB + P_SLOT_BYTES;
    static constexpr uint32_t QS_UB = KS_UB + 2 * 512 * 2;
    static constexpr uint32_t MAX_UB = QS_UB + 512;
    static constexpr uint32_t DEN_UB = MAX_UB + 512;
    static constexpr uint32_t ALPHA_UB = DEN_UB + 512;
    static constexpr uint32_t SUM_UB = ALPHA_UB + 3 * 512;
    static constexpr uint32_t VM_UB = SUM_UB + HALF * 4;
    static constexpr uint32_t PARTIAL_SUMS_UB = P_UB;
    static constexpr uint32_t END = VM_UB + 128 * sizeof(float);

    static_assert(PV_SLOT_BYTES >= HALF * 128 * 4, "PV slot cannot hold FP32 PV");
    static_assert(P_SLOT_BYTES >= HALF * 8 * 4, "output scratch cannot hold softmax partial sums");
    static_assert(END <= UB_HIGH_WATER, "wide softmax exceeds UB");
};

// Wide-pipeline L1 buffer allocations
constexpr uint32_t MX_P_L1 = 496 * 1024;
constexpr uint32_t MX_V_L1 = 320 * 1024;
constexpr uint32_t MX_P_BLOCK_NUM = M * N / 32 / 32;
constexpr uint32_t MX_P_M_BLOCKS = M / 16;
constexpr uint32_t MX_P_N_BLOCKS = N / 64;
constexpr uint32_t L0A_MX_P_OFFSET = 32768;

struct MxScaleCache {
    GM_ADDR base = nullptr;
};

// Initialize constant P scale 2^-8 in L0A for MXFP8 PV MMA
__aicore__ inline void InitMxP(FixedResource &r) {
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

template <uint32_t BOffset>
__aicore__ inline __attribute__((always_inline)) void PrefetchWideK(FixedResource &r, GM_ADDR b, uint32_t parity) {
    using namespace AscendC;
    using namespace Catlass;
    using namespace tla;
    using T = int8_t;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, T, layout::RowMajor, T, layout::ColumnMajor, float, layout::RowMajor>;
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
__aicore__ inline __attribute__((always_inline)) void PrefetchWideV(FixedResource &r, GM_ADDR b, uint32_t parity) {
    using namespace AscendC;
    using namespace Catlass;
    using namespace tla;
    using T = float8_e4m3_t;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, T, layout::RowMajor, T, layout::RowMajor, float, layout::RowMajor>;
    event_t ev = static_cast<event_t>(6 + parity);
    WaitFlag<HardEvent::MTE1_MTE2>(ev);
    GlobalTensor<T> bg;
    bg.SetGlobalBuffer((__gm__ T *)b);
    auto gb = tla::MakeTensor(bg, tla::MakeLayout<T, layout::RowMajor>(N, 128), Catlass::Arch::PositionGM{});
    auto lb = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(BOffset + N * 128 + parity * 65536),
                             tla::MakeLayout<T, typename Copy::LayoutTagL1B>(N, 128), Catlass::Arch::PositionL1{});
    typename Copy::template CopyGmToL1B<decltype(gb)> copy;
    copy(lb, gb);
    SetFlag<HardEvent::MTE2_MTE1>(ev);
}

// Split-M GEMM QK on Cube
template <uint32_t BOffset>
__aicore__ inline __attribute__((always_inline)) void GemmQkSplit(FixedResource &r,
                                                                  uint32_t ubOffset, bool loadA, uint32_t kParity) {
    using namespace AscendC;
    using namespace Catlass;
    using namespace tla;
    constexpr uint32_t k = 128, n = N;
    using T = int8_t;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, T, layout::RowMajor, T, layout::ColumnMajor, float, layout::RowMajor>;

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
        Catlass::Gemm::Tile::TileMmadTla<Arch, T, typename Copy::LayoutTagL1A> mma;
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

// Cube Gemm PV with Catlass templates
template <class T, class BLayout, bool LocalA = false, uint32_t BOffset = WIDE_B_L1, bool Emit = true,
          bool Initialize = true, bool PreloadedB = false>
__aicore__ inline __attribute__((always_inline)) void Gemm(FixedResource &r, GM_ADDR a, GM_ADDR b, uint32_t k,
                                                           uint32_t n, uint32_t ubOffset, uint32_t parity = 0,
                                                           bool loadA = true, GM_ADDR mxScale = nullptr,
                                                           MxScaleCache *mxCache = nullptr, uint32_t mxK = 0,
                                                           uint32_t mxStart = 0, uint32_t vParity = 0) {
    using namespace AscendC;
    using namespace Catlass;
    using namespace tla;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, T, layout::RowMajor, T, BLayout, float, layout::RowMajor>;
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
    auto sb = tla::MakeTensor(r.l1Buf.GetBufferByByte<float8_e8m0_t>(MX_V_L1),
                              tla::MakeMxScaleLayout<float8_e8m0_t, layout::nN, true>(scaleK / 32, n),
                              tla::MakeCoord(0, 0), Catlass::Arch::PositionL1{});

    if constexpr (LocalA) {
        if (!cached || mxCache->base != scaleBase) {
            GlobalTensor<float8_e8m0_t> sg;
            sg.SetGlobalBuffer((__gm__ float8_e8m0_t *)scaleBase);
            auto gs = tla::MakeTensor(sg, tla::MakeMxScaleLayout<float8_e8m0_t, layout::RowMajor, true>(scaleK / 32, n),
                                      Catlass::Arch::PositionGM{});
            Catlass::Gemm::Tile::TileCopyTla<Arch, decltype(gs), decltype(sb)> copyScale;
            copyScale(sb, gs);
            if constexpr (PreloadedB) Sync<HardEvent::MTE2_MTE1>();
            if (cached) mxCache->base = scaleBase;
        }
    }

    if constexpr (PreloadedB)
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(6 + vParity));
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
            tla::GetTile(sb, tla::MakeCoord(scaleStart / 32, 0), tla::MakeShape(k / 32, n)));
    } else {
        if (loadA) c0a(aa, tla::GetTile(la, tla::MakeCoord(0, 0), tla::MakeShape(M, k)));
        c0b(bb, tla::GetTile(lb, tla::MakeCoord(0, 0), tla::MakeShape(k, n)));
    }

    if constexpr (PreloadedB) SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(6 + vParity));
    SetFlag<HardEvent::MTE1_MTE2>(slot);
    Sync<HardEvent::MTE1_M>();

    WaitFlag<HardEvent::FIX_M>(slot);
    auto cc = tla::MakeTensor(r.l0CBuf.template GetBufferByByte<Acc>(l0cOffset),
                              tla::MakeLayout<Acc, layout::L0C>(M, n), Catlass::Arch::PositionL0C{});
    Catlass::Gemm::Tile::TileMmadTla<Arch, T, typename Copy::LayoutTagL1A> mma;
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
        Catlass::Gemm::Tile::CopyL0CToUBTla<Arch, decltype(cc), decltype(u),
                                            Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M> copy;
        copy(u, cc);
    }
    SetFlag<HardEvent::FIX_M>(slot);
}

// Metadata structures for wide stream pipeline
struct WideOutputInfo {
    uint32_t ordinal, b, h, qstart;
    bool first, last;
};

struct CubeFragmentInfo {
    uint64_t kb;
    uint32_t part;
};

template <uint32_t PHASE>
__aicore__ inline __attribute__((always_inline)) void ProcessWideCubeStep(
    FixedResource &r, CubeFragmentInfo *pending, uint64_t fragment, bool hasQk,
    GM_ADDR k8, GM_ADDR v8, GM_ADDR vsmx, MxScaleCache &mxCache, const SageParams &p,
    uint32_t count, bool prefetchPv = true)
{
    using namespace AscendC;
    constexpr uint32_t DP = 128;
    if (hasQk) {
        const auto &current = pending[fragment & 3];
        GemmQkSplit<WIDE_B_L1>(r, (fragment & 3) * HALF * N * 2,
                               current.part == 0, PHASE);
        if (current.part + 2 < count * 2)
            PrefetchWideK<WIDE_B_L1>(r, k8 + (current.kb + (current.part + 2) * N) * DP, PHASE);
    }
    if (fragment >= 2 && prefetchPv) {
        const auto &next = pending[(fragment - 2) & 3];
        PrefetchWideV<WIDE_B_L1>(r, v8 + (next.kb + next.part * N) * DP, PHASE);
    }
    if (fragment >= 3) {
        uint64_t previous = fragment - 3;
        const auto &old = pending[previous & 3];
        uint32_t group = previous / 2;
        EventHandshake::WaitFromBothAiv<PIPE_MTE1>(TokenSignal::P_READY);
        uint32_t parity = previous & 3;
        uint32_t pvOffset = WideVectorLayout::PV_UB + (group & 1) * WideVectorLayout::PV_SLOT_BYTES;
        if constexpr (PHASE == 0) {
            if (group >= 2) {
                EventHandshake::WaitFromBothAiv<PIPE_FIX>(TokenSignal::PV_RELEASE);
            }
            Gemm<float8_e4m3_t, Catlass::layout::RowMajor, true, WIDE_B_L1, true, false, true>(
                r, v8, nullptr, N, DP, pvOffset, parity, true,
                vsmx + (old.kb + old.part * N) / 32 * DP, &mxCache, p.kp, old.part * N, (1 - PHASE));
            EventHandshake::NotifyBothAiv<PIPE_FIX>(TokenSignal::PV_READY);
        } else {
            Gemm<float8_e4m3_t, Catlass::layout::RowMajor, true, WIDE_B_L1, false, true, true>(
                r, v8, nullptr, N, DP, pvOffset, parity, true,
                vsmx + (old.kb + old.part * N) / 32 * DP, &mxCache, p.kp, old.part * N, (1 - PHASE));
        }
    }
}

// Standard (non-wide) 1 AIC : 2 AIV Symmetrical M=64 Attention Kernel
template <class T, uint32_t DP, bool CAUSAL = false>
class StandardAttentionKernel {
public:
    __aicore__ inline __attribute__((always_inline)) void Init(
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, GM_ADDR mxScale,
        GM_ADDR vMean, GM_ADDR out, GM_ADDR lse, const SageParams &params)
    {
        using namespace AscendC;
        p_ = params;
        p_.dp = DP;
        q8_ = q8;
        k8_ = k8;
        v8_ = v8;
        vsmx_ = mxScale;
        out_ = out;
        lse_ = lse;
        subBlock_ = GetSubBlockIdx();

        qsg_.SetGlobalBuffer((__gm__ half *)qs);
        ksg_.SetGlobalBuffer((__gm__ half *)ks);
        vmg_.SetGlobalBuffer((__gm__ float *)vMean);
        og_.SetGlobalBuffer((__gm__ T *)out);
        lg_.SetGlobalBuffer((__gm__ float *)lse);

        accum_ = resource_.ubBuf.template GetBufferByByte<float>(ACCUM_UB);
        qo_ = resource_.ubBuf.template GetBufferByByte<T>(P_UB);
        qsu_ = resource_.ubBuf.template GetBufferByByte<half>(QS_UB);
        maxima_ = resource_.ubBuf.template GetBufferByByte<half>(MAX_UB);
        denom_ = resource_.ubBuf.template GetBufferByByte<float>(DEN_UB);
        vmean_ = resource_.ubBuf.template GetBufferByByte<float>(VM_UB);
    }

    __aicore__ inline __attribute__((always_inline)) void Process() {
        using namespace AscendC;
        auto &r = resource_;
        MxScaleCache mxCache;
        SageParams p = p_;
        GM_ADDR q8 = q8_;
        GM_ADDR k8 = k8_;
        GM_ADDR v8 = v8_;
        uint32_t core = GetBlockIdx();
        if ASCEND_IS_AIV core /= 2;

        if ASCEND_IS_AIC {
            InitMxP(r);
            SetFlag<HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
            SetFlag<HardEvent::MTE1_MTE2>(EventHandshake::PV_EVENT);
            SetFlag<HardEvent::M_MTE1>(EventHandshake::QK_EVENT);
            SetFlag<HardEvent::M_MTE1>(EventHandshake::PV_EVENT);
            SetFlag<HardEvent::FIX_M>(EventHandshake::QK_EVENT);
            SetFlag<HardEvent::FIX_M>(EventHandshake::PV_EVENT);
        }

        uint32_t tiles = (p.sq + M - 1) / M;
        const uint64_t total = uint64_t(p.b) * p.hq * tiles;
        const uint64_t taskEnd = DieTaskEnd(p, total, core);
        uint32_t ordinal = 0;

        for (uint64_t task = core; task < taskEnd; task += p.cores, ++ordinal) {
            auto taskInfo = sage::scheduler::SnakeScheduler<CAUSAL>::Decode(p, task, ordinal);
            uint32_t b = taskInfo.b, h = taskInfo.h;
            uint32_t qstart = taskInfo.qstart;
            uint64_t qb = taskInfo.qb, kb = taskInfo.kb;

            if ASCEND_IS_AIV {
                Sync<HardEvent::V_MTE2>();
                Duplicate(accum_, 0.0f, HALF * p.dp);
                DataCopy(qsu_, qsg_[qb + qstart + subBlock_ * HALF], HALF);
                Sync<HardEvent::MTE2_S>();
                Duplicate(maxima_, half(SAGE_FP16_MIN), 128);
                Duplicate(denom_, 0.0f, 64);
                PipeBarrier<PIPE_V>();
            }

            uint32_t limit = p.sk;
            if constexpr (CAUSAL) limit = (p.sk < qstart + M ? p.sk : qstart + M);
            uint32_t count = (limit + N - 1) / N;

            if ASCEND_IS_AIV {
                SetFlag<HardEvent::V_MTE2>(EventHandshake::SCALE_EVENT_BASE);
                SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + 1));
                for (uint32_t slot = 0; slot < 2; ++slot) {
                    if (slot < count) {
                        event_t ev = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + slot);
                        WaitFlag<HardEvent::V_MTE2>(ev);
                        DataCopy(r.ubBuf.template GetBufferByByte<half>(KS_UB + slot * N * 2), ksg_[kb + slot * N], N);
                        SetFlag<HardEvent::MTE2_V>(ev);
                    }
                }
            }

            for (uint32_t step = 0; step < count + LOOKAHEAD; ++step) {
                uint32_t start = step * N, parity = step % PIPE_SLOTS;
                uint32_t previous = (step + PIPE_SLOTS - LOOKAHEAD) % PIPE_SLOTS;
                if ASCEND_IS_AIC {
                    if (step < count) {
                        Gemm<int8_t, Catlass::layout::ColumnMajor, false, B_L1>(
                            r, q8 + (qb + qstart) * p.dp, k8 + (kb + start) * p.dp, p.dp, N,
                            SCORE_UB + parity * HALF * N * 2, 0, step == 0);
                        EventHandshake::NotifyBothAiv<PIPE_FIX>(TokenSignal::QK_READY);
                    }
                    if (step >= LOOKAHEAD) {
                        EventHandshake::WaitFromBothAiv<PIPE_MTE1>(TokenSignal::P_READY);
                        if (step >= LOOKAHEAD + PV_SLOTS) {
                            EventHandshake::WaitFromBothAiv<PIPE_FIX>(TokenSignal::PV_RELEASE);
                        }
                        Gemm<float8_e4m3_t, Catlass::layout::RowMajor, true, B_L1>(
                            r, v8, v8 + (kb + (step - LOOKAHEAD) * N) * p.dp, N, p.dp,
                            PV_UB + ((step - LOOKAHEAD) % PV_SLOTS) * PV_SLOT_BYTES, previous, true,
                            vsmx_ + (kb + (step - LOOKAHEAD) * N) / 32 * p.dp, &mxCache, p.kp, (step - LOOKAHEAD) * N);
                        EventHandshake::NotifyBothAiv<PIPE_FIX>(TokenSignal::PV_READY);
                    }
                } else {
                    if (step < count) {
                        auto scores = r.ubBuf.template GetBufferByByte<half>(SCORE_UB + parity * HALF * N * 2);
                        auto p8 = r.ubBuf.template GetBufferByByte<float8_e4m3_t>(P_UB + parity * P_SLOT_BYTES);
                        auto alpha = r.ubBuf.template GetBufferByByte<float>(ALPHA_UB + (step % ALPHA_SLOTS) * 512);
                        EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::QK_READY);
                        auto ksu = r.ubBuf.template GetBufferByByte<half>(KS_UB + (step & 1) * N * 2);
                        event_t scaleEvent = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + (step & 1));
                        WaitFlag<HardEvent::MTE2_V>(scaleEvent);
                        const bool isFull = (qstart + subBlock_ * HALF + HALF <= p.sq && start + N <= p.sk &&
                                             (!CAUSAL || start + N <= qstart + subBlock_ * HALF + 1));
                        const uint32_t qOffset = qstart + subBlock_ * HALF;
                        auto scratch = r.ubBuf.template GetBufferByByte<float>(SUM_UB);

                        if (isFull) {
                            SimdSoftmax::SoftmaxHalfVF<HALF, N, true, P_NZ_PAD_ROWS, CAUSAL>(
                                (__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)maxima_.GetPhyAddr(),
                                (__ubuf__ float *)denom_.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                                (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(), (__ubuf__ half *)qsu_.GetPhyAddr(),
                                (__ubuf__ half *)ksu.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                                qOffset, p.sq, start, p.sk);
                        } else {
                            SimdSoftmax::SoftmaxHalfVF<HALF, N, false, P_NZ_PAD_ROWS, CAUSAL>(
                                (__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)maxima_.GetPhyAddr(),
                                (__ubuf__ float *)denom_.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                                (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(), (__ubuf__ half *)qsu_.GetPhyAddr(),
                                (__ubuf__ half *)ksu.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                                qOffset, p.sq, start, p.sk);
                        }
                        SetFlag<HardEvent::V_MTE2>(scaleEvent);
                        if (step + 2 < count) {
                            WaitFlag<HardEvent::V_MTE2>(scaleEvent);
                            DataCopy(ksu, ksg_[kb + (step + 2) * N], N);
                            SetFlag<HardEvent::MTE2_V>(scaleEvent);
                        }
                        Sync<HardEvent::V_MTE3>();
                        auto pl1 = r.l1Buf.template GetBufferByByte<float8_e4m3_t>(P_L1 + parity * M * N);
                        DataCopyParams cp{N / 32, HALF, P_NZ_PAD_ROWS, HALF};
                        DataCopy(pl1[subBlock_ * HALF * 32], p8, cp);
                        EventHandshake::NotifyCube<PIPE_MTE3>(TokenSignal::P_READY);
                    }
                    if (step >= LOOKAHEAD) {
                        uint32_t index = step - LOOKAHEAD;
                        if (index & 1) {
                            EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                            EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                            if (p.accum == SagePvAccum::FP32_FP16) {
                                auto pv0 = r.ubBuf.template GetBufferByByte<float>(PV_UB);
                                auto pv1 = r.ubBuf.template GetBufferByByte<float>(PV_UB + PV_SLOT_BYTES);
                                OutAccumulator::RoundPartialFp16((__ubuf__ float *)pv0.GetPhyAddr(), HALF * p.dp);
                                OutAccumulator::RoundPartialFp16((__ubuf__ float *)pv1.GetPhyAddr(), HALF * p.dp);
                            }
                            auto alpha0 = r.ubBuf.template GetBufferByByte<float>(ALPHA_UB + ((index - 1) % ALPHA_SLOTS) * 512);
                            auto alpha1 = r.ubBuf.template GetBufferByByte<float>(ALPHA_UB + (index % ALPHA_SLOTS) * 512);
                            OutAccumulator::AccumulateFp32Pair<DP>(
                                (__ubuf__ float *)accum_.GetPhyAddr(),
                                (__ubuf__ float *)r.ubBuf.template GetBufferByByte<float>(PV_UB).GetPhyAddr(),
                                (__ubuf__ float *)r.ubBuf.template GetBufferByByte<float>(PV_UB + PV_SLOT_BYTES).GetPhyAddr(),
                                (__ubuf__ float *)alpha0.GetPhyAddr(), (__ubuf__ float *)alpha1.GetPhyAddr());
                            EventHandshake::NotifyCube<PIPE_V>(TokenSignal::PV_RELEASE);
                            EventHandshake::NotifyCube<PIPE_V>(TokenSignal::PV_RELEASE);
                        } else if (index + 1 == count) {
                            EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                            if (p.accum == SagePvAccum::FP32_FP16) {
                                auto pv0 = r.ubBuf.template GetBufferByByte<float>(PV_UB);
                                OutAccumulator::RoundPartialFp16((__ubuf__ float *)pv0.GetPhyAddr(), HALF * p.dp);
                            }
                            auto alpha = r.ubBuf.template GetBufferByByte<float>(ALPHA_UB + (index % ALPHA_SLOTS) * 512);
                            OutAccumulator::AccumulateFp32<DP>(
                                (__ubuf__ float *)accum_.GetPhyAddr(),
                                (__ubuf__ float *)r.ubBuf.template GetBufferByByte<float>(PV_UB).GetPhyAddr(),
                                (__ubuf__ float *)alpha.GetPhyAddr());
                            EventHandshake::NotifyCube<PIPE_V>(TokenSignal::PV_RELEASE);
                        }
                    }
                }
            }
            if ASCEND_IS_AIC {
                for (uint32_t drain = 0; drain < (count < PV_SLOTS ? count : PV_SLOTS); ++drain) {
                    EventHandshake::WaitFromBothAiv<PIPE_FIX>(TokenSignal::PV_RELEASE);
                }
            }
            if ASCEND_IS_AIV {
                WaitFlag<HardEvent::V_MTE2>(EventHandshake::SCALE_EVENT_BASE);
                WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + 1));
                Sync<HardEvent::V_MTE2>();
                if (p.smoothV) {
                    uint32_t kh = CalcKh(h, p.hq, p.hk);
                    DataCopy(vmean_, vmg_[(uint64_t(b) * p.hk + kh) * p.dp], p.dp);
                } else {
                    Duplicate(vmean_, 0.0f, 128);
                }
                Sync<HardEvent::MTE2_V>();
                OutAccumulator::NormalizeCast<T, DP>(
                    (__ubuf__ T *)qo_.GetPhyAddr(), (__ubuf__ float *)accum_.GetPhyAddr(),
                    (__ubuf__ float *)denom_.GetPhyAddr(),
                    (__ubuf__ float *)r.ubBuf.template GetBufferByByte<float>(SUM_UB).GetPhyAddr(),
                    (__ubuf__ float *)vmean_.GetPhyAddr());
                PipeBarrier<PIPE_V>();
                uint32_t first = qstart + subBlock_ * HALF;
                uint32_t validRows = first < p.sq ? (p.sq - first < HALF ? p.sq - first : HALF) : 0;
                if (p.d == p.dp) {
                    if (validRows)
                        WriteRow(og_, ((uint64_t(b) * p.hq + h) * p.sq + first) * p.d, qo_, validRows * p.d);
                } else {
                    for (uint32_t i = 0; i < validRows; ++i)
                        WriteRow(og_, ((uint64_t(b) * p.hq + h) * p.sq + first + i) * p.d, qo_[i * p.dp], p.d);
                }
                if (p.returnLse) {
                    auto scratch = r.ubBuf.template GetBufferByByte<float>(P_UB);
                    OutAccumulator::StoreTileLse(lg_, denom_, maxima_, scratch, first, validRows, b, h, p);
                }
            }
        }
        if ASCEND_IS_AIC {
            WaitFlag<HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
            WaitFlag<HardEvent::MTE1_MTE2>(EventHandshake::PV_EVENT);
            WaitFlag<HardEvent::M_MTE1>(EventHandshake::QK_EVENT);
            WaitFlag<HardEvent::M_MTE1>(EventHandshake::PV_EVENT);
            WaitFlag<HardEvent::FIX_M>(EventHandshake::QK_EVENT);
            WaitFlag<HardEvent::FIX_M>(EventHandshake::PV_EVENT);
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    FixedResource resource_;
    SageParams p_;
    GM_ADDR q8_;
    GM_ADDR k8_;
    GM_ADDR v8_;
    GM_ADDR vsmx_;
    GM_ADDR out_;
    GM_ADDR lse_;
    uint32_t subBlock_;
    AscendC::GlobalTensor<half> qsg_, ksg_;
    AscendC::GlobalTensor<float> vmg_, lg_;
    AscendC::GlobalTensor<T> og_;
    AscendC::LocalTensor<float> accum_, denom_, vmean_;
    AscendC::LocalTensor<T> qo_;
    AscendC::LocalTensor<half> qsu_, maxima_;
};

// Wide 2-block streaming attention kernel
template <class T, bool CAUSAL = false>
__aicore__ inline __attribute__((always_inline)) void ExecuteWideStream(
    GM_ADDR q, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, GM_ADDR mxScale,
    GM_ADDR vMean, GM_ADDR out, GM_ADDR lse, const SageParams &p)
{
    using namespace AscendC;
    FixedResource r;
    MxScaleCache mxCache;
    uint32_t core = GetBlockIdx();
    if ASCEND_IS_AIV core /= 2;

    const uint64_t totalTasks = DieTaskEnd(p, uint64_t(p.b) * p.hq * (p.sq / M), core);
    if (core >= totalTasks) return;

    uint32_t subBlock = GetSubBlockIdx();
    GlobalTensor<T> qg, og;
    qg.SetGlobalBuffer((__gm__ T *)q);
    qg.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    og.SetGlobalBuffer((__gm__ T *)out);
    GlobalTensor<half> qsg, ksg;
    qsg.SetGlobalBuffer((__gm__ half *)qs);
    ksg.SetGlobalBuffer((__gm__ half *)ks);
    GlobalTensor<float> vmg, lg;
    vmg.SetGlobalBuffer((__gm__ float *)vMean);
    lg.SetGlobalBuffer((__gm__ float *)lse);

    auto accum = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::ACCUM_UB);
    auto denom = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::DEN_UB);
    auto qsu = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::QS_UB);
    auto vmean = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::VM_UB);
    auto qo = r.ubBuf.template GetBufferByByte<T>(WideVectorLayout::P_UB);

    if ASCEND_IS_AIC {
        InitMxP(r);
        SetFlag<HardEvent::MTE1_MTE2>(4);
        SetFlag<HardEvent::MTE1_MTE2>(5);
        SetFlag<HardEvent::MTE1_MTE2>(6);
        SetFlag<HardEvent::MTE1_MTE2>(7);
        SetFlag<HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
        SetFlag<HardEvent::MTE1_MTE2>(EventHandshake::PV_EVENT);
        SetFlag<HardEvent::M_MTE1>(EventHandshake::QK_EVENT);
        SetFlag<HardEvent::M_MTE1>(EventHandshake::PV_EVENT);
        SetFlag<HardEvent::FIX_M>(EventHandshake::QK_EVENT);
        SetFlag<HardEvent::FIX_M>(EventHandshake::QK_SECOND_FIX_EVENT);
        SetFlag<HardEvent::FIX_M>(EventHandshake::PV_EVENT);
    } else {
        auto first = sage::scheduler::SnakeScheduler<CAUSAL>::Decode(p, core, 0);
        uint64_t firstScale = first.kb;
        uint64_t secondScale = sage::scheduler::SnakeScheduler<CAUSAL>::LookaheadScale(p, first, 1, totalTasks);
        SetFlag<HardEvent::V_MTE2>(EventHandshake::SCALE_EVENT_BASE);
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + 1));
        for (uint32_t slot = 0; slot < 2; ++slot) {
            uint64_t offset = slot ? secondScale : firstScale;
            if (offset != ~uint64_t(0)) {
                event_t ev = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + slot);
                WaitFlag<HardEvent::V_MTE2>(ev);
                DataCopy(r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::KS_UB + slot * 1024), ksg[offset], 512);
                SetFlag<HardEvent::MTE2_V>(ev);
            }
        }
    }

    WideOutputInfo outputPending[3];
    CubeFragmentInfo cubePending[4];
    uint64_t fragment = 0, retired = 0;
    uint32_t ordinal = 0;

    for (uint64_t task = core; task < totalTasks; task += p.cores, ++ordinal) {
        auto current = sage::scheduler::SnakeScheduler<CAUSAL>::Decode(p, task, ordinal);
        if ASCEND_IS_AIC {
            PrefetchWideK<WIDE_B_L1>(r, k8 + current.kb * 128, 0);
            PrefetchWideK<WIDE_B_L1>(r, k8 + (current.kb + N) * 128, 1);
            for (uint32_t part = 0; part < current.count * 2; part += 2, fragment += 2) {
                cubePending[fragment & 3] = {current.kb, part};
                ProcessWideCubeStep<0>(r, cubePending, fragment, true, k8, v8, mxScale, mxCache, p, current.count);
                cubePending[(fragment + 1) & 3] = {current.kb, part + 1};
                ProcessWideCubeStep<1>(r, cubePending, fragment + 1, true, k8, v8, mxScale, mxCache, p, current.count);
            }
        } else {
            for (uint32_t step = 0; step < current.count; ++step, fragment += 2) {
                const uint64_t group = fragment / 2;
                bool deferPair = false;
                if (group >= 2 && retired <= group - 2) {
                    const auto &old = outputPending[retired % 3];
                    deferPair = (retired + 1 < group && old.ordinal == outputPending[(retired + 1) % 3].ordinal);
                    if (deferPair) {
                        EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                        EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                    } else {
                        if (old.last) {
                            EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                            Sync<HardEvent::V_MTE2>();
                            if (p.smoothV) {
                                uint32_t kh = CalcKh(old.h, p.hq, p.hk);
                                DataCopy(vmean, vmg[(uint64_t(old.b) * p.hk + kh) * p.dp], p.dp);
                            } else {
                                Duplicate(vmean, 0.0f, 128);
                            }
                            Sync<HardEvent::MTE2_V>();
                            auto pv = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::PV_UB + (retired & 1) * WideVectorLayout::PV_SLOT_BYTES);
                            auto alpha = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::ALPHA_UB + (retired % 3) * 512);
                            auto scratch = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::SUM_UB);
                            OutAccumulator::FinalizeWideFp32<T, 128, false>(
                                (__ubuf__ T *)qo.GetPhyAddr(), (__ubuf__ float *)accum.GetPhyAddr(),
                                (__ubuf__ float *)pv.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                                (__ubuf__ float *)denom.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                                (__ubuf__ float *)vmean.GetPhyAddr());
                            PipeBarrier<PIPE_V>();
                            uint32_t firstRow = old.qstart + subBlock * HALF;
                            uint32_t validRows = firstRow < p.sq ? (p.sq - firstRow < HALF ? p.sq - firstRow : HALF) : 0;
                            if (validRows)
                                WriteRow(og, ((uint64_t(old.b) * p.hq + old.h) * p.sq + firstRow) * p.d, qo, validRows * p.d);
                            if (p.returnLse) {
                                auto lseScratch = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::P_UB);
                                auto maximaOld = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::MAX_UB + (old.ordinal & 1) * 256);
                                OutAccumulator::StoreTileLse(lg, denom, maximaOld, lseScratch, firstRow, validRows, old.b, old.h, p);
                            }
                        }
                        ++retired;
                    }
                }

                const bool first = step == 0, last = step + 1 == current.count;
                auto maxima = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::MAX_UB + (current.ordinal & 1) * 256);

                if (first) {
                    constexpr uint32_t Q_IN_UB = WideVectorLayout::SCORE_UB;
                    constexpr uint32_t Q_FP32_UB = Q_IN_UB + HALF * 128 * sizeof(T);
                    constexpr uint32_t Q_INT8_UB = Q_FP32_UB + HALF * 128 * sizeof(float);
                    constexpr uint32_t Q_RAW_SCALE = Q_INT8_UB + HALF * 128 * sizeof(int8_t);
                    constexpr uint32_t Q_INV_SCALE = Q_RAW_SCALE + HALF * sizeof(float);

                    auto input = r.ubBuf.template GetBufferByByte<T>(Q_IN_UB);
                    auto x = r.ubBuf.template GetBufferByByte<float>(Q_FP32_UB);
                    auto qOut = r.ubBuf.template GetBufferByByte<int8_t>(Q_INT8_UB);
                    auto scales = r.ubBuf.template GetBufferByByte<float>(Q_RAW_SCALE);
                    auto inverseScales = r.ubBuf.template GetBufferByByte<float>(Q_INV_SCALE);

                    Sync<HardEvent::MTE3_MTE2>();
                    Sync<HardEvent::V_MTE2>();
                    uint64_t qToken = current.qb + current.qstart + subBlock * HALF;
                    DataCopy(input, qg[qToken * 128], HALF * 128);
                    Sync<HardEvent::MTE2_V>();

                    Cast(x, input, RoundMode::CAST_NONE, HALF * 128);
                    PipeBarrier<PIPE_V>();

                    QuantRowsVF((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)x.GetPhyAddr(),
                                (__ubuf__ float *)scales.GetPhyAddr(), (__ubuf__ int8_t *)qOut.GetPhyAddr(),
                                (__ubuf__ half *)qsu.GetPhyAddr(), p.qFactor,
                                (__ubuf__ float *)inverseScales.GetPhyAddr(), 128, 0, HALF, p.qg);

                    Sync<HardEvent::V_MTE3>();
                    auto ql1 = r.l1Buf.template GetBufferByByte<int8_t>(subBlock * HALF * 32);
                    DataCopyParams plane{HALF, 1, 128 / 32 - 1, 0};
                    #pragma unroll
                    for (uint32_t cb = 0; cb < 128 / 32; ++cb)
                        DataCopy(ql1[cb * M * 32], qOut[cb * 32], plane);

                    EventHandshake::NotifyCube<PIPE_MTE3>(TokenSignal::Q_READY);
                    Duplicate(maxima, half(SAGE_FP16_MIN), 128);
                    PipeBarrier<PIPE_V>();
                }

                auto scores = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::SCORE_UB + (group & 1) * 2 * HALF * N * 2);
                auto p8 = scores.template ReinterpretCast<float8_e4m3_t>();
                auto alpha = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::ALPHA_UB + (group % 3) * 512);

                EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::QK_READY);
                EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::QK_READY);

                auto ksu = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::KS_UB + (group & 1) * 512 * 2);
                event_t ev = static_cast<event_t>(EventHandshake::SCALE_EVENT_BASE + (group & 1));
                WaitFlag<HardEvent::MTE2_V>(ev);
                auto scoresNext = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::SCORE_UB + ((group & 1) * 2 + 1) * HALF * N * 2);
                auto partialSums = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::PARTIAL_SUMS_UB);

                if constexpr (CAUSAL) {
                    if (step + 1 == current.count) {
                        SimdSoftmax::SoftmaxWideFirstVF<true>(
                            (__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)scoresNext.GetPhyAddr(),
                            (__ubuf__ half *)maxima.GetPhyAddr(),
                            (__ubuf__ float *)alpha.GetPhyAddr(), (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(),
                            (__ubuf__ half *)qsu.GetPhyAddr(), (__ubuf__ half *)ksu.GetPhyAddr(),
                            (__ubuf__ float *)partialSums.GetPhyAddr(),
                            static_cast<int32_t>(int64_t(current.qstart) + int64_t(subBlock) * HALF - int64_t(step) * 512));
                    } else {
                        SimdSoftmax::SoftmaxWideFirstVF<false>(
                            (__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)scoresNext.GetPhyAddr(),
                            (__ubuf__ half *)maxima.GetPhyAddr(),
                            (__ubuf__ float *)alpha.GetPhyAddr(), (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(),
                            (__ubuf__ half *)qsu.GetPhyAddr(), (__ubuf__ half *)ksu.GetPhyAddr(),
                            (__ubuf__ float *)partialSums.GetPhyAddr());
                    }
                } else {
                    SimdSoftmax::SoftmaxWideFirstVF<false>(
                        (__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)scoresNext.GetPhyAddr(),
                        (__ubuf__ half *)maxima.GetPhyAddr(),
                        (__ubuf__ float *)alpha.GetPhyAddr(), (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(),
                        (__ubuf__ half *)qsu.GetPhyAddr(), (__ubuf__ half *)ksu.GetPhyAddr(),
                        (__ubuf__ float *)partialSums.GetPhyAddr());
                }

                SetFlag<HardEvent::V_MTE2>(ev);
                uint64_t nextScale = sage::scheduler::SnakeScheduler<CAUSAL>::LookaheadScale(p, current, step + 2, totalTasks);
                if (nextScale != ~uint64_t(0)) {
                    WaitFlag<HardEvent::V_MTE2>(ev);
                    DataCopy(ksu, ksg[nextScale], 512);
                    SetFlag<HardEvent::MTE2_V>(ev);
                }

                Sync<HardEvent::V_MTE3>();
                auto pl1 = r.l1Buf.template GetBufferByByte<float8_e4m3_t>((group & 1) * 2 * M * N);
                DataCopyParams cp{N / 32, HALF, HALF, HALF};
                DataCopy(pl1[subBlock * HALF * 32], p8, cp);
                EventHandshake::NotifyCube<PIPE_MTE3>(TokenSignal::P_READY);

                auto p8Next = scoresNext.template ReinterpretCast<float8_e4m3_t>();
                SimdSoftmax::SoftmaxWideSecondVF(
                    (__ubuf__ half *)scoresNext.GetPhyAddr(), (__ubuf__ half *)maxima.GetPhyAddr(),
                    (__ubuf__ float *)alpha.GetPhyAddr(),
                    (__ubuf__ float8_e4m3_t *)p8Next.GetPhyAddr(),
                    (__ubuf__ float *)partialSums.GetPhyAddr());
                Sync<HardEvent::V_MTE3>();
                DataCopy(pl1[M * N + subBlock * HALF * 32], p8Next, cp);
                EventHandshake::NotifyCube<PIPE_MTE3>(TokenSignal::P_READY);

                outputPending[group % 3] = {current.ordinal, current.b, current.h, current.qstart, first, last};

                if (deferPair) {
                    const auto firstInfo = outputPending[retired % 3];
                    const auto secondInfo = outputPending[(retired + 1) % 3];
                    Sync<HardEvent::V_MTE2>();
                    if (p.smoothV) {
                        uint32_t kh = CalcKh(secondInfo.h, p.hq, p.hk);
                        DataCopy(vmean, vmg[(uint64_t(secondInfo.b) * p.hk + kh) * p.dp], p.dp);
                    } else {
                        Duplicate(vmean, 0.0f, 128);
                    }
                    Sync<HardEvent::MTE2_V>();
                    auto pv0 = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::PV_UB + (retired & 1) * WideVectorLayout::PV_SLOT_BYTES);
                    auto pv1 = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::PV_UB + ((retired + 1) & 1) * WideVectorLayout::PV_SLOT_BYTES);
                    auto a0 = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::ALPHA_UB + (retired % 3) * 512);
                    auto a1 = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::ALPHA_UB + ((retired + 1) % 3) * 512);
                    auto scratch = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::SUM_UB);

                    if (firstInfo.first && secondInfo.last) {
                        OutAccumulator::UpdateWidePair<T, 128, true, true>(
                            (__ubuf__ T *)qo.GetPhyAddr(), (__ubuf__ float *)accum.GetPhyAddr(),
                            (__ubuf__ float *)pv0.GetPhyAddr(), (__ubuf__ float *)pv1.GetPhyAddr(),
                            (__ubuf__ float *)a0.GetPhyAddr(), (__ubuf__ float *)a1.GetPhyAddr(),
                            (__ubuf__ float *)denom.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                            (__ubuf__ float *)vmean.GetPhyAddr());
                    } else if (firstInfo.first) {
                        OutAccumulator::UpdateWidePair<T, 128, true, false>(
                            (__ubuf__ T *)qo.GetPhyAddr(), (__ubuf__ float *)accum.GetPhyAddr(),
                            (__ubuf__ float *)pv0.GetPhyAddr(), (__ubuf__ float *)pv1.GetPhyAddr(),
                            (__ubuf__ float *)a0.GetPhyAddr(), (__ubuf__ float *)a1.GetPhyAddr(),
                            (__ubuf__ float *)denom.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                            (__ubuf__ float *)vmean.GetPhyAddr());
                    } else if (secondInfo.last) {
                        OutAccumulator::UpdateWidePair<T, 128, false, true>(
                            (__ubuf__ T *)qo.GetPhyAddr(), (__ubuf__ float *)accum.GetPhyAddr(),
                            (__ubuf__ float *)pv0.GetPhyAddr(), (__ubuf__ float *)pv1.GetPhyAddr(),
                            (__ubuf__ float *)a0.GetPhyAddr(), (__ubuf__ float *)a1.GetPhyAddr(),
                            (__ubuf__ float *)denom.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                            (__ubuf__ float *)vmean.GetPhyAddr());
                    } else {
                        OutAccumulator::UpdateWidePair<T, 128, false, false>(
                            (__ubuf__ T *)qo.GetPhyAddr(), (__ubuf__ float *)accum.GetPhyAddr(),
                            (__ubuf__ float *)pv0.GetPhyAddr(), (__ubuf__ float *)pv1.GetPhyAddr(),
                            (__ubuf__ float *)a0.GetPhyAddr(), (__ubuf__ float *)a1.GetPhyAddr(),
                            (__ubuf__ float *)denom.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                            (__ubuf__ float *)vmean.GetPhyAddr());
                    }
                    if (secondInfo.last) {
                        PipeBarrier<PIPE_V>();
                        uint32_t firstRow = secondInfo.qstart + subBlock * HALF;
                        uint32_t validRows = firstRow < p.sq ? (p.sq - firstRow < HALF ? p.sq - firstRow : HALF) : 0;
                        if (validRows)
                            WriteRow(og, ((uint64_t(secondInfo.b) * p.hq + secondInfo.h) * p.sq + firstRow) * p.d, qo, validRows * p.d);
                        if (p.returnLse) {
                            auto lseScratch = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::P_UB);
                            auto maximaOld = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::MAX_UB + (secondInfo.ordinal & 1) * 256);
                            OutAccumulator::StoreTileLse(lg, denom, maximaOld, lseScratch, firstRow, validRows, secondInfo.b, secondInfo.h, p);
                        }
                    }
                    retired += 2;
                }
            }
        }
    }

    if ASCEND_IS_AIC {
        ProcessWideCubeStep<0>(r, cubePending, fragment, false, k8, v8, mxScale, mxCache, p, 0);
        ProcessWideCubeStep<1>(r, cubePending, fragment + 1, false, k8, v8, mxScale, mxCache, p, 0);
        ProcessWideCubeStep<0>(r, cubePending, fragment + 2, false, k8, v8, mxScale, mxCache, p, 0, false);
    } else {
        const uint64_t groups = fragment / 2;
        while (retired < groups) {
            const auto old = outputPending[retired % 3];
            if (old.last) {
                EventHandshake::WaitFromCube<PIPE_V>(TokenSignal::PV_READY);
                Sync<HardEvent::V_MTE2>();
                if (p.smoothV) {
                    uint32_t kh = CalcKh(old.h, p.hq, p.hk);
                    DataCopy(vmean, vmg[(uint64_t(old.b) * p.hk + kh) * p.dp], p.dp);
                } else {
                    Duplicate(vmean, 0.0f, 128);
                }
                Sync<HardEvent::MTE2_V>();
                auto pv = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::PV_UB + (retired & 1) * WideVectorLayout::PV_SLOT_BYTES);
                auto alpha = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::ALPHA_UB + (retired % 3) * 512);
                auto scratch = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::SUM_UB);
                OutAccumulator::FinalizeWideFp32<T, 128, false>(
                    (__ubuf__ T *)qo.GetPhyAddr(), (__ubuf__ float *)accum.GetPhyAddr(),
                    (__ubuf__ float *)pv.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                    (__ubuf__ float *)denom.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                    (__ubuf__ float *)vmean.GetPhyAddr());
                PipeBarrier<PIPE_V>();
                uint32_t firstRow = old.qstart + subBlock * HALF;
                uint32_t validRows = firstRow < p.sq ? (p.sq - firstRow < HALF ? p.sq - firstRow : HALF) : 0;
                if (validRows)
                    WriteRow(og, ((uint64_t(old.b) * p.hq + old.h) * p.sq + firstRow) * p.d, qo, validRows * p.d);
                if (p.returnLse) {
                    auto lseScratch = r.ubBuf.template GetBufferByByte<float>(WideVectorLayout::P_UB);
                    auto maximaOld = r.ubBuf.template GetBufferByByte<half>(WideVectorLayout::MAX_UB + (old.ordinal & 1) * 256);
                    OutAccumulator::StoreTileLse(lg, denom, maximaOld, lseScratch, firstRow, validRows, old.b, old.h, p);
                }
            }
            ++retired;
        }
    }

    if ASCEND_IS_AIC {
        for (uint32_t i = 0; i < (fragment / 2 < 2 ? fragment / 2 : 2); ++i) {
            EventHandshake::WaitFromBothAiv<PIPE_FIX>(TokenSignal::PV_RELEASE);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(4);
        WaitFlag<HardEvent::MTE1_MTE2>(5);
        WaitFlag<HardEvent::MTE1_MTE2>(6);
        WaitFlag<HardEvent::MTE1_MTE2>(7);
        WaitFlag<HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
        WaitFlag<HardEvent::MTE1_MTE2>(EventHandshake::PV_EVENT);
        WaitFlag<HardEvent::M_MTE1>(EventHandshake::QK_EVENT);
        WaitFlag<HardEvent::M_MTE1>(EventHandshake::PV_EVENT);
        WaitFlag<HardEvent::FIX_M>(EventHandshake::QK_EVENT);
        WaitFlag<HardEvent::FIX_M>(EventHandshake::QK_SECOND_FIX_EVENT);
        WaitFlag<HardEvent::FIX_M>(EventHandshake::PV_EVENT);
    }
    PipeBarrier<PIPE_ALL>();
}

// Unified AsyncMegaPipe Facade coordinating Cube-Vector 4-stage pipeline
template <class T, bool WIDE>
class AsyncMegaPipe {
public:
    __aicore__ static inline __attribute__((always_inline)) void Execute(
        GM_ADDR q, GM_ADDR k8, GM_ADDR v8, GM_ADDR q8, GM_ADDR qs, GM_ADDR ks,
        GM_ADDR mxScale, GM_ADDR vMean, GM_ADDR out, GM_ADDR lse, const SageParams &p)
    {
        if constexpr (WIDE) {
            if (p.causal)
                ExecuteWideStream<T, true>(q, k8, v8, qs, ks, mxScale, vMean, out, lse, p);
            else
                ExecuteWideStream<T, false>(q, k8, v8, qs, ks, mxScale, vMean, out, lse, p);
        } else {
            if (p.causal) {
                if (p.dp == 128) {
                    StandardAttentionKernel<T, 128, true> kernel;
                    kernel.Init(q8, k8, v8, qs, ks, mxScale, vMean, out, lse, p);
                    kernel.Process();
                } else {
                    StandardAttentionKernel<T, 64, true> kernel;
                    kernel.Init(q8, k8, v8, qs, ks, mxScale, vMean, out, lse, p);
                    kernel.Process();
                }
            } else {
                if (p.dp == 128) {
                    StandardAttentionKernel<T, 128, false> kernel;
                    kernel.Init(q8, k8, v8, qs, ks, mxScale, vMean, out, lse, p);
                    kernel.Process();
                } else {
                    StandardAttentionKernel<T, 64, false> kernel;
                    kernel.Init(q8, k8, v8, qs, ks, mxScale, vMean, out, lse, p);
                    kernel.Process();
                }
            }
        }
    }
};

#endif

} // namespace sage::pipeline
