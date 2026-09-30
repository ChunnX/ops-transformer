#pragma once

#include "../core/platform_traits.h"
#include "../core/numerical_types.h"
#include "../memory/ring_buffer_layout.h"
#include "../memory/accompanying_v_cache.h"
#include "../orchestration/credit_handshake.h"

#if defined(__CCE__)
#include "kernel_operator.h"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#endif

namespace sage::microkernel {

#if defined(__CCE__)
using AscendC::RoundMode;
using memory::Sync;
using AscendC::HardEvent;

struct MxScaleCache {
    GM_ADDR base = nullptr;
};

// Split-M GEMM-1 (Q * K^T) on Cube

template <uint32_t BOffset>
__aicore__ inline __attribute__((always_inline)) void GemmQkSplit(memory::FixedResource &r,
                                                                  uint32_t ubOffset, bool loadA, uint32_t kParity) {
    using namespace AscendC;
    using namespace Catlass;
    using namespace tla;
    constexpr uint32_t k = 128, n = N;
    using T = int8_t;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<memory::Arch, T, layout::RowMajor, T, layout::ColumnMajor, float, layout::RowMajor>;

    WaitFlag<HardEvent::MTE1_MTE2>(EventHandshake::QK_EVENT);
    auto la = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(0),
                              tla::MakeLayout<T, typename Copy::LayoutTagL1A>(M, k), Catlass::Arch::PositionL1{});
    auto lb = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(BOffset + kParity * 65536),
                              tla::MakeLayout<T, typename Copy::LayoutTagL1B>(k, n), Catlass::Arch::PositionL1{});
    if (loadA) {
        EventHandshake::WaitFromBothAiv<PIPE_MTE1>(TokenSignal::Q_READY);
    }
    event_t kEvent = static_cast<event_t>(EventHandshake::K_PREFETCH_EVENT_BASE + kParity);
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
        Catlass::Gemm::Tile::TileMmadTla<memory::Arch, T, typename Copy::LayoutTagL1A> mma;
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

// Cube GEMM-2 (P * V) with Delayed FP16 MMA & FP32 Buffering

template <class T, class BLayout, bool LocalA = false, uint32_t BOffset = WIDE_B_L1, bool Emit = true,
          bool Initialize = true, bool PreloadedB = false>
__aicore__ inline __attribute__((always_inline)) void Gemm(memory::FixedResource &r, GM_ADDR a, GM_ADDR b, uint32_t k,
                                                           uint32_t n, uint32_t ubOffset, uint32_t parity = 0,
                                                           bool loadA = true, GM_ADDR mxScale = nullptr,
                                                           MxScaleCache *mxCache = nullptr, uint32_t mxK = 0,
                                                           uint32_t mxStart = 0, uint32_t vParity = 0) {
    using namespace AscendC;
    using namespace Catlass;
    using namespace tla;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<memory::Arch, T, layout::RowMajor, T, BLayout, float, layout::RowMajor>;
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
                Catlass::Gemm::Tile::TileCopyTla<memory::Arch, decltype(gs), decltype(sb)> copyScale;
                copyScale(sb, gs);
                if (cached) mxCache->base = scaleBase;
            }
        }
    }

    if constexpr (!LocalA) {
        if (loadA) {
            GlobalTensor<T> ag;
            ag.SetGlobalBuffer((__gm__ T *)a);
            auto ga = tla::MakeTensor(ag, tla::MakeLayout<T, layout::RowMajor>(M, k), Catlass::Arch::PositionGM{});
            typename Copy::template CopyGmToL1A<decltype(ga)> copyA;
            copyA(la, ga);
        }
        GlobalTensor<T> bg;
        bg.SetGlobalBuffer((__gm__ T *)b);
        auto gb = tla::MakeTensor(bg, tla::MakeLayout<T, BLayout>(k, n), Catlass::Arch::PositionGM{});
        typename Copy::template CopyGmToL1B<decltype(gb)> copyB;
        copyB(lb, gb);
    } else if constexpr (!PreloadedB) {
        GlobalTensor<T> bv;
        bv.SetGlobalBuffer((__gm__ T *)b);
        auto gv = tla::MakeTensor(bv, tla::MakeLayout<T, BLayout>(k, n), Catlass::Arch::PositionGM{});
        typename Copy::template CopyGmToL1B<decltype(gv)> copyB;
        copyB(lb, gv);
    }

    if constexpr (PreloadedB)
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(EventHandshake::V_PREFETCH_EVENT_BASE + (vParity & 1)));
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

    if constexpr (PreloadedB) SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(EventHandshake::V_PREFETCH_EVENT_BASE + (vParity & 1)));
    SetFlag<HardEvent::MTE1_MTE2>(slot);
    Sync<HardEvent::MTE1_M>();

    WaitFlag<HardEvent::FIX_M>(slot);
    auto cc = tla::MakeTensor(r.l0CBuf.template GetBufferByByte<Acc>(l0cOffset),
                              tla::MakeLayout<Acc, layout::L0C>(M, n), Catlass::Arch::PositionL0C{});
    Catlass::Gemm::Tile::TileMmadTla<memory::Arch, T, typename Copy::LayoutTagL1A> mma;
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
        Catlass::Gemm::Tile::CopyL0CToUBTla<memory::Arch, decltype(cc), decltype(u),
                                            Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M> copy;
        copy(u, cc);
    }
    SetFlag<HardEvent::FIX_M>(slot);
}

#endif // __CCE__

} // namespace sage::microkernel

namespace sage {
#if defined(__CCE__)
using microkernel::MxScaleCache;
using microkernel::GemmQkSplit;
using microkernel::Gemm;
#endif
}
