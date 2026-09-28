/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "attention_common.h"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
namespace Sage {
// P uses a unit E8M0 scale resident in L0A. V scales occupy the L1 gap
// above the QK/PV operand slots; oversized heads are copied tile by tile.
constexpr uint32_t MX_P_L1 = 496 * 1024, MX_V_L1 = 320 * 1024;
constexpr uint32_t MX_P_BLOCK_NUM = M * N / 32 / 32;
constexpr uint32_t MX_P_M_BLOCKS = M / 16;
constexpr uint32_t MX_P_N_BLOCKS = N / 64;
constexpr uint32_t L0A_MX_P_OFFSET = 32768;
struct MxScaleCache {
    GM_ADDR base = nullptr;
};
__aicore__ inline void InitMxP(FixedResource &r)
{
    InitConstValueParams<int16_t> cp;
    cp.repeatTimes = 1;
    cp.blockNum = MX_P_BLOCK_NUM;
    cp.initValue = 0x7f7f;
    cp.dstGap = 0;
    InitConstValue(r.l1Buf.GetBufferByByte<int16_t>(MX_P_L1), cp);
    Sync<HardEvent::MTE2_MTE1>();
    const uint64_t mxDst = reinterpret_cast<uint64_t>(r.l0ABuf.GetBufferByByte<float8_e4m3_t>(L0A_MX_P_OFFSET).GetPhyAddr()) / 16;
    load_cbuf_to_ca_mx(mxDst, (__cbuf__ void *)r.l1Buf.GetBufferByByte<float8_e8m0_t>(MX_P_L1).GetPhyAddr(), 0, 0,
                       MX_P_M_BLOCKS, MX_P_N_BLOCKS, MX_P_N_BLOCKS, MX_P_N_BLOCKS);
    Sync<HardEvent::MTE1_M>();
}
template <uint32_t BOffset>
__aicore__ inline __attribute__((always_inline)) void PrefetchWideV(FixedResource &r, GM_ADDR b, uint32_t parity)
{
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
template <class T, class BLayout, bool LocalA = false, uint32_t BOffset = B_L1, bool Emit = true,
          bool Initialize = true, bool PreloadedB = false>
__aicore__ inline __attribute__((always_inline)) void Gemm(FixedResource &r, GM_ADDR a, GM_ADDR b, uint32_t k,
                                                           uint32_t n, uint32_t ubOffset, uint32_t parity = 0,
                                                           bool loadA = true, GM_ADDR mxScale = nullptr,
                                                           MxScaleCache *mxCache = nullptr, uint32_t mxK = 0,
                                                           uint32_t mxStart = 0, uint32_t vParity = 0)
{
    using namespace Catlass;
    using namespace tla;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, T, layout::RowMajor, T, BLayout, float, layout::RowMajor>;
    using Acc = typename Copy::ElementAccumulator;
    using L0T = std::conditional_t<LocalA, AscendC::mx_fp8_e4m3_t, T>;
    // QK and PV own separate L1/L0 slots; ownership events are armed once.
    constexpr event_t slot = static_cast<event_t>(LocalA ? PV_EVENT : QK_EVENT);
    constexpr uint32_t l0Offset = LocalA ? 32768 : 0;
    constexpr uint32_t l0cOffset = LocalA ? 131072 : 0;
    WaitFlag<HardEvent::MTE1_MTE2>(slot);
    GlobalTensor<T> ag, bg;
    ag.SetGlobalBuffer((__gm__ T *)a);
    bg.SetGlobalBuffer((__gm__ T *)b);
    auto ga = tla::MakeTensor(ag, tla::MakeLayout<T, layout::RowMajor>(M, k), Catlass::Arch::PositionGM{});
    auto gb = tla::MakeTensor(bg, tla::MakeLayout<T, BLayout>(k, n), Catlass::Arch::PositionGM{});
    auto la = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(LocalA ? P_L1 + parity * M * N : 0),
                              tla::MakeLayout<T, typename Copy::LayoutTagL1A>(M, k), Catlass::Arch::PositionL1{});
    auto lb = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(BOffset + (LocalA ? N * 128 : 0) + (PreloadedB ? vParity * 65536 : 0)),
                              tla::MakeLayout<T, typename Copy::LayoutTagL1B>(k, n), Catlass::Arch::PositionL1{});
    if constexpr (!LocalA) {
        if (loadA) {
            typename Copy::template CopyGmToL1A<decltype(ga)> ca;
            ca(la, ga);
        }
    }
    typename Copy::template CopyGmToL1B<decltype(gb)> cb;
    if constexpr (!PreloadedB) cb(lb, gb);
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
            if (cached)
                mxCache->base = scaleBase;
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
    // The QK and PV L0A partitions are disjoint. Q remains valid for the
    // complete query tile, just like its L1 copy; only the first QK loads it.
    if constexpr (LocalA) {
        auto aaData =
            tla::MakeTensor(aa.data().template ReinterpretCast<T>(), aa.layout(), Catlass::Arch::PositionL0A{});
        c0a(aaData, tla::GetTile(la, tla::MakeCoord(0, 0), tla::MakeShape(M, k)));
        c0b(bb, tla::GetTile(lb, tla::MakeCoord(0, 0), tla::MakeShape(k, n)),
            tla::GetTile(sb, tla::MakeCoord(scaleStart / 32, 0), tla::MakeShape(k / 32, n)));
    } else {
        if (loadA)
            c0a(aa, tla::GetTile(la, tla::MakeCoord(0, 0), tla::MakeShape(M, k)));
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
        // Preserve the first PV partial in L0C for the second MMAD.
    } else if constexpr (!LocalA) {
        static constexpr auto format = CO2Layout::ROW_MAJOR;
        static constexpr FixpipeConfig cfg{format, true};
        FixpipeParamsArch3510<format> fp;
        fp.mSize = HALF;
        fp.nSize = n;
        fp.srcStride = M;
        fp.dstStride = n;
        fp.params.ndNum = 1;
        fp.params.srcNdStride = 0;
        fp.params.dstNdStride = 0;
        fp.quantPre = QuantMode_t::DEQF16;
        fp.deqScalar = DEQ_SCALAR_DEQF16;
        fp.dualDstCtl = 0;
        fp.subBlockId = false;
        auto dst = r.ubBuf.template GetBufferByByte<half>(ubOffset);
        auto src = r.l0CBuf.template GetBufferByByte<int32_t>(l0cOffset);
        Fixpipe<half, int32_t, cfg>(dst, src, fp);
        fp.subBlockId = true;
        Fixpipe<half, int32_t, cfg>(dst, src[HALF * 16], fp);
    } else {
        // No precision conversion: 3510 can split M into both AIV UBs in one call.
        auto u = tla::MakeTensor(r.ubBuf.template GetBufferByByte<float>(ubOffset),
                                 tla::MakeLayout<float, layout::RowMajor>(M, n), Catlass::Arch::PositionUB{});
        Catlass::Gemm::Tile::CopyL0CToUBTla<Arch, decltype(cc), decltype(u),
                                            Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>
            copy;
        copy(u, cc);
    }
    SetFlag<HardEvent::FIX_M>(slot);
}
__aicore__ inline void Notify(uint32_t flag)
{
    SetBothAiv<PIPE_FIX>(flag);
}
} // namespace Sage
