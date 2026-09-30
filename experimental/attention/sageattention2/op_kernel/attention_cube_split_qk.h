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
#include "attention_block_cube.h"
namespace Sage {
// Each QK half owns a 64 KiB L0C slot. Its Fixpipe can run while the
// following half MMAD uses the other slot; PV retains its existing partition.
constexpr uint32_t QK_SECOND_FIX_EVENT = 6;
template <uint32_t BOffset>
__aicore__ inline __attribute__((always_inline)) void PrefetchWideK(FixedResource &r, GM_ADDR b, uint32_t parity)
{
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
__aicore__ inline __attribute__((always_inline)) void GemmQkSplit(FixedResource &r,
                                                                  uint32_t ubOffset, bool loadA, uint32_t kParity)
{
    using namespace Catlass;
    using namespace tla;
    constexpr uint32_t k = 128, n = N;
    using T = int8_t;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, T, layout::RowMajor, T, layout::ColumnMajor, float,
                                                        layout::RowMajor>;
    WaitFlag<HardEvent::MTE1_MTE2>(QK_EVENT);
    auto la = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(0),
                              tla::MakeLayout<T, typename Copy::LayoutTagL1A>(M, k), Catlass::Arch::PositionL1{});
    auto lb = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(BOffset + kParity * 65536),
                              tla::MakeLayout<T, typename Copy::LayoutTagL1B>(k, n), Catlass::Arch::PositionL1{});
    if (loadA) {
        WaitBothAiv<PIPE_MTE1>(Q_READY);
    }
    event_t kEvent = static_cast<event_t>(4 + kParity);
    WaitFlag<HardEvent::MTE2_MTE1>(kEvent);
    WaitFlag<HardEvent::M_MTE1>(QK_EVENT);
    auto bb = tla::MakeTensor(r.l0BBuf.template GetBufferByByte<T>(0),
                              tla::MakeLayout<T, typename Copy::LayoutTagL0B>(k, n), Catlass::Arch::PositionL0B{});
    typename Copy::CopyL1ToL0B c0b;
    c0b(bb, tla::GetTile(lb, tla::MakeCoord(0, 0), tla::MakeShape(k, n)));
    #pragma unroll
    for (uint32_t sub = 0; sub < 2; ++sub) {
        auto aa =
            tla::MakeTensor(r.l0ABuf.template GetBufferByByte<T>(sub * HALF * k),
                            tla::MakeLayout<T, typename Copy::LayoutTagL0A>(HALF, k), Catlass::Arch::PositionL0A{});
        if (loadA) {
            typename Copy::CopyL1ToL0A c0a;
            c0a(aa, tla::GetTile(la, tla::MakeCoord(sub * HALF, 0), tla::MakeShape(HALF, k)));
        }
    }
    SetFlag<HardEvent::MTE1_MTE2>(kEvent);
    SetFlag<HardEvent::MTE1_MTE2>(QK_EVENT);
    Sync<HardEvent::MTE1_M>();
    #pragma unroll
    for (uint32_t sub = 0; sub < 2; ++sub) {
        event_t ev = static_cast<event_t>(sub ? QK_SECOND_FIX_EVENT : QK_EVENT);
        WaitFlag<HardEvent::FIX_M>(ev);
        auto aa =
            tla::MakeTensor(r.l0ABuf.template GetBufferByByte<T>(sub * HALF * k),
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
        CrossCoreSetFlag<4, PIPE_FIX>(QK_READY + sub * AIV_FLAG_OFFSET);
    }
    SetFlag<HardEvent::M_MTE1>(QK_EVENT);
}
} // namespace Sage
