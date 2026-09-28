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
__aicore__ inline __attribute__((always_inline)) void GemmQkSplit(FixedResource &r, GM_ADDR a,
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
        GlobalTensor<T> ag;
        ag.SetGlobalBuffer((__gm__ T *)a);
        auto ga = tla::MakeTensor(ag, tla::MakeLayout<T, layout::RowMajor>(M, k), Catlass::Arch::PositionGM{});
        typename Copy::template CopyGmToL1A<decltype(ga)> ca;
        ca(la, ga);
        Sync<HardEvent::MTE2_MTE1>();
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
    // Overlap MMAD and Fixpipe: MMAD 0 and MMAD 1 run consecutively on Cube (PIPE_M),
    // while Fixpipe 0 runs concurrently on Fixpipe (PIPE_FIX) during MMAD 1.
    WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(QK_EVENT));
    auto aa0 = tla::MakeTensor(r.l0ABuf.template GetBufferByByte<T>(0),
                               tla::MakeLayout<T, typename Copy::LayoutTagL0A>(HALF, k), Catlass::Arch::PositionL0A{});
    auto cc0 = tla::MakeTensor(r.l0CBuf.template GetBufferByByte<int32_t>(0),
                               tla::MakeLayout<int32_t, layout::L0C>(HALF, n), Catlass::Arch::PositionL0C{});
    Catlass::Gemm::Tile::TileMmadTla<Arch, T, typename Copy::LayoutTagL1A> mma0;
    mma0(cc0, aa0, bb, HALF, n, k, true);
    SetFlag<HardEvent::M_FIX>(static_cast<event_t>(QK_EVENT));

    WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(QK_SECOND_FIX_EVENT));
    auto aa1 = tla::MakeTensor(r.l0ABuf.template GetBufferByByte<T>(HALF * k),
                               tla::MakeLayout<T, typename Copy::LayoutTagL0A>(HALF, k), Catlass::Arch::PositionL0A{});
    auto cc1 = tla::MakeTensor(r.l0CBuf.template GetBufferByByte<int32_t>(HALF * n * 4),
                               tla::MakeLayout<int32_t, layout::L0C>(HALF, n), Catlass::Arch::PositionL0C{});
    Catlass::Gemm::Tile::TileMmadTla<Arch, T, typename Copy::LayoutTagL1A> mma1;
    mma1(cc1, aa1, bb, HALF, n, k, true);
    SetFlag<HardEvent::M_FIX>(static_cast<event_t>(QK_SECOND_FIX_EVENT));
    SetFlag<HardEvent::M_MTE1>(QK_EVENT);

    // Fixpipe for Sub 0:
    WaitFlag<HardEvent::M_FIX>(static_cast<event_t>(QK_EVENT));
    static constexpr FixpipeConfig cfg{CO2Layout::NZ, true};
    FixpipeParamsArch3510<CO2Layout::NZ> fp0;
    fp0.mSize = HALF;
    fp0.nSize = n;
    fp0.srcStride = HALF;
    fp0.dstStride = HALF * 16;
    fp0.quantPre = QuantMode_t::DEQF16;
    fp0.deqScalar = DEQ_SCALAR_DEQF16;
    fp0.dualDstCtl = 0;
    fp0.subBlockId = false;
    auto dst = r.ubBuf.template GetBufferByByte<half>(ubOffset);
    auto src0 = r.l0CBuf.template GetBufferByByte<int32_t>(0);
    Fixpipe<half, int32_t, cfg>(dst, src0, fp0);
    SetFlag<HardEvent::FIX_M>(static_cast<event_t>(QK_EVENT));
    CrossCoreSetFlag<4, PIPE_FIX>(QK_READY);

    // Fixpipe for Sub 1:
    WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(QK_SECOND_FIX_EVENT));
    FixpipeParamsArch3510<CO2Layout::NZ> fp1;
    fp1.mSize = HALF;
    fp1.nSize = n;
    fp1.srcStride = HALF;
    fp1.dstStride = HALF * 16;
    fp1.quantPre = QuantMode_t::DEQF16;
    fp1.deqScalar = DEQ_SCALAR_DEQF16;
    fp1.dualDstCtl = 0;
    fp1.subBlockId = true;
    auto src1 = r.l0CBuf.template GetBufferByByte<int32_t>(HALF * n * 4);
    Fixpipe<half, int32_t, cfg>(dst, src1, fp1);
    SetFlag<HardEvent::FIX_M>(static_cast<event_t>(QK_SECOND_FIX_EVENT));
    CrossCoreSetFlag<4, PIPE_FIX>(QK_READY + AIV_FLAG_OFFSET);
}
} // namespace Sage
