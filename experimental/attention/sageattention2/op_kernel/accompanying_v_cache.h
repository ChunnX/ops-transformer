#pragma once

#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"
#include "credit_handshake.h"
#include "ring_buffer_layout.h"

#if defined(__CCE__)
#include "kernel_operator.h"
#include "catlass/gemm/tile/tile_copy.hpp"

namespace sage::memory {

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

// Tile-wise Co-streaming V-scale slice into ping-pong buffer: net releases 174 KB L1
template <uint32_t BOffset>
__aicore__ inline __attribute__((always_inline)) void PrefetchWideV(FixedResource &r, GM_ADDR b, GM_ADDR vScale, uint32_t parity) {
    using namespace AscendC;
    using namespace Catlass;
    using namespace tla;
    using T = float8_e4m3_t;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, T, layout::RowMajor, T, layout::RowMajor, float, layout::RowMajor>;
    event_t ev = static_cast<event_t>(6 + (parity & 1));
    WaitFlag<HardEvent::MTE1_MTE2>(ev);
    GlobalTensor<T> bg;
    bg.SetGlobalBuffer((__gm__ T *)b);
    auto gb = tla::MakeTensor(bg, tla::MakeLayout<T, layout::RowMajor>(N, 128), Catlass::Arch::PositionGM{});
    auto lb = tla::MakeTensor(r.l1Buf.template GetBufferByByte<T>(BOffset + N * 128 + (parity & 1) * 65536),
                             tla::MakeLayout<T, typename Copy::LayoutTagL1B>(N, 128), Catlass::Arch::PositionL1{});
    typename Copy::template CopyGmToL1B<decltype(gb)> copy;
    copy(lb, gb);

    // Accompanying V-scale streaming
    if (vScale) {
        uint32_t scaleOffset = (parity & 1) ? V_SCALE_L1_PONG : V_SCALE_L1_PING;
        GlobalTensor<float8_e8m0_t> sg;
        sg.SetGlobalBuffer((__gm__ float8_e8m0_t *)vScale);
        auto gs = tla::MakeTensor(sg, tla::MakeMxScaleLayout<float8_e8m0_t, layout::RowMajor, true>(N / 32, 128),
                                  Catlass::Arch::PositionGM{});
        auto sb = tla::MakeTensor(r.l1Buf.template GetBufferByByte<float8_e8m0_t>(scaleOffset),
                                  tla::MakeMxScaleLayout<float8_e8m0_t, layout::nN, true>(N / 32, 128),
                                  tla::MakeCoord(0, 0), Catlass::Arch::PositionL1{});
        Catlass::Gemm::Tile::TileCopyTla<Arch, decltype(gs), decltype(sb)> copyScale;
        copyScale(sb, gs);
    }
    SetFlag<HardEvent::MTE2_MTE1>(ev);
}

} // namespace sage::memory

namespace sage {
using memory::PrefetchWideK;
using memory::PrefetchWideV;
}

#endif // __CCE__
