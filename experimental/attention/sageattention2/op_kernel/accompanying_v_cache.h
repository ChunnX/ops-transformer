#pragma once

#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"
#include "credit_handshake.h"
#include "ring_buffer_layout.h"

#if defined(__CCE__)
#include "kernel_operator.h"
#include "catlass/gemm/tile/tile_copy.hpp"

namespace sage::memory {

// Hardware handshake event channels for wide double-buffered staging
struct WidePipelineSync {
    static constexpr event_t K_SYNC_CHANNEL_BASE = static_cast<event_t>(4);
    static constexpr event_t V_SYNC_CHANNEL_BASE = static_cast<event_t>(6);

    static __aicore__ inline event_t ResolveKeySyncEvent(uint32_t slotParity) noexcept {
        return static_cast<event_t>(K_SYNC_CHANNEL_BASE + (slotParity & 1U));
    }

    static __aicore__ inline event_t ResolveValSyncEvent(uint32_t slotParity) noexcept {
        return static_cast<event_t>(V_SYNC_CHANNEL_BASE + (slotParity & 1U));
    }
};

// Accompanying V-cache streaming coordinator
// Co-streams per-tile V scale slice directly into ping-pong buffer, releasing 174 KB L1
template <uint32_t BaseL1Offset>
class AccompanyingVCacheManager {
public:
    static constexpr uint32_t PING_PONG_STEP_BYTES = 65536U;
    static constexpr uint32_t V_TILE_L1_DISPLACEMENT = sage::core::N * 128U;

    static constexpr uint32_t ComputeKeySlotOffset(uint32_t phase) noexcept {
        return BaseL1Offset + (phase & 1U) * PING_PONG_STEP_BYTES;
    }

    static constexpr uint32_t ComputeValSlotOffset(uint32_t phase) noexcept {
        return BaseL1Offset + V_TILE_L1_DISPLACEMENT + (phase & 1U) * PING_PONG_STEP_BYTES;
    }

    static constexpr uint32_t ComputeScaleSlotOffset(uint32_t phase) noexcept {
        return (phase & 1U) ? V_SCALE_L1_PONG : V_SCALE_L1_PING;
    }

    static __aicore__ inline __attribute__((always_inline))
    void StreamKeyTile(FixedResource &res, GM_ADDR gmKey, uint32_t phase) {
        using KeyCopyPlan = Catlass::Gemm::Tile::PackedTileCopyTla<
            Catlass::Arch::Ascend950, int8_t, Catlass::layout::RowMajor,
            int8_t, Catlass::layout::ColumnMajor, float, Catlass::layout::RowMajor>;

        const event_t syncEvent = WidePipelineSync::ResolveKeySyncEvent(phase);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(syncEvent);

        AscendC::GlobalTensor<int8_t> srcGmTensor;
        srcGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(gmKey));
        auto gmTile = tla::MakeTensor(srcGmTensor, tla::MakeLayout<int8_t, Catlass::layout::ColumnMajor>(128, sage::core::N), Catlass::Arch::PositionGM{});

        const uint32_t l1ByteAddr = ComputeKeySlotOffset(phase);
        auto l1Tile = tla::MakeTensor(res.l1Buf.template GetBufferByByte<int8_t>(l1ByteAddr), tla::MakeLayout<int8_t, typename KeyCopyPlan::LayoutTagL1B>(128, sage::core::N), Catlass::Arch::PositionL1{});

        typename KeyCopyPlan::template CopyGmToL1B<decltype(gmTile)>{}(l1Tile, gmTile);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(syncEvent);
    }

    static __aicore__ inline __attribute__((always_inline))
    void StreamValueTileWithScale(FixedResource &res, GM_ADDR gmVal, GM_ADDR gmValScale, uint32_t phase) {
        using ValCopyPlan = Catlass::Gemm::Tile::PackedTileCopyTla<
            Catlass::Arch::Ascend950, float8_e4m3_t, Catlass::layout::RowMajor,
            float8_e4m3_t, Catlass::layout::RowMajor, float, Catlass::layout::RowMajor>;

        const event_t syncEvent = WidePipelineSync::ResolveValSyncEvent(phase);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(syncEvent);

        AscendC::GlobalTensor<float8_e4m3_t> srcGmTensor;
        srcGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float8_e4m3_t *>(gmVal));
        auto gmTile = tla::MakeTensor(srcGmTensor, tla::MakeLayout<float8_e4m3_t, Catlass::layout::RowMajor>(sage::core::N, 128), Catlass::Arch::PositionGM{});

        const uint32_t l1ByteAddr = ComputeValSlotOffset(phase);
        auto l1Tile = tla::MakeTensor(res.l1Buf.template GetBufferByByte<float8_e4m3_t>(l1ByteAddr), tla::MakeLayout<float8_e4m3_t, typename ValCopyPlan::LayoutTagL1B>(sage::core::N, 128), Catlass::Arch::PositionL1{});

        typename ValCopyPlan::template CopyGmToL1B<decltype(gmTile)>{}(l1Tile, gmTile);

        if (gmValScale != nullptr) {
            const uint32_t scaleOffset = ComputeScaleSlotOffset(phase);
            AscendC::GlobalTensor<float8_e8m0_t> scaleGmTensor;
            scaleGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float8_e8m0_t *>(gmValScale));
            auto gmScaleTile = tla::MakeTensor(scaleGmTensor, tla::MakeMxScaleLayout<float8_e8m0_t, Catlass::layout::RowMajor, true>(sage::core::N / 32, 128), Catlass::Arch::PositionGM{});
            auto l1ScaleTile = tla::MakeTensor(res.l1Buf.template GetBufferByByte<float8_e8m0_t>(scaleOffset), tla::MakeMxScaleLayout<float8_e8m0_t, Catlass::layout::nN, true>(sage::core::N / 32, 128), tla::MakeCoord(0, 0), Catlass::Arch::PositionL1{});

            Catlass::Gemm::Tile::TileCopyTla<Catlass::Arch::Ascend950, decltype(gmScaleTile), decltype(l1ScaleTile)>{}(l1ScaleTile, gmScaleTile);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(syncEvent);
    }
};

template <uint32_t BOffset>
__aicore__ inline __attribute__((always_inline)) void PrefetchWideK(FixedResource &r, GM_ADDR b, uint32_t parity) {
    AccompanyingVCacheManager<BOffset>::StreamKeyTile(r, b, parity);
}

template <uint32_t BOffset>
__aicore__ inline __attribute__((always_inline)) void PrefetchWideV(FixedResource &r, GM_ADDR b, GM_ADDR vScale, uint32_t parity) {
    AccompanyingVCacheManager<BOffset>::StreamValueTileWithScale(r, b, vScale, parity);
}

} // namespace sage::memory
#endif // __CCE__
