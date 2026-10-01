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
    static constexpr event_t K_SYNC_CHANNEL_BASE = static_cast<event_t>(orchestration::EventHandshake::K_PREFETCH_EVENT_BASE);
    static constexpr event_t V_SYNC_CHANNEL_BASE = static_cast<event_t>(orchestration::EventHandshake::V_PREFETCH_EVENT_BASE);

    static __aicore__ inline event_t ResolveKeySyncEvent(uint32_t slotParity) noexcept {
        return static_cast<event_t>(K_SYNC_CHANNEL_BASE + (slotParity & 1U));
    }

    static __aicore__ inline event_t ResolveValSyncEvent(uint32_t slotParity) noexcept {
        return static_cast<event_t>(V_SYNC_CHANNEL_BASE + (slotParity & 1U));
    }
};

// Generic Global-to-L1 matrix tile transfer helper
template <typename Element, typename LayoutB>
struct GlobalToL1TileTransfer {
    using CopyPlan = Catlass::Gemm::Tile::PackedTileCopyTla<
        Catlass::Arch::Ascend950, Element, Catlass::layout::RowMajor,
        Element, LayoutB, float, Catlass::layout::RowMajor>;

    __aicore__ static inline void Transfer(
        FixedResource &res, GM_ADDR gmAddr, uint32_t l1ByteAddr, uint32_t rows, uint32_t cols)
    {
        AscendC::GlobalTensor<Element> srcGmTensor;
        srcGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(gmAddr));
        auto gmTile = tla::MakeTensor(srcGmTensor, tla::MakeLayout<Element, LayoutB>(rows, cols), Catlass::Arch::PositionGM{});
        auto l1Tile = tla::MakeTensor(res.l1Buf.template GetBufferByByte<Element>(l1ByteAddr),
            tla::MakeLayout<Element, typename CopyPlan::LayoutTagL1B>(rows, cols), Catlass::Arch::PositionL1{});
        typename CopyPlan::template CopyGmToL1B<decltype(gmTile)>{}(l1Tile, gmTile);
    }
};

template <typename Element, typename LayoutB>
__aicore__ static inline void CopyGlobalTileToL1(
    FixedResource &res, GM_ADDR gmAddr, uint32_t l1ByteAddr, uint32_t rows, uint32_t cols)
{
    GlobalToL1TileTransfer<Element, LayoutB>::Transfer(res, gmAddr, l1ByteAddr, rows, cols);
}

// Global-to-L1 MX scale slice transfer helper
template <typename ScaleType = float8_e8m0_t>
__aicore__ static inline void CopyGlobalMxScaleToL1(
    FixedResource &res, GM_ADDR gmScaleAddr, uint32_t l1ByteAddr, uint32_t rows, uint32_t cols)
{
    AscendC::GlobalTensor<ScaleType> scaleGmTensor;
    scaleGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ScaleType *>(gmScaleAddr));
    auto gmScaleTile = tla::MakeTensor(scaleGmTensor, tla::MakeMxScaleLayout<ScaleType, Catlass::layout::RowMajor, true>(rows, cols), Catlass::Arch::PositionGM{});
    auto l1ScaleTile = tla::MakeTensor(res.l1Buf.template GetBufferByByte<ScaleType>(l1ByteAddr), tla::MakeMxScaleLayout<ScaleType, Catlass::layout::nN, true>(rows, cols), tla::MakeCoord(0U, 0U), Catlass::Arch::PositionL1{});
    Catlass::Gemm::Tile::TileCopyTla<Catlass::Arch::Ascend950, decltype(gmScaleTile), decltype(l1ScaleTile)>{}(l1ScaleTile, gmScaleTile);
}

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
        return (phase & 1U) ? sage::core::V_SCALE_L1_PONG : sage::core::V_SCALE_L1_PING;
    }

    static __aicore__ inline __attribute__((always_inline))
    void StreamKeyTile(FixedResource &res, GM_ADDR gmKey, uint32_t phase) {
        const event_t syncEvent = WidePipelineSync::ResolveKeySyncEvent(phase);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(syncEvent);

        CopyGlobalTileToL1<int8_t, Catlass::layout::ColumnMajor>(
            res, gmKey, ComputeKeySlotOffset(phase), 128U, sage::core::N);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(syncEvent);
    }

    static __aicore__ inline __attribute__((always_inline))
    void StreamValueTileWithScale(FixedResource &res, GM_ADDR gmVal, GM_ADDR gmValScale, uint32_t phase) {
        const event_t syncEvent = WidePipelineSync::ResolveValSyncEvent(phase);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(syncEvent);

        CopyGlobalTileToL1<float8_e4m3_t, Catlass::layout::RowMajor>(
            res, gmVal, ComputeValSlotOffset(phase), sage::core::N, 128U);

        if (gmValScale != nullptr) {
            CopyGlobalMxScaleToL1(res, gmValScale, ComputeScaleSlotOffset(phase), sage::core::N / 32U, 128U);
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
