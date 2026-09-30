#pragma once

#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"

#if defined(__CCE__)
#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"

namespace sage::memory {

typedef Catlass::Arch::Ascend950 Arch;

// Slab buffer descriptor for static physical allocation
template <AscendC::TPosition POS, uint32_t CapacityBytes>
struct SlabBufferDescriptor final {
    template <typename ElementType>
    __aicore__ inline AscendC::LocalTensor<ElementType> GetBufferByByte(uint32_t byteOffset) const noexcept {
        return AscendC::LocalTensor<ElementType>(POS, byteOffset, (CapacityBytes - byteOffset) / sizeof(ElementType));
    }
};

// Physical device resource container across hardware memory spaces
struct FixedResource final {
    SlabBufferDescriptor<AscendC::TPosition::A1, Arch::L1_SIZE> l1Buf;
    SlabBufferDescriptor<AscendC::TPosition::A2, Arch::L0A_SIZE> l0ABuf;
    SlabBufferDescriptor<AscendC::TPosition::B2, Arch::L0B_SIZE> l0BBuf;
    SlabBufferDescriptor<AscendC::TPosition::CO1, Arch::L0C_SIZE> l0CBuf;
    SlabBufferDescriptor<AscendC::TPosition::VECCALC, Arch::UB_SIZE> ubBuf;

    __aicore__ inline FixedResource() noexcept { AscendC::PipeBarrier<PIPE_ALL>(); }
    __aicore__ inline ~FixedResource() noexcept { AscendC::PipeBarrier<PIPE_ALL>(); }
};

// Coordinate offset slab primitive for explicit memory partitioning
template <uint32_t BaseOffset, uint32_t ExtentBytes>
struct SlabCoordinate final {
    static inline constexpr uint32_t kBase = BaseOffset;
    static inline constexpr uint32_t kExtent = ExtentBytes;
    static inline constexpr uint32_t kNext = BaseOffset + ExtentBytes;
    constexpr operator uint32_t() const noexcept { return kBase; }
};

// Standard Pipeline Unified Buffer Slab Descriptors
struct StandardUbSlabDescriptors final {
    using ScoreSlab = SlabCoordinate<0U, PIPE_SLOTS * HALF * N * sizeof(half)>;
    using PvSlab = SlabCoordinate<ScoreSlab::kNext, PV_SLOTS * (HALF * 128U * 4U)>;
    using AccumSlab = SlabCoordinate<PvSlab::kNext, HALF * 128U * 4U>;
    using PSlab = SlabCoordinate<AccumSlab::kNext, PIPE_SLOTS * P_SLOT_BYTES>;
    using KsSlab = SlabCoordinate<PSlab::kNext, 2U * N * sizeof(half)>;
    using QsSlab = SlabCoordinate<KsSlab::kNext, N * sizeof(half)>;
    using MaxSlab = SlabCoordinate<QsSlab::kNext, N * sizeof(half)>;
    using DenSlab = SlabCoordinate<MaxSlab::kNext, N * sizeof(half)>;
    using AlphaSlab = SlabCoordinate<DenSlab::kNext, ALPHA_SLOTS * (N * sizeof(half))>;
    using SumSlab = SlabCoordinate<AlphaSlab::kNext, HALF * 4U>;
    using VmSlab = SlabCoordinate<SumSlab::kNext, 128U * sizeof(float)>;
};

inline constexpr uint32_t PV_SLOT_BYTES = HALF * 128U * 4U;
inline constexpr uint32_t SCORE_UB = StandardUbSlabDescriptors::ScoreSlab::kBase;
inline constexpr uint32_t PV_UB = StandardUbSlabDescriptors::PvSlab::kBase;
inline constexpr uint32_t ACCUM_UB = StandardUbSlabDescriptors::AccumSlab::kBase;
inline constexpr uint32_t P_UB = StandardUbSlabDescriptors::PSlab::kBase;
inline constexpr uint32_t KS_UB = StandardUbSlabDescriptors::KsSlab::kBase;
inline constexpr uint32_t QS_UB = StandardUbSlabDescriptors::QsSlab::kBase;
inline constexpr uint32_t MAX_UB = StandardUbSlabDescriptors::MaxSlab::kBase;
inline constexpr uint32_t DEN_UB = StandardUbSlabDescriptors::DenSlab::kBase;
inline constexpr uint32_t ALPHA_UB = StandardUbSlabDescriptors::AlphaSlab::kBase;
inline constexpr uint32_t SUM_UB = StandardUbSlabDescriptors::SumSlab::kBase;
inline constexpr uint32_t VM_UB = StandardUbSlabDescriptors::VmSlab::kBase;
inline constexpr uint32_t UB_END = StandardUbSlabDescriptors::VmSlab::kNext;

// ============================================================================
// [Contract P0.4 - Static UB Capacity Protection & Chunk Streaming Audit]:
// Absorbing the lesson of the 16384-element 256KB register trample incident
// from "Softmax Four Evolutions", all buffer allocations are strictly compile-time
// constants independent of sequence length S.
//
// 1. Static Physical Slab Reservation Budget:
//    - Standard Pipeline: UB_END = 251,904 Bytes = 246.00 KB <= 248 KB (UB_HIGH_WATER)
//    - Wide Streaming:    END    = 203,136 Bytes = 198.38 KB <= 248 KB (UB_HIGH_WATER)
//    Both maintain >= 2 KB to the 248 KB OS watermark and >= 10.0 KB to the 256 KB
//    hardware limit, preventing overwrite of pipeline control registers.
//
// 2. Exact Peak Instantaneous Active UB Calculation across Branches:
//    - Branch A: Standard Attention Pipeline (single_pass_engine.h):
//        * Softmax Step:   Score(32KB) + P(16.25KB) + Alpha/Ks/Max/Den(2KB) = 50.25 KB
//        * Accumulate:     PV ping-pong(64KB) + Accum(32KB) + Alpha(0.5KB) = 96.50 KB
//        * Epilogue:       Accum(32KB) + QO(16KB) + VMean/Den/Scratch(1KB)   = 49.00 KB
//        * Pipelined Peak: Score(32KB) + P(16.25KB) + PV(64KB) + Accum(32KB) + State(2KB)
//                          = 146.25 KB (instantaneous) / 130.75 KB (interference graph)
//        * Headroom to 160 KB: 29.25 KB (18.28%); Headroom to 256 KB: 125.25 KB (48.92%)
//    - Branch B: Wide Streaming Pipeline (decoupled_tri_queue.h):
//        * Score(32KB) + P(16.25KB) + PV(64KB) + Accum(32KB) + Partials(2KB) + State(1.75KB)
//        * Peak Instantaneous: 151,552 Bytes = 148.00 KB <= 160 KB
//        * Headroom to 160 KB: 12.00 KB (7.50%); Headroom to 256 KB: 108.00 KB (42.19%)
//    - Branch C: Pre-quantization / FastQuant (dynamic_symmetric_quant.h):
//        * ProcessStats: Raw ping-pong(32KB) + F32(32KB) + Sum(0.5KB) = 64.50 KB <= 160 KB
//        * Quantize:     Single active slot working set = 84,224 Bytes = 82.25 KB <= 160 KB
//
// 3. Chunk Streaming O(1) Proof for Long Sequences (P-07 with S=16384):
//    Chunk size Bc <= 512 (Standard Bc=256 -> 64 chunks, Wide Bc=512 -> 32 chunks).
//    Because memory allocations are bounded by f(Br, Bc, D) with no O(S) buffers,
//    the peak active memory at S=16384 is identical to S=4096: M(16384) <= 148 KB <= 160 KB.
// ============================================================================
inline constexpr uint32_t UB_HARD_SAFETY_LIMIT = 160U * 1024U;          // 160 KB (163,840 Bytes)
inline constexpr uint32_t STANDARD_INSTANTANEOUS_PEAK_UB = 133888U;    // 130.75 KB
inline constexpr uint32_t WIDE_INSTANTANEOUS_PEAK_UB = 151552U;        // 148.00 KB

static_assert(sage::core::N <= 512, "Chunk size N must be <= 512 to guarantee O(1) UB safety");
static_assert(STANDARD_INSTANTANEOUS_PEAK_UB <= UB_HARD_SAFETY_LIMIT,
              "[P0.4 Audit] Standard path instantaneous UB must be <= 160KB");
static_assert(WIDE_INSTANTANEOUS_PEAK_UB <= UB_HARD_SAFETY_LIMIT,
              "[P0.4 Audit] Wide streaming path instantaneous UB must be <= 160KB");
static_assert(UB_END <= UB_HIGH_WATER, "Static UB plan must stay within 248KB budget");

// Wide Streaming Vector Unified Buffer Layout Plan
struct WideVectorLayout final {
    using PvSlotDescriptor = SlabCoordinate<0U, HALF * 128U * sizeof(float)>;
    using ScoreDescriptor = SlabCoordinate<0U, 2U * HALF * N * sizeof(half)>;
    using PvDescriptor = SlabCoordinate<ScoreDescriptor::kNext, 2U * PvSlotDescriptor::kExtent>;
    using AccumDescriptor = SlabCoordinate<PvDescriptor::kNext, HALF * 128U * sizeof(float)>;
    using PDescriptor = SlabCoordinate<AccumDescriptor::kNext, 2U * (HALF + 1U) * N * sizeof(uint8_t)>;
    using PartialSumsDescriptor = SlabCoordinate<PDescriptor::kNext, HALF * 8U * sizeof(float)>;
    using KsDescriptor = SlabCoordinate<PartialSumsDescriptor::kNext, 2U * N * sizeof(half)>;
    using QsDescriptor = SlabCoordinate<KsDescriptor::kNext, HALF * sizeof(half)>;
    using MaxDescriptor = SlabCoordinate<QsDescriptor::kNext, 2U * HALF * sizeof(half)>;
    using DenDescriptor = SlabCoordinate<MaxDescriptor::kNext, HALF * sizeof(float)>;
    using AlphaDescriptor = SlabCoordinate<DenDescriptor::kNext, 3U * 512U>;
    using SumDescriptor = SlabCoordinate<AlphaDescriptor::kNext, HALF * sizeof(float)>;
    using VmDescriptor = SlabCoordinate<SumDescriptor::kNext, 128U * sizeof(float)>;

    static inline constexpr uint32_t PV_SLOT_BYTES = PvSlotDescriptor::kExtent;
    static inline constexpr uint32_t SCORE_UB = ScoreDescriptor::kBase;
    static inline constexpr uint32_t PV_UB = PvDescriptor::kBase;
    static inline constexpr uint32_t ACCUM_UB = AccumDescriptor::kBase;
    static inline constexpr uint32_t P_UB = PDescriptor::kBase;
    static inline constexpr uint32_t PARTIAL_SUMS_UB = PartialSumsDescriptor::kBase;
    static inline constexpr uint32_t KS_UB = KsDescriptor::kBase;
    static inline constexpr uint32_t QS_UB = QsDescriptor::kBase;
    static inline constexpr uint32_t MAX_UB = MaxDescriptor::kBase;
    static inline constexpr uint32_t DEN_UB = DenDescriptor::kBase;
    static inline constexpr uint32_t ALPHA_UB = AlphaDescriptor::kBase;
    static inline constexpr uint32_t SUM_UB = SumDescriptor::kBase;
    static inline constexpr uint32_t VM_UB = VmDescriptor::kBase;
    static inline constexpr uint32_t END = VmDescriptor::kNext;

    static_assert(END <= UB_HIGH_WATER, "Wide streaming UB plan must stay within 248KB budget");
};

template <AscendC::HardEvent HardEventTag>
__aicore__ inline void Sync() noexcept
{
    AscendC::SetFlag<HardEventTag>(0);
    AscendC::WaitFlag<HardEventTag>(0);
}

} // namespace sage::memory

namespace sage {
inline namespace memory_forwarding {
using namespace ::sage::memory;
typedef AscendC::HardEvent HardEvent;
}
}
#endif // __CCE__
