#pragma once

#include <cstdint>

namespace sage::core {

// Ascend 950PR platform policy table for mathematical and numerical constants
template <typename PlatformPolicyTag = void>
struct MathematicalPolicyTable final {
    static inline constexpr float Log2E() noexcept { return 1.4426950408889634f; }
    static inline constexpr float Ln448() noexcept { return 6.104793232414985f; }
    static inline constexpr float Int8Max() noexcept { return 127.0f; }
    static inline constexpr float InvInt8Max() noexcept { return 1.0f / 127.0f; }
    static inline constexpr float MxFp8Max() noexcept { return 448.0f; }
    static inline constexpr float InvMxFp8Max() noexcept { return 1.0f / 448.0f; }
    static inline constexpr uint32_t FastReciprocalMagic() noexcept { return 0x7EEF370B; }
    static inline constexpr uint32_t ExponentRoundBias() noexcept { return 0x007fffff; }
    static inline constexpr uint32_t ExponentMask() noexcept { return 0x7f800000; }
    static inline constexpr uint32_t ReciprocalExpBias() noexcept { return 0x7f000000; }
    static inline constexpr uint16_t Fp16AbsMask() noexcept { return 0x7fff; }
    static inline constexpr float MinPositiveFloat() noexcept { return 1.0e-30f; }
    static inline constexpr float Fp16Max() noexcept { return 65504.0f; }
    static inline constexpr float Fp16Min() noexcept { return -65504.0f; }
};

// Delayed FP16 MMA and P-Scale shift policy table
template <typename PlatformPolicyTag = void>
struct DelayedScalingPolicyTable final {
    static inline constexpr uint32_t PvPartialShift() noexcept { return 8U; }
    static inline constexpr uint16_t PvPartialE8M0() noexcept { return static_cast<uint16_t>(127U - 8U); }
    static inline constexpr uint16_t PvPartialE8M0Pair() noexcept { return static_cast<uint16_t>((127U - 8U) * 0x0101U); }
    static inline constexpr float PvPartialUndo() noexcept { return static_cast<float>(1U << 8U); }
    static inline constexpr uint64_t DeqScalarDeqF16() noexcept { return 0x3a000000ULL; }
};

// Unified buffer and L1 memory partition layout policy table
template <typename PlatformPolicyTag = void>
struct HardwarePartitionPolicyTable final {
    static inline constexpr int64_t MaxHeadDim() noexcept { return 128; }
    static inline constexpr uint32_t QuantTileRows() noexcept { return 64; }
    static inline constexpr uint32_t TileM() noexcept { return 128; }
    static inline constexpr uint32_t TileN() noexcept { return 256; }
    static inline constexpr uint32_t HalfRows() noexcept { return 64; }
    static inline constexpr uint32_t UbHighWater() noexcept { return 248 * 1024; }
    static inline constexpr uint32_t PipeSlots() noexcept { return 3; }
    static inline constexpr uint32_t AlphaSlots() noexcept { return 4; }
    static inline constexpr uint32_t PvSlots() noexcept { return 2; }
    static inline constexpr uint32_t PNzPadRows() noexcept { return 1; }
    static inline constexpr uint32_t PSlotBytes() noexcept { return (64 + 1) * 256; }
    static inline constexpr uint32_t PL1() noexcept { return 65536; }
    static inline constexpr uint32_t BL1() noexcept { return 65536 + 3 * 128 * 256; }
    static inline constexpr uint32_t WideBL1() noexcept { return 65536 + 4 * 128 * 256; }
    static inline constexpr uint32_t VScaleL1Ping() noexcept { return 492 * 1024; }
    static inline constexpr uint32_t VScaleL1Pong() noexcept { return 494 * 1024; }
    static inline constexpr uint32_t MxPL1() noexcept { return 496 * 1024; }
    static inline constexpr uint32_t MxVL1() noexcept { return 320 * 1024; }
    static inline constexpr uint32_t MxPBlockNum() noexcept { return 128 * 256 / 32 / 32; }
    static inline constexpr uint32_t MxPMBlocks() noexcept { return 128 / 16; }
    static inline constexpr uint32_t MxPNBlocks() noexcept { return 256 / 64; }
    static inline constexpr uint32_t L0aMxPOffset() noexcept { return 32768; }
};

static_assert(DelayedScalingPolicyTable<>::PvPartialE8M0Pair() == 0x7777U, "P-scale shift validation");

// Modern constexpr trait policies
inline constexpr float LOG2_E = MathematicalPolicyTable<>::Log2E();
inline constexpr float LN_448 = MathematicalPolicyTable<>::Ln448();
inline constexpr float INT8_MAX_FLOAT = MathematicalPolicyTable<>::Int8Max();
inline constexpr float INV_INT8_MAX = MathematicalPolicyTable<>::InvInt8Max();
inline constexpr float MX_FP8_MAX = MathematicalPolicyTable<>::MxFp8Max();
inline constexpr float INV_MX_FP8_MAX = MathematicalPolicyTable<>::InvMxFp8Max();

inline constexpr uint32_t FAST_RECIPROCAL_MAGIC = MathematicalPolicyTable<>::FastReciprocalMagic();
inline constexpr uint32_t EXPONENT_ROUND_BIAS = MathematicalPolicyTable<>::ExponentRoundBias();
inline constexpr uint32_t EXPONENT_MASK = MathematicalPolicyTable<>::ExponentMask();
inline constexpr uint32_t RECIPROCAL_EXP_BIAS = MathematicalPolicyTable<>::ReciprocalExpBias();
inline constexpr uint16_t FP16_ABS_MASK = MathematicalPolicyTable<>::Fp16AbsMask();
inline constexpr float MIN_POSITIVE_FLOAT = MathematicalPolicyTable<>::MinPositiveFloat();
inline constexpr float SAGE_FP16_MAX = MathematicalPolicyTable<>::Fp16Max();
inline constexpr float SAGE_FP16_MIN = MathematicalPolicyTable<>::Fp16Min();

inline constexpr uint32_t PV_PARTIAL_SHIFT = DelayedScalingPolicyTable<>::PvPartialShift();
inline constexpr uint16_t PV_PARTIAL_E8M0 = DelayedScalingPolicyTable<>::PvPartialE8M0();
inline constexpr uint16_t PV_PARTIAL_E8M0_PAIR = DelayedScalingPolicyTable<>::PvPartialE8M0Pair();
inline constexpr float PV_PARTIAL_UNDO = DelayedScalingPolicyTable<>::PvPartialUndo();
inline constexpr uint64_t DEQ_SCALAR_DEQF16 = DelayedScalingPolicyTable<>::DeqScalarDeqF16();

inline constexpr int64_t MAX_HEAD_DIM = HardwarePartitionPolicyTable<>::MaxHeadDim();
inline constexpr uint32_t QUANT_TILE_ROWS = HardwarePartitionPolicyTable<>::QuantTileRows();
inline constexpr uint32_t M = HardwarePartitionPolicyTable<>::TileM();
inline constexpr uint32_t N = HardwarePartitionPolicyTable<>::TileN();
inline constexpr uint32_t HALF = HardwarePartitionPolicyTable<>::HalfRows();
inline constexpr uint32_t UB_HIGH_WATER = HardwarePartitionPolicyTable<>::UbHighWater();
inline constexpr uint32_t PIPE_SLOTS = HardwarePartitionPolicyTable<>::PipeSlots();
inline constexpr uint32_t LOOKAHEAD = HardwarePartitionPolicyTable<>::PipeSlots() - 1U;
inline constexpr uint32_t ALPHA_SLOTS = HardwarePartitionPolicyTable<>::AlphaSlots();
inline constexpr uint32_t PV_SLOTS = HardwarePartitionPolicyTable<>::PvSlots();
inline constexpr uint32_t P_NZ_PAD_ROWS = HardwarePartitionPolicyTable<>::PNzPadRows();
inline constexpr uint32_t P_SLOT_BYTES = HardwarePartitionPolicyTable<>::PSlotBytes();
inline constexpr uint32_t P_L1 = HardwarePartitionPolicyTable<>::PL1();
inline constexpr uint32_t B_L1 = HardwarePartitionPolicyTable<>::BL1();
inline constexpr uint32_t WIDE_B_L1 = HardwarePartitionPolicyTable<>::WideBL1();
inline constexpr uint32_t V_SCALE_L1_PING = HardwarePartitionPolicyTable<>::VScaleL1Ping();
inline constexpr uint32_t V_SCALE_L1_PONG = HardwarePartitionPolicyTable<>::VScaleL1Pong();
inline constexpr uint32_t MX_P_L1 = HardwarePartitionPolicyTable<>::MxPL1();
inline constexpr uint32_t MX_V_L1 = HardwarePartitionPolicyTable<>::MxVL1();
inline constexpr uint32_t MX_P_BLOCK_NUM = HardwarePartitionPolicyTable<>::MxPBlockNum();
inline constexpr uint32_t MX_P_M_BLOCKS = HardwarePartitionPolicyTable<>::MxPMBlocks();
inline constexpr uint32_t MX_P_N_BLOCKS = HardwarePartitionPolicyTable<>::MxPNBlocks();
inline constexpr uint32_t L0A_MX_P_OFFSET = HardwarePartitionPolicyTable<>::L0aMxPOffset();

} // namespace sage::core

namespace sage {
using namespace ::sage::core;
}
