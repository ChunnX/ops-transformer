#pragma once

#include <cstdint>

namespace sage::core {

// Mathematical Constants and Transcendental Magic Numbers
constexpr float LOG2_E = 1.4426950408889634f;
constexpr float LN_448 = 6.104793232414985f;
constexpr float INT8_MAX_FLOAT = 127.0f;
constexpr float INV_INT8_MAX = 1.0f / INT8_MAX_FLOAT;
constexpr float MX_FP8_MAX = 448.0f;
constexpr float INV_MX_FP8_MAX = 1.0f / MX_FP8_MAX;

// Taylor-Newton Fast Reciprocal Magic Number and Exponent Masks
constexpr uint32_t FAST_RECIPROCAL_MAGIC = 0x7EEF370B;
constexpr uint32_t EXPONENT_ROUND_BIAS = 0x007fffff;
constexpr uint32_t EXPONENT_MASK = 0x7f800000;
constexpr uint32_t RECIPROCAL_EXP_BIAS = 0x7f000000;
constexpr uint16_t FP16_ABS_MASK = 0x7fff;
constexpr float MIN_POSITIVE_FLOAT = 1.0e-30f;

// Floating-point extreme limits
constexpr float SAGE_FP16_MAX = 65504.0f;
constexpr float SAGE_FP16_MIN = -SAGE_FP16_MAX;

// Delayed FP16 MMA and P-Scale Shift Constants
// To activate native FP16 MMA physical throughput without overflow, P scores
// carry a 2^-8 scale shift during Cube MMAD, which is undone during Softmax normalization.
constexpr uint32_t PV_PARTIAL_SHIFT = 8;
constexpr uint16_t PV_PARTIAL_E8M0 = 127U - PV_PARTIAL_SHIFT;
constexpr uint16_t PV_PARTIAL_E8M0_PAIR = static_cast<uint16_t>(PV_PARTIAL_E8M0 * 0x0101U);
constexpr float PV_PARTIAL_UNDO = static_cast<float>(1U << PV_PARTIAL_SHIFT);
static_assert(PV_PARTIAL_E8M0_PAIR == 0x7777U, "Both P-scale bytes must carry 2^-shift");

// Catlass Fixpipe dequantization scalar: 2^-11 (0x3a000000)
constexpr uint64_t DEQ_SCALAR_DEQF16 = 0x3a000000ULL;

// Hardware Tile Dimensions and Buffer Budgets
constexpr int64_t MAX_HEAD_DIM = 128;
constexpr uint32_t QUANT_TILE_ROWS = 64;
constexpr uint32_t M = 128;
constexpr uint32_t N = 256;
constexpr uint32_t HALF = 64;

// Static Unified Buffer high-water mark (budget: 248 KB)
constexpr uint32_t UB_HIGH_WATER = 248 * 1024;

// Pipeline slot depths
constexpr uint32_t PIPE_SLOTS = 3;
constexpr uint32_t LOOKAHEAD = PIPE_SLOTS - 1;
constexpr uint32_t ALPHA_SLOTS = 4;
constexpr uint32_t PV_SLOTS = 2;
constexpr uint32_t P_NZ_PAD_ROWS = 1;
constexpr uint32_t P_SLOT_BYTES = (HALF + P_NZ_PAD_ROWS) * N;

// L1 memory offsets (Standard path)
constexpr uint32_t P_L1 = 65536;
constexpr uint32_t B_L1 = P_L1 + PIPE_SLOTS * M * N;
constexpr uint32_t WIDE_B_L1 = P_L1 + 4 * M * N;

// Co-streaming V-scale ping-pong offsets (Releases 174 KB of L1 physical capacity)
constexpr uint32_t V_SCALE_L1_PING = 492 * 1024;
constexpr uint32_t V_SCALE_L1_PONG = 494 * 1024;
constexpr uint32_t MX_P_L1 = 496 * 1024;
constexpr uint32_t MX_V_L1 = 320 * 1024;
constexpr uint32_t MX_P_BLOCK_NUM = M * N / 32 / 32;
constexpr uint32_t MX_P_M_BLOCKS = M / 16;
constexpr uint32_t MX_P_N_BLOCKS = N / 64;
constexpr uint32_t L0A_MX_P_OFFSET = 32768;

} // namespace sage::core

namespace sage {
using namespace core;
}
