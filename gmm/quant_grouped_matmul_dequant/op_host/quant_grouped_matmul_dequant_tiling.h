/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_matmul_dequant_tiling.h
 * \brief Host-side tiling contract for the unified qgmmd kernel.
 *
 * The TilingData struct is the single source of truth exchanged between
 * the host tiling function and the kernel template-dispatch entry.
 * The bitfield `flags` packs the template-dispatch flags; see
 * op_kernel/quant_grouped_matmul_dequant_tiling_key.h for the bit layout
 * (mirror must stay in sync).
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_QUANT_MATMUL_DEQUANT_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_QUANT_MATMUL_DEQUANT_H
#include <cstdint>
#include <vector>

namespace optiling {

// Core count and tiling-key enum kept from the pre-rewrite code for
// ABI forward-compat (host may emit any of the four keys, but today
// the kernel-side dispatcher routes all of them to the unified class).
constexpr int32_t NUM_OF_AICORE = 8;
constexpr uint32_t BATCH_MODE = 1;

// Fractal / alignment constants used by CheckInOutShapes and the unified
// cost model.
constexpr int32_t FLOAT16_PERBLOCK  = 16;
constexpr int32_t BLOCK_SIZE        = 32;
constexpr int32_t K_FRACTAL_INT8    = 32;
constexpr int32_t NM_FRACTAL_INT8   = 16;
constexpr int32_t L0C_FRACTAL       = 16;
constexpr int32_t L0_ADDR_ALIGN     = 512;

// L2 headroom for the L2-aware cost-model branch.
constexpr uint64_t L2_HEADROOM_BYTES = 14ULL * 1024 * 1024;

// Blocked-ZN / fractal / candidate constants and memory sizes are
// sourced from op_kernel/quant_grouped_matmul_dequant_config.h - that
// file is host-safe (no __aicore__ content). Host budget math uses
// _Params.ubSize / l0ASize / ... (runtime platform query), not the
// compile-time constants; the constants are consumed only for
// SoC-invariant algorithmic values (candidate arrays, MMAD_K_MAX,
// BLOCKED_ZN_NG, SMALL_M_THRESHOLD).

// Kernel never touches workspace during compute, but CANN still
// requires a SYS_WORKSPACE allocation.
constexpr int32_t SYS_WORKSPACE_310P = 2 * 1024 * 1024;

// Flag bit positions (must mirror
// op_kernel/quant_grouped_matmul_dequant_tiling_key.h::QgmmdFlagBit).
constexpr uint32_t QGMMD_FLAG_DYNAMIC_QUANT    = 1u << 0;
constexpr uint32_t QGMMD_FLAG_HAS_SMOOTH       = 1u << 1;
constexpr uint32_t QGMMD_FLAG_PER_TOKEN        = 1u << 2;
constexpr uint32_t QGMMD_FLAG_SCALE_IS_INT64   = 1u << 3;
constexpr uint32_t QGMMD_FLAG_BLOCKED_ZN       = 1u << 4;
constexpr uint32_t QGMMD_FLAG_SMALL_M          = 1u << 5;
constexpr uint32_t QGMMD_FLAG_L2_WEIGHT_FITS   = 1u << 6;
constexpr uint32_t QGMMD_FLAG_HAS_X_RAW_L1     = 1u << 7;
constexpr uint32_t QGMMD_FLAG_USE_L2_HINT      = 1u << 8;
constexpr uint32_t QGMMD_FLAG_DBL0C             = 1u << 9;  // L0C ping-pong across cbs
constexpr uint32_t QGMMD_FLAG_USE_XZN_WORKSPACE = 1u << 10; // xZN GM workspace stage enabled

// Forward-compat tiling keys: the tiling emits one of these, but the
// kernel dispatcher routes all four into the same unified class.
enum class QuantMatmulDequantTilingKey : uint64_t {
  GEMV    = 10000001,
  NORMAL  = 10000002,
  GROUPED = 10000003,
  SWIFT   = 10000004,
  UNDFINED = 10000099
};

// Weight storage-shape rank (auto-detected in CheckInOutShapes).
enum class QgmmdWeightFormat : uint32_t {
    ND              = 3,  // [E, N, K] (grouped) or [N, K] (non-grouped)
    ZN_STANDARD     = 5,  // [E, K/32, N/16, 16, 32] (or non-grouped 4-D)
    ZN_BLOCKED      = 6,  // [E, num_blocks, K/32, blockN1, 16, 32]
    ZN_NON_GROUPED  = 4,  // [K/32, N/16, 16, 32]
};

struct QuantMatmulDequantCompileInfo {
  uint32_t coreNum;
  uint64_t ubSize;
  uint64_t l1Size;
  uint64_t l2Size;
  uint64_t l0CSize;
  uint64_t l0ASize;
  uint64_t l0BSize;
  platform_ascendc::SocVersion socVersion;
  std::string socVersionStr = "";
};

// Internal tiling parameters (host-only; not serialized into TilingData).
struct QuantMatmulDequantParam {
  // Platform
  uint64_t coreNum;
  uint64_t ubSize;
  uint64_t l1Size;
  uint64_t l0ASize;
  uint64_t l0BSize;
  uint64_t l0CSize;

  // Attrs
  bool perToken;
  bool dynamicQuant;
  bool smoothScale;
  bool scaleIsInt64;
  bool hasBias;
  bool xScaleIsHalf;

  // Shape (problem size)
  uint64_t originE;
  uint64_t originM;
  uint64_t originN;
  uint64_t originK;
  uint64_t originKAligned32;
  uint64_t originKAligned512;
  uint64_t fracM;
  uint64_t fracK;
  uint64_t fracN;

  // Weight format
  QgmmdWeightFormat weightFormat;
  uint32_t weightBlockN1;   // 0 (flat) or BLOCKED_ZN_NG

  // Tile pick (filled by cost model)
  uint32_t Mb, Nb, Kb;
  uint32_t kPasses;
  uint32_t wChunkKPasses;       // K-passes coalesced per MTE2 burst (1 = none)
  uint32_t phaseXChunk;
  uint32_t phaseXFullPairs;
  uint32_t phaseXTail;
  uint32_t MCoreNum, NCoreNum;
  uint32_t maxOneTurnToken;
  uint32_t maxWeightColOneTurn;
  uint32_t perCoreIters;
  uint32_t remainderIters;
  uint32_t nBlockSize;          // cbs that fit L2 simultaneously
  uint64_t ubKMask;
  bool     smallM;
  bool     l2WeightFits;
  bool     hasXRawL1;
  bool     useL2Hint;
  bool     dbL0c;               // enable L0C ping-pong across cbs
  bool     useXZNWorkspace;     // stage quantized x as ZN in GM workspace
  uint64_t xZNWorkspaceBytes;   // = alignedM * originKAligned32 when staging enabled, 0 otherwise
};

// ============================================================
// Lean TilingData. Serialized to kernel-side struct by
// REGISTER_TILING_DATA_CLASS. Field order matters - kernel-side tiling
// parser unpacks by offset.
// ============================================================
BEGIN_TILING_DATA_DEF(QuantMatmulDequantTilingData)

// Core layout
TILING_DATA_FIELD_DEF(uint32_t, coreNum);            // blockDim (= 8 on 310P)
TILING_DATA_FIELD_DEF(uint32_t, MCoreNum);           // M-axis cores
TILING_DATA_FIELD_DEF(uint32_t, NCoreNum);           // N-axis cores

// Problem shape
TILING_DATA_FIELD_DEF(uint32_t, M);
TILING_DATA_FIELD_DEF(uint32_t, K);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, E);                  // expert count (1 if non-grouped)
TILING_DATA_FIELD_DEF(uint32_t, originKAligned32);   // ceil(K, 32)
TILING_DATA_FIELD_DEF(uint32_t, originKAligned512);  // ceil(K, 512)
TILING_DATA_FIELD_DEF(uint32_t, fracM);              // ceil(M, 16)
TILING_DATA_FIELD_DEF(uint32_t, fracK);              // ceil(K, 32)
TILING_DATA_FIELD_DEF(uint32_t, fracN);              // ceil(N, 16)

// Tile
TILING_DATA_FIELD_DEF(uint32_t, Mb);                 // tile rows (elements)
TILING_DATA_FIELD_DEF(uint32_t, Nb);                 // tile cols (elements)
TILING_DATA_FIELD_DEF(uint32_t, Kb);                 // tile inner-K (elements)
TILING_DATA_FIELD_DEF(uint32_t, kPasses);            // K-split passes per (cb, tile)
TILING_DATA_FIELD_DEF(uint32_t, wChunkKPasses);      // kp's bundled per MTE2 burst (1 = none)

// Phase X chunk schedule (dynamic-quant slice size + ping/pong pattern)
TILING_DATA_FIELD_DEF(uint32_t, phaseXChunk);
TILING_DATA_FIELD_DEF(uint32_t, phaseXFullPairs);
TILING_DATA_FIELD_DEF(uint32_t, phaseXTail);

// Group / distribution
TILING_DATA_FIELD_DEF(uint32_t, maxOneTurnToken);    // Mb cap per pass
TILING_DATA_FIELD_DEF(uint32_t, maxWeightColOneTurn); // Nb cap per pass
TILING_DATA_FIELD_DEF(uint32_t, perCoreIters);       // iter-level distribution
TILING_DATA_FIELD_DEF(uint32_t, remainderIters);     // cores with an extra iter

// Weight format
TILING_DATA_FIELD_DEF(uint32_t, weightBlockN1);      // 0 or BLOCKED_ZN_NG

// N-block + circular cb iteration. nBlockSize cbs of (Nb*K bytes) of
// weight fit in L2 simultaneously. Cores stagger their cb order by
// (coreIdx % nBlockSize) so all 8 cores fetch DIFFERENT cbs at the same
// step - L2 catches each cb once, the other 7 cores hit L2 (114 B/cyc)
// instead of GM (17 B/cyc, 6.7x).
TILING_DATA_FIELD_DEF(uint32_t, nBlockSize);

// Bit-packed template-dispatch flags (see QGMMD_FLAG_* above).
TILING_DATA_FIELD_DEF(uint32_t, flags);

// K-tail pad mask (bits set on the K lanes past originK) for Phase X cast.
TILING_DATA_FIELD_DEF(uint64_t, ubKMask);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantGroupedMatmulDequant, QuantMatmulDequantTilingData)

class QuantMatmulDequantTiling {
public:
  ge::graphStatus runTiling(gert::TilingContext* context, bool is_grouped = false);
private:
  bool GetPlatformInfo(gert::TilingContext* context);
  bool GetCheckAttr(gert::TilingContext* context);
  bool CheckInOutShapes(gert::TilingContext* context);
  bool GetTilingData(gert::TilingContext* context);
  bool SetTilingData(gert::TilingContext* context);
  bool SetLaunchInfo(gert::TilingContext* context);

  bool isGrouped;
  QuantMatmulDequantTilingKey tilingKey;
  QuantMatmulDequantTilingData tilingData;
  QuantMatmulDequantParam _Params;
};

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_QUANT_MATMUL_DEQUANT_H
