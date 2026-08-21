/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_mat_mul_allto_allv_v2_tiling.h
 * \brief AIV memory-semantics parameters for GroupedMatMulAlltoAllvV2.
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_V2_TILING_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_V2_TILING_H

#include <cstdint>
#include "../../quant_grouped_mat_mul_allto_allv/op_kernel/quant_grouped_mat_mul_allto_allv_tiling.h"

namespace MC2KernelTemplate {

struct GroupedMatMulAlltoAllvV2Info {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t rankSize;
    uint32_t expertPerRank;
    uint32_t isTransposeB;
    uint32_t maxOutputSize;
    uint64_t commBufferSize;
};

struct GmmA2avCoCTiling {
    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t ubMoveNum;
    int32_t swizzlCount;
    int32_t swizzlDirect;
};

// GroupedMatMulAlltoAllvV2 reuses the quant operator's mature common tiling as
// its base. Keep the V2-only fields in a grouped-op wrapper so the quant
// operator's device ABI is not changed.
struct GroupedMatMulAlltoAllvV2TilingData {
    QuantGmmA2avTilingData baseTiling;
    GroupedMatMulAlltoAllvV2Info info;
    GmmA2avCoCTiling cocTiling;
};

// The first correctness-oriented version derives Info from TaskTilingInfo and
// uses fixed CoC values. These constants are the single source of truth shared
// by CATLASS and AIV communication until host-side tile tuning is introduced.
namespace Gmma2avV2Tiling {
constexpr int32_t L1_TILE_M = 128;
constexpr int32_t L1_TILE_K = 256;
constexpr int32_t L1_TILE_N = 256;
constexpr int32_t L0_TILE_M = 128;
constexpr int32_t L0_TILE_K = 64;
constexpr int32_t L0_TILE_N = 256;
constexpr uint32_t UB_MOVE_ELEMENTS = 32U * 1024U;
constexpr int32_t SWIZZLE_COUNT = 1;
constexpr int32_t SWIZZLE_DIRECTION = 0;
} // namespace Gmma2avV2Tiling

} // namespace MC2KernelTemplate

// REGISTER_TILING_DEFAULT expands the selected tiling type as an unqualified
// name in framework headers, so expose the grouped-op wrapper at global scope.
using GroupedMatMulAlltoAllvV2TilingData = MC2KernelTemplate::GroupedMatMulAlltoAllvV2TilingData;

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_V2_TILING_H
