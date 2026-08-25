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
 * \file grouped_mat_mul_allto_allv_aiv_tiling.h
 * \brief AIV memory-semantics parameters for GroupedMatMulAlltoAllvV2.
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_AIV_TILING_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_AIV_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace MC2KernelTemplate {

constexpr uint32_t GMMA2AV_AIV_MAX_COUNT_NUM = 1024U;

// AIV only consumes the fields below. Keep this definition local instead of
// inheriting TaskTilingInfo from the quant/non-AIV implementation.
struct GroupedMatMulAlltoAllvAivTaskTilingInfo {
    uint64_t BSK;
    uint64_t BS;
    uint64_t H1;
    uint64_t H2;
    uint64_t A;
    uint64_t N1;
    uint64_t N2;
    uint64_t epWorldSize;
    uint64_t e;
    uint64_t aivCoreNum;
    uint64_t aicCoreNum;
    int32_t sendCnt[GMMA2AV_AIV_MAX_COUNT_NUM];
};

struct GmmA2avCoCTiling {
    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t ubMoveNum;
    int32_t swizzlCount;
    int32_t swizzlDirect;
};

// Serialized only for the AIV tiling keys. Non-AIV keys continue to serialize
// QuantGmmA2avTilingData directly and do not share this ABI.
struct GroupedMatMulAlltoAllvAivTilingData {
    // NNopbase reads this header from offset zero to allocate the communicator
    // context passed to the kernel. It is a runtime ABI requirement, not a
    // dependency on the quant/non-AIV tiling wrapper.
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    GroupedMatMulAlltoAllvAivTaskTilingInfo taskTilingInfo;
    uint64_t commBufferSize;
    uint64_t cumsumWorkspaceOffset;
    GmmA2avCoCTiling cocTiling;
};

// The correctness-oriented AIV path uses fixed CoC values. These constants are
// the single source of truth shared by CATLASS and AIV communication until
// host-side tile tuning is introduced.
namespace Gmma2avAivTiling {
constexpr int32_t L1_TILE_M = 128;
constexpr int32_t L1_TILE_K = 256;
constexpr int32_t L1_TILE_N = 256;
constexpr int32_t L0_TILE_M = 128;
constexpr int32_t L0_TILE_K = 64;
constexpr int32_t L0_TILE_N = 256;
constexpr uint32_t UB_MOVE_ELEMENTS = 32U * 1024U;
constexpr int32_t SWIZZLE_COUNT = 1;
constexpr int32_t SWIZZLE_DIRECTION = 0;
constexpr uint64_t WORKSPACE_ALIGNMENT = 512UL;
constexpr uint64_t CONTROL_REGION_BYTES = 16UL * 1024UL * 1024UL;
constexpr uint64_t SEND_COUNT_STAGING_FROM_TAIL = 1024UL * 1024UL;
constexpr uint64_t SYNC_REGION_FROM_TAIL = 512UL * 1024UL;
constexpr uint64_t SYNC_EPOCH_FROM_TAIL = 256UL * 1024UL;
constexpr uint64_t SYNC_SLOT_BYTES = 64UL;
// Three setup barriers plus the data-ready and result-copy-complete barriers.
constexpr uint32_t CROSS_RANK_SYNC_COUNT = 5U;
} // namespace Gmma2avAivTiling

} // namespace MC2KernelTemplate

// GET_TILING_DATA_WITH_STRUCT expands the selected type as an unqualified name
// in some framework versions.
using GroupedMatMulAlltoAllvAivTilingData = MC2KernelTemplate::GroupedMatMulAlltoAllvAivTilingData;

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_AIV_TILING_H
