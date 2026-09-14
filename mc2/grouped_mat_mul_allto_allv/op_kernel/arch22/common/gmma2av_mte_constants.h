/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// UB regions and transaction invariants for the arch22 MTE pipeline.
#ifndef COMMON_GMMA2AV_MTE_CONSTANTS_H
#define COMMON_GMMA2AV_MTE_CONSTANTS_H

#include "../grouped_mat_mul_allto_allv_mte_tiling.h"

namespace MC2KernelTemplate {

namespace GmmA2avMteDetail {
constexpr uint32_t DEFAULT_UB_MOVE_ELEMENTS = Gmma2avMteTiling::UB_MOVE_ELEMENTS;
constexpr uint32_t USED_UB_BYTES = 192U * 1024U;
constexpr uint32_t CONTROL_UB_OFFSET_BYTES = 64U * 1024U;
constexpr uint32_t DESTINATION_OFFSET_UB_OFFSET_BYTES = 72U * 1024U;
constexpr uint32_t SOURCE_OFFSET_UB_OFFSET_BYTES = 80U * 1024U;
constexpr uint32_t PONG_UB_OFFSET_BYTES = 96U * 1024U;
constexpr uint32_t CONTROL_MTE_BYTES = 32U;
constexpr uint32_t CONTROL_MTE_ELEMENTS = CONTROL_MTE_BYTES / sizeof(int32_t);
constexpr uint32_t SYNC_MTE_ELEMENTS = CONTROL_MTE_BYTES / sizeof(int64_t);
// The control scratch spans [CONTROL_UB_OFFSET_BYTES, DESTINATION_OFFSET_UB_
// OFFSET_BYTES) = 8 KiB. Expert-ready staging reuses it in a disjoint phase
// from the peer-count staging, so the full region is available; one batched
// flag transaction may cover up to 128 consecutive 64-byte slots.
constexpr uint32_t FLAG_STAGING_ELEMENTS =
    (DESTINATION_OFFSET_UB_OFFSET_BYTES - CONTROL_UB_OFFSET_BYTES) / sizeof(int64_t);
constexpr uint32_t SYNC_SLOT_ELEMENTS = Gmma2avMteTiling::SYNC_SLOT_BYTES / sizeof(int64_t);
static_assert(CONTROL_MTE_BYTES % sizeof(int64_t) == 0U,
              "The synchronization transaction must contain complete int64 elements");
constexpr uint32_t INVALID_TOKEN_OFFSET = 0xffffffffU;
static_assert(CONTROL_MTE_BYTES % sizeof(int32_t) == 0U,
              "The control transaction must contain complete int32 elements");
static_assert(Gmma2avMteTiling::SYNC_SLOT_BYTES >= CONTROL_MTE_BYTES,
              "Every synchronization slot must hold one control transaction");
static_assert(Gmma2avMteTiling::PAYLOAD_UB_REGION_BYTES <= CONTROL_UB_OFFSET_BYTES,
              "The payload ping buffer must fit before the control scratch");
static_assert(SOURCE_OFFSET_UB_OFFSET_BYTES % CONTROL_MTE_BYTES == 0U,
              "The source-offset buffer must be control-transaction aligned");
static_assert(CONTROL_UB_OFFSET_BYTES + Gmma2avMteTiling::MAX_COUNT_NUM * sizeof(int32_t) <=
                  DESTINATION_OFFSET_UB_OFFSET_BYTES,
              "The peer count table must fit before the destination-offset buffer");
static_assert(DESTINATION_OFFSET_UB_OFFSET_BYTES + Gmma2avMteTiling::MAX_COUNT_NUM * sizeof(uint32_t) <=
                  SOURCE_OFFSET_UB_OFFSET_BYTES,
              "The destination-offset table must fit before the source-offset buffer");
static_assert(SOURCE_OFFSET_UB_OFFSET_BYTES + Gmma2avMteTiling::MAX_COUNT_NUM * sizeof(uint32_t) <=
                  PONG_UB_OFFSET_BYTES,
              "The source-offset table must fit before the payload pong buffer");
static_assert(PONG_UB_OFFSET_BYTES + Gmma2avMteTiling::PAYLOAD_UB_REGION_BYTES <= USED_UB_BYTES,
              "The payload pong buffer must fit in the allocated UB");
} // namespace GmmA2avMteDetail


} // namespace MC2KernelTemplate

#endif // COMMON_GMMA2AV_MTE_CONSTANTS_H
