/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_EP_EXCEPTION_DUMP_DEFS_H
#define MOE_EP_EXCEPTION_DUMP_DEFS_H

#include <cstdint>

constexpr uint32_t MOE_EP_DUMP_LAYOUT_VERSION = 1U;
constexpr uint32_t MOE_EP_DUMP_REGION_COUNT = 6U;
constexpr uint64_t MOE_EP_DUMP_METADATA_BYTES = 64UL * 1024UL;
constexpr uint64_t MOE_EP_PER_CORE_DIAG_SLOT_BYTES = 512UL;
constexpr uint64_t MOE_EP_PER_CORE_DIAG_SLOT_COUNT = 100UL;
constexpr uint64_t MOE_EP_PER_CORE_DIAG_BYTES =
    MOE_EP_PER_CORE_DIAG_SLOT_COUNT * MOE_EP_PER_CORE_DIAG_SLOT_BYTES;
constexpr uint64_t MOE_EP_FIXED_PREFIX_BYTES = MOE_EP_DUMP_METADATA_BYTES + MOE_EP_PER_CORE_DIAG_BYTES;

enum MoeEpDumpRegionIndex : uint32_t {
    MOE_EP_DUMP_REGION_PER_CORE_DIAG = 0U,
    MOE_EP_DUMP_REGION_COUNT_NOTIFY,
    MOE_EP_DUMP_REGION_EXPERT_COUNT,
    MOE_EP_DUMP_REGION_DISPATCH_SLOT_STATE,
    MOE_EP_DUMP_REGION_COMBINE_TOKEN_STATE,
    MOE_EP_DUMP_REGION_HYBRID_SCALEOUT_STATUS,
};

struct MoeEpDumpRegion {
    uint64_t offset;
    uint64_t size;
};

struct MoeEpDumpMetadata {
    uint32_t layoutVersion;
    uint32_t nmt;
    uint32_t topK;
    uint32_t hidden;
    uint32_t epWorldSize;
    uint32_t localExpertNum;
    uint32_t aivNum;
    uint32_t networkMode;
    uint32_t serverNum;
    uint32_t rankNumPerServer;
    MoeEpDumpRegion regions[MOE_EP_DUMP_REGION_COUNT];
};

#endif // MOE_EP_EXCEPTION_DUMP_DEFS_H