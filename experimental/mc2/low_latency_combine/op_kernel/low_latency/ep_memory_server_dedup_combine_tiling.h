// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef COMBINE_DEDUP_EP_MEMORY_SERVER_DEDUP_COMBINE_TILING_H
#define COMBINE_DEDUP_EP_MEMORY_SERVER_DEDUP_COMBINE_TILING_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace CombineDedup {
constexpr uint32_t kEpServerDedupCombineCoreNum = 64;

// Independent Host/AICore wire contract. Host validates before narrowing.
struct EpServerDedupCombineTilingData {
    uint32_t bs;
    uint32_t h;
    uint16_t moeExpertNum;
    uint8_t topK;
    uint8_t dtype;
    uint8_t quantMode;
    uint8_t reserved[3];
};
static_assert(std::is_standard_layout<EpServerDedupCombineTilingData>::value, "standard layout tiling");
static_assert(sizeof(EpServerDedupCombineTilingData) == 16, "Combine tiling is 16 bytes");
static_assert(alignof(EpServerDedupCombineTilingData) == 4, "scalar GM alignment");
static_assert(offsetof(EpServerDedupCombineTilingData, bs) == 0, "BS offset");
static_assert(offsetof(EpServerDedupCombineTilingData, h) == 4, "H offset");
static_assert(offsetof(EpServerDedupCombineTilingData, moeExpertNum) == 8, "E offset");
static_assert(offsetof(EpServerDedupCombineTilingData, topK) == 10, "K offset");
static_assert(offsetof(EpServerDedupCombineTilingData, dtype) == 11, "dtype offset");
static_assert(offsetof(EpServerDedupCombineTilingData, quantMode) == 12, "quantMode offset");
static_assert(offsetof(EpServerDedupCombineTilingData, reserved) == 13, "reserved zeros offset");
} // namespace CombineDedup
#endif
