/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_COUNT_LAYOUT_H
#define MEGA_MOE_COUNT_LAYOUT_H

#include <cstdint>

namespace MegaMoeCountLayout {
// GM requests within one 512-byte region can serialize even for distinct words.
// Only the first int32_t of each MTE slot stores the count/epoch; UB tables stay dense.
constexpr uint32_t SLOT_BYTES = 512U;
// Match the host expert limit and keep the prefix fixed across shape changes so
// route/payload data never occupies an inactive count slot in a reused window.
constexpr int64_t MAX_SLOTS = 2048LL;
constexpr int64_t REGION_BYTES = MAX_SLOTS * SLOT_BYTES;
} // namespace MegaMoeCountLayout

#endif // MEGA_MOE_COUNT_LAYOUT_H
