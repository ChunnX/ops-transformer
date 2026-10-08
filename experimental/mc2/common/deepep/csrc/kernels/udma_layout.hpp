// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Adapted from DeepEP; original copyright and license notices follow.

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of this repository.
 */
#pragma once
#include <cstdint>

namespace ascend_deepep::udma_layout {
// One Session, one immutable SDK configuration. Operators reuse this bank only
// after the previous operator has drained its queues on the ordered stream.
constexpr uint32_t kDispatchOwners = 8U;
constexpr uint32_t kCombineOwners = 16U;
constexpr uint32_t kQpCount = 2U * kCombineOwners;

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__ inline uint32_t CrossQp(uint32_t owner, uint32_t lane)
#else
inline constexpr uint32_t CrossQp(uint32_t owner, uint32_t lane)
#endif
{
    // Matches Combine's 2*owner+lane and the pinned SDK's alternating Clos
    // routes. Lane 0/1 retain their logical 3:1 membership, not new producers.
    // MESH peer QP0 is a different physical SQ; never apply this map to FM.
    return 2U * owner + lane;
}
static_assert(kQpCount == 32U);
static_assert(kDispatchOwners <= kCombineOwners);
} // namespace ascend_deepep::udma_layout
