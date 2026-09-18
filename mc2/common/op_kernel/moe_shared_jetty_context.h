/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_SHARED_JETTY_CONTEXT_H
#define MOE_SHARED_JETTY_CONTEXT_H

#include <cstdint>

namespace Mc2Aclnn {
constexpr uint32_t MOE_SHARED_JETTY_NUM = 56;
constexpr uint32_t MOE_SHARED_JETTY_WORLD_SIZE = 8;
constexpr uint32_t MOE_SHARED_JETTY_LANES = MOE_SHARED_JETTY_NUM / (MOE_SHARED_JETTY_WORLD_SIZE - 1);
constexpr uint32_t MOE_SHARED_JETTY_MAX_RANK_SIZE = 1024;

// Each handle shares one local Jetty across peers ordered by rank, excluding self.
struct MoeSharedJettyContext {
    uint32_t epRankId = 0;
    uint32_t sharedJettyNum = 0;
    uint64_t epHcclBuffer[MOE_SHARED_JETTY_MAX_RANK_SIZE] = {};
    uint64_t multiChannelHandles[MOE_SHARED_JETTY_NUM] = {};
};
} // namespace Mc2Aclnn

#endif // MOE_SHARED_JETTY_CONTEXT_H
