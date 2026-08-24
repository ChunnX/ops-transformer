/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ARCH92_CORE_CONTEXT_H
#define MEGA_MOE_ARCH92_CORE_CONTEXT_H

#include "mega_moe_workspace.h"

namespace MegaMoeImpl {

struct MegaMoeArch92ArchitecturePolicy {
    static constexpr bool PERSIST_DISPATCH_CUMSUM = true;
    static constexpr bool USE_GM_HANDOFF = true;
    static constexpr WorkspaceLayoutPolicy WORKSPACE_LAYOUT = WorkspaceLayoutPolicy::GM_HANDOFF_EXPERT_WAVE;

    __aicore__ static inline uint32_t GetPhysicalBlockIdx()
    {
        // KERNEL_TYPE_MIX_AIC_1_1 gives the 1C1V AIC and AIV the same physical block index.
        return static_cast<uint32_t>(GetBlockIdx());
    }
};

} // namespace MegaMoeImpl

#endif // MEGA_MOE_ARCH92_CORE_CONTEXT_H
