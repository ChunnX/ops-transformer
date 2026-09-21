/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_EXECUTION_POLICY_H
#define MEGA_MOE_EXECUTION_POLICY_H

#include "mega_moe_workspace.h"

namespace MegaMoeImpl {

struct MegaMoeDefaultExecutionPolicy {
    static constexpr bool PERSIST_DISPATCH_CUMSUM = false;
    static constexpr bool USE_GM_HANDOFF = false;
    template <bool IsA4W4>
    static constexpr bool FP4_SWIGLU_OUTPUT = false;
    static constexpr WorkspaceLayoutPolicy WORKSPACE_LAYOUT = WorkspaceLayoutPolicy::DEFAULT;
};

// arch92 keeps the expert Wave lifetime without introducing the arch35 W4 prologue.
struct MegaMoeArch92ExecutionPolicy {
    static constexpr bool PERSIST_DISPATCH_CUMSUM = true;
    static constexpr bool USE_GM_HANDOFF = true;
    template <bool IsA4W4>
    static constexpr bool FP4_SWIGLU_OUTPUT = IsA4W4;
    static constexpr WorkspaceLayoutPolicy WORKSPACE_LAYOUT = WorkspaceLayoutPolicy::GM_HANDOFF_EXPERT_WAVE;
};

} // namespace MegaMoeImpl

#endif // MEGA_MOE_EXECUTION_POLICY_H
