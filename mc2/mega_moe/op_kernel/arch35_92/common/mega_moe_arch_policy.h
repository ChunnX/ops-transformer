/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ARCH_POLICY_H
#define MEGA_MOE_ARCH_POLICY_H

#include "kernel_operator.h"

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__) || defined(__DAV_310R6_CUBE__) || \
    defined(__DAV_310R6_VEC__) || defined(__DAV_920R1_CUBE__) || defined(__DAV_920R1_VEC__) || \
    defined(__NPU_ARCH__)
#ifndef MEGA_MOE_IS_KERNEL_COMPILE
#define MEGA_MOE_IS_KERNEL_COMPILE
#endif
#endif

namespace MegaMoeImpl {

using namespace AscendC;

// 只描述 CV 配比和物理 core id 映射。Wave、Workspace 和 handoff 属于执行策略，不能塞进这里。
struct MegaMoeArch35HardwarePolicy {
    static constexpr uint32_t VECTOR_CORE_COUNT_PER_CUBE = 2U;

    __aicore__ static uint32_t GetLogicalBlockIdx()
    {
        return GetBlockIdx() / GetTaskRation();
    }

    __aicore__ static uint32_t GetVectorJobIdx()
    {
        return GetBlockIdx();
    }

    __aicore__ static uint32_t GetVectorJobCount()
    {
        return GetBlockNum() * VECTOR_CORE_COUNT_PER_CUBE;
    }

    __aicore__ static uint32_t GetVectorRoleIdx()
    {
        return static_cast<uint32_t>(GetSubBlockIdx());
    }

    __aicore__ static bool RunsComputeVectorWork()
    {
        return GetSubBlockIdx() == 0U;
    }

    __aicore__ static bool RunsDispatchVectorWork()
    {
        return GetSubBlockIdx() == 1U;
    }

    __aicore__ static uint32_t GetFirstVectorPhysicalBlockIdx()
    {
        return GetVectorJobIdx() - GetVectorRoleIdx();
    }
};

// arch92 是 1C1V。routed Wave 保留 A8W4 的 AIV1 职责，删除 AIV0 prologue；通用后处理映射到唯一 AIV。
struct MegaMoeArch92HardwarePolicy {
    static constexpr uint32_t VECTOR_CORE_COUNT_PER_CUBE = 1U;

    __aicore__ static uint32_t GetLogicalBlockIdx()
    {
        return GetBlockIdx();
    }

    __aicore__ static uint32_t GetVectorJobIdx()
    {
        return GetBlockIdx();
    }

    __aicore__ static uint32_t GetVectorJobCount()
    {
        return GetBlockNum();
    }

    __aicore__ static uint32_t GetVectorRoleIdx()
    {
        return 0U;
    }

    __aicore__ static bool RunsComputeVectorWork()
    {
        return true;
    }

    __aicore__ static bool RunsDispatchVectorWork()
    {
        return true;
    }

    __aicore__ static uint32_t GetFirstVectorPhysicalBlockIdx()
    {
        return GetVectorJobIdx();
    }
};

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
#define MEGA_MOE_ARCH92_DEVICE 1
using MegaMoeHardwarePolicy = MegaMoeArch92HardwarePolicy;
#else
using MegaMoeHardwarePolicy = MegaMoeArch35HardwarePolicy;
#endif

} // namespace MegaMoeImpl

#endif // MEGA_MOE_ARCH_POLICY_H
