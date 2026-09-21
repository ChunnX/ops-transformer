/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_HOST_ARCH_POLICY_H
#define MEGA_MOE_HOST_ARCH_POLICY_H

#include <cstdint>

namespace optiling {

struct MegaMoeHostArchitecturePolicy {
    bool isArch92{false};
    uint32_t vectorCoreCountPerCube{2U};

    static MegaMoeHostArchitecturePolicy FromNpuArch(int32_t npuArch)
    {
        // 使用平台返回的 DAV 编号，避免 Host 代码依赖 device 侧 __NPU_ARCH__。
        bool arch92 = npuArch == 9201 || npuArch == 9202;
        return {.isArch92 = arch92, .vectorCoreCountPerCube = arch92 ? 1U : 2U};
    }

    bool HasExpectedCvRatio(uint32_t aicNum, uint32_t aivNum) const
    {
        return aivNum == aicNum * vectorCoreCountPerCube;
    }
};

} // namespace optiling

#endif // MEGA_MOE_HOST_ARCH_POLICY_H
