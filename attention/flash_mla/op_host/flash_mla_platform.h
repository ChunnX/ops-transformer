/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_PLATFORM_H_
#define FLASH_MLA_PLATFORM_H_

#include "../op_kernel/flash_mla_schedule_config.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"

namespace FlashMla {
inline bool GetMlaTilingPlatform(const gert::TilingContext *context, MlaPlatformInfo &out)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return false;
    }
    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    // PlatformAscendC groups A2 and A3 under ASCEND910B (arch22).
    if (platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910B) {
        return false;
    }
    out = MlaPlatformInfo{};
    // The framework supplies the effective resources in this invocation's context.
    out.aicNum = platform.GetCoreNumAic();
    out.aivNum = platform.GetCoreNumAiv();
    out.sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    return out.aicNum != 0 && out.aicNum <= kMlaMaxAicNum && (out.aivNum == out.aicNum || out.aivNum == 2 * out.aicNum);
}
} // namespace FlashMla
#endif // FLASH_MLA_PLATFORM_H_
