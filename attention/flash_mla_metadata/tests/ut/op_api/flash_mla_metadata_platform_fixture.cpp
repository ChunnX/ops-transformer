/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/platform.h"

namespace {
struct PlatformResources {
    fe::PlatFormInfos *infos = nullptr;
    uint32_t cubeCoreNum = 0;
    uint32_t vectorCoreNum = 0;
    bool active = false;
};
thread_local PlatformResources resources;
} // namespace

namespace FlashMlaTest {
__attribute__((visibility("default"))) void SetPlatformResources(
    fe::PlatFormInfos *platformInfos, uint32_t cubeCoreNum, uint32_t vectorCoreNum)
{
    resources = {platformInfos, cubeCoreNum, vectorCoreNum, true};
}

__attribute__((visibility("default"))) void ResetPlatformResources()
{
    resources = {};
}
} // namespace FlashMlaTest

// Linker wrappers apply only while this operator's fixture is active.
// All other tests keep using the shared platform stub.
extern "C" {
fe::PlatFormInfos *__real__ZNK2op12PlatformInfo16GetPlatformInfosEv(const op::PlatformInfo *);
uint32_t __real__ZNK2op12PlatformInfo14GetCubeCoreNumEv(const op::PlatformInfo *);
uint32_t __real__ZNK2op12PlatformInfo16GetVectorCoreNumEv(const op::PlatformInfo *);

__attribute__((visibility("default"))) fe::PlatFormInfos *
__wrap__ZNK2op12PlatformInfo16GetPlatformInfosEv(const op::PlatformInfo *platform)
{
    return resources.active ? resources.infos : __real__ZNK2op12PlatformInfo16GetPlatformInfosEv(platform);
}

__attribute__((visibility("default"))) uint32_t
__wrap__ZNK2op12PlatformInfo14GetCubeCoreNumEv(const op::PlatformInfo *platform)
{
    return resources.active ? resources.cubeCoreNum : __real__ZNK2op12PlatformInfo14GetCubeCoreNumEv(platform);
}

__attribute__((visibility("default"))) uint32_t
__wrap__ZNK2op12PlatformInfo16GetVectorCoreNumEv(const op::PlatformInfo *platform)
{
    return resources.active ? resources.vectorCoreNum : __real__ZNK2op12PlatformInfo16GetVectorCoreNumEv(platform);
}
} // extern "C"
