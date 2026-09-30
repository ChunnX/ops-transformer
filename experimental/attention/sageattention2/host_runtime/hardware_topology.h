#pragma once

#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include "acl/acl_rt.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace sage::host_runtime {

struct PlatformHardwareSpec {
    uint32_t aicCoreCount = 0;
    uint32_t aivCoreCount = 0;
};

inline bool DiscoverHardwareTopology(const gert::TilingContext *context, uint32_t &aicCores, uint32_t &aivCores)
{
    thread_local static uint32_t tlCachedAic = 0;
    thread_local static uint32_t tlCachedAiv = 0;

    if (tlCachedAic > 0 && tlCachedAiv >= 2 * tlCachedAic) {
        aicCores = tlCachedAic;
        aivCores = tlCachedAiv;
        return true;
    }

    if (context && context->GetPlatformInfo()) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        aicCores = platform.GetCoreNumAic();
        aivCores = platform.GetCoreNumAiv();
    }

    if (!aicCores && context && context->GetCompileInfo()) {
        const auto *spec = static_cast<const PlatformHardwareSpec *>(context->GetCompileInfo());
        aicCores = spec->aicCoreCount;
        aivCores = spec->aivCoreCount;
    }

    if (!aicCores || aivCores < 2 * aicCores) {
        static auto fnGetDevice = reinterpret_cast<aclError (*)(int32_t *)>(dlsym(RTLD_DEFAULT, "aclrtGetDevice"));
        static auto fnGetDevInfo = reinterpret_cast<aclError (*)(uint32_t, aclrtDevAttr, int64_t *)>(
            dlsym(RTLD_DEFAULT, "aclrtGetDeviceInfo"));

        int32_t devId = 0;
        int64_t hwAic = 0, hwAiv = 0;
        if (fnGetDevice && fnGetDevInfo && fnGetDevice(&devId) == ACL_SUCCESS &&
            fnGetDevInfo(devId, ACL_DEV_ATTR_CUBE_CORE_NUM, &hwAic) == ACL_SUCCESS &&
            fnGetDevInfo(devId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &hwAiv) == ACL_SUCCESS &&
            hwAic > 0 && hwAic < UINT32_MAX && hwAiv >= 2 * hwAic && hwAiv < UINT32_MAX) {
            aicCores = static_cast<uint32_t>(hwAic);
            aivCores = static_cast<uint32_t>(hwAiv);
        }
    }

    if (aicCores > 0 && aivCores >= 2 * aicCores) {
        tlCachedAic = aicCores;
        tlCachedAiv = aivCores;
        return true;
    }
    return false;
}

} // namespace sage::host_runtime

namespace optiling {
using sage::host_runtime::PlatformHardwareSpec;
using sage::host_runtime::DiscoverHardwareTopology;
}
