/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef COMMON_NN_ACLNN_UTIL_H
#define COMMON_NN_ACLNN_UTIL_H

#define ACLNN_API __attribute__((visibility("default")))

#include "opdev/platform.h"
#include <set>

namespace Ops {
namespace NN {
namespace AclnnUtil {

using namespace op;

inline static bool IsRegbase()
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    const static std::set<NpuArch> regbaseNpuArchs = {NpuArch::DAV_3510};
    return regbaseNpuArchs.find(npuArch) != regbaseNpuArchs.end();
}

inline static bool IsRegbase(NpuArch npuArch)
{
    const static std::set<NpuArch> regbaseNpuArchs = {NpuArch::DAV_3510};
    return regbaseNpuArchs.find(npuArch) != regbaseNpuArchs.end();
}

} // namespace AclnnUtil
} // namespace NN
} // namespace Ops

#endif
