#ifndef COMMON_NN_ACLNN_UTIL_ADD_LAYER_NORM_QUANT_H
#define COMMON_NN_ACLNN_UTIL_ADD_LAYER_NORM_QUANT_H

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
    const static std::set<NpuArch> regbaseNpuArchs = {
        NpuArch::DAV_3510,
        NpuArch::DAV_5102
    };
    return regbaseNpuArchs.find(npuArch) != regbaseNpuArchs.end();
}

inline static bool IsRegbase(NpuArch npuArch)
{
    const static std::set<NpuArch> regbaseNpuArchs = {
        NpuArch::DAV_3510,
        NpuArch::DAV_5102
    };
    return regbaseNpuArchs.find(npuArch) != regbaseNpuArchs.end();
}

inline static bool IsSupportedArch(NpuArch npuArch)
{
    const static std::set<NpuArch> supportedNpuArchs = {
        NpuArch::DAV_3510,
        NpuArch::DAV_5102,
        NpuArch::DAV_2201
    };
    return supportedNpuArchs.find(npuArch) != supportedNpuArchs.end();
}

} // namespace AclnnUtil
} // namespace NN
} // namespace Ops

#endif