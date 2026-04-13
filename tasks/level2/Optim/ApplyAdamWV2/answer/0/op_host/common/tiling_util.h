/**
 * Bench：与 `ops-nn-dev/common/inc/tiling_base/tiling_util.h` 接口一致；
 * 当前 CANN 包中 `NpuArch` 枚举与 ops 仓不完全一致时，用本桩保证可编译（regbase 判定保守为 false，与旧 Bench TilingPrepare 中 isRegbase=false 一致）。
 */
#pragma once

#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "common/log.h"

namespace Ops {
namespace NN {
namespace OpTiling {
static const gert::Shape g_vec_1_shape = {1};

static inline bool IsRegbaseSocVersion(const gert::TilingParseContext* context)
{
    (void)context;
    return false;
}

static inline bool IsRegbaseSocVersion(const gert::TilingContext* context)
{
    (void)context;
    return false;
}

inline const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}
} // namespace OpTiling
} // namespace NN
} // namespace Ops
