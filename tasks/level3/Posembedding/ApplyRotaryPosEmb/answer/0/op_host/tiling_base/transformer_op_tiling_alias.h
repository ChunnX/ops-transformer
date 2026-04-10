/**
 * 将 `.codex/references/common` 中的 `OpTiling::*` 映射到算子源码使用的
 * `Ops::Transformer::OpTiling::*`，避免改业务 tiling 命名空间。
 */
#pragma once

#include "tiling_base.h"
#include "tiling_templates_registry.h"
#include "tiling_util.h"

namespace Ops {
namespace Transformer {
namespace OpTiling {

using TilingBaseClass = ::OpTiling::TilingBaseClass;
using TilingRegistry = ::OpTiling::TilingRegistry;
using TilingCases = ::OpTiling::TilingCases;
using Register = ::OpTiling::Register;
using RegisterNew = ::OpTiling::RegisterNew;
using ::OpTiling::IsRegbaseSocVersion;

} // namespace OpTiling
} // namespace Transformer
} // namespace Ops
