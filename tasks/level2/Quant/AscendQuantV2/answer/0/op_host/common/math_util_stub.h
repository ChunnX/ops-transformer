/**
 * Bench / msopgen 下替代缺失的 `util/math_util.h` 中 `Ops::Base` 常用函数（见 `.codex/references/compile_debug.md` §3）。
 */
#pragma once

#include "graph/types.h"
#include <cstdint>
#include <string>

namespace Ops {
namespace Base {

inline int64_t CeilDiv(int64_t num, int64_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}

inline int64_t FloorAlign(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a / b) * b;
}

inline int64_t CeilAlign(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

inline std::string ToString(ge::DataType dt)
{
    return std::string("dt_") + std::to_string(static_cast<int32_t>(dt));
}

} // namespace Base
} // namespace Ops
