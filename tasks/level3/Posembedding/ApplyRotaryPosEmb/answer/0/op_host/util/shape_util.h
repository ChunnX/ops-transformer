/**
 * Bench：替代缺失的 `util/shape_util.h`，提供 infershape 中 `Ops::Base::ToString(Shape)` / `IsUnknownRank`。
 */
#pragma once

#include <sstream>
#include <string>

#include "exe_graph/runtime/shape.h"

namespace Ops {
namespace Base {

inline bool IsUnknownRank(const gert::Shape& shape)
{
    if (shape.GetDimNum() == 1 && shape.GetDim(0) == -2) {
        return true;
    }
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) < 0) {
            return true;
        }
    }
    return false;
}

inline std::string ToString(const gert::Shape& shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << shape.GetDim(i);
    }
    oss << "]";
    return oss.str();
}

} // namespace Base
} // namespace Ops
