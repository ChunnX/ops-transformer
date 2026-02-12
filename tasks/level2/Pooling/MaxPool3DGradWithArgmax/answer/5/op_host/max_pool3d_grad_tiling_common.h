#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_SCATTER_TILING_TEMPLATE_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_SCATTER_TILING_TEMPLATE_H
#include <cstdint>
#include "common/log.h"
#include "max_pool3d_grad_tiling_constants.h"

namespace optiling {
inline uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

template <typename T>
inline T CeilAlign(T x, T y)
{
    return y == 0 ? x : (x + y - 1) / y * y;
}

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_SCATTER_TILING_TEMPLATE_H