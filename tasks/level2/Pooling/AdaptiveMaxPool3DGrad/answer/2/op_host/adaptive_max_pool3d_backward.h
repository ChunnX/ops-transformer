#ifndef OP_API_INC_LEVEL0_ADAPTIVE_MAX_POOL3D_GRAD_H_
#define OP_API_INC_LEVEL0_ADAPTIVE_MAX_POOL3D_GRAD_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* AdaptiveMaxPool3DGrad(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_ADAPTIVE_MAX_POOL3D_GRAD_H_