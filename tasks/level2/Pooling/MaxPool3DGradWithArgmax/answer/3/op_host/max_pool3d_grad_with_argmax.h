#ifndef OP_API_INC_LEVEL0_MAX_POOL3D_GRAD_WITH_ARGMAX_H_
#define OP_API_INC_LEVEL0_MAX_POOL3D_GRAD_WITH_ARGMAX_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* MaxPool3DGradWithArgmax(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
    aclOpExecutor* executor);
} // namespace l0op

#endif