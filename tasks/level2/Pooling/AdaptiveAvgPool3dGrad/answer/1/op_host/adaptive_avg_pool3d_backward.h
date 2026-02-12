#ifndef PTA_NPU_OP_API_INC_LEVEL0_ADAPTIVE_AVG_POOL3D_GRAD_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_ADAPTIVE_AVG_POOL3D_GRAD_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* AdaptiveAvgPool3dGrad(const aclTensor* gradOutput, const aclTensor* self, aclOpExecutor* executor);
} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_ADAPTIVE_AVG_POOL3D_GRAD_OP_H_