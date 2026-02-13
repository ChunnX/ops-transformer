#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_AVGPOOL3D_BACKWARD_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_AVGPOOL3D_BACKWARD_OP_H_

#include <string>
#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* AvgPool3DGrad(
    const aclTensor* self, const aclTensor* shapeOrigInput, const aclTensor* gradOutput, const aclIntArray* ksize,
    const aclIntArray* strides, const aclIntArray* pads, bool ceilMode, bool countIncludePad, int divisorOverride,
    const std::string& dataFormat, aclOpExecutor* executor);
}

#endif