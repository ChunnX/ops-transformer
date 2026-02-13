#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_AVGPOOL3D_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_AVGPOOL3D_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* AvgPool3D(
    const aclTensor* input, const aclIntArray *kernelSize, const aclIntArray *stride, const aclIntArray *pad,
    const bool ceilMode, const bool countIncludePad, const int64_t divisorOverride, const std::string &dataFormat,
    aclOpExecutor* executor);
}

#endif