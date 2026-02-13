#ifndef OP_API_INC_LEVEL0_MAX_POOL3D_WITH_ARGMAX_V2_H_
#define OP_API_INC_LEVEL0_MAX_POOL3D_WITH_ARGMAX_V2_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<const aclTensor*, const aclTensor*> MaxPool3DWithArgmaxV2Ncdhw(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, std::string dataFormat, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_MAX_POOL3D_WITH_ARGMAX_V2_H_