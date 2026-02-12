#ifndef OP_API_INC_LEVEL2_ACLNN_MAX_POOL3D_GRAD_WITH_ARGMAX_H_
#define OP_API_INC_LEVEL2_ACLNN_MAX_POOL3D_GRAD_WITH_ARGMAX_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus
aclnnMaxPool3dWithArgmaxBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif