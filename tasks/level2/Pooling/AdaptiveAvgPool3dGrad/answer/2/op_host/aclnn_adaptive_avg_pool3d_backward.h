#ifndef OP_API_INC_AdaptiveAvgPool3dBackward_H_
#define OP_API_INC_AdaptiveAvgPool3dBackward_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

ACLNN_API aclnnStatus
aclnnAdaptiveAvgPool3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_AdaptiveAvgPool3dBackward_H_