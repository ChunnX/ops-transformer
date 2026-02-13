#ifndef OP_API_INC_LEVEL2_ACLNN_ADAPTIVE_MAX_POOL3D_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_ADAPTIVE_MAX_POOL3D_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnAdaptiveMaxPool3dBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput,
    uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus
aclnnAdaptiveMaxPool3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_ADAPTIVE_MAX_POOL3D_BACKWARD_H_