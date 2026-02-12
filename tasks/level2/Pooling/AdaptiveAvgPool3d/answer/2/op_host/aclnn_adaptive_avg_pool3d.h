#ifndef OP_API_INC_ADAPTIVE_AVG_POOL3D_H_
#define OP_API_INC_ADAPTIVE_AVG_POOL3D_H_

#include <array>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnAdaptiveAvgPool3dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

ACLNN_API aclnnStatus
aclnnAdaptiveAvgPool3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif