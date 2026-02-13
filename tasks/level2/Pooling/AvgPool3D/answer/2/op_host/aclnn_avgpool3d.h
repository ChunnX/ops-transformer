#ifndef OP_API_INC_AVGPOOL3D_H_
#define OP_API_INC_AVGPOOL3D_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnAvgPool3dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    bool ceilMode, bool countIncludePad, int64_t divisorOverride, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnAvgPool3d(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_AVGPOOL3D_H_