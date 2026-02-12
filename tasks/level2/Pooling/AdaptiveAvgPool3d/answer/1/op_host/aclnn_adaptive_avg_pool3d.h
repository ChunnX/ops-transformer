#ifndef OP_API_INC_ADAPTIVE_AVG_POOL3D_H_
#define OP_API_INC_ADAPTIVE_AVG_POOL3D_H_

#include <array>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAdaptiveAvgPool3d的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnAdaptiveAvgPool3dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnAdaptiveAvgPool3d的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus
aclnnAdaptiveAvgPool3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif