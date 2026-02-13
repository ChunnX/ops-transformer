#ifndef OP_API_INC_CTCLOSSBACKWARD_H_
#define OP_API_INC_CTCLOSSBACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnCtcLossBackwardGetWorkspaceSize(
    const aclTensor* gradOut, const aclTensor* logProbs, const aclTensor* targets, const aclIntArray* inputLengths,
    const aclIntArray* targetLengths, const aclTensor* negLogLikelihood, const aclTensor* logAlpha, int64_t blank,
    bool zeroInfinity, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus
aclnnCtcLossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_CTCLOSSBACKWARD_H_