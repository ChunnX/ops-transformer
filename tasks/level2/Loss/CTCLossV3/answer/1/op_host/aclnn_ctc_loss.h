#ifndef OP_API_INC_CTCLOSS_H_
#define OP_API_INC_CTCLOSS_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnCtcLossGetWorkspaceSize(
    const aclTensor* logProbs, const aclTensor* targets, const aclIntArray* inputLengths,
    const aclIntArray* targetlengths, int64_t blank, bool zeroInfinity, aclTensor* negLogLikelihoodOut,
    aclTensor* logAlphaOut, uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus
aclnnCtcLoss(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_CTCLOSS_H_