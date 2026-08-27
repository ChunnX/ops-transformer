#ifndef OPS_TRANSFORMER_CANNBOTDSL_ACLNN_FLASH_KDA_H_
#define OPS_TRANSFORMER_CANNBOTDSL_ACLNN_FLASH_KDA_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnFlashKdaGetWorkspaceSize(
    const aclTensor* k, const aclTensor* v, const aclTensor* q,
    const aclTensor* beta, const aclTensor* g, const aclTensor* aLog,
    const aclTensor* dtBias, const aclTensor* initialState, double scale,
    double lowerBound, const aclTensor* out, const aclTensor* finalState,
    uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnFlashKda(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OPS_TRANSFORMER_CANNBOTDSL_ACLNN_FLASH_KDA_H_
