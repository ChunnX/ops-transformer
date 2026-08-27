#ifndef OPS_TRANSFORMER_CANNBOTDSL_ACLNN_CUSTOM_FLASH_KDA_H_
#define OPS_TRANSFORMER_CANNBOTDSL_ACLNN_CUSTOM_FLASH_KDA_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * L2 delivery API assembled from the public l0op::FlashKda entry.
 *
 * scale must be finite and representable as float; lowerBound must be in
 * [-5, 0]. On failure, non-null workspaceSize/executor outputs are reset to
 * 0/null.
 */
ACLNN_API aclnnStatus aclnnCustomFlashKdaGetWorkspaceSize(
    const aclTensor* k, const aclTensor* v, const aclTensor* q,
    const aclTensor* beta, const aclTensor* g, const aclTensor* aLog,
    const aclTensor* dtBias, const aclTensor* initialState, double scale,
    double lowerBound, const aclTensor* out, const aclTensor* finalState,
    uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnCustomFlashKda(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OPS_TRANSFORMER_CANNBOTDSL_ACLNN_CUSTOM_FLASH_KDA_H_
