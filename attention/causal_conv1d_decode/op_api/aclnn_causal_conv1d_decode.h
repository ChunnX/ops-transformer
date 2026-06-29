#ifndef OP_API_INC_CAUSAL_CONV1D_DECODE_H_
#define OP_API_INC_CAUSAL_CONV1D_DECODE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnCausalConv1dDecodeGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, aclTensor *convStates, const aclTensor *queryStartLoc,
    const aclTensor *cacheIndices, const aclTensor *bias, const aclTensor *numAcceptedTokens,
    int64_t activationMode, int64_t padSlotId, int64_t convMode, aclTensor *y, uint64_t *workspaceSize,
    aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnCausalConv1dDecode(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
