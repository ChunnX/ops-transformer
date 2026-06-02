#ifndef ACLNN_DISPATCH_NORMAL_H_
#define ACLNN_DISPATCH_NORMAL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnDispatchNormalGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *topkIdx, const aclTensor *sendOffset,
    const aclTensor *sendTokenIdx, const aclTensorList *recvOffset, const aclTensorList *expertOutputOffset,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t cclBufferSize, int64_t quantMode,
    int64_t realMaxBs, int64_t globalBs, int32_t round, int32_t perRoundTokens, const aclTensor *recvX,
    const aclTensor *xScales, const aclTensor *assistInfoForCombine, const aclTensor *dispatchWaitRecvCostStats,
    const aclTensor *extHandleData, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnDispatchNormal(void *workspace, uint64_t workspaceSize,
                                                                       aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
