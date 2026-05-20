#include <string.h>
#include "graph/types.h"
#include "aclnn_dispatch_normal.h"
#include "aclnnInner_dispatch_normal.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnDispatchNormalGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *topkIdx, const aclTensor *sendOffset,
    const aclTensor *sendTokenIdx, const aclTensorList *recvOffset, const aclTensorList *expertOutputOffset,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t cclBufferSize, int64_t quantMode,
    int64_t realMaxBs, int64_t globalBs, int32_t round, int32_t perRoundTokens, const aclTensor *recvX,
    const aclTensor *xScales, const aclTensor *assistInfoForCombine, const aclTensor *dispatchWaitRecvCostStats,
    const aclTensor *extHandleData, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerDispatchNormalGetWorkspaceSize(
        context, x, topkIdx, sendOffset, sendTokenIdx, recvOffset, expertOutputOffset, epWorldSize, epRankId,
        moeExpertNum, cclBufferSize, quantMode, realMaxBs, globalBs, round, perRoundTokens, recvX, xScales,
        assistInfoForCombine, dispatchWaitRecvCostStats, extHandleData, workspaceSize, executor);
}

aclnnStatus aclnnDispatchNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerDispatchNormal(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
