#include "aclnn_moe_ep_dispatch.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnnInner_moe_ep_dispatch.h"

using namespace Ops::Transformer;
using namespace op;

extern "C" {

static aclnnStatus CheckNotNull(const aclTensor* context,
                                 const aclTensor* x,
                                 const aclTensor* topkIdx,
                                 const aclTensor* topkWeights,
                                 const aclTensor* cachedHandleDstBufferSlotIdx,
                                 aclTensor* numRecvTokensPerRankHostPinned,
                                 aclTensor* numRecvTokensPerExpertHostPinned,
                                 aclTensor* dstBufferSlotIdx,
                                 aclTensor* recvSrcMetadataInternal) {
    CHECK_RET(context != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(x != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(numRecvTokensPerRankHostPinned != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(numRecvTokensPerExpertHostPinned != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dstBufferSlotIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(recvSrcMetadataInternal != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    bool cachedMode = (cachedHandleDstBufferSlotIdx != nullptr);
    if (!cachedMode) {
        CHECK_RET(topkIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(topkWeights != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(int64_t epWorldSize, int64_t epRankId,
                                int64_t numExperts, int64_t numMaxTokensPerRank,
                                int64_t cclBufferSize,
                                int64_t expertAlignment, bool doCpuSync,
                                const aclTensor* cachedHandleDstBufferSlotIdx) {
    CHECK_RET(epWorldSize > 1, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(epRankId >= 0 && epRankId < epWorldSize, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(numExperts % epWorldSize == 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(numMaxTokensPerRank > 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(cclBufferSize > 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(expertAlignment > 0, ACLNN_ERR_PARAM_INVALID);

    bool cachedMode = (cachedHandleDstBufferSlotIdx != nullptr);
    CHECK_RET(!(cachedMode && doCpuSync), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus MoeEpDispatchGetWorkspaceSize(
    const aclTensor* context, const aclTensor* x,
    const aclTensor* topkIdx, const aclTensor* topkWeights,
    const aclTensor* cachedHandleDstBufferSlotIdx,
    int64_t epWorldSize, int64_t epRankId,
    int64_t numExperts, int64_t numMaxTokensPerRank, int64_t cclBufferSize,
    int64_t expertAlignment, bool doCpuSync,
    aclTensor* numRecvTokensPerRankHostPinned,
    aclTensor* numRecvTokensPerExpertHostPinned,
    aclTensor* dstBufferSlotIdx, aclTensor* recvSrcMetadataInternal,
    uint64_t* workspaceSize, aclOpExecutor** executor) {
    OP_LOGD("MoeEpDispatch", "Begin to do MoeEpDispatchGetWorkspaceSize");
    CHECK_RET(CheckNotNull(context, x, topkIdx, topkWeights,
                            cachedHandleDstBufferSlotIdx,
                            numRecvTokensPerRankHostPinned,
                            numRecvTokensPerExpertHostPinned,
                            dstBufferSlotIdx, recvSrcMetadataInternal),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckParams(epWorldSize, epRankId, numExperts, numMaxTokensPerRank,
                          cclBufferSize, expertAlignment, doCpuSync,
                          cachedHandleDstBufferSlotIdx),
              ACLNN_ERR_PARAM_INVALID);

    return aclnnInnerMoeEpDispatchGetWorkspaceSize(
        context, x, topkIdx, topkWeights,
        cachedHandleDstBufferSlotIdx,
        epWorldSize, epRankId,
        numExperts, numMaxTokensPerRank, cclBufferSize, expertAlignment, doCpuSync,
        numRecvTokensPerRankHostPinned, numRecvTokensPerExpertHostPinned,
        dstBufferSlotIdx, recvSrcMetadataInternal,
        workspaceSize, executor);
}

aclnnStatus aclnnMoeEpDispatchGetWorkspaceSize(
    const aclTensor* context, const aclTensor* x,
    const aclTensor* topkIdx, const aclTensor* topkWeights,
    const aclTensor* cachedHandleDstBufferSlotIdx,
    int64_t epWorldSize, int64_t epRankId,
    int64_t numExperts, int64_t numMaxTokensPerRank, int64_t cclBufferSize,
    int64_t expertAlignment, bool doCpuSync,
    aclTensor* numRecvTokensPerRankHostPinned,
    aclTensor* numRecvTokensPerExpertHostPinned,
    aclTensor* dstBufferSlotIdx, aclTensor* recvSrcMetadataInternal,
    uint64_t* workspaceSize, aclOpExecutor** executor) {
    return MoeEpDispatchGetWorkspaceSize(
        context, x, topkIdx, topkWeights,
        cachedHandleDstBufferSlotIdx,
        epWorldSize, epRankId,
        numExperts, numMaxTokensPerRank, cclBufferSize, expertAlignment, doCpuSync,
        numRecvTokensPerRankHostPinned, numRecvTokensPerExpertHostPinned,
        dstBufferSlotIdx, recvSrcMetadataInternal,
        workspaceSize, executor);
}

enum NnopbaseHcclServerType { NNOPBASE_HCCL_SERVER_TYPE_MTE = 0, NNOPBASE_HCCL_SERVER_TYPE_AICPU = 1 };
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnMoeEpDispatch(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream) {
    NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    return aclnnInnerMoeEpDispatch(workspace, workspaceSize, executor, stream);
}

}  // extern "C"