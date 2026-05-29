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

static aclnnStatus CheckNotNull(const aclTensor* x,
                                 const aclTensor* topkIdx,
                                 const aclTensor* topkWeights,
                                 const aclTensor* cachedHandleNumRecvTokensPerExpert,
                                 const aclTensor* cachedHandleDstBufferSlotIdx,
                                 const aclTensor* cachedHandleRecvSrcMetadata,
                                 const char* groupEp,
                                 aclTensor* numRecvTokensPerRankHostPinned,
                                 aclTensor* numRecvTokensPerExpertHostPinned,
                                 aclTensor* dstBufferSlotIdx,
                                 aclTensor* recvSrcMetadataInternal) {
    CHECK_RET(x != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(groupEp != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(numRecvTokensPerRankHostPinned != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(numRecvTokensPerExpertHostPinned != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dstBufferSlotIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(recvSrcMetadataInternal != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    bool anyCached = (cachedHandleNumRecvTokensPerExpert != nullptr) ||
                     (cachedHandleDstBufferSlotIdx != nullptr) ||
                     (cachedHandleRecvSrcMetadata != nullptr);
    bool allCached = (cachedHandleNumRecvTokensPerExpert != nullptr) &&
                     (cachedHandleDstBufferSlotIdx != nullptr) &&
                     (cachedHandleRecvSrcMetadata != nullptr);
    CHECK_RET(anyCached == allCached, ACLNN_ERR_PARAM_INVALID);

    if (!allCached) {
        CHECK_RET(topkIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(topkWeights != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(int64_t epWorldSize, int64_t epRankId,
                                int64_t numExperts, int64_t numMaxTokensPerRank,
                                int64_t expertAlignment, bool doCpuSync,
                                const aclTensor* cachedHandleNumRecvTokensPerExpert,
                                const aclTensor* cachedHandleDstBufferSlotIdx,
                                const aclTensor* cachedHandleRecvSrcMetadata) {
    CHECK_RET(epWorldSize > 1, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(epRankId >= 0 && epRankId < epWorldSize, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(numExperts % epWorldSize == 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(numMaxTokensPerRank > 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(expertAlignment > 0, ACLNN_ERR_PARAM_INVALID);

    bool allCached = (cachedHandleNumRecvTokensPerExpert != nullptr) &&
                     (cachedHandleDstBufferSlotIdx != nullptr) &&
                     (cachedHandleRecvSrcMetadata != nullptr);
    CHECK_RET(!(allCached && doCpuSync), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus MoeEpDispatchGetWorkspaceSize(
    const aclTensor* x, const aclTensor* topkIdx, const aclTensor* topkWeights,
    const aclTensor* cachedHandleNumRecvTokensPerExpert,
    const aclTensor* cachedHandleDstBufferSlotIdx,
    const aclTensor* cachedHandleRecvSrcMetadata,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t numExperts, int64_t numMaxTokensPerRank, int64_t expertAlignment,
    bool doCpuSync,
    aclTensor* numRecvTokensPerRankHostPinned,
    aclTensor* numRecvTokensPerExpertHostPinned,
    aclTensor* dstBufferSlotIdx, aclTensor* recvSrcMetadataInternal,
    uint64_t* workspaceSize, aclOpExecutor** executor) {
    OP_LOGD("MoeEpDispatch", "Begin to do MoeEpDispatchGetWorkspaceSize");
    CHECK_RET(CheckNotNull(x, topkIdx, topkWeights,
                            cachedHandleNumRecvTokensPerExpert,
                            cachedHandleDstBufferSlotIdx,
                            cachedHandleRecvSrcMetadata,
                            groupEp, numRecvTokensPerRankHostPinned,
                            numRecvTokensPerExpertHostPinned,
                            dstBufferSlotIdx, recvSrcMetadataInternal),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckParams(epWorldSize, epRankId, numExperts, numMaxTokensPerRank,
                          expertAlignment, doCpuSync,
                          cachedHandleNumRecvTokensPerExpert,
                          cachedHandleDstBufferSlotIdx,
                          cachedHandleRecvSrcMetadata),
              ACLNN_ERR_PARAM_INVALID);

    return aclnnInnerMoeEpDispatchGetWorkspaceSize(
        x, topkIdx, topkWeights,
        cachedHandleNumRecvTokensPerExpert,
        cachedHandleDstBufferSlotIdx,
        cachedHandleRecvSrcMetadata,
        const_cast<char*>(groupEp), epWorldSize, epRankId,
        numExperts, numMaxTokensPerRank, expertAlignment, doCpuSync,
        numRecvTokensPerRankHostPinned, numRecvTokensPerExpertHostPinned,
        dstBufferSlotIdx, recvSrcMetadataInternal,
        workspaceSize, executor);
}

aclnnStatus aclnnMoeEpDispatchGetWorkspaceSize(
    const aclTensor* x, const aclTensor* topkIdx, const aclTensor* topkWeights,
    const aclTensor* cachedHandleNumRecvTokensPerExpert,
    const aclTensor* cachedHandleDstBufferSlotIdx,
    const aclTensor* cachedHandleRecvSrcMetadata,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t numExperts, int64_t numMaxTokensPerRank, int64_t expertAlignment,
    bool doCpuSync,
    aclTensor* numRecvTokensPerRankHostPinned,
    aclTensor* numRecvTokensPerExpertHostPinned,
    aclTensor* dstBufferSlotIdx, aclTensor* recvSrcMetadataInternal,
    uint64_t* workspaceSize, aclOpExecutor** executor) {
    return MoeEpDispatchGetWorkspaceSize(
        x, topkIdx, topkWeights,
        cachedHandleNumRecvTokensPerExpert,
        cachedHandleDstBufferSlotIdx,
        cachedHandleRecvSrcMetadata,
        groupEp, epWorldSize, epRankId,
        numExperts, numMaxTokensPerRank, expertAlignment, doCpuSync,
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