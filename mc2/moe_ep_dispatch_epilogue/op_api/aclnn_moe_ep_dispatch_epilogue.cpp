#include "aclnn_moe_ep_dispatch_epilogue.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnnInner_moe_ep_dispatch_epilogue.h"

using namespace Ops::Transformer;
using namespace op;

extern "C" {

static aclnnStatus CheckNotNull(const aclTensor* context,
                                 const aclTensor* dstBufferSlotIdx,
                                 const aclTensor* recvSrcMetadataInternal,
                                 const aclTensor* numRecvTokensPerRankHostPinned,
                                 const aclTensor* numRecvTokensPerExpertHostPinned,
                                 aclTensor* recvX,
                                 aclTensor* recvTopkIdx,
                                 aclTensor* recvTopkWeights,
                                 aclTensor* recvSrcMetadata,
                                 aclTensor* numRecvTokensPerRank,
                                 aclTensor* numRecvTokensPerExpert) {
    CHECK_RET(context != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dstBufferSlotIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(recvSrcMetadataInternal != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(numRecvTokensPerRankHostPinned != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(numRecvTokensPerExpertHostPinned != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(recvX != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(recvTopkIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(recvTopkWeights != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(recvSrcMetadata != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(numRecvTokensPerRank != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(numRecvTokensPerExpert != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(int64_t epWorldSize, int64_t epRankId,
                                int64_t numExperts, int64_t numMaxTokensPerRank,
                                int64_t cclBufferSize,
                                int64_t expertAlignment) {
    CHECK_RET(epWorldSize > 1, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(epRankId >= 0 && epRankId < epWorldSize, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(numExperts % epWorldSize == 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(numMaxTokensPerRank > 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(cclBufferSize > 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(expertAlignment > 0, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus MoeEpDispatchEpilogueGetWorkspaceSize(
    const aclTensor* context,
    const aclTensor* dstBufferSlotIdx,
    const aclTensor* recvSrcMetadataInternal,
    const aclTensor* numRecvTokensPerRankHostPinned,
    const aclTensor* numRecvTokensPerExpertHostPinned,
    int64_t epWorldSize, int64_t epRankId,
    int64_t numExperts, int64_t numMaxTokensPerRank, int64_t cclBufferSize,
    int64_t expertAlignment,
    aclTensor* recvX, aclTensor* recvTopkIdx, aclTensor* recvTopkWeights,
    aclTensor* recvSrcMetadata, aclTensor* numRecvTokensPerRank,
    aclTensor* numRecvTokensPerExpert,
    uint64_t* workspaceSize, aclOpExecutor** executor) {
    OP_LOGD("MoeEpDispatchEpilogue", "Begin to do MoeEpDispatchEpilogueGetWorkspaceSize");
    CHECK_RET(CheckNotNull(context, dstBufferSlotIdx, recvSrcMetadataInternal,
                            numRecvTokensPerRankHostPinned, numRecvTokensPerExpertHostPinned,
                            recvX, recvTopkIdx, recvTopkWeights,
                            recvSrcMetadata, numRecvTokensPerRank, numRecvTokensPerExpert),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckParams(epWorldSize, epRankId, numExperts, numMaxTokensPerRank,
                          cclBufferSize, expertAlignment),
              ACLNN_ERR_PARAM_INVALID);

    return aclnnInnerMoeEpDispatchEpilogueGetWorkspaceSize(
        context, dstBufferSlotIdx, recvSrcMetadataInternal,
        numRecvTokensPerRankHostPinned, numRecvTokensPerExpertHostPinned,
        epWorldSize, epRankId,
        numExperts, numMaxTokensPerRank, cclBufferSize, expertAlignment,
        recvX, recvTopkIdx, recvTopkWeights,
        recvSrcMetadata, numRecvTokensPerRank, numRecvTokensPerExpert,
        workspaceSize, executor);
}

aclnnStatus aclnnMoeEpDispatchEpilogueGetWorkspaceSize(
    const aclTensor* context,
    const aclTensor* dstBufferSlotIdx,
    const aclTensor* recvSrcMetadataInternal,
    const aclTensor* numRecvTokensPerRankHostPinned,
    const aclTensor* numRecvTokensPerExpertHostPinned,
    int64_t epWorldSize, int64_t epRankId,
    int64_t numExperts, int64_t numMaxTokensPerRank, int64_t cclBufferSize,
    int64_t expertAlignment,
    aclTensor* recvX, aclTensor* recvTopkIdx, aclTensor* recvTopkWeights,
    aclTensor* recvSrcMetadata, aclTensor* numRecvTokensPerRank,
    aclTensor* numRecvTokensPerExpert,
    uint64_t* workspaceSize, aclOpExecutor** executor) {
    return MoeEpDispatchEpilogueGetWorkspaceSize(
        context, dstBufferSlotIdx, recvSrcMetadataInternal,
        numRecvTokensPerRankHostPinned, numRecvTokensPerExpertHostPinned,
        epWorldSize, epRankId,
        numExperts, numMaxTokensPerRank, cclBufferSize, expertAlignment,
        recvX, recvTopkIdx, recvTopkWeights,
        recvSrcMetadata, numRecvTokensPerRank, numRecvTokensPerExpert,
        workspaceSize, executor);
}

enum NnopbaseHcclServerType { NNOPBASE_HCCL_SERVER_TYPE_MTE = 0, NNOPBASE_HCCL_SERVER_TYPE_AICPU = 1 };
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnMoeEpDispatchEpilogue(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream) {
    NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    return aclnnInnerMoeEpDispatchEpilogue(workspace, workspaceSize, executor, stream);
}

}  // extern "C"