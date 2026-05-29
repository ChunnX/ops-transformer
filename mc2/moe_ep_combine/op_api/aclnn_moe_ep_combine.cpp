#include "aclnn_moe_ep_combine.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnnInner_moe_ep_combine.h"

using namespace Ops::Transformer;
using namespace op;

extern "C" {

static aclnnStatus CheckNotNull(const aclTensor* x,
                                 const aclTensor* dstBufferSlotIdx,
                                 const aclTensor* recvSrcMetadata,
                                 const aclTensor* numRecvTokensPerRank,
                                 const char* groupEp,
                                 aclTensor* combinedX) {
    CHECK_RET(x != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dstBufferSlotIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(recvSrcMetadata != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(numRecvTokensPerRank != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(groupEp != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(combinedX != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(int64_t epWorldSize, int64_t epRankId,
                                int64_t numExperts, int64_t numMaxTokensPerRank,
                                const aclTensor* biasOptional0,
                                const aclTensor* biasOptional1) {
    CHECK_RET(epWorldSize > 1, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(epRankId >= 0 && epRankId < epWorldSize, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(numExperts % epWorldSize == 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(numMaxTokensPerRank > 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(biasOptional0 == nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(biasOptional1 == nullptr, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus MoeEpCombineGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* dstBufferSlotIdx,
    const aclTensor* recvSrcMetadata,
    const aclTensor* numRecvTokensPerRank,
    const aclTensor* topkWeightsOptional,
    const aclTensor* biasOptional0,
    const aclTensor* biasOptional1,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t numExperts, int64_t numMaxTokensPerRank,
    aclTensor* combinedX, aclTensor* combinedTopkWeightsOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor) {
    OP_LOGD("MoeEpCombine", "Begin to do MoeEpCombineGetWorkspaceSize");
    CHECK_RET(CheckNotNull(x, dstBufferSlotIdx, recvSrcMetadata,
                            numRecvTokensPerRank, groupEp, combinedX),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckParams(epWorldSize, epRankId, numExperts, numMaxTokensPerRank,
                          biasOptional0, biasOptional1),
              ACLNN_ERR_PARAM_INVALID);

    return aclnnInnerMoeEpCombineGetWorkspaceSize(
        x, dstBufferSlotIdx, recvSrcMetadata, numRecvTokensPerRank,
        topkWeightsOptional, biasOptional0, biasOptional1,
        const_cast<char*>(groupEp), epWorldSize, epRankId,
        numExperts, numMaxTokensPerRank,
        combinedX, combinedTopkWeightsOptional,
        workspaceSize, executor);
}

aclnnStatus aclnnMoeEpCombineGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* dstBufferSlotIdx,
    const aclTensor* recvSrcMetadata,
    const aclTensor* numRecvTokensPerRank,
    const aclTensor* topkWeightsOptional,
    const aclTensor* biasOptional0,
    const aclTensor* biasOptional1,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t numExperts, int64_t numMaxTokensPerRank,
    aclTensor* combinedX, aclTensor* combinedTopkWeightsOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor) {
    return MoeEpCombineGetWorkspaceSize(
        x, dstBufferSlotIdx, recvSrcMetadata, numRecvTokensPerRank,
        topkWeightsOptional, biasOptional0, biasOptional1,
        groupEp, epWorldSize, epRankId,
        numExperts, numMaxTokensPerRank,
        combinedX, combinedTopkWeightsOptional,
        workspaceSize, executor);
}

enum NnopbaseHcclServerType { NNOPBASE_HCCL_SERVER_TYPE_MTE = 0, NNOPBASE_HCCL_SERVER_TYPE_AICPU = 1 };
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnMoeEpCombine(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream) {
    NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    return aclnnInnerMoeEpCombine(workspace, workspaceSize, executor, stream);
}

}  // extern "C"