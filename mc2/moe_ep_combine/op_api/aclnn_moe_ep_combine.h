#ifndef OP_API_INC_MOE_EP_COMBINE_
#define OP_API_INC_MOE_EP_COMBINE_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

extern "C" {

/**
 * @brief MoE Expert-Parallel Combine: reverse routing + sum (corresponding to DeepEP V2 launch_combine).
 *
 * @param [in] x                     expert-side tokens [A, hidden] bf16
 * @param [in] dstBufferSlotIdx      [num_tokens, top_k] int32 handle field
 * @param [in] recvSrcMetadata       [A, 3] int32 handle field
 * @param [in] numRecvTokensPerRank  [ep_world_size] int32 handle field
 * @param [in] topkWeightsOptional   [num_tokens, top_k] float; nullptr for training forward
 * @param [in] biasOptional0         nullptr (reserved, first release must be null)
 * @param [in] biasOptional1         nullptr (reserved, first release must be null)
 * @param [in] groupEp               HCCL EP group name
 * @param [in] epWorldSize           EP world size
 * @param [in] epRankId              EP rank id
 * @param [in] numExperts            total expert count
 * @param [in] numMaxTokensPerRank   max tokens per rank
 * @param [out] combinedX            [num_tokens, hidden] bf16
 * @param [out] combinedTopkWeightsOptional  [num_tokens, top_k] float; only when topkWeightsOptional != nullptr
 * @param [out] workspaceSize
 * @param [out] executor
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnMoeEpCombineGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* dstBufferSlotIdx,
    const aclTensor* recvSrcMetadata,
    const aclTensor* numRecvTokensPerRank,
    const aclTensor* topkWeightsOptional,
    const aclTensor* biasOptional0,
    const aclTensor* biasOptional1,
    const char* groupEp,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t numExperts,
    int64_t numMaxTokensPerRank,
    aclTensor* combinedX,
    aclTensor* combinedTopkWeightsOptional,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief execute moe_ep_combine operator
 */
ACLNN_API aclnnStatus aclnnMoeEpCombine(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);

}  // extern "C"

#endif  // OP_API_INC_MOE_EP_COMBINE_