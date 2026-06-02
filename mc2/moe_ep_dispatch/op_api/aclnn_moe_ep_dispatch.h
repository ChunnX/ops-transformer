#ifndef OP_API_INC_MOE_EP_DISPATCH_
#define OP_API_INC_MOE_EP_DISPATCH_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

extern "C" {

/**
 * @brief MoE Expert-Parallel Dispatch: SendPhase + RecvPhase, write HCCL Window
 *        + host-pinned counter (corresponding to DeepEP V2 launch_dispatch).
 *
 * @param [in] context             communication context int32 tensor
 * @param [in] x                   token data [num_tokens, hidden] bf16
 * @param [in] topkIdx             top-k expert indices [num_tokens, top_k] int32; nullptr in cached mode
 * @param [in] topkWeights         top-k weights [num_tokens, top_k] float; nullptr in cached mode
 * @param [in] cachedHandleDstBufferSlotIdx  [num_tokens, top_k] int32; nullptr in non-cached mode
 * @param [in] epWorldSize         EP world size
 * @param [in] epRankId            EP rank id
 * @param [in] numExperts          total expert count
 * @param [in] numMaxTokensPerRank max tokens per rank upper bound
 * @param [in] cclBufferSize       HCCL buffer size
 * @param [in] expertAlignment     alignment for expert token count
 * @param [in] doCpuSync           true: write host-pinned counter for precise alloc
 * @param [out] numRecvTokensPerRankHostPinned   [ep_world_size] int32 host-pinned + dev-mapped
 * @param [out] numRecvTokensPerExpertHostPinned [num_local_experts] int64 host-pinned + dev-mapped
 * @param [out] dstBufferSlotIdx   [num_tokens, top_k] int32 device
 * @param [out] recvSrcMetadataInternal [ep*nmt, 3] int32 device (upper-bound alloc)
 * @param [out] workspaceSize      workspace size output
 * @param [out] executor           executor output
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnMoeEpDispatchGetWorkspaceSize(
    const aclTensor* context,
    const aclTensor* x,
    const aclTensor* topkIdx,
    const aclTensor* topkWeights,
    const aclTensor* cachedHandleDstBufferSlotIdx,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t numExperts,
    int64_t numMaxTokensPerRank,
    int64_t cclBufferSize,
    int64_t expertAlignment,
    bool doCpuSync,
    aclTensor* numRecvTokensPerRankHostPinned,
    aclTensor* numRecvTokensPerExpertHostPinned,
    aclTensor* dstBufferSlotIdx,
    aclTensor* recvSrcMetadataInternal,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief execute moe_ep_dispatch operator
 */
ACLNN_API aclnnStatus aclnnMoeEpDispatch(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);

}  // extern "C"

#endif  // OP_API_INC_MOE_EP_DISPATCH_