#ifndef OP_API_INC_MOE_EP_DISPATCH_EPILOGUE_
#define OP_API_INC_MOE_EP_DISPATCH_EPILOGUE_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

extern "C" {

/**
 * @brief MoE EP Dispatch Epilogue: compact copy from HCCL Window to caller-preallocated
 *        recv_x (corresponding to DeepEP V2 launch_dispatch_copy_epilogue).
 *
* @param [in] context             communication context int32 tensor
 * @param [in] dstBufferSlotIdx           [num_tokens, top_k] int32 from Dispatch
 * @param [in] recvSrcMetadataInternal    [ep*nmt, 3] int32 from Dispatch (upper-bound)
 * @param [in] numRecvTokensPerRankHostPinned   [ep_world_size] int32 host-pinned from Dispatch
 * @param [in] numRecvTokensPerExpertHostPinned  [num_local_experts] int64 host-pinned from Dispatch
 * @param [in] epWorldSize                EP world size
 * @param [in] epRankId                   EP rank id
 * @param [in] numExperts                 total expert count
 * @param [in] numMaxTokensPerRank        max tokens per rank
 * @param [in] cclBufferSize              HCCL buffer size
 * @param [in] expertAlignment            alignment for expert token count
 * @param [out] recvX                     [A_alloc, hidden] bf16 caller preallocated
 * @param [out] recvTopkIdx               [A_alloc, top_k] int32
 * @param [out] recvTopkWeights           [A_alloc, top_k] float
 * @param [out] recvSrcMetadata           [A_alloc, 3] int32 handle field
 * @param [out] numRecvTokensPerRank      [ep_world_size] int32 handle field (device tensor)
 * @param [out] numRecvTokensPerExpert    [num_local_experts] int64 handle field
 * @param [out] workspaceSize
 * @param [out] executor
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnMoeEpDispatchEpilogueGetWorkspaceSize(
    const aclTensor* context,
    const aclTensor* dstBufferSlotIdx,
    const aclTensor* recvSrcMetadataInternal,
    const aclTensor* numRecvTokensPerRankHostPinned,
    const aclTensor* numRecvTokensPerExpertHostPinned,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t numExperts,
    int64_t numMaxTokensPerRank,
    int64_t cclBufferSize,
    int64_t expertAlignment,
    aclTensor* recvX,
    aclTensor* recvTopkIdx,
    aclTensor* recvTopkWeights,
    aclTensor* recvSrcMetadata,
    aclTensor* numRecvTokensPerRank,
    aclTensor* numRecvTokensPerExpert,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief execute moe_ep_dispatch_epilogue operator
 */
ACLNN_API aclnnStatus aclnnMoeEpDispatchEpilogue(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);

}  // extern "C"

#endif  // OP_API_INC_MOE_EP_DISPATCH_EPILOGUE_