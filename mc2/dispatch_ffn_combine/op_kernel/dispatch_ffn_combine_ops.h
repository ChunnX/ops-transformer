/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef DISPATCH_FFN_COMBINE_OPS_H
#define DISPATCH_FFN_COMBINE_OPS_H

#include "kernel_operator.h"
#include "gmmsq_base.h"

using namespace AscendC;

namespace DispatchFFNCombineImpl {

struct CombineOpsParams {
    GM_ADDR x;
    GM_ADDR topkIds;
    GM_ADDR topkWeights;
    GM_ADDR weight1;
    GM_ADDR weight2;
    GM_ADDR weightScales1;
    GM_ADDR weightScales2;
    GM_ADDR yOut;
    GM_ADDR expertTokenNumsOut;
    GM_ADDR workspaceGM;
    GM_ADDR peermemBuffer[128];  // 最大128个rank的peermem buffer
    uint32_t epWorldSize;
    uint32_t moeExpertNum;
    uint32_t bs;
    uint32_t h;
    uint32_t n;
    uint32_t k;
    uint32_t rankId;
    
    CombineOpsParams() = default;
};

/**
 * @brief Initialize routing - dispatch tokens to experts
 * @param params Combined parameters
 * @param tokenPerExpert Output tensor for token count per expert
 * @param dispatchBuffer Buffer for dispatched tokens
 */
__aicore__ inline void InitRouting(
    const CombineOpsParams& params,
    GlobalTensor<int32_t>& tokenPerExpert,
    GM_ADDR dispatchBuffer);

/**
 * @brief Synchronize token per expert across ranks
 * @param params Combined parameters
 * @param tokenPerExpert Token count per expert tensor
 * @param preSumBeforeRank Prefix sum before each rank
 * @param cumsumMM Cumulative sum matrix
 */
__aicore__ inline void CrossRankSyncTokenPerExpert(
    const CombineOpsParams& params,
    GlobalTensor<int32_t>& tokenPerExpert,
    GlobalTensor<int32_t>& preSumBeforeRank,
    GlobalTensor<int32_t>& cumsumMM);

/**
 * @brief Dispatch tokens to expert buffers
 * @param params Combined parameters
 * @param groupIdx Current expert group index
 * @param tokenPerExpert Token count per expert
 * @param preSumBeforeRank Prefix sum before each rank
 * @param cumsumMM Cumulative sum matrix
 * @param aGlobal Input tensor
 * @param aScaleGlobal Input scale tensor
 * @param dispatchBuffer Buffer for dispatched tokens
 */
__aicore__ inline void DispatchTokens(
    const CombineOpsParams& params,
    uint32_t groupIdx,
    GlobalTensor<int32_t>& tokenPerExpert,
    GlobalTensor<int32_t>& preSumBeforeRank,
    GlobalTensor<int32_t>& cumsumMM,
    GlobalTensor<bfloat16_t>& aGlobal,
    GlobalTensor<fp8_e8m0_t>& aScaleGlobal,
    GM_ADDR dispatchBuffer);

/**
 * @brief Finalize routing - combine expert outputs
 * @param params Combined parameters
 * @param combinedOutput Combined output tensor
 * @param expandedRowIdx Expanded row indices
 * @param expertOutputs Expert output tensors
 * @param topkWeights Top-k weights
 * @param topkIds Top-k expert IDs
 */
__aicore__ inline void FinalizeRouting(
    const CombineOpsParams& params,
    GlobalTensor<bfloat16_t>& combinedOutput,
    GlobalTensor<int32_t>& expandedRowIdx,
    GM_ADDR expertOutputs,
    GlobalTensor<bfloat16_t>& topkWeights,
    GlobalTensor<int32_t>& topkIds);

/**
 * @brief Reset token per expert buffer
 * @param tokenPerExpert Token count per expert tensor
 * @param expertPerRank Number of experts per rank
 * @param epWorldSize EP world size
 */
__aicore__ inline void ResetTokenPerExpert(
    GlobalTensor<int32_t>& tokenPerExpert,
    uint32_t expertPerRank,
    uint32_t epWorldSize);

// ============================================================================
// Interface implementations (placeholder for external implementation)
// ============================================================================

__aicore__ inline void InitRouting(
    const CombineOpsParams& params,
    GlobalTensor<int32_t>& tokenPerExpert,
    GM_ADDR dispatchBuffer)
{
    // TODO: Implement routing initialization
    // This should call moe_init_routing_v3 if available
}

__aicore__ inline void CrossRankSyncTokenPerExpert(
    const CombineOpsParams& params,
    GlobalTensor<int32_t>& tokenPerExpert,
    GlobalTensor<int32_t>& preSumBeforeRank,
    GlobalTensor<int32_t>& cumsumMM)
{
    // TODO: Implement cross-rank synchronization
    // This should synchronize token counts across EP ranks
}

__aicore__ inline void DispatchTokens(
    const CombineOpsParams& params,
    uint32_t groupIdx,
    GlobalTensor<int32_t>& tokenPerExpert,
    GlobalTensor<int32_t>& preSumBeforeRank,
    GlobalTensor<int32_t>& cumsumMM,
    GlobalTensor<bfloat16_t>& aGlobal,
    GlobalTensor<fp8_e8m0_t>& aScaleGlobal,
    GM_ADDR dispatchBuffer)
{
    // TODO: Implement token dispatch
    // This should dispatch tokens to expert buffers
}

__aicore__ inline void FinalizeRouting(
    const CombineOpsParams& params,
    GlobalTensor<bfloat16_t>& combinedOutput,
    GlobalTensor<int32_t>& expandedRowIdx,
    GM_ADDR expertOutputs,
    GlobalTensor<bfloat16_t>& topkWeights,
    GlobalTensor<int32_t>& topkIds)
{
    // TODO: Implement routing finalization
    // This should call moe_finalize_routing_v2 if available
}

__aicore__ inline void ResetTokenPerExpert(
    GlobalTensor<int32_t>& tokenPerExpert,
    uint32_t expertPerRank,
    uint32_t epWorldSize)
{
    // TODO: Implement token per expert reset
    // This should reset token counts for next iteration
}

}  // namespace DispatchFFNCombineImpl

#endif  // DISPATCH_FFN_COMBINE_OPS_H
