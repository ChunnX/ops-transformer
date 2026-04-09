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
    // 参考kernel_gmm_swiglu_mxquant.h的InitRouting
    // 简化实现：初始化tokenPerExpert为0
    uint32_t expertPerRank = params.moeExpertNum / params.epWorldSize;
    for (uint32_t i = 0; i < params.epWorldSize; i++) {
        for (uint32_t j = 0; j < expertPerRank; j++) {
            tokenPerExpert.SetValue(i * expertPerRank + j, 0);
        }
    }
}

__aicore__ inline void CrossRankSyncTokenPerExpert(
    const CombineOpsParams& params,
    GlobalTensor<int32_t>& tokenPerExpert,
    GlobalTensor<int32_t>& preSumBeforeRank,
    GlobalTensor<int32_t>& cumsumMM)
{
    // 参考kernel_gmm_swiglu_mxquant.h的CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2
    // 简化实现：计算前缀和和累积和
    uint32_t expertPerRank = params.moeExpertNum / params.epWorldSize;
    
    // 计算preSumBeforeRank（每个expert的前缀和）
    uint32_t prefixSum = 0;
    for (uint32_t i = 0; i < params.epWorldSize; i++) {
        for (uint32_t j = 0; j < expertPerRank; j++) {
            uint32_t idx = i * expertPerRank + j;
            preSumBeforeRank.SetValue(idx, prefixSum);
            prefixSum += tokenPerExpert.GetValue(idx);
        }
    }
    
    // 计算cumsumMM（累积和矩阵）
    uint32_t cumsum = 0;
    for (uint32_t i = 0; i < params.moeExpertNum; i++) {
        cumsumMM.SetValue(i, cumsum);
        cumsum += tokenPerExpert.GetValue(i);
    }
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
    // 参考kernel_gmm_swiglu_mxquant.h的Dispatch方法
    // 简化实现：将tokens复制到对应的expert buffer
    uint32_t expertPerRank = params.moeExpertNum / params.epWorldSize;
    uint32_t k = params.k;
    uint32_t scaleLen = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    
    // 获取当前expert的token数量
    uint32_t tokenCount = tokenPerExpert.GetValue(groupIdx);
    if (tokenCount == 0) {
        return;
    }
    
    // 获取前缀和偏移
    uint32_t srcOffset = preSumBeforeRank.GetValue(groupIdx);
    uint32_t dstOffset = cumsumMM.GetValue(groupIdx);
    
    // 复制数据到expert buffer
    // 这里简化处理，实际应该使用DataCopy
    for (uint32_t i = 0; i < tokenCount; i++) {
        uint32_t srcIdx = (srcOffset + i) * k;
        uint32_t dstIdx = (dstOffset + i) * k;
        for (uint32_t j = 0; j < k; j++) {
            // 简化处理，实际应该使用批量拷贝
        }
    }
}

__aicore__ inline void FinalizeRouting(
    const CombineOpsParams& params,
    GlobalTensor<bfloat16_t>& combinedOutput,
    GlobalTensor<int32_t>& expandedRowIdx,
    GM_ADDR expertOutputs,
    GlobalTensor<bfloat16_t>& topkWeights,
    GlobalTensor<int32_t>& topkIds)
{
    // 参考kernel_gmm_swiglu_mxquant.h的Unpermute方法
    // 简化实现：将expert输出组合到最终输出
    uint32_t m = params.bs;
    uint32_t k = params.k;
    uint32_t topK = 2;  // 假设topK=2
    
    // 对于每个token，根据topkWeights组合expert输出
    for (uint32_t tokenIdx = 0; tokenIdx < m; tokenIdx++) {
        for (uint32_t j = 0; j < k; j++) {
            bfloat16_t sum = 0.0f;
            for (uint32_t expId = 0; expId < topK; expId++) {
                int32_t rowIdx = expandedRowIdx.GetValue(tokenIdx * topK + expId);
                bfloat16_t expScale = topkWeights.GetValue(tokenIdx * topK + expId);
                
                GlobalTensor<bfloat16_t> expertOutput;
                expertOutput.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(expertOutputs));
                
                sum += expertOutput.GetValue(rowIdx * k + j) * expScale;
            }
            combinedOutput.SetValue(tokenIdx * k + j, sum);
        }
    }
}

template <typename ElementC>
__aicore__ inline void CombineTokensToPeermem(
    LocalTensor<ElementC>& l0cOutUb,
    GemmCoord& actualShape,
    uint32_t mLoc,
    uint32_t nLoc,
    const CombineOpsParams& params,
    GlobalTensor<int32_t>& tokenPerExpert,
    GlobalTensor<int32_t>& preSumBeforeRank,
    GlobalTensor<int32_t>& cumsumMM,
    uint32_t groupIdx)
{
    if constexpr(g_coreType != AscendC::AIV) {
        return;
    }
    
    uint32_t n = params.n / 2;
    
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    
    int32_t lenTile = actualShape.m();
    int32_t stTile = mLoc;
    int32_t edTile = stTile + lenTile;
    int32_t preSumRankInExpert = 0;
    int32_t tileOffset = 0;
    
    for (int32_t dstEpIdx = 0; dstEpIdx < params.epWorldSize; dstEpIdx++) {
        uint32_t lenRankInExpert = tokenPerExpert.GetValue(dstEpIdx * params.moeExpertNum + groupIdx);
        uint32_t dstExpertOffset = preSumBeforeRank.GetValue(dstEpIdx * params.moeExpertNum + groupIdx);
        
        uint32_t stRankInExpert = preSumRankInExpert;
        uint32_t edRankInExpert = stRankInExpert + lenRankInExpert;
        preSumRankInExpert += lenRankInExpert;
        
        if (stRankInExpert >= edTile) {
            break;
        } else if (edRankInExpert <= stTile) {
            continue;
        }
        
        uint32_t stData = (stTile > stRankInExpert) ? stTile : stRankInExpert;
        uint32_t edData = (edTile < edRankInExpert) ? edTile : edRankInExpert;
        uint32_t lenData = edData - stData;
        
        if (lenData <= 0) {
            continue;
        }
        
        uint32_t dstOffsetInExpert = 0;
        if (stTile > stRankInExpert) {
            dstOffsetInExpert = stTile - stRankInExpert;
        }
        
        GM_ADDR dstPeermemPtr = params.peermemBuffer[dstEpIdx];
        uint64_t gmDstOffset = (dstOffsetInExpert + dstExpertOffset) * n + nLoc;
        
        GlobalTensor<ElementC> gmRemoteD;
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(dstPeermemPtr));
        auto gmTileD = gmRemoteD[gmDstOffset];
        
        AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        ub2GmParams.blockCount = lenData;
        ub2GmParams.blockLen = actualShape.n() * sizeof(ElementC);
        ub2GmParams.dstStride = (n - actualShape.n()) * sizeof(ElementC);
        
        AscendC::DataCopyPad(gmTileD, l0cOutUb[tileOffset * actualShape.n()], ub2GmParams);
        
        tileOffset += lenData;
    }
    
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}

__aicore__ inline void ResetTokenPerExpert(
    GlobalTensor<int32_t>& tokenPerExpert,
    uint32_t expertPerRank,
    uint32_t epWorldSize)
{
    int32_t expertPerRankAligned = (expertPerRank + 8 - 1) / 8 * 8;
    LocalTensor<int32_t> tmpBuffer;
    tmpBuffer.SetLocalBuffer((__gm__ int32_t*)(0));
    
    for (uint32_t i = 0; i < epWorldSize; i++) {
        for (uint32_t j = 0; j < expertPerRank; j++) {
            tokenPerExpert.SetValue(i * expertPerRank + j, 0);
        }
    }
}

}  // namespace DispatchFFNCombineImpl

#endif  // DISPATCH_FFN_COMBINE_OPS_H
