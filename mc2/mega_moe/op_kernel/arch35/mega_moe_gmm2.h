/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_gmm2.h
 * \brief Single-core GMM2 implementation and combine post-processing.
 */

#ifndef MEGA_MOE_GMM2_H
#define MEGA_MOE_GMM2_H

#include "kernel_operator.h"
#include "mega_moe_impl.h"

using namespace AscendC;

namespace MegaMoeGmm2Impl {
using Gmm2ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

// 连续逻辑 tile 的调度信息；单核模块不直接推导物理核映射。
struct Gmm2TileSchedule {
    uint32_t tileId;
    uint32_t tileNum;
};

// 当前 expert 的 GMM2 shape、全局地址视图和 group 信息。
struct Gmm2ProblemContext {
    const Gmm2ProblemShape& problemShape;
    const GMMAddrInfo& addrInfo;
    uint32_t expertBeforeCnt;
    uint32_t expertIdx;
};

// GMM2 在多个 expert 之间需要延续的可变运行状态。
struct Gmm2RuntimeState {
    uint32_t& startBlockIdx;
    int32_t& vecSetSyncCom;
    uint16_t& pingPongIdx;
};

// 量化 Combine 后处理使用的调度、shape 和 buffer 信息。
// gmm2SyncCounterAddr 已由调用方按 expert 和物理核映射到唯一 cache line。
struct Gmm2CombineContext {
    Gmm2TileSchedule tileSchedule;
    uint32_t rankId;
    uint32_t k;
    uint32_t mExpert;
    GM_ADDR gmm2SyncCounterAddr;
    int64_t combineUbTensorSize;
    bool enablePostCombine;
};

// 重置 GMM2→Combine 同步计数器所需的全局信息。
struct Gmm2ResetContext {
    Gmm2TileSchedule tileSchedule;
    uint32_t expertPerRank;
    GM_ADDR gmm2SyncCounterAddr;
};

// ===============================================================
// ComputeAlignedTileRange：按逻辑 tile 分割一维区间，并保持每段起点按 align 对齐
// ===============================================================
__aicore__ inline void ComputeAlignedTileRange(const Gmm2TileSchedule& tileSchedule, int32_t totalLength,
    int32_t align, int32_t& tileLength, int32_t& tileOffset)
{
    tileLength = 0;
    tileOffset = 0;
    if (totalLength <= 0 || align <= 0 || tileSchedule.tileNum == 0 ||
        tileSchedule.tileId >= tileSchedule.tileNum) {
        return;
    }
    uint32_t lengthPerTile = Ops::Base::CeilDiv(
        static_cast<uint32_t>(totalLength), tileSchedule.tileNum);
    uint32_t alignedLengthPerTile = Ops::Base::CeilAlign(lengthPerTile, static_cast<uint32_t>(align));
    uint32_t offset = tileSchedule.tileId * alignedLengthPerTile;
    if (offset >= static_cast<uint32_t>(totalLength)) {
        return;
    }
    tileOffset = static_cast<int32_t>(offset);
    tileLength = static_cast<int32_t>((alignedLengthPerTile < static_cast<uint32_t>(totalLength) - offset) ?
        alignedLengthPerTile : static_cast<uint32_t>(totalLength) - offset);
}

// ==================================================================================
// ResetGmm2CombineSyncCountersSingleCore：每个逻辑 tile 清零自己负责的计数器区间
// ==================================================================================
__aicore__ inline void ResetGmm2CombineSyncCountersSingleCore(const Gmm2ResetContext& resetContext,
    LocalTensor<int32_t>& resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    int32_t totalCounters = static_cast<int32_t>(static_cast<int64_t>(resetContext.expertPerRank) *
        resetContext.tileSchedule.tileNum * INT_CACHELINE);
    int32_t tileLength = 0;
    int32_t tileOffset = 0;
    ComputeAlignedTileRange(resetContext.tileSchedule, totalCounters, ALIGN_32, tileLength, tileOffset);
    if (tileLength == 0) {
        return;
    }
    GlobalTensor<int32_t> gmm2CombineSyncCounter;
    gmm2CombineSyncCounter.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(resetContext.gmm2SyncCounterAddr));
    Duplicate(resetTensor, 0, tileLength);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    DataCopy(gmm2CombineSyncCounter[tileOffset], resetTensor, tileLength);
}

// ==================================================================================
// ComputeCombineUbTensorSize：计算量化 Combine 后处理在 UB 中需要的元素数
// ==================================================================================
template <int32_t CombineQuantMode>
__aicore__ inline int64_t ComputeCombineUbTensorSize(uint32_t k)
{
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT || g_coreType != AIV) {
        return 0;
    }
    uint32_t nAlign32 = Ops::Base::CeilAlign(k, static_cast<uint32_t>(ALIGN_32));
    uint32_t nScale = Ops::Base::CeilDiv(k, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
    uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k + nScale, static_cast<uint32_t>(ALIGN_32));
    uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantTokenSizeBytes;
    return static_cast<int64_t>((singleTokenBytes * 2) / sizeof(bfloat16_t));
}

// ==================================================================================
// RunGmm2MatmulSingleCore：按 A8W4/A4W4、NZ/ND 路径选择底层 GMM2 单核实现
// ==================================================================================
template <int32_t CombineQuantMode, typename QuantOutType, typename SwigluQuantOutType, typename WeightType,
    typename QuantScaleType, bool EnableA8W4, bool EnableA4W4>
__aicore__ inline void RunGmm2MatmulSingleCore(const Params& params, const Gmm2ProblemContext& problemContext,
    Gmm2RuntimeState& runtimeState)
{
    if constexpr (EnableA8W4 || EnableA4W4) {
        MegaMoeImpl::GroupMatmul2CombineA8W4<CombineQuantMode, SwigluQuantOutType, WeightType, bfloat16_t,
            QuantScaleType, QuantScaleType>(params, problemContext.problemShape, problemContext.addrInfo,
            runtimeState.startBlockIdx, runtimeState.vecSetSyncCom, problemContext.expertBeforeCnt,
            runtimeState.pingPongIdx, problemContext.expertIdx);
    } else {
        // A8W8_NZ 和 generic ND 共用 GroupMatmul2，仅 IsWeightNZ 模板参数不同。
        if (params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleType, QuantScaleType, true>(params, problemContext.problemShape, problemContext.addrInfo,
                runtimeState.startBlockIdx, runtimeState.vecSetSyncCom, problemContext.expertBeforeCnt,
                runtimeState.pingPongIdx, problemContext.expertIdx);
        } else {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleType, QuantScaleType>(params, problemContext.problemShape, problemContext.addrInfo,
                runtimeState.startBlockIdx, runtimeState.vecSetSyncCom, problemContext.expertBeforeCnt,
                runtimeState.pingPongIdx, problemContext.expertIdx);
        }
    }
}

// ==================================================================================
// RunGmm2CombinePostSingleCore：等待当前 row-group 的 GMM2 N tile 完成，再执行 CombineSend
// ==================================================================================
template <int32_t CombineQuantMode>
__aicore__ inline void RunGmm2CombinePostSingleCore(const Params& params,
    const Gmm2ProblemContext& problemContext, const Gmm2CombineContext& combineContext)
{
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        if (!combineContext.enablePostCombine) {
            return;
        }
        MegaMoeCombineImpl::CombineTileSchedule tileSchedule{
            combineContext.tileSchedule.tileId, combineContext.tileSchedule.tileNum};
        MegaMoeCombineImpl::CombineExpertContext expertContext{
            problemContext.expertIdx, combineContext.rankId, combineContext.k,
            problemContext.expertBeforeCnt, combineContext.mExpert};
        MegaMoeCombineImpl::CombineBufferContext bufferContext{
            problemContext.addrInfo.gmm2OutGlobal, combineContext.gmm2SyncCounterAddr,
            combineContext.combineUbTensorSize};
        MegaMoeCombineImpl::ProcessCombineSingleCore<CombineQuantMode>(
            tileSchedule, expertContext, bufferContext, params);
    }
}

// ==================================================================================
// ProcessGmm2SingleCore：GMM2 单核总入口，串联矩阵计算和量化 Combine 后处理
// ==================================================================================
template <int32_t CombineQuantMode, typename QuantOutType, typename SwigluQuantOutType, typename WeightType,
    typename QuantScaleType, bool EnableA8W4, bool EnableA4W4>
__aicore__ inline void ProcessGmm2SingleCore(const Params& params, const Gmm2ProblemContext& problemContext,
    const Gmm2CombineContext& combineContext, Gmm2RuntimeState& runtimeState)
{
    RunGmm2MatmulSingleCore<CombineQuantMode, QuantOutType, SwigluQuantOutType, WeightType, QuantScaleType,
        EnableA8W4, EnableA4W4>(params, problemContext, runtimeState);
    RunGmm2CombinePostSingleCore<CombineQuantMode>(params, problemContext, combineContext);
}

// ==================================================================================
// FinalizeGmm2SingleCore：非量化 Combine 路径等待 GMM2 ping-pong 同步收尾
// ==================================================================================
template <int32_t CombineQuantMode>
__aicore__ inline void FinalizeGmm2SingleCore(Gmm2RuntimeState& runtimeState)
{
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        EndGMM2Sync(runtimeState.vecSetSyncCom, runtimeState.pingPongIdx);
    }
}

} // namespace MegaMoeGmm2Impl

#endif // MEGA_MOE_GMM2_H
