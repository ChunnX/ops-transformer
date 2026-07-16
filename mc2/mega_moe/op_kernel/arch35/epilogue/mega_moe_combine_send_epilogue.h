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
 * \file mega_moe_combine_send_epilogue.h
 * \brief CombineSend epilogue that connects GMM2 completion, AIV compute, and communication.
 */

#ifndef MEGA_MOE_COMBINE_SEND_EPILOGUE_H
#define MEGA_MOE_COMBINE_SEND_EPILOGUE_H

#include "kernel_operator.h"
#include "../mega_moe_base.h"
#if __has_include("../../../common/mc2_kernel_utils.h")
#include "../../../common/mc2_kernel_utils.h"
#else
#include "../../../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "../mega_moe_impl_base.h"
#include "../aiv_comm/mega_moe_combine_send_comm.h"
#include "../aiv_compute/mega_moe_combine_send_compute.h"
using namespace AscendC;

namespace MegaMoeCombineImpl {
// =============================================
// CombineTokenGroup：处理一个 token group 的 Combine 操作，从 GMM2 输出读取数据，量化后发送到目标 rank
// =============================================
template <uint8_t QuantMode, typename T>
__aicore__ inline void CombineTokenGroup(uint32_t tokenStart, uint32_t tokenCount, uint32_t n, uint32_t groupIdx,
                                         uint32_t rankId, GM_ADDR gmm2OutAddr, const Params &params,
                                         LocalTensor<int32_t> &tripleTensor, int64_t ubTensorSize, int64_t offset,
                                         uint32_t quantTokenSizeBytes)
{
    LocalTensor<T> combineUbTensor(TPosition::VECIN, offset, ubTensorSize);
    offset += ubTensorSize * sizeof(T);

    uint32_t nScale = Ops::Base::CeilDiv(n, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t nAlign32 = Ops::Base::CeilAlign(n, static_cast<uint32_t>(ALIGN_32));
    uint32_t floatTempSize = nScale + nScale / 2;
    LocalTensor<float> floatTemp = LocalTensor<float>(TPosition::VECIN, offset, floatTempSize);

    GlobalTensor<T> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gmm2OutAddr));

    using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;

    uint32_t singleTokenElems = (nAlign32 * sizeof(T) + quantTokenSizeBytes) / sizeof(T);
    DataCopyPadExtParams<T> copyPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyExtParams gm2UbParams{static_cast<uint16_t>(1), static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};

    for (uint32_t i = 0; i < tokenCount; i++) {
        uint32_t pingPong = i % 2;
        LocalTensor<T> ubBf16 = combineUbTensor[pingPong * singleTokenElems];
        LocalTensor<T> ubQuantData = ubBf16[nAlign32];

        // MTE2: read from GM
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID3>();
        AscendC::DataCopyPad(ubBf16, gmm2OutGm[(tokenStart + i) * n], gm2UbParams, copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID4>();

        // V: quantize
        QuantMxFp8<QuantMode, T>(ubQuantData, ubBf16, floatTemp, n);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID5>();

        // MTE3: send to GM
        LocalTensor<Fp8Type> ubQuantDataFp8 = ubQuantData.template ReinterpretCast<Fp8Type>();
        CombineQuantizedTokens<Fp8Type>(i, 1, n, nScale, groupIdx, rankId, tripleTensor, ubQuantDataFp8, params);
    }

    // Wait for all MTE3 operations to complete
    SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
}

__aicore__ inline void ComputeCombineTileGrouping(uint32_t tileId, uint32_t numGroups, uint32_t totalTiles,
    uint32_t& myGroup, uint32_t& myIdxInGrp, uint32_t& myGrpSize)
{
    // 将 totalTiles 尽量均匀分到 numGroups，前 remainder 个 group 多分配一个 tile。
    uint32_t baseSize = totalTiles / numGroups;
    uint32_t remainder = totalTiles % numGroups;
    uint32_t boundary = remainder * (baseSize + 1);

    if (tileId < boundary) {
        myGroup = tileId / (baseSize + 1);
        myIdxInGrp = tileId % (baseSize + 1);
        myGrpSize = baseSize + 1;
    } else {
        uint32_t adjusted = tileId - boundary;
        myGroup = remainder + adjusted / baseSize;
        myIdxInGrp = adjusted % baseSize;
        myGrpSize = baseSize;
    }
}

// ProcessCombineSingleCore 将输入拆为三类独立上下文：
// 1) tile 调度：连续逻辑 tile ID 和 tile 数；
// 2) expert 上下文：定位 triple 与计算分组所需的 expert/rank/shape；
// 3) buffer 上下文：Combine 使用的 GM/UB 地址。
// 逻辑 tile ID 由外部计算，物理核对应的同步计数器地址也由外部映射后传入。
struct CombineTileSchedule {
    uint32_t tileId;
    uint32_t tileNum;
};

struct CombineExpertContext {
    uint32_t expertIdx;
    uint32_t rankId;
    uint32_t k;
    uint32_t expertBeforeCnt;
    uint32_t mExpert;
};

struct CombineBufferContext {
    GM_ADDR gmm2OutAddr;
    GM_ADDR gmm2SyncCounterAddr;
    int64_t combineUbTensorSize;
};

// 量化 CombineSend 开始前清零 GMM2 完成计数器所需的信息。
struct CombineSyncCounterResetContext {
    CombineTileSchedule tileSchedule;
    uint32_t expertPerRank;
    GM_ADDR gmm2SyncCounterAddr;
    uint32_t resetBatchElementCount;
};

// ===============================================================
// ComputeCombineAlignedTileRange：按逻辑 tile 分割计数器区间，并保持每段起点对齐
// ===============================================================
__aicore__ inline void ComputeCombineAlignedTileRange(const CombineTileSchedule& tileSchedule,
    int32_t totalLength, int32_t align, int32_t& tileLength, int32_t& tileOffset)
{
    tileLength = 0;
    tileOffset = 0;
    if (totalLength <= 0 || align <= 0 || tileSchedule.tileNum == 0 ||
        tileSchedule.tileId >= tileSchedule.tileNum) {
        return;
    }
    uint32_t lengthPerTile = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLength), tileSchedule.tileNum);
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
// ResetCombineSyncCountersSingleCore：每个逻辑 tile 清零自己负责的 GMM2 完成计数器
// ==================================================================================
__aicore__ inline void ResetCombineSyncCountersSingleCore(const CombineSyncCounterResetContext& resetContext,
    LocalTensor<int32_t>& resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (resetContext.resetBatchElementCount == 0) {
        return;
    }
    int32_t totalCounters = static_cast<int32_t>(static_cast<int64_t>(resetContext.expertPerRank) *
        resetContext.tileSchedule.tileNum * INT_CACHELINE);
    int32_t tileLength = 0;
    int32_t tileOffset = 0;
    ComputeCombineAlignedTileRange(resetContext.tileSchedule, totalCounters, ALIGN_32, tileLength, tileOffset);
    if (tileLength == 0) {
        return;
    }
    GlobalTensor<int32_t> gmm2CombineSyncCounter;
    gmm2CombineSyncCounter.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(resetContext.gmm2SyncCounterAddr));
    Duplicate(resetTensor, 0, resetContext.resetBatchElementCount);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    for (int32_t resetElementOffset = 0; resetElementOffset < tileLength;
         resetElementOffset += resetContext.resetBatchElementCount) {
        int32_t currentBatchElementCount =
            tileLength - resetElementOffset < static_cast<int32_t>(resetContext.resetBatchElementCount) ?
                tileLength - resetElementOffset : static_cast<int32_t>(resetContext.resetBatchElementCount);
        DataCopy(gmm2CombineSyncCounter[tileOffset + resetElementOffset], resetTensor, currentBatchElementCount);
    }
}

// ==================================================================================
// ComputeCombineUbTensorSize：计算量化 CombineSend 在 UB 中需要的元素数
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

// =============================================
// ProcessCombineSingleCore：generic combine-quant 路径的单核后处理
//                            等待本 expert 的 row-group 完成，再读取 triple 和 GMM2 输出
// =============================================
template <int32_t CombineQuantMode>
__aicore__ inline void ProcessCombineSingleCore(const CombineTileSchedule& tileSchedule,
    const CombineExpertContext& expertContext, const CombineBufferContext& bufferContext, const Params& params)
{
    uint32_t tokenGroupsThisExpert = Ops::Base::CeilDiv(expertContext.mExpert, MegaMoeImpl::L1_TILE_M_256);
    if (tokenGroupsThisExpert == 0 || tileSchedule.tileNum == 0) {
        return;
    }

    // 按 token row-group 对逻辑 tile 分组，组内 tile 分担该 row-group 的 token。
    uint32_t myGroup = 0;
    uint32_t myIdxInGrp = 0;
    uint32_t myGrpSize = 0;
    ComputeCombineTileGrouping(tileSchedule.tileId, tokenGroupsThisExpert, tileSchedule.tileNum, myGroup,
                               myIdxInGrp, myGrpSize);
    if (myGroup >= tokenGroupsThisExpert || myGrpSize == 0) {
        return;
    }

    // 同步计数器地址已由调用方按 expert 和物理核映射完成。
    // 计数达到 nTilesPerGroup 表示 GMM2 已产出当前 row-group 的全部 N tile。
    uint32_t nTilesPerGroup = Ops::Base::CeilDiv(expertContext.k, MegaMoeImpl::L1_TILE_N);
    __gm__ int32_t *myCounterAddr = reinterpret_cast<__gm__ int32_t *>(bufferContext.gmm2SyncCounterAddr);
    while (AscendC::ReadGmByPassDCache(myCounterAddr) != nTilesPerGroup) {
        int64_t st = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - st < 100) {
        };
    }

    // 量化 token 由 k 字节数据和 nScale 字节 scale 组成，每条记录按 32B 对齐。
    uint32_t nScale = Ops::Base::CeilDiv(expertContext.k, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(
        expertContext.k + nScale, static_cast<uint32_t>(ALIGN_32));
    uint32_t tokenStart = myGroup * MegaMoeImpl::L1_TILE_M_256;
    uint32_t tokenCount = (MegaMoeImpl::L1_TILE_M_256 < expertContext.mExpert - tokenStart) ?
        MegaMoeImpl::L1_TILE_M_256 : expertContext.mExpert - tokenStart;
    // 组内再均分 token，尾 tile 可能少于 tokensPerTile 或无任务。
    uint32_t tokensPerTile = Ops::Base::CeilDiv(tokenCount, myGrpSize);
    int32_t myTokenOffset = myIdxInGrp * tokensPerTile;
    int32_t myTokenCount = 0;
    if (myTokenOffset < static_cast<int32_t>(tokenCount)) {
        myTokenCount = (tokensPerTile < tokenCount - myTokenOffset) ?
            tokensPerTile : tokenCount - myTokenOffset;
    }
    if (myTokenCount <= 0) {
        return;
    }

    // 从 workspace 搬入本 tile 对应的 (rank, tokenIndex, topkIndex) 三元组。
    AscendC::SetCtrlSpr<60, 60>(0);
    int64_t offset = 0;
    LocalTensor<int32_t> tripleTensor = LocalTensor<int32_t>(TPosition::VECIN, offset, myTokenCount * TRIPLE_SIZE);
    offset += myTokenCount * TRIPLE_SIZE * sizeof(int32_t);
    AscendC::GlobalTensor<int32_t> tripleGm;
    tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.tripleInfoPtr
        + (expertContext.expertBeforeCnt + tokenStart + myTokenOffset) * TRIPLE_SIZE * sizeof(int32_t)));
    AscendC::DataCopy(tripleTensor, tripleGm, myTokenCount * TRIPLE_SIZE);
    PipeBarrier<PIPE_MTE2>();
    // 读取 GMM2 结果，完成量化并按三元组发送到目标 rank。
    CombineTokenGroup<CombineQuantMode, bfloat16_t>(
        tokenStart + myTokenOffset, myTokenCount, expertContext.k, expertContext.expertIdx, expertContext.rankId,
        bufferContext.gmm2OutAddr, params, tripleTensor, bufferContext.combineUbTensorSize,
        offset, quantTokenSizeBytes);
}

}  // namespace MegaMoeCombineImpl

#endif // MEGA_MOE_COMBINE_SEND_EPILOGUE_H
