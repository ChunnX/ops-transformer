/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_softmax_lse_kernel_base.h
 * \brief Arch35 kernel base class
 */

#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_KERNEL_BASE_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_KERNEL_BASE_H

#include "dense_lightning_indexer_softmax_lse_tiling_regbase.h"
#include "dense_lightning_indexer_softmax_lse_regbase_common.h"
#include "dense_lightning_indexer_softmax_lse_cube_block.h"
#include "dense_lightning_indexer_softmax_lse_vector_block.h"

namespace DenseLISoftmaxLseKernelRegbase {

TEMPLATES_DEF
class DenseLISoftmaxLseKernelBase {
public:
    ARGS_TRAITS;
    __aicore__ inline DenseLISoftmaxLseKernelBase(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR weights,
                                GM_ADDR actualSeqLengthsQ, GM_ADDR actualSeqLengths,
                                GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                GM_ADDR workspace, GM_ADDR user,
                                const optiling::DenseLightningIndexerSoftmaxLseRegBaseTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(DenseLISoftmaxLseRunInfo &runInfo);
    __aicore__ inline void ProcessSoftmax(DenseLISoftmaxLseRunInfo &runInfo);
    __aicore__ inline void InitMMResBuf();
    __aicore__ inline void FreeBuf();
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    __aicore__ inline void SetConstInfo();
    __aicore__ inline void SetRunInfo(DenseLISoftmaxLseRunInfo &runInfo, int64_t taskId, int64_t bIdx, 
                                      int64_t gS1Idx, int64_t s2Idx);
    __aicore__ inline void CalcMultiCoreOffset(int64_t &bStartIdx, int64_t &gS1StartIdx, 
                                                int64_t &bEndIdx, int64_t &gS1EndIdx);
    __aicore__ inline int64_t FindBIndex(int64_t bIndex, int64_t curIndex, int64_t &accumulateLen);
    __aicore__ inline int32_t GetActualSeqLens(int32_t bIdx, int32_t defaultLens,
                                                GlobalTensor<int64_t> &actualSeqLensGm, int64_t &accumLen);
    __aicore__ inline int64_t GetEndGS1Etx(int32_t bIdx, int32_t defaultLens,
                                           GlobalTensor<int64_t> &actualSeqLensGm);
    __aicore__ inline void CalcS2LoopParams(DenseLISoftmaxLseRunInfo &runInfo);
    __aicore__ inline void CalcGS1LoopParams(DenseLISoftmaxLseRunInfo &runInfo, int64_t bN2LoopIdx);

    TPipe *pipe = nullptr;
    const optiling::DenseLightningIndexerSoftmaxLseRegBaseTilingData *__restrict tilingData = nullptr;

    GlobalTensor<int64_t> actualSeqLengthsQueryGm;
    GlobalTensor<int64_t> actualSeqLengthsKeyGm;

    GlobalTensor<T> mm1ResGm;
    GlobalTensor<T> vec1ResGm;

    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;

    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;

    DenseLISoftmaxLseConstInfo constInfo;
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;

    static constexpr uint32_t M_BASE_SIZE = 512;
    static constexpr uint32_t S2_BASE_SIZE = 512;
    static constexpr uint32_t D_SIZE = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t S1_BASE_SIZE = 8;
    static constexpr uint32_t SYNC_MODE = 2;
    static constexpr uint32_t C0_SIZE = 16;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::Init(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights,
    GM_ADDR actualSeqLengthsQ, GM_ADDR actualSeqLengths,
    GM_ADDR softmaxMax, GM_ADDR softmaxSum,
    GM_ADDR workspace, GM_ADDR user,
    const optiling::DenseLightningIndexerSoftmaxLseRegBaseTilingData *__restrict tiling,
    TPipe *tPipe)
{
    pipe = tPipe;
    tilingData = tiling;
    SetConstInfo();

    if (actualSeqLengthsQ != nullptr) {
        actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsQ, constInfo.bSize);
    } else {
        actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsQ, 0);
    }
    if (actualSeqLengths != nullptr) {
        actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengths, constInfo.bSize);
    } else {
        actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengths, 0);
    }

    InitMMResBuf();
    InitWorkspace(workspace);

    if ASCEND_IS_AIV {
        vecBlock.InitParams(constInfo, tilingData);
        vecBlock.InitGlobalBuffer(weights, softmaxMax, softmaxSum, mm1ResGm, vec1ResGm);
        vecBlock.InitBuffers(pipe);
    } else if ASCEND_IS_AIC {
        cubeBlock.SetCubeBlockParams(tPipe, &l1BufferManager);
        cubeBlock.InitCubeBuffers();
        cubeBlock.InitGlobalBuffer(query, key, mm1ResGm);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::SetConstInfo()
{
    if ASCEND_IS_AIV {
        constInfo.aivIdx = GetBlockIdx();
        constInfo.aicIdx = constInfo.aivIdx / 2;
        constInfo.subBlockIdx = constInfo.aivIdx % MODE_NUM_2;
    } else {
        constInfo.aicIdx = GetBlockIdx();
    }

    auto &baseInfo = tilingData->baseParams;
    constInfo.bSize = baseInfo.bSize;
    constInfo.n2Size = baseInfo.n2Size;
    constInfo.gSize = baseInfo.gSize;
    constInfo.s1Size = baseInfo.s1Size;
    constInfo.s2Size = baseInfo.s2Size;
    constInfo.dSize = baseInfo.dSize;
    constInfo.kSize = baseInfo.kSize;
    constInfo.s1BaseSize = baseInfo.s1BaseSize;
    constInfo.s2BaseSize = baseInfo.s2BaseSize;
    constInfo.mBaseSize = baseInfo.mBaseSize;
    constInfo.sparseMode = baseInfo.sparseMode;
    constInfo.preTokens = baseInfo.preTokens;
    constInfo.nextTokens = baseInfo.nextTokens;
    constInfo.scaleValue = baseInfo.scaleValue;
    constInfo.layoutType = static_cast<DenseLISoftmaxLseLayout>(baseInfo.layoutType);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::InitMMResBuf()
{
    l1BufferManager.Init(pipe, L1_MAX_SIZE);
    ubBufferManager.Init(pipe, UB_MAX_SIZE);
    bmm1Buffers.Init(ubBufferManager, 32 * 256 * sizeof(T));

    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SYNC_MM1_TO_V1_FLAG[0]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SYNC_MM1_TO_V1_FLAG[1]);
    } else if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_GATHER_TO_MM1_FLAG[0]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_GATHER_TO_MM1_FLAG[1]);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::InitWorkspace(__gm__ uint8_t *workspace)
{
    int64_t mm1ResOffset = constInfo.mBaseSize * constInfo.s2BaseSize * sizeof(T);
    int64_t vec1ResOffset = constInfo.s1BaseSize * DenseLISoftmaxLseAlign<uint64_t>(
        constInfo.s2Size > 128 * 1024 ? 128 * 1024 : constInfo.s2Size, 
        static_cast<uint64_t>(constInfo.s2BaseSize)) * sizeof(float);

    int64_t coreTotalOffset = constInfo.aicIdx * (mm1ResOffset + vec1ResOffset);

    uint64_t offset = 0;
    mm1ResGm.SetGlobalBuffer((__gm__ T *)(workspace + offset + coreTotalOffset));
    offset += mm1ResOffset;
    vec1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + offset + coreTotalOffset));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::FreeBuf()
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_GATHER_TO_MM1_FLAG[0]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_GATHER_TO_MM1_FLAG[1]);
    } else if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_MM1_TO_V1_FLAG[0]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_MM1_TO_V1_FLAG[1]);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::CalcMultiCoreOffset(
    int64_t &bStartIdx, int64_t &gS1StartIdx, int64_t &bEndIdx, int64_t &gS1EndIdx)
{
    int64_t actualSum = 0;
    int64_t bS1Index = tilingData->multiCoreParams.bS1Index[constInfo.aicIdx];
    int64_t bS1EndIndex = constInfo.aicIdx + 1 < optiling::MAX_CORE_NUM_REGBASE ?
                          tilingData->multiCoreParams.bS1Index[constInfo.aicIdx + 1] :
                          tilingData->multiCoreParams.totalSize;

    if constexpr (LAYOUT_T == DenseLISoftmaxLseLayout::TND) {
        bStartIdx = FindBIndex(0, bS1Index, actualSum);
        gS1StartIdx = bS1Index - actualSum;
        bEndIdx = FindBIndex(bStartIdx, bS1EndIndex - 1, actualSum);
        gS1EndIdx = bS1EndIndex - actualSum;
    } else {
        bStartIdx = bS1Index / constInfo.s1Size;
        bEndIdx = (bS1EndIndex - 1) / constInfo.s1Size;
        gS1StartIdx = bS1Index - bStartIdx * constInfo.s1Size;
        gS1EndIdx = bS1EndIndex - bEndIdx * constInfo.s1Size;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::FindBIndex(
    int64_t bIndex, int64_t curIndex, int64_t &accumulateLen)
{
    for (int index = bIndex; index < constInfo.bSize; index++) {
        int64_t actualLen = actualSeqLengthsQueryGm.GetValue(index);
        if (curIndex < actualLen) {
            return index;
        }
        accumulateLen = actualLen;
    }
    return tilingData->multiCoreParams.totalSize >= curIndex ? constInfo.bSize : -1;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::GetActualSeqLens(
    int32_t bIdx, int32_t defaultLens, GlobalTensor<int64_t> &actualSeqLensGm, int64_t &accumLen)
{
    if (actualSeqLensGm.GetSize() <= 0) {
        return defaultLens;
    }

    if (bIdx == 0) {
        accumLen = 0;
        return actualSeqLensGm.GetValue(0);
    } else {
        accumLen = actualSeqLensGm.GetValue(bIdx - 1);
        return (actualSeqLensGm.GetValue(bIdx) - accumLen);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::GetEndGS1Etx(
    int32_t bIdx, int32_t defaultLens, GlobalTensor<int64_t> &actualSeqLensGm)
{
    if (actualSeqLensGm.GetSize() <= 0) {
        return defaultLens;
    }

    if (bIdx == 0) {
        return actualSeqLensGm.GetValue(0);
    } else {
        return (actualSeqLensGm.GetValue(bIdx) - actualSeqLensGm.GetValue(bIdx - 1));
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::SetRunInfo(
    DenseLISoftmaxLseRunInfo &runInfo, int64_t taskId, int64_t bIdx, 
    int64_t gS1Idx, int64_t s2Idx)
{
    runInfo.taskId = taskId;
    runInfo.bIdx = bIdx;
    runInfo.gS1Idx = gS1Idx;
    runInfo.s2Idx = s2Idx;

    int64_t accumS1Len = 0;
    int64_t accumS2Len = 0;
    if constexpr (LAYOUT_T == DenseLISoftmaxLseLayout::TND) {
        runInfo.actS1Size = GetActualSeqLens(runInfo.bIdx, constInfo.s1Size, actualSeqLengthsQueryGm, accumS1Len);
        runInfo.actS2Size = GetActualSeqLens(runInfo.bIdx, constInfo.s2Size, actualSeqLengthsKeyGm, accumS2Len);
        runInfo.accumS1Idx = accumS1Len + gS1Idx;
        runInfo.accumS2Idx = accumS2Len;
    } else {
        runInfo.actS1Size = constInfo.s1Size;
        runInfo.actS2Size = constInfo.s2Size;
        runInfo.accumS1Idx = bIdx * constInfo.s1Size + gS1Idx;
        runInfo.accumS2Idx = bIdx * constInfo.s2Size;
    }

    runInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = runInfo.actS1Size * constInfo.gSize - gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        runInfo.actMBaseSize = remainedGS1Size;
    }

    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = DenseLISoftmaxLseCeilDiv<uint32_t>(runInfo.actS2Size, constInfo.s2BaseSize);
    if (s2Idx == s2SplitNum - 1) {
        uint32_t tailSize = runInfo.actS2Size % constInfo.s2BaseSize;
        runInfo.actualSingleProcessSInnerSize = tailSize == 0 ? constInfo.s2BaseSize : tailSize;
    }

    runInfo.actualSingleProcessSInnerSizeAlign = DenseLISoftmaxLseAlign<uint32_t>(
        runInfo.actualSingleProcessSInnerSize, static_cast<uint32_t>(32));

    runInfo.isFirstS2InnerLoop = s2Idx == 0;
    runInfo.s2LoopTimes = DenseLISoftmaxLseCeilDiv<uint32_t>(runInfo.actS2Size, constInfo.s2BaseSize);

    if constexpr (LAYOUT_T == DenseLISoftmaxLseLayout::TND) {
        runInfo.queryTensorOffset = runInfo.accumS1Idx * constInfo.gSize * constInfo.dSize;
        runInfo.keyTensorOffset = runInfo.accumS2Idx * K_HEAD_NUM * constInfo.dSize;
        runInfo.weightsTensorOffset = runInfo.accumS1Idx * constInfo.gSize;
        runInfo.reduceMaxOutGmOffset = runInfo.accumS1Idx;
    } else {
        runInfo.queryTensorOffset = bIdx * constInfo.s1Size * constInfo.gSize * constInfo.dSize + 
                                    gS1Idx * constInfo.mBaseSize * constInfo.dSize;
        runInfo.keyTensorOffset = bIdx * constInfo.s2Size * K_HEAD_NUM * constInfo.dSize;
        runInfo.weightsTensorOffset = bIdx * constInfo.s1Size * constInfo.gSize + 
                                       gS1Idx * constInfo.s1BaseSize * constInfo.gSize;
        runInfo.reduceMaxOutGmOffset = bIdx * constInfo.s1Size + gS1Idx * constInfo.s1BaseSize;
    }

    runInfo.reduceSumGmOutOffset = s2Idx * constInfo.s2BaseSize;
    runInfo.isLastS2InnerLoop = s2Idx == runInfo.s2LoopTimes - 1;
    runInfo.loop = taskId;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::Process()
{
    ProcessMain();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::ProcessMain()
{
    int64_t bStartIdx, gS1StartIdx, bEndIdx, gS1EndIdx;
    CalcMultiCoreOffset(bStartIdx, gS1StartIdx, bEndIdx, gS1EndIdx);

    DenseLISoftmaxLseRunInfo runInfo;
    int64_t taskId = 0;

    for (int64_t bIdx = bStartIdx; bIdx <= bEndIdx; bIdx++) {
        bool lastB = (bIdx == bEndIdx);
        int64_t gS1StartIdxThisBatch = (bIdx == bStartIdx) ? gS1StartIdx : 0;
        int64_t gS1EndIdxThisBatch = 0;

        if constexpr (LAYOUT_T == DenseLISoftmaxLseLayout::TND) {
            gS1EndIdxThisBatch = (!lastB) ? GetEndGS1Etx(bIdx, constInfo.s1Size, actualSeqLengthsQueryGm) : gS1EndIdx;
        } else {
            gS1EndIdxThisBatch = (!lastB) ? constInfo.s1Size : gS1EndIdx;
        }

        int64_t gS1SplitNum = DenseLISoftmaxLseCeilDiv<int64_t>(gS1EndIdxThisBatch, constInfo.s1BaseSize);

        for (int64_t gS1LoopIdx = gS1StartIdxThisBatch / constInfo.s1BaseSize; 
             gS1LoopIdx < gS1SplitNum; gS1LoopIdx++) {

            for (int64_t s2Idx = 0; s2Idx < DenseLISoftmaxLseCeilDiv<int64_t>(
                 GetActualSeqLens(bIdx, constInfo.s2Size, actualSeqLengthsKeyGm, runInfo.accumS2Idx), 
                 constInfo.s2BaseSize); s2Idx++) {
                
                SetRunInfo(runInfo, taskId, bIdx, gS1LoopIdx * constInfo.s1BaseSize, s2Idx);
                ProcessBaseBlock(runInfo);
                taskId++;
            }

            if ASCEND_IS_AIV {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                ProcessSoftmax(runInfo);
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_MM1_TO_V1_FLAG[0]);
            }
        }
    }

    FreeBuf();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::ProcessBaseBlock(
    DenseLISoftmaxLseRunInfo &runInfo)
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_MM1_TO_V1_FLAG[runInfo.loop % MODE_NUM_2]);
        cubeBlock.ComputeMm1(this->bmm1Buffers.Get(), runInfo, constInfo);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_MM1_TO_V1_FLAG[runInfo.loop % MODE_NUM_2]);
    } else {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_MM1_TO_V1_FLAG[runInfo.loop % MODE_NUM_2]);
        vecBlock.ProcessVec1(this->bmm1Buffers.GetPre(), runInfo);
        if (!runInfo.isLastS2InnerLoop) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_MM1_TO_V1_FLAG[runInfo.loop % MODE_NUM_2]);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseKernelBase<TEMPLATE_ARGS>::ProcessSoftmax(
    DenseLISoftmaxLseRunInfo &runInfo)
{
    vecBlock.ProcessSoftmax(runInfo);
}

} // namespace DenseLISoftmaxLseKernelRegbase

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_KERNEL_BASE_H