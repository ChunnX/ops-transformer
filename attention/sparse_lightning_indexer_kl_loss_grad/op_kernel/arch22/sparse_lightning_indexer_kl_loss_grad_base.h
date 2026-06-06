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
 * \file sparse_lightning_indexer_kl_loss_grad_base.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_BASE_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_BASE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_lightning_indexer_kl_loss_grad_common.h"
#include "../sparse_lightning_indexer_kl_loss_grad_metadata.h"
#include "sparse_lightning_indexer_kl_loss_grad_tiling.h"
#include "sparse_lightning_indexer_kl_loss_grad_vector.h"
#include "sparse_lightning_indexer_kl_loss_grad_vector2.h"
#include "sparse_lightning_indexer_kl_loss_grad_service_cube.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SLIT> class SparseLightningIndexerKLLossGradBase {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename SLIT::inputQT;
    using KV_T = typename SLIT::inputKT;
    using OUT_T = typename SLIT::outputT;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = KV_T;
    using MM12_OUT_T = T;
    using MM3_OUT_T = T;

    static constexpr bool hasRope = SLIT::hasRope;
    static constexpr bool deterministic = SLIT::deterministic;
    static constexpr uint32_t topKSize = static_cast<uint32_t>(SLIT::topKRange);
    static constexpr SLILayout LAYOUT_T = SLIT::inputQLayout;
    static constexpr SLILayout KV_LAYOUT_T = SLIT::inputKLayout;

    __aicore__ inline SparseLightningIndexerKLLossGradBase(){};
    __aicore__ inline void Init(__gm__ uint8_t *q, __gm__ uint8_t *k,
                                __gm__ uint8_t *w, __gm__ uint8_t *sparseIndices,
                                __gm__ uint8_t *attnSoftmaxL1Norm,
                                __gm__ uint8_t *cuSeqlensQ,
                                __gm__ uint8_t *cuSeqlensK,
                                __gm__ uint8_t *sequsedQ,
                                __gm__ uint8_t *sequsedK,
                                __gm__ uint8_t *cmpResidualK,
                                __gm__ uint8_t *metadata,
                                __gm__ uint8_t *dq, __gm__ uint8_t *dk,
                                __gm__ uint8_t *dw, __gm__ uint8_t *softmaxOut,
                                __gm__ uint8_t *workspace,
                                const optiling::SparseLightningIndexerKLLossGradTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void InitConstInfo();
    __aicore__ inline void InitBuffer(TPipe *pipe);
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    __aicore__ inline void Process();
    __aicore__ inline void GetRunInfo(int64_t taskId, int64_t bIdx, int64_t s1Idx, int64_t s1IdxEnd, int64_t accumS1Len, int64_t accumS2Len,
        int32_t actualSeqLensQ, int32_t actualSeqLensK, SLIKLLossGradRunInfo &runInfo);

private:
    __aicore__ inline int32_t GetActualSeqLens(int32_t bIdx, int32_t defaultLens,
        GlobalTensor<int32_t> &actualSeqLensGm, SLILayout layout, int64_t &accumLen);
    __aicore__ inline int32_t GetS2SparseLen(int32_t s1Idx, int32_t actualSeqLensQ, int32_t actualSeqLensK, SLISparseMode sparseMode);
    __aicore__ inline int64_t FindBIndex(int64_t bIndex, int64_t curIndex, int64_t &accumulateLen);
    __aicore__ inline int64_t GetEndS1(int64_t bIdx);
    __aicore__ inline int64_t GetMetadataTotalSize();
    __aicore__ inline int64_t GetMetadataBS1Index(uint32_t coreIdx);
    __aicore__ inline int64_t GetEndS1Etx(int32_t bIdx, int32_t defaultLens,
        GlobalTensor<int32_t> &actualSeqLensGm, SLILayout layout);
    __aicore__ inline void CalcMultiCoreOffset(int64_t &bStartIdx, int64_t &s1StartIdx, int64_t &bEndIdx, int64_t &s1EndIdx);
    __aicore__ inline int64_t CalcBS1Loop();

    TPipe *pipe = nullptr;
    const optiling::SparseLightningIndexerKLLossGradTilingData *__restrict tilingData = nullptr;
    bool hasMetadata = false;
    SLIKLLossGradConstInfo constInfo;
    SLIKLLossGradRunInfo runInfos[3];

    // vector and cube class
    SLIKLLossVectorService<SLIT> vectorService;
    SLIKLLossVector2Service<SLIT> vector2Service;
    SLITMatmulService<SLIT> matmulService;

    // input GM
    GlobalTensor<Q_T> queryGm, queryIndexGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<KV_T> keyGm, keyIndexGm;
    GlobalTensor<Q_ROPE_T> queryRopeGm;
    GlobalTensor<K_ROPE_T> keyRopeGm;
    GlobalTensor<T> attnSoftmaxL1NormGm;
    GlobalTensor<int32_t> topKIndexGm;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm, actualSeqLengthsKeyGm;
    GlobalTensor<int32_t> metadataGm;
    // output GM
    GlobalTensor<OUT_T> dQueryIndexGm, dKeyIndexGm;
    GlobalTensor<T> dWeightGm, softmaxOutGm;
    // workspace
    GlobalTensor<KV_T> gatherPRes, gatherSYRes;
    GlobalTensor<MM12_OUT_T> bmm1Res, bmm2Res;
    GlobalTensor<T> psySyncGm;
    GlobalTensor<KV_T> reluGradRes;
    GlobalTensor<T> scatterAddRes;
    GlobalTensor<MM3_OUT_T> bmm3Res;
    GlobalTensor<T> reluGm;
    // local tensor
    TBuf<> gatherTbuf;
    TBuf<> mm1Tbuf;
    TBuf<> mm2TBuf;         // 复用 -> mm4 scatterAdd reluGrad
};

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::Init(
    __gm__ uint8_t *q, __gm__ uint8_t *k,
    __gm__ uint8_t *w, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *attnSoftmaxL1Norm,
    __gm__ uint8_t *cuSeqlensQ,
    __gm__ uint8_t *cuSeqlensK,
    __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedK,
    __gm__ uint8_t *cmpResidualK,
    __gm__ uint8_t *metadata,
    __gm__ uint8_t *dq, __gm__ uint8_t *dk,
    __gm__ uint8_t *dw, __gm__ uint8_t *softmaxOut,
    __gm__ uint8_t *workspace,
    const optiling::SparseLightningIndexerKLLossGradTilingData *__restrict tiling,
    TPipe *tPipe)
{
    // init tiling data
    pipe = tPipe;
    tilingData = tiling;

    InitConstInfo();

    // init input global buffer
    (void)sequsedQ;
    (void)sequsedK;
    (void)cmpResidualK;
    queryGm.SetGlobalBuffer((__gm__ Q_T *)q);
    keyGm.SetGlobalBuffer((__gm__ KV_T *)k);
    queryIndexGm.SetGlobalBuffer((__gm__ Q_T *)q);
    keyIndexGm.SetGlobalBuffer((__gm__ KV_T *)k);
    weightGm.SetGlobalBuffer((__gm__ T *)w);
    topKIndexGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    attnSoftmaxL1NormGm.SetGlobalBuffer((__gm__ T *)attnSoftmaxL1Norm);
    if constexpr (SLIT::hasRope) {
        queryRopeGm.SetGlobalBuffer((__gm__ Q_ROPE_T *)q);
        keyRopeGm.SetGlobalBuffer((__gm__ K_ROPE_T *)k);
    }
    if (cuSeqlensQ != nullptr) {
        actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ, constInfo.bSize + 1);
    } else {
        actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ, 0);
    }
    if (cuSeqlensK != nullptr) {
        actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensK, constInfo.bSize + 1);
    } else {
        actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensK, 0);
    }
    hasMetadata = metadata != nullptr;
    if (hasMetadata) {
        metadataGm.SetGlobalBuffer((__gm__ int32_t *)metadata, optiling::SLI_METADATA_SIZE);
    }

    // init output global buffer
    dQueryIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dq);
    dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dk);
    dWeightGm.SetGlobalBuffer((__gm__ T *)dw);
    softmaxOutGm.SetGlobalBuffer((__gm__ T *)softmaxOut);
    InitWorkspace(workspace);
    InitBuffer(pipe);

    if ASCEND_IS_AIV {
        // InitVecOP
        vectorService.InitParams(constInfo, tilingData, metadataGm, hasMetadata);
        vectorService.InitVector0GM(keyGm, keyRopeGm, keyIndexGm, topKIndexGm,
                                    actualSeqLengthsQueryGm, actualSeqLengthsKeyGm,
                                    gatherPRes, gatherSYRes);
        vectorService.InitVector1GM(attnSoftmaxL1NormGm, bmm2Res, weightGm, psySyncGm,
                                    softmaxOutGm, dWeightGm, reluGm, reluGradRes,
                                    actualSeqLengthsQueryGm, actualSeqLengthsKeyGm);
        vectorService.InitVector2GM(bmm3Res, topKIndexGm, scatterAddRes);
    } else if ASCEND_IS_AIC {
        // initCubeOP
        matmulService.InitParams(constInfo);

        matmulService.InitMm1GlobalTensor(queryGm, gatherPRes, queryRopeGm, actualSeqLengthsQueryGm, actualSeqLengthsKeyGm, bmm1Res, dk);
        matmulService.InitMm2GlobalTensor(queryIndexGm, gatherSYRes, bmm2Res);
        matmulService.InitMm5GlobalTensor(reluGradRes, queryIndexGm, bmm3Res, topKIndexGm);
        matmulService.InitMm6GlobalTensor(reluGradRes, gatherSYRes, dQueryIndexGm);
    }
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::InitConstInfo()
{
    if ASCEND_IS_AIV {
        constInfo.aivIdx = GetBlockIdx(); // vec:0-47
        constInfo.aicIdx = constInfo.aivIdx / 2;
        constInfo.subBlockIdx = constInfo.aivIdx % 2;
    } else {
        constInfo.aicIdx = GetBlockIdx(); // cube:0-23
    }

    auto &baseInfo = tilingData->baseParams;
    constInfo.bSize = baseInfo.bSize;
    constInfo.n2Size = baseInfo.n2Size;
    constInfo.gSizeQuery = baseInfo.gSizeQuery;
    constInfo.gSizeQueryIndex = baseInfo.gSizeQueryIndex;
    constInfo.s1Size = baseInfo.s1Size;
    constInfo.s2Size = baseInfo.s2Size;
    constInfo.kSize = baseInfo.kSize;

    constInfo.dSizeQuery = baseInfo.dSizeQuery;
    constInfo.dSizeQueryIndex = baseInfo.dSizeQueryIndex;
    constInfo.gSizeQueryIndexAlign16 = ((constInfo.gSizeQueryIndex + 15) / 16) * 16;
    constInfo.sparseMode = static_cast<SLISparseMode>(baseInfo.sparseMode);
    constInfo.scaleValue = baseInfo.scaleValue;
    constInfo.cmpRatio = baseInfo.cmpRatio;
    constInfo.hasSoftmaxInput = baseInfo.hasSoftmaxInput;
    constInfo.gatherKeySize = topKSize * (constInfo.dSizeQuery + constInfo.dSizeRope);
    constInfo.gatherKeyIndexSize = topKSize * constInfo.dSizeQueryIndex;
    if constexpr (!SLIT::hasRope) {
        constInfo.dSizeQueryRope = 0;
        constInfo.gatherKeySize = topKSize * (constInfo.dSizeQuery);
    }
    constInfo.tilingInfo = tilingData->vectorParams.softmaxYTilingData;
    constInfo.simpleSoftMaxTilingInfo = tilingData->vectorParams.simpleSoftmaxPTilingData;
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::InitWorkspace(__gm__ uint8_t *workspace)
{
    int64_t pOffset = topKSize * (constInfo.dSizeQuery + constInfo.dSizeQueryRope) * sizeof(KV_T); // * 2;
    int64_t syOffset = topKSize * constInfo.dSizeQueryIndex * sizeof(KV_T); // * 2;
    int64_t bmm1Offset = constInfo.gSizeQuery * topKSize * sizeof(float); // * 2;
    int64_t psySyncSize = topKSize * sizeof(float) * 2;
    int64_t bmm2Offset = constInfo.gSizeQueryIndex * topKSize * sizeof(float); // * 2;
    int64_t reluGradOffset = constInfo.gSizeQueryIndex * topKSize * sizeof(float); // * 2;
    int64_t bmm3Offset =  topKSize * constInfo.dSizeQueryIndex * sizeof(float); // * 2;

    int64_t coreTotalOffset = constInfo.aicIdx *
            (pOffset * constInfo.gatherKeyDbNum + syOffset * constInfo.gatherKeyIndexDbNum +
            bmm1Offset * 2 + bmm2Offset * 2 + reluGradOffset * 2 + psySyncSize * 2);

    int64_t totalOffset = GetBlockNum() *
            (pOffset * constInfo.gatherKeyDbNum + syOffset * constInfo.gatherKeyIndexDbNum +
            bmm1Offset * 2 + bmm2Offset * 2 + reluGradOffset * 2 + psySyncSize * 2);

    uint64_t offset = 0;
    // workspace 按核分, 每个核内不同workspace相邻
    gatherPRes.SetGlobalBuffer(
        (__gm__ KV_T *)(workspace + offset + coreTotalOffset));
    offset += pOffset * constInfo.gatherKeyDbNum;

    gatherSYRes.SetGlobalBuffer(
        (__gm__ KV_T *)(workspace + offset + coreTotalOffset));
    offset += syOffset * constInfo.gatherKeyIndexDbNum;

    bmm1Res.SetGlobalBuffer(
        (__gm__ MM12_OUT_T *)(workspace + offset + coreTotalOffset));
    offset += bmm1Offset * 2;

    psySyncGm.SetGlobalBuffer(
        (__gm__ T *)(workspace + offset + coreTotalOffset));
    offset += psySyncSize * 2;

    bmm2Res.SetGlobalBuffer(
        (__gm__ MM12_OUT_T *)(workspace + offset + coreTotalOffset));
    reluGm.SetGlobalBuffer(
        (__gm__ T *)(bmm2Res.GetPhyAddr()));
    offset += bmm2Offset * 2;

    reluGradRes.SetGlobalBuffer(
        (__gm__ OUT_T *)(workspace + offset + coreTotalOffset));
    offset += reluGradOffset * 2;

    bmm3Res.SetGlobalBuffer(
        (__gm__ MM3_OUT_T *)(workspace + totalOffset));
    totalOffset += bmm3Offset * GetBlockNum() * 2;

    scatterAddRes.SetGlobalBuffer(
        (__gm__ T *)(workspace + totalOffset));
    if ASCEND_IS_AIV {
        int64_t totalCost = 0;
        int64_t totalCostQ = 0;

        if constexpr (KV_LAYOUT_T == SLILayout::TND) {
            totalCostQ = actualSeqLengthsQueryGm.GetValue(constInfo.bSize);
            totalCost = actualSeqLengthsKeyGm.GetValue(constInfo.bSize);
        } else {
            totalCostQ = constInfo.bSize * constInfo.s1Size;
            totalCost = constInfo.bSize * constInfo.s2Size;
        }

        int64_t totalCoreNum = GetBlockNum() * GetTaskRation();
        int64_t avgCost = CeilDiv(totalCost, totalCoreNum);
        int64_t avgCostQ = CeilDiv(totalCostQ, totalCoreNum);
        int32_t t2Start = Min(constInfo.aivIdx * avgCost, totalCost);
        int32_t t2End = Min(t2Start + avgCost, totalCost);
        int32_t t2StartQ = Min(constInfo.aivIdx * avgCostQ, totalCostQ);
        int32_t t2EndQ = Min(t2StartQ + avgCostQ, totalCostQ);
        int64_t qBaseOffset = constInfo.gSizeQuery * constInfo.dSizeQuery;

        // AscendC::InitOutput(dQueryIndexGm[t2StartQ * qBaseOffset], qBaseOffset * (t2EndQ - t2StartQ), static_cast<T>(0));
        // AscendC::InitOutput(dWeightGm[t2Start * constInfo.gSizeQuery], constInfo.gSizeQuery * (t2End - t2Start), static_cast<T>(0));
        AscendC::InitOutput(scatterAddRes[t2Start * constInfo.dSizeQueryIndex], constInfo.dSizeQueryIndex * (t2End - t2Start), static_cast<T>(0));
    }
    SyncAll();
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::InitBuffer(TPipe *pipe)
{
    if ASCEND_IS_AIC {
        matmulService.InitBuffers(pipe);
    } else {
        vectorService.InitBuffers(pipe);
    }
}
template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::FindBIndex(int64_t bIndex, int64_t curIndex, int64_t &accumulateLen)
{
    for (int index = bIndex; index < constInfo.bSize; index++) {
        int64_t actualLen = this->actualSeqLengthsQueryGm.GetValue(index + 1);
        if (curIndex < actualLen) {
            return index;
        }
        accumulateLen = actualLen;
    }
    return GetMetadataTotalSize() >= curIndex ? constInfo.bSize : -1;
}

template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::GetEndS1(int64_t bIdx)
{
    int64_t end = constInfo.aicIdx + 1 < optiling::MAX_CORE_NUM ?
                      GetMetadataBS1Index(bIdx + 1) : GetMetadataTotalSize();
    return end - GetMetadataBS1Index(bIdx);
}

template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::GetMetadataTotalSize()
{
    return hasMetadata ? metadataGm.GetValue(optiling::SLI_META_TOTAL_SIZE_INDEX) : tilingData->multiCoreParams.totalSize;
}

template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::GetMetadataBS1Index(uint32_t coreIdx)
{
    if (hasMetadata) {
        return metadataGm.GetValue(optiling::GetSliMetaBS1IndexAttr(coreIdx));
    }
    return tilingData->multiCoreParams.bS1Index[coreIdx];
}

template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::GetEndS1Etx(int32_t bIdx,
    int32_t defaultLens, GlobalTensor<int32_t> &actualSeqLensGm, SLILayout layout)
{
    if (actualSeqLensGm.GetSize() <= 0) {
        return defaultLens;
    }

    if (layout == SLILayout::TND) {
        return actualSeqLensGm.GetValue(bIdx + 1) - actualSeqLensGm.GetValue(bIdx);
    } else {
        assert(false, "do not support current layout!\n");
        return 0;
    }
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::CalcMultiCoreOffset(int64_t &bStartIdx, int64_t &s1StartIdx, int64_t &bEndIdx, int64_t &s1EndIdx)
{
    int64_t actualSum = 0;
    int64_t bS1Index = GetMetadataBS1Index(constInfo.aicIdx);
    int64_t bS1EndIndex = constInfo.aicIdx + 1 < optiling::MAX_CORE_NUM ?
                GetMetadataBS1Index(constInfo.aicIdx + 1) : GetMetadataTotalSize();
    if constexpr (LAYOUT_T == SLILayout::TND) {
        bStartIdx = FindBIndex(0, bS1Index, actualSum);
        s1StartIdx = bS1Index - actualSum;
        bEndIdx = FindBIndex(bStartIdx, bS1EndIndex - 1, actualSum);
        s1EndIdx = bS1EndIndex - actualSum;
    } else {
        bStartIdx = bS1Index / constInfo.s1Size;
        bEndIdx = (bS1EndIndex-1) / constInfo.s1Size;
        s1StartIdx = bS1Index - bStartIdx * constInfo.s1Size;
        s1EndIdx = bS1EndIndex - bEndIdx * constInfo.s1Size;
    }
}

template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::CalcBS1Loop() {
    int64_t maxLoop = 0;
    int32_t coreNum = GetBlockNum();
    int64_t bS1Index, bS1EndIndex;
    for (int32_t aicIdx = 0; aicIdx < coreNum; aicIdx++) {
        bS1Index = GetMetadataBS1Index(aicIdx);
        bS1EndIndex = aicIdx + 1 < optiling::MAX_CORE_NUM ?
                GetMetadataBS1Index(aicIdx + 1) : GetMetadataTotalSize();
        maxLoop = Max(maxLoop, bS1EndIndex - bS1Index);
    }
    return maxLoop;
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::Process()
{
    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
    } else {
        matmulService.AllocEventID();
    }

    int64_t bStartIdx, s1StartIdx, bEndIdx, s1EndIdx;
    CalcMultiCoreOffset(bStartIdx, s1StartIdx, bEndIdx, s1EndIdx);

    int64_t taskId = 0;
    int64_t extraLoopTimes = 0;
    for (int64_t bIdx = bStartIdx; bIdx <= bEndIdx; bIdx++) {
        bool lastB = (bIdx == bEndIdx);
        int64_t s1StartIdxThisBatch = 0;
        int64_t s1EndIdxThisBatch = 0;

        int64_t accumS1Len = 0;
        int64_t accumS2Len = 0;
        int32_t actualSeqLensQ = 0;
        int32_t actualSeqLensK = 0;
        if constexpr (LAYOUT_T == SLILayout::TND) {
            s1StartIdxThisBatch = (bIdx == bStartIdx) ? s1StartIdx : 0;
            s1EndIdxThisBatch = (!lastB) ? GetEndS1Etx(bIdx, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_T) : s1EndIdx;
            actualSeqLensQ = GetActualSeqLens(bIdx, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_T, accumS1Len);
            actualSeqLensK = GetActualSeqLens(bIdx, constInfo.s2Size, actualSeqLengthsKeyGm, KV_LAYOUT_T, accumS2Len);
        } else if constexpr (LAYOUT_T == SLILayout::BSND) {
            s1StartIdxThisBatch = (bIdx == bStartIdx) ? s1StartIdx : 0;
            s1EndIdxThisBatch = (!lastB) ? constInfo.s1Size : s1EndIdx;
        }
        if (lastB) {
            extraLoopTimes = 2;// 最后一个Batch需要额外循环两次，因为preload方式会产生尾巴
        }

        if (constInfo.cmpRatio == 4) {
            int32_t realKvLen = 0;
            if constexpr (LAYOUT_T == SLILayout::TND) {
                realKvLen = GetS2SparseLen(s1StartIdxThisBatch, actualSeqLensQ, actualSeqLensK, constInfo.sparseMode);
            } else {
                realKvLen = GetS2SparseLen(s1StartIdxThisBatch, constInfo.s1Size, constInfo.s2Size, constInfo.sparseMode);
            }

            if (realKvLen <= 0 && s1StartIdxThisBatch < constInfo.cmpRatio - 1) {
                // init invalid s1 loop output[s1StartIdxThisBatch, cmpRatio - 1]
                int64_t s1Offset = 0;
                if constexpr (LAYOUT_T == SLILayout::TND) {
                    s1Offset = accumS1Len;
                } else if constexpr (LAYOUT_T == SLILayout::BSND) {
                    s1Offset = bIdx * constInfo.s1Size;
                }
                int64_t qBaseOffset = constInfo.gSizeQuery * constInfo.dSizeQuery;
                int64_t s1CleanLen = Min(constInfo.cmpRatio - 1, s1EndIdxThisBatch) - s1StartIdxThisBatch;
                AscendC::InitOutput(dQueryIndexGm[(s1Offset + s1StartIdxThisBatch) * qBaseOffset], qBaseOffset * s1CleanLen, static_cast<OUT_T>(0));
                AscendC::InitOutput(dWeightGm[(s1Offset + s1StartIdxThisBatch) * constInfo.gSizeQuery], constInfo.gSizeQuery * s1CleanLen, static_cast<T>(0));
                AscendC::InitOutput(softmaxOutGm[(s1Offset + s1StartIdxThisBatch) * constInfo.n2Size * topKSize],
                                    constInfo.n2Size * topKSize * s1CleanLen, static_cast<T>(0));

                // skip invalid s1 loop
                s1StartIdxThisBatch = Min(s1EndIdxThisBatch, constInfo.cmpRatio - 1);
            }
        }

        for (int64_t s1Idx = s1StartIdxThisBatch; s1Idx < s1EndIdxThisBatch + extraLoopTimes; s1Idx++) {
            SLIKLLossGradRunInfo &runInfoNeg2 = runInfos[(taskId + 1) % 3];       // 上2轮
            SLIKLLossGradRunInfo &runInfoNeg1 = runInfos[(taskId + 2) % 3];       // 上1轮
            SLIKLLossGradRunInfo &runInfo0 = runInfos[taskId % 3];                // 当前轮

            GetRunInfo(taskId, bIdx, s1Idx, s1EndIdxThisBatch, accumS1Len, accumS2Len, actualSeqLensQ, actualSeqLensK, runInfo0);
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag<2, PIPE_MTE3>(14);
            } else {
                CrossCoreSetFlag<2, PIPE_MTE2>(14);
            }

            if (runInfo0.isValid) {
                if ASCEND_IS_AIV {
                    vectorService.ProcessVector0(runInfo0);  // V0
                }
            }

            if (runInfoNeg1.isValid) {
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C1_TO_V1_P_FLAG[runInfoNeg1.taskIdMod2]);
                    matmulService.ComputeMm2(runInfoNeg1);   // C1
                }

                if ASCEND_IS_AIV {
                    vectorService.ProcessVector1(runInfoNeg1); // V1
                }
                if ASCEND_IS_AIC {
                    matmulService.ComputeMm5(runInfoNeg1); // C2
                    matmulService.ComputeMm6(runInfoNeg1); // C2
                }
            }

            if (runInfoNeg2.isValid) {
                if ASCEND_IS_AIV {
                    if constexpr (deterministic){
                        vectorService.ProcessDeterVector2(runInfoNeg2);
                    } else {
                        vectorService.ProcessVector2(runInfoNeg2); // V2 ScatterAdd
                    }
                    runInfoNeg2.isValid = false;
                }
            }

            taskId++;
        }
    }

    if constexpr (deterministic){
        int64_t maxLoop = CalcBS1Loop();
        if ASCEND_IS_AIV {
            // 关于SYNC_V2_TO_C2_DETER_SA_FLAG的同步，是为了防止SYNC_C2_TO_V2_SA_FLAG累加到上限
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V2_TO_C2_DETER_SA_FLAG);
        }
        for (; taskId < maxLoop + extraLoopTimes; taskId++) {
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_V2_TO_C2_DETER_SA_FLAG);
                CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C2_TO_V2_SA_FLAG[(taskId - extraLoopTimes) & 1]);
            }
            if ASCEND_IS_AIV {
                SLIKLLossGradRunInfo &runInfo = runInfos[0];
                runInfo.taskId = taskId - extraLoopTimes;
                runInfo.taskIdMod2 = runInfo.taskId & 1;
                vectorService.ProcessDeterVector2(runInfo);
                CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V2_TO_C2_DETER_SA_FLAG);
            }
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_V2_TO_C2_DETER_SA_FLAG);
        }
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
    }
    if ASCEND_IS_AIV {
        vector2Service.InitParams(constInfo, tilingData);
        vector2Service.InitVector2GM(scatterAddRes, topKIndexGm, dKeyIndexGm, actualSeqLengthsQueryGm, actualSeqLengthsKeyGm);
        vector2Service.InitBuffers(pipe);
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        vector2Service.AllocEventID();
        vector2Service.ProcessVector2();
        vector2Service.FreeEventID();
    }
}

template <typename SLIT>
__aicore__ inline int32_t SparseLightningIndexerKLLossGradBase<SLIT>::GetActualSeqLens(int32_t bIdx,
    int32_t defaultLens, GlobalTensor<int32_t> &actualSeqLensGm, SLILayout layout, int64_t &accumLen)
{
    if (actualSeqLensGm.GetSize() <= 0) {
        return defaultLens;
    }

    if (layout == SLILayout::TND) {
        accumLen = actualSeqLensGm.GetValue(bIdx);
        return actualSeqLensGm.GetValue(bIdx + 1) - accumLen;
    } else {
        return 0;
    }
}

template <typename SLIT>
__aicore__ inline int32_t SparseLightningIndexerKLLossGradBase<SLIT>::GetS2SparseLen(int32_t s1Idx,
    int32_t actualSeqLensQ, int32_t actualSeqLensK, SLISparseMode sparseMode)
{
    if (sparseMode == SLISparseMode::RightDown) {
        if (constInfo.cmpRatio != 0) {
            return (actualSeqLensK * constInfo.cmpRatio - actualSeqLensQ + s1Idx + 1) / constInfo.cmpRatio;
        } else {
            return Max(actualSeqLensK - actualSeqLensQ + s1Idx + 1, 0);
        }
    } else if (sparseMode == SLISparseMode::NoMask) {
        return actualSeqLensK;
    } else {
        return 0;
    }
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::GetRunInfo(int64_t taskId, int64_t bIdx, int64_t s1Idx,
    int64_t s1IdxEnd, int64_t accumS1Len, int64_t accumS2Len, int32_t actualSeqLensQ, int32_t actualSeqLensK, SLIKLLossGradRunInfo &runInfo)
{
    if (s1Idx >= s1IdxEnd) {        // extra循环阶段，不生产任务
        runInfo.isValid = false;
        return;
    }

    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;

    runInfo.bIdx = bIdx;
    runInfo.s1Idx = s1Idx;
    if constexpr (LAYOUT_T == SLILayout::TND) {
        runInfo.actS1Size = actualSeqLensQ;
        runInfo.actS2Size = actualSeqLensK;
        runInfo.accumS1Idx = accumS1Len + s1Idx;
        runInfo.accumS2Idx = accumS2Len;
    } else if constexpr (LAYOUT_T == SLILayout::BSND) {
        runInfo.actS1Size = constInfo.s1Size;
        runInfo.actS2Size = constInfo.s2Size;
        runInfo.accumS1Idx = bIdx * constInfo.s1Size + s1Idx;
        runInfo.accumS2Idx = bIdx * constInfo.s2Size;
    }

    runInfo.s2SparseLen = GetS2SparseLen(runInfo.s1Idx, runInfo.actS1Size, runInfo.actS2Size, constInfo.sparseMode);
    runInfo.s2RealSize = Min(topKSize, runInfo.s2SparseLen);
    if (constInfo.cmpRatio != 0){
        runInfo.s2RealSize = Max(1, runInfo.s2RealSize);
    }

    runInfo.kRealSize = runInfo.s2RealSize;
    runInfo.kRealSizeAlign8 = (runInfo.kRealSize + 7) >> 3 << 3;
    runInfo.s2LoopTimes = CeilDiv(runInfo.s2RealSize, constInfo.s2BaseSize);
    runInfo.s2TailSize = (runInfo.s2RealSize % constInfo.s2BaseSize == 0) ?
        constInfo.s2BaseSize : (runInfo.s2RealSize % constInfo.s2BaseSize);

    runInfo.kLoopTimes = CeilDiv(runInfo.kRealSize, runInfo.kBaseSize);
    runInfo.kTailSize = (runInfo.kRealSize % runInfo.kBaseSize == 0) ?
        runInfo.kBaseSize : (runInfo.kRealSize % runInfo.kBaseSize);

    if constexpr (LAYOUT_T == SLILayout::TND) {
        runInfo.queryTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQuery * (constInfo.dSizeQuery);
        runInfo.queryRopeTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQuery * (constInfo.dSizeQueryRope);
        runInfo.queryIndexTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQueryIndex * constInfo.dSizeQueryIndex;
    } else if constexpr (LAYOUT_T == SLILayout::BSND) {
        runInfo.queryTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQuery * (constInfo.dSizeQuery);
        runInfo.queryRopeTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQuery * (constInfo.dSizeQueryRope);
        runInfo.queryIndexTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQueryIndex * constInfo.dSizeQueryIndex;
    }

    if constexpr (LAYOUT_T == SLILayout::TND) {
        runInfo.topkGmBaseOffset = runInfo.accumS1Idx * topKSize;
    } else {
        runInfo.topkGmBaseOffset = runInfo.bIdx * constInfo.s1Size * topKSize + runInfo.s1Idx * topKSize;
    }

    runInfo.calcP = ((runInfo.taskIdMod2 == 0 && constInfo.subBlockIdx == 0) ||
        (runInfo.taskIdMod2 != 0 && constInfo.subBlockIdx != 0));

    runInfo.isValid = true;
}

#endif // SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_BASE_H
