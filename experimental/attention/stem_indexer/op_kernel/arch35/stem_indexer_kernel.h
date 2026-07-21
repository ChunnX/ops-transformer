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
 * \file stem_indexer_kernel.h
 * \brief
 */

#ifndef stem_indexer_KERNEL_H
#define stem_indexer_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stem_indexer_common.h"
#include "stem_indexer_service_vector.h"
#include "stem_indexer_service_cube.h"
#include "../stem_indexer_metadata.h"

namespace SIKernel {
using namespace SICommon;
using namespace matmul;
using namespace optiling;
using namespace optiling::detail;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SIT>
class SIPreload {
public:
    __aicore__ inline SIPreload(){};
    __aicore__ inline void Init(__gm__ uint8_t *qflat, __gm__ uint8_t *kflat, __gm__ uint8_t *vbias,
                                __gm__ uint8_t *qSeqLens, __gm__ uint8_t *kvSeqLens,
                                __gm__ uint8_t *numPromptTokens, __gm__ uint8_t *metadata,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen,
                                __gm__ uint8_t *workspace, const StemIndexerTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    using Q_T = typename SIT::queryType;
    using K_T = typename SIT::keyType;
    using OUT_T = typename SIT::outputType;
    static constexpr SI_LAYOUT Q_LAYOUT_T = SIT::layout;
    static constexpr SI_LAYOUT K_LAYOUT_T = SIT::keyLayout;

    using SCORE_T = uint32_t;

    SIMatmul<SIT> matmulService;
    SIVector<SIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 256;
    static constexpr uint32_t S2_BASE_SIZE = 128;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;

    // for workspace double
    static constexpr uint32_t WS_DOUBLE = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t vBiasCoreOffset = 0ULL;
    uint64_t keyScaleCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;
    uint64_t indiceLenCoreOffset = 0ULL;
    bool isUsedCoreEqZero = false;
    // ================================Global Buffer区=================================

    GlobalTensor<uint32_t> metadataGm;
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<float> vbiasGm;
    GlobalTensor<int32_t> qSeqLensGm;
    GlobalTensor<int32_t> kvSeqLensGm;
    GlobalTensor<int32_t> numPromptTokensGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> sparseSeqLenGm;

    // ================================类成员变量====================================
    // AIC/AIV information.
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    SICommon::ConstInfo constInfo{};
    SICommon::TempLoopInfo tempLoopInfo{};
    SICommon::SplitCoreInfo splitCoreInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const StemIndexerTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK);
    // ================================Split Core================================
    __aicore__ inline void SplitCoreByAICPU(uint32_t cubeCoreIdx, uint32_t vecCoreIdx,
                                            GlobalTensor<uint32_t> &metadataGm);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t gS1Idx, uint32_t actS1Size, uint32_t actS2Size);
    __aicore__ inline uint32_t CalcS2ValidSize(uint32_t gS1Idx, uint32_t actS1Size, uint32_t actS2Size);
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                            SICommon::RunInfo runInfo);
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<int32_t> &seqLensGm, uint32_t defaultSeqLen);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, SICommon::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::InitTilingData(const StemIndexerTilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = tilingData->qHeadNum;
    constInfo.kvHeadNum = tilingData->kvHeadNum;
    constInfo.gSize = tilingData->gSize;
    constInfo.maxQb = tilingData->maxQb;
    constInfo.maxKb = tilingData->maxKb;
    constInfo.qSeqSize = tilingData->maxQb;
    constInfo.kSeqSize = tilingData->maxKb;
    constInfo.headDim = tilingData->headDim;
    constInfo.sparseCount = tilingData->maxKb;
    constInfo.causal = tilingData->causal;
    constInfo.stemBlockSize = tilingData->stemBlockSize;
    constInfo.stemStride = tilingData->stemStride;
    constInfo.initialBlocks = tilingData->initialBlocks;
    constInfo.windowSize = tilingData->windowSize;
    constInfo.rSquare = tilingData->rSquare;
    constInfo.kBlockNumRateMedium = tilingData->kBlockNumRateMedium;
    constInfo.kBlockNumBiasMedium = tilingData->kBlockNumBiasMedium;
    constInfo.kBlockNumRateLarge = tilingData->kBlockNumRateLarge;
    constInfo.kBlockNumBiasLarge = tilingData->kBlockNumBiasLarge;
    constInfo.mBaseSize = tilingData->mBaseSize;
    constInfo.s2BaseSize = tilingData->s2BaseSize;
    constInfo.s1BaseSize = CeilDiv(tilingData->mBaseSize, tilingData->gSize);
    constInfo.alpha = tilingData->alpha;
    constInfo.outputLayout = Q_LAYOUT_T;  // 输出和输入形状一致
    constInfo.attenMaskFlag = tilingData->causal != 0U;
    if (Q_LAYOUT_T == SI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == SI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::InitActualSeqLen(__gm__ uint8_t *qSeqLens,
                                                          __gm__ uint8_t *kvSeqLens)
{
    if (qSeqLens == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        qSeqLensGm.SetGlobalBuffer((__gm__ int32_t *)qSeqLens, constInfo.actualLenQDims);
    }
    if (kvSeqLens == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        kvSeqLensGm.SetGlobalBuffer((__gm__ int32_t *)kvSeqLens, constInfo.actualLenDims);
    }
}

template <typename SIT>
__aicore__ inline uint32_t SIPreload<SIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                             GlobalTensor<int32_t> &seqLensGm,
                                                             uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) { // TND
        return static_cast<uint32_t>(seqLensGm.GetValue(bIdx) - seqLensGm.GetValue(bIdx - 1));
    } else {
        return static_cast<uint32_t>(seqLensGm.GetValue(bIdx));
    }
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size)
{
    uint32_t qTokenLen = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, qSeqLensGm,
                                         constInfo.qSeqSize * constInfo.stemBlockSize);
    uint32_t kTokenLen = GetActualSeqLen(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, kvSeqLensGm,
                                         constInfo.kSeqSize * constInfo.stemBlockSize);
    actS1Size = CeilDiv(qTokenLen, constInfo.stemBlockSize);
    actS2Size = CeilDiv(kTokenLen, constInfo.stemBlockSize);
}

template <typename SIT>
__aicore__ inline uint32_t SIPreload<SIT>::GetS2BaseBlockNumOnMask(uint32_t gS1Idx, uint32_t actS1Size,
                                                                    uint32_t actS2Size)
{
    uint32_t s2ValidSize = CalcS2ValidSize(gS1Idx, actS1Size, actS2Size);
    if (s2ValidSize == 0U) {
        return 0;
    }
    return CeilDiv(s2ValidSize, constInfo.s2BaseSize);
}

template <typename SIT>
__aicore__ inline uint32_t SIPreload<SIT>::CalcS2ValidSize(uint32_t gS1Idx, uint32_t actS1Size,
                                                            uint32_t actS2Size)
{
    uint32_t validS2SizeWithWindow = 0U;
    if (actS2Size == 0U) {
        return 0U;
    }
    if (!constInfo.attenMaskFlag) {
        validS2SizeWithWindow = actS2Size;
    } else {
        if (actS1Size == 0U || constInfo.gSize == 0U) {
            return 0U;
        }

        uint64_t totalMSize = static_cast<uint64_t>(actS1Size) * constInfo.gSize;
        uint64_t mBlockStart = static_cast<uint64_t>(gS1Idx) * constInfo.mBaseSize;
        if (mBlockStart >= totalMSize) {
            return 0U;
        }
        uint64_t mBlockEnd = Min(mBlockStart + constInfo.mBaseSize, totalMSize);
        uint64_t mBlockLen = mBlockEnd - mBlockStart;
        uint32_t firstS1Idx = static_cast<uint32_t>(mBlockStart % actS1Size);
        uint32_t lastS1Idx = static_cast<uint32_t>((mBlockEnd - 1U) % actS1Size);
        uint32_t maxS1Idx = (mBlockLen >= actS1Size || lastS1Idx < firstS1Idx) ? (actS1Size - 1U) : lastS1Idx;
        uint32_t qBlockEnd = maxS1Idx + 1U;

        // For causal mode, KV may contain prefix/cache blocks before the current Q range.
        // Shift qBlockEnd by the KV-Q block offset to get the visible K block count.
        int64_t validS2Size = static_cast<int64_t>(actS2Size) - static_cast<int64_t>(actS1Size) +
                              static_cast<int64_t>(qBlockEnd);
        validS2SizeWithWindow = static_cast<uint32_t>(Max(validS2Size, static_cast<int64_t>(0)));
    }
    return (validS2SizeWithWindow > constInfo.windowSize) ? (validS2SizeWithWindow - constInfo.windowSize) : 0U;
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::SplitCoreByAICPU(uint32_t cubeCoreIdx, uint32_t vecCoreIdx,
                                                        GlobalTensor<uint32_t> &metadataGm)
{
    uint32_t liCoreEnableIndex = GetAttrAbsIndex(cubeCoreIdx, SLI_CORE_ENABLE_INDEX);
    uint32_t bN2StartIndex = GetAttrAbsIndex(cubeCoreIdx, SLI_BN2_START_INDEX);
    uint32_t mStartIndex = GetAttrAbsIndex(cubeCoreIdx, SLI_M_START_INDEX);
    uint32_t s2StartIndex = GetAttrAbsIndex(cubeCoreIdx, SLI_S2_START_INDEX);
    uint32_t bN2EndIndex = GetAttrAbsIndex(cubeCoreIdx, SLI_BN2_END_INDEX);
    uint32_t mEndIndex = GetAttrAbsIndex(cubeCoreIdx, SLI_M_END_INDEX);
    uint32_t s2EndIndex = GetAttrAbsIndex(cubeCoreIdx, SLI_S2_END_INDEX);

    uint32_t liZeroCoreEnableIndex = GetAttrAbsIndex(0, SLI_CORE_ENABLE_INDEX);
    if (metadataGm.GetValue(liZeroCoreEnableIndex) == 0) {
        isUsedCoreEqZero = true;
    }
    if (metadataGm.GetValue(liCoreEnableIndex) == 0) {
        splitCoreInfo.isCoreEnable = false;
        return;
    } else {
        splitCoreInfo.isCoreEnable = true;
    }

    splitCoreInfo.bN2Start = metadataGm.GetValue(bN2StartIndex);
    splitCoreInfo.gS1Start = metadataGm.GetValue(mStartIndex);
    splitCoreInfo.s2Start = metadataGm.GetValue(s2StartIndex);
    splitCoreInfo.bN2End = metadataGm.GetValue(bN2EndIndex);
    splitCoreInfo.gS1End = metadataGm.GetValue(mEndIndex);
    splitCoreInfo.s2End  = metadataGm.GetValue(s2EndIndex);

    if (splitCoreInfo.s2End != 0) {
        // Move s2End back by one block while keeping bN2End and gS1End unchanged.
        splitCoreInfo.s2End = splitCoreInfo.s2End - 1;
    } else {
        if (splitCoreInfo.gS1End != 0) {
            // Move gS1End back by one when s2End is 0, while keeping bN2End unchanged.
            // Use bIdx to get actual S2 for s2End calculation.
            splitCoreInfo.gS1End = splitCoreInfo.gS1End - 1;
            // Get current actual S2.
            uint32_t bIdx = splitCoreInfo.bN2End / constInfo.kvHeadNum;
            uint32_t actS1Size, actS2Size;
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
            // S2 block count.
            uint32_t s2BaseNum = GetS2BaseBlockNumOnMask(splitCoreInfo.gS1End, actS1Size, actS2Size);
            splitCoreInfo.s2End = (s2BaseNum == 0U) ? 0U : s2BaseNum - 1U;
        } else {
            // Move bN2End back by one when both gS1End and s2End are 0.
            // Use bIdx to get actual S1/S2 for gS1End and s2End calculation.
            splitCoreInfo.bN2End = splitCoreInfo.bN2End - 1;

            // Get current actual S1 and S2.
            uint32_t bIdx = splitCoreInfo.bN2End / constInfo.kvHeadNum;
            uint32_t actS1Size, actS2Size;
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);

            // Block count in the M(gS1) dimension.
            uint32_t gS1BaseNum = static_cast<uint32_t>(
                CeilDiv(static_cast<uint64_t>(actS1Size) * constInfo.gSize,
                        static_cast<uint64_t>(constInfo.mBaseSize)));
            splitCoreInfo.gS1End = (gS1BaseNum == 0U) ? 0U : gS1BaseNum - 1U;

            // S2 block count.
            uint32_t s2BaseNum = GetS2BaseBlockNumOnMask(splitCoreInfo.gS1End, actS1Size, actS2Size);
            splitCoreInfo.s2End = (s2BaseNum == 0U) ? 0U : s2BaseNum - 1U;
        }
    }
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == SI_LAYOUT::TND) {
            uint32_t tBase = 0;
            if (bIdx > 0) {
                uint32_t tBaseIdx = (constInfo.batchSupperFlag) ? bIdx : bIdx - 1;
                tBase = qSeqLensGm.GetValue(tBaseIdx);
            }
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                for (uint32_t gIdx = 0; gIdx < constInfo.gSize; gIdx++) {
                    uint64_t indiceOutOffset =
                        (((tBase + s1Idx) * constInfo.kvHeadNum + n2Idx) * constInfo.gSize + gIdx) *
                        constInfo.sparseCount;
                    uint64_t indiceLenOffset =
                        ((tBase + s1Idx) * constInfo.kvHeadNum + n2Idx) * constInfo.gSize + gIdx;
                    GlobalTensor<int32_t> indiceOut = sparseIndicesGm[indiceOutOffset];
                    AscendC::InitGlobalMemory(indiceOut, constInfo.sparseCount,
                                              static_cast<int32_t>(SICommon::INVALID_IDX));
                    vectorService.SetSparseSeqLenZero(indiceLenOffset);
                }
            }
        } else if (constInfo.outputLayout == SI_LAYOUT::BNSD) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                for (uint32_t gIdx = 0; gIdx < constInfo.gSize; gIdx++) {
                    uint64_t indiceOutOffset =
                        (((bIdx * constInfo.kvHeadNum + n2Idx) * constInfo.gSize + gIdx) * constInfo.qSeqSize +
                         s1Idx) *
                        constInfo.sparseCount;
                    uint64_t indiceLenOffset =
                        ((bIdx * constInfo.kvHeadNum + n2Idx) * constInfo.gSize + gIdx) * constInfo.qSeqSize + s1Idx;
                    GlobalTensor<int32_t> indiceOut = sparseIndicesGm[indiceOutOffset];
                    AscendC::InitGlobalMemory(indiceOut, constInfo.sparseCount,
                                              static_cast<int32_t>(SICommon::INVALID_IDX));
                    vectorService.SetSparseSeqLenZero(indiceLenOffset);
                }
            }
        }
    }
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::Init(__gm__ uint8_t *qflat, __gm__ uint8_t *kflat, __gm__ uint8_t *vbias,
                                            __gm__ uint8_t *qSeqLens, __gm__ uint8_t *kvSeqLens,
                                            __gm__ uint8_t *numPromptTokens, __gm__ uint8_t *metadata,
                                            __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen,
                                            __gm__ uint8_t *workspace,
                                            const StemIndexerTilingData *__restrict tiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx();  // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx();  // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(qSeqLens, kvSeqLens);
    numPromptTokensGm.SetGlobalBuffer((__gm__ int32_t *)numPromptTokens, constInfo.batchSize);

    // Get split information calculated by AICPU metadata.
    metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);
    SplitCoreByAICPU(aiCoreIdx, tmpBlockIdx, metadataGm);

    pipe = tPipe;

    if ASCEND_IS_AIV {
        vbiasGm.SetGlobalBuffer((__gm__ float *)vbias);
        sparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        sparseSeqLenGm.SetGlobalBuffer((__gm__ int32_t *)sparseSeqLen);
        vectorService.InitParams(constInfo, tiling);
        vectorService.InitVecInputTensor(vbiasGm, numPromptTokensGm, sparseIndicesGm, sparseSeqLenGm);
    } else {
        queryGm.SetGlobalBuffer((__gm__ Q_T *)qflat);
        keyGm.SetGlobalBuffer((__gm__ K_T *)kflat);
        matmulService.InitParams(constInfo);
        matmulService.InitMm1GlobalTensor(queryGm, keyGm);
    }

    InitBuffers();
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kvHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kvHeadNum;
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }
    tempLoopInfo.s2ValidSize = CalcS2ValidSize(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.s2ValidSize % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;

    bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
    uint32_t s2BlockNum = CeilDiv(tempLoopInfo.s2ValidSize, constInfo.s2BaseSize);
    if (s2BlockNum == 0U) {
        tempLoopInfo.s2LoopEnd = 0U;
        return;
    }
    uint32_t tileS2LoopEnd = s2BlockNum - 1U;
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : tileS2LoopEnd;
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    tempLoopInfo.needDealActS1LessThanS1 = false;
    tempLoopInfo.s2ValidSize = 0U;
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    uint64_t promptTokenLen = GetActualSeqLen(tempLoopInfo.bIdx, static_cast<uint32_t>(constInfo.batchSize), false,
                                                  numPromptTokensGm, 0U);
    tempLoopInfo.promptLen = CeilDiv((uint32_t)promptTokenLen, constInfo.stemBlockSize);
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (bN2LoopIdx == splitCoreInfo.bN2End) ? splitCoreInfo.gS1End : gS1SplitNum - 1;
    if constexpr (Q_LAYOUT_T == SI_LAYOUT::BNSD) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, SICommon::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.n2Idx = tempLoopInfo.n2Idx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.promptLen = tempLoopInfo.promptLen;

    runInfo.s2Start = splitCoreInfo.s2Start;
    runInfo.s2LoopEnd = tempLoopInfo.s2LoopEnd;

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.s2ValidSize + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;

    if (runInfo.isFirstS2InnerLoop) {

        queryCoreOffset = runInfo.bIdx * constInfo.qHeadNum * constInfo.qSeqSize * constInfo.headDim +
                          runInfo.n2Idx * constInfo.gSize * constInfo.qSeqSize * constInfo.headDim;

        indiceOutCoreOffset = runInfo.bIdx * constInfo.qHeadNum * constInfo.qSeqSize * constInfo.kSeqSize +
                              runInfo.n2Idx * constInfo.gSize * constInfo.qSeqSize * constInfo.kSeqSize;

        indiceLenCoreOffset = runInfo.bIdx * constInfo.qHeadNum * constInfo.qSeqSize +
                              runInfo.n2Idx * constInfo.gSize * constInfo.qSeqSize;
    }
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
    runInfo.indiceLenOffset = indiceLenCoreOffset;

    runInfo.tensorKeyOffset = runInfo.bIdx * constInfo.kvHeadNum * constInfo.kSeqSize * constInfo.headDim +
                              runInfo.n2Idx * constInfo.kSeqSize * constInfo.headDim +
                              runInfo.s2Idx * constInfo.s2BaseSize * constInfo.headDim;

    runInfo.tensorVBiasOffset = runInfo.bIdx * constInfo.kvHeadNum * constInfo.kSeqSize +
                                runInfo.n2Idx * constInfo.kSeqSize;
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::Process()
{
    if (isUsedCoreEqZero) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }

    ProcessMain();
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2;  // 2 means c:v = 1:2
        uint64_t totalIndiceSize =
            constInfo.batchSize * constInfo.qHeadNum * constInfo.qSeqSize * constInfo.sparseCount;
        uint64_t singleCoreIndiceSize = SICommon::Align((totalIndiceSize + aivCoreNum - 1) / aivCoreNum,
                                                         GM_ALIGN_BYTES / sizeof(int32_t));
        uint64_t indiceBaseSize = static_cast<uint64_t>(tmpBlockIdx) * singleCoreIndiceSize;
        if (indiceBaseSize < totalIndiceSize) {
            uint64_t dealIndiceSize = (indiceBaseSize + singleCoreIndiceSize <= totalIndiceSize)
                                          ? singleCoreIndiceSize
                                          : totalIndiceSize - indiceBaseSize;
            GlobalTensor<int32_t> indiceOut = sparseIndicesGm[indiceBaseSize];
            AscendC::InitGlobalMemory(indiceOut, dealIndiceSize, static_cast<int32_t>(SICommon::INVALID_IDX));
        }

        uint64_t totalLenSize = constInfo.batchSize * constInfo.qHeadNum * constInfo.qSeqSize;
        uint64_t singleCoreLenSize = SICommon::Align((totalLenSize + aivCoreNum - 1) / aivCoreNum,
                                                      GM_ALIGN_BYTES / sizeof(int32_t));
        uint64_t lenBaseSize = static_cast<uint64_t>(tmpBlockIdx) * singleCoreLenSize;
        if (lenBaseSize < totalLenSize) {
            uint64_t dealLenSize =
                (lenBaseSize + singleCoreLenSize <= totalLenSize) ? singleCoreLenSize : totalLenSize - lenBaseSize;
            GlobalTensor<int32_t> indiceLen = sparseSeqLenGm[lenBaseSize];
            AscendC::InitGlobalMemory(indiceLen, dealLenSize, static_cast<int32_t>(0));
        }
    }
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::ProcessMain()
{
    if (!splitCoreInfo.isCoreEnable) {
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_VC_EVENT + 0);
        CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_VC_EVENT + 1);
    } else {
        matmulService.AllocEventID();
    }

    SICommon::RunInfo runInfo;
    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
            continue;
        }

        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            // s2ValidSize excludes windowSize, so initial blocks covering it can use direct output.
            if (tempLoopInfo.s2ValidSize <= constInfo.initialBlocks) {
                vectorService.ProcessDirectOutput(tempLoopInfo, gS1LoopIdx);
                splitCoreInfo.s2Start = 0;
                continue;
            }

            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_VC_EVENT + 0);
        CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_VC_EVENT + 1);
    }
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, SICommon::RunInfo runInfo)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(runInfo);
    } else {
        if (runInfo.isFirstS2InnerLoop) {
            vectorService.initS2LenToNeg(runInfo.gS1Idx, runInfo.actMBaseSize, runInfo.actS1Size,
                                         runInfo.indiceOutOffset);
        }
        vectorService.ProcessVec1(runInfo);
        vectorService.ProcessTopK(runInfo, runInfo.isFirstS2InnerLoop, runInfo.isLastS2InnerLoop);
    }
}

}  // namespace SIKernel
#endif  // stem_indexer_KERNEL_H
