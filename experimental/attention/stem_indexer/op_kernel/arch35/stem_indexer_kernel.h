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

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;   // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;    // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL;  // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    uint32_t actQTokenLen = 0ULL;
    uint32_t actKVTokenLen = 0ULL;
    uint32_t promptTokenLen = 0ULL;
    uint32_t s2ValidSize = 0ULL;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false;  // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;            // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;          // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;         // S2方向循环的尾基本块大小
    bool isDecode = false;
};

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
    // aic、aiv核信息
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
    __aicore__ inline void SplitCoreByAICPU(uint32_t cubeCoreIdx, uint32_t vecCoreIdx, GlobalTensor<uint32_t> &metadataGm);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2Size);
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
    __aicore__ inline uint32_t GetPromptTokenLen(uint32_t bIdx);
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
    constInfo.flattenDim = tilingData->flattenDim;
    constInfo.qSeqSize = tilingData->maxQb;
    constInfo.kSeqSize = tilingData->maxKb;
    constInfo.headDim = tilingData->flattenDim;
    constInfo.sparseCount = tilingData->maxKb;
    constInfo.causal = tilingData->causal;
    constInfo.stemBlockSize = tilingData->stemBlockSize;
    constInfo.stemStride = tilingData->stemStride;
    constInfo.initialBlocks = tilingData->initialBlocks;
    constInfo.windowSize = tilingData->windowSize;
    constInfo.mBaseSize = tilingData->mBaseSize;
    constInfo.nBaseSize = tilingData->nBaseSize;
    constInfo.s2BaseSize = tilingData->nBaseSize;
    constInfo.s1BaseSize = CeilDiv(tilingData->mBaseSize, tilingData->gSize);
    constInfo.rSquare = tilingData->rSquare;
    constInfo.alpha = tilingData->alpha;
    constInfo.kBlockNumRateMedium = tilingData->kBlockNumRateMedium;
    constInfo.kBlockNumBiasMedium = tilingData->kBlockNumBiasMedium;
    constInfo.kBlockNumRateLarge = tilingData->kBlockNumRateLarge;
    constInfo.kBlockNumBiasLarge = tilingData->kBlockNumBiasLarge;
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
    uint32_t promptTokenLen = GetPromptTokenLen(bIdx);
    uint32_t lastKvTokenIdx = (kTokenLen > 0U) ? (kTokenLen - 1U) : 0U;
    tempLoopInfo.actQTokenLen = qTokenLen;
    tempLoopInfo.actKVTokenLen = kTokenLen;
    tempLoopInfo.promptTokenLen = promptTokenLen;
    tempLoopInfo.isDecode = (qTokenLen == 1U) && (promptTokenLen >= lastKvTokenIdx);
    actS1Size = Min(CeilDiv(qTokenLen, constInfo.stemBlockSize), static_cast<uint32_t>(constInfo.qSeqSize));
    actS2Size = Min(CeilDiv(kTokenLen, constInfo.stemBlockSize), static_cast<uint32_t>(constInfo.kSeqSize));
}

template <typename SIT>
__aicore__ inline uint32_t SIPreload<SIT>::GetPromptTokenLen(uint32_t bIdx)
{
    uint32_t promptEnd = static_cast<uint32_t>(numPromptTokensGm.GetValue(bIdx));
    if (bIdx == 0) {
        return promptEnd;
    }
    uint32_t promptStart = static_cast<uint32_t>(numPromptTokensGm.GetValue(bIdx - 1));
    return (promptEnd > promptStart) ? (promptEnd - promptStart) : 0U;
}

template <typename SIT>
__aicore__ inline uint32_t SIPreload<SIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                     uint32_t actS2Size)
{
    uint32_t s2ValidSize = CalcS2ValidSize(s1gIdx, actS1Size, actS2Size);
    if (s2ValidSize == 0U) {
        return 0;
    }
    return CeilDiv(s2ValidSize, constInfo.s2BaseSize);
}

template <typename SIT>
__aicore__ inline uint32_t SIPreload<SIT>::CalcS2ValidSize(uint32_t gS1Idx, uint32_t actS1Size,
                                                            uint32_t actS2Size)
{
    if (actS2Size == 0U) {
        return 0U;
    }
    if (!constInfo.attenMaskFlag || tempLoopInfo.isDecode) {
        return actS2Size;
    }
    uint32_t qBlockStart = gS1Idx * constInfo.s1BaseSize;
    if (qBlockStart >= actS1Size) {
        return 0U;
    }
    uint32_t qBlockEnd = Min(qBlockStart + constInfo.s1BaseSize, actS1Size);
    // For causal mode, KV may contain prefix/cache blocks before the current Q range.
    // Shift qBlockEnd by the KV-Q block offset to get the visible K block count.
    int32_t validS2Size = static_cast<int32_t>(actS2Size) - static_cast<int32_t>(actS1Size) +
                          static_cast<int32_t>(qBlockEnd);
    validS2Size = Min(validS2Size, static_cast<int32_t>(actS2Size));
    validS2Size = Max(validS2Size, static_cast<int32_t>(0));
    return static_cast<uint32_t>(validS2Size);
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::SplitCoreByAICPU(uint32_t cubeCoreIdx, uint32_t vecCoreIdx, GlobalTensor<uint32_t> &metadataGm)
{
    uint32_t liCoreEnableIndex = GetAttrAbsIndex(cubeCoreIdx, LI_CORE_ENABLE_INDEX);
    uint32_t bN2StartIndex = GetAttrAbsIndex(cubeCoreIdx, LI_BN2_START_INDEX);
    uint32_t mStartIndex = GetAttrAbsIndex(cubeCoreIdx, LI_M_START_INDEX);
    uint32_t s2StartIndex = GetAttrAbsIndex(cubeCoreIdx, LI_S2_START_INDEX);
    uint32_t bN2EndIndex = GetAttrAbsIndex(cubeCoreIdx, LI_BN2_END_INDEX);
    uint32_t mEndIndex = GetAttrAbsIndex(cubeCoreIdx, LI_M_END_INDEX);
    uint32_t s2EndIndex = GetAttrAbsIndex(cubeCoreIdx, LI_S2_END_INDEX);

    uint32_t liZeroCoreEnableIndex = GetAttrAbsIndex(0, LI_CORE_ENABLE_INDEX);
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
        // 此时只需要s2End往前退一格，bN2End和gS1End都不变
        splitCoreInfo.s2End = splitCoreInfo.s2End - 1;
    } else {
        if (splitCoreInfo.gS1End != 0) {
            // splitCoreInfo.gS1End != 0 splitCoreInfo.s2End == 0 时，gS1End需要往前退一格, bN2End不变
            // 此时需要使用bIdx获取实际Actal S2来计算出 s2End
            splitCoreInfo.gS1End = splitCoreInfo.gS1End - 1;
            // 需要获取当前的Actaul S2
            uint32_t bIdx = splitCoreInfo.bN2End / constInfo.kHeadNum;
            uint32_t actS1Size, actS2Size;
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
            // s2的切块数量
            uint32_t s2BaseNum = GetS2BaseBlockNumOnMask(splitCoreInfo.gS1End, actS1Size, actS2Size);
            splitCoreInfo.s2End = (s2BaseNum == 0U) ? 0U : s2BaseNum - 1U;
        } else { 
            // splitCoreInfo.gS1End == 0 splitCoreInfo.s2End == 0 时，bN2End需要往前退一格
            // 此时需要使用bIdx获取实际Actal S1和S2来计算出 gS1End 和 s2End
            splitCoreInfo.bN2End = splitCoreInfo.bN2End - 1;

            // 需要获取当前的Actaul S1 S2
            uint32_t bIdx = splitCoreInfo.bN2End / constInfo.kHeadNum;
            uint32_t actS1Size, actS2Size;
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);

            // s1的切块数量
            uint32_t s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
            splitCoreInfo.gS1End = s1GBaseNum - 1;

            // s2的切块数量
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
                        (((tBase + s1Idx) * constInfo.kHeadNum + n2Idx) * constInfo.gSize + gIdx) *
                        constInfo.sparseCount;
                    uint64_t indiceLenOffset =
                        ((tBase + s1Idx) * constInfo.kHeadNum + n2Idx) * constInfo.gSize + gIdx;
                    GlobalTensor<int32_t> indiceOut = sparseIndicesGm[indiceOutOffset];
                    GlobalTensor<int32_t> indiceLen = sparseSeqLenGm[indiceLenOffset];
                    AscendC::InitGlobalMemory(indiceOut, constInfo.sparseCount,
                                              static_cast<int32_t>(constInfo.INVALID_IDX));
                    AscendC::InitGlobalMemory(indiceLen, 1, static_cast<int32_t>(0));
                }
            }
        } else if (constInfo.outputLayout == SI_LAYOUT::BNSD) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                for (uint32_t gIdx = 0; gIdx < constInfo.gSize; gIdx++) {
                    uint64_t indiceOutOffset =
                        (((bIdx * constInfo.kHeadNum + n2Idx) * constInfo.gSize + gIdx) * constInfo.qSeqSize +
                         s1Idx) *
                        constInfo.sparseCount;
                    uint64_t indiceLenOffset =
                        ((bIdx * constInfo.kHeadNum + n2Idx) * constInfo.gSize + gIdx) * constInfo.qSeqSize + s1Idx;
                    GlobalTensor<int32_t> indiceOut = sparseIndicesGm[indiceOutOffset];
                    GlobalTensor<int32_t> indiceLen = sparseSeqLenGm[indiceLenOffset];
                    AscendC::InitGlobalMemory(indiceOut, constInfo.sparseCount,
                                              static_cast<int32_t>(constInfo.INVALID_IDX));
                    AscendC::InitGlobalMemory(indiceLen, 1, static_cast<int32_t>(0));
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
    
    // 获取 AICPU metadata 算好的分核信息。
    metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);
    SplitCoreByAICPU(aiCoreIdx, tmpBlockIdx, metadataGm);

    pipe = tPipe;

    if ASCEND_IS_AIV {
        vbiasGm.SetGlobalBuffer((__gm__ float *)vbias);
        numPromptTokensGm.SetGlobalBuffer((__gm__ int32_t *)numPromptTokens);
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

    // if (GetBlockIdx() == 0) {
    //     AscendC::printf("[StemIndexer Init] bSize=%u qHeadNum=%u kvHeadNum=%u gSize=%u maxQb=%u maxKb=%u "
    //                     "flattenDim=%u usedCoreNum=%u causal=%u stemBlockSize=%u stemStride=%u "
    //                     "initialBlocks=%u windowSize=%u mBaseSize=%u nBaseSize=%u scoreScale=%f alpha=%f "
    //                     "kBlockNumRateMedium=%f kBlockNumBiasMedium=%u kBlockNumRateLarge=%f "
    //                     "kBlockNumBiasLarge=%u scoreWorkspaceBytes=%u\n",
    //                     static_cast<uint32_t>(constInfo.batchSize), static_cast<uint32_t>(constInfo.qHeadNum),
    //                     static_cast<uint32_t>(constInfo.kvHeadNum), static_cast<uint32_t>(constInfo.gSize),
    //                     static_cast<uint32_t>(constInfo.maxQb), static_cast<uint32_t>(constInfo.maxKb),
    //                     static_cast<uint32_t>(constInfo.flattenDim), constInfo.usedCoreNum, constInfo.causal,
    //                     constInfo.stemBlockSize, constInfo.stemStride, constInfo.initialBlocks, constInfo.windowSize,
    //                     constInfo.mBaseSize, constInfo.nBaseSize, constInfo.scoreScale, constInfo.alpha,
    //                     constInfo.kBlockNumRateMedium, constInfo.kBlockNumBiasMedium, constInfo.kBlockNumRateLarge,
    //                     constInfo.kBlockNumBiasLarge, static_cast<uint32_t>(singleCoreScoreWorkspaceBytes));
    // }

    InitBuffers();
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
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
    tempLoopInfo.s2ValidSize =
        CalcS2ValidSize(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
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
    tempLoopInfo.s2LoopEnd = isEnd ? Min(splitCoreInfo.s2End, tileS2LoopEnd) : tileS2LoopEnd;

}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    tempLoopInfo.needDealActS1LessThanS1 = false;
    tempLoopInfo.s2ValidSize = 0U;
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
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
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.isValid = s2LoopIdx <= tempLoopInfo.s2LoopEnd;

    runInfo.s2Start = splitCoreInfo.s2Start;
    runInfo.s2LoopEnd = tempLoopInfo.s2LoopEnd;

    if (!runInfo.isValid) {
        return;  // 需要验证， v1 时候需要runInfo
    }

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.s2ValidSize + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        SICommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, SICommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (Q_LAYOUT_T == SI_LAYOUT::TND) {
            uint32_t qSeqLensGmIdx = (constInfo.batchSupperFlag) ? runInfo.bIdx : runInfo.bIdx - 1;
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : qSeqLensGm.GetValue(qSeqLensGmIdx);
        } else {  // BNSD
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
        }
        uint64_t qBatchOffset = actualSeqQPrefixSum * constInfo.qHeadNum;
        uint64_t qHeadOffset = runInfo.n2Idx * constInfo.gSize * constInfo.qSeqSize;
        uint64_t qBlockOffset = runInfo.gS1Idx * constInfo.s1BaseSize;
        // qflat: B,N1,Qb,D. gS1Idx is a Qb tile index; M = QbTile * G.
        queryCoreOffset = (qBatchOffset + qHeadOffset + qBlockOffset) *
                          constInfo.headDim;
        // sparse_indices: B,N2,G,S1,S2
        indiceOutCoreOffset = (qBatchOffset + qHeadOffset) * tempLoopInfo.actS2Size;
        // sparse_seq_len: B,N2,G,S1
        indiceLenCoreOffset = qBatchOffset + qHeadOffset;
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == SI_LAYOUT::TND) { // T N2 D
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : kvSeqLensGm.GetValue(runInfo.bIdx - 1);
        actualSeqKPrefixSum = actualSeqKPrefixSum / constInfo.cmpRatio;
    } else {
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
    }
    runInfo.numPromptTokens = GetPromptTokenLen(runInfo.bIdx);
    uint64_t kBatchOffset = actualSeqKPrefixSum * constInfo.kHeadNum;
    uint64_t kHeadOffset = runInfo.n2Idx * constInfo.kSeqSize;
    keyCoreOffset =
        (kBatchOffset + kHeadOffset + runInfo.s2Idx * constInfo.s2BaseSize) * constInfo.headDim;
    keyScaleCoreOffset = kBatchOffset + kHeadOffset + runInfo.s2Idx * constInfo.s2BaseSize;
    weightsCoreOffset = kBatchOffset + kHeadOffset + runInfo.s2Idx * constInfo.s2BaseSize;
    vBiasCoreOffset = (runInfo.bIdx * constInfo.kHeadNum * runInfo.actS2Size) + (runInfo.n2Idx * runInfo.actS2Size);
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset;
    runInfo.tensorKeyScaleOffset = keyScaleCoreOffset;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.tensorVBiasOffset = vBiasCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
    runInfo.indiceLenOffset = indiceLenCoreOffset;
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
            AscendC::InitGlobalMemory(indiceOut, dealIndiceSize, static_cast<int32_t>(constInfo.INVALID_IDX));
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
    if(!splitCoreInfo.isCoreEnable){
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<SICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(SICommon::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreSetFlag<SICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(SICommon::ConstInfo::CROSS_VC_EVENT + 1);
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
            // 短路分支：s2ValidSize <= constInfo.initialBlocks时直接输出连续索引
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
        CrossCoreWaitFlag<SICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(SICommon::ConstInfo::CROSS_VC_EVENT + 0); 
        CrossCoreWaitFlag<SICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(SICommon::ConstInfo::CROSS_VC_EVENT + 1); 
    }
}

template <typename SIT>
__aicore__ inline void SIPreload<SIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, SICommon::RunInfo runInfo)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(runInfo);
    } else {
        vectorService.ProcessVec1(runInfo);
        if (runInfo.isFirstS2InnerLoop) {
            vectorService.initS2LenToNeg(runInfo.actMBaseSize, runInfo.indiceOutOffset);
        }
        vectorService.ProcessTopK(runInfo, runInfo.isFirstS2InnerLoop, runInfo.isLastS2InnerLoop);
    }
}

}  // namespace SIKernel
#endif  // stem_indexer_KERNEL_H
