/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_kernel_nonquant.h
 * \brief
 */

#ifndef FIA_KERNEL_NONQUANT_H
#define FIA_KERNEL_NONQUANT_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../fia_public_define.h"
#include "../vector_common.h"
#include "kernel_common.h"
#include "../memory_copy.h"
#include "fia_block_cube_nonquant.h"
#include "fia_block_cube_nonquant_gqa.h"
#include "fia_block_vec_nonquant.h"
#include "fia_block_vec_flashdecode.h"

using namespace optiling;
using namespace matmul;
using namespace AttentionCommon;
using namespace fa_base_vector;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename FIAT>
class FiaKernelNonQuant {
public:
    __aicore__ inline FiaKernelNonQuant(){};
    __aicore__ inline void Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
         __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
         __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
         __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
         __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
         __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
         __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
         __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
         __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *learnableSink, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
         __gm__ uint8_t *workspace, const FusedInferAttentionScoreTilingData *__restrict tiling,
         __gm__ uint8_t *gmTiling, TPipe *tPipe, bool isPrefix = false);
    __aicore__ inline void Process();

protected:
    // =================================模板参数与数据类型定义=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;
    using OUT_T = typename FIAT::outputType;
    using ORIGIN_T = typename FIAT::orginalType;
    static constexpr bool PAGE_ATTENTION = FIAT::pageAttention;
    static constexpr bool FLASH_DECODE = FIAT::flashDecode;
    static constexpr FIA_LAYOUT LAYOUT_T = FIAT::layout;
    static constexpr FIA_LAYOUT KV_LAYOUT_T = FIAT::kvLayout;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_T>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_T, PAGE_ATTENTION>();
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    static constexpr uint8_t PER_CHANNEL_MODE = 0; // 伪量化: K V per-channel
    static constexpr uint8_t ANTIQUANT_MODE = FIAT::antiquantMode;
    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool ANTIQUANT_PER_CHANNEL = (ANTIQUANT && (ANTIQUANT_MODE == PER_CHANNEL_MODE));
    using Q_ROPE_T = typename AscendC::Conditional<ANTIQUANT, Q_T, ORIGIN_T>::type;
    using K_ROPE_T = typename AscendC::Conditional<ANTIQUANT, KV_T, ORIGIN_T>::type;    

    using UPDATE_T = typename AscendC::Conditional<QUANT || ANTIQUANT, half, T>::type;
    using TMP_T = typename AscendC::Conditional<ANTIQUANT, half, T>::type;
    using MM1_OUT_T = typename AscendC::Conditional<QUANT, int32_t, TMP_T>::type;
    using MM2_OUT_T = typename AscendC::Conditional<QUANT, half, TMP_T>::type;
    using PSE_T = typename AscendC::Conditional<IsSameType<Q_T, int8_t>::value, half, Q_T>::type;

    // ==============================Service Define==============================
    FiaBlockCubeNonQuantGqa<FIAT> matmulService;
    FiaBlockVecNonQuant<FIAT> vectorService;
    FiaBlockVecFlashDecode<FIAT> fdService;

    // =================================常量区=================================
    static constexpr uint32_t PRELOAD_NUM = 2;
    static constexpr uint32_t N_BUFFER_M_BASIC_SIZE = 256;
    static constexpr uint32_t FIA_PRELOAD_TASK_CACHE_SIZE = 3;

    static constexpr uint32_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint32_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint32_t SYNC_C2_V2_FLAG = 9;
    static constexpr uint32_t SYNC_C2_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_NUPDATE_C2_FLAG = 5;
    static constexpr int64_t fdPrefetchLen = 2;

    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;

    // ==============================TilingData&TPipe==============================
    const FusedInferAttentionScoreTilingData *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;

    // ================================Required Global Tensor=================================
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<bfloat16_t> sinkGm;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;
    __gm__ uint8_t *key_ = nullptr;
    __gm__ uint8_t *value_ = nullptr;

    // ================================Optional Global Tensor=================================
    GlobalTensor<PSE_T> pseShiftGm;
    // actual seq lens
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    // post quant
    GlobalTensor<float> quantScale2Gm;
    GlobalTensor<float> quantOffset2Gm;
    GlobalTensor<bfloat16_t> quantScale2Bf16Gm;
    GlobalTensor<bfloat16_t> quantOffset2Bf16Gm;
    // left padding
    GlobalTensor<int64_t> queryPaddingSizeGm;
    GlobalTensor<int64_t> kvPaddingSizeGm;
    // block table
    GlobalTensor<int32_t> blockTableGm;
    // share prefix
    GlobalTensor<KV_T> keySharePrefixGm;
    GlobalTensor<KV_T> valueSharePrefixGm;
    GlobalTensor<uint64_t> actualSeqLengthsPrefixGm;

    // ===========================Workspace Global Tensor===========================
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    // ================================Task Info====================================
    ConstInfo constInfo{};
    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;
    uint64_t bn2IdxInCurCore = 0ULL;
    // offset
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorARopeCoreOffset = 0ULL;
    uint64_t tensorBRopeCoreOffset = 0ULL;
    uint64_t attenMaskCoreOffset = 0ULL;
    uint64_t s2BatchBaseOffset = 0;

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    ActualSeqLensParser<Q_MODE> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE> kvActSeqLensParser;
    uint32_t curS2Start;
    uint32_t curS2End;
    // ===============================Util functions================================
    template <typename T>
    __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    template <typename T1, typename T2>
    __aicore__ inline T1 Min(T1 a, T2 b)
    {
        return (a > b) ? (b) : (a);
    }

    template <typename T1, typename T2>
    __aicore__ inline T1 Max(T1 a, T2 b)
    {
        return (a > b) ? (a) : (b);
    }
    // ================================Init functions==================================
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    __aicore__ inline void InitActualSeqLenQ(__gm__ uint8_t *actualSeqLengthsQ);
    __aicore__ inline void InitActualSeqLenKV(__gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void InitOutputSingleCore();
    // ================================Tool============================================
    __aicore__ inline uint32_t GetBIdx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetN2Idx(uint32_t bN2Idx);
    __aicore__ inline void GetSafeActToken(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &safePreToken,
                                           int64_t &safeNextToken);
    // ================================Process functions================================
    __aicore__ inline void FlashAttention();
    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfo &info);
    __aicore__ inline void CalcAccumOffset(RunInfo &info);
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    __aicore__ inline void ComputeMm2(const RunInfo &info);
    __aicore__ inline void ComputeVec1(const RunInfo &info);
    __aicore__ inline void ComputeVec2(const RunInfo &info);
    __aicore__ inline void FlashDecode();
    // ================================PIPE Control=====================================
    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur);
    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur);
    __aicore__ inline void CalcCurS2StartEnd(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur);
    __aicore__ inline void UpdateAxisInfo(uint32_t &bN2Cur, uint32_t &gS1Cur, uint32_t &s2Cur);
    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE]);
    __aicore__ inline void DealZeroActSeqLen(uint32_t &bN2Cur, uint32_t &gS1Cur, uint32_t &s2Cur);
    __aicore__ inline bool ShouldExecuteTask(RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE]);
    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE]);
};

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::InitTilingData()
{
    usedCoreNum = tilingData->baseParams.usedCoreNum;

    constInfo.scaleValue = tilingData->baseParams.scaleValue;
    constInfo.batchSize = tilingData->baseParams.bSize;
    constInfo.gSize = tilingData->baseParams.gSize;
    constInfo.kvHeadNum = tilingData->baseParams.n2Size;
    constInfo.qHeadNum = constInfo.gSize * constInfo.kvHeadNum;
    constInfo.kvSeqSize = tilingData->baseParams.s2Size;
    constInfo.qSeqSize = tilingData->baseParams.s1Size;

    constInfo.attenMaskFlag = (tilingData->maskParams.attenMaskFlag != 0) ? true : false;
    constInfo.attenMaskSize = tilingData->maskParams.attenMaskSize;
    constInfo.attenMaskStride = tilingData->maskParams.attenMaskStride;
    constInfo.sparseMode = tilingData->maskParams.sparseMode;
    constInfo.preToken = tilingData->maskParams.preToken;
    constInfo.nextToken = tilingData->maskParams.nextToken;
    constInfo.isRowInvalid = (tilingData->maskParams.isRowInvalid != 0);

    constInfo.softmaxLseFlag = tilingData->baseParams.softmaxLseFlag;

    constInfo.maxBlockNumPerBatch = tilingData->pageAttenParams.maxBlockNumPerBatch;
    constInfo.kvCacheBlockSize = tilingData->pageAttenParams.blockSize;
    constInfo.outputLayout = static_cast<FIA_LAYOUT>(tilingData->baseParams.outputLayout);
    constInfo.mBaseSize = tilingData->innerSplitParams.mBaseSize;
    constInfo.s2BaseSize = tilingData->innerSplitParams.s2BaseSize;
    constInfo.batchContinuous = tilingData->baseParams.batchContinuous;
    constInfo.l2CacheOffFlag = tilingData->baseParams.l2CacheOffFlag;

    constInfo.headDim = tilingData->baseParams.headDim;
    constInfo.headDimRope = tilingData->baseParams.headDimRope;
    constInfo.headDimAlign = Align(constInfo.headDim, (uint64_t)BYTE_BLOCK);

    constInfo.mmResUbSize = tilingData->workspaceParams.mm1ResSize;
    constInfo.bmm2ResUbSize = tilingData->workspaceParams.mm2ResSize;
    constInfo.vec1ResUbSize = constInfo.mmResUbSize;

    constInfo.preLoadNum = PRELOAD_NUM;
    constInfo.nBufferMBaseSize = N_BUFFER_M_BASIC_SIZE;
    constInfo.syncV0C1 = SYNC_V0_C1_FLAG;
    constInfo.syncC1V1 = SYNC_C1_V1_FLAG;
    constInfo.syncV1C2 = SYNC_V1_C2_FLAG;
    constInfo.syncC2V2 = SYNC_C2_V2_FLAG;
    constInfo.syncC2V1 = SYNC_C2_V1_FLAG;
    constInfo.syncV1NupdateC2 = SYNC_V1_NUPDATE_C2_FLAG;
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::InitOutputSingleCore()
{
    if (usedCoreNum != 0) {
        uint32_t initOutputEventId = 0U;
        SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        uint64_t tSize = constInfo.batchSize * constInfo.qSeqSize;
        if constexpr (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::NTD) {
            tSize = qActSeqLensParser.GetTSize();
        }
        uint64_t totalOutputSize = tSize * constInfo.qHeadNum * constInfo.headDim;
        uint64_t singleCoreSize = (totalOutputSize + (2 * usedCoreNum) - 1) / (2 * usedCoreNum); // 2 means c:v = 1:2
        uint64_t tailSize = totalOutputSize - tmpBlockIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        matmul::InitOutput<OUT_T>(attentionOutGm[tmpBlockIdx * singleCoreSize], singleInitOutputSize, 0);
        SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);

        if (constInfo.softmaxLseFlag) {
            float lseInitValue = constInfo.FLOAT_INF;
            uint64_t totalLseSize = tSize * constInfo.qHeadNum;
            uint64_t singleCoreLseSize = (totalLseSize + (2 * usedCoreNum) - 1) / (2 * usedCoreNum); // 2 means c:v = 1:2;
            uint64_t tailLseSize = totalLseSize - tmpBlockIdx * singleCoreLseSize;
            uint64_t singleInitOutputLseSize = tailLseSize < singleCoreLseSize ? tailLseSize : singleCoreLseSize;
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm[tmpBlockIdx * singleCoreLseSize], singleInitOutputLseSize, lseInitValue);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
        WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        SyncAll();
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::InitActualSeqLenQ(__gm__ uint8_t *actualSeqLengthsQ)
{
    constInfo.actualLenQDims = tilingData->baseParams.actualSeqS1Dims;
    constInfo.accumQSeqFlag = tilingData->baseParams.accumQSeqFlag;
    if (constInfo.actualLenQDims != 0) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    qActSeqLensParser.Init(actualSeqLengthsGmQ, constInfo.actualLenQDims, constInfo.qSeqSize);
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::InitActualSeqLenKV(__gm__ uint8_t *actualSeqLengths)
{
    constInfo.actualLenDims = tilingData->baseParams.actualSeqS2Dims;
    constInfo.accumKVSeqFlag = tilingData->baseParams.accumKVSeqFlag;
    if (constInfo.actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, constInfo.actualLenDims);
    }
    kvActSeqLensParser.Init(actualSeqLengthsGm, constInfo.actualLenDims, constInfo.kvSeqSize);
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::InitWorkspace(__gm__ uint8_t *workspace)
{
    // workspace 内存排布
    // |Q--|mm1ResGm(存S)|vec1ResGm(存A1,A2)|mm2ResGm(存O)|vec2ResGm
    // |Core0_Q1-Core0_Q2-Core1_Q1-Core1_Q2....Core32_Q1-Core32_Q2|Core0_mmRes
    static constexpr uint32_t dbWorkspaceRatio = PRELOAD_NUM;
    uint64_t offset = 0;
    // mm1Res
    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T);

    // vec1Res
    vec1ResGm.SetGlobalBuffer(
        (__gm__ KV_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T);

    // mm2Res
    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);

    // vec2Res
    vec2ResGm.SetGlobalBuffer(
        (__gm__ UPDATE_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(UPDATE_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(UPDATE_T);

    // flash decode input
    if constexpr (FLASH_DECODE) {
        accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset = offset + tilingData->workspaceParams.fdAccumOutSize * sizeof(float);
        lseSumFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        lseMaxFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset) + tilingData->workspaceParams.fdLogSumExpSize / 2);
        offset = offset + tilingData->workspaceParams.fdLogSumExpSize * sizeof(float);
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
    __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *learnableSink, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, const FusedInferAttentionScoreTilingData *__restrict tiling,
    __gm__ uint8_t *gmTiling, TPipe *tPipe, bool isPrefix)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    // init tiling data
    tilingData = tiling;

    InitTilingData();
    // // 初始化计算参数
    InitActualSeqLenQ(actualSeqLengthsQ);
    InitActualSeqLenKV(actualSeqLengths);
    InitCalcParamsEach();
    constInfo.ropeSplitMode = (queryRope != nullptr);

    pipe = tPipe;
    keyPtr = key;
    valuePtr = value;

    // init global buffer
    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    if (constInfo.softmaxLseFlag) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    }

    if ASCEND_IS_AIV {
        InitOutputSingleCore();
    }

    InitWorkspace(workspace);

    if ASCEND_IS_AIC {
        matmulService.InitParams(constInfo);
        matmulService.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths,
            deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,
            blockTable, queryPaddingSize, kvPaddingSize,
            keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
            keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
            queryRope, keyRope, keyRopeAntiquantScale,
            attentionOut, softmaxLse);
        matmulService.InitMm1GlobalTensor(mm1ResGm);
        matmulService.InitMm2GlobalTensor(vec1ResGm, mm2ResGm);
    } else {
        if constexpr (FLASH_DECODE) {
            fdService.InitParams(constInfo);
            fdService.InitGlobalTensor(lseMaxFdGm, lseSumFdGm, accumOutGm, attentionOutGm, 
                                       actualSeqLengthsGmQ, actualSeqLengthsGm);
            if (constInfo.softmaxLseFlag) {
                fdService.InitSoftmaxLseGm(softmaxLseGm);
            }
            if (learnableSink != nullptr) {
                sinkGm.SetGlobalBuffer((__gm__ bfloat16_t *)learnableSink);
                fdService.InitLearnableSinkGm(sinkGm);
            }
        }
        vectorService.InitParams(constInfo);
        vectorService.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths,
            deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,
            blockTable, queryPaddingSize, kvPaddingSize,
            keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
            keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
            queryRope, keyRope, keyRopeAntiquantScale, learnableSink,
            attentionOut, softmaxLse);
        vectorService.InitVec1GlobalTensor(vec1ResGm, mm1ResGm);
        vectorService.InitVec2GlobalTensor(vec2ResGm, mm2ResGm);
        vectorService.InitFlashDecodeGlobalTensor(accumOutGm, lseMaxFdGm, lseSumFdGm);
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::InitCalcParamsEach()
{
    // 这里是编译器优化写法，定义一个局部数组变量coreSidxEnd(存在栈上)，使用copy_data_align64接口
    // 可以只从ub中拷贝tiling中coreSidxEnd的内容到栈上，而非将整个increFlashAttentionCoreParams
    // 内容拷贝到栈，减少拷贝时间。
#ifdef ASCENDC_CPU_DEBUG
    const uint32_t *bN2End = tilingData->outerSplitParams.bN2End;
    const uint32_t *gS1End = tilingData->outerSplitParams.gS1End;
    const uint32_t *s2End = tilingData->outerSplitParams.s2End;
    const uint32_t *s2SplitStartIdxOfCore = tilingData->fdParams.s2SplitStartIdxOfCore;
#else
    uint32_t bN2End[ARRAY_SIZE(tilingData->outerSplitParams.bN2End)];
    uint32_t gS1End[ARRAY_SIZE(tilingData->outerSplitParams.gS1End)];
    uint32_t s2End[ARRAY_SIZE(tilingData->outerSplitParams.s2End)];
    uint32_t s2SplitStartIdxOfCore[ARRAY_SIZE(tilingData->fdParams.s2SplitStartIdxOfCore)];
    copy_data_align64((uint8_t *)bN2End, (uint8_t *)(tilingData->outerSplitParams.bN2End), sizeof(bN2End));
    copy_data_align64((uint8_t *)gS1End, (uint8_t *)(tilingData->outerSplitParams.gS1End), sizeof(gS1End));
    copy_data_align64((uint8_t *)s2End, (uint8_t *)(tilingData->outerSplitParams.s2End), sizeof(s2End));
    copy_data_align64((uint8_t *)s2SplitStartIdxOfCore,
                      (uint8_t *)(tilingData->fdParams.s2SplitStartIdxOfCore), sizeof(s2SplitStartIdxOfCore));
#endif

    // 任务起始位置
    if (aiCoreIdx == 0) {
        constInfo.bN2Start = 0;
        constInfo.gS1Start = 0;
        constInfo.s2Start = 0;
    } else {
        constInfo.bN2Start = bN2End[aiCoreIdx - 1];
        constInfo.gS1Start = gS1End[aiCoreIdx - 1];
        constInfo.s2Start = s2End[aiCoreIdx - 1];
    }
    // 任务结束位置
    constInfo.bN2End = bN2End[aiCoreIdx];
    constInfo.gS1End = gS1End[aiCoreIdx];
    constInfo.s2End = s2End[aiCoreIdx];

    // 首个S1G块、最后一个S1G块的S2是否被切分
    constInfo.headS2Split = (constInfo.s2Start != 0);
    constInfo.tailS2Split = (constInfo.s2End != 0);

    // xxx
    constInfo.coreStartKVSplitPos = s2SplitStartIdxOfCore[aiCoreIdx];
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::CalcAccumOffset(RunInfo &info)
{
    if ASCEND_IS_AIV {
#ifdef ASCENDC_CPU_DEBUG
        const uint32_t *bN2IdxOfFdHead = tilingData->fdParams.bN2IdxOfFdHead;
        const uint32_t *gS1IdxOfFdHead = tilingData->fdParams.gS1IdxOfFdHead;
        const uint32_t *s2SplitNumOfFdHead = tilingData->fdParams.s2SplitNumOfFdHead;
#else
        uint32_t bN2IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.bN2IdxOfFdHead)];
        uint32_t gS1IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1IdxOfFdHead)];
        uint32_t s2SplitNumOfFdHead[ARRAY_SIZE(tilingData->fdParams.s2SplitNumOfFdHead)];
        copy_data_align64((uint8_t *)bN2IdxOfFdHead, (uint8_t *)(tilingData->fdParams.bN2IdxOfFdHead),
                        sizeof(bN2IdxOfFdHead));
        copy_data_align64((uint8_t *)gS1IdxOfFdHead, (uint8_t *)(tilingData->fdParams.gS1IdxOfFdHead),
                        sizeof(gS1IdxOfFdHead));
        copy_data_align64((uint8_t *)s2SplitNumOfFdHead, (uint8_t *)(tilingData->fdParams.s2SplitNumOfFdHead),
                        sizeof(s2SplitNumOfFdHead));
#endif
        uint64_t accumTmpOutNum = 0;
        int taskId = 0;
        uint32_t curbN2Idx = info.bIdx * constInfo.kvHeadNum + info.n2Idx;
        while (bN2IdxOfFdHead[taskId] != curbN2Idx || gS1IdxOfFdHead[taskId] * constInfo.mBaseSize != info.gS1Idx) {
            accumTmpOutNum += s2SplitNumOfFdHead[taskId]; // 计算前面的workspace数
            taskId++;
        }
        info.accumTmpOutNum = accumTmpOutNum;
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::GetSafeActToken(int64_t actSeqLensQ, int64_t actSeqLensKv,
                                                           int64_t &safePreToken, int64_t &safeNextToken) 
{
    if (constInfo.sparseMode == DEFAULT_MASK) {
        safePreToken = Max(-actSeqLensKv, safePreToken);
        safePreToken = Min(safePreToken, actSeqLensQ);
        safeNextToken = Max(-actSeqLensQ, safeNextToken);
        safeNextToken = Min(safeNextToken, actSeqLensKv);
    } else if (constInfo.sparseMode == BAND) {
        safePreToken = Max(-actSeqLensQ, safePreToken);
        safePreToken = Min(safePreToken, actSeqLensKv);
        safeNextToken = Max(-actSeqLensKv, safeNextToken);
        safeNextToken = Min(safeNextToken, actSeqLensQ);
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                                      RunInfo &info)
{
    info.loop = loop;

    info.bIdx = GetBIdx(bN2Cur);
    info.n2Idx = GetN2Idx(bN2Cur);
    info.gS1Idx = gS1Cur * constInfo.mBaseSize;
    info.s2Idx = s2Cur;

    info.actS1Size = actSeqLensQ;
    info.actS2Size = actSeqLensKv;

    info.actMBaseSize = constInfo.mBaseSize;
    uint64_t gS1Size = info.actS1Size * constInfo.gSize;
    if (((gS1Cur + 1) * constInfo.mBaseSize) > gS1Size) {
        info.actMBaseSize = gS1Size - gS1Cur * constInfo.mBaseSize;
    }
    info.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    if (((s2Cur + 1) * constInfo.s2BaseSize) > info.actS2Size) {
        info.actualSingleProcessSInnerSize = info.actS2Size - s2Cur * constInfo.s2BaseSize;
    }
    info.actualSingleProcessSInnerSizeAlign =
        Align((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)BYTE_BLOCK);

    if (constInfo.batchContinuous) {
        info.isChangeBatch = false;
    } else {
        if (loop == 0) { // 第一个有效任务才需要重置KV的tensor
            info.isChangeBatch = true;
        } else {
            info.isChangeBatch = (info.n2Idx == 0 && s2Cur == curS2Start);
        }
    }

    int64_t safePreToken = constInfo.preToken;
    int64_t safeNextToken = constInfo.nextToken;
    GetSafeActToken(info.actS1Size, info.actS2Size, safePreToken, safeNextToken);
    if (constInfo.sparseMode == BAND) {
        info.preTokensPerBatch = safePreToken;
        info.nextTokensPerBatch =
            static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size) + safeNextToken;
    } else if ((constInfo.sparseMode == DEFAULT_MASK) && constInfo.attenMaskFlag) {
        info.nextTokensPerBatch = safeNextToken;
        info.preTokensPerBatch = 
            static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size) + safePreToken;
    } else {
        info.nextTokensPerBatch = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size);
        info.preTokensPerBatch = 0;
    }

    // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
    // 情况2: loop=0时, 如果(bN2Start, gS1Start, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
    // 情况3: loop=0时, 如果(bN2Start, gS1Start, s2Start)任务无效, 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
    info.isFirstSInnerLoop = ((loop == 0) || (s2Cur == curS2Start));
    if (info.isFirstSInnerLoop) {
        bn2IdxInCurCore++;
    }
    info.bn2IdxInCurCore = bn2IdxInCurCore - 1;

    info.tndIsS2SplitCore = false;
    info.tndCoreStartKVSplitPos = 0;
    info.isLastS2Loop = (s2Cur + 1 == curS2End);
    info.curSInnerLoopTimes = curS2End - curS2Start;

    if (constInfo.bN2Start == constInfo.bN2End && constInfo.gS1Start == constInfo.gS1End) {
        // 所有任务属于同一个S1G
        info.tndIsS2SplitCore = true;
        info.tndCoreStartKVSplitPos = constInfo.coreStartKVSplitPos;
    } else {
        if (constInfo.headS2Split && (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1Start)) {
            // 当前任务属于第一个S1G, 并且第一个S1G的S2被切分了
            info.tndIsS2SplitCore = true;
            info.tndCoreStartKVSplitPos = constInfo.coreStartKVSplitPos;
        } else if (constInfo.tailS2Split && (bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1End)) {
            // 当前任务属于最后一个S1G, 并且最后一个S1G的S2被切分了
            info.tndIsS2SplitCore = true;
        }
    }

    if constexpr (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {
            CalcAccumOffset(info);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::ComputeMm1(const RunInfo &info)
{
    matmulService.ComputeMm1(info);
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::ComputeMm2(const RunInfo &info)
{
    matmulService.ComputeMm2(info);
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::ComputeVec1(const RunInfo &info)
{
    vectorService.ComputeVec1(info);
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::ComputeVec2(const RunInfo &info)
{
    vectorService.ComputeVec2(info);
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::FlashDecode()
{
    fdService.InitBuffers(pipe);
    AscendC::ICachePreLoad(fdPrefetchLen);
    SyncAll();
    if ASCEND_IS_AIV {
        #ifdef ASCENDC_CPU_DEBUG
            const uint32_t *bN2IdxOfFdHead = tilingData->fdParams.bN2IdxOfFdHead;
            const uint32_t *gS1IdxOfFdHead = tilingData->fdParams.gS1IdxOfFdHead;
            const uint32_t *s2SplitNumOfFdHead = tilingData->fdParams.s2SplitNumOfFdHead;
            const uint32_t *gS1IdxEndOfFdHead = tilingData->fdParams.gS1IdxEndOfFdHead;
            const uint32_t *gS1IdxEndOfFdHeadSplit = tilingData->fdParams.gS1IdxEndOfFdHeadSplit;
            const uint32_t *gS1SplitNumOfFdHead = tilingData->fdParams.gS1SplitNumOfFdHead;
            const uint32_t *gS1LastPartSizeOfFdHead = tilingData->fdParams.gS1LastPartSizeOfFdHead;
        #else
            uint32_t bN2IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.bN2IdxOfFdHead)];
            uint32_t gS1IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1IdxOfFdHead)];
            uint32_t s2SplitNumOfFdHead[ARRAY_SIZE(tilingData->fdParams.s2SplitNumOfFdHead)];
            uint32_t gS1IdxEndOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1IdxEndOfFdHead)];
            uint32_t gS1IdxEndOfFdHeadSplit[ARRAY_SIZE(tilingData->fdParams.gS1IdxEndOfFdHeadSplit)];
            uint32_t gS1SplitNumOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1SplitNumOfFdHead)];
            uint32_t gS1LastPartSizeOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1LastPartSizeOfFdHead)];
            copy_data_align64((uint8_t *)bN2IdxOfFdHead, (uint8_t *)(tilingData->fdParams.bN2IdxOfFdHead),
                        sizeof(bN2IdxOfFdHead));
            copy_data_align64((uint8_t *)gS1IdxOfFdHead, (uint8_t *)(tilingData->fdParams.gS1IdxOfFdHead),
                        sizeof(gS1IdxOfFdHead));
            copy_data_align64((uint8_t *)s2SplitNumOfFdHead, (uint8_t *)(tilingData->fdParams.s2SplitNumOfFdHead),
                        sizeof(s2SplitNumOfFdHead));
            copy_data_align64((uint8_t *)gS1IdxEndOfFdHead, (uint8_t *)(tilingData->fdParams.gS1IdxEndOfFdHead),
                        sizeof(gS1IdxEndOfFdHead));
            copy_data_align64((uint8_t *)gS1IdxEndOfFdHeadSplit,
                        (uint8_t *)(tilingData->fdParams.gS1IdxEndOfFdHeadSplit),
                        sizeof(gS1IdxEndOfFdHeadSplit));
            copy_data_align64((uint8_t *)gS1SplitNumOfFdHead, (uint8_t *)(tilingData->fdParams.gS1SplitNumOfFdHead),
                        sizeof(gS1SplitNumOfFdHead));
            copy_data_align64((uint8_t *)gS1LastPartSizeOfFdHead,
                        (uint8_t *)(tilingData->fdParams.gS1LastPartSizeOfFdHead),
                        sizeof(gS1LastPartSizeOfFdHead));
        #endif
        FDparams fdParams = {bN2IdxOfFdHead, gS1IdxOfFdHead, s2SplitNumOfFdHead, gS1SplitNumOfFdHead, gS1LastPartSizeOfFdHead,
                gS1IdxEndOfFdHead, gS1IdxEndOfFdHeadSplit, tilingData->fdParams.usedVecNumOfFd,
                tilingData->fdParams.gS1BaseSizeOfFd};
        fdService.AllocEventID();
        fdService.InitDecodeParams();
        fdService.FlashDecode(fdParams);
        fdService.FreeEventID();
    }
}

template <typename FIAT>
__aicore__ inline uint32_t FiaKernelNonQuant<FIAT>::GetBIdx(uint32_t bN2Idx)
{
    return (bN2Idx / constInfo.kvHeadNum);
}

template <typename FIAT>
__aicore__ inline uint32_t FiaKernelNonQuant<FIAT>::GetN2Idx(uint32_t bN2Idx)
{
    return (bN2Idx % constInfo.kvHeadNum);
}

template <typename FIAT>
__aicore__ inline bool FiaKernelNonQuant<FIAT>::ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
{
    return ((bN2Cur != constInfo.bN2End) || (gS1Cur != constInfo.gS1End) || (s2Cur != constInfo.s2End));
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::DealZeroActSeqLen(uint32_t &bN2Cur, uint32_t &gS1Cur, uint32_t &s2Cur)
{
    // 对整个batch的结果置0
    if constexpr (POST_QUANT) { // out int8
  
    } else {
        uint32_t n2Idx = GetN2Idx(bN2Cur);
        uint32_t bIdx = GetBIdx(bN2Cur);
        if (constInfo.outputLayout == FIA_LAYOUT::BSND || constInfo.outputLayout == FIA_LAYOUT::BSH) {
            OffsetCalculator<GmFormat::BSNGD> offsetCalculator;
            offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize, constInfo.qSeqSize, constInfo.headDim, 
                                  actualSeqLengthsGmQ, constInfo.actualLenQDims);
            DealActSeqLenIsZero<GmFormat::BSNGD, OUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if (constInfo.outputLayout == FIA_LAYOUT::BNSD) {
            OffsetCalculator<GmFormat::BNGSD> offsetCalculator;
            offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize, constInfo.qSeqSize, constInfo.headDim, 
                                  actualSeqLengthsGmQ, constInfo.actualLenQDims);
            DealActSeqLenIsZero<GmFormat::BNGSD, OUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if (constInfo.outputLayout == FIA_LAYOUT::NBSD) {
            OffsetCalculator<GmFormat::NGBSD> offsetCalculator;
            offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize, constInfo.qSeqSize, constInfo.headDim, 
                                  actualSeqLengthsGmQ, constInfo.actualLenQDims);
            DealActSeqLenIsZero<GmFormat::NGBSD, OUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if (constInfo.outputLayout == FIA_LAYOUT::TND) {
            OffsetCalculator<GmFormat::TNGD> offsetCalculator;
            offsetCalculator.Init(constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim, actualSeqLengthsGmQ, constInfo.actualLenQDims);
            DealActSeqLenIsZero<GmFormat::TNGD, OUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if (constInfo.outputLayout == FIA_LAYOUT::NTD) {
            OffsetCalculator<GmFormat::NGTD> offsetCalculator;
            offsetCalculator.Init(constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim, actualSeqLengthsGmQ, constInfo.actualLenQDims);
            DealActSeqLenIsZero<GmFormat::NGTD, OUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::UpdateAxisInfo(uint32_t &bN2Cur, uint32_t &gS1Cur, uint32_t &s2Cur)
{
    uint64_t s2LoopTimes = (actSeqLensKv + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
    uint64_t gS1LoopTimes = (gS1Size + constInfo.mBaseSize - 1) / constInfo.mBaseSize;

    // 当前S2未处理完
    if (s2Cur + 1 < s2LoopTimes) {
        s2Cur++;
        return;
    }

    // 当前BN2未处理完
    s2Cur = 0;
    if (gS1Cur + 1 < gS1LoopTimes) {
        gS1Cur++;
        return;
    }

    // 当前BN2已处理完
    gS1Cur = 0;
    bN2Cur++;
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                                      RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE])
{
    RunInfo &extraInfo0 = extraInfo[loop % FIA_PRELOAD_TASK_CACHE_SIZE];       // 本轮任务
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % FIA_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % FIA_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    CalcParams(loop, bN2Cur, gS1Cur, s2Cur, extraInfo0);
    extraInfo0.isValid = true;
}

template <typename FIAT>
__aicore__ inline bool FiaKernelNonQuant<FIAT>::ShouldExecuteTask(RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE])
{
    for (uint32_t i = 0; i < FIA_PRELOAD_TASK_CACHE_SIZE; i++) {
        if (extraInfo[i].isValid) {
            return true;
        }
    }
    return false;
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::ExecuteTask(uint64_t loop, RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE])
{
    RunInfo &extraInfo0 = extraInfo[loop % FIA_PRELOAD_TASK_CACHE_SIZE];       // 本轮任务
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % FIA_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % FIA_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    if (extraInfo0.isValid) {
        if ASCEND_IS_AIC {
            ComputeMm1(extraInfo0);
        }
    }
    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            ComputeVec1(extraInfo2);
        }
        if ASCEND_IS_AIC {
            ComputeMm2(extraInfo2);
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            ComputeVec2(extraInfo1);
        }
        extraInfo1.isValid = false;
    }
}

template <typename FIAT>
__aicore__ inline TASK_DEAL_MODE FiaKernelNonQuant<FIAT>::GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
{
    uint32_t bIdx = GetBIdx(bN2Cur);
    if (constInfo.actualLenDims == 0 && !constInfo.batchContinuous) {
        actSeqLensKv = fa_base_kernel::SeqLenFromTensorList<LAYOUT_T>(keyPtr, bIdx);
    } else {
        actSeqLensKv = kvActSeqLensParser.GetActualSeqLength(bIdx);
    }
    uint64_t s2LoopTimes = (actSeqLensKv + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;

    actSeqLensQ = qActSeqLensParser.GetActualSeqLength(bIdx);
    uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
    uint64_t gS1LoopTimes = (gS1Size + constInfo.mBaseSize - 1) / constInfo.mBaseSize;

    if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
        if (gS1Cur == 0 && s2Cur == 0) {
            return TASK_DEAL_MODE::DEAL_ZERO;
        }
        return TASK_DEAL_MODE::SKIP;
    }

    CalcCurS2StartEnd(bN2Cur, gS1Cur, s2Cur);
    if (s2Cur < curS2Start || s2Cur >= curS2End) {
        return TASK_DEAL_MODE::SKIP;
    }
    return TASK_DEAL_MODE::CREATE_TASK;
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::CalcCurS2StartEnd(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
{
    uint32_t s2End;
    if ((bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1End)) { // 当前任务属于最后一个S1G
        s2End = constInfo.s2End;
    } else {
        s2End = (actSeqLensKv + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }

    if (constInfo.attenMaskFlag == 0 || FLASH_DECODE) {
        curS2Start = 0;
        curS2End = s2End;
        return;
    }

    uint32_t gs1Idx = gS1Cur * constInfo.mBaseSize;
    int64_t sIdx;
    uint32_t s1BaseSize;
    if constexpr (GetOutUbFormat<LAYOUT_T>() == UbFormat::S1G) {
        sIdx = static_cast<int64_t>(gs1Idx / constInfo.gSize);
        s1BaseSize = constInfo.mBaseSize / constInfo.gSize + 1;
    } else {
        sIdx = static_cast<int64_t>(gs1Idx % actSeqLensQ);
        s1BaseSize = constInfo.mBaseSize;
    }

    if (sIdx + static_cast<int64_t>(s1BaseSize) > static_cast<int64_t>(actSeqLensQ)) {
        curS2Start = 0;
        curS2End = s2End;
        return;
    }

    uint32_t s2Start = bN2Cur == constInfo.bN2Start ? constInfo.s2Start : 0;
    int64_t safePreToken = constInfo.preToken;
    int64_t safeNextToken = constInfo.nextToken;
    GetSafeActToken(actSeqLensQ, actSeqLensKv, safePreToken, safeNextToken);

    int64_t preTokenLeftUp = (constInfo.sparseMode != BAND) ? safePreToken :
        (static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + safePreToken);
    int64_t nextTokenLeftUp;
    if (constInfo.sparseMode == DEFAULT_MASK || constInfo.sparseMode == ALL_MASK 
        || constInfo.sparseMode == LEFT_UP_CAUSAL) {
        nextTokenLeftUp = safeNextToken;
    } else if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
    } else {
        nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + safeNextToken;
    }

    int64_t s2FirstToken = ClipSInnerToken(sIdx - preTokenLeftUp, static_cast<int64_t>(s2Start), static_cast<int64_t>(actSeqLensKv));
    curS2Start = static_cast<uint32_t>(s2FirstToken / constInfo.s2BaseSize);

    int64_t s2LastToken = ClipSInnerToken(sIdx + nextTokenLeftUp + static_cast<int64_t>(s1BaseSize), 0, static_cast<int64_t>(s2End * constInfo.s2BaseSize));
    curS2End = static_cast<uint32_t>((s2LastToken + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize);
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::FlashAttention()
{
    RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE];

    uint32_t bN2Cur = constInfo.bN2Start;
    uint32_t gS1Cur = constInfo.gS1Start;
    uint32_t s2Cur = constInfo.s2Start;

    uint64_t createdTaskCount = 0;
    uint64_t executedTaskCount = 0;

    bool shouldDispatchTask = true;
    bool shouldExecuteTask = false;
    while (shouldDispatchTask || shouldExecuteTask) {
        // 分发任务
        shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
        if (shouldDispatchTask) {
            TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
            if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                // 创建任务
                CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, extraInfo);
                createdTaskCount++;
                UpdateAxisInfo(bN2Cur, gS1Cur, s2Cur);
            } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                DealZeroActSeqLen(bN2Cur, gS1Cur, s2Cur);
                UpdateAxisInfo(bN2Cur, gS1Cur, s2Cur);
                continue;
            } else if (taskDealMode == TASK_DEAL_MODE::SKIP) {
                UpdateAxisInfo(bN2Cur, gS1Cur, s2Cur);
                continue;
            }
        }
        // 执行任务
        shouldExecuteTask = ShouldExecuteTask(extraInfo);
        if (shouldExecuteTask) {
            ExecuteTask(executedTaskCount, extraInfo);
            executedTaskCount++;
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaKernelNonQuant<FIAT>::Process()
{
    // usedCoreNum: 使用的总核数
    if (aiCoreIdx < usedCoreNum) {
        if ASCEND_IS_AIC {
            matmulService.InitBuffers(pipe);
            matmulService.AllocEventID();
        } else {
            vectorService.InitBuffers(pipe);
            vectorService.AllocEventID();
        }
        FlashAttention();
        if ASCEND_IS_AIC {
            matmulService.FreeEventID();
        } else {
            vectorService.FreeEventID();
        }
    }

    if constexpr (FLASH_DECODE) {
        FlashDecode();
    }
}

#endif // FIA_KERNEL_NONQUANT_H