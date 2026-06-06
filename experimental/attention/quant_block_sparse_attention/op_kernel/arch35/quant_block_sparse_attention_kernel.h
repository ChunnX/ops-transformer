/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// BSA-COPY-OF: FIA fullquant GQA local migration.
// TODO: 待实现 - 当前已完成自包含迁移和命名/include 适配；sparseSeqLen/sparseIndices 流水、per-token qScale、vScale、空 sparse 输出和 mask sparse offset 仍需按 BSA 方案 2.4 实现。

/*!
 * ile quant_block_sparse_attention_kernel.h
 * \brief
 */

#ifndef QUANT_BLOCK_SPARSE_ATTENTION_KERNEL_H_
#define QUANT_BLOCK_SPARSE_ATTENTION_KERNEL_H_
#include "quant_block_sparse_attention_block_cube.h"
#include "quant_block_sparse_attention_block_vec.h"
#include "quant_block_sparse_attention_common_arch35.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_block_sparse_attention_attenmask.h"

#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"
#include "common/quant_block_sparse_attention_tiling_regbase.h"

#include "common/quant_block_sparse_attention_kvcache.h"
#include "kernel_operator_list_tensor_intf.h"
#include "adv_api/utils/init_global_memory.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
namespace BSAKernelBase {
template <bool isFp8>
struct Bmm2ResBuffSel {
    using Type = BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
};
}

template <typename CubeBlockType, typename VecBlockType>
class QuantBlockSparseAttentionKernel {
public:
    ARGS_TRAITS;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;
    __aicore__ inline QuantBlockSparseAttentionKernel() {};

    __aicore__ inline void InitBaseAPI(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                            __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                            __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
                            __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, 
                            __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, 
                            __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale, 
                            __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen, 
                            __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxMax, 
                            __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                            __gm__ uint8_t *workspace, const QuantBlockSparseAttentionTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                            __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                            __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                            __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
                            __gm__ uint8_t *queryRope,
                            __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen,
                            __gm__ uint8_t *keyRope, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
                            __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                            __gm__ uint8_t *softmaxOut, __gm__ uint8_t *workspace,
                            const QuantBlockSparseAttentionTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx, RunParamStr &runParam);
    __aicore__ inline void ComputeBmm1Tail(RunInfo &runInfo, RunParamStr &runParam);
    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvLen);
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr &runParam,
        RunInfo &runInfo);
    __aicore__ inline void ProcessMainLoop();

    TPipe *pipe;

    const QuantBlockSparseAttentionTilingData *__restrict tilingData;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr bool isFp8 = CubeBlockType::isFp8;
    static constexpr bool isMxfp8FullQuant = s1BaseSize == 128 && s2BaseSize == 512;
    static constexpr TPosition bmm2OutPos = CubeBlockType::bmm2OutPos;
    static constexpr bool bmm2Write2Ub = CubeBlockType::bmm2Write2Ub;
    static constexpr bool splitD =  CubeBlockType::splitD;
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint64_t SYNC_C1_V1_FLAG[2] = {0, 1};
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4};
    static constexpr uint64_t SYNC_C2_V2_FLAG[2] = {5, 6};
    BufferManager<BufferType::GM> gmBufferManager;
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers;

    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    using bmm2ResBufferType = typename BSAKernelBase::Bmm2ResBuffSel<isFp8>::Type;
    bmm2ResBufferType bmm2Buffers;

    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers; 
    CVSharedParams<isPa> sharedParams;
    using keyGmType = GlobalTensor<INPUT_T>;
    keyGmType keyGm;
    __gm__ int64_t *actualSeqQlenAddr;
    __gm__ int64_t *actualSeqKvlenAddr;
    int32_t aicIdx;

    ConstInfo constInfo;
    AttenMaskInfo attenMaskInfo;
    uint64_t s1OuterSizeAcc;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;
    uint64_t s1ScaleNumAcc;
    uint64_t s2ScaleNumAcc;

    int64_t actualKVPrefixLen = 0;

    CubeBlockType cubeBlock;
    VecBlockType vecBlock;

private:
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index,
                                                    RunParamStr &runParam);
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::InitBaseAPI(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
    __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, 
    __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *learnableSink,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const QuantBlockSparseAttentionTilingData *__restrict tiling, TPipe *tPipe)
{
    fa_base_matmul::idCounterNum = 0;
    constInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
    } else {
        constInfo.aivIdx = GetBlockIdx();
        this->aicIdx = constInfo.aivIdx >> 1;
        this->tilingData = tiling;
    }
    this->pipe = tPipe;
    vecBlock.InitVecBlock(tPipe, this->tilingData, this->sharedParams, this->aicIdx, constInfo.subBlockIdx, 
        attenMaskInfo);
    vecBlock.CleanOutput(softmaxLse, attentionOut, constInfo);
    InitMMResBuf();
    if ASCEND_IS_AIC {
        cubeBlock.InitCubeBlock(pipe, &l1BufferManager, query, key, value, blockTable);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);
        auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0);
        auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
        #pragma unroll
        for (int i = 0; i < sizeof(CVSharedParams<isPa>) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
            *tempTiling = *tempTilingSSbuf;
        }
    }

    this->ComputeConstexpr();
    this->InitGlobalBuffer(query, key, value, dropMask, paddingMask, attenMask, prefix, actualSeqLengths,
        actualSeqLengthsKv, deqScaleQ, deqScaleK, deqScaleV, pScale,
        keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, blockTable, queryPaddingSize, 
        kvPaddingSize, learnableSink, softmaxMax, softmaxSum, softmaxOut, workspace, tiling, tPipe);
    this->InitLocalBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
    __gm__ uint8_t *pScale, __gm__ uint8_t *keySharedPrefix,
    __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t*actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *learnableSink,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *workspace,
    const QuantBlockSparseAttentionTilingData *__restrict tiling, TPipe *tPipe)
{
    keyGm.SetGlobalBuffer((__gm__ INPUT_T *)(key));
    if (constInfo.isQHasLeftPadding) {
        constInfo.queryRightPaddingSize = ((__gm__ int64_t *)queryPaddingSize)[0];
        if (constInfo.queryRightPaddingSize < 0) {
            constInfo.queryRightPaddingSize = 0;
        }
    }
    if (constInfo.isKVHasLeftPadding) {
        constInfo.kvRightPaddingSize = ((__gm__ int64_t *)kvPaddingSize)[0];
        if (constInfo.kvRightPaddingSize < 0) {
            constInfo.kvRightPaddingSize = 0;
        }
    }
    if constexpr (hasAtten) {
        attenMaskInfo.prefixNAddr = prefix;
    }
    if constexpr (qLayout == LayOutTypeEnum::LAYOUT_TND) {
        actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
        actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
    }

    uint64_t singleCoreOffset = 0;
    if constexpr (!bmm2Write2Ub) {
        int64_t bmm2ResBlock = this->sharedParams.dSizeV;
        if constexpr (splitD) {
            bmm2ResBlock = (int64_t)dVTemplateType;
        }
        int64_t mm2ResultSize = (s1BaseSize) * bmm2ResBlock;
        int64_t mm2Offset = CeilDiv(mm2ResultSize, 128) * 128 * sizeof(T);
        int64_t vec2ResultSize = (s1BaseSize) * constInfo.dBasicBlock;
        int64_t vec2Offset = CeilDiv(vec2ResultSize, 128) * 128 * sizeof(T);
        singleCoreOffset = mm2Offset;
        if constexpr (splitD) {
            singleCoreOffset = mm2Offset + vec2Offset;
        }
        int64_t totalOffset = this->aicIdx * 3 * singleCoreOffset;
        gmBufferManager.Init(workspace + totalOffset);
        bmm2ResGmBuffers.Init(gmBufferManager, mm2Offset);
        workspace += (totalOffset + mm2Offset * 3);
    }
    vecBlock.InitGlobalBuffer(deqScaleQ, deqScaleK, deqScaleV, pScale,
        prefix, attenMask, blockTable, queryPaddingSize, kvPaddingSize, softmaxMax, softmaxSum, workspace, singleCoreOffset, this->aicIdx, constInfo);
    cubeBlock.InitCubeInput(key, value, &sharedParams, &attenMaskInfo, actualSeqQlenAddr, actualSeqKvlenAddr);
    cubeBlock.InitDequantParams(deqScaleQ, deqScaleK, deqScaleV);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::InitMMResBuf()
{
    uint32_t mm1OutDtype = sizeof(T);
    uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
    if constexpr (isMxfp8FullQuant) {
        mm1ResultSize /= 2;
    }
    constexpr uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    constexpr uint32_t mm2LeftSize = s1BaseSize * s2BaseSize * sizeof(INPUT_T);
    l1BufferManager.Init(pipe, 524288);
    l1PBuffers.Init(l1BufferManager, mm2LeftSize);
    if constexpr (bmm2Write2Ub) {
        ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize * 2);
        bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
        if ASCEND_IS_AIV {
            bmm2Buffers.Get().SetCrossCore();
            bmm2Buffers.Get().SetCrossCore();
        }
    } else {
        ubBufferManager.Init(pipe, mm1ResultSize * 2);
    }
    bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
    if ASCEND_IS_AIV {
        bmm1Buffers.Get().SetCrossCore();
        bmm1Buffers.Get().SetCrossCore();
    }
}
 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    constInfo.s1BaseSize = s1BaseSize;
    constInfo.s2BaseSize = s2BaseSize;

    constInfo.bSize = sharedParams.bSize;
    constInfo.t1Size = sharedParams.t1Size;
    constInfo.t2Size = sharedParams.t2Size;
    constInfo.n2Size = sharedParams.n2Size;
    constInfo.s1Size = sharedParams.s1Size;
    constInfo.s2Size = sharedParams.s2Size;
    constInfo.dSize = sharedParams.dSize;
    constInfo.dSizeV = sharedParams.dSizeV;
    constInfo.combineDim = (constInfo.dSize + constInfo.dSizeV + sizeof(float) + 31U) / 32U * 32U;
    constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    constInfo.dSizeRope = 0;
    constInfo.gSize = sharedParams.gSize;
    constInfo.s1OuterSize = sharedParams.s1OuterSize;
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
    constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
    constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
    constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
    constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
    constInfo.s2BaseN2Dv = s2BaseSize * constInfo.n2Dv;
    constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;
    constInfo.layoutType = sharedParams.layoutType;

    constInfo.s1D = constInfo.s1Size * constInfo.dSize;
    constInfo.gD = constInfo.gSize * constInfo.dSize;
    constInfo.n2GD = constInfo.n2Size * constInfo.gD; 
    constInfo.gS1D = constInfo.gS1 * constInfo.dSize;

    if constexpr (qLayout == LayOutTypeEnum::LAYOUT_TND) {
        constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;

        constInfo.mm1Ka = constInfo.n2Size * constInfo.gSize * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        constInfo.mm2Kb = constInfo.n2Dv;
        if (sharedParams.isGqa) {
            constInfo.mm1Ka = constInfo.dSize;
        }
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
            if (sharedParams.isPfaGS1Merge == 1) {
                constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
            } else if (sharedParams.isGqa == 1) {
                constInfo.attentionOutStride = 0;
            }
        }
    }

    if ASCEND_IS_AIV {
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        if constexpr (hasAtten) {
            attenMaskInfo.preTokens = sharedParams.preTokens;
            attenMaskInfo.nextTokens = sharedParams.nextTokens;
            attenMaskInfo.attenMaskShapeType = inputParamsRegbase.attenMaskShapeType;
            attenMaskInfo.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
            attenMaskInfo.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
            attenMaskInfo.bandIndex = inputParamsRegbase.bandIndex;
        }
        attenMaskInfo.compressMode = inputParamsRegbase.attenMaskCompressMode;
        constInfo.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
    }

    InitUniqueConstInfo();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::GetSeqQlenKvlenByBoidx(int64_t boIdx,
    int64_t &actualSeqQlen, int64_t &actualSeqKvlen)
{
    if (unlikely(boIdx == 0)) {
        actualSeqQlen = actualSeqQlenAddr[0];
        actualSeqKvlen = actualSeqKvlenAddr[0];
        return;
    }
    actualSeqQlen = actualSeqQlenAddr[boIdx] - actualSeqQlenAddr[boIdx - 1];
    actualSeqKvlen = actualSeqKvlenAddr[boIdx] - actualSeqKvlenAddr[boIdx - 1];
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::ComputeAxisIdx(
    int64_t multiCoreInnerIdx, RunParamStr &runParam)
{
    if constexpr (qLayout == LayOutTypeEnum::LAYOUT_TND) {
        GetSeqQlenKvlenByBoidx(runParam.boIdx, runParam.actualS1Size, runParam.actualS2Size);
        int64_t actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(runParam.actualS1Size, this->s1BaseSize) * constInfo.n2G);

        while (multiCoreInnerIdx >= actualS1Outersize) {
            this->s1OuterSizeAcc = actualS1Outersize;
            this->s1SizeAcc += runParam.actualS1Size;
            this->s2SizeAcc += runParam.actualS2Size;
            runParam.b1SSOffset += runParam.actualS1Size * runParam.actualS2Size;
            runParam.boIdx++;
            if (runParam.boIdx >= this->sharedParams.bSize) {
                break;
            }
            GetSeqQlenKvlenByBoidx(runParam.boIdx, runParam.actualS1Size, runParam.actualS2Size);
            actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(runParam.actualS1Size, this->s1BaseSize) * constInfo.n2G);
        }

        int64_t tmpS1Outersize = CeilDiv(runParam.actualS1Size, this->s1BaseSize);
        actualS1Outersize = multiCoreInnerIdx - this->s1OuterSizeAcc;
        runParam.n2oIdx = actualS1Outersize / tmpS1Outersize / this->sharedParams.gSize;
        runParam.goIdx = actualS1Outersize / tmpS1Outersize % this->sharedParams.gSize;
        runParam.s1oIdx = actualS1Outersize % tmpS1Outersize;
    } else {
        runParam.boIdx = multiCoreInnerIdx / constInfo.n2GS1o;
        runParam.n2oIdx = multiCoreInnerIdx % constInfo.n2GS1o / constInfo.gS1o;
        runParam.goIdx = multiCoreInnerIdx % constInfo.gS1o / constInfo.s1OuterSize;
        runParam.s1oIdx = multiCoreInnerIdx % constInfo.s1OuterSize;
        runParam.b1SSOffset = runParam.boIdx * constInfo.s1S2;
        runParam.actualS1Size = constInfo.s1Size;
        runParam.actualS2Size = constInfo.s2Size;
    }
    runParam.s1RealSize = Min(s1BaseSize, runParam.actualS1Size - runParam.s1oIdx * s1BaseSize);
    runParam.halfS1RealSize = (runParam.s1RealSize + 1) >> 1;
    runParam.firstHalfS1RealSize = runParam.halfS1RealSize;
    if (constInfo.subBlockIdx == 1) {
        runParam.halfS1RealSize = runParam.s1RealSize - runParam.halfS1RealSize;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    runInfo.s2StartIdx = runParam.s2LineStartIdx;
    runInfo.s2EndIdx = runParam.s2LineEndIdx;
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }
    if constexpr (qLayout == LayOutTypeEnum::LAYOUT_TND) {
        runInfo.boIdx = runParam.boIdx;
        runInfo.s1ScaleNumAcc = s1ScaleNumAcc;
        runInfo.s2ScaleNumAcc = s2ScaleNumAcc;
        runInfo.s1SizeAcc = s1SizeAcc;
        runInfo.s2SizeAcc = s2SizeAcc;
    } else {
        runInfo.s2SizeAcc = runInfo.boIdx * constInfo.s2Size;
    }
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.queryOffset = runParam.tensorQOffset;
    runInfo.qRopeOffset = runParam.qRopeNBGOffset;
    runInfo.sOuterOffset = runParam.sOuterOffset;
    this->ComputeBmm1Tail(runInfo, runParam);
    InitUniqueRunInfo(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::ComputeBmm1Tail(
    RunInfo &runInfo, RunParamStr &runParam)
{
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.s1RealSizeAlign32 = runParam.s1RealSizeAlign32;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;

    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize;

    runInfo.s2RealSize = s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    if (runInfo.s2StartIdx + (runInfo.s2LoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
        runInfo.s2RealSize = runInfo.s2EndIdx - runInfo.s2LoopCount * runInfo.s2RealSize - runInfo.s2StartIdx;
        runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    if constexpr (POST_QUANT) {
        this->constInfo.isPostQuantPerChnl = this->sharedParams.isPostQuantPerChnl;
        this->constInfo.isPostQuantBF16 = this->sharedParams.isPostQuantBF16;
    }
    this->constInfo.isRowInvalid = this->sharedParams.isRowInvalid;
    this->constInfo.headNumRatio = this->sharedParams.headNumRatio;
    this->constInfo.isGqa = this->sharedParams.isGqa;
    this->constInfo.isPfaGS1Merge = this->sharedParams.isPfaGS1Merge;
    this->constInfo.isKvContinuous = this->sharedParams.isKvContinuous;
    this->constInfo.actualSeqLenSize = this->sharedParams.actualSeqLengthsSize;
    this->constInfo.actualSeqLenKVSize = this->sharedParams.actualSeqLengthsKVSize;
    this->constInfo.isActualLenDimsNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsNull);
    this->constInfo.isActualLenDimsKVNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsKVNull);
    this->constInfo.isQHasLeftPadding = static_cast<bool>(this->sharedParams.isQHasLeftPadding);
    this->constInfo.isKVHasLeftPadding = static_cast<bool>(this->sharedParams.isKVHasLeftPadding);
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = this->sharedParams.blockTableDim2;
        this->constInfo.blockSize = this->sharedParams.blockSize;
        this->constInfo.paLayoutType = this->sharedParams.paLayoutType;
        this->constInfo.paBlockNumSum = this->sharedParams.paBlockNumSum;
        this->constInfo.paBlockStride = this->sharedParams.paBlockStride;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr &runParam, RunInfo &runInfo)
{
    InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS>(runParam, runInfo);
    ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS>(runParam, this->constInfo, runInfo.s2LoopCount + runInfo.s2StartIdx / this->constInfo.s2BaseSize, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    int32_t actualCoreNums = this->sharedParams.coreNum;
    if (this->aicIdx >= actualCoreNums) {
        return;
    }
    int64_t gS1StartIdx;
    uint32_t bnStartIdx;
    uint32_t bnEndIdx;
    int64_t s2LoopLimit;
    int64_t nextGs1Idx = this->sharedParams.multiCoreInnerLimit;
    bnStartIdx = this->sharedParams.bnStartIdx;
    gS1StartIdx = this->sharedParams.multiCoreInnerOffset;
    bnEndIdx = this->sharedParams.bnEndIdx;
    if (nextGs1Idx != 0) {
        bnEndIdx++;
    }
    int64_t taskId = 0;
    bool notLast = true;
    bool isLastBmm1 = false;
    LocalTensor<INPUT_T> scmTensor[2];
    RunInfo runInfo[4];
    RunParamStr runParam;

    int64_t multiCoreInnerIdx = 1;
    for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; ++bnIdx) {
        bool lastBN = (bnIdx == bnEndIdx - 1);
        runParam.boIdx = bnIdx / this->constInfo.n2Size;
        runParam.n2oIdx = bnIdx % this->constInfo.n2Size;
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS>(runParam, this->constInfo, this->attenMaskInfo,
            this->keyGm, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS>(runParam, this->constInfo, lastBN,
                                                                      nextGs1Idx);
        int64_t tempGS1End = lastBN ? (runParam.s1LoopTimes + 3) : runParam.s1LoopTimes;
        for (int64_t gS1Index = gS1StartIdx; gS1Index < tempGS1End; ++gS1Index) {
            bool notLastThreeLoop = true;
            bool notLastTwoLoop = true;
            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.s1LoopTimes;
                switch (extraGS1) {
                    case -1:
                        isLastBmm1 = true;
                        break;
                    case 0:
                        notLastThreeLoop = false;
                        break;
                    case 1:
                        notLastThreeLoop = false;
                        notLastTwoLoop = false;
                        break;
                    case 2:
                        notLast = false;
                        notLastThreeLoop = false;
                        notLastTwoLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastThreeLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS>(
                    runParam, this->constInfo, gS1Index, this->actualSeqQlenAddr);
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS>(runParam, this->constInfo);
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }

            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if (notLastThreeLoop) {
                    RunInfo &runInfo1 = runInfo[taskId & 3];
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo);
                    }
                }
                if (taskId > 0 && notLastTwoLoop) {
                    if ASCEND_IS_AIV {
                        auto &runInfo3 = runInfo[(taskId + 3) & 3];
                    }
                }
                if (taskId > 1 && notLast) {
                    if ASCEND_IS_AIC {
                        RunInfo &runInfo2 = runInfo[(taskId + 2) & 3];
                        if constexpr (bmm2Write2Ub) {
                            this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo2,
                                this->constInfo);
                        } else {
                            this->cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo2,
                                this->constInfo);
                        }
                    }
                }
                if (taskId > 2) {
                    if ASCEND_IS_AIV {
                        RunInfo &runInfo3 = runInfo[(taskId + 1) & 3];
                        if constexpr (bmm2Write2Ub) {
                        } else {
                        }
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::Process()
{
    if (this->sharedParams.needInit) {
        SyncAll<false>();
    }
    ProcessMainLoop();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam)
{
    constexpr uint64_t fp8QBlockSize = 128U;
    constexpr uint64_t fp8KvBlockSize = 256U;
    if (this->constInfo.isGqa) {
        runParam.goIdx = gS1Index / this->constInfo.s1OuterSize;
    } else {
        runParam.goIdx = bnIndex % this->constInfo.headNumRatio;
    }
    runParam.s1oIdx = gS1Index % this->constInfo.s1OuterSize;
}
}
#endif
