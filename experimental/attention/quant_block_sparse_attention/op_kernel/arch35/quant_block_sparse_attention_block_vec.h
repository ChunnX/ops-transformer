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
 * ile quant_block_sparse_attention_block_vec.h
 * \brief
 */
#ifndef QUANT_BLOCK_SPARSE_ATTENTION_BLOCK_VEC_H_
#define QUANT_BLOCK_SPARSE_ATTENTION_BLOCK_VEC_H_
#include "quant_block_sparse_attention_block_vec_base.h"
#include "common/util_regbase.h"
#include "common/flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "vf/vf_post_quant.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;


namespace BaseApi {
TEMPLATES_DEF
class BSABlockVec
    : public BSABlockVecBase<BSABlockVec<TEMPLATE_ARGS>, TEMPLATE_ARGS> {
public:
    using BaseClass = BSABlockVecBase<BSABlockVec<TEMPLATE_ARGS>, TEMPLATE_ARGS>;
public:
    /* ================编译期常量信息======================= */
    static constexpr uint32_t bufferSizeByte32K = 32768;
    static constexpr uint32_t gSplitMax = 16;
    static constexpr uint32_t preloadTimes = 3;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                  IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    /* =====================GM变量========================== */
    GlobalTensor<float> softmaxLseGm;

    using postQuantGmType = typename std::conditional<POST_QUANT, GlobalTensor<float>, int8_t>::type;
    postQuantGmType postQuantScaleGm;
    postQuantGmType postQuantOffsetGm;
    using postQuantBf16GmType = typename std::conditional<POST_QUANT, GlobalTensor<bfloat16_t>, int8_t>::type;
    postQuantBf16GmType postQuantScaleBf16Gm;
    postQuantBf16GmType postQuantOffsetBf16Gm;

    /* =====================UB变量========================== */
    TBuf<> lseTmpBuff;
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;
    TQue<QuePosition::VECIN, 1> postQuantScaleQue;; // postQuant
    TQue<QuePosition::VECIN, 1> postQuantOffsetQue;; // postQuant
    TQue<QuePosition::VECIN, 1> sinkQue; // AttentionSink

    __aicore__ inline BSABlockVec() {};
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx);
    __aicore__ inline void CleanOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitDropOut(__gm__ uint8_t *dropMask, __gm__ uint8_t *workspace) {}
    __aicore__ inline void InitGlobalBuffer(
        __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
        __gm__ uint8_t *pScale, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
        __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxMax,
        __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitUniqueLocalBuffer(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitPostQuant(ConstInfo<isInfer, hasRope> &constInfo, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset);
    __aicore__ inline void GenerateDropoutMask(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<uint8_t> &dropMaskUb) {}
    __aicore__ inline void SoftmaxDataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb,
                                              LocalTensor<float> &maxUb);
    __aicore__ inline void SoftmaxDataCopyOutFp8(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
                                                 LocalTensor<half> &sumUb, LocalTensor<half> &maxUb) {}
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                               int64_t vec2S1Idx, int64_t vec2CalcSize);

    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(ConstInfo<isInfer, hasRope> &constInfo, RunInfo<isInfer> &runInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t dSizeAligned64);

    template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
    __aicore__ inline void PostQuantPerChnl(ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut,
    LocalTensor<VEC2_RES_T> &vec2ResUb, uint64_t perChannelQuantOffset, uint32_t gSplitSize, uint32_t s1RowCount, uint32_t splitOffset, int64_t dSizeAligned64,
    GlobalTensor<POSTQUANT_PARAMS_T> postQuantScaleGm, GlobalTensor<POSTQUANT_PARAMS_T> postQuantOffsetGm);

private:
    __aicore__ inline void InitOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitLseOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void GetActualSeqLenKV(ConstInfo<isInfer, hasRope> &constInfo, GlobalTensor<INPUT_T> &keyGm, 
        __gm__ int64_t *actualSeqKvlenAddr, int64_t boIdx, int64_t &actualSeqKvLen);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void CopySinkIn(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void Vec1SinkCompute(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb);

    __aicore__ inline void Vec1SinkComputeGSFused(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb);

};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx)
{
    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
    sharedParams.bSize = inputParamsRegbase.bSize;
    sharedParams.t1Size = inputParamsRegbase.t1Size;
    sharedParams.t2Size = inputParamsRegbase.t2Size;
    sharedParams.n2Size = inputParamsRegbase.n2Size;
    sharedParams.gSize = inputParamsRegbase.gSize;
    sharedParams.s1Size = inputParamsRegbase.s1Size;
    sharedParams.s2Size = inputParamsRegbase.s2Size;
    sharedParams.dSize = inputParamsRegbase.dSize;
    sharedParams.dSizeV = inputParamsRegbase.dSizeV;
    if constexpr (hasRope) {
        sharedParams.dSizeRope = inputParamsRegbase.dSizeRope;
    } else {
        sharedParams.dSizeRope = 0;
    }
    sharedParams.preTokens = inputParamsRegbase.preTokens;
    sharedParams.nextTokens = inputParamsRegbase.nextTokens;
    sharedParams.s1SparseValidSize = inputParamsRegbase.s1SparseValidSize;
    sharedParams.s2SparseValidSize = inputParamsRegbase.s2SparseValidSize;
    sharedParams.bandIndex = inputParamsRegbase.bandIndex;
    sharedParams.implMode = inputParamsRegbase.implMode;
    sharedParams.layoutType = inputParamsRegbase.layoutType;
    sharedParams.sparseType = inputParamsRegbase.sparseType;
    sharedParams.compressMode = inputParamsRegbase.attenMaskCompressMode;
    sharedParams.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
    sharedParams.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
 
    if constexpr (POST_QUANT) {
        sharedParams.isPostQuantPerChnl = inputParamsRegbase.isPostQuantPerChnl;
        sharedParams.isPostQuantBF16 = inputParamsRegbase.isPostQuantBF16;
    }
    sharedParams.transposeLayout = inputParamsRegbase.transposeLayout;
    sharedParams.fromFused = inputParamsRegbase.fromFused;
    sharedParams.isRowInvalid = inputParamsRegbase.isRowInvalid;
    sharedParams.headNumRatio = inputParamsRegbase.headNumRatio;
    sharedParams.isGqa = inputParamsRegbase.isGqa;
    sharedParams.isPfaGS1Merge = (inputParamsRegbase.isGqa && sharedParams.s1Size > 1);
    sharedParams.isKvContinuous = inputParamsRegbase.isKvContinuous;
    sharedParams.actualSeqLengthsSize = inputParamsRegbase.actualSeqLengthsSize;
    sharedParams.actualSeqLengthsKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
    sharedParams.isActualSeqLengthsNull = inputParamsRegbase.isActualSeqLengthsNull;
    sharedParams.isActualSeqLengthsKVNull = inputParamsRegbase.isActualSeqLengthsKVNull;
    sharedParams.isQHasLeftPadding = inputParamsRegbase.isQHasLeftPadding;
    sharedParams.isKVHasLeftPadding = inputParamsRegbase.isKVHasLeftPadding;
    // pageAttention
    if constexpr (isPa) {
        sharedParams.blockTableDim2 = inputParamsRegbase.blockTableDim2;
        sharedParams.blockSize = inputParamsRegbase.blockSize;
        sharedParams.paLayoutType = inputParamsRegbase.paLayoutType;
        sharedParams.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
        sharedParams.paBlockStride = inputParamsRegbase.paBlockStride;
    }
    // prefix
    if constexpr (enableKVPrefix) {
        sharedParams.isActualSharedPrefixLenNull = inputParamsRegbase.isActualSharedPrefixLenNull;
        sharedParams.kvPrefixSize = inputParamsRegbase.prefixSeqInnerSize;
    }
    auto &multiCoreParamsRegbase = this->tilingData->multiCoreParamsRegbase;
    sharedParams.s1OuterSize = multiCoreParamsRegbase.s1OuterSize;
    sharedParams.coreNum = multiCoreParamsRegbase.coreNum;
    /* 多核切分偏移计算 */
    sharedParams.multiCoreInnerOffset = multiCoreParamsRegbase.sparseStartIdx[aicIdx];
    sharedParams.multiCoreInnerLimit = multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1];
    sharedParams.bnStartIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx];
    sharedParams.bnEndIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];
    sharedParams.needInit = this->tilingData->initOutputParams.needInit;

    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
            #pragma unroll
            for (int i = 0; i < sizeof(CVSharedParams<isInfer, isPa>) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }

}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *attentionOut, ConstInfo<isInfer, hasRope> &constInfo) 
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if constexpr (POST_QUANT) {
            this->attentionOutInitGm.SetGlobalBuffer((__gm__ half *)attentionOut);
        }
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        constInfo.isSoftmaxLseEnable = this->tilingData->inputParamsRegbase.isSoftMaxLseEnable;
        if (this->tilingData->initOutputParams.needInit == 1) {
            InitOutputSingleCore(constInfo);
            // lse output
            if (constInfo.isSoftmaxLseEnable) {
                SyncAll();
                InitLseOutputSingleCore(constInfo);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    BaseClass::InitCommonGlobalBuffer(
        deqScaleQ, deqScaleK, deqScaleV, pScale, postQuantScale, prefix, attenMask, blockTable,
        learnableSink, workspace, constInfo);
    if constexpr (POST_QUANT) {
        this->InitPostQuant(constInfo, postQuantScale, postQuantOffset);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::InitPostQuant(ConstInfo<isInfer, hasRope> &constInfo, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset)
{
    if constexpr (POST_QUANT) {
        constInfo.isPostQuantOffsetExist = false;
        if (!constInfo.isPostQuantPerChnl && !constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleGm.SetGlobalBuffer((__gm__ float *)postQuantScale);
                constInfo.postQuantScaleValue = postQuantScaleGm.GetValue(0);
            }
            if (postQuantOffset != nullptr) {
                postQuantOffsetGm.SetGlobalBuffer((__gm__ float *)postQuantOffset);
                constInfo.postQuantOffsetValue = postQuantOffsetGm.GetValue(0);
            } else {
                constInfo.postQuantOffsetValue = 0.0;
            }
        }
        
        if (!constInfo.isPostQuantPerChnl && constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantScale);
                constInfo.postQuantScaleValue = ToFloat(postQuantScaleBf16Gm.GetValue(0));
            }
            if (postQuantOffset != nullptr) {
                postQuantOffsetBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantOffset);
                constInfo.postQuantOffsetValue = ToFloat(postQuantOffsetBf16Gm.GetValue(0));
            } else {
                constInfo.postQuantOffsetValue = 0.0;
            }
        }

        if (constInfo.isPostQuantPerChnl && !constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                this->postQuantScaleGm.SetGlobalBuffer((__gm__ float *)postQuantScale);
            }
            if (postQuantOffset != nullptr) {
                constInfo.isPostQuantOffsetExist = true;
                postQuantOffsetGm.SetGlobalBuffer((__gm__ float *)postQuantOffset);
            }
        }

        if (constInfo.isPostQuantPerChnl && constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantScale);
            }
            if (postQuantOffset != nullptr) {
                constInfo.isPostQuantOffsetExist = true;
                postQuantOffsetBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantOffset);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::InitUniqueLocalBuffer(ConstInfo<isInfer, hasRope> &constInfo)
{
    if (constInfo.isSoftmaxLseEnable) {
        // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
        this->tPipe->InitBuffer(softmaxLseQueue, 1, (BaseClass::s1BaseSize >> 1U) * sizeof(float) * 8);
    }
    if constexpr (POST_QUANT) {
        this->tPipe->InitBuffer(postQuantScaleQue, 1, 2048); // 2K
        if (constInfo.isPostQuantOffsetExist) {
            this->tPipe->InitBuffer(postQuantOffsetQue, 1, 2048); // 2K
        }
    }
    if (constInfo.learnableSinkFlag) {
        this->tPipe->InitBuffer(sinkQue, 1, 256); // buffer size = 256 bytes
    }
}



TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::SoftmaxDataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb,
    LocalTensor<float> &maxUb)
{
    if (constInfo.learnableSinkFlag) {
        if (constInfo.isGqa) {
            this->Vec1SinkComputeGSFused(runInfo, constInfo, sumUb, maxUb);
        } else {
            this->Vec1SinkCompute(runInfo, constInfo, sumUb, maxUb);
        }
    }
    SoftmaxLseCopyOut(sumUb, maxUb, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::Vec1SinkCompute(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb) 
{
    int64_t sinkOffset = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx;
    auto sinkRaw = this->sinkGm.GetValue(sinkOffset);
    float sinkValue;
    if constexpr (IsSameType<decltype(sinkRaw), half>::value) {
        sinkValue = static_cast<float>(sinkRaw);
    } else {
        sinkValue = ToFloat(sinkRaw);
    }
    SinkSubExpAddVF<float>(sumUb, maxUb, sinkValue, runInfo.halfS1RealSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::Vec1SinkComputeGSFused(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb) 
{
    CopySinkIn(runInfo, constInfo);
    LocalTensor<INPUT_T> sinkUb = sinkQue.DeQue<INPUT_T>();
    SinkSubExpAddGSFusedVF<float, INPUT_T>(sinkUb, sumUb, maxUb, runInfo.halfS1RealSize);
    sinkQue.FreeTensor(sinkUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::CopySinkIn(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
{
    LocalTensor<INPUT_T> sinkUbBf16 = sinkQue.AllocTensor<INPUT_T>();
    int64_t sinkOffset = runInfo.n2oIdx * constInfo.gSize + constInfo.subBlockIdx * runInfo.halfS1RealSize;
    DataCopyExtParams sinkCopyParams;
    sinkCopyParams.blockCount = 1; // 进行一次连续拷贝
    sinkCopyParams.blockLen = runInfo.halfS1RealSize * sizeof(INPUT_T); // 实际需要拷贝的字节数
    sinkCopyParams.srcStride = 0; // 源地址连续
    sinkCopyParams.dstStride = 0; // 目的地址连续

    DataCopyPadExtParams<INPUT_T> sinkCopyPadParams{};
    DataCopyPad(sinkUbBf16, this->sinkGm[sinkOffset], sinkCopyParams, sinkCopyPadParams);
    sinkQue.EnQue(sinkUbBf16);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::GetActualSeqLenKV(ConstInfo<isInfer, hasRope> &constInfo, 
    GlobalTensor<INPUT_T> &keyGm, __gm__ int64_t *actualSeqKvlenAddr, int64_t boIdx, int64_t &actualSeqLen)
{
    int64_t s2InCurrentBatch = constInfo.s2Size;
    if (constInfo.isKvContinuous == 0) {
        ListTensorDesc keyListTensorDesc((__gm__ void *)keyGm.GetPhyAddr());
        AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;
        uint64_t dimInfo[4];
        kvTensorDesc.SetShapeAddr(&dimInfo[0]);
        keyListTensorDesc.GetDesc(kvTensorDesc, boIdx);
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            s2InCurrentBatch = kvTensorDesc.GetShape(2);
        } else {
            s2InCurrentBatch = kvTensorDesc.GetShape(1);
        }
    }
    if (constInfo.isActualLenDimsKVNull) {
        actualSeqLen = s2InCurrentBatch;
    } else {
        actualSeqLen = (constInfo.actualSeqLenKVSize == 1) ? actualSeqKvlenAddr[0] : actualSeqKvlenAddr[boIdx];
        if constexpr ((layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) && (!isPa)) {
            if (boIdx > 0) {
                actualSeqLen -= actualSeqKvlenAddr[boIdx - 1];
            }
        }
    }
    if (constInfo.isKVHasLeftPadding) {
        int64_t kvLeftPaddingSize = constInfo.s2Size - actualSeqLen - constInfo.kvRightPaddingSize;
        if (kvLeftPaddingSize < 0) {
            actualSeqLen = 0;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::SoftmaxLseCopyOut(
    LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }

    if (!constInfo.isSoftmaxLseEnable) {
        return;
    }
    LocalTensor<float> lseUb = this->softmaxLseQueue.template AllocTensor<float>();
    ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.halfS1RealSize);
    softmaxLseQueue.template EnQue(lseUb);
    softmaxLseQueue.DeQue<float>();
    DataCopyExtParams intriParams1;
    intriParams1.blockLen = sizeof(float);
    intriParams1.blockCount = runInfo.halfS1RealSize;
    intriParams1.srcStride = 0;
    if (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) {
        intriParams1.dstStride = constInfo.isGqa ? 0 : sizeof(float) * (constInfo.n2G - 1);
    } else {
        intriParams1.dstStride = 0;
    }
    DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], lseUb, intriParams1);
    softmaxLseQueue.FreeTensor(lseUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo)
{
    auto &initParams = this->tilingData->initOutputParams;
    uint32_t tailSize = (initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize) > 0 ?
        (initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize) : 0;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    if constexpr (POST_QUANT) {
        InitOutput<half>(this->attentionOutInitGm[constInfo.aivIdx * initParams.singleCoreSize / 2], singleInitOutputSize / 2, 0.0);
    } else {
        InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * initParams.singleCoreSize], singleInitOutputSize, 0.0);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::InitLseOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo)
{
    int64_t coreNum = GetBlockNum() * GetTaskRation();
    auto &initParams = this->tilingData->initOutputParams;
    if (coreNum != 0 && constInfo.aivIdx < coreNum) {
        int64_t singleCoreLseSize = initParams.totalSoftMaxLseOutputSize / coreNum;
        if (constInfo.aivIdx == coreNum - 1) {
            singleCoreLseSize += initParams.totalSoftMaxLseOutputSize % coreNum;
        }
        InitOutput<float>(softmaxLseGm[constInfo.aivIdx * (initParams.totalSoftMaxLseOutputSize / coreNum)], 
            singleCoreLseSize, 3e+99); // 3e+99:set the value of invalid batch to inf
    }
}











TEMPLATES_DEF_NO_DEFAULT
template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::PostQuantPerChnl(
    ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb,
    uint64_t perChannelQuantOffset, uint32_t gSplitSize, uint32_t s1RowCount, uint32_t splitOffset, int64_t dSizeAligned64,
    GlobalTensor<POSTQUANT_PARAMS_T> postQuantScaleGm, GlobalTensor<POSTQUANT_PARAMS_T> postQuantOffsetGm)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<POSTQUANT_PARAMS_T> copyInPadParams;
    copyInParams.blockCount = gSplitSize;
    copyInParams.blockLen = constInfo.dSizeV * sizeof(POSTQUANT_PARAMS_T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (dSizeAligned64 - constInfo.dSizeV) / (32 / sizeof(POSTQUANT_PARAMS_T));  // 32: datablock size

    LocalTensor<POSTQUANT_PARAMS_T> postQuantScaleUb =
        this->postQuantScaleQue.template AllocTensor<POSTQUANT_PARAMS_T>();
    DataCopyPad(postQuantScaleUb, postQuantScaleGm[perChannelQuantOffset], copyInParams, copyInPadParams);
    this->postQuantScaleQue.template EnQue(postQuantScaleUb);
    this->postQuantScaleQue.template DeQue<POSTQUANT_PARAMS_T>();
    if (constInfo.isPostQuantOffsetExist) {
        LocalTensor<POSTQUANT_PARAMS_T> postQuantOffsetUb =
            this->postQuantOffsetQue.template AllocTensor<POSTQUANT_PARAMS_T>();
        DataCopyPad(postQuantOffsetUb, postQuantOffsetGm[perChannelQuantOffset], copyInParams, copyInPadParams);
        this->postQuantOffsetQue.template EnQue(postQuantOffsetUb);
        this->postQuantOffsetQue.template DeQue<POSTQUANT_PARAMS_T>();
        PostQuantPerChnlImpl<T, OUTPUT_T, POSTQUANT_PARAMS_T>(
            attenOut[splitOffset], vec2ResUb[splitOffset], postQuantScaleUb, postQuantOffsetUb, gSplitSize, s1RowCount,
            constInfo.dSizeV, dSizeAligned64);
        this->postQuantOffsetQue.FreeTensor(postQuantOffsetUb);
    } else {
        PostQuantPerChnlImpl<T, OUTPUT_T, POSTQUANT_PARAMS_T>(
            attenOut[splitOffset], vec2ResUb[splitOffset], postQuantScaleUb, gSplitSize, s1RowCount, constInfo.dSizeV, dSizeAligned64);
    }
    this->postQuantScaleQue.FreeTensor(postQuantScaleUb);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::PostQuant(ConstInfo<isInfer, hasRope> &constInfo,
                                                                      RunInfo<isInfer> &runInfo, LocalTensor<OUTPUT_T> &attenOut,
                                                                      LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                      int64_t vec2S1Idx, int64_t dSizeAligned64)
{
    uint32_t s1RowCount = constInfo.isGqa ? 1U : runInfo.vec2S1RealSize; // s1=1, gS合轴, bn2分核
    uint32_t gRowCount = constInfo.isGqa ? runInfo.vec2S1RealSize : 1U;  // s1>1, bn1分核
    if (constInfo.isPostQuantPerChnl) {
        uint64_t perChannelQuantGQAOffset = runInfo.n2oIdx * constInfo.gDv + vec2S1Idx * runInfo.vec2S1BaseSize * constInfo.dSizeV +
                                            runInfo.sOuterOffset * constInfo.dSizeV;
        uint64_t perChannelQuantOffset = constInfo.isGqa ?
                                                perChannelQuantGQAOffset :
                                                runInfo.n2oIdx * constInfo.gDv + runInfo.goIdx * constInfo.dSizeV;
        uint32_t gSplitSize = constInfo.isPostQuantBF16 ? (2048U / ((uint32_t)dSizeAligned64 * sizeof(bfloat16_t))) :
                                                            (2048U / ((uint32_t)dSizeAligned64 * sizeof(float)));
        gSplitSize = gSplitSize > gRowCount ? gRowCount : gSplitSize;
        uint32_t loopCount = (gRowCount + gSplitSize - 1) / gSplitSize;
        uint32_t tailSplitSize = gRowCount - (loopCount - 1) * gSplitSize;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t startRow = i * gSplitSize;
            if (i + 1 == loopCount) {
                gSplitSize = tailSplitSize;
            }
            uint32_t splitOffset = startRow * dSizeAligned64;
            if (constInfo.isPostQuantBF16) {
                PostQuantPerChnl(constInfo, attenOut, vec2ResUb, perChannelQuantOffset + startRow * constInfo.dSizeV,
                                    gSplitSize, s1RowCount, splitOffset, dSizeAligned64, postQuantScaleBf16Gm, postQuantOffsetBf16Gm);
            } else {
                PostQuantPerChnl(constInfo, attenOut, vec2ResUb, perChannelQuantOffset + startRow * constInfo.dSizeV,
                                    gSplitSize, s1RowCount, splitOffset, dSizeAligned64, postQuantScaleGm, postQuantOffsetGm);
            }
        }
    } else {
        PostQuantPerTensorImpl<T, OUTPUT_T, true>(attenOut, vec2ResUb, constInfo.postQuantScaleValue, constInfo.postQuantOffsetValue, 
                                                 runInfo.vec2S1RealSize, constInfo.dSizeV, dSizeAligned64);
    }
}

}
#endif // QUANT_BLOCK_SPARSE_ATTENTION_BLOCK_VEC_H_
