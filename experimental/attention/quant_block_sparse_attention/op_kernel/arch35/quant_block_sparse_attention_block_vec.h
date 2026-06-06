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
#include "common/quant_block_sparse_attention_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
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
    static constexpr bool POST_QUANT = false;
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                  IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    /* =====================GM变量========================== */
    GlobalTensor<float> softmaxLseGm;


    /* =====================UB变量========================== */
    TBuf<> lseTmpBuff;
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;

    __aicore__ inline BSABlockVec() {};
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx);
    __aicore__ inline void CleanOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
        ConstInfo<isInfer> &constInfo);
    __aicore__ inline void InitDropOut(__gm__ uint8_t *dropMask, __gm__ uint8_t *workspace) {}
    __aicore__ inline void InitGlobalBuffer(
        __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
        __gm__ uint8_t *pScale,
        __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *softmaxMax,
        __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
        ConstInfo<isInfer> &constInfo);
    __aicore__ inline void InitUniqueLocalBuffer(ConstInfo<isInfer> &constInfo);
    __aicore__ inline void GenerateDropoutMask(RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo, LocalTensor<uint8_t> &dropMaskUb) {}
    __aicore__ inline void SoftmaxDataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo, LocalTensor<float> &sumUb,
                                              LocalTensor<float> &maxUb);
    __aicore__ inline void SoftmaxDataCopyOutFp8(RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo,
                                                 LocalTensor<half> &sumUb, LocalTensor<half> &maxUb) {}
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                               int64_t vec2S1Idx, int64_t vec2CalcSize);

private:
    __aicore__ inline void InitOutputSingleCore(ConstInfo<isInfer> &constInfo);
    __aicore__ inline void InitLseOutputSingleCore(ConstInfo<isInfer> &constInfo);
    __aicore__ inline void GetActualSeqLenKV(ConstInfo<isInfer> &constInfo, GlobalTensor<INPUT_T> &keyGm, 
        __gm__ int64_t *actualSeqKvlenAddr, int64_t boIdx, int64_t &actualSeqKvLen);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo);

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
    sharedParams.preTokens = inputParamsRegbase.preTokens;
    sharedParams.nextTokens = inputParamsRegbase.nextTokens;
    sharedParams.s1SparseValidSize = inputParamsRegbase.s1SparseValidSize;
    sharedParams.s2SparseValidSize = inputParamsRegbase.s2SparseValidSize;
    sharedParams.bandIndex = inputParamsRegbase.bandIndex;
    sharedParams.layoutType = inputParamsRegbase.layoutType;
    sharedParams.sparseType = inputParamsRegbase.sparseType;
    sharedParams.compressMode = inputParamsRegbase.attenMaskCompressMode;
    sharedParams.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
    sharedParams.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
 
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
    __gm__ uint8_t *attentionOut, ConstInfo<isInfer> &constInfo) 
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
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
    __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
    ConstInfo<isInfer> &constInfo)
{
    BaseClass::InitCommonGlobalBuffer(
        deqScaleQ, deqScaleK, deqScaleV, pScale, prefix, attenMask, blockTable,
        workspace, constInfo);
}


TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::InitUniqueLocalBuffer(ConstInfo<isInfer> &constInfo)
{
    if (constInfo.isSoftmaxLseEnable) {
        // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
        this->tPipe->InitBuffer(softmaxLseQueue, 1, (BaseClass::s1BaseSize >> 1U) * sizeof(float) * 8);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::SoftmaxDataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo, LocalTensor<float> &sumUb,
    LocalTensor<float> &maxUb)
{
    SoftmaxLseCopyOut(sumUb, maxUb, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::GetActualSeqLenKV(ConstInfo<isInfer> &constInfo, 
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
    LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp, RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo)
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
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo<isInfer> &constInfo)
{
    auto &initParams = this->tilingData->initOutputParams;
    uint32_t tailSize = (initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize) > 0 ?
        (initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize) : 0;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * initParams.singleCoreSize], singleInitOutputSize, 0.0);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void BSABlockVec<TEMPLATE_ARGS>::InitLseOutputSingleCore(ConstInfo<isInfer> &constInfo)
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

}
#endif // QUANT_BLOCK_SPARSE_ATTENTION_BLOCK_VEC_H_
