/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file flash_attention_score_antiquant_processor.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_ANTIQUANT_PROCESSOR_H
#define FLASH_ATTENTION_SCORE_ANTIQUANT_PROCESSOR_H

#include "../../../incre_flash_attention/regbase/opkernel/vector_api/vf_antiquant_w4.h"
#include "../../../incre_flash_attention/regbase/opkernel/vector_api/vf_antiquant_w8.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using matmul::MatmulType;
using namespace optiling;
namespace BaseApi {
struct AntiquantTaskParamBaseAPI {
    uint32_t batchSize;
    uint32_t seqSize;
    uint32_t kvHeadNum;
    uint32_t headDim;
    uint32_t headDimAlignBlock;
    uint64_t kvStep;

    uint64_t kvGmOffset;
    uint32_t copySplitS;
    uint32_t copyTotalS;

    uint32_t s2BatchOffset;
    bool isLoadAntiqParam;
    bool isFreeAntiqParam;
    bool isExistOffset;
    uint64_t antiqParamOffset;

    uint64_t bIdx;
    uint64_t n2Idx;
    uint64_t s2Idx;
    uint32_t singleSInnerSize;
    uint32_t flashDecodeS2Idx;
    uint32_t sInnerLoopSize;
    uint32_t antiqSeqSize;
    bool isPertensor;
    bool isPerHead;
    uint32_t kvCacheBlockSize;
    uint32_t maxBlockNumPerSeq;
    uint32_t paKvShapeType;
    uint64_t kvPaddingBeginOffset;
};
template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
class AntiquantProcessorBaseAPI
{
public:
    static constexpr bool PAGE_ATTENTION = isPa;
    static constexpr bool FLASH_DECODE = isFd;
    static constexpr bool KVFP4 = (IsSameType<KV_T, fp4x2_e1m2_t>::value || IsSameType<KV_T, fp4x2_e2m1_t>::value);
    static constexpr bool KVINT4 = IsSameType<KV_T, int4b_t>::value;
    static constexpr bool PAGE_ATTENTION_ANTIQUANT = (antiquantMode == AntiquantTypeEnum::PER_TOKEN_PAGE_ATTENTION ||
        antiquantMode == AntiquantTypeEnum::PER_TOKEN_HEAD_PAGE_ATTENTION);
    using ANTIQ_PARAMS_T = typename std::conditional<ANTIQUANT_PER_TOKEN, T, Q_T>::type;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;

    __aicore__ inline AntiquantProcessorBaseAPI(){};

    __aicore__ inline void ProcessBaseAPI(Buffer<BufferType::L1> &outBufAntiRes, GlobalTensor<KV_T> &kvGm,
                                   GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
                                   GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm, GlobalTensor<int32_t>& blockTableGm,
                                   TQue<QuePosition::VECIN, 1>& kvInputQue, TQue<QuePosition::VECOUT, 1>& kvOutputQue,
                                   TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                   TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue, TBuf<> kvAntiqMxScaleRes,
                                   const AntiquantTaskParamBaseAPI& taskParam, int32_t taskId, bool isBeforeHalf, int32_t s2RealSize);

    __aicore__ inline void CopyAntiqScaleE8M0(LocalTensor<Q_T> dstLocal, GlobalTensor<Q_T>& srcGm, uint64_t offset,
                                              uint32_t rowCnt, uint32_t grpNum);
    __aicore__ inline void LoadAntiquantParamsPerTokenGroup(GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
                                                            TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                                            TBuf<> kvAntiqMxScaleRes, const AntiquantTaskParamBaseAPI& taskParam,
                                                            bool isBeforeHalf, int32_t s2RealSize);
    __aicore__ inline void LoadAntiquantParamsPerToken(GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
                                                       GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm,
                                                       GlobalTensor<int32_t>& blockTableGm,
                                                       TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                                       TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
                                                       const AntiquantTaskParamBaseAPI& taskParam, bool isBeforeHalf, int32_t s2RealSize);

    __aicore__ inline void CopyAntiqParam(LocalTensor<ANTIQ_PARAMS_T> dstLocal, GlobalTensor<ANTIQ_PARAMS_T>& srcGm,
                                          uint32_t rowCnt, const AntiquantTaskParamBaseAPI& taskParam);
    __aicore__ inline void LoadAntiquantParamsPerChannel(GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
                                   GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm,
                                   TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                   TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
                                    const AntiquantTaskParamBaseAPI& taskParam);
    __aicore__ inline void FreeAntiquantParams(TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                   TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
                                    const AntiquantTaskParamBaseAPI& taskParam);
    __aicore__ inline void AntiquantBaseAPI(uint32_t copyLoopIdx, TQue<QuePosition::VECIN, 1>& kvInputQue,
                                     TQue<QuePosition::VECOUT, 1>& kvOutputQue, LocalTensor<Q_T> outScm,
                                     GlobalTensor<KV_T> &kvGm, GlobalTensor<int32_t>& blockTableGm,
                                     uint32_t dealRowCount, const AntiquantTaskParamBaseAPI& taskParam,
                                     int32_t taskId, bool isBeforeHalf, int32_t s2RealSize);
    __aicore__ inline void CopyKV(LocalTensor<KV_T> dstLocal, GlobalTensor<KV_T>& srcGm, uint64_t offset,
                                  uint32_t rowCnt, uint32_t headDim, uint32_t kvHeadNum,
                                  uint32_t paKvShapeType);
    __aicore__ inline void CopyKVPageAttention(LocalTensor<KV_T> dstLocal, GlobalTensor<KV_T> &srcGm,
                                               GlobalTensor<int32_t>& blockTableGm, const AntiquantTaskParamBaseAPI& taskParam,
                                               uint32_t curSequence, uint32_t dealRowCount);
    __aicore__ inline void CopyAntiquantParamsPageAttention(LocalTensor<ANTIQ_PARAMS_T> dstLocal, GlobalTensor<ANTIQ_PARAMS_T>& srcGm,
                                                            GlobalTensor<int32_t>& blockTableGm, const AntiquantTaskParamBaseAPI& taskParam,
                                                            DataCopyExtParams copyInParams,
                                                            DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams);
    __aicore__ inline void AntiquantVec(LocalTensor<Q_T>& antiqResUb, LocalTensor<KV_T>& antiqInUb, uint32_t copyLoopIdx,
                                        uint32_t dealRowCount, const AntiquantTaskParamBaseAPI& taskParam, bool isBeforeHalf);
    __aicore__ inline void CopyAntiquantResToL1BaseAPI(LocalTensor<Q_T>& antiqResScm, LocalTensor<Q_T>& antiqResUb,
                                                uint32_t copyLoopIdx, uint32_t dealRowCount,
                                                const AntiquantTaskParamBaseAPI& taskParam, int32_t taskId, bool isBeforeHalf, int32_t s2RealSize);

private:
    LocalTensor<ANTIQ_PARAMS_T> antiqScale;
    LocalTensor<ANTIQ_PARAMS_T> antiqOffset;
};

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::ProcessBaseAPI(
    Buffer<BufferType::L1> &outBufAntiRes, GlobalTensor<KV_T> &kvGm, GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
    GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm, GlobalTensor<int32_t>& blockTableGm,
    TQue<QuePosition::VECIN, 1>& kvInputQue, TQue<QuePosition::VECOUT, 1>& kvOutputQue,
    TQue<QuePosition::VECIN, 1>& antiqScaleInputQue, TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
    TBuf<> kvAntiqMxScaleRes, const AntiquantTaskParamBaseAPI& taskParam, int32_t taskId, bool isBeforeHalf, int32_t s2RealSize)
{
    if (taskParam.isLoadAntiqParam) {
        if constexpr (KVFP4) {
            LoadAntiquantParamsPerTokenGroup(antiqScaleGm, antiqScaleInputQue, kvAntiqMxScaleRes, taskParam, isBeforeHalf, s2RealSize);
        } else if constexpr (ANTIQUANT_PER_TOKEN) {
            LoadAntiquantParamsPerToken(antiqScaleGm, antiqOffsetGm, blockTableGm, antiqScaleInputQue,
                                        antiqOffsetInputQue, taskParam, isBeforeHalf, s2RealSize);
        } else {
            LoadAntiquantParamsPerChannel(antiqScaleGm, antiqOffsetGm, antiqScaleInputQue, antiqOffsetInputQue,
                                          taskParam);
        }
    }
    uint32_t loopCnt = (taskParam.copyTotalS + taskParam.copySplitS - 1) / taskParam.copySplitS;
    uint32_t tailCopyS = taskParam.copyTotalS - (loopCnt - 1) * taskParam.copySplitS;
    LocalTensor<Q_T> scmTensor = outBufAntiRes.GetTensor<Q_T>();
    CrossCoreWaitFlag<4, PIPE_MTE3>(CV_L1_EVENT[taskId % 2]);
    for (uint32_t i = 0, actCopyS = taskParam.copySplitS; i < loopCnt; i++) {
        if (i + 1 == loopCnt) {
            actCopyS = tailCopyS;
        }
        AntiquantBaseAPI(i, kvInputQue, kvOutputQue, scmTensor, kvGm, blockTableGm, actCopyS, taskParam, taskId, isBeforeHalf, s2RealSize);
    }
    CrossCoreSetFlag<4, PIPE_MTE3>(VC_L1_EVENT[taskId % 2]);
    if (taskParam.isFreeAntiqParam) {
        if constexpr (KVFP4) {
        } else {
            FreeAntiquantParams(antiqScaleInputQue, antiqOffsetInputQue, taskParam);
        }
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyAntiqScaleE8M0(
    LocalTensor<Q_T> dstLocal, GlobalTensor<Q_T>& srcGm, uint64_t offset, uint32_t rowCnt, uint32_t grpNum)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<Q_T> copyInPadParams{};
    copyInParams.blockCount = 1;
    copyInParams.dstStride = 0;
    copyInParams.blockLen = grpNum * rowCnt * sizeof(int8_t);
    copyInParams.srcStride = 0;
    DataCopyPad(dstLocal, srcGm[offset], copyInParams, copyInPadParams);
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::LoadAntiquantParamsPerTokenGroup(
    GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm, TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
    TBuf<> kvAntiqMxScaleRes, const AntiquantTaskParamBaseAPI& taskParam, bool isBeforeHalf, int32_t s2RealSize)
{
    int32_t subBlockIdx;
    if (isBeforeHalf) {
        subBlockIdx = 0;
    } else {
        subBlockIdx = 1;
    }
    uint32_t grpSize = 32;
    uint32_t grpNum = taskParam.headDim / grpSize;
    uint64_t scaleOffset = 0;
    scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.seqSize * grpNum +
                  taskParam.n2Idx * taskParam.seqSize * grpNum + taskParam.s2Idx * taskParam.singleSInnerSize * grpNum +
                  taskParam.kvPaddingBeginOffset * grpNum + subBlockIdx * (s2RealSize / 2) * grpNum;
    if constexpr (FLASH_DECODE) {
        scaleOffset += taskParam.flashDecodeS2Idx * taskParam.sInnerLoopSize * grpNum;
    }

    LocalTensor<Q_T> antiqScaleE8M0Ub = antiqScaleInputQue.template AllocTensor<Q_T>();
    CopyAntiqScaleE8M0(antiqScaleE8M0Ub, antiqScaleGm, scaleOffset / 2, taskParam.copyTotalS,
                       grpNum);
    antiqScaleInputQue.template EnQue(antiqScaleE8M0Ub);
    antiqScaleE8M0Ub = antiqScaleInputQue.DeQue<Q_T>();

    antiqScale = kvAntiqMxScaleRes.Get<ANTIQ_PARAMS_T>();
    AntiqScaleByVF<Q_T, ANTIQ_PARAMS_T>(antiqScaleE8M0Ub, antiqScale, taskParam.copyTotalS, grpNum);
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::LoadAntiquantParamsPerToken(
    GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm, GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm,
    GlobalTensor<int32_t>& blockTableGm, TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
    TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue, const AntiquantTaskParamBaseAPI& taskParam, bool isBeforeHalf, int32_t s2RealSize)
{
    int32_t subBlockIdx;
    if (isBeforeHalf) {
        subBlockIdx = 0;
    } else {
        subBlockIdx = 1;
    }
    uint64_t scaleOffset = 0;
    if (taskParam.isPerHead) {
        scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.seqSize + taskParam.n2Idx * taskParam.seqSize +
                      taskParam.s2Idx * taskParam.singleSInnerSize + taskParam.kvPaddingBeginOffset + subBlockIdx * (s2RealSize / 2);
    } else {
        scaleOffset = taskParam.bIdx * taskParam.antiqSeqSize + taskParam.s2Idx * taskParam.singleSInnerSize +
                      taskParam.kvPaddingBeginOffset + subBlockIdx * (s2RealSize / 2);
    }
    if constexpr (FLASH_DECODE) {
        scaleOffset += taskParam.flashDecodeS2Idx * taskParam.sInnerLoopSize;
    }

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams{};
    copyInParams.blockCount = 1;
    copyInParams.dstStride = 0;
    copyInParams.blockLen = taskParam.copyTotalS * sizeof(ANTIQ_PARAMS_T);
    copyInParams.srcStride = 0;
    LocalTensor<ANTIQ_PARAMS_T> tmpUb = antiqScaleInputQue.template AllocTensor<ANTIQ_PARAMS_T>();
    if constexpr (PAGE_ATTENTION_ANTIQUANT) {
        CopyAntiquantParamsPageAttention(tmpUb, antiqScaleGm, blockTableGm, taskParam, copyInParams, copyInPadParams);
    } else {
        DataCopyPad(tmpUb, antiqScaleGm[scaleOffset], copyInParams, copyInPadParams);
    }
    antiqScaleInputQue.template EnQue(tmpUb);
    antiqScale = antiqScaleInputQue.DeQue<ANTIQ_PARAMS_T>();

    if (taskParam.isExistOffset) {
        LocalTensor<ANTIQ_PARAMS_T> tmpUb = antiqOffsetInputQue.template AllocTensor<ANTIQ_PARAMS_T>();
        if constexpr (PAGE_ATTENTION_ANTIQUANT) {
            CopyAntiquantParamsPageAttention(tmpUb, antiqOffsetGm, blockTableGm, taskParam, copyInParams, copyInPadParams);
        } else {
            DataCopyPad(tmpUb, antiqOffsetGm[scaleOffset], copyInParams, copyInPadParams);
        }
        antiqOffsetInputQue.template EnQue(tmpUb);
        antiqOffset = antiqOffsetInputQue.DeQue<ANTIQ_PARAMS_T>();
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyAntiqParam(LocalTensor<ANTIQ_PARAMS_T> dstLocal, GlobalTensor<ANTIQ_PARAMS_T>& srcGm,
                                          uint32_t rowCnt, const AntiquantTaskParamBaseAPI& taskParam)
{
    if (taskParam.isPertensor) {
        Duplicate(dstLocal, srcGm.GetValue(0), rowCnt * dBaseSize);
    } else if (taskParam.isPerHead) {
        Duplicate(dstLocal, srcGm.GetValue(taskParam.n2Idx), rowCnt * dBaseSize);
    } else {
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams;
        copyInParams.blockCount = rowCnt;
        copyInParams.blockLen = taskParam.headDim * sizeof(ANTIQ_PARAMS_T);
        copyInParams.srcStride = static_cast<int64_t>(copyInParams.blockLen) * (-1);
        copyInParams.dstStride = (dBaseSize - taskParam.headDim) * sizeof(ANTIQ_PARAMS_T) / BYTE_BLOCK;

        copyInPadParams.isPad = false;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = 0;
        copyInPadParams.paddingValue = 0;
    
        DataCopyPad(dstLocal, srcGm[taskParam.antiqParamOffset], copyInParams, copyInPadParams);
    }
}
template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::LoadAntiquantParamsPerChannel(
    GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm, GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm,
    TQue<QuePosition::VECIN, 1>& antiqScaleInputQue, TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
    const AntiquantTaskParamBaseAPI& taskParam)
{
    uint32_t qElementCntPerReg = 256 / sizeof(ANTIQ_PARAMS_T);
    uint32_t loopCount = qElementCntPerReg / taskParam.headDimAlignBlock;
    if( loopCount == 0){
        loopCount =1;
    }
    LocalTensor<ANTIQ_PARAMS_T> tmpUb = antiqScaleInputQue.template AllocTensor<ANTIQ_PARAMS_T>();
    CopyAntiqParam(tmpUb, antiqScaleGm, loopCount, taskParam);

    antiqScaleInputQue.template EnQue(tmpUb);
    antiqScale = antiqScaleInputQue.DeQue<ANTIQ_PARAMS_T>();
    if (taskParam.isExistOffset) {
        LocalTensor<ANTIQ_PARAMS_T> tmpUb = antiqOffsetInputQue.template AllocTensor<ANTIQ_PARAMS_T>();
        CopyAntiqParam(tmpUb, antiqOffsetGm, loopCount, taskParam);
        antiqOffsetInputQue.template EnQue(tmpUb);
        antiqOffset = antiqOffsetInputQue.DeQue<ANTIQ_PARAMS_T>();
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::FreeAntiquantParams(TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                                                     TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
                                                                     const AntiquantTaskParamBaseAPI& taskParam)
{
    antiqScaleInputQue.FreeTensor(antiqScale);
    if (taskParam.isExistOffset) {
        antiqOffsetInputQue.FreeTensor(antiqOffset);
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::AntiquantBaseAPI(uint32_t copyLoopIdx,
                                                           TQue<QuePosition::VECIN, 1>& kvInputQue,
                                                           TQue<QuePosition::VECOUT, 1>& kvOutputQue,
                                                           LocalTensor<Q_T> outScm, GlobalTensor<KV_T> &kvGm,
                                                           GlobalTensor<int32_t>& blockTableGm, uint32_t dealRowCount,
                                                           const AntiquantTaskParamBaseAPI& taskParam, int32_t taskId, bool isBeforeHalf, int32_t s2RealSize)
{
    LocalTensor<KV_T> tmpUb = kvInputQue.template AllocTensor<KV_T>();
    if constexpr (!PAGE_ATTENTION) {
        uint64_t kvOffset = taskParam.kvGmOffset + copyLoopIdx * taskParam.copySplitS * taskParam.kvStep;
        CopyKV(tmpUb, kvGm, kvOffset, dealRowCount, taskParam.headDim, taskParam.kvHeadNum,
               taskParam.paKvShapeType);
    } else {
        uint64_t curSeqence = taskParam.s2BatchOffset + copyLoopIdx * taskParam.copySplitS;
        CopyKVPageAttention(tmpUb, kvGm, blockTableGm, taskParam, curSeqence, dealRowCount);
    }
    kvInputQue.template EnQue(tmpUb);
    kvInputQue.DeQue<KV_T>();

    LocalTensor<Q_T> outUb = kvOutputQue.template AllocTensor<Q_T>();
    AntiquantVec(outUb, tmpUb, copyLoopIdx, dealRowCount, taskParam, isBeforeHalf);
    kvInputQue.FreeTensor(tmpUb);

    kvOutputQue.template EnQue(outUb);
    kvOutputQue.DeQue<Q_T>();
    CopyAntiquantResToL1BaseAPI(outScm, outUb, copyLoopIdx, dealRowCount, taskParam, taskId, isBeforeHalf, s2RealSize);
    kvOutputQue.FreeTensor(outUb);
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyAntiquantParamsPageAttention(LocalTensor<ANTIQ_PARAMS_T> dstLocal,
                                                                      GlobalTensor<ANTIQ_PARAMS_T>& srcGm,
                                                                      GlobalTensor<int32_t>& blockTableGm,
                                                                      const AntiquantTaskParamBaseAPI& taskParam,
                                                                      DataCopyExtParams copyInParams,
                                                                      DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams)
{
    uint32_t useKvHeadNum = 1;
    uint32_t useN2Idx = 0;
    if (taskParam.isPerHead) {
        useKvHeadNum = taskParam.kvHeadNum;
        useN2Idx = taskParam.n2Idx;
    }
    uint64_t blockTableBaseOffset = taskParam.bIdx * taskParam.maxBlockNumPerSeq;
    uint64_t dstOffset = 0;
    uint32_t copyFinishElmenCnt = 0;
    uint32_t curSequence = taskParam.s2BatchOffset;
    while (copyFinishElmenCnt < taskParam.copyTotalS) {
        uint64_t blockIdOffset = curSequence / taskParam.kvCacheBlockSize;
        uint64_t remainElmenCnt = curSequence % taskParam.kvCacheBlockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyElmenCnt = taskParam.kvCacheBlockSize - remainElmenCnt;
        if (copyElmenCnt + copyFinishElmenCnt > taskParam.copyTotalS) {
            copyElmenCnt = taskParam.copyTotalS - copyFinishElmenCnt;
        }
        uint64_t srcOffset = idInBlockTable * taskParam.kvCacheBlockSize * useKvHeadNum +
                             taskParam.kvCacheBlockSize * useN2Idx + remainElmenCnt;
        copyInParams.blockLen = copyElmenCnt * sizeof(ANTIQ_PARAMS_T);
        DataCopyPad(dstLocal[dstOffset], srcGm[srcOffset], copyInParams, copyInPadParams);
        dstOffset += copyElmenCnt;
        copyFinishElmenCnt += copyElmenCnt;
        curSequence += copyElmenCnt;
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyKV(LocalTensor<KV_T> dstLocal,
                                                                             GlobalTensor<KV_T> &srcGm, uint64_t offset,
                                                                             uint32_t rowCnt, uint32_t headDim,
                                                                             uint32_t kvHeadNum, uint32_t paKvShapeType)
{
    uint32_t typeElementSize = ONE_BLK_SIZE / sizeof(KV_T);

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<KV_T> copyInPadParams;

    if constexpr (KVINT4 || KVFP4) {
        typeElementSize = ONE_BLK_SIZE * 2;
        copyInParams.blockCount = rowCnt;
        copyInParams.blockLen = headDim * sizeof(KV_T) / 2; 
    } else {
        copyInParams.blockCount = rowCnt;
        copyInParams.blockLen = headDim * sizeof(KV_T);
    }

    copyInParams.dstStride = (dBaseSize - headDim) / typeElementSize;

    if constexpr (PAGE_ATTENTION) {
        if (paKvShapeType == 0) {
            copyInParams.srcStride = copyInParams.blockLen * (kvHeadNum - 1);
        } else {
            copyInParams.srcStride = 0;
        }
    } else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            copyInParams.srcStride = 0;
        } else {
            copyInParams.srcStride = copyInParams.blockLen * (kvHeadNum - 1);
        }
    }

    copyInPadParams.isPad = false;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = 0;
    copyInPadParams.paddingValue = 0;

    DataCopyPad(dstLocal, srcGm[offset], copyInParams, copyInPadParams);
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyKVPageAttention(LocalTensor<KV_T> dstLocal,
                                                                     GlobalTensor<KV_T> &srcGm,
                                                                     GlobalTensor<int32_t>& blockTableGm,
                                                                     const AntiquantTaskParamBaseAPI& taskParam,
                                                                     uint32_t curSequence, uint32_t dealRowCount)
{
    uint64_t blockTableBaseOffset = taskParam.bIdx * taskParam.maxBlockNumPerSeq;
    uint32_t copyFinishRowCnt = 0;
    while (copyFinishRowCnt < dealRowCount) {
        uint64_t blockIdOffset = curSequence / taskParam.kvCacheBlockSize;
        uint64_t reaminRowCnt = curSequence % taskParam.kvCacheBlockSize;
        uint64_t idInBlockTable =
            blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyRowCnt = taskParam.kvCacheBlockSize - reaminRowCnt;
        if (copyFinishRowCnt + copyRowCnt > dealRowCount) {
            copyRowCnt = dealRowCount - copyFinishRowCnt;
        }
        uint64_t curOffset = 0;
        if (taskParam.paKvShapeType == 0) {
            curOffset =
                (idInBlockTable * taskParam.kvCacheBlockSize + reaminRowCnt) * taskParam.kvHeadNum * taskParam.headDim +
                (uint64_t)(taskParam.n2Idx * taskParam.headDim);
        } else {
            curOffset = idInBlockTable * taskParam.kvHeadNum * taskParam.kvCacheBlockSize * taskParam.headDim +
                        (uint64_t)(taskParam.n2Idx * taskParam.kvCacheBlockSize * taskParam.headDim) +
                        reaminRowCnt * taskParam.headDim;
        }

        CopyKV(dstLocal[copyFinishRowCnt * dBaseSize], srcGm, curOffset, copyRowCnt, taskParam.headDim,
               taskParam.kvHeadNum, taskParam.paKvShapeType);

        copyFinishRowCnt += copyRowCnt;
        curSequence += copyRowCnt;
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::AntiquantVec(LocalTensor<Q_T>& antiqResUb, LocalTensor<KV_T>& antiqInUb,
                                                              uint32_t copyLoopIdx, uint32_t dealRowCount,
                                                              const AntiquantTaskParamBaseAPI& taskParam, bool isBeforeHalf)
{
    if constexpr (KVFP4) {
        uint32_t grpNum = taskParam.headDim / 32;
        uint32_t perTokenScaleOffset = copyLoopIdx * taskParam.copySplitS * grpNum * 2;
        LocalTensor<ANTIQ_PARAMS_T> antiqScaleWithOffset = antiqScale[perTokenScaleOffset];
        AntiquantVF<Q_T, KV_T, ANTIQ_PARAMS_T, dBaseSize, false>(antiqInUb, antiqResUb, antiqOffset,
                                                                 antiqScaleWithOffset, dealRowCount, taskParam.headDim);
    } else if constexpr (ANTIQUANT_PER_TOKEN) {
        uint32_t perTokenScaleOffset = copyLoopIdx * taskParam.copySplitS;
        LocalTensor<ANTIQ_PARAMS_T> antiqScaleWithOffset = antiqScale[perTokenScaleOffset];
        if (taskParam.isExistOffset) {
            LocalTensor<ANTIQ_PARAMS_T> antiqOffsetWithOffset = antiqOffset[perTokenScaleOffset];
            AntiquantVF<Q_T, KV_T, ANTIQ_PARAMS_T, dBaseSize, true, true>(antiqInUb, antiqResUb, antiqOffsetWithOffset,
                                                                          antiqScaleWithOffset, dealRowCount, taskParam.headDim);
        } else {
            AntiquantVF<Q_T, KV_T, ANTIQ_PARAMS_T, dBaseSize, false, true>(antiqInUb, antiqResUb, antiqOffset,
                                                                           antiqScaleWithOffset, dealRowCount, taskParam.headDim);
        }
    } else {
        if (taskParam.isExistOffset) {
            AntiquantVF<Q_T, KV_T, ANTIQ_PARAMS_T, dBaseSize, true>(antiqInUb, antiqResUb, antiqOffset, antiqScale,
                                                                    dealRowCount, taskParam.headDim);
        } else {
            AntiquantVF<Q_T, KV_T, ANTIQ_PARAMS_T, dBaseSize, false>(antiqInUb, antiqResUb, antiqOffset, antiqScale,
                                                                     dealRowCount, taskParam.headDim);
        }
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyAntiquantResToL1BaseAPI(
        LocalTensor<Q_T>& antiqResScm, LocalTensor<Q_T>& antiqResUb, uint32_t copyLoopIdx,
        uint32_t dealRowCount, const AntiquantTaskParamBaseAPI& taskParam, int32_t taskId, bool isBeforeHalf, int32_t s2RealSize)
{
    uint16_t elementTypeSize = ONE_BLK_SIZE / sizeof(Q_T);
    uint16_t dstStep = ALIGN((uint16_t)s2RealSize, (uint16_t)16);
    int32_t subBlockIdx;
    if (isBeforeHalf) {
        subBlockIdx = 0;
    } else {
        subBlockIdx = 1;
    }
    uint64_t outOffset = subBlockIdx * (s2RealSize / 2) * 16 + copyLoopIdx * taskParam.copySplitS * 16;

    struct DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = taskParam.headDimAlignBlock / elementTypeSize;
    dataCopyParams.blockLen = dealRowCount;
    dataCopyParams.srcStride = 1;
    dataCopyParams.dstStride = dstStep - dealRowCount;

    DataCopy(antiqResScm[outOffset], antiqResUb, dataCopyParams);
}
}
#endif
