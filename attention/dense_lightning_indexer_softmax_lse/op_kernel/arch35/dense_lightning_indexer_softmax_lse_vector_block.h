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
 * \file dense_lightning_indexer_softmax_lse_vector_block.h
 * \brief Arch35 vector block implementation
 */

#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_VECTOR_BLOCK_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_VECTOR_BLOCK_H

#include "dense_lightning_indexer_softmax_lse_regbase_common.h"
#include "vf/vf_process_vec0.h"
#include "vf/vf_process_vec1.h"

namespace DenseLISoftmaxLseKernelRegbase {

TEMPLATES_DEF
class DenseLISoftmaxLseBlockVec {
public:
    __aicore__ inline DenseLISoftmaxLseBlockVec(){};
    __aicore__ inline void InitParams(const DenseLISoftmaxLseConstInfo &vecConstInfo,
                                      const optiling::DenseLightningIndexerSoftmaxLseRegBaseTilingData *tiling);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR weights, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                            GlobalTensor<T> &mm1Res, GlobalTensor<float> &vec1Res);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &mm1ResBuf,
                                       DenseLISoftmaxLseRunInfo &runInfo);
    __aicore__ inline void ProcessSoftmax(DenseLISoftmaxLseRunInfo &runInfo);

    GlobalTensor<INPUT_T> weightsGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;

    const optiling::DenseLightningIndexerSoftmaxLseRegBaseTilingData *tilingData;
    TPipe *pipe;

private:
    static constexpr int64_t UB_ROW_SIZE = 8;
    static constexpr int64_t GROUP_INNER_SIZE = 16;
    static constexpr event_t EVENT_ID0 = (event_t)4;
    static constexpr event_t EVENT_ID2 = (event_t)6;

    static constexpr float SOFTMAX_MIN_NUM = -2e38;

    DenseLISoftmaxLseConstInfo constInfo;
    TBuf<> scaleTbuf;
    TQue<QuePosition::VECIN, 1> reduceCacheQue;
    TQue<QuePosition::VECOUT, 1> reduceOutQue;
    TQue<QuePosition::VECIN, 1> weightInQue;
    TQue<QuePosition::VECIN, 1> reduceMaxSrcQue;
    TQue<QuePosition::VECIN, 1> reduceSumSrcQue;

    LocalTensor<INPUT_T> weightsUb;
    LocalTensor<T> mm1ResUb;
    float maxValue_;

    __aicore__ inline void CopyInWeights(DenseLISoftmaxLseRunInfo &runInfo);
    __aicore__ inline void ProcessReduceMax(DenseLISoftmaxLseRunInfo &runInfo, uint32_t pingPongFlag);
    __aicore__ inline void ProcessReduceSum(DenseLISoftmaxLseRunInfo &runInfo, uint32_t pingPongFlag);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockVec<TEMPLATE_ARGS>::InitParams(
    const DenseLISoftmaxLseConstInfo &vecConstInfo,
    const optiling::DenseLightningIndexerSoftmaxLseRegBaseTilingData *tiling)
{
    this->constInfo = vecConstInfo;
    tilingData = tiling;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR weights, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
    GlobalTensor<T> &mm1Res, GlobalTensor<float> &vec1Res)
{
    weightsGm.SetGlobalBuffer((__gm__ INPUT_T *)weights);
    softmaxMaxGm.SetGlobalBuffer((__gm__ T *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ T *)softmaxSum);
    this->mm1ResGm = mm1Res;
    this->vec1ResGm = vec1Res;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockVec<TEMPLATE_ARGS>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(this->scaleTbuf, DenseLISoftmaxLseConstInfo::BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(this->reduceCacheQue, 1, DenseLISoftmaxLseConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->reduceOutQue, 1, DenseLISoftmaxLseConstInfo::BUFFER_SIZE_BYTE_2K);
    pipe->InitBuffer(this->weightInQue, 1, DenseLISoftmaxLseConstInfo::BUFFER_SIZE_BYTE_128);
    pipe->InitBuffer(this->reduceMaxSrcQue, 1, DenseLISoftmaxLseConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->reduceSumSrcQue, 1, DenseLISoftmaxLseConstInfo::BUFFER_SIZE_BYTE_256);

    weightsUb = scaleTbuf.Get<INPUT_T>();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockVec<TEMPLATE_ARGS>::CopyInWeights(
    DenseLISoftmaxLseRunInfo &runInfo)
{
    LocalTensor<INPUT_T> weightInUb = weightInQue.template AllocTensor<INPUT_T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = GROUP_INNER_SIZE * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<INPUT_T> padParams;
    DataCopyPad(weightInUb, weightsGm[runInfo.weightsTensorOffset], dataCopyParams, padParams);
    this->weightInQue.template EnQue(weightInUb);
    this->weightInQue.template DeQue<INPUT_T>();
    this->weightInQue.template FreeTensor(weightInUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockVec<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &mm1ResBuf,
    DenseLISoftmaxLseRunInfo &runInfo)
{
    LocalTensor<T> mmRes = mm1ResBuf.template GetTensor<T>();
    auto reduceCacheTensor = this->reduceCacheQue.template AllocTensor<T>();
    auto reduceOutTensor = this->reduceOutQue.template AllocTensor<T>();
    LocalTensor<INPUT_T> weightInUb = weightInQue.template AllocTensor<INPUT_T>();

    uint32_t mSize = runInfo.actMBaseSize;
    uint32_t nSize = runInfo.actualSingleProcessSInnerSizeAlign;

    uint32_t outerG = DenseLISoftmaxLseCeilDiv<uint32_t>(constInfo.gSize, GROUP_INNER_SIZE);

    for (uint32_t outerGidx = 0; outerGidx < outerG; outerGidx++) {
        uint32_t procGnum = (outerGidx != outerG - 1) ? GROUP_INNER_SIZE : 
                           (constInfo.gSize - outerGidx * GROUP_INNER_SIZE);

        uint64_t mmGmOffset = outerGidx * GROUP_INNER_SIZE * nSize;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = procGnum;
        dataCopyParams.blockLen = nSize * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams;
        DataCopyPad(mmRes[mmGmOffset], mm1ResGm[runInfo.loop % MODE_NUM_2 * constInfo.mBaseSize * constInfo.s2BaseSize + 
                   outerGidx * GROUP_INNER_SIZE * nSize], dataCopyParams, padParams);

        if constexpr (!IsSameType<INPUT_T, float>::value) {
            LocalTensor<float> weightsFloatUb = weightInUb.template ReinterpretCast<float>();
            AscendC::Cast(weightsFloatUb, weightInUb[outerGidx * GROUP_INNER_SIZE], 
                         RoundMode::CAST_NONE, procGnum);
            AscendC::PipeBarrier<PIPE_V>();
        }

        ProcessVec1Vf<T, INPUT_T>(reduceCacheTensor, mmRes[mmGmOffset], weightInUb, 
                                  procGnum, nSize, outerGidx);
    }

    DenseLISoftmaxLseKRunInfo kRunInfo;
    kRunInfo.s2RealBaseSize = nSize;
    ProcessVec2Vf<T>(reduceOutTensor, reduceCacheTensor, mSize, nSize);

    this->reduceOutQue.template EnQue(reduceOutTensor);
    this->reduceOutQue.template DeQue<T>();
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = runInfo.actualSingleProcessSInnerSize * sizeof(float);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    DataCopyPad(vec1ResGm[runInfo.reduceSumGmOutOffset], reduceOutTensor, copyOutParams);

    this->weightInQue.template FreeTensor(weightInUb);
    this->reduceCacheQue.template FreeTensor(reduceCacheTensor);
    this->reduceOutQue.template FreeTensor(reduceOutTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockVec<TEMPLATE_ARGS>::ProcessSoftmax(
    DenseLISoftmaxLseRunInfo &runInfo)
{
    uint32_t actS1Size = runInfo.actS1Size;
    uint32_t actS2Size = runInfo.actS2Size;
    uint32_t s1Idx = runInfo.s1Idx;

    uint32_t maskedActS2Size = (actS2Size - actS1Size + s1Idx + 1) > actS2Size ?
                               actS2Size : (actS2Size - actS1Size + s1Idx + 1);

    uint64_t reduceOutGmOffset = runInfo.reduceMaxOutGmOffset + s1Idx;
    uint64_t reduceSrcGmOffset = runInfo.s1BlockInnerLoop * DenseLISoftmaxLseAlign<uint64_t>(
        actS2Size > 128 * 1024 ? 128 * 1024 : actS2Size, static_cast<uint64_t>(constInfo.s2BaseSize));

    uint32_t pingPongFlag = runInfo.loop % MODE_NUM_2;

    auto reduceMaxSrc = reduceMaxSrcQue.template AllocTensor<T>();
    Duplicate(reduceMaxSrc, SOFTMAX_MIN_NUM, VEC_REDUCE_BASESIZE);
    AscendC::PipeBarrier<PIPE_V>();
    reduceMaxSrcQue.template FreeTensor(reduceMaxSrc);

    ProcessReduceMax(runInfo, pingPongFlag);
    ProcessReduceSum(runInfo, pingPongFlag);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockVec<TEMPLATE_ARGS>::ProcessReduceMax(
    DenseLISoftmaxLseRunInfo &runInfo, uint32_t pingPongFlag)
{
    uint32_t actS2Size = runInfo.actS2Size;
    uint32_t actS1Size = runInfo.actS1Size;
    uint32_t s1Idx = runInfo.s1Idx;
    uint32_t maskedActS2Size = (actS2Size - actS1Size + s1Idx + 1) > actS2Size ?
                               actS2Size : (actS2Size - actS1Size + s1Idx + 1);

    uint32_t blockNum = DenseLISoftmaxLseCeilDiv<uint32_t>(maskedActS2Size, VEC_REDUCE_BASESIZE);

    LocalTensor<T> reduceMaxSrc0 = reduceMaxSrcQue.template AllocTensor<T>();
    LocalTensor<T> reduceMaxSrc1 = reduceMaxSrc0[VEC_REDUCE_BASESIZE];
    LocalTensor<T> reduceMaxDstTensor = reduceMaxSrc0[3 * VEC_REDUCE_BASESIZE];
    LocalTensor<T> sharedTmpTensor = reduceMaxSrc0[2 * VEC_REDUCE_BASESIZE];

    SetFlag<HardEvent::V_S>(EVENT_ID0);
    for (uint32_t blockIdx = 0; blockIdx <= blockNum - 1; blockIdx++) {
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        uint32_t curBlockSize = (blockIdx != blockNum - 1) ? VEC_REDUCE_BASESIZE :
            (maskedActS2Size - VEC_REDUCE_BASESIZE * blockIdx);

        if (blockIdx > 0) {
            uint64_t mask = 64;
            uint8_t repeatTime = 1;
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Copy(reduceMaxSrc1, reduceMaxDstTensor, mask, repeatTime, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            Duplicate(reduceMaxSrc1, SOFTMAX_MIN_NUM, VEC_REDUCE_BASESIZE);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(curBlockSize * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{true, 0,
            static_cast<uint8_t>(DenseLISoftmaxLseAlign<uint32_t>(curBlockSize, 8) - curBlockSize),
            static_cast<float>(SOFTMAX_MIN_NUM)};
        AscendC::DataCopyPad(reduceMaxSrc0, vec1ResGm[runInfo.s1BlockInnerLoop * constInfo.s2BaseSize + 
                               VEC_REDUCE_BASESIZE * blockIdx], dataCopyParams, padParams);

        reduceMaxSrcQue.template EnQue<T>(reduceMaxSrc0);
        reduceMaxSrc0 = reduceMaxSrcQue.template DeQue<T>();

        if (curBlockSize < VEC_REDUCE_BASESIZE) {
            Duplicate(reduceMaxSrc0[DenseLISoftmaxLseAlign<uint32_t>(curBlockSize, 8)],
                     static_cast<float>(SOFTMAX_MIN_NUM), 
                     VEC_REDUCE_BASESIZE - DenseLISoftmaxLseAlign<uint32_t>(curBlockSize, 8));
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Max(reduceMaxDstTensor, reduceMaxSrc0, reduceMaxSrc1, VEC_REDUCE_BASESIZE);
        AscendC::PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
    }

    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax<T>(reduceMaxDstTensor, reduceMaxDstTensor, sharedTmpTensor, VEC_REDUCE_BASESIZE, false);
    reduceMaxSrcQue.template EnQue<T>(reduceMaxSrc0);
    reduceMaxSrc0 = reduceMaxSrcQue.template DeQue<T>();

    DataCopyExtParams copyOutParams{1, 1 * sizeof(float), 0, 0, 0};
    AscendC::DataCopyPad(softmaxMaxGm[runInfo.reduceMaxOutGmOffset + runInfo.s1Idx], 
                         reduceMaxDstTensor, copyOutParams);
    maxValue_ = reduceMaxDstTensor.GetValue(0);
    reduceMaxSrcQue.template FreeTensor(reduceMaxSrc0);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockVec<TEMPLATE_ARGS>::ProcessReduceSum(
    DenseLISoftmaxLseRunInfo &runInfo, uint32_t pingPongFlag)
{
    uint32_t actS2Size = runInfo.actS2Size;
    uint32_t actS1Size = runInfo.actS1Size;
    uint32_t s1Idx = runInfo.s1Idx;
    uint32_t maskedActS2Size = (actS2Size - actS1Size + s1Idx + 1) > actS2Size ?
                               actS2Size : (actS2Size - actS1Size + s1Idx + 1);

    uint32_t blockNum = DenseLISoftmaxLseCeilDiv<uint32_t>(maskedActS2Size, VEC_REDUCE_BASESIZE);

    LocalTensor<T> reduceSumSrc0 = reduceSumSrcQue.template AllocTensor<T>();
    LocalTensor<T> reduceSumSrc1 = reduceSumSrc0[VEC_REDUCE_BASESIZE];
    LocalTensor<T> reduceSumDstTensor = reduceSumSrc0[3 * VEC_REDUCE_BASESIZE];
    LocalTensor<T> sharedTmpTensor = reduceSumSrc0[2 * VEC_REDUCE_BASESIZE];
    LocalTensor<T> subMaxTensor = reduceSumSrc0[4 * VEC_REDUCE_BASESIZE];

    Duplicate(subMaxTensor, maxValue_, VEC_REDUCE_BASESIZE);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_S>(EVENT_ID4);
    for (uint32_t blockIdx = 0; blockIdx <= blockNum - 1; blockIdx++) {
        WaitFlag<HardEvent::V_S>(EVENT_ID4);
        uint32_t curBlockSize = (blockIdx != blockNum - 1) ? VEC_REDUCE_BASESIZE :
                                (maskedActS2Size - VEC_REDUCE_BASESIZE * blockIdx);

        if (blockIdx > 0) {
            uint64_t mask = 64;
            uint8_t repeatTime = 1;
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Copy(reduceSumSrc1, reduceSumDstTensor, mask, repeatTime, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            Duplicate(reduceSumSrc1, 0.0f, VEC_REDUCE_BASESIZE);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(curBlockSize * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{true, 0,
            static_cast<uint8_t>(DenseLISoftmaxLseAlign<uint32_t>(curBlockSize, 8) - curBlockSize),
            static_cast<float>(SOFTMAX_MIN_NUM)};
        AscendC::DataCopyPad(reduceSumSrc0, vec1ResGm[runInfo.s1BlockInnerLoop * constInfo.s2BaseSize + 
                               VEC_REDUCE_BASESIZE * blockIdx], dataCopyParams, padParams);

        reduceSumSrcQue.template EnQue<T>(reduceSumSrc0);
        reduceSumSrc0 = reduceSumSrcQue.template DeQue<T>();

        if (curBlockSize < VEC_REDUCE_BASESIZE) {
            Duplicate(reduceSumSrc0[DenseLISoftmaxLseAlign<uint32_t>(curBlockSize, 8)],
                     static_cast<float>(SOFTMAX_MIN_NUM),
                     VEC_REDUCE_BASESIZE - DenseLISoftmaxLseAlign<uint32_t>(curBlockSize, 8));
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Sub(reduceSumSrc0, reduceSumSrc0, subMaxTensor, VEC_REDUCE_BASESIZE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(reduceSumSrc0, reduceSumSrc0, VEC_REDUCE_BASESIZE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(reduceSumDstTensor, reduceSumSrc0, reduceSumSrc1, VEC_REDUCE_BASESIZE);
        AscendC::PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID4);
    }

    WaitFlag<HardEvent::V_S>(EVENT_ID4);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<T>(reduceSumDstTensor, reduceSumDstTensor, sharedTmpTensor, VEC_REDUCE_BASESIZE);
    reduceSumSrcQue.template EnQue<T>(reduceSumSrc0);
    reduceSumSrc0 = reduceSumSrcQue.template DeQue<T>();

    DataCopyExtParams copyOutParams{1, 1 * sizeof(float), 0, 0, 0};
    AscendC::DataCopyPad(softmaxSumGm[runInfo.reduceMaxOutGmOffset + runInfo.s1Idx], 
                         reduceSumDstTensor, copyOutParams);
    reduceSumSrcQue.template FreeTensor(reduceSumSrc0);
}

TEMPLATES_DEF
class DenseLISoftmaxLseBlockVecDummy {
public:
    __aicore__ inline void InitParams(const DenseLISoftmaxLseConstInfo &vecConstInfo,
                                      const optiling::DenseLightningIndexerSoftmaxLseRegBaseTilingData *tiling){};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR weights, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                            GlobalTensor<T> &mm1Res, GlobalTensor<float> &vec1Res){};
    __aicore__ inline void InitBuffers(TPipe *pipe){};
    __aicore__ inline void ProcessVec1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &mm1ResBuf,
                                       DenseLISoftmaxLseRunInfo &runInfo){};
    __aicore__ inline void ProcessSoftmax(DenseLISoftmaxLseRunInfo &runInfo){};
};

} // namespace DenseLISoftmaxLseKernelRegbase

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_VECTOR_BLOCK_H