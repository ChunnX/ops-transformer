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
 * \brief Vector block for A5 (arch35) implementation
 */

#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_VECTOR_BLOCK_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_VECTOR_BLOCK_H

#include "dense_lightning_indexer_softmax_lse_regbase_common.h"

namespace DenseLISoftmaxLseRegbase {

TEMPLATES_DEF
class DenseLISoftmaxLseVecBlock {
public:
    __aicore__ inline DenseLISoftmaxLseVecBlock(){};
    __aicore__ inline void InitParams(const DenseLISoftmaxLseConstInfo &vecConstInfo,
                                      const optiling::DenseLISoftmaxLseRegBaseTilingData *tiling);
    __aicore__ inline void InitGlobalBuffer(
        GM_ADDR weights, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
        GlobalTensor<T> &mm1ResGm, GlobalTensor<T> &vec1ResGm);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void CopyInWeight(DenseLISoftmaxLseRunInfo &runInfo);
    __aicore__ inline void ProcessVec0(
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &mm1ResBuf,
        DenseLISoftmaxLseRunInfo &runInfo);
    __aicore__ inline void ProcessSoftmax(DenseLISoftmaxLseRunInfo &runInfo);

    GlobalTensor<INPUT_T> weightGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<T> mm1ResGm;
    GlobalTensor<T> vec1ResGm;

    const optiling::DenseLISoftmaxLseRegBaseTilingData *tilingData;
    TPipe *pipe;
    DenseLISoftmaxLseConstInfo constInfo;

private:
    static constexpr event_t EVENT_ID0 = (event_t)4;
    TBuf<> weightTbuf;
    TQue<QuePosition::VECIN, 1> mm1InQue;
    TQue<QuePosition::VECOUT, 1> reduceSumOutQue;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseVecBlock<TEMPLATE_ARGS>::InitParams(
    const DenseLISoftmaxLseConstInfo &vecConstInfo,
    const optiling::DenseLISoftmaxLseRegBaseTilingData *tiling)
{
    this->constInfo = vecConstInfo;
    tilingData = tiling;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseVecBlock<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR weights, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
    GlobalTensor<T> &mm1ResGm, GlobalTensor<T> &vec1ResGm)
{
    weightGm.SetGlobalBuffer((__gm__ INPUT_T *)weights);
    softmaxMaxGm.SetGlobalBuffer((__gm__ T *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ T *)softmaxSum);
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseVecBlock<TEMPLATE_ARGS>::InitBuffers(TPipe *pipe)
{
    this->pipe = pipe;
    pipe->InitBuffer(weightTbuf, DenseLISoftmaxLseConstInfo::BUFFER_SIZE_BYTE_512);
    pipe->InitBuffer(mm1InQue, 1, VEC_SY_BASESIZE * sizeof(T));
    pipe->InitBuffer(reduceSumOutQue, 1, VEC_P_BASESIZE * sizeof(T));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseVecBlock<TEMPLATE_ARGS>::CopyInWeight(
    DenseLISoftmaxLseRunInfo &runInfo)
{
    LocalTensor<INPUT_T> weightUb = weightTbuf.Get<INPUT_T>();
    DataCopy(weightUb, weightGm[runInfo.weightTensorOffset], constInfo.gSizeQueryIndex);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseVecBlock<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &mm1ResBuf,
    DenseLISoftmaxLseRunInfo &runInfo)
{
    LocalTensor<T> mm1Ub = mm1ResBuf.GetTensor<T>();
    LocalTensor<INPUT_T> weightUb = weightTbuf.Get<INPUT_T>();
    LocalTensor<T> reduceSumUb = reduceSumOutQue.AllocTensor<T>();

    Max(mm1Ub, mm1Ub, (T)0, runInfo.s2RealSize * runInfo.nRealSize);

    Mul(mm1Ub, mm1Ub, weightUb, runInfo.s2RealSize * runInfo.nRealSize);

    ReduceSum(reduceSumUb, mm1Ub, mm1Ub, runInfo.s2RealSize);

    reduceSumOutQue.EnQue(reduceSumUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseVecBlock<TEMPLATE_ARGS>::ProcessSoftmax(
    DenseLISoftmaxLseRunInfo &runInfo)
{
    LocalTensor<T> reduceSumUb = reduceSumOutQue.DeQue<T>();
    LocalTensor<T> maxUb = reduceSumOutQue.AllocTensor<T>();
    LocalTensor<T> sumUb = reduceSumOutQue.AllocTensor<T>();

    ReduceMax(maxUb, reduceSumUb, reduceSumUb, runInfo.s2RealSize);

    Sub(reduceSumUb, reduceSumUb, maxUb, runInfo.s2RealSize);
    Exp(reduceSumUb, reduceSumUb, runInfo.s2RealSize);

    ReduceSum(sumUb, reduceSumUb, reduceSumUb, runInfo.s2RealSize);

    DataCopy(softmaxMaxGm[runInfo.weightTensorOffset], maxUb, constInfo.gSizeQueryIndex);
    DataCopy(softmaxSumGm[runInfo.weightTensorOffset], sumUb, constInfo.gSizeQueryIndex);

    reduceSumOutQue.FreeTensor(reduceSumUb);
}

TEMPLATES_DEF
class DenseLISoftmaxLseVecBlockDummy {
public:
    __aicore__ inline DenseLISoftmaxLseVecBlockDummy(){};
    __aicore__ inline void InitParams(const DenseLISoftmaxLseConstInfo &vecConstInfo,
                                      const optiling::DenseLISoftmaxLseRegBaseTilingData *tiling) {};
    __aicore__ inline void InitGlobalBuffer(
        GM_ADDR weights, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
        GlobalTensor<T> &mm1ResGm, GlobalTensor<T> &vec1ResGm) {};
    __aicore__ inline void InitBuffers(TPipe *pipe) {};
    __aicore__ inline void CopyInWeight(DenseLISoftmaxLseRunInfo &runInfo) {};
    __aicore__ inline void ProcessVec0(
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &mm1ResBuf,
        DenseLISoftmaxLseRunInfo &runInfo) {};
    __aicore__ inline void ProcessSoftmax(DenseLISoftmaxLseRunInfo &runInfo) {};
};

}

#endif