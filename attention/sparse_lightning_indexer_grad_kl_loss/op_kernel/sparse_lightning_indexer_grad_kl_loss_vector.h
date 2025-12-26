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
 * \file sparse_lightning_indexer_grad_kl_loss_vector.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_lightning_indexer_grad_kl_loss_common.h"
#include "sparse_lightning_indexer_grad_kl_loss_tiling.h"

using namespace AscendC;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SLIT> 
class SLIKLLossVectorService {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename SLIT::inputQT;
    using KV_T = typename SLIT::inputKT;
    using OUT_T = typename SLIT::outputT;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = KV_T;
    using MM12_OUT_T = T;
    using MM5_OUT_T = T;

    static constexpr SLITopKRange topKRange = SLIT::topKRange;
    static constexpr SLILayout LAYOUT_T = SLIT::inputQLayout;
    static constexpr SLILayout KV_LAYOUT_T = SLIT::inputKLayout;
    static constexpr UBAllocPolicy<topKRange> ubAllocPolicy;

    __aicore__ inline SLIKLLossVectorService(){};
    __aicore__ inline void InitParams(const SLIGradKLLossConstInfo &vecConstInfo,
        const optiling::SparseLightningIndexerGradKLLossTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    // =============== vector 0 functions ==============
    __aicore__ inline void InitVector0GM(GlobalTensor<KV_T> &key, GlobalTensor<KV_T> &keyRope,
        GlobalTensor<KV_T> &keyIndex, GlobalTensor<int32_t> &topK,
        GlobalTensor<int64_t> &actualSeqLengthsQ, GlobalTensor<int64_t> &actualSeqLengthsKV,
        GlobalTensor<KV_T> &gatherPRes, GlobalTensor<KV_T> &gatherSYRes);
    __aicore__ inline void ProcessVector0(SLIGradKLLossRunInfo &runInfo);

    // =============== vector 1 functions ==============
    __aicore__ inline void InitVector1GM(const GlobalTensor<MM12_OUT_T> &bmm1Res,
                                    const GlobalTensor<T> &softmaxMax, const GlobalTensor<T> &softmaxSum,
                                    const GlobalTensor<MM12_OUT_T> &bmm2Res, const GlobalTensor<Q_T> &weight,
                                    const GlobalTensor<T> psySync, GlobalTensor<T> &loss,
                                    GlobalTensor<OUT_T> &dWeight, GlobalTensor<T> &reluGm,
                                    GlobalTensor<KV_T> &reluGradRes);
    __aicore__ inline void ProcessVector1(SLIGradKLLossRunInfo &runInfo);

    // =============== vector 2 functions ==============
    __aicore__ inline void ProcessVector2(SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void InitVector2GM(const GlobalTensor<MM5_OUT_T> &bmm5Res, const GlobalTensor<int32_t> &topK,
        GlobalTensor<T> &scatterAddRes);

private:
    // =============== vector 0 functions ==============
    template <event_t IdStart, bool gatherRope>
    __aicore__ inline void MergeKv(const SLIGradKLLossRunInfo &runInfo);
    template <bool gatherRope>
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
        int64_t keyBNBOffset,int64_t s2IdLimit, const SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline int64_t GetKeyRopeGmOffset(int64_t realS2Idx, const SLIGradKLLossRunInfo &runInfo, int64_t s2IdLimit);
    __aicore__ inline int64_t GetKeyGmOffset(int64_t realS2Idx, const SLIGradKLLossRunInfo &runInfo, int64_t s2IdLimit, int32_t dValue);
    template <bool gatherRope>
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
        int64_t realS2Idx2, const SLIGradKLLossRunInfo &runInfo);
    template <event_t IdStart, bool gatherRope>
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2GmStartOffset,
        int64_t mergeMte3Idx, const SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, const SLIGradKLLossRunInfo &runInfo);

    // =============== vector 1 functions ==============
    __aicore__ inline void VectorP(SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void PreloadWeight(SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void VectorSy(SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void ReLUGrad(LocalTensor<KV_T> &reluGradOutTensor, LocalTensor<T> &ReLuTensor, LocalTensor<T> &subResTensor,
        int32_t kRealSize, int32_t kRealSizeAlign8, SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void DataCopyReluRes(LocalTensor<T> &reluResTensor, GlobalTensor<T> &reluGm,
        int32_t kRealSize, int32_t kRealSizeAlign8, int64_t reluResOffset);
    __aicore__ inline void VectorDwDqDk(SLIGradKLLossRunInfo &runInfo, int32_t kLoopIdx);
    __aicore__ inline void VectorLoss(SLIGradKLLossRunInfo &runInfo, int32_t kLoopIdx);
    // =============== vector 2 functions ==============
    __aicore__ inline void ScatterAddCopyOutSingle(const LocalTensor<MM5_OUT_T> &srcUb, int64_t keyBNBOffset);

    // =============== vector common variable ==============
    SLIGradKLLossConstInfo constInfo;
    const optiling::SparseLightningIndexerGradKLLossTilingData *__restrict tilingData;

    // global tensor
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> keyIndexGm;
    GlobalTensor<KV_T> keyRopeGm;
    GlobalTensor<int32_t> topKGm;

    GlobalTensor<MM12_OUT_T> bmm1ResGm;
    GlobalTensor<T> softmaxMaxGm; 
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<MM12_OUT_T> bmm2ResGm;
    GlobalTensor<Q_T> weightGm;
    GlobalTensor<T> psySyncGm, psySyncOtherGm;
    GlobalTensor<T> tempGm;
    GlobalTensor<T> lossGm;
    GlobalTensor<OUT_T> dWeightGm;
    GlobalTensor<T> reluGm;
    GlobalTensor<KV_T> reluGradResGm;

    GlobalTensor<MM5_OUT_T> bmm5ResGm;

    // local tensor
    TBuf<> mm1Tbuf;
    TBuf<> mm2TBuf;         // 复用 -> mm4 scatterAdd reluGrad
    TBuf<> sharedTBuf;
    TBuf<> resPSYTBuf;
    TBuf<> reduceSumDwTBuf;
    TBuf<> v1TmpTBuf;

    TBuf<> weightTBuf;
    TBuf<> weightInTBuf;
    TBuf<> lossSumTBuf;
    TBuf<> dwTBuf;
    TBuf<> maskBuf;
    TBuf<> scatterAddBuf;

    // =============== vector0 variable ==============
    GlobalTensor<KV_T> gatherPResGm;
    GlobalTensor<KV_T> gatherSYResGm;

    TBuf<> gatherTbuf;
    
    LocalTensor<KV_T> gatherKeyUb;
    LocalTensor<KV_T> gatherKeyIndexUb;
    static constexpr int32_t blockBytes = 32;

    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<KV_T> ropeMergUb_;

    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> srcTensor;
    GlobalTensor<KV_T> srcRopeTensor;

    GatherParams gatherParams;

    static constexpr uint32_t N_QUERY_INDEX_SPECIAL = 8;
    static constexpr int64_t UB_ROW_SIZE = 8;
    static constexpr event_t EVENT_ID0 = (event_t)4;
    static constexpr event_t EVENT_ID2 = (event_t)6;
    // =============== vector1 variable ==============
    LocalTensor<T> weightUb;
    LocalTensor<KV_T> weightInUb;

    LocalTensor<T> reluResUb;
    LocalTensor<T> reduceSumYResTmpBuffer;
    LocalTensor<T> reduceSumYResUb;
    LocalTensor<T> reduceSumPResUb;
    LocalTensor<uint8_t> reduceSumTmpBuffer;
    LocalTensor<uint8_t> softmaxTmpBuffer;
    LocalTensor<T> reduceSumDwResUb;

    event_t eventIdMte2ToV4SY;
    event_t eventIdVToV4SY;
    event_t eventIdVToMte24SY;
    event_t eventIdVToMte2P[2];
    event_t eventIdVToMte2Weight[2];

    event_t eventIdMTE3ToVTmp;

    // =============== vector2 variable ==============
    GlobalTensor<T> scatterAddResGm;
    LocalTensor<T> scatterAddUb;
    event_t eventIdScatterAdd;
    event_t eventIdScatterAddPong;
    int32_t scatterAddPingpong = 0;

    // =============== vector3 variable ==============
};

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitParams(const SLIGradKLLossConstInfo &vecConstInfo,
    const optiling::SparseLightningIndexerGradKLLossTilingData *__restrict tilingData)
{
    this->constInfo = vecConstInfo;
    this->tilingData = tilingData;
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitVector0GM(
    GlobalTensor<KV_T> &key, GlobalTensor<KV_T> &keyRope, GlobalTensor<KV_T> &keyIndex, GlobalTensor<int32_t> &topK,
    GlobalTensor<int64_t> &actualSeqLengthsQ, GlobalTensor<int64_t> &actualSeqLengthsKV,
    GlobalTensor<KV_T> &gatherPRes, GlobalTensor<KV_T> &gatherSYRes)
{
    this->keyGm = key;
    this->keyRopeGm = keyRope;
    this->keyIndexGm = keyIndex;
    this->topKGm = topK;

    this->gatherPResGm = gatherPRes;
    this->gatherSYResGm = gatherSYRes;
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitVector1GM(const GlobalTensor<MM12_OUT_T> &bmm1Res,
                                                        const GlobalTensor<T> &softmaxMax, const GlobalTensor<T> &softmaxSum,
                                                        const GlobalTensor<MM12_OUT_T> &bmm2Res, const GlobalTensor<Q_T> &weight,
                                                        const GlobalTensor<T> psySync, GlobalTensor<T> &loss,
                                                        GlobalTensor<OUT_T> &dWeight, GlobalTensor<T> &reluGm,
                                                        GlobalTensor<KV_T> &reluGradRes)
{
    this->bmm1ResGm = bmm1Res;
    this->softmaxMaxGm = softmaxMax;
    this->softmaxSumGm = softmaxSum;
    this->bmm2ResGm = bmm2Res;
    this->weightGm = weight;
    this->psySyncGm = psySync[constInfo.subBlockIdx * constInfo.kSize * 2];
    this->psySyncOtherGm = psySync[!constInfo.subBlockIdx * constInfo.kSize * 2];
    
    this->lossGm = loss;
    this->dWeightGm = dWeight;
    this->reluGm = reluGm;
    this->reluGradResGm = reluGradRes;
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitVector2GM(const GlobalTensor<MM5_OUT_T> &bmm5Res,
    const GlobalTensor<int32_t> &topK, GlobalTensor<T> &scatterAddRes)
{
    this->bmm5ResGm = bmm5Res;
    this->topKGm = topK;
    this->scatterAddResGm = scatterAddRes;
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitBuffers(TPipe *pipe)
{
    // 可选输入的buffer空间，保持和stage1处理的size一致
    pipe->InitBuffer(this->gatherTbuf, ubAllocPolicy.gatherUbSize);
    pipe->InitBuffer(this->mm1Tbuf, ubAllocPolicy.mm1UbSize);
    pipe->InitBuffer(this->mm2TBuf, ubAllocPolicy.mm2UbSize);
    pipe->InitBuffer(this->sharedTBuf, ubAllocPolicy.sharedUbSize);
    pipe->InitBuffer(this->resPSYTBuf, ubAllocPolicy.resPSYUbSize);
    pipe->InitBuffer(this->reduceSumDwTBuf, ubAllocPolicy.reduceSumDwUbSize);
    pipe->InitBuffer(this->v1TmpTBuf, ubAllocPolicy.v1TmpUbSize);
    pipe->InitBuffer(this->lossSumTBuf, ubAllocPolicy.lossSumUbSize);
    pipe->InitBuffer(this->weightTBuf, ubAllocPolicy.weightUbSize);
    pipe->InitBuffer(this->weightInTBuf, ubAllocPolicy.weightInUbSize);
    pipe->InitBuffer(this->dwTBuf, ubAllocPolicy.dwUbSize);
    pipe->InitBuffer(this->maskBuf, ubAllocPolicy.maskUbSize);
    pipe->InitBuffer(this->scatterAddBuf, ubAllocPolicy.scatterAddUbSize);

    gatherKeyUb = gatherTbuf.Get<KV_T>();
    gatherKeyIndexUb = gatherKeyUb[SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_9K * 2];

    weightUb = weightTBuf.Get<T>();
    weightInUb = weightInTBuf.Get<KV_T>();
    reluResUb = mm2TBuf.Get<T>();
    reduceSumTmpBuffer = sharedTBuf.GetWithOffset<uint8_t>(8*1024, 8*1024);
    reduceSumYResTmpBuffer = sharedTBuf.GetWithOffset<T>(constInfo.kSize, 2*8*1024);
    reduceSumYResUb = resPSYTBuf.Get<T>();
    reduceSumPResUb = resPSYTBuf.template Get<T>();
    reduceSumDwResUb = reduceSumDwTBuf.Get<T>();
    softmaxTmpBuffer = sharedTBuf.Get<uint8_t>();
    scatterAddUb = scatterAddBuf.Get<T>();
}

template <typename SLIT> __aicore__ inline void SLIKLLossVectorService<SLIT>::AllocEventID()
{
    eventIdMTE3ToVTmp = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    eventIdScatterAdd = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    eventIdScatterAddPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    eventIdVToMte2Weight[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdVToMte2Weight[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdVToMte2P[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdVToMte2P[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMTE3ToVTmp);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdScatterAdd);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdScatterAddPong);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[0]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[1]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[0]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[1]);
}

template <typename SLIT> __aicore__ inline void SLIKLLossVectorService<SLIT>::FreeEventID()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMTE3ToVTmp);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdScatterAdd);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdScatterAddPong);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[0]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[1]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[0]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[1]);

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMTE3ToVTmp);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdScatterAdd);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdScatterAddPong);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2Weight[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2Weight[1]);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[0]);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[1]);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::ProcessVector0(SLIGradKLLossRunInfo &runInfo)
{
    gatherParams.dValue = constInfo.dSizeQuery;
    gatherParams.dRopeValue = constInfo.dSizeRope;
    gatherParams.gatherResGmEleSize = constInfo.gatherKeySize;

    this->kvMergeGm_.SetGlobalBuffer((__gm__ KV_T *)gatherPResGm.GetPhyAddr());
    this->srcTensor.SetGlobalBuffer((__gm__ KV_T *)keyGm.GetPhyAddr());
    this->srcRopeTensor.SetGlobalBuffer((__gm__ KV_T *)keyRopeGm.GetPhyAddr());
    this->kvMergUb_ = gatherKeyUb;
    this->ropeMergUb_ = gatherKeyUb[UB_ROW_SIZE * gatherParams.dValue * 2];

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + 1);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + 1);

    for (int32_t i = 0; i < runInfo.s2LoopTimes; ++i) {
        runInfo.s2Idx = i;
	
        gatherParams.s2ProcessSize = (i >= runInfo.s2LoopTimes - 1) ? runInfo.s2TailSize : constInfo.s2BaseSize;
        MergeKv<EVENT_ID0, SLIT::hasRope>(runInfo);
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V0_TO_C1_P_FLAG[i & 1]);
    }

    gatherParams.dValue = constInfo.dSizeQueryIndex;
    gatherParams.dRopeValue = 0;
    gatherParams.gatherResGmEleSize = constInfo.gatherKeyIndexSize;
    this->kvMergeGm_.SetGlobalBuffer((__gm__ KV_T *)gatherSYResGm.GetPhyAddr());
    this->srcTensor.SetGlobalBuffer((__gm__ KV_T *)keyIndexGm.GetPhyAddr());
    this->kvMergUb_ = gatherKeyIndexUb;

    for (int32_t i = 0; i < runInfo.s2LoopTimes; ++i) {
        runInfo.s2Idx = i;
        gatherParams.s2ProcessSize = (i >= runInfo.s2LoopTimes - 1) ? runInfo.s2TailSize : constInfo.s2BaseSize;
        MergeKv<EVENT_ID2, false>(runInfo);
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V0_TO_C1_SY_FLAG[i & 1]);
    }

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + 1);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + 1);
}

template <typename SLIT>
template <bool gatherRope>
__aicore__ inline void
SLIKLLossVectorService<SLIT>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                       int64_t keyBNBOffset, int64_t s2IdLimit, const SLIGradKLLossRunInfo &runInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        (realS2Idx + constInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx : constInfo.sparseBlockSize);
    DataCopyExtParams intriParams;
    intriParams.blockLen = validS2Count * gatherParams.dValue * sizeof(KV_T);
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    DataCopyPad(kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dValue + (mte2Size - mte3Size) * gatherParams.dValue],
                srcTensor[keyBNBOffset * gatherParams.dValue], intriParams, padParams);

    if constexpr (gatherRope) {
        intriParams.blockLen = validS2Count * gatherParams.dRopeValue * sizeof(KV_T);
        DataCopyPad(ropeMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dRopeValue + (mte2Size - mte3Size) * gatherParams.dRopeValue],
                    srcRopeTensor[keyBNBOffset * gatherParams.dRopeValue], intriParams, padParams);
    }
    mte2Size += validS2Count;
}

template <typename SLIT>
__aicore__ inline int64_t SLIKLLossVectorService<SLIT>::GetKeyRopeGmOffset(int64_t realS2Idx,
    const SLIGradKLLossRunInfo &runInfo, int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t tensorBRopeOffset = runInfo.accumS2Idx * gatherParams.dRopeValue;
    int64_t realKeyRopeGmOffset = (tensorBRopeOffset + realS2Idx * gatherParams.dRopeValue) / gatherParams.dRopeValue;
    return realKeyRopeGmOffset;
}

template <typename SLIT>
__aicore__ inline int64_t SLIKLLossVectorService<SLIT>::GetKeyGmOffset(int64_t realS2Idx,
    const SLIGradKLLossRunInfo &runInfo, int64_t s2IdLimit, int32_t dValue)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t tensorBOffset = runInfo.accumS2Idx * dValue;
    int64_t realKeyGmOffset = (tensorBOffset + realS2Idx * dValue) / dValue;
    return realKeyGmOffset;
}

template <typename SLIT>
template <bool gatherRope>
__aicore__ inline void SLIKLLossVectorService<SLIT>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
    int64_t realS2Idx1, int64_t realS2Idx2, const SLIGradKLLossRunInfo &runInfo)
{
    int64_t s2IdLimit = runInfo.s2SparseLen;

    int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit, gatherParams.dValue);
    int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit, gatherParams.dValue);
    if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
        return;
    }

    int64_t keySrcStride = 0;
    int64_t keyRopeSrcStride = 0;
    keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) :
                    (keyOffset2 - keyOffset1)) - constInfo.sparseBlockSize) * gatherParams.dValue * sizeof(KV_T);
    if constexpr (gatherRope) {
        int64_t keyRopeOffset1 = GetKeyRopeGmOffset(realS2Idx1, runInfo, s2IdLimit);
        int64_t keyRopeOffset2 = GetKeyRopeGmOffset(realS2Idx2, runInfo, s2IdLimit);
        keyRopeSrcStride = ((keyRopeOffset1 > keyRopeOffset2 ? (keyRopeOffset1 - keyRopeOffset2) :
                            (keyRopeOffset2 - keyRopeOffset1)) - constInfo.sparseBlockSize) *
                                gatherParams.dRopeValue * sizeof(KV_T);
    }

    bool key1LessThankey2 = (realS2Idx1 > realS2Idx2);
    bool strideInvalid = (keySrcStride >= INT32_MAX) || (keySrcStride < 0) ||
        (keyRopeSrcStride >= INT32_MAX || keyRopeSrcStride < 0);
    bool copyOutOfRange = (realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
        realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit);

    if (key1LessThankey2 || strideInvalid || copyOutOfRange) {
        // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
        CopyInSingleKv<gatherRope>(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, runInfo);
        CopyInSingleKv<gatherRope>(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, runInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockLen = constInfo.sparseBlockSize * gatherParams.dValue * sizeof(KV_T);
        intriParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyOffset1 > -1 ? keyOffset1 : keyOffset2;
        if (keyOffset2 > -1 && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        DataCopyPad(kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dValue + (mte2Size - mte3Size) * gatherParams.dValue],
                    srcTensor[startGmOffset * gatherParams.dValue], intriParams, padParams);

        if constexpr (gatherRope) {
            intriParams.blockLen = constInfo.sparseBlockSize * gatherParams.dRopeValue * sizeof(KV_T);
            intriParams.srcStride = keyRopeSrcStride;
            DataCopyPad(ropeMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dRopeValue + (mte2Size - mte3Size) * gatherParams.dRopeValue],
                        srcRopeTensor[startGmOffset * gatherParams.dRopeValue], intriParams, padParams);
        }

        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

template <typename SLIT>
template <event_t IdStart, bool gatherRope>
__aicore__ inline void SLIKLLossVectorService<SLIT>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
    int64_t s2GmStartOffset, int64_t mergeMte3Idx, const SLIGradKLLossRunInfo &runInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(mergeMte3Idx + IdStart);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mergeMte3Idx + IdStart);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = mte2Size - mte3Size;
    dataCopyParams.blockLen = gatherParams.dValue * sizeof(KV_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    int64_t gmStartOffset = runInfo.taskIdMod2 * gatherParams.gatherResGmEleSize +
        runInfo.s2Idx * constInfo.s2BaseSize * (gatherParams.dValue + gatherParams.dRopeValue);

    DataCopyPad(kvMergeGm_[gmStartOffset + (s2GmStartOffset + mte3Size) * gatherParams.dValue],
                kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dValue], dataCopyParams);

    if constexpr (gatherRope) {
        dataCopyParams.blockLen = gatherParams.dRopeValue * sizeof(KV_T);
        DataCopyPad(kvMergeGm_[gmStartOffset + constInfo.s2BaseSize * gatherParams.dValue + (s2GmStartOffset + mte3Size) *
            gatherParams.dRopeValue], ropeMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dRopeValue], dataCopyParams);
    }
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
    const SLIGradKLLossRunInfo &runInfo)
{
    int64_t topkGmIdx = (s2GmOffset + runInfo.s2Idx * constInfo.s2BaseSize) / constInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= constInfo.kSize)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topKGm.GetValue(runInfo.topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + runInfo.s2Idx * constInfo.s2BaseSize) % constInfo.sparseBlockSize);
}

template <typename SLIT>
template <event_t IdStart, bool gatherRope>
__aicore__ inline void SLIKLLossVectorService<SLIT>::MergeKv(const SLIGradKLLossRunInfo &runInfo)
{
    int64_t s2ProcessSize = gatherParams.s2ProcessSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);

    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize: s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit; s2GmOffsetArray += 2 * constInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, runInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult<IdStart, gatherRope>(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            mergeMte3Idx = 1 - mergeMte3Idx;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, s2IdxArray1, runInfo);
        CopyInKv<gatherRope>(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo);
        if ((mte2Size - mte3Size + 2 * constInfo.sparseBlockSize > UB_ROW_SIZE) ||
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult<IdStart, gatherRope>(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            mergeMte3Idx = 1 - mergeMte3Idx;
            needWaitMte3ToMte2 = true;
        }
    }
    return;
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::ProcessVector1(SLIGradKLLossRunInfo &runInfo)
{
    eventIdMte2ToV4SY = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    eventIdVToMte24SY = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());

    CrossCoreWaitFlag(SYNC_C1_TO_V1_P_FLAG[runInfo.taskIdMod2]);
    CrossCoreWaitFlag(SYNC_C1_TO_V1_SY_FLAG[runInfo.taskIdMod2]);
    PreloadWeight(runInfo);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMTE3ToVTmp);
    // 保证 reduceN 轴的S'Y在同一个Vec核内 将P和S'Y任务分割
    if (runInfo.calcP) {
        VectorP(runInfo);
    } else {
        VectorSy(runInfo); // 计算S'Y部分，一直到Softmax
    }
    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag<0x1, PIPE_MTE2>(0x8);
    for (int32_t kLoopIdx = 0; kLoopIdx < runInfo.kLoopTimes; ++kLoopIdx) {
        VectorDwDqDk(runInfo, kLoopIdx); // Sub + Mul + Mul + ReluGrad
        if (kLoopIdx >= runInfo.kLoopTimes - 1) {
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V1_TO_C2_DW_FLAG[runInfo.taskIdMod2]);
        }
        VectorLoss(runInfo, kLoopIdx);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMTE3ToVTmp);

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV4SY);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte24SY);
}

template <typename SLIT>
__aicore__ inline void
SLIKLLossVectorService<SLIT>::ScatterAddCopyOutSingle(const LocalTensor<MM5_OUT_T> &srcUb, int64_t keyBNBOffset)
{
    if (keyBNBOffset < 0) {
        return;
    }
    LocalTensor<T> srcTmpUb = srcUb.template ReinterpretCast<T>();
    DataCopy(scatterAddResGm[keyBNBOffset * constInfo.dSizeQueryIndex], srcTmpUb, constInfo.dSizeQueryIndex);
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::ProcessVector2(SLIGradKLLossRunInfo &runInfo)
{
    CrossCoreWaitFlag(SYNC_C2_TO_V2_SA_FLAG[runInfo.taskIdMod2]);

    int32_t v0RealKSize, v1RealKSize, vRealKSize;
    int64_t coreKOffset;
    v0RealKSize = CeilDiv(runInfo.kRealSize, 2);
    v0RealKSize = Min(SLIGAlign(v0RealKSize, 2), runInfo.kRealSize);
    v1RealKSize = runInfo.kRealSize - v0RealKSize;
    if (constInfo.subBlockIdx == 0) {
        vRealKSize = v0RealKSize;
        coreKOffset = 0;
    } else {
        vRealKSize = v1RealKSize;
        coreKOffset = v0RealKSize;
    }
    if (vRealKSize <= 0) {
        return;
    }

    GlobalTensor<MM5_OUT_T> srcGm = bmm5ResGm[(runInfo.taskIdMod2 * constInfo.kSize + coreKOffset) * constInfo.dSizeQueryIndex];
    LocalTensor<MM5_OUT_T> scatterAddTmpUb;
    int32_t kSplitSize = ubAllocPolicy.scatterAddUbSize / 2 / sizeof(T) / constInfo.dSizeQueryIndex;
    int32_t kTailSize = (vRealKSize % kSplitSize == 0) ? kSplitSize : (vRealKSize % kSplitSize);
    int32_t kProcessSize = kSplitSize;
    int32_t kLoopTimes = CeilDiv(vRealKSize, kSplitSize);
    int64_t realS2Idx1, realS2Idx2, s2GmOffset;
    event_t eventIdArr[2] = {eventIdScatterAdd, eventIdScatterAddPong};
    runInfo.s2Idx = 0;

    SetAtomicAdd<T>();
    for (int32_t kLoopIdx = 0; kLoopIdx < kLoopTimes; ++kLoopIdx) {
        if (kLoopIdx >= kLoopTimes - 1) {
            kProcessSize = kTailSize;
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
        scatterAddTmpUb = scatterAddUb[scatterAddPingpong * kSplitSize * constInfo.dSizeQueryIndex].template ReinterpretCast<MM5_OUT_T>();
        DataCopy(scatterAddTmpUb, srcGm[kLoopIdx * kSplitSize * constInfo.dSizeQueryIndex], kProcessSize * constInfo.dSizeQueryIndex);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);

        constexpr int32_t topKSplitSize = 2;
        int32_t topKTailSize = (kProcessSize % topKSplitSize == 0) ? topKSplitSize : (kProcessSize % topKSplitSize);
        int32_t topKProcessSize = topKSplitSize;
        int32_t topKLoopTimes = CeilDiv(kProcessSize, topKSplitSize);
        for (int32_t topKIdx = 0; topKIdx < topKLoopTimes; ++topKIdx) {
            if (topKIdx >= topKLoopTimes - 1) {
                topKProcessSize = topKTailSize;
            }
            s2GmOffset = coreKOffset + kLoopIdx * kSplitSize + topKIdx * topKSplitSize;
            GetRealS2Idx(s2GmOffset, realS2Idx1, runInfo);
            GetRealS2Idx(s2GmOffset + 1, realS2Idx2, runInfo);

            int64_t s2IdLimit = runInfo.s2SparseLen;
            int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit, constInfo.dSizeQueryIndex);
            int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit, constInfo.dSizeQueryIndex);
            if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
                return;
            }

            int64_t keySrcStride = 0;
            keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) :
                            (keyOffset2 - keyOffset1)) - constInfo.sparseBlockSize) * constInfo.dSizeQueryIndex * sizeof(T);

            bool strideInvalid = (keySrcStride >= INT32_MAX) || (keySrcStride < 0);
            bool copyOutOfRange = (realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
                realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit);
            bool key1LessThankey2 = (realS2Idx1 > realS2Idx2);

            int64_t ub1Offset = topKIdx * topKSplitSize * constInfo.dSizeQueryIndex;
            int64_t ub2Offset = ub1Offset + constInfo.dSizeQueryIndex;

            if (strideInvalid || copyOutOfRange || key1LessThankey2) {
                // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
                ScatterAddCopyOutSingle(scatterAddTmpUb[ub1Offset], keyOffset1);
                ScatterAddCopyOutSingle(scatterAddTmpUb[ub2Offset], keyOffset2);
            } else {
                DataCopyExtParams dataCopyParams;
                dataCopyParams.blockLen = constInfo.dSizeQueryIndex * sizeof(T);
                dataCopyParams.srcStride = 0;
                dataCopyParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
                dataCopyParams.dstStride = keySrcStride;
                int64_t keyStartOffset = (keyOffset1 >= 0) ? keyOffset1 : keyOffset2;
                int64_t ubStartOffset = (keyOffset1 >= 0) ? ub1Offset : ub2Offset;
                LocalTensor<T> srcTmpUb = scatterAddTmpUb.template ReinterpretCast<T>();
                DataCopyPad(scatterAddResGm[keyStartOffset * constInfo.dSizeQueryIndex], srcTmpUb[ubStartOffset], dataCopyParams);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
        scatterAddPingpong = 1 - scatterAddPingpong;
    }
    SetAtomicNone();
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorP(SLIGradKLLossRunInfo &runInfo)
{
    int64_t bmm1Offset = runInfo.taskIdMod2 * constInfo.gSizeQuery * constInfo.kSize;
    int64_t nLoops = CeilDiv(constInfo.gSizeQuery, runInfo.nBaseSizeP);
    int64_t kLoops = CeilDiv(runInfo.kRealSize, K_BASE_SIZE);

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    LocalTensor<T> mulsResUb[2] = {mm1Tbuf.Get<T>(), mm2TBuf.Get<T>()};
    LocalTensor<uint8_t> sharedTmpBuffer = sharedTBuf.GetWithOffset<uint8_t>(16 * 1024, 0);
    LocalTensor<T> reduceSumPTmpBuffer = sharedTBuf.GetWithOffset<T>(K_BASE_SIZE, 16 * 1024);

    int64_t softmaxInputSize = CeilDiv(runInfo.nBaseSizeP, 8) * 8;
    LocalTensor<T> softmaxMaxTmp[2] = {sharedTBuf.GetWithOffset<T>(softmaxInputSize, 0), sharedTBuf.GetWithOffset<T>(softmaxInputSize, softmaxInputSize * sizeof(T))};
    LocalTensor<T> softmaxSumTmp[2] = {sharedTBuf.GetWithOffset<T>(softmaxInputSize, 2 * softmaxInputSize * sizeof(T)), sharedTBuf.GetWithOffset<T>(softmaxInputSize, 3 * softmaxInputSize * sizeof(T))};
    LocalTensor<T> softmaxMaxUb = sharedTBuf.GetWithOffset<T>(softmaxInputSize * 8, 1024);
    LocalTensor<T> softmaxSumUb = sharedTBuf.GetWithOffset<T>(softmaxInputSize * 8, 2048);

    uint8_t pingpong = 0;
    for (int64_t n = 0; n < nLoops; n++) { // nLoops=4
        int64_t curNSize = (n == nLoops - 1) ? constInfo.gSizeQuery - (runInfo.nBaseSizeP * n) : runInfo.nBaseSizeP;
        for (int64_t k = 0; k < kLoops; k++) {
            int64_t curKOffset = K_BASE_SIZE * k;
            int64_t curKSize = (k == kLoops - 1) ? runInfo.kRealSize - curKOffset : K_BASE_SIZE;
            int64_t curKSizeAlign8 = CeilDiv(curKSize, 8) * 8;
            int64_t bmm1NOffset = n * runInfo.nBaseSizeP * constInfo.kSize + k * K_BASE_SIZE;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[pingpong]);
            DataCopyExtParams copyParams = {static_cast<uint16_t>(curNSize),
                                            static_cast<uint32_t>(curKSize * sizeof(T)),
                                            static_cast<uint32_t>((constInfo.kSize - curKSize) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> copyPadParams = {false, 0, 0, 0};
            AscendC::DataCopyPad(mulsResUb[pingpong], bmm1ResGm[bmm1Offset + bmm1NOffset], copyParams, copyPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::Muls(mulsResUb[pingpong], mulsResUb[pingpong], constInfo.scaleValue, curNSize * curKSizeAlign8);
            PipeBarrier<PIPE_V>();
            
            DataCopyExtParams softmaxCopyParams = {static_cast<uint16_t>(1),
                                                    static_cast<uint32_t>(runInfo.nBaseSizeP * sizeof(T)), // 8 * 4 = 32Bytes
                                                    static_cast<uint32_t>(0), 0, 0};
            DataCopyPadExtParams<T> softmaxCopyPadParams = {false, 0, 0, 0};
            int64_t softmaxInputOffset = 0;
            if constexpr (LAYOUT_T == SLILayout::BSND) {
                softmaxInputOffset = runInfo.bIdx * (constInfo.s1Size * constInfo.gSizeQuery) +
                    runInfo.s1Idx * constInfo.gSizeQuery + n * runInfo.nBaseSizeP;
            } else if constexpr (LAYOUT_T == SLILayout::TND) {
                softmaxInputOffset = runInfo.accumS1Idx * constInfo.gSizeQuery + n * runInfo.nBaseSizeP;
            }
            AscendC::DataCopyPad(softmaxMaxTmp[pingpong], softmaxMaxGm[softmaxInputOffset], softmaxCopyParams, softmaxCopyPadParams);
            AscendC::DataCopyPad(softmaxSumTmp[pingpong], softmaxSumGm[softmaxInputOffset], softmaxCopyParams, softmaxCopyPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::Brcb(softmaxMaxUb, softmaxMaxTmp[pingpong], curNSize, {1, 8});
            AscendC::Brcb(softmaxSumUb, softmaxSumTmp[pingpong], curNSize, {1, 8});
            PipeBarrier<PIPE_V>();
            LocalTensor<uint8_t> softmaxTmpUb = sharedTBuf.GetWithOffset<uint8_t>(8*1024, 0);
            SoftMaxTiling simpleSoftMaxTilingInfo = tilingData->vectorParams.simpleSoftmaxPTilingData;
            AscendC::SimpleSoftMax<T>(mulsResUb[pingpong], softmaxSumUb, softmaxMaxUb, mulsResUb[pingpong],
                                      softmaxTmpUb, simpleSoftMaxTilingInfo, 
                                      {static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSizeAlign8),
                                      static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSize)});
            PipeBarrier<PIPE_V>();
            uint32_t reduceShape[] = {static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSizeAlign8)};
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(reduceSumPTmpBuffer, mulsResUb[pingpong], sharedTmpBuffer, reduceShape, true);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[pingpong]);
            if (n == 0) {
                AscendC::DataCopy(reduceSumPResUb[curKOffset], reduceSumPTmpBuffer, curKSizeAlign8);
            } else {
                AscendC::Add(reduceSumPResUb[curKOffset], reduceSumPResUb[curKOffset], reduceSumPTmpBuffer, curKSizeAlign8);
            }
            PipeBarrier<PIPE_V>();
            pingpong = 1 - pingpong;
        }
    }
    PipeBarrier<PIPE_V>();
    float gRec = 1.0f / static_cast<float>(static_cast<int64_t>(constInfo.gSizeQuery));
    AscendC::Muls(reduceSumPResUb, reduceSumPResUb, gRec, constInfo.kSize);
    PipeBarrier<PIPE_V>();
    event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3);
    AscendC::DataCopy(psySyncGm, reduceSumPResUb, constInfo.kSize);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::PreloadWeight(SLIGradKLLossRunInfo &runInfo)
{
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    int64_t weightOffset = 0;
    if constexpr (LAYOUT_T == SLILayout::BSND) {
        weightOffset = runInfo.bIdx * (constInfo.s1Size * constInfo.gSizeQueryIndex) +
            runInfo.s1Idx * constInfo.gSizeQueryIndex;
    } else if constexpr (LAYOUT_T == SLILayout::TND) {
        weightOffset = runInfo.accumS1Idx * constInfo.gSizeQueryIndex;
    }
    int64_t weightDBOffset = runInfo.taskIdMod2 * AlignTo(constInfo.gSizeQueryIndex, static_cast<uint32_t>(16));
    // weight 可以常驻, 所以直接搬运, 减少搬运切片

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[runInfo.taskIdMod2]);
    if (constInfo.gSizeQueryIndex == N_QUERY_INDEX_SPECIAL) {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(constInfo.gSizeQueryIndex * sizeof(KV_T)), 0, 0, 0};
        DataCopyPadExtParams<KV_T> copyPadParams = {true, 0, (uint8_t)(N_QUERY_INDEX_SPECIAL), 0.0};
        AscendC::DataCopyPad(weightInUb[weightDBOffset], weightGm[weightOffset], copyParams, copyPadParams);
    } else {
        AscendC::DataCopy(weightInUb[weightDBOffset], weightGm[weightOffset], constInfo.gSizeQueryIndex);
    }    
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
    AscendC::Cast(weightUb[weightDBOffset], weightInUb[weightDBOffset], AscendC::RoundMode::CAST_NONE, constInfo.gSizeQueryIndex);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[runInfo.taskIdMod2]);
    PipeBarrier<PIPE_V>();
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorSy(SLIGradKLLossRunInfo &runInfo)
{
    {
        // weight UB cast To weight.GetValue and release eventID
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    }
    // 核内保留一份BS1Idx
    int64_t realKSize = runInfo.s2RealSize; // int64_t &realKSize = runInfo.s2RealSize;
    int64_t realKSizeAlign = CeilDiv(realKSize, 8) * 8; 
 
    // 切B, S1做循环, bmm2 单次存储 [N1_INDEX, TopK] 大小的数据块
    // 为了softmax K轴尽量不切分, 只能在N1方向上切
    int64_t bmm2ResSize = constInfo.gSizeQueryIndex * constInfo.kSize; //  sizeof(float)
    int64_t bmm2ResOffset = runInfo.taskIdMod2 * bmm2ResSize;
    // relu无法复用LocalTensor 所以再 8/ 2 = 4行
    int64_t nSplitSize = ubAllocPolicy.mm2UbSize / constInfo.kSize / sizeof(float);
    // 一个Cube 做 G 行, 分给两个Vector, 每个做 G / 2 的大小
    int64_t nLoopSize = CeilDiv(constInfo.gSizeQueryIndex, nSplitSize);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
    for (int64_t n = 0; n < nLoopSize; n++) {
        int64_t bmm2Offset = n * nSplitSize * constInfo.kSize; // 4 * 2048
        DataCopyExtParams copyBmm2Params = {static_cast<uint16_t>(nSplitSize), static_cast<uint32_t>(realKSize * sizeof(T)), static_cast<uint32_t>((constInfo.kSize - realKSize) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> copyBmm2PadParams = {true, 0, (uint8_t)(realKSizeAlign - realKSize), 0.0};
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
        AscendC::DataCopyPad<T>(reluResUb, bmm2ResGm[bmm2ResOffset + bmm2Offset], copyBmm2Params, copyBmm2PadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4SY);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4SY);
        int64_t weightOffset = !constInfo.subBlockIdx * AlignTo(constInfo.gSizeQueryIndex, static_cast<uint32_t>(16));
        for (int32_t i = 0; i < nSplitSize; i++) {
            float weightValue = weightUb[weightOffset].GetValue(n*nSplitSize + i);
            int64_t mulOffset = i * realKSizeAlign;
            event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
            AscendC::Muls<T>(reluResUb[mulOffset], reluResUb[mulOffset], weightValue, realKSizeAlign);
        }
        PipeBarrier<PIPE_V>();
        
        LocalTensor<T> reduceUb = nSplitSize > 1 ? reduceSumYResTmpBuffer : reluResUb;
        if (nSplitSize > 1) {
            // 使用高阶API做自选维度的reduce 不支持源操作数与目的操作数地址重叠。不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
            uint32_t reduceShape[] = { static_cast<uint32_t>(nSplitSize), static_cast<uint32_t>(realKSizeAlign) };
            constexpr bool isReuse = true;
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, isReuse>(reduceUb, reluResUb, reduceSumTmpBuffer, reduceShape, true);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
            PipeBarrier<PIPE_V>();
        }

        if (n == 0) {
            AscendC::DataCopy(reduceSumYResUb, reduceUb, realKSizeAlign);
        } else {
            AscendC::Add(reduceSumYResUb, reduceSumYResUb, reduceUb, realKSizeAlign);
        }

        if (nSplitSize == 1) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
        }
        PipeBarrier<PIPE_V>();
    }

    SoftMaxShapeInfo softmaxShapeInfoData = {
        static_cast<uint32_t>(1),
        static_cast<uint32_t>(realKSizeAlign),
        static_cast<uint32_t>(1),
        static_cast<uint32_t>(realKSize),
    };
    SoftMaxTiling tilingInfo = tilingData->vectorParams.softmaxYTilingData;
    AscendC::SoftMax<T>(reduceSumYResUb, reduceSumYResUb, softmaxTmpBuffer, tilingInfo,
                        {static_cast<uint32_t>(1), static_cast<uint32_t>(realKSizeAlign),
                         static_cast<uint32_t>(1), static_cast<uint32_t>(realKSize)});
    PipeBarrier<PIPE_V>();
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(realKSize * sizeof(float)), 0, 0, 0}; 
    event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3);
    AscendC::DataCopyPad(psySyncGm[constInfo.kSize], reduceSumYResUb, copyParams);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::ReLUGrad(LocalTensor<KV_T> &reluGradOutTensor, LocalTensor<T> &ReLuTensor,
    LocalTensor<T> &subResTensor, int32_t kRealSize, int32_t kRealSizeAlign8, SLIGradKLLossRunInfo &runInfo)
{
    // maskTensor需要创建  subResTensor：mul之后的结果
    LocalTensor<T> maskUb = this->maskBuf.template Get<T>();
    CompareScalar(maskUb, subResTensor, static_cast<T>(0.0), AscendC::CMPMODE::GT, K_BASE_SIZE);
    PipeBarrier<PIPE_V>();
    //UB 新建了一个 maskBuffer
    Select(ReLuTensor, maskUb, ReLuTensor, static_cast<T>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, kRealSizeAlign8);
    PipeBarrier<PIPE_V>();
    Cast(reluGradOutTensor, ReLuTensor, AscendC::RoundMode::CAST_ROUND, kRealSizeAlign8);
    PipeBarrier<PIPE_V>();
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::DataCopyReluRes(LocalTensor<T> &reluResTensor, GlobalTensor<T> &reluGm,
    int32_t kRealSize, int32_t kRealSizeAlign8, int64_t reluResOffset)
{
    constexpr static int32_t blockSize = blockBytes / sizeof(T);
    if (likely(kRealSizeAlign8 == kRealSize)) {
        DataCopy(reluResTensor, reluGm[reluResOffset], kRealSize);
    } else {
        DataCopyParams dataCopyParams;
        DataCopyPadParams dataCopyPadParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = 0;
        dataCopyParams.blockLen = kRealSize * sizeof(T);
        dataCopyPadParams.rightPadding = kRealSizeAlign8 - kRealSize;
        dataCopyPadParams.paddingValue = 0;
        DataCopyPad(reluResTensor, reluGm[reluResOffset], dataCopyParams,
                    dataCopyPadParams);
    }
}

template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorDwDqDk(SLIGradKLLossRunInfo &runInfo, int32_t kLoopIdx)
{
    LocalTensor<T> v1TmpUb = v1TmpTBuf.template Get<T>();
    LocalTensor<T> reluResUbPingPong[2] = {mm2TBuf.template Get<T>(), mm2TBuf.template Get<T>()[2048]};
    LocalTensor<T> reduceSumPUbSingleK, reduceSumYUbSingleK;
    LocalTensor<T> weightTensor = weightTBuf.template Get<T>();
    LocalTensor<KV_T> reluGradUbPingPong[2] = {mm2TBuf.template Get<KV_T>()[8192], mm2TBuf.template Get<KV_T>()[10240]};
    LocalTensor<uint8_t> tmpUb = sharedTBuf.template Get<uint8_t>()[8192];
    LocalTensor<T> mulLeftUb = sharedTBuf.template Get<T>();
    LocalTensor<T> reduceSumPrevResUb = this->dwTBuf.template Get<T>();
    LocalTensor<KV_T> dwOutTensor = this->dwTBuf.template Get<KV_T>();
    event_t vToMte2PingPong[2] = {event_t(6), event_t(7)};
    event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    event_t mte3ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    event_t mte3ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    event_t mte3ToVPingPong[2] = {mte3ToVPing, mte3ToVPong};

    int64_t kLoopOffset = kLoopIdx * K_BASE_SIZE;
    int32_t kRealSize = (kLoopIdx == runInfo.kLoopTimes - 1) ? runInfo.kTailSize : K_BASE_SIZE;
    int32_t kRealSizeAlign8 = BlockAlign<T>(kRealSize);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
    if (runInfo.calcP) {
        reduceSumPUbSingleK = resPSYTBuf.Get<T>()[kLoopOffset];
        reduceSumYUbSingleK = mm2TBuf.template Get<T>()[6144];
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
        DataCopyReluRes(reduceSumYUbSingleK, psySyncOtherGm, kRealSize, kRealSizeAlign8, constInfo.kSize + kLoopOffset);
    } else {
        reduceSumPUbSingleK = mm2TBuf.template Get<T>()[6144];
        reduceSumYUbSingleK = resPSYTBuf.Get<T>()[kLoopOffset];
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
        DataCopyReluRes(reduceSumPUbSingleK, psySyncOtherGm, kRealSize, kRealSizeAlign8, kLoopOffset);
    }

    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);

    Sub(v1TmpUb, reduceSumYUbSingleK, reduceSumPUbSingleK, kRealSize);
    PipeBarrier<PIPE_V>();
    SumParams sumParams;
    sumParams.outter = 1;
    sumParams.inner = kRealSizeAlign8;
    sumParams.n = kRealSize;

    DataCopyParams dataCopyReluGradParams;
    dataCopyReluGradParams.blockLen = kRealSize * sizeof(KV_T);
    dataCopyReluGradParams.srcStride = 0;
    dataCopyReluGradParams.blockCount = 1;
    dataCopyReluGradParams.dstStride = 0;

    int64_t subBlockGQueryIndexOffset = constInfo.subBlockIdx * (constInfo.gSizeQueryIndex >> 1);
    int64_t reluGmOffset = (runInfo.taskIdMod2 * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset) * constInfo.kSize + kLoopOffset;
    int64_t weightOffset = runInfo.taskIdMod2 * AlignTo(constInfo.gSizeQueryIndex, static_cast<uint32_t>(16));
    SetFlag<HardEvent::MTE3_V>(mte3ToVPingPong[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVPingPong[1]);
    SetFlag<HardEvent::V_MTE2>(vToMte2PingPong[0]);
    SetFlag<HardEvent::V_MTE2>(vToMte2PingPong[1]);
    for (size_t nIdx = 0; nIdx < constInfo.gSizeQueryIndex >> 1; nIdx++) {
        uint8_t pingPong = nIdx & 1;
        WaitFlag<HardEvent::V_MTE2>(vToMte2PingPong[pingPong]);
        int64_t reluResOffset = reluGmOffset + nIdx * constInfo.kSize; // ((kRealSize + 15) / 16 * 16);
        DataCopyReluRes(reluResUbPingPong[pingPong], reluGm, kRealSize, kRealSizeAlign8, reluResOffset);

        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
        Mul(mulLeftUb, v1TmpUb, reluResUbPingPong[pingPong], kRealSizeAlign8);
        PipeBarrier<PIPE_V>();

        if (kLoopIdx == 0) {
            // 首轮直接输出到dwUb上
            AscendC::Sum(reduceSumPrevResUb[nIdx * 16], mulLeftUb, tmpUb, sumParams);
        } else {
            AscendC::Sum(reduceSumDwResUb[nIdx * 16], mulLeftUb, tmpUb, sumParams);
        }
        PipeBarrier<PIPE_V>();

        Muls(mulLeftUb, v1TmpUb, weightTensor.GetValue(nIdx + weightOffset + subBlockGQueryIndexOffset), kRealSizeAlign8);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVPingPong[pingPong]);
        ReLUGrad(reluGradUbPingPong[pingPong], mulLeftUb, reluResUbPingPong[pingPong], kRealSize, kRealSizeAlign8, runInfo);
        SetFlag<HardEvent::V_MTE2>(vToMte2PingPong[pingPong]);

        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);
        DataCopyPad(reluGradResGm[reluResOffset], reluGradUbPingPong[pingPong], dataCopyReluGradParams);
        SetFlag<HardEvent::MTE3_V>(mte3ToVPingPong[pingPong]);
    }
    if (kLoopIdx > 0) {
        PipeBarrier<PIPE_V>();
        Add(reduceSumPrevResUb, reduceSumPrevResUb, reduceSumDwResUb, constInfo.gSizeQueryIndex * 16);
    }
    WaitFlag<HardEvent::MTE3_V>(mte3ToVPingPong[0]);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVPingPong[1]);
    WaitFlag<HardEvent::V_MTE2>(vToMte2PingPong[0]);
    WaitFlag<HardEvent::V_MTE2>(vToMte2PingPong[1]);

    if (kLoopIdx == runInfo.kLoopTimes - 1) {
        PipeBarrier<PIPE_V>();
        Cast(dwOutTensor, reduceSumPrevResUb, AscendC::RoundMode::CAST_ROUND, constInfo.gSizeQueryIndex * 16);
        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);

        // 拷出到GM
        DataCopyParams dataCopyParams;
        dataCopyParams.blockLen = sizeof(KV_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.blockCount =  constInfo.gSizeQueryIndex >> 1;
        dataCopyParams.dstStride = 0;
        if constexpr (LAYOUT_T == SLILayout::BSND) {
            DataCopyPad(dWeightGm[runInfo.bIdx * (constInfo.s1Size * constInfo.gSizeQueryIndex) +
                runInfo.s1Idx * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset], dwOutTensor, dataCopyParams);
        } else if constexpr (LAYOUT_T == SLILayout::TND) {
            DataCopyPad(this->dWeightGm[runInfo.accumS1Idx * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset], dwOutTensor, dataCopyParams);
        }
    }

    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToVPing);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToVPong);
}


template <typename SLIT> 
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorLoss(SLIGradKLLossRunInfo &runInfo, int32_t kLoopIdx)
{
    bool isLastBasicBlock = kLoopIdx == runInfo.kLoopTimes - 1;
    int64_t realKSize = isLastBasicBlock ? runInfo.kTailSize : K_BASE_SIZE;
    int32_t v0RealKSizeAlign = BlockAlign<T>(CeilDiv(realKSize, 2));
    int32_t v0RealKSize = realKSize;
    if (realKSize > 8) {
        v0RealKSize = v0RealKSizeAlign;
    }
    int32_t v1RealKSize = Max(realKSize - v0RealKSizeAlign, 0);
    int32_t v1RealKSizeAlign = BlockAlign<T>(v1RealKSize);

    int32_t vRealKSize, vRealKSizeAlign;
    int64_t kLoopOffset = kLoopIdx * K_BASE_SIZE;
    int64_t coreOffset = 0;
    if (constInfo.subBlockIdx == 0) {
        vRealKSize = v0RealKSize;
        vRealKSizeAlign = v0RealKSizeAlign;
    } else {
        vRealKSize = v1RealKSize;
        vRealKSizeAlign = v1RealKSizeAlign;
        coreOffset += v0RealKSizeAlign;
    }

    int64_t calcSize = vRealKSize;
    int64_t calcSizeAlign = vRealKSizeAlign;
    const float MIN_VALUE = 1e-8;
    LocalTensor<T> reduceSumPTmpBuffer = sharedTBuf.GetWithOffset<T>(4 * 1024, 0);
    LocalTensor<uint8_t> sharedTmpBuffer = sharedTBuf.GetWithOffset<uint8_t>(8 * 1024, 4 * 1024);
    LocalTensor<T> vecTmpUb;
    LocalTensor<T> lossSumUb;
    LocalTensor<T> tmpLossUb;
    if (constInfo.subBlockIdx) {
        vecTmpUb = v1TmpTBuf.GetWithOffset<T>(4 * 1024, 4 * 1024);
        lossSumUb = lossSumTBuf.GetWithOffset<T>(8, 32);
        tmpLossUb = sharedTBuf.GetWithOffset<T>(8, 12 * 1024 + 32);
    } else {
        vecTmpUb = v1TmpTBuf.GetWithOffset<T>(4 * 1024, 0);
        lossSumUb = lossSumTBuf.GetWithOffset<T>(8, 0);
        tmpLossUb = sharedTBuf.GetWithOffset<T>(8, 12 * 1024);
    }

    if (vRealKSize != 0) {
        LocalTensor<T> reduceSumPTmpUb, reduceSumYResTmpUb;
        if (runInfo.calcP) {  // 当前核做的是VectorP
            reduceSumPTmpUb = resPSYTBuf.Get<T>();
            reduceSumPTmpUb = reduceSumPTmpUb[kLoopOffset + coreOffset];
            reduceSumYResTmpUb = mm2TBuf.template Get<T>()[6144 + coreOffset];
        } else {  // 当前核做的是VectorSY
            reduceSumPTmpUb = mm2TBuf.template Get<T>()[6144 + coreOffset];
            reduceSumYResTmpUb = resPSYTBuf.Get<T>();
            reduceSumYResTmpUb = reduceSumYResTmpUb[kLoopOffset + coreOffset];
        }
        PipeBarrier<PIPE_V>();
        AscendC::Maxs<T>(reduceSumPTmpUb, reduceSumPTmpUb, MIN_VALUE, calcSize);
        PipeBarrier<PIPE_V>();
        AscendC::Maxs<T>(reduceSumYResTmpUb, reduceSumYResTmpUb, MIN_VALUE, calcSize);
        PipeBarrier<PIPE_V>();
        
        AscendC::Log(reduceSumPTmpBuffer, reduceSumPTmpUb, calcSize);
        AscendC::Log(reduceSumYResTmpUb, reduceSumYResTmpUb, calcSize);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(vecTmpUb, reduceSumPTmpBuffer, reduceSumYResTmpUb, calcSize);
        PipeBarrier<PIPE_V>();
        AscendC::Mul(vecTmpUb, vecTmpUb, reduceSumPTmpUb, calcSize);
        PipeBarrier<PIPE_V>();
        if (kLoopIdx == 0) {
            AscendC::Sum(lossSumUb, vecTmpUb, sharedTmpBuffer,
                    {static_cast<uint32_t>(1), static_cast<uint32_t>(calcSizeAlign), static_cast<uint32_t>(calcSize)});
            PipeBarrier<PIPE_V>();
        } else {
            AscendC::Sum(tmpLossUb, vecTmpUb, sharedTmpBuffer,
                    {static_cast<uint32_t>(1), static_cast<uint32_t>(calcSizeAlign), static_cast<uint32_t>(calcSize)});
            PipeBarrier<PIPE_V>();
            //与之前基本块的loss相加
            AscendC::Add(lossSumUb, lossSumUb, tmpLossUb, 8);
            PipeBarrier<PIPE_V>();
        }
    }

    if (isLastBasicBlock && (runInfo.kLoopTimes > 1 || vRealKSize != 0)) {
        event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMTE3);
        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopyPad(lossGm, lossSumUb,
                            {static_cast<uint32_t>(1), static_cast<uint32_t>(4),
                            static_cast<uint32_t>(0), static_cast<uint32_t>(0)});
        SetAtomicNone();
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMTE3);
    }
}

#endif // SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR_H
