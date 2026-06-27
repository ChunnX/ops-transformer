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
 * \file bsa_select_block_mask_vector.h
 * \brief
 */
#ifndef BSA_SELECT_BLOCK_MASK_VECTOR_H
#define BSA_SELECT_BLOCK_MASK_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bsa_select_block_mask_common.h"
#include "bsa_select_block_mask_tiling_data.h"
#include "bsa_select_block_mask_pool.h"
#include "bsa_select_block_mask_softmax.h"

template <typename BSAT>
class BSAVectorService {
public:
    using T = float;
    using IN_T = typename BSAT::inputT;
    using OUT_T = typename BSAT::outputT;

    static constexpr BSALayout LAYOUT_Q = BSAT::layoutQ;
    static constexpr BSALayout LAYOUT_KV = BSAT::layoutKV;

    __aicore__ inline BSAVectorService() {};
    __aicore__ inline void InitParams(const BSAConstInfo &constInfo,
                                      const optiling::BSASelectBlockMaskTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitGM(GlobalTensor<IN_T> &qCmpGm, GlobalTensor<IN_T> &kCmpGm,
                                   GlobalTensor<half> &attnScorFp16eGm, GlobalTensor<T> &ScoreFp32Gm,
                                   GlobalTensor<IN_T> &queryGm, GlobalTensor<IN_T> &keyGm,
                                   GlobalTensor<int64_t> &actualBlockLenQGm,
                                   GlobalTensor<int64_t> &actualBlockLenKVGm,
                                   GlobalTensor<T> &qCmpOutGm, GlobalTensor<T> &kCmpOutGm, GlobalTensor<half> &softmaxOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    __aicore__ inline void PoolingSingleQBlock(uint32_t batchIdx, uint32_t headIdx, uint32_t qBlockIdx,
                                                GlobalTensor<IN_T> &queryGm,
                                                GlobalTensor<int64_t> &actualBlockLenQGm,
                                                GlobalTensor<IN_T> &qCmpGm,
                                                GlobalTensor<T> &qCmpOutGm);
    __aicore__ inline void PoolingSingleKBlock(uint32_t batchIdx, uint32_t headIdx, uint32_t kBlockIdx,
                                                GlobalTensor<IN_T> &keyGm,
                                                GlobalTensor<int64_t> &actualBlockLenKVGm,
                                                GlobalTensor<IN_T> &kCmpGm,
                                                GlobalTensor<T> &kCmpOutGm);
    __aicore__ inline void OnlineSoftmaxFirstPassChunk(uint32_t qChunkStart, uint32_t qChunkSize,
                                                        uint32_t kChunkStart, uint32_t kChunkSize);
    __aicore__ inline void SoftmaxSecondPassAndCast(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart, uint32_t kChunkSize, uint32_t batchIdx, uint32_t headIdx);

private:
    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData;

    GlobalTensor<IN_T> qCmpGmLocal, kCmpGmLocal;
    GlobalTensor<IN_T> attnScoreGmLocal;
    GlobalTensor<T> softmaxTmpGmLocal;
    GlobalTensor<IN_T> queryGmLocal, keyGmLocal;
    GlobalTensor<int64_t> actualBlockLenQGmLocal, actualBlockLenKVGmLocal;

    TBuf<> uBuf_;

    LocalTensor<IN_T> poolInputUb_;
    LocalTensor<T> poolAccumUb_;
    LocalTensor<IN_T> poolOutputUb_;
    LocalTensor<T> scoreUb_;
    LocalTensor<T> softmaxTmpUb_;
    LocalTensor<T> castOutputUb_;
    LocalTensor<T> mScalarUb_;
    LocalTensor<T> lScalarUb_;

    event_t eventIdMte2ToV;
    event_t eventIdVToMte3;
    event_t eventIdMte3ToV;

    // vector op
    BSAVecPoolService<BSAT> poolOP;
    BSAVecSmService<BSAT> softmaxOP;
};

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::InitParams(const BSAConstInfo &constInfo,
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData)
{
    // this->constInfo = constInfo;
    // this->tilingData = tilingData;
    poolOP.InitParams(constInfo, tilingData);
    softmaxOP.InitParams(constInfo, tilingData);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(uBuf_, BSAConstInfo::BUFFER_SIZE_BYTE_192K);
    poolOP.InitBuffers(&uBuf_);
    softmaxOP.InitBuffers(&uBuf_);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::InitGM(
    GlobalTensor<IN_T> &qCmpGm, GlobalTensor<IN_T> &kCmpGm,
    GlobalTensor<half> &attnScorFp16eGm, GlobalTensor<T> &ScoreFp32Gm,
    GlobalTensor<IN_T> &queryGm, GlobalTensor<IN_T> &keyGm,
    GlobalTensor<int64_t> &actualBlockLenQGm, GlobalTensor<int64_t> &actualBlockLenKVGm,
    GlobalTensor<T> &qCmpOutGm, GlobalTensor<T> &kCmpOutGm, GlobalTensor<half> &softmaxOutGm)
{
    // this->qCmpGmLocal = qCmpGm;
    // this->kCmpGmLocal = kCmpGm;
    // this->attnScoreGmLocal = softmaxOutGm;
    // this->queryGmLocal = queryGm;
    // this->keyGmLocal = keyGm;
    // this->actualBlockLenQGmLocal = actualBlockLenQGm;
    // this->actualBlockLenKVGmLocal = actualBlockLenKVGm;

    poolOP.InitGM(qCmpGm, kCmpGm, queryGm, keyGm, actualBlockLenQGm, actualBlockLenKVGm, qCmpOutGm, kCmpOutGm);
    // FIX: softmaxOP.InitGM expects GlobalTensor<T> (FP32) for attnScorFp16eGm parameter
    // Pass ScoreFp32Gm (FP32 workspace) instead of attnScorFp16eGm (IN_T)
    softmaxOP.InitGM(ScoreFp32Gm, attnScorFp16eGm, softmaxOutGm);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::AllocEventID()
{
    // eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    // eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    // eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    // SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    // SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    // SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);

    poolOP.AllocEventID();
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::FreeEventID()
{
    poolOP.FreeEventID();
    // WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    // WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    // WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    // GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    // GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
    // GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::PoolingSingleQBlock(
    uint32_t batchIdx, uint32_t headIdx, uint32_t qBlockIdx,
    GlobalTensor<IN_T> &queryGm,
    GlobalTensor<int64_t> &actualBlockLenQGm,
    GlobalTensor<IN_T> &qCmpGm,
    GlobalTensor<T> &qCmpOutGm)
{
    // TODO: 单个query block的均值池化
    poolOP.PoolingSingleQBlock(batchIdx, headIdx, qBlockIdx, queryGm, actualBlockLenQGm, qCmpGm, qCmpOutGm);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::PoolingSingleKBlock(
    uint32_t batchIdx, uint32_t headIdx, uint32_t kBlockIdx,
    GlobalTensor<IN_T> &keyGm,
    GlobalTensor<int64_t> &actualBlockLenKVGm,
    GlobalTensor<IN_T> &kCmpGm,
    GlobalTensor<T> &kCmpOutGm)
{
    // TODO: 单个key block的均值池化（同PoolingSingleQBlock逻辑）
    poolOP.PoolingSingleKBlock(batchIdx, headIdx, kBlockIdx, keyGm, actualBlockLenKVGm, kCmpGm, kCmpOutGm);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::OnlineSoftmaxFirstPassChunk(
    uint32_t qChunkStart, uint32_t qChunkSize,
    uint32_t kChunkStart, uint32_t kChunkSize)
{
    softmaxOP.OnlineSoftmaxFirstPassChunk(qChunkStart, qChunkSize, kChunkStart, kChunkSize);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::SoftmaxSecondPassAndCast(
    uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart, uint32_t kChunkSize, uint32_t batchIdx, uint32_t headIdx)
{
    softmaxOP.SoftmaxSecondPassAndCast(qChunkStart, qChunkSize, kChunkStart, kChunkSize, batchIdx, headIdx);
}

#endif // BSA_SELECT_BLOCK_MASK_VECTOR_H
