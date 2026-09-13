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
 * \file sparse_flash_attention_grad_post.h
 * \brief common post process
 */

#pragma once
#include "kernel_operator.h"
using namespace AscendC;
namespace SFAG_KEY_OWNED_FULL_REDUCE {

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const bool HAS_ROPE>
class SparseFlashAttentionGradPost {
public:
    __aicore__ inline SparseFlashAttentionGradPost(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
                                __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                                __gm__ uint8_t *dq_rope, __gm__ uint8_t *dk_rope, __gm__ uint8_t *workspace,
                                const TILING_TYPE *__restrict ordTilingData, TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void InitIndex(uint64_t startIdx, int64_t curG, int64_t &curS, int64_t headDim,
                                     int64_t headDimAlign, GM_ADDR seqS);
    __aicore__ inline void NZ2ND(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, uint64_t sLen,
                                 uint64_t ubOffset, uint64_t srcUbOffset, int64_t headDimAlign);
    __aicore__ inline void NZVecClc(GlobalTensor<float> srcGm, GlobalTensor<OUT_TYPE> dstGm, uint64_t dataSize,
                                    GM_ADDR seqS, int64_t curG, int64_t &curS, int64_t headDim, int64_t headDimAlign,
                                    bool needMuls, int64_t flag);
    __aicore__ inline void NZProcess();
    __aicore__ inline void ComputeDataCopyOffset(int64_t curG, int64_t &curS, int64_t headDim, int64_t headDimAlign);

    constexpr static uint32_t BUFFER_NUM = 1;
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    // NZ buffer
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePing;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePong;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuePing;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuePong;
    TBuf<> tmpBufPing;
    TBuf<> tmpBufPong;

    AscendC::GlobalTensor<OUT_TYPE> dqGm;
    AscendC::GlobalTensor<OUT_TYPE> dkGm;
    AscendC::GlobalTensor<OUT_TYPE> dvGm;
    AscendC::GlobalTensor<OUT_TYPE> dqRopeGm;
    AscendC::GlobalTensor<OUT_TYPE> dkRopeGm;
    // input
    AscendC::GlobalTensor<float> dqWorkSpaceGm;
    AscendC::GlobalTensor<float> dkWorkSpaceGm;
    AscendC::GlobalTensor<float> dvWorkSpaceGm;

    const TILING_TYPE *__restrict tilingData;
    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint64_t C0_SIZE = 16;
    constexpr static uint64_t VEC_REPEAT = 8;
    constexpr static uint32_t cal_block_num = 32 / sizeof(float);

    int64_t usedCoreNum;
    int64_t cBlockIdx;
    // query
    int64_t ubBaseSize;
    uint32_t nzReservedSize;
    int64_t qPostBlockFactor;
    uint64_t qPostBlockTotal;
    int64_t qPostBaseNum;
    int64_t qPostTailNum;
    uint64_t qSizeAlign;
    int64_t kPostBlockFactor;
    uint64_t kPostBlockTotal;
    int64_t kPostBaseNum;
    int64_t kPostTailNum;
    uint64_t kSizeAlign;
    int64_t vPostBlockFactor;
    uint64_t vPostBlockTotal;
    int64_t vPostBaseNum;
    int64_t vPostTailNum;
    uint64_t vSizeAlign;

    // org shape info
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t d2;
    int64_t dAlign;
    int64_t d2Align;
    int64_t dimDqk;
    int64_t dimRope;

    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t TND = 3;

    constexpr static uint32_t ND = 0;
    constexpr static uint32_t NZ = 1;

    GM_ADDR actual_seq_qlen_addr;
    GM_ADDR actual_seq_kvlen_addr;

    uint64_t bIdx;
    uint64_t nIdx;
    uint64_t sIdx;

    uint64_t scrOffsetBase = 0;
    uint64_t dstOffsetBase = 0;
    uint64_t copyInSrcOffset = 0;
    uint64_t copyOutDstOffset = 0;
    uint64_t copyOutDstStride = 0;
};

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const bool HAS_ROPE>
__aicore__ inline void
SparseFlashAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, HAS_ROPE>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *actual_seq_qlen,
    __gm__ uint8_t *actual_seq_kvlen, __gm__ uint8_t *dq_rope, __gm__ uint8_t *dk_rope, __gm__ uint8_t *workspace,
    const TILING_TYPE *__restrict ordTilingData, TPipe *pipe_in)
{
    cBlockIdx = GetBlockIdx();

    tilingData = ordTilingData;
    pipe = pipe_in;

    dqGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dq);
    dkGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dk);
    dvGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dv);
    dqRopeGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dq_rope);
    dkRopeGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dk_rope);

    // tiling_data
    usedCoreNum = tilingData->postTilingData.coreNum;
    ubBaseSize = tilingData->postTilingData.postUbBaseSize;
    nzReservedSize = tilingData->postTilingData.nzReservedSize;
    qPostBlockFactor = tilingData->postTilingData.qPostBlockFactor;
    qPostBlockTotal = tilingData->postTilingData.qPostBlockTotal;
    qPostBaseNum = tilingData->postTilingData.qPostBaseNum;
    qPostTailNum = tilingData->postTilingData.qPostTailNum;
    kPostBlockFactor = tilingData->postTilingData.kPostBlockFactor;
    kPostBlockTotal = tilingData->postTilingData.kPostBlockTotal;
    kPostBaseNum = tilingData->postTilingData.kPostBaseNum;
    kPostTailNum = tilingData->postTilingData.kPostTailNum;
    vPostBlockFactor = tilingData->postTilingData.vPostBlockFactor;
    vPostBlockTotal = tilingData->postTilingData.vPostBlockTotal;
    vPostBaseNum = tilingData->postTilingData.vPostBaseNum;
    vPostTailNum = tilingData->postTilingData.vPostTailNum;
    qSizeAlign = tilingData->postTilingData.qSizeAlign;
    kSizeAlign = tilingData->postTilingData.kSizeAlign;
    vSizeAlign = tilingData->postTilingData.vSizeAlign;

    dimDqk = tilingData->opInfo.D;
    dimRope = tilingData->opInfo.ropeD;



    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
    if constexpr (CAST_DV) {
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      tilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
    }


        pipe->InitBuffer(inQueue, 1, ubBaseSize * 2);
        pipe->InitBuffer(outQueue, 1, ubBaseSize);

}











template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const bool HAS_ROPE>
__aicore__ inline void
SparseFlashAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, HAS_ROPE>::Process()
{

    uint64_t totalD = dimDqk + dimRope;
    // init q
    uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
    uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;

    if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
        qEnd = qPostBlockTotal;
    }
    uint64_t dqOutGmOffset = cBlockIdx * qPostBlockFactor * (qPostBaseNum / totalD) * dimDqk;
    uint64_t dqRopeOutGmOffset = cBlockIdx * qPostBlockFactor * (qPostBaseNum / totalD) * dimRope;

    SetFlag<HardEvent::V_MTE2>(0);
    for (uint64_t i = qBegin; i < qEnd; i = i + qPostBaseNum) {
        AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
        AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
        uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
        WaitFlag<HardEvent::V_MTE2>(0);
        DataCopy(vecIn, dqWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
        inQueue.EnQue(vecIn);
        inQueue.template DeQue<float>();

        if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
            Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();

            DataCopy(dqGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
        } else {
            Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
            PIPE_BARRIER(PIPE_V);

            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();

            DataCopyParams repeatParams;
            repeatParams.blockCount = dataSize / totalD;
            repeatParams.blockLen = dimDqk * sizeof(OUT_TYPE) / 32;
            repeatParams.srcStride = dimRope * sizeof(OUT_TYPE) / 32;
            repeatParams.dstStride = 0;

            DataCopy(dqGm[dqOutGmOffset], vecOut, repeatParams);
            dqOutGmOffset += repeatParams.blockCount * dimDqk;

            if constexpr (HAS_ROPE) {
                // if ROPE_ENABLED: scatter --> dq & dq_rope
                repeatParams.blockLen = dimRope * sizeof(OUT_TYPE) / 32;
                repeatParams.srcStride = dimDqk * sizeof(OUT_TYPE) / 32;
                DataCopy(dqRopeGm[dqRopeOutGmOffset], vecOut[dimDqk], repeatParams);
                dqRopeOutGmOffset += repeatParams.blockCount * dimRope;
            }
        }
        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
        SetFlag<HardEvent::V_MTE2>(0);
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    PIPE_BARRIER(PIPE_ALL);
    // init k
    uint64_t kBegin = cBlockIdx * kPostBlockFactor * kPostBaseNum;
    uint64_t kEnd = (cBlockIdx + 1) * kPostBlockFactor * kPostBaseNum;
    if (((cBlockIdx + 1) * kPostBlockFactor * kPostBaseNum) > kPostBlockTotal) {
        kEnd = kPostBlockTotal;
    }
    uint64_t dkOutGmOffset = cBlockIdx * kPostBlockFactor * (kPostBaseNum / totalD) * dimDqk;
    uint64_t dkRopeOutGmOffset = cBlockIdx * kPostBlockFactor * (kPostBaseNum / totalD) * dimRope;

    SetFlag<HardEvent::V_MTE2>(0);
    for (uint64_t i = kBegin; i < kEnd; i = i + kPostBaseNum) {
        AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
        AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
        uint64_t dataSize = i + kPostBaseNum < kPostBlockTotal ? kPostBaseNum : kPostTailNum;
        WaitFlag<HardEvent::V_MTE2>(0);
        DataCopy(vecIn, dkWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
        inQueue.EnQue(vecIn);
        inQueue.template DeQue<float>();
        if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
            PIPE_BARRIER(PIPE_ALL);
            Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();
            PIPE_BARRIER(PIPE_ALL);
            DataCopy(dkGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
        } else {
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();

            DataCopyParams repeatParams;
            repeatParams.blockCount = dataSize / totalD;
            repeatParams.blockLen = dimDqk * sizeof(OUT_TYPE) / 32;
            repeatParams.srcStride = dimRope * sizeof(OUT_TYPE) / 32;
            repeatParams.dstStride = 0;

            DataCopy(dkGm[dkOutGmOffset], vecOut, repeatParams);
            dkOutGmOffset += repeatParams.blockCount * dimDqk;

            if constexpr (HAS_ROPE) {
                // if ROPE_ENABLED: scatter --> dk & dk_rope
                repeatParams.blockLen = dimRope * sizeof(OUT_TYPE) / 32;
                repeatParams.srcStride = dimDqk * sizeof(OUT_TYPE) / 32;
                DataCopy(dkRopeGm[dkRopeOutGmOffset], vecOut[dimDqk], repeatParams);
                dkRopeOutGmOffset += repeatParams.blockCount * dimRope;
            }
        }
        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
        SetFlag<HardEvent::V_MTE2>(0);
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    PIPE_BARRIER(PIPE_ALL);

    // init v
    if constexpr (CAST_DV && !AscendC::IsSameType<OUT_TYPE, float>::value) {
        uint64_t vBegin = cBlockIdx * vPostBlockFactor * vPostBaseNum;
        uint64_t vEnd = (cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum;
        if (((cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum) > vPostBlockTotal) {
            vEnd = vPostBlockTotal;
        }

        SetFlag<HardEvent::V_MTE2>(0);
        for (uint64_t i = vBegin; i < vEnd; i = i + vPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
            uint64_t dataSize = i + vPostBaseNum < vPostBlockTotal ? vPostBaseNum : vPostTailNum;
            WaitFlag<HardEvent::V_MTE2>(0);
            DataCopy(vecIn, dvWorkSpaceGm[i], (dataSize + 7) / 8 * 8);
            inQueue.EnQue(vecIn);
            inQueue.template DeQue<float>();

            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();

            DataCopy(dvGm[i], vecOut, (dataSize + 15) / 16 * 16);
            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
            SetFlag<HardEvent::V_MTE2>(0);
        }
        WaitFlag<HardEvent::V_MTE2>(0);
    }
}

} // namespace SFAG_KEY_OWNED_FULL_REDUCE
