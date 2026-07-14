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
 * \file apply_rotary_pos_emb_compute_ab_cast_perf.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_COMPUTE_AB_CAST_PERF_H
#define APPLY_ROTARY_POS_EMB_COMPUTE_AB_CAST_PERF_H

#include "kernel_operator.h"
#include "apply_rotary_pos_emb_base.h"

namespace ApplyRotaryPosEmb {
using namespace AscendC;

template <typename T1, typename T2>
class ARPEComputeABCastPerf : public ApplyRotaryPosEmbBase<T1> {
public:
    __aicore__ inline ARPEComputeABCastPerf(){};
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR workspace,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void Process(const ApplyRotaryPosEmbTilingData* tilingData);

private:
    __aicore__ inline void ProcessPreCore(const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ProcessLastCore(const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ProcessOneBatch(
        const int64_t coreBatchIndex, const int64_t batchB, const ApplyRotaryPosEmbTilingData* tilingData);

    __aicore__ inline void CopyInTensor(
        const int64_t coreBatchIndex, const int64_t preCBatchB, GlobalTensor<T1>& gm, const int64_t coreOffset,
        const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qUb,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void CopyOutTensor(
        const int64_t coreBatchIndex, const int64_t preCBatchB, GlobalTensor<T1>& gm, const int64_t coreOffset,
        const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qOutUb,
        const ApplyRotaryPosEmbTilingData* tilingData);

    __aicore__ inline void ComputeRotaryNoCast(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
        LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb, const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ComputeRotaryCast(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T2>& cosC1,
        LocalTensor<T2>& sinC1, LocalTensor<T1>& outUb, const ApplyRotaryPosEmbTilingData* tilingData);

    __aicore__ inline void SmallQC(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T2>& mul1Ub, LocalTensor<T2>& sinC1,
        LocalTensor<T2>& mul2Ub, LocalTensor<T2>& cosC1,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void LargeQC(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T2>& mul1Ub, LocalTensor<T2>& sinC1,
        LocalTensor<T2>& mul2Ub, LocalTensor<T2>& cosC1,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void SmallQCFP32(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& mul1Ub, LocalTensor<T1>& sinSize,
        LocalTensor<T1>& mul2Ub, LocalTensor<T1>& cosSize,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void LargeQCFP32(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& mul1Ub, LocalTensor<T1>& sinSize,
        LocalTensor<T1>& mul2Ub, LocalTensor<T1>& cosSize,
        const ApplyRotaryPosEmbTilingData* tilingData);
    template <typename T>
    __aicore__ inline void MulsNegFirstHalfSplit(LocalTensor<T>& sinTensor, T negScalar, int64_t batchB,
        const ApplyRotaryPosEmbTilingData* tilingData);

    constexpr static int32_t bufferNum = 1;
    constexpr static int32_t bufferNumdb = 2;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> qInQueue;
    TQue<QuePosition::VECIN, bufferNum> cosInQueue;
    TQue<QuePosition::VECIN, bufferNum> sinInQueue;
    TQue<QuePosition::VECOUT, bufferNum> qOutQueue;
    TBuf<QuePosition::VECCALC> mul1;
    TBuf<QuePosition::VECCALC> mul2;
#if ORIG_DTYPE_QUERY != DT_FLOAT
    TBuf<QuePosition::VECCALC> cosC;
    TBuf<QuePosition::VECCALC> sinC;
#endif

    GlobalTensor<T1> qGm;
    GlobalTensor<T1> kGm;
    GlobalTensor<T1> cosGm;
    GlobalTensor<T1> sinGm;
    UnaryRepeatParams mulRepeatP;
    BinaryRepeatParams repeatParams;
    uint64_t blockIdx = 0;
    uint64_t oneBlock = 0;
};

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::Init(
    GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR workspace,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    qGm.SetGlobalBuffer((__gm__ T1*)q);
    kGm.SetGlobalBuffer((__gm__ T1*)k);
    cosGm.SetGlobalBuffer((__gm__ T1*)cos);
    sinGm.SetGlobalBuffer((__gm__ T1*)sin);
    pipe.InitBuffer(cosInQueue, bufferNumdb, tilingData->cosPart1Ub);
    pipe.InitBuffer(qInQueue, bufferNumdb, tilingData->qPart1Ub);
    pipe.InitBuffer(sinInQueue, bufferNumdb, tilingData->cosPart1Ub);
    pipe.InitBuffer(mul1, tilingData->q2q1Part1Ub);
    pipe.InitBuffer(qOutQueue, bufferNumdb, tilingData->qPart1Ub);
    pipe.InitBuffer(mul2, tilingData->q2q1Part1Ub);
#if ORIG_DTYPE_QUERY != DT_FLOAT
    pipe.InitBuffer(cosC, tilingData->sin1UbSize);
    pipe.InitBuffer(sinC, tilingData->sin1UbSize);
#endif

    mulRepeatP.dstBlkStride = 1;
    mulRepeatP.srcBlkStride = 1;
    mulRepeatP.dstRepStride = tilingData->dstRepSBr;
    mulRepeatP.srcRepStride = tilingData->dstRepSBr;

    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;

    oneBlock = tilingData->lastDim / tilingData->blockMoveQ;
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::SmallQC(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T2>& mul1Ub, LocalTensor<T2>& sinC1,
    LocalTensor<T2>& mul2Ub, LocalTensor<T2>& cosC1, const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t fullMask = tilingData->mask;
    int64_t fullChunks = (tilingData->lastDim + fullMask - 1) / fullMask;
    for (int64_t ii = 0; ii < cNum; ii++) {
        for (int32_t j = 0; j < fullChunks; j++) {
            int64_t offset = j * fullMask;
            int64_t curMask = (offset + fullMask > tilingData->lastDim) ? (tilingData->lastDim - offset) : fullMask;
            int64_t remainB = preCBatchB;
            int64_t baseB = 0;
            while (remainB > 0) {
                uint8_t curRep = (remainB > UINT8_MAX) ? UINT8_MAX : static_cast<uint8_t>(remainB);
                int64_t off  = ii * tilingData->lastDim + offset + baseB * cNum * tilingData->lastDim;
                int64_t soff = offset + baseB * tilingData->lastDim;
                Mul<T2>(mul1Ub[off], mul1Ub[off], sinC1[soff], curMask, curRep, repeatParams);
                Mul<T2>(mul2Ub[off], mul2Ub[off], cosC1[soff], curMask, curRep, repeatParams);
                remainB -= curRep;
                baseB += curRep;
            }
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::LargeQC(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T2>& mul1Ub, LocalTensor<T2>& sinC1,
    LocalTensor<T2>& mul2Ub, LocalTensor<T2>& cosC1, const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t fullMask = tilingData->mask;
    int64_t fullChunks = (tilingData->lastDim + fullMask - 1) / fullMask;
    int64_t batchStride = cNum * tilingData->lastDim;
    for (int64_t i = 0; i < preCBatchB; i++) {
        for (int32_t j = 0; j < fullChunks; j++) {
            int64_t offset = j * fullMask;
            int64_t curMask = (offset + fullMask > tilingData->lastDim) ? (tilingData->lastDim - offset) : fullMask;
            int64_t off = i * batchStride + offset;
            int64_t sOff = i * tilingData->lastDim + offset;
            Mul<T2>(mul1Ub[off], mul1Ub[off], sinC1[sOff], curMask, cNum, repeatParams);
            Mul<T2>(mul2Ub[off], mul2Ub[off], cosC1[sOff], curMask, cNum, repeatParams);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::SmallQCFP32(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& mul1Ub, LocalTensor<T1>& sinSize,
    LocalTensor<T1>& mul2Ub, LocalTensor<T1>& cosSize,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t fullMask = tilingData->mask;
    int64_t fullChunks = (tilingData->lastDim + fullMask - 1) / fullMask;
    for (int64_t ii = 0; ii < cNum; ii++) {
        for (int32_t j = 0; j < fullChunks; j++) {
            int64_t offset = j * fullMask;
            int64_t curMask = (offset + fullMask > tilingData->lastDim) ? (tilingData->lastDim - offset) : fullMask;
            int64_t off = ii * tilingData->lastDim + offset;
            int64_t remainB = preCBatchB;
            int64_t baseB = 0;
            while (remainB > 0) {
                uint8_t curRep = (remainB > UINT8_MAX) ? UINT8_MAX : static_cast<uint8_t>(remainB);
                int64_t moff = off + baseB * cNum * tilingData->lastDim;
                int64_t soff = offset + baseB * tilingData->lastDim;
                Mul<T1>(mul1Ub[moff], mul1Ub[moff], sinSize[soff], curMask, curRep, repeatParams);
                Mul<T1>(mul2Ub[moff], mul2Ub[moff], cosSize[soff], curMask, curRep, repeatParams);
                remainB -= curRep;
                baseB += curRep;
            }
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::LargeQCFP32(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& mul1Ub, LocalTensor<T1>& sinSize,
    LocalTensor<T1>& mul2Ub, LocalTensor<T1>& cosSize,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t fullMask = tilingData->mask;
    int64_t fullChunks = (tilingData->lastDim + fullMask - 1) / fullMask;
    int64_t batchStride = cNum * tilingData->lastDim;
    for (int64_t i = 0; i < preCBatchB; i++) {
        for (int32_t j = 0; j < fullChunks; j++) {
            int64_t offset = j * fullMask;
            int64_t curMask = (offset + fullMask > tilingData->lastDim) ? (tilingData->lastDim - offset) : fullMask;
            int64_t off = i * batchStride + offset;
            int64_t sOff = i * tilingData->lastDim + offset;
            Mul<T1>(mul1Ub[off], mul1Ub[off], sinSize[sOff], curMask, cNum, repeatParams);
            Mul<T1>(mul2Ub[off], mul2Ub[off], cosSize[sOff], curMask, cNum, repeatParams);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::ComputeRotaryNoCast(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
    LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb, const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<T1> mul2Ub = mul2.Get<T1>();
    LocalTensor<T1> mul1Ub = mul1.Get<T1>();
    DataCopy(mul2Ub, qSize, preCBatchB * cNum * tilingData->lastDim);
    PipeBarrier<PIPE_V>();
    qInQueue.FreeTensor(qSize);

    DataCopyParams rotCopy;
    rotCopy.blockCount = preCBatchB * cNum;
    rotCopy.blockLen = tilingData->blockLenq2q1;
    rotCopy.srcStride = tilingData->blockLenq2q1;
    rotCopy.dstStride = tilingData->blockLenq2q1;
    DataCopy(mul1Ub, mul2Ub[tilingData->halfNum], rotCopy);
    DataCopy(mul1Ub[tilingData->halfNum], mul2Ub, rotCopy);
    PipeBarrier<PIPE_V>();

    if (cNum >= preCBatchB) {
        repeatParams.dstRepStride = tilingData->dstRepSBr;
        repeatParams.src0RepStride = tilingData->dstRepSBr;
        repeatParams.src1RepStride = 0;
        LargeQCFP32(preCBatchB, cNum, mul1Ub, sinSize, mul2Ub, cosSize, tilingData);
    } else {
        repeatParams.dstRepStride = cNum * tilingData->dstRepSBr;
        repeatParams.src0RepStride = cNum * tilingData->dstRepSBr;
        repeatParams.src1RepStride = tilingData->dstRepSBr;
        SmallQCFP32(preCBatchB, cNum, mul1Ub, sinSize, mul2Ub, cosSize, tilingData);
    }
    PipeBarrier<PIPE_V>();
    Add(outUb, mul1Ub, mul2Ub, preCBatchB * cNum * tilingData->lastDim);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::ComputeRotaryCast(
   const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T2>& cosC1,
   LocalTensor<T2>& sinC1, LocalTensor<T1>& outUb,
   const ApplyRotaryPosEmbTilingData* tilingData)
{
#if ORIG_DTYPE_QUERY != DT_FLOAT
    LocalTensor<T2> mul2Ub = mul2.Get<T2>();
    LocalTensor<T2> mul1Ub = mul1.Get<T2>();
    Cast(mul2Ub, qSize, RoundMode::CAST_NONE, preCBatchB * cNum * tilingData->lastDim);
    PipeBarrier<PIPE_V>();
    qInQueue.FreeTensor(qSize);

    DataCopyParams rotCopy;
    rotCopy.blockCount = preCBatchB * cNum;
    rotCopy.blockLen = tilingData->blockLenq2q1;
    rotCopy.srcStride = tilingData->blockLenq2q1;
    rotCopy.dstStride = tilingData->blockLenq2q1;

    DataCopy(mul1Ub, mul2Ub[tilingData->halfNum], rotCopy);
    DataCopy(mul1Ub[tilingData->halfNum], mul2Ub, rotCopy);
    PipeBarrier<PIPE_V>();

    if (cNum >= preCBatchB) {
        repeatParams.dstRepStride = tilingData->dstRepSBr;
        repeatParams.src0RepStride = tilingData->dstRepSBr;
        repeatParams.src1RepStride = 0;
        LargeQC(preCBatchB, cNum, mul1Ub, sinC1, mul2Ub, cosC1, tilingData);
    } else {
        repeatParams.dstRepStride = cNum * tilingData->dstRepSBr;
        repeatParams.src0RepStride = cNum * tilingData->dstRepSBr;
        repeatParams.src1RepStride = tilingData->dstRepSBr;
        SmallQC(preCBatchB, cNum, mul1Ub, sinC1, mul2Ub, cosC1, tilingData);
    }
    PipeBarrier<PIPE_V>();
    Add(mul2Ub, mul1Ub, mul2Ub, preCBatchB * cNum * tilingData->lastDim);
    PipeBarrier<PIPE_V>();
    Cast(outUb, mul2Ub, RoundMode::CAST_RINT, preCBatchB * cNum * tilingData->lastDim);
#endif
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::CopyInTensor(
    const int64_t coreBatchIndex, const int64_t preCBatchB, GlobalTensor<T1>& gm, const int64_t coreOffset,
    const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qUb,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t gmBatchOffset = blockIdx * coreOffset + coreBatchIndex * tilingData->preCBatchB * (cNum * dim3);
    if (dim3 > tilingData->lastDim) {
        int64_t batchSize = cNum * dim3;
        int64_t ubBatchSize = cNum * tilingData->lastDim;
        if (cNum == 1) {
            DataCopyParams copyIn;
            copyIn.blockCount = 1;
            copyIn.blockLen = tilingData->blockMoveQ;
            copyIn.srcStride = 0;
            copyIn.dstStride = 0;
            for (int i = 0; i < preCBatchB; i++) {
                DataCopy(qUb[i * ubBatchSize], gm[gmBatchOffset + i * batchSize], copyIn);
            }
        } else {
            DataCopyParams copyIn;
            copyIn.blockCount = cNum;
            copyIn.blockLen = tilingData->blockMoveQ;
            copyIn.srcStride = dim3 / oneBlock - tilingData->blockMoveQ;
            copyIn.dstStride = tilingData->lastDim / oneBlock - tilingData->blockMoveQ;
            for (int i = 0; i < preCBatchB; i++) {
                DataCopy(qUb[i * ubBatchSize], gm[gmBatchOffset + i * batchSize], copyIn);
            }
        }
    } else {
        DataCopy(qUb, gm[gmBatchOffset], preCBatchB * cdNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::CopyOutTensor(
    const int64_t coreBatchIndex, const int64_t preCBatchB, GlobalTensor<T1>& gm, const int64_t coreOffset,
    const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qOutUb,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t gmBatchOffset = blockIdx * coreOffset + coreBatchIndex * tilingData->preCBatchB * (cNum * dim3);
    if (dim3 > tilingData->lastDim) {
        int64_t batchSize = cNum * dim3;
        int64_t ubBatchSize = cNum * tilingData->lastDim;
        if (cNum == 1) {
            DataCopyParams copyOut;
            copyOut.blockCount = 1;
            copyOut.blockLen = tilingData->blockMoveQ;
            copyOut.srcStride = 0;
            copyOut.dstStride = 0;
            for (int i = 0; i < preCBatchB; i++) {
                DataCopy(gm[gmBatchOffset + i * batchSize], qOutUb[i * ubBatchSize], copyOut);
            }
        } else {
            DataCopyParams copyOut;
            copyOut.blockCount = cNum;
            copyOut.blockLen = tilingData->blockMoveQ;
            copyOut.srcStride = tilingData->lastDim / oneBlock - tilingData->blockMoveQ;
            copyOut.dstStride = dim3 / oneBlock - tilingData->blockMoveQ;
            for (int i = 0; i < preCBatchB; i++) {
                DataCopy(gm[gmBatchOffset + i * batchSize], qOutUb[i * ubBatchSize], copyOut);
            }
        }
    } else {
        DataCopy(gm[gmBatchOffset], qOutUb, preCBatchB * cdNum);
    }
}

template <typename T1, typename T2>
template <typename T>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::MulsNegFirstHalfSplit(
    LocalTensor<T>& sinTensor, T negScalar, int64_t batchB,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t remainB = batchB;
    int64_t baseB = 0;
    while (remainB > 0) {
        uint8_t curRep = (remainB > UINT8_MAX) ? UINT8_MAX : static_cast<uint8_t>(remainB);
        Muls(sinTensor[baseB * tilingData->lastDim], sinTensor[baseB * tilingData->lastDim],
             negScalar, tilingData->halfNum, curRep, mulRepeatP);
        remainB -= curRep;
        baseB += curRep;
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::ProcessOneBatch(
    const int64_t coreBatchIndex, const int64_t batchB, const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<T1> cosUb = cosInQueue.AllocTensor<T1>();
    LocalTensor<T1> sinUb = sinInQueue.AllocTensor<T1>();
    DataCopy(cosUb,
             cosGm[blockIdx * tilingData->cosCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->coscdNum],
             batchB * tilingData->coscdNum);
    DataCopy(sinUb,
             sinGm[blockIdx * tilingData->cosCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->coscdNum],
             batchB * tilingData->coscdNum);

    // CopyIn Q
    LocalTensor<T1> qUb = qInQueue.AllocTensor<T1>();
    CopyInTensor(coreBatchIndex, batchB, qGm, tilingData->qCoreOffset, tilingData->qcNum, tilingData->qcdNum,
                    tilingData->qDim3, qUb, tilingData);
    qInQueue.EnQue(qUb);
    cosInQueue.EnQue(cosUb);
    sinInQueue.EnQue(sinUb);
    LocalTensor<T1> cosSize = cosInQueue.DeQue<T1>();
    LocalTensor<T1> sinSize = sinInQueue.DeQue<T1>();
    LocalTensor<T1> qSize = qInQueue.DeQue<T1>();

#if ORIG_DTYPE_QUERY != DT_FLOAT
    LocalTensor<T2> cosC1 = cosC.Get<T2>();
    LocalTensor<T2> sinC1 = sinC.Get<T2>();
    Cast(cosC1, cosSize, RoundMode::CAST_NONE, batchB * tilingData->lastDim);
    Cast(sinC1, sinSize, RoundMode::CAST_NONE, batchB * tilingData->lastDim);
    PipeBarrier<PIPE_V>();
    MulsNegFirstHalfSplit(sinC1, T2(-1.0), batchB, tilingData);
#else
    MulsNegFirstHalfSplit(sinSize, T1(-1.0), batchB, tilingData);
#endif

    // Compute Q
    LocalTensor<T1> qOutUb = qOutQueue.AllocTensor<T1>();
#if ORIG_DTYPE_QUERY != DT_FLOAT
    ComputeRotaryCast(batchB, tilingData->qcNum, qSize, cosC1, sinC1, qOutUb, tilingData);
#else
    ComputeRotaryNoCast(batchB, tilingData->qcNum, qSize, cosSize, sinSize, qOutUb, tilingData);
#endif

    // CopyIn K
    LocalTensor<T1> kUb = qInQueue.AllocTensor<T1>();
    CopyInTensor(coreBatchIndex, batchB, kGm, tilingData->kCoreOffset, tilingData->kcNum, tilingData->kcdNum,
                    tilingData->kDim3, kUb, tilingData);
    qOutQueue.EnQue(qOutUb);

    // CopyOut Q
    LocalTensor<T1> qOutUbSize = qOutQueue.DeQue<T1>();
    CopyOutTensor(coreBatchIndex, batchB, qGm, tilingData->qCoreOffset, tilingData->qcNum, tilingData->qcdNum,
                    tilingData->qDim3, qOutUbSize, tilingData);
    qOutQueue.FreeTensor(qOutUbSize);
    qInQueue.EnQue(kUb);

    // Compute K
    LocalTensor<T1> kSize = qInQueue.DeQue<T1>();
    LocalTensor<T1> kOutUb = qOutQueue.AllocTensor<T1>();
#if ORIG_DTYPE_QUERY != DT_FLOAT
    ComputeRotaryCast(batchB, tilingData->kcNum, kSize, cosC1, sinC1, kOutUb, tilingData);
#else
    ComputeRotaryNoCast(batchB, tilingData->kcNum, kSize, cosSize, sinSize, kOutUb, tilingData);
#endif
    cosInQueue.FreeTensor(cosSize);
    sinInQueue.FreeTensor(sinSize);
    qOutQueue.EnQue(kOutUb);

    // CopyOut K
    LocalTensor<T1> kOutUbSize = qOutQueue.DeQue<T1>();
    CopyOutTensor(coreBatchIndex, batchB, kGm, tilingData->kCoreOffset, tilingData->kcNum, tilingData->kcdNum,
                    tilingData->kDim3, kOutUbSize, tilingData);
    qOutQueue.FreeTensor(kOutUbSize);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::ProcessPreCore(const ApplyRotaryPosEmbTilingData* tilingData)
{
    for (int64_t i = 0; i < tilingData->preCLTimes; i++) {
        ProcessOneBatch(i, tilingData->preCBatchB, tilingData);
    }
    if (tilingData->preCBatchL > 0) {
        ProcessOneBatch(tilingData->preCLTimes, tilingData->preCBatchL, tilingData);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::ProcessLastCore(const ApplyRotaryPosEmbTilingData* tilingData)
{
    for (int64_t i = 0; i < tilingData->lastCLTimes; i++) {
        ProcessOneBatch(i, tilingData->preCBatchB, tilingData);
    }
    if (tilingData->lastCBatchL > 0) {
        ProcessOneBatch(tilingData->lastCLTimes, tilingData->lastCBatchL, tilingData);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCastPerf<T1, T2>::Process(const ApplyRotaryPosEmbTilingData* tilingData)
{
    if (blockIdx >= tilingData->useCoreNum) {
        return;
    }

    if (blockIdx == tilingData->useCoreNum - 1) {
        ProcessLastCore(tilingData);
    } else {
        ProcessPreCore(tilingData);
    }
}

} // namespace ApplyRotaryPosEmb

#endif // APPLY_ROTARY_POS_EMB_COMPUTE_AB_CAST_PERF_H
