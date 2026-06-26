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
 * \file apply_rotary_pos_emb_compute_ab_cast.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_COMPUTE_AB_CAST_H
#define APPLY_ROTARY_POS_EMB_COMPUTE_AB_CAST_H

#include "kernel_operator.h"
#include "apply_rotary_pos_emb_base.h"

namespace ApplyRotaryPosEmb {
using namespace AscendC;

template <typename T1, typename T2>
class ARPEComputeABCast : public ApplyRotaryPosEmbBase<T1> {
public:
    __aicore__ inline ARPEComputeABCast(){};
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR workspace,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void Process(const ApplyRotaryPosEmbTilingData* tilingData);

private:
    __aicore__ inline void ProcessPreCore(const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ProcessLastCore(const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ProcessOneTensor(
        GlobalTensor<T1>& gm, const int64_t coreOffset, const int64_t cNum, const int64_t cdNum,
        const int64_t dim3, const int64_t coreBatchB, const int64_t coreBatchL,
        const ApplyRotaryPosEmbTilingData* tilingData);

    __aicore__ inline void CopyInTensor(
        const int64_t coreBatchIndex, const int64_t preCBatchB, GlobalTensor<T1>& gm, const int64_t coreOffset,
        const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qUb,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void CopyOutTensor(
        const int64_t coreBatchIndex, const int64_t preCBatchB, GlobalTensor<T1>& gm, const int64_t coreOffset,
        const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qOutUb,
        const ApplyRotaryPosEmbTilingData* tilingData);

    __aicore__ inline void ComputeRotary(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
        LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb, const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ComputeRotaryNoCast(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
        LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb, const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ComputeRotaryBF16(
        const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
        LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb, const ApplyRotaryPosEmbTilingData* tilingData);

    constexpr static int32_t bufferNum = 1;
    constexpr static int32_t bufferNumdb = 2;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNumdb> qInQueue;
    TQue<QuePosition::VECIN, bufferNumdb> cosInQueue;
    TQue<QuePosition::VECIN, bufferNumdb> sinInQueue;
    TQue<QuePosition::VECOUT, bufferNumdb> qOutQueue;
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
};

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::Init(
    GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR workspace,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    qGm.SetGlobalBuffer((__gm__ T1*)q);
    kGm.SetGlobalBuffer((__gm__ T1*)k);
    cosGm.SetGlobalBuffer((__gm__ T1*)cos);
    sinGm.SetGlobalBuffer((__gm__ T1*)sin);
    // 【泛化需求】qInQueue一次仅放Q或K，size按preCBatchB*singleN*alignedLastDim分配
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
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ComputeRotaryNoCast(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
    LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb, const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<T1> mul2Ub = mul2.Get<T1>();
    LocalTensor<T1> mul1Ub = mul1.Get<T1>();
    Muls(sinSize, sinSize, T1(-1.0), tilingData->halfNum, preCBatchB, mulRepeatP);

    // 【泛化需求】使用DataCopy旋转单张量，stride按alignedLastDim
    DataCopyParams rotCopy;
    rotCopy.blockCount = preCBatchB * cNum;
    rotCopy.blockLen = tilingData->blockLenq2q1;
    rotCopy.srcStride = tilingData->dstRepSBr;
    rotCopy.dstStride = tilingData->dstRepSBr;
    DataCopy(outUb, qSize[tilingData->halfNum], rotCopy);
    DataCopy(outUb[tilingData->halfNum], qSize, rotCopy);

    repeatParams.dstRepStride = tilingData->dstRepSBr;
    repeatParams.src0RepStride = tilingData->dstRepSBr;
    repeatParams.src1RepStride = 0;
    int64_t batchStride = cNum * tilingData->alignedLastDim;
    for (int64_t i = 0; i < preCBatchB; i++) {
        int64_t qOff = i * batchStride;
        int64_t sOff = i * tilingData->lastDim;
        Mul<T1>(mul1Ub[qOff], outUb[qOff], sinSize[sOff], tilingData->mask, cNum, repeatParams);
        Mul<T1>(mul2Ub[qOff], qSize[qOff], cosSize[sOff], tilingData->mask, cNum, repeatParams);
    }
    Add(outUb, mul1Ub, mul2Ub, preCBatchB * cNum * tilingData->lastDim);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ComputeRotaryBF16(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
    LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb, const ApplyRotaryPosEmbTilingData* tilingData)
{
#if ORIG_DTYPE_QUERY != DT_FLOAT
    LocalTensor<T2> qFP32 = mul2.Get<T2>();
    LocalTensor<T2> rotOrOut = mul1.Get<T2>();
    LocalTensor<T2> cosC1 = cosC.Get<T2>();
    LocalTensor<T2> sinC1 = sinC.Get<T2>();

    // 【泛化需求】BF16按实际lastDim cast到FP32计算，UB中按computeAlignedLastDim排布
    Cast(qFP32, qSize, RoundMode::CAST_NONE, preCBatchB * cNum * tilingData->lastDim);
    Cast(cosC1, cosSize, RoundMode::CAST_NONE, preCBatchB * tilingData->lastDim);
    Cast(sinC1, sinSize, RoundMode::CAST_NONE, preCBatchB * tilingData->lastDim);

    repeatParams.dstRepStride = tilingData->dstRepSBr;
    repeatParams.src0RepStride = tilingData->dstRepSBr;
    repeatParams.src1RepStride = 0;

    int64_t batchStride = cNum * tilingData->computeAlignedLastDim;
    float negOne = -1.0f;
    int64_t fullMask = tilingData->mask;
    int64_t fullChunks = (tilingData->lastDim + fullMask - 1) / fullMask;
    for (int64_t i = 0; i < preCBatchB; i++) {
        int64_t qOff = i * batchStride;
        int64_t sOff = i * tilingData->lastDim;
        // first half: -q[halfNum:] * sin[:halfNum]
        Mul<T2>(rotOrOut[qOff], qFP32[qOff + tilingData->halfNum], sinC1[sOff], tilingData->halfNum, cNum,
                 repeatParams);
        Muls<T2>(rotOrOut[qOff], rotOrOut[qOff], negOne, tilingData->halfNum, cNum, mulRepeatP);
        // second half: q[:halfNum] * sin[halfNum:]
        Mul<T2>(rotOrOut[qOff + tilingData->halfNum], qFP32[qOff], sinC1[sOff + tilingData->halfNum],
                 tilingData->halfNum, cNum, repeatParams);
        // q * cos -> qFP32
        // 【泛化需求】FP32计算时按tilingData->mask分chunk处理，支持lastDim>64
        for (int64_t k = 0; k < fullChunks; k++) {
            int64_t offset = k * fullMask;
            int64_t curMask = (offset + fullMask > tilingData->lastDim) ? (tilingData->lastDim - offset) : fullMask;
            Mul<T2>(qFP32[qOff + offset], qFP32[qOff + offset], cosC1[sOff + offset], curMask, cNum, repeatParams);
        }
        // q_embed = rotate(q)*sin + q*cos
        Add(rotOrOut[qOff], rotOrOut[qOff], qFP32[qOff], cNum * tilingData->lastDim);
    }
    Cast(outUb, rotOrOut, RoundMode::CAST_ROUND, preCBatchB * cNum * tilingData->lastDim);
#endif
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ComputeRotary(
    const int64_t preCBatchB, const int64_t cNum, LocalTensor<T1>& qSize, LocalTensor<T1>& cosSize,
    LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb, const ApplyRotaryPosEmbTilingData* tilingData)
{
#if ORIG_DTYPE_QUERY != DT_FLOAT
    ComputeRotaryBF16(preCBatchB, cNum, qSize, cosSize, sinSize, outUb, tilingData);
#else
    ComputeRotaryNoCast(preCBatchB, cNum, qSize, cosSize, sinSize, outUb, tilingData);
#endif
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::CopyInTensor(
    const int64_t coreBatchIndex, const int64_t preCBatchB, GlobalTensor<T1>& gm, const int64_t coreOffset,
    const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qUb,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t gmBatchOffset = blockIdx * coreOffset + coreBatchIndex * tilingData->preCBatchB * cdNum;
    if (dim3 > tilingData->lastDim) { // 部分维度旋转处理
        // 【泛化需求】仅搬运当前tensor（Q或K）的旋转部分；UB内按alignedLastDim布局
        int64_t oneBlock = tilingData->lastDim / tilingData->blockMoveQ;
        DataCopyParams copyIn;
        copyIn.blockCount = cNum;
        copyIn.blockLen = tilingData->blockMoveQ;
        copyIn.srcStride = dim3 / oneBlock - tilingData->blockMoveQ;
        copyIn.dstStride = tilingData->alignedLastDim / oneBlock - tilingData->blockMoveQ;
        int64_t batchSize = cNum * dim3;
        int64_t ubBatchSize = cNum * tilingData->alignedLastDim;
        for (int i = 0; i < preCBatchB; i++) {
            DataCopy(qUb[i * ubBatchSize], gm[gmBatchOffset + i * batchSize], copyIn);
        }
    } else { // 全维度旋转处理
        // 【泛化需求】仅搬运当前tensor的preCBatchB*cdNum个元素
        DataCopy(qUb, gm[gmBatchOffset], preCBatchB * cdNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::CopyOutTensor(
    const int64_t coreBatchIndex, const int64_t preCBatchB, GlobalTensor<T1>& gm, const int64_t coreOffset,
    const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qOutUb,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t gmBatchOffset = blockIdx * coreOffset + coreBatchIndex * tilingData->preCBatchB * cdNum;
    if (dim3 > tilingData->lastDim) { // 部分维度旋转处理
        // 【泛化需求】仅写回当前tensor的旋转部分；UB内按alignedLastDim布局
        int64_t oneBlock = tilingData->lastDim / tilingData->blockMoveQ;
        DataCopyParams copyOut;
        copyOut.blockCount = cNum;
        copyOut.blockLen = tilingData->blockMoveQ;
        copyOut.srcStride = tilingData->alignedLastDim / oneBlock - tilingData->blockMoveQ;
        copyOut.dstStride = dim3 / oneBlock - tilingData->blockMoveQ;
        int64_t batchSize = cNum * dim3;
        int64_t ubBatchSize = cNum * tilingData->alignedLastDim;
        for (int i = 0; i < preCBatchB; i++) {
            DataCopy(gm[gmBatchOffset + i * batchSize], qOutUb[i * ubBatchSize], copyOut);
        }
    } else { // 全维度旋转处理
        // 【泛化需求】仅写回当前tensor的preCBatchB*cdNum个元素
        DataCopy(gm[gmBatchOffset], qOutUb, preCBatchB * cdNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ProcessOneTensor(
    GlobalTensor<T1>& gm, const int64_t coreOffset, const int64_t cNum, const int64_t cdNum, const int64_t dim3,
    const int64_t coreBatchB, const int64_t coreBatchL, const ApplyRotaryPosEmbTilingData* tilingData)
{
    for (int64_t i = 0; i < coreBatchB; i++) {
        LocalTensor<T1> qUb = qInQueue.AllocTensor<T1>();
        LocalTensor<T1> cosUb = cosInQueue.AllocTensor<T1>();
        LocalTensor<T1> sinUb = sinInQueue.AllocTensor<T1>();
        // 【泛化需求】cos/sin复用，当前tensor单独搬运
        DataCopy(cosUb, cosGm[blockIdx * tilingData->cosCoreOffset + i * tilingData->preCBatchB * tilingData->coscdNum],
                 tilingData->preCBatchB * tilingData->coscdNum);
        DataCopy(sinUb, sinGm[blockIdx * tilingData->cosCoreOffset + i * tilingData->preCBatchB * tilingData->coscdNum],
                 tilingData->preCBatchB * tilingData->coscdNum);
        CopyInTensor(i, tilingData->preCBatchB, gm, coreOffset, cNum, cdNum, dim3, qUb, tilingData);
        qInQueue.EnQue(qUb);
        cosInQueue.EnQue(cosUb);
        sinInQueue.EnQue(sinUb);

        LocalTensor<T1> qSize = qInQueue.DeQue<T1>();
        LocalTensor<T1> cosSize = cosInQueue.DeQue<T1>();
        LocalTensor<T1> sinSize = sinInQueue.DeQue<T1>();
        LocalTensor<T1> qOutUb = qOutQueue.AllocTensor<T1>();
        ComputeRotary(tilingData->preCBatchB, cNum, qSize, cosSize, sinSize, qOutUb, tilingData);
        qInQueue.FreeTensor(qSize);
        cosInQueue.FreeTensor(cosSize);
        sinInQueue.FreeTensor(sinSize);
        qOutQueue.EnQue(qOutUb);

        LocalTensor<T1> qOutUbSize = qOutQueue.DeQue<T1>();
        CopyOutTensor(i, tilingData->preCBatchB, gm, coreOffset, cNum, cdNum, dim3, qOutUbSize, tilingData);
        qOutQueue.FreeTensor(qOutUbSize);
    }
    if (coreBatchL > 0) {
        LocalTensor<T1> qUb = qInQueue.AllocTensor<T1>();
        LocalTensor<T1> cosUb = cosInQueue.AllocTensor<T1>();
        LocalTensor<T1> sinUb = sinInQueue.AllocTensor<T1>();
        DataCopy(
            cosUb,
            cosGm[blockIdx * tilingData->cosCoreOffset + coreBatchB * tilingData->preCBatchB * tilingData->coscdNum],
            coreBatchL * tilingData->coscdNum);
        DataCopy(
            sinUb,
            sinGm[blockIdx * tilingData->cosCoreOffset + coreBatchB * tilingData->preCBatchB * tilingData->coscdNum],
            coreBatchL * tilingData->coscdNum);
        CopyInTensor(coreBatchB, coreBatchL, gm, coreOffset, cNum, cdNum, dim3, qUb, tilingData);
        qInQueue.EnQue(qUb);
        cosInQueue.EnQue(cosUb);
        sinInQueue.EnQue(sinUb);

        LocalTensor<T1> qSize = qInQueue.DeQue<T1>();
        LocalTensor<T1> cosSize = cosInQueue.DeQue<T1>();
        LocalTensor<T1> sinSize = sinInQueue.DeQue<T1>();
        LocalTensor<T1> qOutUb = qOutQueue.AllocTensor<T1>();
        ComputeRotary(coreBatchL, cNum, qSize, cosSize, sinSize, qOutUb, tilingData);
        qInQueue.FreeTensor(qSize);
        cosInQueue.FreeTensor(cosSize);
        sinInQueue.FreeTensor(sinSize);
        qOutQueue.EnQue(qOutUb);

        LocalTensor<T1> qOutUbSize = qOutQueue.DeQue<T1>();
        CopyOutTensor(coreBatchB, coreBatchL, gm, coreOffset, cNum, cdNum, dim3, qOutUbSize, tilingData);
        qOutQueue.FreeTensor(qOutUbSize);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ProcessPreCore(const ApplyRotaryPosEmbTilingData* tilingData)
{
    // 【泛化需求】先处理Q，再处理K；qInQueue一次仅存放其中一种
    ProcessOneTensor(qGm, tilingData->qCoreOffset, tilingData->qcNum, tilingData->qcdNum, tilingData->qDim3,
                     tilingData->preCLTimes, tilingData->preCBatchL, tilingData);
    ProcessOneTensor(kGm, tilingData->kCoreOffset, tilingData->kcNum, tilingData->kcdNum, tilingData->kDim3,
                     tilingData->preCLTimes, tilingData->preCBatchL, tilingData);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::ProcessLastCore(const ApplyRotaryPosEmbTilingData* tilingData)
{
    // 【泛化需求】先处理Q，再处理K；qInQueue一次仅存放其中一种
    ProcessOneTensor(qGm, tilingData->qCoreOffset, tilingData->qcNum, tilingData->qcdNum, tilingData->qDim3,
                     tilingData->lastCLTimes, tilingData->lastCBatchL, tilingData);
    ProcessOneTensor(kGm, tilingData->kCoreOffset, tilingData->kcNum, tilingData->kcdNum, tilingData->kDim3,
                     tilingData->lastCLTimes, tilingData->lastCBatchL, tilingData);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABCast<T1, T2>::Process(const ApplyRotaryPosEmbTilingData* tilingData)
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

#endif // APPLY_ROTARY_POS_EMB_COMPUTE_AB_CAST_H
