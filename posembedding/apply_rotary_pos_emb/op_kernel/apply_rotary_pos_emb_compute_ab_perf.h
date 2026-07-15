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
 * \file apply_rotary_pos_emb_compute_ab_perf.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_COMPUTE_AB_PERF_H
#define APPLY_ROTARY_POS_EMB_COMPUTE_AB_PERF_H

#include "kernel_operator.h"
#include "apply_rotary_pos_emb_base.h"

namespace ApplyRotaryPosEmb {
using namespace AscendC;

template <typename T1, typename T2>
class ARPEComputeABPerf : public ApplyRotaryPosEmbBase<T1> {
public:
    __aicore__ inline ARPEComputeABPerf(){};
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR workspace,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void Process(const ApplyRotaryPosEmbTilingData* tilingData);

private:
    __aicore__ inline void ProcessPreCore(const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ProcessLastCore(const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void CopyIn(
        const int64_t coreBatchIndex, const int64_t totalBatch, const int64_t innerTimes, const int64_t innerRemain,
        const ApplyRotaryPosEmbTilingData* tilingData);

    __aicore__ inline void CopyInTensor(
        const int64_t coreBatchIndex, const int64_t totalBatch, GlobalTensor<T1>& gm, const int64_t coreOffset,
        const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qUb,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void CopyOutTensor(
        const int64_t coreBatchIndex, const int64_t totalBatch, GlobalTensor<T1>& gm, const int64_t coreOffset,
        const int64_t cNum, const int64_t cdNum, const int64_t dim3, LocalTensor<T1>& qOutUb,
        const ApplyRotaryPosEmbTilingData* tilingData);

    // fp32 计算：mul1Ub = outUb(rotate_half) * sin ; mul2Ub = qSize * cos ; outUb = mul1Ub + mul2Ub
    __aicore__ inline void CTotaryNoCast(
        const int64_t j, const int64_t curBatch, const int64_t cNum, LocalTensor<T1>& qSize,
        LocalTensor<T1>& cosSize, LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void LargeQCFP32(
        const int64_t j, const int64_t curBatch, const int64_t cNum, LocalTensor<T1>& mul1Ub,
        LocalTensor<T1>& outUb, LocalTensor<T1>& sinSize, LocalTensor<T1>& mul2Ub, LocalTensor<T1>& qSize,
        LocalTensor<T1>& cosSize, const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void SmallQCFP32(
        const int64_t j, const int64_t curBatch, const int64_t cNum, LocalTensor<T1>& mul1Ub,
        LocalTensor<T1>& outUb, LocalTensor<T1>& sinSize, LocalTensor<T1>& mul2Ub, LocalTensor<T1>& qSize,
        LocalTensor<T1>& cosSize, const ApplyRotaryPosEmbTilingData* tilingData);

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

    GlobalTensor<T1> qGm;
    GlobalTensor<T1> kGm;
    GlobalTensor<T1> cosGm;
    GlobalTensor<T1> sinGm;
    DataCopyParams copyInq2q1;  // 旋转(前后半交换) DataCopy 参数
    UnaryRepeatParams mulRepeatP;
    BinaryRepeatParams repeatParams;
    uint64_t blockIdx = 0;
    uint64_t oneBlock = 0;
};

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABPerf<T1, T2>::Init(
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

    copyInq2q1.blockLen = tilingData->blockLenq2q1;
    copyInq2q1.srcStride = tilingData->blockLenq2q1;
    copyInq2q1.dstStride = tilingData->blockLenq2q1;

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
__aicore__ inline void ARPEComputeABPerf<T1, T2>::LargeQCFP32(
    const int64_t j, const int64_t curBatch, const int64_t cNum, LocalTensor<T1>& mul1Ub,
    LocalTensor<T1>& outUb, LocalTensor<T1>& sinSize, LocalTensor<T1>& mul2Ub, LocalTensor<T1>& qSize,
    LocalTensor<T1>& cosSize, const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t fullMask = tilingData->mask;
    int64_t fullChunks = (tilingData->lastDim + fullMask - 1) / fullMask;
    for (int64_t i = 0; i < curBatch; i++) {
        int64_t offmul = i * cNum * tilingData->lastDim;
        int64_t offset = j * tilingData->comBatchBB * cNum * tilingData->lastDim + offmul;
        int64_t offsin = j * tilingData->comBatchBB * tilingData->lastDim + i * tilingData->lastDim;
        for (int32_t k = 0; k < fullChunks; k++) {
            int64_t off2 = k * fullMask;
            int64_t curMask = (off2 + fullMask > tilingData->lastDim) ? (tilingData->lastDim - off2) : fullMask;
            Mul<T1>(mul1Ub[offmul + off2], outUb[offset + off2], sinSize[offsin + off2], curMask, cNum, repeatParams);
            Mul<T1>(mul2Ub[offmul + off2], qSize[offset + off2], cosSize[offsin + off2], curMask, cNum, repeatParams);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABPerf<T1, T2>::SmallQCFP32(
    const int64_t j, const int64_t curBatch, const int64_t cNum, LocalTensor<T1>& mul1Ub,
    LocalTensor<T1>& outUb, LocalTensor<T1>& sinSize, LocalTensor<T1>& mul2Ub, LocalTensor<T1>& qSize,
    LocalTensor<T1>& cosSize, const ApplyRotaryPosEmbTilingData* tilingData)
{
    int64_t fullMask = tilingData->mask;
    int64_t fullChunks = (tilingData->lastDim + fullMask - 1) / fullMask;
    int64_t offsin = j * tilingData->comBatchBB * tilingData->lastDim;
    int64_t offsetBase = j * tilingData->comBatchBB * cNum * tilingData->lastDim;
    for (int64_t ii = 0; ii < cNum; ii++) {
        int64_t offi = ii * tilingData->lastDim;
        for (int32_t k = 0; k < fullChunks; k++) {
            int64_t off2 = k * fullMask;
            int64_t curMask = (off2 + fullMask > tilingData->lastDim) ? (tilingData->lastDim - off2) : fullMask;
            int64_t remainB = curBatch;
            int64_t baseB = 0;
            while (remainB > 0) {
                uint8_t curRep = (remainB > UINT8_MAX) ? UINT8_MAX : static_cast<uint8_t>(remainB);
                int64_t batchOff = baseB * cNum * tilingData->lastDim;
                int64_t sinOff = baseB * tilingData->lastDim;
                Mul<T1>(mul1Ub[offi + off2 + batchOff], outUb[offsetBase + offi + off2 + batchOff],
                        sinSize[offsin + off2 + sinOff], curMask, curRep, repeatParams);
                Mul<T1>(mul2Ub[offi + off2 + batchOff], qSize[offsetBase + offi + off2 + batchOff],
                        cosSize[offsin + off2 + sinOff], curMask, curRep, repeatParams);
                remainB -= curRep;
                baseB += curRep;
            }
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABPerf<T1, T2>::CTotaryNoCast(
    const int64_t j, const int64_t curBatch, const int64_t cNum, LocalTensor<T1>& qSize,
    LocalTensor<T1>& cosSize, LocalTensor<T1>& sinSize, LocalTensor<T1>& outUb,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<T1> mul2Ub = mul2.Get<T1>();
    LocalTensor<T1> mul1Ub = mul1.Get<T1>();

    if (cNum >= curBatch) {
        repeatParams.dstRepStride = tilingData->dstRepSBr;
        repeatParams.src0RepStride = tilingData->dstRepSBr;
        repeatParams.src1RepStride = 0;
        LargeQCFP32(j, curBatch, cNum, mul1Ub, outUb, sinSize, mul2Ub, qSize, cosSize, tilingData);
    } else {
        repeatParams.dstRepStride = cNum * tilingData->dstRepSBr;
        repeatParams.src0RepStride = cNum * tilingData->dstRepSBr;
        repeatParams.src1RepStride = tilingData->dstRepSBr;
        SmallQCFP32(j, curBatch, cNum, mul1Ub, outUb, sinSize, mul2Ub, qSize, cosSize, tilingData);
    }
    PipeBarrier<PIPE_V>();
    Add(outUb[j * tilingData->comBatchBB * cNum * tilingData->lastDim], mul1Ub, mul2Ub,
        curBatch * cNum * tilingData->lastDim);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABPerf<T1, T2>::CopyInTensor(
    const int64_t coreBatchIndex, const int64_t totalBatch, GlobalTensor<T1>& gm, const int64_t coreOffset,
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
            for (int i = 0; i < totalBatch; i++) {
                DataCopy(qUb[i * ubBatchSize], gm[gmBatchOffset + i * batchSize], copyIn);
            }
        } else {
            DataCopyParams copyIn;
            copyIn.blockCount = cNum;
            copyIn.blockLen = tilingData->blockMoveQ;
            copyIn.srcStride = dim3 / oneBlock - tilingData->blockMoveQ;
            copyIn.dstStride = tilingData->lastDim / oneBlock - tilingData->blockMoveQ;
            for (int i = 0; i < totalBatch; i++) {
                DataCopy(qUb[i * ubBatchSize], gm[gmBatchOffset + i * batchSize], copyIn);
            }
        }
    } else {
        DataCopy(qUb, gm[gmBatchOffset], totalBatch * cdNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABPerf<T1, T2>::CopyOutTensor(
    const int64_t coreBatchIndex, const int64_t totalBatch, GlobalTensor<T1>& gm, const int64_t coreOffset,
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
            for (int i = 0; i < totalBatch; i++) {
                DataCopy(gm[gmBatchOffset + i * batchSize], qOutUb[i * ubBatchSize], copyOut);
            }
        } else {
            DataCopyParams copyOut;
            copyOut.blockCount = cNum;
            copyOut.blockLen = tilingData->blockMoveQ;
            copyOut.srcStride = tilingData->lastDim / oneBlock - tilingData->blockMoveQ;
            copyOut.dstStride = dim3 / oneBlock - tilingData->blockMoveQ;
            for (int i = 0; i < totalBatch; i++) {
                DataCopy(gm[gmBatchOffset + i * batchSize], qOutUb[i * ubBatchSize], copyOut);
            }
        }
    } else {
        DataCopy(gm[gmBatchOffset], qOutUb, totalBatch * cdNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABPerf<T1, T2>::CopyIn(
    const int64_t coreBatchIndex, const int64_t totalBatch, const int64_t innerTimes, const int64_t innerRemain,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<T1> cosUb = cosInQueue.AllocTensor<T1>();
    LocalTensor<T1> sinUb = sinInQueue.AllocTensor<T1>();
    DataCopy(cosUb,
             cosGm[blockIdx * tilingData->cosCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->coscdNum],
             totalBatch * tilingData->coscdNum);
    DataCopy(sinUb,
             sinGm[blockIdx * tilingData->cosCoreOffset + coreBatchIndex * tilingData->preCBatchB * tilingData->coscdNum],
             totalBatch * tilingData->coscdNum);

    // CopyIn Q
    LocalTensor<T1> qUb = qInQueue.AllocTensor<T1>();
    CopyInTensor(coreBatchIndex, totalBatch, qGm, tilingData->qCoreOffset, tilingData->qcNum, tilingData->qcdNum,
                    tilingData->qDim3, qUb, tilingData);
    qInQueue.EnQue(qUb);
    cosInQueue.EnQue(cosUb);
    sinInQueue.EnQue(sinUb);
    LocalTensor<T1> cosSize = cosInQueue.DeQue<T1>();
    LocalTensor<T1> sinSize = sinInQueue.DeQue<T1>();
    LocalTensor<T1> qSize = qInQueue.DeQue<T1>();

    // FP32: sin 前半统一取反一次，Q/K 复用同一 sinSize
    int64_t remainB = totalBatch;
    int64_t baseB = 0;
    while (remainB > 0) {
        uint8_t curRep = (remainB > UINT8_MAX) ? UINT8_MAX : static_cast<uint8_t>(remainB);
        Muls(sinSize[baseB * tilingData->lastDim], sinSize[baseB * tilingData->lastDim], T1(-1.0),
             tilingData->halfNum, curRep, mulRepeatP);
        remainB -= curRep;
        baseB += curRep;
    }
    PipeBarrier<PIPE_V>();

    // 旋转 Q：DataCopy 把整批 q前后半交换写入 qOutUb
    LocalTensor<T1> qOutUb = qOutQueue.AllocTensor<T1>();
    copyInq2q1.blockCount = totalBatch * tilingData->qcNum;
    DataCopy(qOutUb, qSize[tilingData->halfNum], copyInq2q1);
    DataCopy(qOutUb[tilingData->halfNum], qSize, copyInq2q1);
    PipeBarrier<PIPE_V>();

    // Compute Q
    for (int64_t j = 0; j < innerTimes; j++) {
        CTotaryNoCast(j, tilingData->comBatchBB, tilingData->qcNum, qSize, cosSize, sinSize, qOutUb, tilingData);
    }
    CTotaryNoCast(innerTimes, innerRemain, tilingData->qcNum, qSize, cosSize, sinSize, qOutUb, tilingData);
    qInQueue.FreeTensor(qSize);

    // CopyIn K
    LocalTensor<T1> kUb = qInQueue.AllocTensor<T1>();
    CopyInTensor(coreBatchIndex, totalBatch, kGm, tilingData->kCoreOffset, tilingData->kcNum, tilingData->kcdNum,
                    tilingData->kDim3, kUb, tilingData);

    // CopyOut Q
    qOutQueue.EnQue(qOutUb);
    LocalTensor<T1> qOutUbSize = qOutQueue.DeQue<T1>();
    CopyOutTensor(coreBatchIndex, totalBatch, qGm, tilingData->qCoreOffset, tilingData->qcNum, tilingData->qcdNum,
                    tilingData->qDim3, qOutUbSize, tilingData);
    qOutQueue.FreeTensor(qOutUbSize);
    qInQueue.EnQue(kUb);

    LocalTensor<T1> kSize = qInQueue.DeQue<T1>();
    LocalTensor<T1> kOutUb = qOutQueue.AllocTensor<T1>();
    copyInq2q1.blockCount = totalBatch * tilingData->kcNum;
    DataCopy(kOutUb, kSize[tilingData->halfNum], copyInq2q1);
    DataCopy(kOutUb[tilingData->halfNum], kSize, copyInq2q1);
    PipeBarrier<PIPE_V>();

    // Compute K
    for (int64_t j = 0; j < innerTimes; j++) {
        CTotaryNoCast(j, tilingData->comBatchBB, tilingData->kcNum, kSize, cosSize, sinSize, kOutUb, tilingData);
    }
    CTotaryNoCast(innerTimes, innerRemain, tilingData->kcNum, kSize, cosSize, sinSize, kOutUb, tilingData);
    qInQueue.FreeTensor(kSize);
    cosInQueue.FreeTensor(cosSize);
    sinInQueue.FreeTensor(sinSize);

    // CopyOut K
    qOutQueue.EnQue(kOutUb);
    LocalTensor<T1> kOutUbSize = qOutQueue.DeQue<T1>();
    CopyOutTensor(coreBatchIndex, totalBatch, kGm, tilingData->kCoreOffset, tilingData->kcNum, tilingData->kcdNum,
                    tilingData->kDim3, kOutUbSize, tilingData);
    qOutQueue.FreeTensor(kOutUbSize);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABPerf<T1, T2>::ProcessPreCore(const ApplyRotaryPosEmbTilingData* tilingData)
{
    for (int64_t i = 0; i < tilingData->preCLTimes; i++) {
        CopyIn(i, tilingData->preCBatchB, tilingData->preCBBTimes, tilingData->comBatchBBL, tilingData);
    }
    CopyIn(tilingData->preCLTimes, tilingData->preCBatchL, tilingData->preCBLTimes, tilingData->comBatchBLL, tilingData);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABPerf<T1, T2>::ProcessLastCore(const ApplyRotaryPosEmbTilingData* tilingData)
{
    for (int64_t i = 0; i < tilingData->lastCLTimes; i++) {
        CopyIn(i, tilingData->preCBatchB, tilingData->preCBBTimes, tilingData->comBatchBBL, tilingData);
    }
    CopyIn(tilingData->lastCLTimes, tilingData->lastCBatchL, tilingData->preCLLTimes, tilingData->comBatchLLL, tilingData);
}

template <typename T1, typename T2>
__aicore__ inline void ARPEComputeABPerf<T1, T2>::Process(const ApplyRotaryPosEmbTilingData* tilingData)
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

#endif // APPLY_ROTARY_POS_EMB_COMPUTE_AB_PERF_H
