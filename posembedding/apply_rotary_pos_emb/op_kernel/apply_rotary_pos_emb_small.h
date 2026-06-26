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
 * \file apply_rotary_pos_emb_small.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_SMALL_H
#define APPLY_ROTARY_POS_EMB_SMALL_H

#include "kernel_operator.h"
#include "apply_rotary_pos_emb_base.h"

namespace ApplyRotaryPosEmb {
using namespace AscendC;

template <typename T1, typename T2>
class ARPESmall : public ApplyRotaryPosEmbBase<T1> {
public:
    __aicore__ inline ARPESmall(){};
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR workspace,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void Process(const ApplyRotaryPosEmbTilingData* tilingData);

private:
    __aicore__ inline void ProcessOneTensor(
        GlobalTensor<T1>& gm, const int64_t coreOffset, const int64_t headNum,
        const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void CopyInTensor(
        LocalTensor<T1>& qUb, LocalTensor<T1>& cosUb, LocalTensor<T1>& sinUb, GlobalTensor<T1>& gm,
        const int64_t coreOffset, const int64_t headNum, const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ComputeRotary(
        LocalTensor<T1>& qUb, LocalTensor<T1>& cosUb, LocalTensor<T1>& sinUb, LocalTensor<T1>& qoutSize,
        const int64_t headNum, const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void ComputeBF16(
        LocalTensor<T1>& qUb, LocalTensor<T1>& cosUb, LocalTensor<T1>& sinUb, LocalTensor<T1>& qoutSize,
        const int64_t headNum, const ApplyRotaryPosEmbTilingData* tilingData);
    __aicore__ inline void CopyOutTensor(
        LocalTensor<T1>& qOutUb, GlobalTensor<T1>& gm, const int64_t coreOffset, const int64_t headNum,
        const ApplyRotaryPosEmbTilingData* tilingData);

    constexpr static int32_t bufferNum = 1;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> qInQueue;
    TQue<QuePosition::VECOUT, bufferNum> qOutQueue;
    TQue<QuePosition::VECIN, bufferNum> cosInQueue;
    TQue<QuePosition::VECIN, bufferNum> sinInQueue;
    TBuf<QuePosition::VECCALC> mul1;
    TBuf<QuePosition::VECCALC> mul2;
    TBuf<QuePosition::VECCALC> cosCast;
    TBuf<QuePosition::VECCALC> sinCast;

    GlobalTensor<T1> qGm;
    GlobalTensor<T1> kGm;
    GlobalTensor<T1> cosGm;
    GlobalTensor<T1> sinGm;
    uint64_t blockIdx = 0;
    BinaryRepeatParams repeatParams;
    UnaryRepeatParams mulRepeatP = {1, 1, 0, 0};
};

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::Init(
    GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR kOut, GM_ADDR workspace,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    qGm.SetGlobalBuffer((__gm__ T1*)q);
    kGm.SetGlobalBuffer((__gm__ T1*)k);
    cosGm.SetGlobalBuffer((__gm__ T1*)cos);
    sinGm.SetGlobalBuffer((__gm__ T1*)sin);
    // 【泛化需求】qInQueue一次仅放Q或K，size按singleN*alignedLastDim分配
    pipe.InitBuffer(cosInQueue, bufferNum, tilingData->cosPart1Ub);
    pipe.InitBuffer(mul1, tilingData->q2q1Part1Ub);
    pipe.InitBuffer(sinInQueue, bufferNum, tilingData->cosPart1Ub);
    pipe.InitBuffer(mul2, tilingData->q2q1Part1Ub);
    pipe.InitBuffer(qInQueue, bufferNum, tilingData->qPart1Ub);
    pipe.InitBuffer(qOutQueue, bufferNum, tilingData->qPart1Ub);

    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    // 【泛化需求】repeat stride使用computeAlignedLastDim，使各头按对齐后长度排布
    repeatParams.dstRepStride = tilingData->dstRepSBr;
    repeatParams.src0RepStride = tilingData->dstRepSBr;
    repeatParams.src1RepStride = 0;

#if ORIG_DTYPE_QUERY != DT_FLOAT
    pipe.InitBuffer(cosCast, tilingData->sin1UbSize);
    pipe.InitBuffer(sinCast, tilingData->sin1UbSize);
#endif
}

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::CopyInTensor(
    LocalTensor<T1>& qUb, LocalTensor<T1>& cosUb, LocalTensor<T1>& sinUb, GlobalTensor<T1>& gm,
    const int64_t coreOffset, const int64_t headNum, const ApplyRotaryPosEmbTilingData* tilingData)
{
    DataCopy(cosUb, cosGm[blockIdx * tilingData->cosCoreOffset], tilingData->coscdNum);
    DataCopy(sinUb, sinGm[blockIdx * tilingData->cosCoreOffset], tilingData->coscdNum);
    // 【泛化需求】只搬运当前tensor（Q或K）的headNum * lastDim个元素
    DataCopy(qUb, gm[blockIdx * coreOffset], headNum * tilingData->lastDim);
}

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::ComputeRotary(
    LocalTensor<T1>& qUb, LocalTensor<T1>& cosUb, LocalTensor<T1>& sinUb, LocalTensor<T1>& qoutSize,
    const int64_t headNum, const ApplyRotaryPosEmbTilingData* tilingData)
{
#if ORIG_DTYPE_QUERY != DT_FLOAT
    ComputeBF16(qUb, cosUb, sinUb, qoutSize, headNum, tilingData);
#else
    LocalTensor<T1> mul2Ub = mul2.Get<T1>();
    LocalTensor<T1> mul1Ub = mul1.Get<T1>();
    T1 negOne = T1(-1.0);

    // q * cos
    Mul<T1>(mul2Ub, qUb, cosUb, tilingData->lastDim, headNum, repeatParams);

    // 【泛化需求】不使用DataCopy旋转，改用两次Mul直接计算rotate(q) * sin
    // first half: -q[half:] * sin[:half]
    Mul<T1>(mul1Ub, qUb[tilingData->halfNum], sinUb, tilingData->halfNum, headNum, repeatParams);
    Muls(mul1Ub, mul1Ub, negOne, tilingData->halfNum, headNum, mulRepeatP);

    // second half: q[:half] * sin[half:]
    Mul<T1>(mul1Ub[tilingData->halfNum], qUb, sinUb[tilingData->halfNum], tilingData->halfNum, headNum, repeatParams);

    // q_embed = q*cos + rotate(q)*sin
    Add(qoutSize, mul2Ub, mul1Ub, headNum * tilingData->lastDim);
#endif
}

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::ComputeBF16(
    LocalTensor<T1>& qUb, LocalTensor<T1>& cosUb, LocalTensor<T1>& sinUb, LocalTensor<T1>& qoutSize,
    const int64_t headNum, const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<T2> qFP32 = mul1.Get<T2>();   // 先存q的FP32视图
    LocalTensor<T2> rotOrOut = mul2.Get<T2>(); // 先存rotate(q)*sin，后存最终结果
    LocalTensor<T2> cosC1 = cosCast.Get<T2>();
    LocalTensor<T2> sinC1 = sinCast.Get<T2>();

    Cast(qFP32, qUb, RoundMode::CAST_NONE, headNum * tilingData->lastDim);
    Cast(cosC1, cosUb, RoundMode::CAST_NONE, tilingData->lastDim);
    Cast(sinC1, sinUb, RoundMode::CAST_NONE, tilingData->lastDim);

    float negOne = -1.0f;

    // rotate(q) * sin -> rotOrOut
    // first half: -q[half:] * sin[:half]
    Mul<T2>(rotOrOut, qFP32[tilingData->halfNum], sinC1, tilingData->halfNum, headNum, repeatParams);
    Muls<T2>(rotOrOut, rotOrOut, negOne, tilingData->halfNum, headNum, mulRepeatP);

    // second half: q[:half] * sin[half:]
    Mul<T2>(rotOrOut[tilingData->halfNum], qFP32, sinC1[tilingData->halfNum], tilingData->halfNum, headNum, repeatParams);

    // q_fp32 * cos_fp32 -> qFP32（覆盖原q视图）
    // 【泛化需求】FP32计算时按tilingData->mask分chunk处理，支持lastDim>64
    int64_t fullMask = tilingData->mask;
    int64_t fullChunks = (tilingData->lastDim + fullMask - 1) / fullMask;
    for (int64_t k = 0; k < fullChunks; k++) {
        int64_t offset = k * fullMask;
        int64_t curMask = (offset + fullMask > tilingData->lastDim) ? (tilingData->lastDim - offset) : fullMask;
        Mul<T2>(qFP32[offset], qFP32[offset], cosC1[offset], curMask, headNum, repeatParams);
    }

    // q_embed = rotate(q)*sin + q*cos
    Add(rotOrOut, rotOrOut, qFP32, headNum * tilingData->lastDim);
    Cast(qoutSize, rotOrOut, RoundMode::CAST_ROUND, headNum * tilingData->lastDim);
}

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::CopyOutTensor(
    LocalTensor<T1>& qOutUb, GlobalTensor<T1>& gm, const int64_t coreOffset, const int64_t headNum,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    // 【泛化需求】只写回当前tensor（Q或K）的headNum * lastDim个元素
    DataCopy(gm[blockIdx * coreOffset], qOutUb, headNum * tilingData->lastDim);
}

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::ProcessOneTensor(
    GlobalTensor<T1>& gm, const int64_t coreOffset, const int64_t headNum,
    const ApplyRotaryPosEmbTilingData* tilingData)
{
    LocalTensor<T1> qUb = qInQueue.AllocTensor<T1>();
    LocalTensor<T1> cosUb = cosInQueue.AllocTensor<T1>();
    LocalTensor<T1> sinUb = sinInQueue.AllocTensor<T1>();
    CopyInTensor(qUb, cosUb, sinUb, gm, coreOffset, headNum, tilingData);
    qInQueue.EnQue(qUb);
    cosInQueue.EnQue(cosUb);
    sinInQueue.EnQue(sinUb);

    LocalTensor<T1> qSize = qInQueue.DeQue<T1>();
    LocalTensor<T1> cosSize = cosInQueue.DeQue<T1>();
    LocalTensor<T1> sinSize = sinInQueue.DeQue<T1>();

    LocalTensor<T1> qOutUb = qOutQueue.AllocTensor<T1>();
    ComputeRotary(qSize, cosSize, sinSize, qOutUb, headNum, tilingData);

    qInQueue.FreeTensor(qSize);
    cosInQueue.FreeTensor(cosSize);
    sinInQueue.FreeTensor(sinSize);
    qOutQueue.EnQue(qOutUb);

    LocalTensor<T1> qOutUbSize = qOutQueue.DeQue<T1>();
    CopyOutTensor(qOutUbSize, gm, coreOffset, headNum, tilingData);
    qOutQueue.FreeTensor(qOutUbSize);
}

template <typename T1, typename T2>
__aicore__ inline void ARPESmall<T1, T2>::Process(const ApplyRotaryPosEmbTilingData* tilingData)
{
    if (blockIdx >= tilingData->useCoreNum) {
        return;
    }
    // 【泛化需求】先处理Q，再处理K；qInQueue一次仅存放其中一种
    ProcessOneTensor(qGm, tilingData->qCoreOffset, tilingData->qcNum, tilingData);
    ProcessOneTensor(kGm, tilingData->kCoreOffset, tilingData->kcNum, tilingData);
}

} // namespace ApplyRotaryPosEmb
#endif // APPLY_ROTARY_POS_EMB_SMALL_H
