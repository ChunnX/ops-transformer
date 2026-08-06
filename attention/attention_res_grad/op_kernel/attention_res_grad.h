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
 * \file attention_res_grad.h
 * \brief attention_res_grad kernel: B-axis split, outer B, inner N, H intact.
 *
 * Loop (per core):
 *   for batch in coreBatchStart_ .. coreBatchEnd_    (outer, B split)
 *       CopyInToken + PrecomputeGradProbs + SoftmaxBackwardN1
 *       for blk in 0 .. N+1  (inner, all blocks)
 *           ComputeGradV + CopyOutToken
 *   WriteGradScoreWeight (partial sum to workspace)
 *   SyncAll
 *   core 0: ReduceAndWriteWeights (cross-core sum, write grad_norm_weight / grad_proj_weight)
 */

#ifndef ATTENTION_RES_GRAD_H
#define ATTENTION_RES_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "attention_res_grad_tiling_data.h"
#include "attention_res_grad_tiling_key.h"

namespace NsAttentionResGrad {
using namespace AscendC;

template <typename T>
__aicore__ inline void KahanSumUpdate(LocalTensor<T>& inputTensor,
                                      LocalTensor<T> sumTensorList[2],
                                      const int32_t len,
                                      int32_t& outPos)
{
    LocalTensor<T> sumTensor = sumTensorList[outPos];
    LocalTensor<T> eTensor = sumTensorList[1 - outPos];
    Sub(inputTensor, inputTensor, eTensor, len);   // y = x - e
    Add(eTensor, inputTensor, sumTensor, len);     // t = y + s
    Sub(sumTensor, eTensor, sumTensor, len);       // e = t - s
    Sub(sumTensor, sumTensor, inputTensor, len);   // e = (t - s) - y
    outPos = 1 - outPos;
}

template <typename T>
class AttentionResGrad {
public:
    __aicore__ inline AttentionResGrad() {};

    __aicore__ inline void Init(GM_ADDR prefixSum, GM_ADDR blockResidual,
                                GM_ADDR projWeight, GM_ADDR normWeight,
                                GM_ADDR gradOutput, GM_ADDR invRms, GM_ADDR probs,
                                GM_ADDR gradPrefixSum, GM_ADDR gradBlockResidual,
                                GM_ADDR gradProjWeight, GM_ADDR gradNormWeight,
                                GM_ADDR workspace,
                                const AttentionResGradTilingData *td);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTiling(const AttentionResGradTilingData *td);
    __aicore__ inline GlobalTensor<T> VAddr(int64_t blk, int64_t batch) const;
    __aicore__ inline void InitScoreWeight();
    __aicore__ inline void CopyInToken(int64_t batch);
    __aicore__ inline void LoadGradOutAndV(int64_t blk);
    __aicore__ inline void PrecomputeGradProbs(int64_t batch);
    __aicore__ inline void SoftmaxBackwardN1();
    __aicore__ inline void ComputeGradV(int64_t blk);
    __aicore__ inline void CopyOutToken(int64_t blk, int64_t batch);
    __aicore__ inline void WriteGradScoreWeight();
    __aicore__ inline void ReduceAndWriteWeights();

    TPipe pipe;

    GlobalTensor<T>     prefixSumGm, blockResGm, projWeightGm, normWeightGm;
    GlobalTensor<T>     gradOutGm, invRmsGm, probsGm;
    GlobalTensor<T>     gradPrefixSumGm, gradBlockResGm;
    GlobalTensor<T>     gradProjWeightGm, gradNormWeightGm;
    GlobalTensor<float> gswWkspGm;
    GlobalTensor<float> wkspAllGm;

    TBuf<TPosition::VECCALC> scoreWeightBuf, halfBuf;
    TBuf<TPosition::VECCALC> floatBuf0, floatBuf1, floatBuf2, gswAccBuf;
    TBuf<TPosition::VECCALC> gswCmpBuf;
    TBuf<TPosition::VECCALC> probsHalfBuf, invrmsHalfBuf;
    TBuf<TPosition::VECCALC> probsFloatBuf, invrmsFloatBuf, gradProbBuf, gradScoreBuf;

    int64_t  batchSize_, numBlocks_, totalBlocks_, hiddenSize_;
    int64_t  coreBatchStart_, coreBatchEnd_, coreBatchCount_;
    int64_t  currentBatch_, coreNum_;
    int32_t  kahanOutPos_;
    uint64_t perCoreWkspBytes_;
    float    invH_;

    event_t  eventMTE2V_;
    event_t  eventVMTE2_;
    event_t  eventMTE3MTE2_;
    event_t  eventVMTE3_;
    event_t  eventVS_;
    event_t  eventSV_;
};

template <typename T>
__aicore__ inline GlobalTensor<T> AttentionResGrad<T>::VAddr(int64_t blk, int64_t batch) const
{
    if (blk < numBlocks_) {
        return blockResGm[static_cast<uint64_t>(batch) * numBlocks_ + static_cast<uint64_t>(blk)];
    } else {
        return prefixSumGm[static_cast<uint64_t>(batch)];
    }
}

template <typename T>
__aicore__ inline void AttentionResGrad<T>::ParseTiling(const AttentionResGradTilingData *td)
{
    batchSize_        = td->batchSize;
    numBlocks_        = td->numBlocks;
    totalBlocks_      = td->totalBlocks;
    hiddenSize_       = td->hiddenSize;
    coreBatchStart_   = td->coreBatchStart;
    coreBatchEnd_     = td->coreBatchEnd;
    coreBatchCount_   = td->coreBatchCount;
    invH_             = 1.0f / static_cast<float>(hiddenSize_);
    currentBatch_     = 0;
    coreNum_          = td->coreNum;
    kahanOutPos_      = 0;
    perCoreWkspBytes_ = td->perCoreWkspBytes;
}

template <typename T>
__aicore__ inline void AttentionResGrad<T>::Init(
    GM_ADDR prefixSum, GM_ADDR blockResidual,
    GM_ADDR projWeight, GM_ADDR normWeight,
    GM_ADDR gradOutput, GM_ADDR invRms, GM_ADDR probs,
    GM_ADDR gradPrefixSum, GM_ADDR gradBlockResidual,
    GM_ADDR gradProjWeight, GM_ADDR gradNormWeight,
    GM_ADDR workspace,
    const AttentionResGradTilingData *td)
{
    ParseTiling(td);

    __gm__ float* userWksp = (__gm__ float*)AscendC::GetUserWorkspace(workspace);

    // per-core batch range is computed in kernel, not host tiling
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t perCore = batchSize_ / coreNum_;
    int64_t rem = batchSize_ % coreNum_;
    if (blockIdx < rem) {
        coreBatchStart_ = blockIdx * (perCore + 1);
        coreBatchEnd_ = coreBatchStart_ + perCore + 1;
    } else {
        coreBatchStart_ = rem * (perCore + 1) + (blockIdx - rem) * perCore;
        coreBatchEnd_ = coreBatchStart_ + perCore;
    }
    coreBatchCount_ = coreBatchEnd_ - coreBatchStart_;

    uint64_t B  = static_cast<uint64_t>(batchSize_);
    uint64_t N  = static_cast<uint64_t>(numBlocks_);
    uint64_t H  = static_cast<uint64_t>(hiddenSize_);
    uint64_t N1 = static_cast<uint64_t>(totalBlocks_);

    prefixSumGm.SetGlobalBuffer((__gm__ T*)prefixSum,  B * H);
    blockResGm.SetGlobalBuffer((__gm__ T*)blockResidual, B * N * H);
    projWeightGm.SetGlobalBuffer((__gm__ T*)projWeight, H);
    normWeightGm.SetGlobalBuffer((__gm__ T*)normWeight, H);
    gradOutGm.SetGlobalBuffer((__gm__ T*)gradOutput, B * H);
    invRmsGm.SetGlobalBuffer((__gm__ T*)invRms, B * N1);
    probsGm.SetGlobalBuffer((__gm__ T*)probs,   B * N1);
    gradPrefixSumGm.SetGlobalBuffer((__gm__ T*)gradPrefixSum, B * H);
    gradBlockResGm.SetGlobalBuffer((__gm__ T*)gradBlockResidual, B * N * H);
    gradProjWeightGm.SetGlobalBuffer((__gm__ T*)gradProjWeight, H);
    gradNormWeightGm.SetGlobalBuffer((__gm__ T*)gradNormWeight, H);
    uint64_t wkspOff = perCoreWkspBytes_ * static_cast<uint64_t>(blockIdx);
    gswWkspGm.SetGlobalBuffer(
        userWksp + (wkspOff / sizeof(float)), H);

    uint64_t totalWkspBytes = td->perCoreWkspBytes * static_cast<uint64_t>(td->coreNum);
    wkspAllGm.SetGlobalBuffer(userWksp, totalWkspBytes / sizeof(float));

    uint32_t HU  = static_cast<uint32_t>(H);
    uint32_t N1U = static_cast<uint32_t>(N1);

    pipe.InitBuffer(scoreWeightBuf,    HU * sizeof(float));
    pipe.InitBuffer(halfBuf,           HU * sizeof(T));
    pipe.InitBuffer(floatBuf0,         HU * sizeof(float));
    pipe.InitBuffer(floatBuf1,         HU * sizeof(float));
    pipe.InitBuffer(floatBuf2,         HU * sizeof(float));
    pipe.InitBuffer(gswAccBuf,         HU * sizeof(float));
    pipe.InitBuffer(gswCmpBuf,         HU * sizeof(float));
    uint32_t n1HalfBytes  = (N1U * sizeof(T) + 31U) / 32U * 32U;
    uint32_t n1FloatBytes = N1U * sizeof(float);
    pipe.InitBuffer(probsHalfBuf,      n1HalfBytes);
    pipe.InitBuffer(invrmsHalfBuf,     n1HalfBytes);
    pipe.InitBuffer(probsFloatBuf,     n1FloatBytes);
    pipe.InitBuffer(invrmsFloatBuf,    n1FloatBytes);
    pipe.InitBuffer(gradProbBuf,       N1U * sizeof(float));
    pipe.InitBuffer(gradScoreBuf,      N1U * sizeof(float));
    // Kahan summation buffers.
}

template <typename T>
__aicore__ inline void AttentionResGrad<T>::InitScoreWeight()
{
    uint32_t H = static_cast<uint32_t>(hiddenSize_);

    LocalTensor<float> sw     = scoreWeightBuf.Get<float>();
    LocalTensor<T>     nh     = halfBuf.Get<T>();

    DataCopyExtParams cp(1, H * sizeof(T), 0, 0, 0);
    DataCopyPad(nh, normWeightGm, cp, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    Cast(sw, nh, RoundMode::CAST_NONE, H);
    SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
    WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);

    LocalTensor<T>     ph = halfBuf.Get<T>();
    LocalTensor<float> pf = floatBuf0.Get<float>();

    DataCopyPad(ph, projWeightGm, cp, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    Cast(pf, ph, RoundMode::CAST_NONE, H);
    SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
    WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);
    Mul(sw, sw, pf, H);
}

template <typename T>
__aicore__ inline void AttentionResGrad<T>::CopyInToken(int64_t batch)
{
    uint32_t H   = static_cast<uint32_t>(hiddenSize_);
    uint32_t N1U = static_cast<uint32_t>(totalBlocks_);

    uint64_t gmOff = static_cast<uint64_t>(batch) * totalBlocks_;

    LocalTensor<T> hb     = halfBuf.Get<T>();
    LocalTensor<T> ppHalf = probsHalfBuf.Get<T>();
    LocalTensor<T> piHalf = invrmsHalfBuf.Get<T>();

    DataCopyExtParams cpH(1, H * sizeof(T), 0, 0, 0);
    DataCopyExtParams cpN1(1, N1U * sizeof(T), 0, 0, 0);
    uint32_t n1RightPad = ((N1U * sizeof(T) + 31U) / 32U * 32U - N1U * sizeof(T)) / sizeof(T);

    DataCopyPad(hb, gradOutGm[static_cast<uint64_t>(batch) * hiddenSize_], cpH, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);

    DataCopyPad(ppHalf, probsGm[gmOff], cpN1, {true, 0U, static_cast<uint8_t>(n1RightPad), 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    Cast(probsFloatBuf.Get<float>(), ppHalf, RoundMode::CAST_NONE, N1U);
    SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
    WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);

    DataCopyPad(piHalf, invRmsGm[gmOff], cpN1, {true, 0U, static_cast<uint8_t>(n1RightPad), 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    Cast(invrmsFloatBuf.Get<float>(), piHalf, RoundMode::CAST_NONE, N1U);
    SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
    WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);
}

template <typename T>
__aicore__ inline void AttentionResGrad<T>::LoadGradOutAndV(int64_t blk)
{
    uint32_t H = static_cast<uint32_t>(hiddenSize_);

    LocalTensor<T>     hb  = halfBuf.Get<T>();
    LocalTensor<float> fb0 = floatBuf0.Get<float>();
    LocalTensor<float> fb1 = floatBuf1.Get<float>();

    DataCopyExtParams cp(1, H * sizeof(T), 0, 0, 0);

    DataCopyPad(hb, gradOutGm[static_cast<uint64_t>(currentBatch_) * hiddenSize_],
                cp, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    Cast(fb0, hb, RoundMode::CAST_NONE, H);
    SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
    WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);

    DataCopyPad(hb, VAddr(blk, currentBatch_), cp, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    Cast(fb1, hb, RoundMode::CAST_NONE, H);
}

template <typename T>
__aicore__ inline void AttentionResGrad<T>::PrecomputeGradProbs(int64_t batch)
{
    uint32_t H   = static_cast<uint32_t>(hiddenSize_);
    uint32_t N1U = static_cast<uint32_t>(totalBlocks_);

    LocalTensor<float> fb0 = floatBuf0.Get<float>();
    LocalTensor<float> fb1 = floatBuf1.Get<float>();
    LocalTensor<float> fb2 = floatBuf2.Get<float>();
    LocalTensor<T>     hb  = halfBuf.Get<T>();
    LocalTensor<float> gp  = gradProbBuf.Get<float>();

    // grad_out was copied to hb in CopyInToken.
    Cast(fb0, hb, RoundMode::CAST_NONE, H);
    SetFlag<HardEvent::V_MTE2>(eventVMTE2_);   // V has consumed hb; MTE2 may overwrite it.
    WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);

    for (uint32_t bb = 0; bb < N1U; bb++) {
        DataCopyExtParams cp(1, H * sizeof(T), 0, 0, 0);
        DataCopyPad(hb, VAddr(static_cast<int64_t>(bb), batch), cp, {false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
        Cast(fb1, hb, RoundMode::CAST_NONE, H);
        SetFlag<HardEvent::V_MTE2>(eventVMTE2_);   // V has consumed hb; MTE2 may overwrite it.
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);

        Mul(fb2, fb1, fb0, H);
        ReduceSum<float>(fb2, fb2, fb2, H);
        SetFlag<HardEvent::V_S>(eventVS_);   // ReduceSum -> GetValue
        WaitFlag<HardEvent::V_S>(eventVS_);
        gp.SetValue(bb, fb2.GetValue(0));
    }
    SetFlag<HardEvent::S_V>(eventSV_);   // Scalar writes to gp are visible to vector.
}

template <typename T>
__aicore__ inline void AttentionResGrad<T>::SoftmaxBackwardN1()
{
    uint32_t N1 = static_cast<uint32_t>(totalBlocks_);

    LocalTensor<float> probsF = probsFloatBuf.Get<float>();
    LocalTensor<float> gp     = gradProbBuf.Get<float>();
    LocalTensor<float> gs     = gradScoreBuf.Get<float>();
    LocalTensor<float> tmp    = floatBuf2.Get<float>();

    WaitFlag<HardEvent::S_V>(eventSV_);   // Wait for scalar writes to gp.
    Mul(tmp, probsF, gp, N1);
    ReduceSum<float>(tmp, tmp, tmp, N1);
    SetFlag<HardEvent::V_S>(eventVS_);   // ReduceSum -> GetValue
    WaitFlag<HardEvent::V_S>(eventVS_);
    float scalar = tmp.GetValue(0);

    Adds(tmp, gp, -scalar, N1);
    Mul(gs, probsF, tmp, N1);
}

template <typename T>
__aicore__ inline void AttentionResGrad<T>::ComputeGradV(int64_t blk)
{
    uint32_t H    = static_cast<uint32_t>(hiddenSize_);
    uint32_t blkU = static_cast<uint32_t>(blk);

    LocalTensor<float> sw     = scoreWeightBuf.Get<float>();
    LocalTensor<float> fb0    = floatBuf0.Get<float>();
    LocalTensor<float> fb1    = floatBuf1.Get<float>();
    LocalTensor<float> fb2    = floatBuf2.Get<float>();
    LocalTensor<float> gswA   = gswAccBuf.Get<float>();
    LocalTensor<float> gswC   = gswCmpBuf.Get<float>();
    LocalTensor<float> probsF = probsFloatBuf.Get<float>();
    LocalTensor<float> invrF  = invrmsFloatBuf.Get<float>();
    LocalTensor<float> gs     = gradScoreBuf.Get<float>();

    // Load grad_output and v[batch, blk, :]
    LoadGradOutAndV(blk);

    // grad_v_from_out = grad_out * probs[blk]

    SetFlag<HardEvent::V_S>(eventVS_);   // Vector writes are visible before scalar GetValue.
    WaitFlag<HardEvent::V_S>(eventVS_);
    float probBlk = probsF.GetValue(blkU);
    Muls(fb2, fb0, probBlk, H);

    // grad_v_from_k = (gradScore * inv_rms) * score_weight
    float gsBlk = gs.GetValue(blkU);
    float irBlk = invrF.GetValue(blkU);
    float sc    = gsBlk * irBlk;

    Muls(fb0, sw, sc, H);
    Add(fb2, fb2, fb0, H);
    // grad_k = grad_scores * score_weight;
    // grad_v_from_k = grad_k * inv_rms  ->  [B, N+1, H] * [B, N+1, 1]  ->  [B, N+1, H]
    // grad_inv_rms = gradScore[blk] * sum(score_weight * v)
    Mul(fb0, sw, fb1, H);
    ReduceSum<float>(fb0, fb0, fb0, H);
    SetFlag<HardEvent::V_S>(eventVS_);   // ReduceSum -> GetValue
    WaitFlag<HardEvent::V_S>(eventVS_);
    float sumSWV = fb0.GetValue(0);
    float gradInvRms = gsBlk * sumSWV;

    // grad_variance = grad_inv_rms * (-0.5) * inv_rms^3
    float gvar = gradInvRms * (-0.5f) * irBlk * irBlk * irBlk;

    // grad_v_from_var = (grad_variance / H) * 2.0 * v
    float varScale = gvar * 2.0f * invH_;
    Muls(fb0, fb1, varScale, H);
    Add(fb2, fb2, fb0, H);

    // grad_score_weight partial accumulation = sc * v (Kahan summation)
    Muls(fb0, fb1, sc, H);
    LocalTensor<float> sumCmp[2] = {gswA, gswC};
    KahanSumUpdate(fb0, sumCmp, static_cast<int32_t>(H), kahanOutPos_);
}

template <typename T>
__aicore__ inline void AttentionResGrad<T>::CopyOutToken(int64_t blk, int64_t batch)
{
    uint32_t H = static_cast<uint32_t>(hiddenSize_);

    LocalTensor<float> fb2 = floatBuf2.Get<float>();
    LocalTensor<T>     hb  = halfBuf.Get<T>();

    PipeBarrier<PIPE_V>();   // Cast must not read fb2 before vector writes complete.
    if constexpr (std::is_same<T, half>::value) {
        Cast(hb, fb2, RoundMode::CAST_NONE, H);
    } else {
        Cast(hb, fb2, RoundMode::CAST_RINT, H);
    }
    SetFlag<HardEvent::V_MTE3>(eventVMTE3_);   // Cast writes hb; MTE3 may read it.
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);

    DataCopyExtParams cp(1, H * sizeof(T), 0, 0, 0);
    if (blk < numBlocks_) {
        uint64_t gmOff =
            (static_cast<uint64_t>(batch) * numBlocks_ + static_cast<uint64_t>(blk)) * hiddenSize_;
        DataCopyPad(gradBlockResGm[gmOff], hb, cp);
    } else {
        uint64_t gmOff = static_cast<uint64_t>(batch) * hiddenSize_;
        DataCopyPad(gradPrefixSumGm[gmOff], hb, cp);
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
}

template <typename T>
__aicore__ inline void AttentionResGrad<T>::WriteGradScoreWeight()
{
    uint32_t H = static_cast<uint32_t>(hiddenSize_);
    DataCopyExtParams cp(1, H * sizeof(float), 0, 0, 0);
    SetFlag<HardEvent::V_MTE3>(eventVMTE3_);   // Kahan accumulation is complete; MTE3 may read it.
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);
    if (kahanOutPos_ == 0) {
        DataCopyPad(gswWkspGm, gswAccBuf.Get<float>(), cp);
    } else {
        DataCopyPad(gswWkspGm, gswCmpBuf.Get<float>(), cp);
    }
}

template <typename T>
__aicore__ inline void AttentionResGrad<T>::ReduceAndWriteWeights()
{
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }

    uint32_t H = static_cast<uint32_t>(hiddenSize_);

    LocalTensor<float> total = floatBuf0.Get<float>();
    LocalTensor<float> temp  = floatBuf1.Get<float>();
    LocalTensor<float> out   = floatBuf2.Get<float>();
    LocalTensor<T>     hb    = halfBuf.Get<T>();

    DataCopyExtParams cpF(1, H * sizeof(float), 0, 0, 0);
    DataCopyExtParams cpT(1, H * sizeof(T), 0, 0, 0);

    Duplicate<float>(total, 0.0f, static_cast<int32_t>(H));

    // cross-core reduce: sum each core's partial gswAcc from workspace
    uint64_t strideFloats = perCoreWkspBytes_ / sizeof(float);
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0U};
    for (int64_t c = 0; c < coreNum_; c++) {
        uint64_t cOff = static_cast<uint64_t>(c) * strideFloats;
        DataCopyPad(temp, wkspAllGm[cOff], cpF, padParams);
        SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
        Add(total, total, temp, H);
        SetFlag<HardEvent::V_MTE2>(eventVMTE2_);   // Add has consumed temp; MTE2 may overwrite it.
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);
    }

    // grad_norm_weight = total * proj_w
    LocalTensor<T> ph = halfBuf.Get<T>();
    DataCopyPad(ph, projWeightGm, cpT, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    Cast(temp, ph, RoundMode::CAST_NONE, H);
    Mul(out, total, temp, H);
    if constexpr (std::is_same<T, half>::value) {
        Cast(hb, out, RoundMode::CAST_NONE, H);
    } else {
        Cast(hb, out, RoundMode::CAST_RINT, H);
    }
    SetFlag<HardEvent::V_MTE3>(eventVMTE3_);   // Cast writes hb; MTE3 may read it.
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);
    DataCopyPad(gradNormWeightGm, hb, cpT);

    // grad_proj_weight = total * norm_w
    DataCopyPad(ph, normWeightGm, cpT, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    Cast(temp, ph, RoundMode::CAST_NONE, H);
    Mul(out, total, temp, H);
    if constexpr (std::is_same<T, half>::value) {
        Cast(hb, out, RoundMode::CAST_NONE, H);
    } else {
        Cast(hb, out, RoundMode::CAST_RINT, H);
    }
    SetFlag<HardEvent::V_MTE3>(eventVMTE3_);   // Cast writes hb; MTE3 may read it.
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);
    DataCopyPad(gradProjWeightGm, hb, cpT);
}

template <typename T>
__aicore__ inline void AttentionResGrad<T>::Process()
{
    eventMTE2V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    eventVMTE2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    eventMTE3MTE2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    eventVMTE3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    eventVS_    = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    eventSV_    = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    InitScoreWeight();
    {
        uint32_t H = static_cast<uint32_t>(hiddenSize_);
        Duplicate<float>(gswAccBuf.Get<float>(), 0.0f, static_cast<int32_t>(H));
        Duplicate<float>(gswCmpBuf.Get<float>(), 0.0f, static_cast<int32_t>(H));
    }

    int64_t N1 = totalBlocks_;
    for (int64_t batch = coreBatchStart_; batch < coreBatchEnd_; batch++) {
        currentBatch_ = batch;

        CopyInToken(batch);
        PrecomputeGradProbs(batch);
        SoftmaxBackwardN1();
        
        for (int64_t blk = 0; blk < N1; blk++) {
            ComputeGradV(blk);
            CopyOutToken(blk, batch);
        }
    }

    WriteGradScoreWeight();
    SyncAll();
    ReduceAndWriteWeights();
}

} // namespace NsAttentionResGrad
#endif // ATTENTION_RES_GRAD_H
