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
 * \file apply_rotary_pos_emb_grad_bab.h
 * \brief BAB split kernel（BSND layout, cos 1S1D）
 * 双路反向：同时处理 Q 和 K
 * 参照 rotary_position_embedding_grad_bab.h
 */
#ifndef APPLY_ROTARY_POS_EMB_GRAD_BAB_H
#define APPLY_ROTARY_POS_EMB_GRAD_BAB_H

#include "apply_rotary_pos_emb_grad_common.h"
#include "apply_rotary_pos_emb_grad_tiling_data.h"

namespace ApplyRotaryPosEmbGrad {
using namespace AscendC;

template <typename T>
class ApplyRotaryPosEmbGradBAB
{
public:
    __aicore__ inline ApplyRotaryPosEmbGradBAB(TPipe* pipe, const ApplyRopeGradRegbaseParams* tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(
        GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR dxQOut, GM_ADDR dxKOut);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;
    const ApplyRopeGradRegbaseParams* tilingData_;
    TPipe* pipe_;
    int64_t blockIdx_ = 0;
    int64_t dSplitCoef_ = HALF_INTERLEAVE_COEF;
    uint32_t dSplitSize_ = 0;
    int64_t dAlign_ = 0;
    int64_t bIdx_ = 0, sIdx_ = 0;
    int64_t bNum_ = 0, sNum_ = 0;
    int64_t maxN_ = 0;
    int64_t ubFactorS_ = 0, ubFactorN_ = 0;
    GlobalTensor<T> gradQGm_, gradKGm_;
    GlobalTensor<T> cosGm_, sinGm_;
    GlobalTensor<T> dxQGm_, dxKGm_;

    TQue<QuePosition::VECIN, 1> gradInQue_;
    TQue<QuePosition::VECIN, 1> cosInQue_;
    TQue<QuePosition::VECIN, 1> sinInQue_;
    TQue<QuePosition::VECOUT, 1> dxOutQue_;

private:
    __aicore__ inline void PrepareParams();
    __aicore__ inline void ProcessNLoop(
        const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum,
        GlobalTensor<T>& gradGm, GlobalTensor<T>& dxGm, int64_t nSize);
    __aicore__ inline void Compute(
        const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
        const LocalTensor<T>& inTensor, const LocalTensor<T>& outTensor,
        const uint32_t currSNum, const uint32_t currDNum);
    __aicore__ inline void ProcessN(
        const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
        const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum,
        GlobalTensor<T>& gradGm, GlobalTensor<T>& dxGm, int64_t nSize);
};

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::Init(
    GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR dxQOut, GM_ADDR dxKOut)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    maxN_ = (tilingData_->nQ > tilingData_->nK) ? tilingData_->nQ : tilingData_->nK;
    dSplitSize_ = tilingData_->d / dSplitCoef_ * sizeof(T);
    dAlign_ = Ops::Base::CeilAlign<int64_t>(tilingData_->d / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(T)) * dSplitCoef_;
    ubFactorN_ = tilingData_->ubFactorN;
    ubFactorS_ = tilingData_->ubFactorS;
    gradQGm_.SetGlobalBuffer((__gm__ T*)gradQ);
    gradKGm_.SetGlobalBuffer((__gm__ T*)gradK);
    cosGm_.SetGlobalBuffer((__gm__ T*)cos);
    sinGm_.SetGlobalBuffer((__gm__ T*)sin);
    dxQGm_.SetGlobalBuffer((__gm__ T*)dxQOut);
    dxKGm_.SetGlobalBuffer((__gm__ T*)dxKOut);
    pipe_->InitBuffer(gradInQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(cosInQue_, bufferNum, ubFactorS_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(sinInQue_, bufferNum, ubFactorS_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(dxOutQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlign_ * sizeof(T));
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::PrepareParams()
{
    bIdx_ = blockIdx_ % tilingData_->blockNumB;
    sIdx_ = blockIdx_ / tilingData_->blockNumB;
    bNum_ = tilingData_->blockFactorB;
    sNum_ = tilingData_->blockFactorS;
    if (bIdx_ == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0) {
        bNum_ = tilingData_->b % tilingData_->blockFactorB;
    }
    if (sIdx_ == tilingData_->blockNumS - 1 && tilingData_->s % tilingData_->blockFactorS != 0) {
        sNum_ = tilingData_->s % tilingData_->blockFactorS;
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    PrepareParams();
    uint32_t bIdxStart = bIdx_ * tilingData_->blockFactorB;
    for (uint32_t bIdx = bIdxStart; bIdx < bIdxStart + bNum_; bIdx++) {
        uint32_t sIdxStart = sIdx_ * tilingData_->blockFactorS;
        uint32_t sLoopCnt = Ops::Base::CeilDiv(sNum_, ubFactorS_);
        for (uint32_t loopIdx = 0; loopIdx < sLoopCnt; loopIdx++) {
            uint32_t currSNum = (loopIdx != sLoopCnt - 1) ? ubFactorS_ : sNum_ - loopIdx * ubFactorS_;
            // === Q Path ===
            ProcessNLoop(bIdx, sIdxStart + loopIdx * ubFactorS_, currSNum, gradQGm_, dxQGm_, tilingData_->nQ);
            // === K Path ===
            ProcessNLoop(bIdx, sIdxStart + loopIdx * ubFactorS_, currSNum, gradKGm_, dxKGm_, tilingData_->nK);
        }
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::ProcessNLoop(
    const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum,
    GlobalTensor<T>& gradGm, GlobalTensor<T>& dxGm, int64_t nSize)
{
    LocalTensor<T> sinTensor = sinInQue_.AllocTensor<T>();
    LocalTensor<T> cosTensor = cosInQue_.AllocTensor<T>();
    int64_t offset = sIdx * tilingData_->d;
    DataCopyExtParams copyParams{static_cast<uint16_t>(currSNum * dSplitCoef_), dSplitSize_, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(sinTensor, sinGm_[offset], copyParams, padParams);
    DataCopyPad(cosTensor, cosGm_[offset], copyParams, padParams);
    sinInQue_.EnQue(sinTensor);
    cosInQue_.EnQue(cosTensor);
    sinTensor = sinInQue_.DeQue<T>();
    cosTensor = cosInQue_.DeQue<T>();
    ProcessN(sinTensor, cosTensor, bIdx, sIdx, currSNum, gradGm, dxGm, nSize);
    sinInQue_.FreeTensor(sinTensor);
    cosInQue_.FreeTensor(cosTensor);
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::ProcessN(
    const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
    const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum,
    GlobalTensor<T>& gradGm, GlobalTensor<T>& dxGm, int64_t nSize)
{
    LocalTensor<T> gradInTensor, dxOutTensor;
    int64_t baseOffset = (bIdx * tilingData_->s + sIdx) * maxN_ * tilingData_->d;
    int64_t ubLoopNumN = Ops::Base::CeilDiv(nSize, ubFactorN_);
    for (uint32_t idxN = 0; idxN < ubLoopNumN; idxN++) {
        int64_t currDNum = (idxN == ubLoopNumN - 1) ? nSize - idxN * ubFactorN_ : ubFactorN_;
        int64_t offset = baseOffset + idxN * ubFactorN_ * tilingData_->d;
        DataCopyExtParams copyParams{static_cast<uint16_t>(currSNum * currDNum * dSplitCoef_), dSplitSize_, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        gradInTensor = gradInQue_.AllocTensor<T>();
        DataCopyPad(gradInTensor, gradGm[offset], copyParams, padParams);
        gradInQue_.EnQue(gradInTensor);
        gradInTensor = gradInQue_.DeQue<T>();
        dxOutTensor = dxOutQue_.AllocTensor<T>();
        Compute(sinTensor, cosTensor, gradInTensor, dxOutTensor, currSNum, currDNum);
        gradInQue_.FreeTensor(gradInTensor);
        dxOutQue_.EnQue(dxOutTensor);
        dxOutTensor = dxOutQue_.DeQue<T>();
        DataCopyPad(dxGm[offset], dxOutTensor, copyParams);
        dxOutQue_.FreeTensor(dxOutTensor);
    }
}

template <typename T>
__aicore__ inline void ApplyRotaryPosEmbGradBAB<T>::Compute(
    const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
    const LocalTensor<T>& inTensor, const LocalTensor<T>& outTensor,
    const uint32_t currSNum, const uint32_t currDNum)
{
    HalfGradAlignVF<T>(sinTensor, cosTensor, inTensor, outTensor, tilingData_->d, dAlign_, currSNum, currDNum);
}

} // namespace ApplyRotaryPosEmbGrad

#endif // APPLY_ROTARY_POS_EMB_GRAD_BAB_H
