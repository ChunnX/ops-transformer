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
 * \file apply_rotary_pos_emb_grad_a_and_b.h
 * \brief A/B split kernel（TND layout, NO_BROADCAST / BROADCAST_BSN）
 * 双路反向：同时处理 Q 和 K
 * 参照 rotary_position_embedding_grad_a_and_b.h
 */
#ifndef APPLY_ROTARY_POS_EMB_GRAD_A_AND_B_H
#define APPLY_ROTARY_POS_EMB_GRAD_A_AND_B_H

#include "apply_rotary_pos_emb_grad_common.h"
#include "apply_rotary_pos_emb_grad_tiling_data.h"

namespace ApplyRotaryPosEmbGrad {
using namespace AscendC;

template <typename T, bool IsBoardCast>
class ApplyRotaryPosEmbGradAAndB
{
public:
    __aicore__ inline ApplyRotaryPosEmbGradAAndB(TPipe* pipe, const ApplyRopeGradRegbaseParams* tiling)
        : pipe_(pipe), tilingData_(tiling){};

    __aicore__ inline ~ApplyRotaryPosEmbGradAAndB(){};

    __aicore__ inline void Init(
        GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR dxQOut, GM_ADDR dxKOut);

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitAllGlobalBuffer(
        GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR dxQOut, GM_ADDR dxKOut);
    __aicore__ inline void InitAllBuffer();
    __aicore__ inline void InitLoopParams();
    __aicore__ inline void ProcessInLoop(
        LocalTensor<T>& cos, LocalTensor<T>& sin, int64_t bStart, int64_t bLength,
        GlobalTensor<T>& gradGm, GlobalTensor<T>& dxGm, int64_t nSize);
    __aicore__ inline void CopyInCosAndSin(int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyIn(
        GlobalTensor<T>& source, int64_t bStart, int64_t bLength, int64_t nSize);
    __aicore__ inline void CopyOut(
        GlobalTensor<T>& target, int64_t bStart, int64_t bLength, int64_t nSize);
    __aicore__ inline void Compute(
        LocalTensor<T>& cos, LocalTensor<T>& sin, int64_t bLength, int64_t nLength);

private:
    TPipe* pipe_;
    const ApplyRopeGradRegbaseParams* tilingData_;

    GlobalTensor<T> gradQGm_, gradKGm_;
    GlobalTensor<T> cosGm_, sinGm_;
    GlobalTensor<T> dxQGm_, dxKGm_;

    TQue<QuePosition::VECIN, 1> gradInQueue_, cosInQueue_, sinInQueue_;
    TQue<QuePosition::VECOUT, 1> dxOutQueue_;

    int64_t blockIdx_ = 0;
    int64_t bBlockStart_ = 0;
    int64_t bBlockLength_ = 0;
    int64_t ubFactorB_ = 0, ubFactorN_ = 0;
    int64_t D_ = 0, dAlign_ = 0, dSplitCoef_ = HALF_INTERLEAVE_COEF;
};

template <typename T, bool IsBoardCast>
__aicore__ inline void ApplyRotaryPosEmbGradAAndB<T, IsBoardCast>::Init(
    GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR dxQOut, GM_ADDR dxKOut)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    InitAllGlobalBuffer(gradQ, gradK, cos, sin, dxQOut, dxKOut);
    InitAllBuffer();
    InitLoopParams();
}

template <typename T, bool IsBoardCast>
__aicore__ inline void ApplyRotaryPosEmbGradAAndB<T, IsBoardCast>::InitAllGlobalBuffer(
    GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR dxQOut, GM_ADDR dxKOut)
{
    gradQGm_.SetGlobalBuffer((__gm__ T*)gradQ);
    gradKGm_.SetGlobalBuffer((__gm__ T*)gradK);
    cosGm_.SetGlobalBuffer((__gm__ T*)cos);
    sinGm_.SetGlobalBuffer((__gm__ T*)sin);
    dxQGm_.SetGlobalBuffer((__gm__ T*)dxQOut);
    dxKGm_.SetGlobalBuffer((__gm__ T*)dxKOut);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void ApplyRotaryPosEmbGradAAndB<T, IsBoardCast>::InitAllBuffer()
{
    ubFactorB_ = tilingData_->ubFactorB;
    ubFactorN_ = tilingData_->ubFactorN;
    D_ = tilingData_->d;
    dAlign_ = Ops::Base::CeilAlign<int64_t>(D_ / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(T)) * dSplitCoef_;

    pipe_->InitBuffer(gradInQueue_, DOUBLE_BUFFER, ubFactorB_ * ubFactorN_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(dxOutQueue_, DOUBLE_BUFFER, ubFactorB_ * ubFactorN_ * dAlign_ * sizeof(T));
    if constexpr (IsBoardCast) {
        pipe_->InitBuffer(cosInQueue_, DOUBLE_BUFFER, ubFactorB_ * dAlign_ * sizeof(T));
        pipe_->InitBuffer(sinInQueue_, DOUBLE_BUFFER, ubFactorB_ * dAlign_ * sizeof(T));
    } else {
        pipe_->InitBuffer(cosInQueue_, DOUBLE_BUFFER, ubFactorB_ * dAlign_ * sizeof(T));
        pipe_->InitBuffer(sinInQueue_, DOUBLE_BUFFER, ubFactorB_ * dAlign_ * sizeof(T));
    }
}

template <typename T, bool IsBoardCast>
__aicore__ inline void ApplyRotaryPosEmbGradAAndB<T, IsBoardCast>::InitLoopParams()
{
    int64_t bIdx = blockIdx_ % tilingData_->blockNumB;
    bBlockLength_ = tilingData_->blockFactorB;
    if (bIdx == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0) {
        bBlockLength_ = tilingData_->b % tilingData_->blockFactorB;
    }
    bBlockStart_ = bIdx * tilingData_->blockFactorB;
}

template <typename T, bool IsBoardCast>
__aicore__ inline void ApplyRotaryPosEmbGradAAndB<T, IsBoardCast>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t ubLoopCount = Ops::Base::CeilDiv(bBlockLength_, ubFactorB_);
    for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
        CopyInCosAndSin(
            bBlockStart_ + ubLoopIdx * ubFactorB_,
            ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
        LocalTensor<T> cosUb = cosInQueue_.template DeQue<T>();
        LocalTensor<T> sinUb = sinInQueue_.template DeQue<T>();

        // === Q Path ===
        ProcessInLoop(
            cosUb, sinUb, bBlockStart_ + ubLoopIdx * ubFactorB_,
            ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_,
            gradQGm_, dxQGm_, tilingData_->nQ);

        // === K Path ===
        ProcessInLoop(
            cosUb, sinUb, bBlockStart_ + ubLoopIdx * ubFactorB_,
            ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_,
            gradKGm_, dxKGm_, tilingData_->nK);

        cosInQueue_.FreeTensor(cosUb);
        sinInQueue_.FreeTensor(sinUb);
    }
}

template <typename T, bool IsBoardCast>
__aicore__ inline void ApplyRotaryPosEmbGradAAndB<T, IsBoardCast>::ProcessInLoop(
    LocalTensor<T>& cos, LocalTensor<T>& sin, int64_t bStart, int64_t bLength,
    GlobalTensor<T>& gradGm, GlobalTensor<T>& dxGm, int64_t nSize)
{
    int64_t ubLoopCount = Ops::Base::CeilDiv(nSize, ubFactorN_);
    for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
        int64_t nLength = (ubLoopIdx != ubLoopCount - 1) ? ubFactorN_ : nSize - ubLoopIdx * ubFactorN_;
        CopyIn(gradGm, bStart, bLength, nSize);
        Compute(cos, sin, bLength, nLength);
        CopyOut(dxGm, bStart, bLength, nSize);
    }
}

template <typename T, bool IsBoardCast>
__aicore__ inline void ApplyRotaryPosEmbGradAAndB<T, IsBoardCast>::CopyInCosAndSin(
    int64_t bStart, int64_t bLength)
{
    LocalTensor<T> cosUb = cosInQueue_.template AllocTensor<T>();
    LocalTensor<T> sinUb = sinInQueue_.template AllocTensor<T>();
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = bLength;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1SrcStride = tilingData_->s * D_ * sizeof(T);
    loopParams.loop1DstStride = ubFactorB_ * dAlign_ * sizeof(T);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPadExtParams<T> copyPadExtparams{false, 0, 0, 0};
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = tilingData_->s * dSplitCoef_;
    copyExtParams.blockLen = D_ * sizeof(T) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPad(cosUb, cosGm_[bStart * tilingData_->s * D_], copyExtParams, copyPadExtparams);
    DataCopyPad(sinUb, sinGm_[bStart * tilingData_->s * D_], copyExtParams, copyPadExtparams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    cosInQueue_.template EnQue(cosUb);
    sinInQueue_.template EnQue(sinUb);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void ApplyRotaryPosEmbGradAAndB<T, IsBoardCast>::CopyIn(
    GlobalTensor<T>& source, int64_t bStart, int64_t bLength, int64_t nSize)
{
    LocalTensor<T> target = gradInQueue_.template AllocTensor<T>();
    LoopModeParams loopParams;
    loopParams.loop2Size = bLength;
    loopParams.loop1Size = 1;
    loopParams.loop2SrcStride = nSize * tilingData_->s * D_ * sizeof(T);
    loopParams.loop2DstStride = ubFactorN_ * ubFactorB_ * dAlign_ * sizeof(T);
    loopParams.loop1SrcStride = 0;
    loopParams.loop1DstStride = 0;
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = tilingData_->s * dSplitCoef_;
    copyExtParams.blockLen = D_ * sizeof(T) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPadExtParams<T> copyPadExtparams{false, 0, 0, 0};
    DataCopyPad(target, source[bStart * nSize * tilingData_->s * D_], copyExtParams, copyPadExtparams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    gradInQueue_.template EnQue(target);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void ApplyRotaryPosEmbGradAAndB<T, IsBoardCast>::CopyOut(
    GlobalTensor<T>& target, int64_t bStart, int64_t bLength, int64_t nSize)
{
    LocalTensor<T> source = dxOutQueue_.template DeQue<T>();
    LoopModeParams loopParams;
    loopParams.loop2Size = bLength;
    loopParams.loop1Size = 1;
    loopParams.loop2DstStride = nSize * tilingData_->s * D_ * sizeof(T);
    loopParams.loop2SrcStride = ubFactorN_ * ubFactorB_ * dAlign_ * sizeof(T);
    loopParams.loop1DstStride = 0;
    loopParams.loop1SrcStride = 0;
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = tilingData_->s * dSplitCoef_;
    copyExtParams.blockLen = D_ * sizeof(T) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPad(target[bStart * nSize * tilingData_->s * D_], source, copyExtParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    dxOutQueue_.FreeTensor(source);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void ApplyRotaryPosEmbGradAAndB<T, IsBoardCast>::Compute(
    LocalTensor<T>& cos, LocalTensor<T>& sin, int64_t bLength, int64_t nLength)
{
    LocalTensor<T> inUb = gradInQueue_.template DeQue<T>();
    LocalTensor<T> outUb = dxOutQueue_.template AllocTensor<T>();
    BatchHalfGradAlignVF<T, IsBoardCast>(
        (__local_mem__ T*)inUb.GetPhyAddr(), (__local_mem__ T*)cos.GetPhyAddr(), (__local_mem__ T*)sin.GetPhyAddr(),
        (__local_mem__ T*)outUb.GetPhyAddr(), tilingData_->s, bLength, nLength, D_, dAlign_, ubFactorB_, ubFactorN_);
    gradInQueue_.FreeTensor(inUb);
    dxOutQueue_.template EnQue(outUb);
}

} // namespace ApplyRotaryPosEmbGrad

#endif // APPLY_ROTARY_POS_EMB_GRAD_A_AND_B_H
