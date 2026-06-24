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
 * \file inplace_partial_rotary_mul_grad_a_and_b.h
 * \brief A/B template kernel for InplacePartialRotaryMulGrad (NO_BROADCAST / BROADCAST_BSN)
 */
#ifndef __INPLACE_PARTIAL_ROTARY_MUL_GRAD_A_AND_B_H__
#define __INPLACE_PARTIAL_ROTARY_MUL_GRAD_A_AND_B_H__

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "inplace_partial_rotary_mul_grad_tiling_data.h"
#include "inplace_partial_rotary_mul_grad_common.h"

namespace InplacePartialRotaryMulGrad {
using namespace AscendC;

template <typename TDY, typename TCOS, bool IsBroadcast>
class InplacePartialRotaryMulGradAAndB {
public:
    __aicore__ inline InplacePartialRotaryMulGradAAndB() {}
    __aicore__ inline void Init(TPipe *pipe, const InplacePartialRotaryMulGradRegbaseParams *tiling, GM_ADDR dy,
                                GM_ADDR cos, GM_ADDR sin, GM_ADDR dx);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInCosAndSin(int64_t bStart, int64_t bLength);
    __aicore__ inline void Compute(const LocalTensor<TCOS> &cosBuffer, const LocalTensor<TCOS> &sinBuffer,
                                   const LocalTensor<TDY> &inBuffer, const LocalTensor<TDY> &outBuffer,
                                   int64_t bUbLength);
    __aicore__ inline void ProcessInLoop(const LocalTensor<TCOS> &cosBuffer,
                                         const LocalTensor<TCOS> &sinBuffer, int64_t bUbStart, int64_t bUbLength);

    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, 1> gradInQueue_;
    TQue<QuePosition::VECOUT, 1> xGradOutQueue_;
    TQue<QuePosition::VECIN, IsBroadcast ? 1 : 1> cosInQueue_;
    TQue<QuePosition::VECIN, IsBroadcast ? 1 : 1> sinInQueue_;

    GlobalTensor<TDY> gradGm_;
    GlobalTensor<TDY> dxGm_;
    GlobalTensor<TCOS> cosGm_;
    GlobalTensor<TCOS> sinGm_;

    const InplacePartialRotaryMulGradRegbaseParams *tilingData_{nullptr};
    DataCopyPadExtParams<TDY> padParamsDy_{false, 0, 0, static_cast<TDY>(0)};
    int64_t blockIdx_{0};
    int64_t bBlockStart_{0};
    int64_t bBlockLength_{0};
    int64_t dSplitCoef_{1};
    int64_t dAlignLenDy_{0};
    int64_t dAlignLenCos_{0};
    uint32_t dSplitSizeDy_{0};
    uint32_t dSplitSizeCos_{0};
    int64_t ubFactorB_{0};
};

template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadcast>::Init(
    TPipe *pipe, const InplacePartialRotaryMulGradRegbaseParams *tiling, GM_ADDR dy, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR dx)
{
    this->pipe_ = pipe;
    this->tilingData_ = tiling;
    this->blockIdx_ = GetBlockIdx();

    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    this->dSplitCoef_ = tilingData_->dSplitCoef;
    this->dAlignLenDy_ = Ops::Base::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_,
                                                       static_cast<int64_t>(BLOCK_TYPE_SIZE / sizeof(TDY))) *
                         dSplitCoef_;
    this->dAlignLenCos_ = Ops::Base::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_,
                                                        static_cast<int64_t>(BLOCK_TYPE_SIZE / sizeof(TCOS))) *
                          dSplitCoef_;
    this->dSplitSizeDy_ = static_cast<uint32_t>(tilingData_->sliceLength / dSplitCoef_ * sizeof(TDY));
    this->dSplitSizeCos_ = static_cast<uint32_t>(tilingData_->sliceLength / dSplitCoef_ * sizeof(TCOS));
    this->ubFactorB_ = tilingData_->ubFactorB;

    this->gradGm_.SetGlobalBuffer((__gm__ TDY *)dy);
    this->dxGm_.SetGlobalBuffer((__gm__ TDY *)dx);
    this->cosGm_.SetGlobalBuffer((__gm__ TCOS *)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ TCOS *)sin);

    this->pipe_->InitBuffer(this->gradInQueue_, 2, ubFactorB_ * dAlignLenDy_ * sizeof(TDY));
    this->pipe_->InitBuffer(this->xGradOutQueue_, 2, ubFactorB_ * dAlignLenDy_ * sizeof(TDY));

    if constexpr (IsBroadcast) {
        this->pipe_->InitBuffer(this->cosInQueue_, 1, dAlignLenCos_ * sizeof(TCOS));
        this->pipe_->InitBuffer(this->sinInQueue_, 1, dAlignLenCos_ * sizeof(TCOS));
    } else {
        this->pipe_->InitBuffer(this->cosInQueue_, 2, ubFactorB_ * dAlignLenCos_ * sizeof(TCOS));
        this->pipe_->InitBuffer(this->sinInQueue_, 2, ubFactorB_ * dAlignLenCos_ * sizeof(TCOS));
    }

    bBlockLength_ = tilingData_->blockFactorB;
    if (blockIdx_ == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0) {
        bBlockLength_ = tilingData_->b % tilingData_->blockFactorB;
    }
    bBlockStart_ = blockIdx_ * tilingData_->blockFactorB;
}

template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadcast>::
    CopyInCosAndSin(int64_t bStart, int64_t bLength)
{
    DataCopyExtParams cosParams = {static_cast<uint16_t>(bLength * dSplitCoef_), dSplitSizeCos_, 0, 0, 0};
    DataCopyPadExtParams<TCOS> padParams{false, 0, 0, 0};

    LocalTensor<TCOS> cosUb = cosInQueue_.template AllocTensor<TCOS>();
    DataCopyPad(cosUb, cosGm_[bStart * tilingData_->sliceLength], cosParams, padParams);
    cosInQueue_.EnQue(cosUb);

    LocalTensor<TCOS> sinUb = sinInQueue_.template AllocTensor<TCOS>();
    DataCopyPad(sinUb, sinGm_[bStart * tilingData_->sliceLength], cosParams, padParams);
    sinInQueue_.EnQue(sinUb);
}

template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadcast>::Compute(
    const LocalTensor<TCOS> &cosBuffer, const LocalTensor<TCOS> &sinBuffer,
    const LocalTensor<TDY> &inBuffer, const LocalTensor<TDY> &outBuffer, int64_t bUbLength)
{
    if (tilingData_->rotaryMode == ROTARY_MODE_INTERLEAVE) {
        if constexpr (IsBroadcast) {
            InterleaveModeGradVF<TDY, TCOS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                            tilingData_->sliceLength, 1, static_cast<uint16_t>(bUbLength));
        } else {
            auto *inUb = (__local_mem__ TDY *)inBuffer.GetPhyAddr();
            auto *outUb = (__local_mem__ TDY *)outBuffer.GetPhyAddr();
            auto *cosUb = (__local_mem__ TCOS *)cosBuffer.GetPhyAddr();
            auto *sinUb = (__local_mem__ TCOS *)sinBuffer.GetPhyAddr();
            BatchInterleaveModeGradVF<TDY, TCOS, IsBroadcast>(
                inUb, cosUb, sinUb, outUb, static_cast<uint16_t>(bUbLength), 1, 1, tilingData_->sliceLength,
                ubFactorB_, 1);
        }
    }
}

template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadcast>::ProcessInLoop(
    const LocalTensor<TCOS> &cosBuffer, const LocalTensor<TCOS> &sinBuffer, int64_t bUbStart,
    int64_t bUbLength)
{
    int64_t gradGmOffset = bUbStart * tilingData_->d + tilingData_->sliceStart;
    DataCopyExtParams inParams = {static_cast<uint16_t>(bUbLength * dSplitCoef_), dSplitSizeDy_, 0, 0, 0};
    DataCopyExtParams outParams = {static_cast<uint16_t>(bUbLength * dSplitCoef_), dSplitSizeDy_, 0, 0, 0};

    LocalTensor<TDY> inBuffer = gradInQueue_.template AllocTensor<TDY>();
    LocalTensor<TDY> outBuffer = xGradOutQueue_.template AllocTensor<TDY>();

    DataCopyPad(inBuffer, gradGm_[gradGmOffset], inParams, padParamsDy_);
    gradInQueue_.EnQue(inBuffer);
    inBuffer = gradInQueue_.template DeQue<TDY>();

    Compute(cosBuffer, sinBuffer, inBuffer, outBuffer, bUbLength);

    gradInQueue_.FreeTensor(inBuffer);
    xGradOutQueue_.EnQue(outBuffer);
    outBuffer = xGradOutQueue_.template DeQue<TDY>();
    DataCopyPad(dxGm_[gradGmOffset], outBuffer, outParams);
    xGradOutQueue_.FreeTensor(outBuffer);
}

template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadcast>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    uint32_t ubLoopCnt = AscendC::Ceil(static_cast<uint64_t>(bBlockLength_), static_cast<uint64_t>(ubFactorB_));

    if constexpr (IsBroadcast) {
        CopyInCosAndSin(0, 1);
        LocalTensor<TCOS> cosBuffer = cosInQueue_.template DeQue<TCOS>();
        LocalTensor<TCOS> sinBuffer = sinInQueue_.template DeQue<TCOS>();

        for (uint32_t ubLoopIdx = 0; ubLoopIdx < ubLoopCnt; ubLoopIdx++) {
            int64_t bUbLength =
                (ubLoopIdx != ubLoopCnt - 1) ? ubFactorB_ : bBlockLength_ - (ubLoopIdx * ubFactorB_);
            ProcessInLoop(cosBuffer, sinBuffer, bBlockStart_ + ubLoopIdx * ubFactorB_, bUbLength);
        }
        cosInQueue_.FreeTensor(cosBuffer);
        sinInQueue_.FreeTensor(sinBuffer);
    } else {
        for (uint32_t ubLoopIdx = 0; ubLoopIdx < ubLoopCnt; ubLoopIdx++) {
            int64_t bStart = bBlockStart_ + ubLoopIdx * ubFactorB_;
            int64_t bUbLength =
                (ubLoopIdx != ubLoopCnt - 1) ? ubFactorB_ : bBlockLength_ - (ubLoopIdx * ubFactorB_);

            CopyInCosAndSin(bStart, bUbLength);
            LocalTensor<TCOS> cosBuffer = cosInQueue_.template DeQue<TCOS>();
            LocalTensor<TCOS> sinBuffer = sinInQueue_.template DeQue<TCOS>();
            ProcessInLoop(cosBuffer, sinBuffer, bStart, bUbLength);
            cosInQueue_.FreeTensor(cosBuffer);
            sinInQueue_.FreeTensor(sinBuffer);
        }
    }
}

} // namespace InplacePartialRotaryMulGrad
#endif
