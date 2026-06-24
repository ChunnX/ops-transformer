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
 * \file inplace_partial_rotary_mul_grad_aba_and_ba.h
 * \brief ABA/BA template kernel for InplacePartialRotaryMulGrad (BNSD layout)
 */
#ifndef __INPLACE_PARTIAL_ROTARY_MUL_GRAD_ABA_AND_BA_H__
#define __INPLACE_PARTIAL_ROTARY_MUL_GRAD_ABA_AND_BA_H__

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "inplace_partial_rotary_mul_grad_tiling_data.h"
#include "inplace_partial_rotary_mul_grad_common.h"

namespace InplacePartialRotaryMulGrad {
using namespace AscendC;

template <typename TDY, typename TCOS, bool IsBroadcast>
class InplacePartialRotaryMulGradABAAndBA {
public:
    __aicore__ inline InplacePartialRotaryMulGradABAAndBA() {}
    __aicore__ inline void Init(TPipe *pipe, const InplacePartialRotaryMulGradRegbaseParams *tiling, GM_ADDR dy,
                                GM_ADDR cos, GM_ADDR sin, GM_ADDR dx);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessInSLoop(int64_t sStart, int64_t sLength);
    __aicore__ inline void ProcessInSBLoop(int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength,
                                            const LocalTensor<TCOS> &cosBuffer,
                                            const LocalTensor<TCOS> &sinBuffer);

    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, 1> gradInQueue_;
    TQue<QuePosition::VECOUT, 1> xGradOutQueue_;
    TQue<QuePosition::VECIN, 1> cosInQueue_;
    TQue<QuePosition::VECIN, 1> sinInQueue_;

    GlobalTensor<TDY> gradGm_;
    GlobalTensor<TDY> dxGm_;
    GlobalTensor<TCOS> cosGm_;
    GlobalTensor<TCOS> sinGm_;

    const InplacePartialRotaryMulGradRegbaseParams *tilingData_{nullptr};
    DataCopyPadExtParams<TDY> padParamsDy_{false, 0, 0, static_cast<TDY>(0)};
    int64_t blockIdx_{0};
    int64_t bBlockStart_{0};
    int64_t bBlockLength_{0};
    int64_t sBlockStart_{0};
    int64_t sBlockLength_{0};
    int64_t dSplitCoef_{1};
    int64_t dAlignLenDy_{0};
    int64_t dAlignLenCos_{0};
    uint32_t dSplitSizeDy_{0};
    uint32_t dSplitSizeCos_{0};
    int64_t ubFactorB_{0};
    int64_t ubFactorS_{0};
    int64_t ubFactorN_{0};
};

template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void
InplacePartialRotaryMulGradABAAndBA<TDY, TCOS, IsBroadcast>::Init(
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
    this->ubFactorS_ = tilingData_->ubFactorS;
    this->ubFactorN_ = tilingData_->ubFactorN;

    this->gradGm_.SetGlobalBuffer((__gm__ TDY *)dy);
    this->dxGm_.SetGlobalBuffer((__gm__ TDY *)dx);
    this->cosGm_.SetGlobalBuffer((__gm__ TCOS *)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ TCOS *)sin);

    int64_t bufferSizeBS = dAlignLenDy_ * sizeof(TDY) * ubFactorB_ * ubFactorS_;
    this->pipe_->InitBuffer(this->gradInQueue_, 2, bufferSizeBS * ubFactorN_);
    this->pipe_->InitBuffer(this->xGradOutQueue_, 2, bufferSizeBS * ubFactorN_);

    if constexpr (IsBroadcast) {
        this->pipe_->InitBuffer(this->cosInQueue_, 2, dAlignLenCos_ * sizeof(TCOS) * ubFactorS_);
        this->pipe_->InitBuffer(this->sinInQueue_, 2, dAlignLenCos_ * sizeof(TCOS) * ubFactorS_);
    } else {
        int64_t bufferSizeCosBS = dAlignLenCos_ * sizeof(TCOS) * ubFactorB_ * ubFactorS_;
        this->pipe_->InitBuffer(this->cosInQueue_, 2, bufferSizeCosBS);
        this->pipe_->InitBuffer(this->sinInQueue_, 2, bufferSizeCosBS);
    }

    int64_t bIdx = blockIdx_ % tilingData_->blockNumB;
    int64_t sIdx = blockIdx_ / tilingData_->blockNumB;
    bBlockStart_ = bIdx * tilingData_->blockFactorB;
    sBlockStart_ = sIdx * tilingData_->blockFactorS;
    bBlockLength_ = tilingData_->blockFactorB;
    sBlockLength_ = tilingData_->blockFactorS;
    if (bIdx == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0) {
        bBlockLength_ = tilingData_->b % tilingData_->blockFactorB;
    }
    if (sIdx == tilingData_->blockNumS - 1 && tilingData_->s % tilingData_->blockFactorS != 0) {
        sBlockLength_ = tilingData_->s % tilingData_->blockFactorS;
    }
}

template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void
InplacePartialRotaryMulGradABAAndBA<TDY, TCOS, IsBroadcast>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    uint32_t ubLoopCntS =
        AscendC::Ceil(static_cast<uint64_t>(sBlockLength_), static_cast<uint64_t>(ubFactorS_));

    for (uint32_t ubLoopIdx = 0; ubLoopIdx < ubLoopCntS; ubLoopIdx++) {
        int64_t sStart = sBlockStart_ + ubLoopIdx * ubFactorS_;
        int64_t sLength =
            (ubLoopIdx != ubLoopCntS - 1) ? ubFactorS_ : sBlockLength_ - (ubLoopIdx * ubFactorS_);
        ProcessInSLoop(sStart, sLength);
    }
}

template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void
InplacePartialRotaryMulGradABAAndBA<TDY, TCOS, IsBroadcast>::ProcessInSLoop(
    int64_t sStart, int64_t sLength)
{
    uint32_t ubLoopCntB =
        AscendC::Ceil(static_cast<uint64_t>(bBlockLength_), static_cast<uint64_t>(ubFactorB_));

    if constexpr (IsBroadcast) {
        DataCopyExtParams cosParams = {static_cast<uint16_t>(sLength * dSplitCoef_), dSplitSizeCos_, 0, 0, 0};
        DataCopyPadExtParams<TCOS> padParams{false, 0, 0, 0};
        int64_t cosGmOffset = sStart * tilingData_->sliceLength;

        LocalTensor<TCOS> cosBuffer = cosInQueue_.template AllocTensor<TCOS>();
        DataCopyPad(cosBuffer, cosGm_[cosGmOffset], cosParams, padParams);
        cosInQueue_.EnQue(cosBuffer);
        cosBuffer = cosInQueue_.template DeQue<TCOS>();

        LocalTensor<TCOS> sinBuffer = sinInQueue_.template AllocTensor<TCOS>();
        DataCopyPad(sinBuffer, sinGm_[cosGmOffset], cosParams, padParams);
        sinInQueue_.EnQue(sinBuffer);
        sinBuffer = sinInQueue_.template DeQue<TCOS>();

        for (uint32_t ubLoopIdx = 0; ubLoopIdx < ubLoopCntB; ubLoopIdx++) {
            int64_t bStart = bBlockStart_ + ubLoopIdx * ubFactorB_;
            int64_t bLength =
                (ubLoopIdx != ubLoopCntB - 1) ? ubFactorB_ : bBlockLength_ - (ubLoopIdx * ubFactorB_);
            ProcessInSBLoop(sStart, sLength, bStart, bLength, cosBuffer, sinBuffer);
        }
        cosInQueue_.FreeTensor(cosBuffer);
        sinInQueue_.FreeTensor(sinBuffer);
    } else {
        for (uint32_t ubLoopIdx = 0; ubLoopIdx < ubLoopCntB; ubLoopIdx++) {
            int64_t bStart = bBlockStart_ + ubLoopIdx * ubFactorB_;
            int64_t bLength =
                (ubLoopIdx != ubLoopCntB - 1) ? ubFactorB_ : bBlockLength_ - (ubLoopIdx * ubFactorB_);

            DataCopyExtParams cosParams = {static_cast<uint16_t>(sLength * dSplitCoef_), dSplitSizeCos_, 0, 0, 0};
            DataCopyPadExtParams<TCOS> padParams{false, 0, 0, 0};
            int64_t cosGmOffset =
                bStart * tilingData_->s * tilingData_->sliceLength + sStart * tilingData_->sliceLength;

            LocalTensor<TCOS> cosBuffer = cosInQueue_.template AllocTensor<TCOS>();
            DataCopyPad(cosBuffer, cosGm_[cosGmOffset], cosParams, padParams);
            cosInQueue_.EnQue(cosBuffer);
            cosBuffer = cosInQueue_.template DeQue<TCOS>();

            LocalTensor<TCOS> sinBuffer = sinInQueue_.template AllocTensor<TCOS>();
            DataCopyPad(sinBuffer, sinGm_[cosGmOffset], cosParams, padParams);
            sinInQueue_.EnQue(sinBuffer);
            sinBuffer = sinInQueue_.template DeQue<TCOS>();

            ProcessInSBLoop(sStart, sLength, bStart, bLength, cosBuffer, sinBuffer);
            cosInQueue_.FreeTensor(cosBuffer);
            sinInQueue_.FreeTensor(sinBuffer);
        }
    }
}

template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void
InplacePartialRotaryMulGradABAAndBA<TDY, TCOS, IsBroadcast>::ProcessInSBLoop(
    int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength,
    const LocalTensor<TCOS> &cosBuffer, const LocalTensor<TCOS> &sinBuffer)
{
    uint32_t ubLoopCntN = static_cast<uint32_t>(AscendC::Ceil(
        static_cast<uint64_t>(tilingData_->n), static_cast<uint64_t>(ubFactorN_)));

    for (uint32_t nLoopIdx = 0; nLoopIdx < ubLoopCntN; nLoopIdx++) {
        uint32_t nLength = (nLoopIdx != ubLoopCntN - 1)
                               ? static_cast<uint32_t>(ubFactorN_)
                               : static_cast<uint32_t>(tilingData_->n - nLoopIdx * ubFactorN_);

        int64_t gradOffset = bStart * tilingData_->n * tilingData_->s * tilingData_->d +
                             nLoopIdx * ubFactorN_ * tilingData_->s * tilingData_->d +
                             sStart * tilingData_->d + tilingData_->sliceStart;

        DataCopyExtParams inParams = {static_cast<uint16_t>(sLength * bLength * nLength * dSplitCoef_), dSplitSizeDy_,
                                      0, 0, 0};
        DataCopyExtParams outParams = {static_cast<uint16_t>(sLength * bLength * nLength * dSplitCoef_), dSplitSizeDy_,
                                       0, 0, 0};

        LocalTensor<TDY> inBuffer = gradInQueue_.template AllocTensor<TDY>();
        LocalTensor<TDY> outBuffer = xGradOutQueue_.template AllocTensor<TDY>();

        DataCopyPad(inBuffer, gradGm_[gradOffset], inParams, padParamsDy_);
        gradInQueue_.EnQue(inBuffer);
        inBuffer = gradInQueue_.template DeQue<TDY>();

        auto *inUb = (__local_mem__ TDY *)inBuffer.GetPhyAddr();
        auto *outUb = (__local_mem__ TDY *)outBuffer.GetPhyAddr();
        auto *cosUb = (__local_mem__ TCOS *)cosBuffer.GetPhyAddr();
        auto *sinUb = (__local_mem__ TCOS *)sinBuffer.GetPhyAddr();

        if (tilingData_->rotaryMode == ROTARY_MODE_INTERLEAVE) {
            BatchInterleaveModeGradVF<TDY, TCOS, IsBroadcast>(
                inUb, cosUb, sinUb, outUb, static_cast<uint16_t>(sLength), static_cast<uint16_t>(bLength),
                static_cast<uint16_t>(nLength), tilingData_->sliceLength, ubFactorS_, ubFactorN_);
        }

        gradInQueue_.FreeTensor(inBuffer);
        xGradOutQueue_.EnQue(outBuffer);
        outBuffer = xGradOutQueue_.template DeQue<TDY>();
        DataCopyPad(dxGm_[gradOffset], outBuffer, outParams);
        xGradOutQueue_.FreeTensor(outBuffer);
    }
}

} // namespace InplacePartialRotaryMulGrad
#endif
