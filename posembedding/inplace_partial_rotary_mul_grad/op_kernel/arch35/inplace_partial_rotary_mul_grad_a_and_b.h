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
                                GM_ADDR cos, GM_ADDR sin, GM_ADDR dyOut);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInCosAndSin(int64_t bStart, int64_t bLength);
    __aicore__ inline void Compute(const LocalTensor<TCOS> &cosBuffer, const LocalTensor<TCOS> &sinBuffer,
                                    const LocalTensor<TDY> &inBuffer, const LocalTensor<TDY> &outBuffer,
                                    int64_t bUbLength);
    __aicore__ inline void ProcessInLoop(const LocalTensor<TCOS> &cosBuffer,
                                          const LocalTensor<TCOS> &sinBuffer, int64_t bUbStart, int64_t bUbLength);

    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, 2> gradInQueue_;
    TQue<QuePosition::VECOUT, 2> xGradOutQueue_;
    TQue<QuePosition::VECIN, IsBroadcast ? 1 : 2> cosInQueue_;
    TQue<QuePosition::VECIN, IsBroadcast ? 1 : 2> sinInQueue_;

    GlobalTensor<TDY> gradGm_;
    GlobalTensor<TDY> dyOutGm_;
    GlobalTensor<TCOS> cosGm_;
    GlobalTensor<TCOS> sinGm_;

    const InplacePartialRotaryMulGradRegbaseParams *tilingData_{nullptr};
    DataCopyPadExtParams<TDY> padParamsX_{false, 0, 0, static_cast<TDY>(0)};
    DataCopyPadExtParams<TCOS> padParamsCos_{false, 0, 0, static_cast<TCOS>(0)};
    int64_t blockIdx_{0};
    int64_t bBlockStart_{0};
    int64_t bBlockLength_{0};
    uint32_t dSplitSizeX_{0};
    uint32_t dSplitSizeCos_{0};
    int64_t dAlign_{0};
    int64_t ubFactorB_{0};
};

template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadcast>::Init(
    TPipe *pipe, const InplacePartialRotaryMulGradRegbaseParams *tiling, GM_ADDR dy, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR dyOut)
{
    pipe_ = pipe;
    tilingData_ = tiling;
    blockIdx_ = GetBlockIdx();

    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    constexpr uint32_t maxTypeSize =
        sizeof(TDY) > sizeof(TCOS) ? sizeof(TDY) : sizeof(TCOS);
    dAlign_ = AscendC::Ceil(tilingData_->sliceLength / tilingData_->dSplitCoef,
                              static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / maxTypeSize)) *
               tilingData_->dSplitCoef;
    dSplitSizeX_ = tilingData_->sliceLength / tilingData_->dSplitCoef * sizeof(TDY);
    dSplitSizeCos_ = tilingData_->sliceLength / tilingData_->dSplitCoef * sizeof(TCOS);
    ubFactorB_ = tilingData_->ubFactorB;

    gradGm_.SetGlobalBuffer((__gm__ TDY *)dy);
    dyOutGm_.SetGlobalBuffer((__gm__ TDY *)dyOut);
    cosGm_.SetGlobalBuffer((__gm__ TCOS *)cos);
    sinGm_.SetGlobalBuffer((__gm__ TCOS *)sin);

    int64_t bufferSizeX = dAlign_ * sizeof(TDY) * ubFactorB_;
    int64_t bufferSizeCos = dAlign_ * sizeof(TCOS) * ubFactorB_;
    pipe_->InitBuffer(gradInQueue_, 2, bufferSizeX);
    pipe_->InitBuffer(xGradOutQueue_, 2, bufferSizeX);

    if constexpr (IsBroadcast) {
        pipe_->InitBuffer(cosInQueue_, 1, dAlign_ * sizeof(TCOS));
        pipe_->InitBuffer(sinInQueue_, 1, dAlign_ * sizeof(TCOS));
    } else {
        pipe_->InitBuffer(cosInQueue_, 2, bufferSizeCos);
        pipe_->InitBuffer(sinInQueue_, 2, bufferSizeCos);
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
    DataCopyExtParams cosParams = {static_cast<uint16_t>(bLength * tilingData_->dSplitCoef), dSplitSizeCos_, 0, 0, 0};
    LocalTensor<TCOS> cosUb = cosInQueue_.template AllocTensor<TCOS>();
    DataCopyPad(cosUb, cosGm_[bStart * tilingData_->sliceLength], cosParams, padParamsCos_);
    cosInQueue_.EnQue(cosUb);

    LocalTensor<TCOS> sinUb = sinInQueue_.template AllocTensor<TCOS>();
    DataCopyPad(sinUb, sinGm_[bStart * tilingData_->sliceLength], cosParams, padParamsCos_);
    sinInQueue_.EnQue(sinUb);
}

template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<TDY, TCOS, IsBroadcast>::Compute(
    const LocalTensor<TCOS> &cosBuffer, const LocalTensor<TCOS> &sinBuffer,
    const LocalTensor<TDY> &inBuffer, const LocalTensor<TDY> &outBuffer, int64_t bUbLength)
{
    if constexpr (IsBroadcast) {
        if (tilingData_->rotaryMode == ROTARY_MODE_HALF) {
            HalfGradAlignVF<TDY, TCOS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                   tilingData_->sliceLength,
                                                   static_cast<uint16_t>(bUbLength), 1);
        } else if (tilingData_->rotaryMode == ROTARY_MODE_INTERLEAVE) {
            InterleaveModeGradVF<TDY, TCOS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                        tilingData_->sliceLength,
                                                        static_cast<uint16_t>(bUbLength), 1);
        } else if (tilingData_->rotaryMode == ROTARY_MODE_QUARTER) {
            QuarterGradAlignVF<TDY, TCOS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                      tilingData_->sliceLength,
                                                      static_cast<uint16_t>(bUbLength), 1);
        } else {
            DeepSeekInterleaveModeGradVF<TDY, TCOS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                            tilingData_->sliceLength,
                                                            static_cast<uint16_t>(bUbLength), 1);
        }
    } else {
        auto *inUb = (__local_mem__ TDY *)inBuffer.GetPhyAddr();
        auto *outUb = (__local_mem__ TDY *)outBuffer.GetPhyAddr();
        auto *cosUb = (__local_mem__ TCOS *)cosBuffer.GetPhyAddr();
        auto *sinUb = (__local_mem__ TCOS *)sinBuffer.GetPhyAddr();
        if (tilingData_->rotaryMode == ROTARY_MODE_HALF) {
            BatchHalfGradAlignVF<TDY, TCOS, IsBroadcast>(
                inUb, cosUb, sinUb, outUb, static_cast<uint16_t>(bUbLength), 1, 1, tilingData_->sliceLength,
                ubFactorB_, 1);
        } else if (tilingData_->rotaryMode == ROTARY_MODE_INTERLEAVE) {
            BatchInterleaveModeGradVF<TDY, TCOS, IsBroadcast>(
                inUb, cosUb, sinUb, outUb, static_cast<uint16_t>(bUbLength), 1, 1, tilingData_->sliceLength,
                ubFactorB_, 1);
        } else if (tilingData_->rotaryMode == ROTARY_MODE_QUARTER) {
            BatchQuarterGradAlignVF<TDY, TCOS, IsBroadcast>(
                inUb, cosUb, sinUb, outUb, static_cast<uint16_t>(bUbLength), 1, 1, tilingData_->sliceLength,
                ubFactorB_, 1);
        } else {
            BatchDeepSeekInterleaveModeGradVF<TDY, TCOS, IsBroadcast>(
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
    uint32_t strideBytesX =
        static_cast<uint32_t>((tilingData_->d - tilingData_->sliceLength) * sizeof(TDY));
    uint32_t blockLenX = static_cast<uint32_t>(tilingData_->sliceLength * sizeof(TDY));

    DataCopyExtParams inParams = {static_cast<uint16_t>(bUbLength), blockLenX, strideBytesX, 0, 0};
    DataCopyExtParams outParams = {static_cast<uint16_t>(bUbLength), blockLenX, 0, strideBytesX, 0};

    LocalTensor<TDY> inBuffer = gradInQueue_.template AllocTensor<TDY>();
    LocalTensor<TDY> outBuffer = xGradOutQueue_.template AllocTensor<TDY>();

    DataCopyPad(inBuffer, gradGm_[gradGmOffset], inParams, padParamsX_);
    gradInQueue_.EnQue(inBuffer);
    inBuffer = gradInQueue_.template DeQue<TDY>();

    Compute(cosBuffer, sinBuffer, inBuffer, outBuffer, bUbLength);

    gradInQueue_.FreeTensor(inBuffer);
    xGradOutQueue_.EnQue(outBuffer);
    outBuffer = xGradOutQueue_.template DeQue<TDY>();
    DataCopyPad(dyOutGm_[gradGmOffset], outBuffer, outParams);
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
