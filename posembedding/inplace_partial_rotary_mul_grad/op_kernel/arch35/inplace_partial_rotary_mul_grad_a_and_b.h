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

template <typename DTYPE_DY, typename DTYPE_COS, bool IsBroadcast>
class InplacePartialRotaryMulGradAAndB {
public:
    __aicore__ inline InplacePartialRotaryMulGradAAndB() {}
    __aicore__ inline void Init(TPipe *pipe, const InplacePartialRotaryMulGradRegbaseParams *tiling, GM_ADDR dy,
                                GM_ADDR cos, GM_ADDR sin, GM_ADDR dyOut);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInCosAndSin(int64_t bStart, int64_t bLength);
    __aicore__ inline void Compute(const LocalTensor<DTYPE_COS> &cosBuffer, const LocalTensor<DTYPE_COS> &sinBuffer,
                                    const LocalTensor<DTYPE_DY> &inBuffer, const LocalTensor<DTYPE_DY> &outBuffer,
                                    int64_t bUbLength);
    __aicore__ inline void ProcessInLoop(const LocalTensor<DTYPE_COS> &cosBuffer,
                                          const LocalTensor<DTYPE_COS> &sinBuffer, int64_t bUbStart, int64_t bUbLength);

    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, 2> gradInQueue_;
    TQue<QuePosition::VECOUT, 2> xGradOutQueue_;
    TQue<QuePosition::VECIN, IsBroadcast ? 1 : 2> cosInQueue_;
    TQue<QuePosition::VECIN, IsBroadcast ? 1 : 2> sinInQueue_;

    GlobalTensor<DTYPE_DY> gradGm_;
    GlobalTensor<DTYPE_DY> dyOutGm_;
    GlobalTensor<DTYPE_COS> cosGm_;
    GlobalTensor<DTYPE_COS> sinGm_;

    const InplacePartialRotaryMulGradRegbaseParams *tilingData_{nullptr};
    DataCopyPadExtParams<DTYPE_DY> padParamsX_{false, 0, 0, static_cast<DTYPE_DY>(0)};
    DataCopyPadExtParams<DTYPE_COS> padParamsCos_{false, 0, 0, static_cast<DTYPE_COS>(0)};
    int64_t blockIdx_{0};
    int64_t bBlockStart_{0};
    int64_t bBlockLength_{0};
    uint32_t dSplitSizeX_{0};
    uint32_t dSplitSizeCos_{0};
    int64_t dAlign_{0};
    int64_t ubFactorB_{0};
};

template <typename DTYPE_DY, typename DTYPE_COS, bool IsBroadcast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<DTYPE_DY, DTYPE_COS, IsBroadcast>::Init(
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
        sizeof(DTYPE_DY) > sizeof(DTYPE_COS) ? sizeof(DTYPE_DY) : sizeof(DTYPE_COS);
    dAlign_ = AscendC::Ceil(tilingData_->sliceLength / tilingData_->dSplitCoef,
                              static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / maxTypeSize)) *
               tilingData_->dSplitCoef;
    dSplitSizeX_ = tilingData_->sliceLength / tilingData_->dSplitCoef * sizeof(DTYPE_DY);
    dSplitSizeCos_ = tilingData_->sliceLength / tilingData_->dSplitCoef * sizeof(DTYPE_COS);
    ubFactorB_ = tilingData_->ubFactorB;

    gradGm_.SetGlobalBuffer((__gm__ DTYPE_DY *)dy);
    dyOutGm_.SetGlobalBuffer((__gm__ DTYPE_DY *)dyOut);
    cosGm_.SetGlobalBuffer((__gm__ DTYPE_COS *)cos);
    sinGm_.SetGlobalBuffer((__gm__ DTYPE_COS *)sin);

    int64_t bufferSizeX = dAlign_ * sizeof(DTYPE_DY) * ubFactorB_;
    int64_t bufferSizeCos = dAlign_ * sizeof(DTYPE_COS) * ubFactorB_;
    pipe_->InitBuffer(gradInQueue_, 2, bufferSizeX);
    pipe_->InitBuffer(xGradOutQueue_, 2, bufferSizeX);

    if constexpr (IsBroadcast) {
        pipe_->InitBuffer(cosInQueue_, 1, dAlign_ * sizeof(DTYPE_COS));
        pipe_->InitBuffer(sinInQueue_, 1, dAlign_ * sizeof(DTYPE_COS));
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

template <typename DTYPE_DY, typename DTYPE_COS, bool IsBroadcast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<DTYPE_DY, DTYPE_COS, IsBroadcast>::
    CopyInCosAndSin(int64_t bStart, int64_t bLength)
{
    DataCopyExtParams cosParams = {static_cast<uint16_t>(bLength * tilingData_->dSplitCoef), dSplitSizeCos_, 0, 0, 0};
    LocalTensor<DTYPE_COS> cosUb = cosInQueue_.AllocTensor<DTYPE_COS>();
    DataCopyPad(cosUb, cosGm_[bStart * tilingData_->sliceLength], cosParams, padParamsCos_);
    cosInQueue_.EnQue(cosUb);

    LocalTensor<DTYPE_COS> sinUb = sinInQueue_.AllocTensor<DTYPE_COS>();
    DataCopyPad(sinUb, sinGm_[bStart * tilingData_->sliceLength], cosParams, padParamsCos_);
    sinInQueue_.EnQue(sinUb);
}

template <typename DTYPE_DY, typename DTYPE_COS, bool IsBroadcast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<DTYPE_DY, DTYPE_COS, IsBroadcast>::Compute(
    const LocalTensor<DTYPE_COS> &cosBuffer, const LocalTensor<DTYPE_COS> &sinBuffer,
    const LocalTensor<DTYPE_DY> &inBuffer, const LocalTensor<DTYPE_DY> &outBuffer, int64_t bUbLength)
{
    auto *inUb = const_cast<__local_mem__ DTYPE_DY *>(inBuffer.GetPhyAddr());
    auto *outUb = const_cast<__local_mem__ DTYPE_DY *>(outBuffer.GetPhyAddr());
    auto *cosUb = const_cast<__local_mem__ DTYPE_COS *>(cosBuffer.GetPhyAddr());
    auto *sinUb = const_cast<__local_mem__ DTYPE_COS *>(sinBuffer.GetPhyAddr());

    if constexpr (IsBroadcast) {
        if (tilingData_->rotaryMode == ROTARY_MODE_HALF) {
            HalfGradAlignVF<DTYPE_DY, DTYPE_COS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                   tilingData_->sliceLength,
                                                   static_cast<uint16_t>(bUbLength), 1);
        } else if (tilingData_->rotaryMode == ROTARY_MODE_INTERLEAVE) {
            InterleaveModeGradVF<DTYPE_DY, DTYPE_COS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                        tilingData_->sliceLength,
                                                        static_cast<uint16_t>(bUbLength), 1);
        } else if (tilingData_->rotaryMode == ROTARY_MODE_QUARTER) {
            QuarterGradAlignVF<DTYPE_DY, DTYPE_COS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                      tilingData_->sliceLength,
                                                      static_cast<uint16_t>(bUbLength), 1);
        } else {
            DeepSeekInterleaveModeGradVF<DTYPE_DY, DTYPE_COS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                            tilingData_->sliceLength,
                                                            static_cast<uint16_t>(bUbLength), 1);
        }
    } else {
        if (tilingData_->rotaryMode == ROTARY_MODE_HALF) {
            BatchHalfGradAlignVF<DTYPE_DY, DTYPE_COS, IsBroadcast>(
                inUb, cosUb, sinUb, outUb, static_cast<uint16_t>(bUbLength), 1, 1, tilingData_->sliceLength,
                ubFactorB_, 1);
        } else if (tilingData_->rotaryMode == ROTARY_MODE_INTERLEAVE) {
            BatchInterleaveModeGradVF<DTYPE_DY, DTYPE_COS, IsBroadcast>(
                inUb, cosUb, sinUb, outUb, static_cast<uint16_t>(bUbLength), 1, 1, tilingData_->sliceLength,
                ubFactorB_, 1);
        } else if (tilingData_->rotaryMode == ROTARY_MODE_QUARTER) {
            BatchQuarterGradAlignVF<DTYPE_DY, DTYPE_COS, IsBroadcast>(
                inUb, cosUb, sinUb, outUb, static_cast<uint16_t>(bUbLength), 1, 1, tilingData_->sliceLength,
                ubFactorB_, 1);
        } else {
            BatchDeepSeekInterleaveModeGradVF<DTYPE_DY, DTYPE_COS, IsBroadcast>(
                inUb, cosUb, sinUb, outUb, static_cast<uint16_t>(bUbLength), 1, 1, tilingData_->sliceLength,
                ubFactorB_, 1);
        }
    }
}

template <typename DTYPE_DY, typename DTYPE_COS, bool IsBroadcast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<DTYPE_DY, DTYPE_COS, IsBroadcast>::ProcessInLoop(
    const LocalTensor<DTYPE_COS> &cosBuffer, const LocalTensor<DTYPE_COS> &sinBuffer, int64_t bUbStart,
    int64_t bUbLength)
{
    int64_t gradGmOffset = bUbStart * tilingData_->d + tilingData_->sliceStart;
    uint32_t strideBytesX =
        static_cast<uint32_t>((tilingData_->d - tilingData_->sliceLength) * sizeof(DTYPE_DY));
    uint32_t blockLenX = static_cast<uint32_t>(tilingData_->sliceLength * sizeof(DTYPE_DY));

    DataCopyExtParams inParams = {static_cast<uint16_t>(bUbLength), blockLenX, strideBytesX, 0, 0};
    DataCopyExtParams outParams = {static_cast<uint16_t>(bUbLength), blockLenX, 0, strideBytesX, 0};

    LocalTensor<DTYPE_DY> inBuffer = gradInQueue_.AllocTensor<DTYPE_DY>();
    LocalTensor<DTYPE_DY> outBuffer = xGradOutQueue_.AllocTensor<DTYPE_DY>();

    DataCopyPad(inBuffer, gradGm_[gradGmOffset], inParams, padParamsX_);
    gradInQueue_.EnQue(inBuffer);
    inBuffer = gradInQueue_.DeQue<DTYPE_DY>();

    Compute(cosBuffer, sinBuffer, inBuffer, outBuffer, bUbLength);

    gradInQueue_.FreeTensor(inBuffer);
    xGradOutQueue_.EnQue(outBuffer);
    outBuffer = xGradOutQueue_.DeQue<DTYPE_DY>();
    DataCopyPad(dyOutGm_[gradGmOffset], outBuffer, outParams);
    xGradOutQueue_.FreeTensor(outBuffer);
}

template <typename DTYPE_DY, typename DTYPE_COS, bool IsBroadcast>
__aicore__ inline void InplacePartialRotaryMulGradAAndB<DTYPE_DY, DTYPE_COS, IsBroadcast>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    uint32_t ubLoopCnt = AscendC::Ceil(static_cast<uint64_t>(bBlockLength_), static_cast<uint64_t>(ubFactorB_));

    if constexpr (IsBroadcast) {
        CopyInCosAndSin(0, 1);
        LocalTensor<DTYPE_COS> cosBuffer = cosInQueue_.DeQue<DTYPE_COS>();
        LocalTensor<DTYPE_COS> sinBuffer = sinInQueue_.DeQue<DTYPE_COS>();

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
            LocalTensor<DTYPE_COS> cosBuffer = cosInQueue_.DeQue<DTYPE_COS>();
            LocalTensor<DTYPE_COS> sinBuffer = sinInQueue_.DeQue<DTYPE_COS>();
            ProcessInLoop(cosBuffer, sinBuffer, bStart, bUbLength);
            cosInQueue_.FreeTensor(cosBuffer);
            sinInQueue_.FreeTensor(sinBuffer);
        }
    }
}

} // namespace InplacePartialRotaryMulGrad
#endif
