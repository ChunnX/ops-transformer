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

template <typename DTYPE_DY, typename DTYPE_COS, bool IsBroadcast>
class InplacePartialRotaryMulGradABAAndBA {
public:
    __aicore__ inline InplacePartialRotaryMulGradABAAndBA() {}
    __aicore__ inline void Init(TPipe *pipe, const InplacePartialRotaryMulGradRegbaseParams *tiling, GM_ADDR dy,
                                GM_ADDR cos, GM_ADDR sin, GM_ADDR dyOut);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessInSLoop(int64_t sStart, int64_t sLength);
    __aicore__ inline void ProcessInSBLoop(int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength,
                                            const LocalTensor<DTYPE_COS> &cosBuffer,
                                            const LocalTensor<DTYPE_COS> &sinBuffer);

    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, 2> gradInQueue_;
    TQue<QuePosition::VECOUT, 2> xGradOutQueue_;
    TQue<QuePosition::VECIN, 2> cosInQueue_;
    TQue<QuePosition::VECIN, 2> sinInQueue_;

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
    int64_t sBlockStart_{0};
    int64_t sBlockLength_{0};
    uint32_t dSplitSizeX_{0};
    uint32_t dSplitSizeCos_{0};
    int64_t dAlign_{0};
    int64_t ubFactorB_{0};
    int64_t ubFactorS_{0};
    int64_t ubFactorN_{0};
};

template <typename DTYPE_DY, typename DTYPE_COS, bool IsBroadcast>
__aicore__ inline void
InplacePartialRotaryMulGradABAAndBA<DTYPE_DY, DTYPE_COS, IsBroadcast>::Init(
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
    ubFactorS_ = tilingData_->ubFactorS;
    ubFactorN_ = tilingData_->ubFactorN;

    gradGm_.SetGlobalBuffer((__gm__ DTYPE_DY *)dy);
    dyOutGm_.SetGlobalBuffer((__gm__ DTYPE_DY *)dyOut);
    cosGm_.SetGlobalBuffer((__gm__ DTYPE_COS *)cos);
    sinGm_.SetGlobalBuffer((__gm__ DTYPE_COS *)sin);

    int64_t bufferSizeBS = dAlign_ * sizeof(DTYPE_DY) * ubFactorB_ * ubFactorS_;
    pipe_->InitBuffer(gradInQueue_, 2, bufferSizeBS * ubFactorN_);
    pipe_->InitBuffer(xGradOutQueue_, 2, bufferSizeBS * ubFactorN_);

    if constexpr (IsBroadcast) {
        pipe_->InitBuffer(cosInQueue_, 2, dAlign_ * sizeof(DTYPE_COS) * ubFactorS_);
        pipe_->InitBuffer(sinInQueue_, 2, dAlign_ * sizeof(DTYPE_COS) * ubFactorS_);
    } else {
        int64_t bufferSizeCosBS = dAlign_ * sizeof(DTYPE_COS) * ubFactorB_ * ubFactorS_;
        pipe_->InitBuffer(cosInQueue_, 2, bufferSizeCosBS);
        pipe_->InitBuffer(sinInQueue_, 2, bufferSizeCosBS);
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

template <typename DTYPE_DY, typename DTYPE_COS, bool IsBroadcast>
__aicore__ inline void
InplacePartialRotaryMulGradABAAndBA<DTYPE_DY, DTYPE_COS, IsBroadcast>::Process()
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

template <typename DTYPE_DY, typename DTYPE_COS, bool IsBroadcast>
__aicore__ inline void
InplacePartialRotaryMulGradABAAndBA<DTYPE_DY, DTYPE_COS, IsBroadcast>::ProcessInSLoop(
    int64_t sStart, int64_t sLength)
{
    uint32_t ubLoopCntB =
        AscendC::Ceil(static_cast<uint64_t>(bBlockLength_), static_cast<uint64_t>(ubFactorB_));

    if constexpr (IsBroadcast) {
        // cos/sin 的 B=1，只拷一份
        DataCopyExtParams cosParams = {static_cast<uint16_t>(sLength * tilingData_->dSplitCoef), dSplitSizeCos_, 0, 0,
                                       0};
        int64_t cosGmOffset = sStart * tilingData_->sliceLength;

        LocalTensor<DTYPE_COS> cosBuffer = cosInQueue_.AllocTensor<DTYPE_COS>();
        DataCopyPad(cosBuffer, cosGm_[cosGmOffset], cosParams, padParamsCos_);
        cosInQueue_.EnQue(cosBuffer);
        cosBuffer = cosInQueue_.DeQue<DTYPE_COS>();

        LocalTensor<DTYPE_COS> sinBuffer = sinInQueue_.AllocTensor<DTYPE_COS>();
        DataCopyPad(sinBuffer, sinGm_[cosGmOffset], cosParams, padParamsCos_);
        sinInQueue_.EnQue(sinBuffer);
        sinBuffer = sinInQueue_.DeQue<DTYPE_COS>();

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

            DataCopyExtParams cosParams = {static_cast<uint16_t>(sLength * tilingData_->dSplitCoef), dSplitSizeCos_,
                                           0, 0, 0};
            int64_t cosGmOffset =
                bStart * tilingData_->s * tilingData_->sliceLength + sStart * tilingData_->sliceLength;

            LocalTensor<DTYPE_COS> cosBuffer = cosInQueue_.AllocTensor<DTYPE_COS>();
            DataCopyPad(cosBuffer, cosGm_[cosGmOffset], cosParams, padParamsCos_);
            cosInQueue_.EnQue(cosBuffer);
            cosBuffer = cosInQueue_.DeQue<DTYPE_COS>();

            LocalTensor<DTYPE_COS> sinBuffer = sinInQueue_.AllocTensor<DTYPE_COS>();
            DataCopyPad(sinBuffer, sinGm_[cosGmOffset], cosParams, padParamsCos_);
            sinInQueue_.EnQue(sinBuffer);
            sinBuffer = sinInQueue_.DeQue<DTYPE_COS>();

            ProcessInSBLoop(sStart, sLength, bStart, bLength, cosBuffer, sinBuffer);
            cosInQueue_.FreeTensor(cosBuffer);
            sinInQueue_.FreeTensor(sinBuffer);
        }
    }
}

template <typename DTYPE_DY, typename DTYPE_COS, bool IsBroadcast>
__aicore__ inline void
InplacePartialRotaryMulGradABAAndBA<DTYPE_DY, DTYPE_COS, IsBroadcast>::ProcessInSBLoop(
    int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength,
    const LocalTensor<DTYPE_COS> &cosBuffer, const LocalTensor<DTYPE_COS> &sinBuffer)
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

        int64_t totalCount = sLength * bLength * nLength;
        uint32_t strideBytesX =
            static_cast<uint32_t>((tilingData_->d - tilingData_->sliceLength) * sizeof(DTYPE_DY));
        uint32_t blockLenX = static_cast<uint32_t>(tilingData_->sliceLength * sizeof(DTYPE_DY));

        DataCopyExtParams inParams = {static_cast<uint16_t>(totalCount), blockLenX, strideBytesX, 0, 0};
        DataCopyExtParams outParams = {static_cast<uint16_t>(totalCount), blockLenX, 0, strideBytesX, 0};

        LocalTensor<DTYPE_DY> inBuffer = gradInQueue_.AllocTensor<DTYPE_DY>();
        LocalTensor<DTYPE_DY> outBuffer = xGradOutQueue_.AllocTensor<DTYPE_DY>();

        DataCopyPad(inBuffer, gradGm_[gradOffset], inParams, padParamsX_);
        gradInQueue_.EnQue(inBuffer);
        inBuffer = gradInQueue_.DeQue<DTYPE_DY>();

        auto *inUb = const_cast<__local_mem__ DTYPE_DY *>(inBuffer.GetPhyAddr());
        auto *outUb = const_cast<__local_mem__ DTYPE_DY *>(outBuffer.GetPhyAddr());
        auto *cosUb = const_cast<__local_mem__ DTYPE_COS *>(cosBuffer.GetPhyAddr());
        auto *sinUb = const_cast<__local_mem__ DTYPE_COS *>(sinBuffer.GetPhyAddr());

        if (tilingData_->rotaryMode == ROTARY_MODE_HALF) {
            BatchHalfGradAlignVF<DTYPE_DY, DTYPE_COS, IsBroadcast>(
                inUb, cosUb, sinUb, outUb, static_cast<uint16_t>(sLength), static_cast<uint16_t>(bLength),
                static_cast<uint16_t>(nLength), tilingData_->sliceLength, ubFactorS_, ubFactorN_);
        } else if (tilingData_->rotaryMode == ROTARY_MODE_INTERLEAVE) {
            BatchInterleaveModeGradVF<DTYPE_DY, DTYPE_COS, IsBroadcast>(
                inUb, cosUb, sinUb, outUb, static_cast<uint16_t>(sLength), static_cast<uint16_t>(bLength),
                static_cast<uint16_t>(nLength), tilingData_->sliceLength, ubFactorS_, ubFactorN_);
        } else if (tilingData_->rotaryMode == ROTARY_MODE_QUARTER) {
            BatchQuarterGradAlignVF<DTYPE_DY, DTYPE_COS, IsBroadcast>(
                inUb, cosUb, sinUb, outUb, static_cast<uint16_t>(sLength), static_cast<uint16_t>(bLength),
                static_cast<uint16_t>(nLength), tilingData_->sliceLength, ubFactorS_, ubFactorN_);
        } else {
            BatchDeepSeekInterleaveModeGradVF<DTYPE_DY, DTYPE_COS, IsBroadcast>(
                inUb, cosUb, sinUb, outUb, static_cast<uint16_t>(sLength), static_cast<uint16_t>(bLength),
                static_cast<uint16_t>(nLength), tilingData_->sliceLength, ubFactorS_, ubFactorN_);
        }

        gradInQueue_.FreeTensor(inBuffer);
        xGradOutQueue_.EnQue(outBuffer);
        outBuffer = xGradOutQueue_.DeQue<DTYPE_DY>();
        DataCopyPad(dyOutGm_[gradOffset], outBuffer, outParams);
        xGradOutQueue_.FreeTensor(outBuffer);
    }
}

} // namespace InplacePartialRotaryMulGrad
#endif
