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
 * \file inplace_partial_rotary_mul_grad_bab.h
 * \brief BAB template kernel for InplacePartialRotaryMulGrad (BSND layout, cosb_==1)
 */
#ifndef __INPLACE_PARTIAL_ROTARY_MUL_GRAD_BAB_H__
#define __INPLACE_PARTIAL_ROTARY_MUL_GRAD_BAB_H__

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "inplace_partial_rotary_mul_grad_tiling_data.h"
#include "inplace_partial_rotary_mul_grad_common.h"

namespace InplacePartialRotaryMulGrad {
using namespace AscendC;

template <typename TDY, typename TCOS>
class InplacePartialRotaryMulGradBAB {
public:
    __aicore__ inline InplacePartialRotaryMulGradBAB() {}
    __aicore__ inline void Init(TPipe *pipe, const InplacePartialRotaryMulGradRegbaseParams *tiling, GM_ADDR dy,
                                GM_ADDR cos, GM_ADDR sin, GM_ADDR dyOut);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessNLoop(int64_t sIdx, uint32_t currSNum, int64_t bIdx);
    __aicore__ inline void ProcessN(int64_t sIdx, uint32_t currSNum, int64_t bIdx, int64_t baseOffset);

    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, 2> gradInQue_;
    TQue<QuePosition::VECIN, 2> cosInQue_;
    TQue<QuePosition::VECIN, 2> sinInQue_;
    TQue<QuePosition::VECOUT, 2> xGradOutQue_;

    GlobalTensor<TDY> gradGm_;
    GlobalTensor<TDY> dyOutGm_;
    GlobalTensor<TCOS> cosGm_;
    GlobalTensor<TCOS> sinGm_;

    const InplacePartialRotaryMulGradRegbaseParams *tilingData_{nullptr};
    DataCopyPadExtParams<TDY> padParamsX_{false, 0, 0, static_cast<TDY>(0)};
    DataCopyPadExtParams<TCOS> padParamsCos_{false, 0, 0, static_cast<TCOS>(0)};
    uint32_t dSplitSizeX_{0};
    uint32_t dSplitSizeCos_{0};
    int64_t bIdx_{0};
    int64_t sIdx_{0};
    int64_t bNum_{0};
    int64_t sNum_{0};
    int64_t dAlign_{0};
    uint32_t ubFactorN_{0};
};

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradBAB<TDY, TCOS>::Init(
    TPipe *pipe, const InplacePartialRotaryMulGradRegbaseParams *tiling, GM_ADDR dy, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR dyOut)
{
    pipe_ = pipe;
    tilingData_ = tiling;

    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }

    constexpr uint32_t maxTypeSize =
        sizeof(TDY) > sizeof(TCOS) ? sizeof(TDY) : sizeof(TCOS);
    dAlign_ = AscendC::Ceil(tilingData_->sliceLength / tilingData_->dSplitCoef,
                              static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / maxTypeSize)) *
               tilingData_->dSplitCoef;
    dSplitSizeX_ = tilingData_->sliceLength / tilingData_->dSplitCoef * sizeof(TDY);
    dSplitSizeCos_ = tilingData_->sliceLength / tilingData_->dSplitCoef * sizeof(TCOS);
    ubFactorN_ = tilingData_->ubFactorN;

    gradGm_.SetGlobalBuffer((__gm__ TDY *)dy);
    dyOutGm_.SetGlobalBuffer((__gm__ TDY *)dyOut);
    cosGm_.SetGlobalBuffer((__gm__ TCOS *)cos);
    sinGm_.SetGlobalBuffer((__gm__ TCOS *)sin);

    int64_t bufferSizeX = dAlign_ * sizeof(TDY) * tilingData_->ubFactorS;
    int64_t bufferSizeCos = dAlign_ * sizeof(TCOS) * tilingData_->ubFactorS;
    pipe_->InitBuffer(gradInQue_, 2, bufferSizeX * ubFactorN_);
    pipe_->InitBuffer(xGradOutQue_, 2, bufferSizeX * ubFactorN_);
    pipe_->InitBuffer(cosInQue_, 2, bufferSizeCos);
    pipe_->InitBuffer(sinInQue_, 2, bufferSizeCos);
}

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradBAB<TDY, TCOS>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }

    bIdx_ = GetBlockIdx() % tilingData_->blockNumB;
    sIdx_ = GetBlockIdx() / tilingData_->blockNumB;
    bNum_ = (bIdx_ == tilingData_->blockNumB - 1 && tilingData_->b % tilingData_->blockFactorB != 0)
                ? tilingData_->b % tilingData_->blockFactorB : tilingData_->blockFactorB;
    sNum_ = (sIdx_ == tilingData_->blockNumS - 1 && tilingData_->s % tilingData_->blockFactorS != 0)
                ? tilingData_->s % tilingData_->blockFactorS : tilingData_->blockFactorS;

    uint32_t sLoopCnt = AscendC::Ceil(static_cast<uint64_t>(sNum_), static_cast<uint64_t>(tilingData_->ubFactorS));

    for (int64_t bIdx = bIdx_; bIdx < bIdx_ + bNum_; bIdx++) {
        for (uint32_t loopIdx = 0; loopIdx < sLoopCnt; loopIdx++) {
            uint32_t currSNum = (loopIdx != sLoopCnt - 1) ? tilingData_->ubFactorS
                                    : sNum_ - (loopIdx * tilingData_->ubFactorS);
            ProcessNLoop(sIdx_ + loopIdx * tilingData_->ubFactorS, currSNum, bIdx);
        }
    }
}

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradBAB<TDY, TCOS>::ProcessNLoop(
    int64_t sIdx, uint32_t currSNum, int64_t bIdx)
{
    LocalTensor<TCOS> cosBuffer = cosInQue_.AllocTensor<TCOS>();
    LocalTensor<TCOS> sinBuffer = sinInQue_.AllocTensor<TCOS>();

    int64_t cosGmOffset = sIdx * tilingData_->sliceLength;
    DataCopyExtParams cosCopyParams = {static_cast<uint16_t>(currSNum * tilingData_->dSplitCoef), dSplitSizeCos_, 0, 0,
                                       0};

    DataCopyPad(cosBuffer, cosGm_[cosGmOffset], cosCopyParams, padParamsCos_);
    cosInQue_.EnQue(cosBuffer);
    cosBuffer = cosInQue_.DeQue<TCOS>();

    DataCopyPad(sinBuffer, sinGm_[cosGmOffset], cosCopyParams, padParamsCos_);
    sinInQue_.EnQue(sinBuffer);
    sinBuffer = sinInQue_.DeQue<TCOS>();

    int64_t baseOffset = (bIdx * tilingData_->s + sIdx) * tilingData_->n * tilingData_->d;
    ProcessN(sIdx, currSNum, bIdx, baseOffset);

    cosInQue_.FreeTensor(cosBuffer);
    sinInQue_.FreeTensor(sinBuffer);
}

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradBAB<TDY, TCOS>::ProcessN(
    int64_t sIdx, uint32_t currSNum, int64_t bIdx, int64_t baseOffset)
{
    uint32_t ubLoopNumN = static_cast<uint32_t>(tilingData_->ubLoopNumN);
    LocalTensor<TCOS> cosBuffer = cosInQue_.DeQue<TCOS>();
    LocalTensor<TCOS> sinBuffer = sinInQue_.DeQue<TCOS>();

    for (uint32_t idxN = 0; idxN < ubLoopNumN; idxN++) {
        uint32_t currDNum =
            (idxN == ubLoopNumN - 1) ? static_cast<uint32_t>(tilingData_->ubTailFactorN) : ubFactorN_;
        int64_t offset = baseOffset + idxN * ubFactorN_ * tilingData_->d + tilingData_->sliceStart;

        LocalTensor<TDY> inBuffer = gradInQue_.AllocTensor<TDY>();
        uint32_t strideBytesX =
            static_cast<uint32_t>((tilingData_->d - tilingData_->sliceLength) * sizeof(TDY));
        uint32_t blockLenX = static_cast<uint32_t>(tilingData_->sliceLength * sizeof(TDY));
        DataCopyExtParams copyInParams = {static_cast<uint16_t>(currSNum * currDNum), blockLenX, strideBytesX, 0, 0};
        DataCopyPad(inBuffer, gradGm_[offset], copyInParams, padParamsX_);
        gradInQue_.EnQue(inBuffer);
        inBuffer = gradInQue_.DeQue<TDY>();

        LocalTensor<TDY> outBuffer = xGradOutQue_.AllocTensor<TDY>();

        // 根据 mode 选择 VF
        if (tilingData_->rotaryMode == ROTARY_MODE_HALF) {
            HalfGradAlignVF<TDY, TCOS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                   tilingData_->sliceLength,
                                                   static_cast<uint16_t>(currSNum), static_cast<uint16_t>(currDNum));
        } else if (tilingData_->rotaryMode == ROTARY_MODE_INTERLEAVE) {
            InterleaveModeGradVF<TDY, TCOS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                        tilingData_->sliceLength, static_cast<uint16_t>(currSNum),
                                                        static_cast<uint16_t>(currDNum));
        } else if (tilingData_->rotaryMode == ROTARY_MODE_QUARTER) {
            QuarterGradAlignVF<TDY, TCOS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                      tilingData_->sliceLength,
                                                      static_cast<uint16_t>(currSNum), static_cast<uint16_t>(currDNum));
        } else {
            DeepSeekInterleaveModeGradVF<TDY, TCOS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                            tilingData_->sliceLength, static_cast<uint16_t>(currSNum),
                                                            static_cast<uint16_t>(currDNum));
        }

        gradInQue_.FreeTensor(inBuffer);
        xGradOutQue_.EnQue(outBuffer);
        outBuffer = xGradOutQue_.DeQue<TDY>();

        DataCopyExtParams copyOutParams = {static_cast<uint16_t>(currSNum * currDNum), blockLenX, 0, strideBytesX, 0};
        DataCopyPad(dyOutGm_[offset], outBuffer, copyOutParams);
        xGradOutQue_.FreeTensor(outBuffer);
    }
}

} // namespace InplacePartialRotaryMulGrad
#endif
