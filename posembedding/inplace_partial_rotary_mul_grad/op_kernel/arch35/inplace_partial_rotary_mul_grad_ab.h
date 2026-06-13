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
 * \file inplace_partial_rotary_mul_grad_ab.h
 * \brief AB template kernel for InplacePartialRotaryMulGrad (SBND layout)
 */

#ifndef __INPLACE_PARTIAL_ROTARY_MUL_GRAD_AB_H__
#define __INPLACE_PARTIAL_ROTARY_MUL_GRAD_AB_H__

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "inplace_partial_rotary_mul_grad_tiling_data.h"
#include "inplace_partial_rotary_mul_grad_common.h"

namespace InplacePartialRotaryMulGrad {
using namespace AscendC;

template <typename TDY, typename TCOS>
class InplacePartialRotaryMulGradAB {
public:
    __aicore__ inline InplacePartialRotaryMulGradAB() {}
    __aicore__ inline void Init(TPipe *pipe, const InplacePartialRotaryMulGradRegbaseABParams *tiling, GM_ADDR dy,
                                GM_ADDR cos, GM_ADDR sin, GM_ADDR dx);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessLoop(int64_t gradGmOffset, const LocalTensor<TCOS> &cosBuffer,
                                       const LocalTensor<TCOS> &sinBuffer, int64_t ubIdx, int64_t bsCount,
                                       int64_t nCount);

    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, 1> gradInQueue_;
    TQue<QuePosition::VECIN, 1> cosInQueue_;
    TQue<QuePosition::VECIN, 1> sinInQueue_;
    TQue<QuePosition::VECOUT, 1> dxQueue_;

    GlobalTensor<TDY> gradGm_;
    GlobalTensor<TDY> dxGm_;
    GlobalTensor<TCOS> cosGm_;
    GlobalTensor<TCOS> sinGm_;

    const InplacePartialRotaryMulGradRegbaseABParams *tilingData_{nullptr};
    DataCopyPadExtParams<TDY> padParamsX_{false, 0, 0, static_cast<TDY>(0)};
    DataCopyPadExtParams<TCOS> padParamsCos_{false, 0, 0, static_cast<TCOS>(0)};
    uint8_t DB_FLAG = 2;
    uint32_t dSplitSizeX_{0};
    uint32_t dSplitSizeCos_{0};
    int64_t bsBlockCount_{0};
    int64_t nBlockCount_{0};
    int64_t sliceAlignX_{0};
};

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradAB<TDY, TCOS>::Init(
    TPipe *pipe, const InplacePartialRotaryMulGradRegbaseABParams *tiling, GM_ADDR dy, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR dx)
{
    pipe_ = pipe;
    tilingData_ = tiling;

    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }

    dSplitSizeX_ = tilingData_->sliceLength / tilingData_->dSplitCoef * sizeof(TDY);
    dSplitSizeCos_ = tilingData_->sliceLength / tilingData_->dSplitCoef * sizeof(TCOS);

    int64_t blockDimBS = GetBlockIdx() / tilingData_->blockNumN;
    int64_t blockDimN = GetBlockIdx() % tilingData_->blockNumN;
    bsBlockCount_ =
        (blockDimBS == tilingData_->blockNumBS - 1) ? tilingData_->blockTailBS : tilingData_->blockFactorBS;
    nBlockCount_ =
        (blockDimN == tilingData_->blockNumN - 1) ? tilingData_->blockTailN : tilingData_->blockFactorN;

    // cos/sin 偏移：沿 BS 维连续 sliceLength 元素
    int64_t cosOffset = blockDimBS * tilingData_->blockFactorBS * tilingData_->sliceLength;
    // dx 偏移：沿 BS 维跨越完整 D，沿 N 维分块
    int64_t offset = blockDimBS * tilingData_->blockFactorBS * tilingData_->d;
    int64_t gradOffset = offset * tilingData_->n + blockDimN * tilingData_->blockFactorN * tilingData_->d +
                         tilingData_->sliceStart;

    cosGm_.SetGlobalBuffer((__gm__ TCOS *)cos + cosOffset);
    sinGm_.SetGlobalBuffer((__gm__ TCOS *)sin + cosOffset);
    gradGm_.SetGlobalBuffer((__gm__ TDY *)dy + gradOffset);
    dxGm_.SetGlobalBuffer((__gm__ TDY *)dx + gradOffset);

    // 统一按 max(sizeof(TDY), sizeof(TCOS)) 对齐分配所有 buffer
    // 确保 dy=fp16+cos=fp32 或 dy=fp32+cos=fp16 两种混合方向都能容纳
    constexpr uint32_t maxTypeSize =
        sizeof(TDY) > sizeof(TCOS) ? sizeof(TDY) : sizeof(TCOS);
    sliceAlignX_ =
        AscendC::Ceil(static_cast<uint64_t>(tilingData_->sliceLength * maxTypeSize),
                      static_cast<uint64_t>(BLOCK_TYPE_SIZE)) /
        sizeof(TDY);
    int64_t bufferSizeX = sliceAlignX_ * sizeof(TDY) * tilingData_->ubFactorBS;

    int64_t bufferSizeCos = sliceAlignX_ * sizeof(TCOS) * tilingData_->ubFactorBS;
    pipe_->InitBuffer(gradInQueue_, DB_FLAG, bufferSizeX * tilingData_->ubFactorN);
    pipe_->InitBuffer(dxQueue_, DB_FLAG, bufferSizeX * tilingData_->ubFactorN);
    pipe_->InitBuffer(cosInQueue_, DB_FLAG, bufferSizeCos);
    pipe_->InitBuffer(sinInQueue_, DB_FLAG, bufferSizeCos);
}

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradAB<TDY, TCOS>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }

    uint32_t bsLoopCnt = AscendC::Ceil(static_cast<uint64_t>(bsBlockCount_),
                                       static_cast<uint64_t>(tilingData_->ubFactorBS));
    uint32_t nLoopCnt =
        AscendC::Ceil(static_cast<uint64_t>(nBlockCount_), static_cast<uint64_t>(tilingData_->ubFactorN));

    for (uint32_t bsLoopIdx = 0; bsLoopIdx < bsLoopCnt; bsLoopIdx++) {
        int64_t gradGmOffset = bsLoopIdx * tilingData_->ubFactorBS * tilingData_->n * tilingData_->d;
        uint32_t currBSNum = (bsLoopIdx != bsLoopCnt - 1)
                                 ? tilingData_->ubFactorBS
                                 : bsBlockCount_ - (bsLoopIdx * tilingData_->ubFactorBS);

        // 搬运 cos/sin（沿 sliceLength，无 stride）
        DataCopyExtParams cosCopyParams = {
            static_cast<uint16_t>(currBSNum * tilingData_->dSplitCoef), dSplitSizeCos_, 0, 0, 0};

        LocalTensor<TCOS> cosBuffer = cosInQueue_.template AllocTensor<TCOS>();
        LocalTensor<TCOS> sinBuffer = sinInQueue_.template AllocTensor<TCOS>();

        DataCopyPad(cosBuffer,
                    cosGm_[bsLoopIdx * tilingData_->ubFactorBS * tilingData_->sliceLength], cosCopyParams,
                    padParamsCos_);
        cosInQueue_.EnQue(cosBuffer);
        cosBuffer = cosInQueue_.template DeQue<TCOS>();

        DataCopyPad(sinBuffer,
                    sinGm_[bsLoopIdx * tilingData_->ubFactorBS * tilingData_->sliceLength], cosCopyParams,
                    padParamsCos_);
        sinInQueue_.EnQue(sinBuffer);
        sinBuffer = sinInQueue_.template DeQue<TCOS>();

        // 内层 N 循环
        for (uint32_t nLoopIdx = 0; nLoopIdx < nLoopCnt; nLoopIdx++) {
            uint32_t currNNum = (nLoopIdx != nLoopCnt - 1)
                                    ? tilingData_->ubFactorN
                                    : nBlockCount_ - (nLoopIdx * tilingData_->ubFactorN);
            ProcessLoop(gradGmOffset, cosBuffer, sinBuffer, nLoopIdx, currBSNum, currNNum);
        }

        cosInQueue_.FreeTensor(cosBuffer);
        sinInQueue_.FreeTensor(sinBuffer);
    }
}

template <typename TDY, typename TCOS>
__aicore__ inline void InplacePartialRotaryMulGradAB<TDY, TCOS>::ProcessLoop(
    int64_t gradGmOffset, const LocalTensor<TCOS> &cosBuffer, const LocalTensor<TCOS> &sinBuffer,
    int64_t ubIdx, int64_t bsCount, int64_t nCount)
{
    int64_t totalCount = bsCount * nCount;
    uint32_t strideBytesX =
        static_cast<uint32_t>((tilingData_->d - tilingData_->sliceLength) * sizeof(TDY));
    uint32_t blockLenX = static_cast<uint32_t>(tilingData_->sliceLength * sizeof(TDY));

    DataCopyExtParams inParams = {static_cast<uint16_t>(totalCount), blockLenX, strideBytesX, 0, 0};
    DataCopyExtParams outParams = {static_cast<uint16_t>(totalCount), blockLenX, 0, strideBytesX, 0};

    LocalTensor<TDY> inBuffer = gradInQueue_.template AllocTensor<TDY>();
    LocalTensor<TDY> outBuffer = dxQueue_.template AllocTensor<TDY>();

    DataCopyPad(inBuffer,
                gradGm_[gradGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->d], inParams,
                padParamsX_);
    gradInQueue_.EnQue(inBuffer);
    inBuffer = gradInQueue_.template DeQue<TDY>();

    // 根据 mode 选择 VF 函数（同一套函数适配所有 dtype 组合）
    if (tilingData_->rotaryMode == ROTARY_MODE_HALF) {
        HalfGradAlignVF<TDY, TCOS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                               tilingData_->sliceLength,
                                               static_cast<uint16_t>(bsCount), static_cast<uint16_t>(nCount));
    } else if (tilingData_->rotaryMode == ROTARY_MODE_INTERLEAVE) {
        InterleaveModeGradVF<TDY, TCOS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                    tilingData_->sliceLength, static_cast<uint16_t>(bsCount),
                                                    static_cast<uint16_t>(nCount));
    } else if (tilingData_->rotaryMode == ROTARY_MODE_QUARTER) {
        QuarterGradAlignVF<TDY, TCOS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                  tilingData_->sliceLength,
                                                  static_cast<uint16_t>(bsCount), static_cast<uint16_t>(nCount));
    } else {
        DeepSeekInterleaveModeGradVF<TDY, TCOS>(sinBuffer, cosBuffer, inBuffer, outBuffer,
                                                        tilingData_->sliceLength, static_cast<uint16_t>(bsCount),
                                                        static_cast<uint16_t>(nCount));
    }

    dxQueue_.EnQue(outBuffer);
    outBuffer = dxQueue_.template DeQue<TDY>();
    gradInQueue_.FreeTensor(inBuffer);

    DataCopyPad(dxGm_[gradGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->d], outBuffer, outParams);
    dxQueue_.FreeTensor(outBuffer);
}

} // namespace InplacePartialRotaryMulGrad
#endif
