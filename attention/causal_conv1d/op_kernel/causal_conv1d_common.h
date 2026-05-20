/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_common.h
 */

#ifndef CAUSAL_CONV1D_COMMON_H
#define CAUSAL_CONV1D_COMMON_H

#include "kernel_operator.h"

namespace NsCausalConv1dCommon {

constexpr int32_t MAX_WIDTH = 4;

constexpr int32_t RING_SLOTS = 5;

__aicore__ inline int32_t SlotCurr(int32_t t)
{
    return (t + 3) % RING_SLOTS;
}

__aicore__ inline int32_t SlotHist(int32_t t, int32_t i)
{
    return (t + 3 - i) % RING_SLOTS;
}

__aicore__ inline int32_t SlotPrefetch(int32_t t)
{
    return (t + 4) % RING_SLOTS;
}

__aicore__ inline constexpr bool IsFnInputModeKey(uint32_t inputModeKey)
{
    return (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN) ||
           (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_BATCH);
}

__aicore__ inline constexpr bool IsUpdateInputModeKey(uint32_t inputModeKey)
{
    return (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_VARLEN) ||
           (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_BATCH);
}

__aicore__ inline constexpr bool IsVarlenInputModeKey(uint32_t inputModeKey)
{
    return (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN) ||
           (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_VARLEN);
}

struct CalcBufLayout {
    AscendC::LocalTensor<float> weightF;
    AscendC::LocalTensor<float> biasF;
    AscendC::LocalTensor<float> accF;
    AscendC::LocalTensor<float> tmpF;
    AscendC::LocalTensor<float> currF;

    __aicore__ inline CalcBufLayout() = default;

    __aicore__ static inline CalcBufLayout FromCalcBuf(AscendC::TBuf<AscendC::QuePosition::VECCALC> &calcBuf,
                                                       int32_t width, int32_t baseDim, bool hasBias)
    {
        CalcBufLayout layout;
        AscendC::LocalTensor<float> calc = calcBuf.template Get<float>();
        layout.weightF = calc;
        layout.biasF = calc[width * baseDim];
        if (hasBias) {
            layout.accF = layout.biasF[baseDim];
        } else {
            layout.accF = layout.biasF;
        }
        layout.tmpF = layout.accF[baseDim];
        layout.currF = layout.tmpF[baseDim];
        return layout;
    }
};

} // namespace NsCausalConv1dCommon

#endif // CAUSAL_CONV1D_COMMON_H
