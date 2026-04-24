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
 * \file vf_process_vec1.h
 * \brief MicroAPI implementations for scale and reduce operations
 */

#ifndef VF_PROCESS_VEC1_H
#define VF_PROCESS_VEC1_H

#include "kernel_tensor.h"
#include "dense_lightning_indexer_softmax_lse_regbase_common.h"

namespace AscendC {
using namespace MicroAPI;
using namespace DenseLISoftmaxLseKernelRegbase;

constexpr uint8_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint8_t B32_VEC_REPEAT_STRIDE = 8;
constexpr uint64_t VEC_REPEAT_BYTES = 256;
constexpr int32_t CONST_TWO = 2;

template <typename T, typename INPUT_T>
__aicore__ inline void ProcessVec1Vf(LocalTensor<T> &reduceCacheBuf, LocalTensor<T> &mmOutUb,
                                     LocalTensor<INPUT_T> &weightsTUb, int64_t groupInner,
                                     int64_t s2Inner, int32_t outerGidx)
{
    LocalTensor<T> weightsUb = weightsTUb.template ReinterpretCast<T>();
    LocalTensor<T> tmpBuff;

    if constexpr (!IsSameType<INPUT_T, float>::value) {
        AscendC::Cast(weightsUb, weightsTUb, RoundMode::CAST_NONE, groupInner);
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::Brcb(tmpBuff, weightsUb,
                  DenseLISoftmaxLseCeilDiv<int64_t>(groupInner, static_cast<int64_t>(B32_BLOCK_ALIGN_NUM)),
                  {1, B32_VEC_REPEAT_STRIDE});
    AscendC::PipeBarrier<PIPE_V>();

    uint64_t countPerRepeat = VEC_REPEAT_BYTES / sizeof(T);
    uint64_t repeatTimes = s2Inner / countPerRepeat;
    for (int32_t i = 0; i < groupInner; i++) {
        if (outerGidx == 0) {
            AscendC::Mul(reduceCacheBuf[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM],
                         countPerRepeat, repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        } else {
            AscendC::Mul(mmOutUb[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM],
                         countPerRepeat, repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        }
    }

    if (outerGidx != 0) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(reduceCacheBuf, mmOutUb, reduceCacheBuf, groupInner * s2Inner);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void ProcessVec2Vf(LocalTensor<T> &dstTensor, LocalTensor<T> &srcTensor,
                                     int32_t rNum, int32_t aNum)
{
    if (rNum == 1) {
        AscendC::Adds<T>(dstTensor, srcTensor, 0, aNum);
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }

    uint32_t dichotomizeAddPow = FindNearestPower2(rNum);
    uint32_t dichotomizeAddDiffSize = rNum - dichotomizeAddPow;
    if (dichotomizeAddDiffSize != 0) {
        AscendC::Add(srcTensor, srcTensor, srcTensor[dichotomizeAddPow * aNum], dichotomizeAddDiffSize * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    int32_t nowRows = dichotomizeAddPow;
    while (nowRows > CONST_TWO) {
        nowRows = nowRows / CONST_TWO;
        AscendC::Add(srcTensor, srcTensor, srcTensor[nowRows * aNum], nowRows * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Add(dstTensor, srcTensor, srcTensor[aNum], aNum);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline uint64_t FindNearestPower2(uint64_t value)
{
    if (value <= CONST_TWO) {
        return value;
    } else {
        const uint64_t pow = 63 - AscendC::ScalarCountLeadingZero(value);
        return (1 << pow);
    }
}

} // namespace AscendC

#endif // VF_PROCESS_VEC1_H