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
 * \file inplace_partial_rotary_mul_grad_common.h
 * \brief Common VF functions for InplacePartialRotaryMulGrad, shared across all templates.
 */

#ifndef __INPLACE_PARTIAL_ROTARY_MUL_GRAD_COMMON_H__
#define __INPLACE_PARTIAL_ROTARY_MUL_GRAD_COMMON_H__

#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"
#include "op_kernel/math_util.h"

namespace InplacePartialRotaryMulGrad {
using namespace AscendC;

constexpr uint32_t HALF_INTERLEAVE_COEF = 2;
constexpr uint32_t QUARTER_MODE_COEF = 4;

__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

constexpr uint32_t BLOCK_TYPE_SIZE = GetUbBlockSize();
constexpr uint32_t VREG_SIZE_BITS = 256U;                            // Ascend950 vector register width
constexpr uint32_t VL_FLOAT32_SIZE = VREG_SIZE_BITS / sizeof(float); // 256/4 = 64 floats per vector

// mode 常量，与 InplacePartialRotaryMulGradMode 对齐
constexpr int64_t ROTARY_MODE_HALF = 0;
constexpr int64_t ROTARY_MODE_INTERLEAVE = 1;
constexpr int64_t ROTARY_MODE_QUARTER = 2;
constexpr int64_t ROTARY_MODE_DEEPSEEK_INTERLEAVE = 3;

// ---- InterleaveModeGradVF ----
// interleave mode (rotary_mode=1):
//   dx[2k]   = cos[2k]*dy[2k]   + sin[2k+1]*dy[2k+1]
//   dx[2k+1] = cos[2k+1]*dy[2k+1] - sin[2k]*dy[2k]
// Matches reference in rotary_position_embedding_grad.
template <typename TDY, typename TCOS>
__aicore__ inline void InterleaveModeGradVF(const LocalTensor<TCOS> &sinTensor, const LocalTensor<TCOS> &cosTensor,
                                            const LocalTensor<TDY> &inTensor, const LocalTensor<TDY> &outTensor,
                                            uint16_t dLen, uint16_t currSNum, uint16_t currNNum)
{
    __local_mem__ TCOS *sinUb = (__local_mem__ TCOS *)sinTensor.GetPhyAddr();
    __local_mem__ TCOS *cosUb = (__local_mem__ TCOS *)cosTensor.GetPhyAddr();
    __local_mem__ TDY *inUb = (__local_mem__ TDY *)inTensor.GetPhyAddr();
    __local_mem__ TDY *outUb = (__local_mem__ TDY *)outTensor.GetPhyAddr();
    uint16_t loopSize = 2 * VL_FLOAT32_SIZE;
    uint16_t loopNum = (dLen + loopSize - 1) / (2 * VL_FLOAT32_SIZE);
    uint16_t dAlignLenDy = Ops::Base::CeilAlign(dLen, static_cast<uint16_t>(BLOCK_TYPE_SIZE / sizeof(TDY)));
    uint16_t dAlignLenCos = Ops::Base::CeilAlign(dLen, static_cast<uint16_t>(BLOCK_TYPE_SIZE / sizeof(TCOS)));


    uint32_t halfNum = dLen / 2;
    uint32_t part1Num = (loopNum - 1) * VL_FLOAT32_SIZE;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = dLen - part1Num - part2Num;
    if (tailNum > VL_FLOAT32_SIZE) {
        part1Num += VL_FLOAT32_SIZE;
        part2Num += (tailNum - VL_FLOAT32_SIZE);
    } else {
        part1Num += tailNum;
    }

    __local_mem__ TDY *currInUb, *currOutUb;
    __local_mem__ TCOS *currSinUb, *currCosUb, *tailSinUb, *tailCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregFormerCos, vregLatterCos, vregFormerSin, vregLatterSin, vregFormerIn,
            vregLatterIn, vregOdd, vregEven;
        MicroAPI::MaskReg pregLoop, pregPart1, pregPart2;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlignLenCos;
            currCosUb = cosUb + sIdx * dAlignLenCos;
            for (uint16_t idxD = 0; idxD < currNNum; idxD++) {
                uint32_t halfCnt = halfNum;
                uint32_t part1Cnt = part1Num;
                uint32_t part2Cnt = part2Num;
                currInUb = inUb + (sIdx * currNNum + idxD) * dAlignLenDy;
                currOutUb = outUb + (sIdx * currNNum + idxD) * dAlignLenDy;
                for (uint16_t i = 0; i < loopNum; i++) {
                    pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                    pregPart1 = MicroAPI::UpdateMask<float>(part1Cnt);
                    pregPart2 = MicroAPI::UpdateMask<float>(part2Cnt);
                    int32_t evenOffSet = (i * 2) * VL_FLOAT32_SIZE;
                    int32_t oddOffset = evenOffSet + VL_FLOAT32_SIZE;
                    ops::LoadOneTensorForDtypeT<TDY>(currInUb, vregFormerIn, pregPart1, evenOffSet);
                    ops::LoadOneTensorForDtypeT<TDY>(currInUb, vregLatterIn, pregPart2, oddOffset);
                    ops::LoadOneTensorForDtypeT<TCOS>(currCosUb, vregFormerCos, pregPart1, evenOffSet);
                    ops::LoadOneTensorForDtypeT<TCOS>(currCosUb, vregLatterCos, pregPart2, oddOffset);
                    ops::LoadOneTensorForDtypeT<TCOS>(currSinUb, vregFormerSin, pregPart1, evenOffSet);
                    ops::LoadOneTensorForDtypeT<TCOS>(currSinUb, vregLatterSin, pregPart2, oddOffset);
                    Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregPart1);
                    Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregPart2);
                    Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregPart1);
                    Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregPart2);
                    MicroAPI::DeInterleave<float>(vregFormerSin, vregLatterSin, vregFormerSin, vregLatterSin);
                    Muls(vregFormerSin, vregFormerSin, float(-1.0), pregLoop);
                    MicroAPI::Interleave<float>(vregFormerSin, vregLatterSin, vregLatterSin, vregFormerSin);
                    Add(vregFormerCos, vregFormerSin, vregFormerCos, pregPart1);
                    Add(vregLatterCos, vregLatterSin, vregLatterCos, pregPart2);
                    ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregFormerCos, pregPart1, evenOffSet);
                    ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregLatterCos, pregPart2, oddOffset);
                }
            }
        }
    }
}

} // namespace InplacePartialRotaryMulGrad
#endif
