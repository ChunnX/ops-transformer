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

// ============================================================================
// VF 函数 — 同一套模板适配所有 TDY × TCOS 组合
// cos/sin 以 LocalTensor<TCOS> 传入，LoadTwoTensorForDtypeT<T> 自动 upcast 到 float
// ============================================================================

// ---- HalfGradAlignVF ----
template <typename TDY, typename TCOS>
__aicore__ inline void HalfGradAlignVF(const LocalTensor<TCOS> &sinTensor, const LocalTensor<TCOS> &cosTensor,
                                       const LocalTensor<TDY> &inTensor, const LocalTensor<TDY> &outTensor,
                                       uint32_t dLen, uint16_t currSNum, uint16_t currDNum)
{
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint16_t repeatTimes = AscendC::Ceil(halfD, static_cast<uint32_t>(VL_FLOAT32_SIZE));


    __local_mem__ TCOS *sinUb = (__local_mem__ TCOS *)sinTensor.GetPhyAddr();
    __local_mem__ TCOS *cosUb = (__local_mem__ TCOS *)cosTensor.GetPhyAddr();
    __local_mem__ TDY *inUb = (__local_mem__ TDY *)inTensor.GetPhyAddr();
    __local_mem__ TDY *outUb = (__local_mem__ TDY *)outTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregIn, vregHalfIn, vregSin, vregHalfSin, vregCos, vregHalfCos, vregOut, vregHalfOut;
        MicroAPI::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            __local_mem__ TCOS *currSinUb = sinUb + sIdx * dLen;
            __local_mem__ TCOS *currCosUb = cosUb + sIdx * dLen;
            for (uint16_t row = 0; row < currDNum; row++) {
                __local_mem__ TDY *currInUb = inUb + (sIdx * currDNum + row) * dLen;
                __local_mem__ TDY *currOutUb = outUb + (sIdx * currDNum + row) * dLen;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = MicroAPI::UpdateMask<float>(updateCnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t halfOffset = offset + halfD;

                    ops::LoadTwoTensorForDtypeT<TDY>(currInUb, currInUb, vregIn, vregHalfIn, preg, preg, offset,
                                                     halfOffset);
                    ops::LoadTwoTensorForDtypeT<TCOS>(currSinUb, currSinUb, vregSin, vregHalfSin, preg, preg, offset,
                                                      halfOffset);
                    ops::LoadTwoTensorForDtypeT<TCOS>(currCosUb, currCosUb, vregCos, vregHalfCos, preg, preg, offset,
                                                      halfOffset);

                    Mul(vregOut, vregCos, vregIn, preg);
                    Mul(vregHalfSin, vregHalfSin, vregHalfIn, preg);
                    Add(vregOut, vregOut, vregHalfSin, preg);
                    Mul(vregSin, vregSin, vregIn, preg);
                    Mul(vregHalfCos, vregHalfCos, vregHalfIn, preg);
                    Sub(vregHalfOut, vregHalfCos, vregSin, preg);

                    ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregOut, preg, offset);
                    ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregHalfOut, preg, halfOffset);
                }
            }
        }
    }
}

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

// ---- QuarterGradAlignVF ----
// quarter mode (rotary_mode=2):
//   dx_q1 = cos_q1 * dy_q1 + sin_q2 * dy_q2
//   dx_q2 = cos_q2 * dy_q2 - sin_q1 * dy_q1
//   dx_q3 = cos_q3 * dy_q3 + sin_q4 * dy_q4
//   dx_q4 = cos_q4 * dy_q4 - sin_q3 * dy_q3
// Matches reference in rotary_position_embedding_grad.
template <typename TDY, typename TCOS>
__aicore__ inline void QuarterGradAlignVF(const LocalTensor<TCOS> &sinTensor, const LocalTensor<TCOS> &cosTensor,
                                          const LocalTensor<TDY> &inTensor, const LocalTensor<TDY> &outTensor,
                                          uint32_t dLen, uint16_t currSNum, uint16_t currDNum)
{
    uint32_t quarterD = dLen / QUARTER_MODE_COEF;
    uint16_t repeatTimes = AscendC::Ceil(quarterD, static_cast<uint32_t>(VL_FLOAT32_SIZE));


    __local_mem__ TCOS *sinUb = (__local_mem__ TCOS *)sinTensor.GetPhyAddr();
    __local_mem__ TCOS *cosUb = (__local_mem__ TCOS *)cosTensor.GetPhyAddr();
    __local_mem__ TDY *inUb = (__local_mem__ TDY *)inTensor.GetPhyAddr();
    __local_mem__ TDY *outUb = (__local_mem__ TDY *)outTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregIn, vregQ1In, vregQ2In, vregQ3In;
        MicroAPI::RegTensor<float> vregSin, vregQ1Sin, vregQ2Sin, vregQ3Sin;
        MicroAPI::RegTensor<float> vregCos, vregQ1Cos, vregQ2Cos, vregQ3Cos;
        MicroAPI::RegTensor<float> vregOut, vregQ1Out, vregQ2Out, vregQ3Out;
        MicroAPI::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            __local_mem__ TCOS *currSinUb = sinUb + sIdx * dLen;
            __local_mem__ TCOS *currCosUb = cosUb + sIdx * dLen;
            for (uint16_t row = 0; row < currDNum; row++) {
                __local_mem__ TDY *currInUb = inUb + (sIdx * currDNum + row) * dLen;
                __local_mem__ TDY *currOutUb = outUb + (sIdx * currDNum + row) * dLen;
                uint32_t updateCnt = quarterD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = MicroAPI::UpdateMask<float>(updateCnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t q1Off = offset + quarterD;
                    int32_t q2Off = q1Off + quarterD;
                    int32_t q3Off = q2Off + quarterD;

                    // Load dy, sin, cos as 4 quarters
                    ops::LoadTwoTensorForDtypeT<TDY>(currInUb, currInUb, vregIn, vregQ1In, preg, preg, offset, q1Off);
                    ops::LoadTwoTensorForDtypeT<TDY>(currInUb, currInUb, vregQ2In, vregQ3In, preg, preg, q2Off, q3Off);
                    ops::LoadTwoTensorForDtypeT<TCOS>(currSinUb, currSinUb, vregSin, vregQ1Sin, preg, preg, offset,
                                                      q1Off);
                    ops::LoadTwoTensorForDtypeT<TCOS>(currSinUb, currSinUb, vregQ2Sin, vregQ3Sin, preg, preg, q2Off,
                                                      q3Off);
                    ops::LoadTwoTensorForDtypeT<TCOS>(currCosUb, currCosUb, vregCos, vregQ1Cos, preg, preg, offset,
                                                      q1Off);
                    ops::LoadTwoTensorForDtypeT<TCOS>(currCosUb, currCosUb, vregQ2Cos, vregQ3Cos, preg, preg, q2Off,
                                                      q3Off);

                    // q1: cos_q1 * dy_q1 + sin_q2 * dy_q2
                    Mul(vregOut, vregCos, vregIn, preg);
                    Mul(vregQ1Sin, vregQ1Sin, vregQ1In, preg);
                    Add(vregOut, vregOut, vregQ1Sin, preg);
                    // q2: cos_q2 * dy_q2 - sin_q1 * dy_q1
                    Mul(vregSin, vregSin, vregIn, preg);
                    Mul(vregQ1Cos, vregQ1Cos, vregQ1In, preg);
                    Sub(vregQ1Out, vregQ1Cos, vregSin, preg);
                    // q3: cos_q3 * dy_q3 + sin_q4 * dy_q4
                    Mul(vregQ2Out, vregQ2Cos, vregQ2In, preg);
                    Mul(vregQ3Sin, vregQ3Sin, vregQ3In, preg);
                    Add(vregQ2Out, vregQ2Out, vregQ3Sin, preg);
                    // q4: cos_q4 * dy_q4 - sin_q3 * dy_q3
                    Mul(vregQ2Sin, vregQ2Sin, vregQ2In, preg);
                    Mul(vregQ3Cos, vregQ3Cos, vregQ3In, preg);
                    Sub(vregQ3Out, vregQ3Cos, vregQ2Sin, preg);

                    ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregOut, preg, offset);
                    ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregQ1Out, preg, q1Off);
                    ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregQ2Out, preg, q2Off);
                    ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregQ3Out, preg, q3Off);
                }
            }
        }
    }
}

// ---- DeepSeekInterleaveModeGradVF ----
// DeepSeek interleave mode (rotary_mode=3): dx formula is identical to half mode —
//   dx_left  = cos_left  * dy_left  + sin_right * dy_right
//   dx_right = cos_right * dy_right - sin_left  * dy_left
// The output must be interleaved (matching forward DeepSeekInterleave's input format).
// Matches reference DeepSeekInterleaveModeGradVF in rotary_position_embedding_grad.
template <typename TDY, typename TCOS>
__aicore__ inline void DeepSeekInterleaveModeGradVF(const LocalTensor<TCOS> &sinTensor,
                                                    const LocalTensor<TCOS> &cosTensor,
                                                    const LocalTensor<TDY> &inTensor, const LocalTensor<TDY> &outTensor,
                                                    uint32_t dLen, uint16_t currSNum, uint16_t currDNum)
{
    uint32_t dAlignLenDy = Ops::Base::CeilAlign<uint32_t>(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(TDY)));
    uint32_t dAlignLenCos = Ops::Base::CeilAlign<uint32_t>(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(TCOS)));
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t loopSize = HALF_INTERLEAVE_COEF * VL_FLOAT32_SIZE;
    uint16_t repeatTimes = AscendC::Ceil(dLen, loopSize);

    // output mask: part1/part2 for the two interleaved output blocks per iteration
    uint32_t part1Num = (repeatTimes - 1) * VL_FLOAT32_SIZE;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = dLen - part1Num - part2Num;
    if (tailNum > VL_FLOAT32_SIZE) {
        part1Num += VL_FLOAT32_SIZE;
        part2Num += (tailNum - VL_FLOAT32_SIZE);
    } else {
        part1Num += tailNum;
    }


    __local_mem__ TCOS *sinUb = (__local_mem__ TCOS *)sinTensor.GetPhyAddr();
    __local_mem__ TCOS *cosUb = (__local_mem__ TCOS *)cosTensor.GetPhyAddr();
    __local_mem__ TDY *inUb = (__local_mem__ TDY *)inTensor.GetPhyAddr();
    __local_mem__ TDY *outUb = (__local_mem__ TDY *)outTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregIn, vregHalfIn, vregSin, vregHalfSin, vregCos, vregHalfCos, vregOut, vregHalfOut;
        MicroAPI::MaskReg pregLoop, pregPart1, pregPart2;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            __local_mem__ TCOS *currSinUb = sinUb + sIdx * dAlignLenCos;
            __local_mem__ TCOS *currCosUb = cosUb + sIdx * dAlignLenCos;
            for (uint16_t row = 0; row < currDNum; row++) {
                __local_mem__ TDY *currInUb = inUb + (sIdx * currDNum + row) * dAlignLenDy;
                __local_mem__ TDY *currOutUb = outUb + (sIdx * currDNum + row) * dAlignLenDy;
                uint32_t halfCnt = halfD;
                uint32_t part1Cnt = part1Num;
                uint32_t part2Cnt = part2Num;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                    pregPart1 = MicroAPI::UpdateMask<float>(part1Cnt);
                    pregPart2 = MicroAPI::UpdateMask<float>(part2Cnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t halfOffset = offset + halfD;
                    int32_t outOffset = offset * HALF_INTERLEAVE_COEF;

                    ops::LoadTwoTensorForDtypeT<TDY>(currInUb, currInUb, vregIn, vregHalfIn, pregLoop, pregLoop, offset,
                                                     halfOffset);
                    ops::LoadTwoTensorForDtypeT<TCOS>(currSinUb, currSinUb, vregSin, vregHalfSin, pregLoop, pregLoop,
                                                      offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<TCOS>(currCosUb, currCosUb, vregCos, vregHalfCos, pregLoop, pregLoop,
                                                      offset, halfOffset);

                    // dx_left = cos_left * dy_left + sin_right * dy_right
                    Mul(vregOut, vregCos, vregIn, pregLoop);
                    Mul(vregHalfSin, vregHalfSin, vregHalfIn, pregLoop);
                    Add(vregOut, vregOut, vregHalfSin, pregLoop);
                    // dx_right = cos_right * dy_right - sin_left * dy_left
                    Mul(vregSin, vregSin, vregIn, pregLoop);
                    Mul(vregHalfCos, vregHalfCos, vregHalfIn, pregLoop);
                    Sub(vregHalfOut, vregHalfCos, vregSin, pregLoop);

                    // Interleave: half format → interleaved format (matching forward input x layout)
                    // Reuse vregCos/vregHalfCos (dead after compute) as interleave outputs
                    MicroAPI::Interleave<float>(vregCos, vregHalfCos, vregOut, vregHalfOut);
                    ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregCos, pregPart1, outOffset);
                    ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregHalfCos, pregPart2, outOffset + VL_FLOAT32_SIZE);
                }
            }
        }
    }
}

// ============================================================================
// Batch VF 函数 — B→N→S 三维循环内置，用于 AAndB(非广播) 和 ABAAndBA
// ============================================================================

// ---- BatchHalfGradAlignVF ----
template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void BatchHalfGradAlignVF(__local_mem__ TDY *in, __local_mem__ TCOS *cos, __local_mem__ TCOS *sin,
                                            __local_mem__ TDY *out, uint16_t sLength, uint16_t bLength,
                                            uint16_t nLength, uint32_t dLen, int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t dHalfSize = dLen / HALF_INTERLEAVE_COEF;
    uint16_t dLoopCount = (dHalfSize + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;

    int32_t bStepUb = ubFactorN * ubFactorS * dLen;
    int32_t nStepUb = ubFactorS * dLen;
    int32_t nStepCosUb = ubFactorS * dLen;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inPart1Reg, inPart2Reg, cosPart1Reg, cosPart2Reg, sinPart1Reg, sinPart2Reg;
        MicroAPI::MaskReg pregLoop;
        __local_mem__ TDY *currInUb, *currOutUb;
        __local_mem__ TCOS *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t count = dHalfSize;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dLen;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dLen;
                    if constexpr (IsBroadcast) {
                        currCosUb = cos + sIdx * dLen;
                        currSinUb = sin + sIdx * dLen;
                    } else {
                        currCosUb = cos + bIdx * nStepCosUb + sIdx * dLen;
                        currSinUb = sin + bIdx * nStepCosUb + sIdx * dLen;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(count);
                        ops::LoadOneTensorForDtypeT<TDY>(currInUb, inPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<TDY>(currInUb, inPart2Reg, pregLoop,
                                                         i * VL_FLOAT32_SIZE + dHalfSize);
                        ops::LoadOneTensorForDtypeT<TCOS>(currCosUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<TCOS>(currCosUb, cosPart2Reg, pregLoop,
                                                          i * VL_FLOAT32_SIZE + dHalfSize);
                        ops::LoadOneTensorForDtypeT<TCOS>(currSinUb, sinPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<TCOS>(currSinUb, sinPart2Reg, pregLoop,
                                                          i * VL_FLOAT32_SIZE + dHalfSize);
                        Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregLoop);
                        Add(cosPart1Reg, cosPart1Reg, sinPart2Reg, pregLoop);
                        Mul(cosPart2Reg, inPart2Reg, cosPart2Reg, pregLoop);
                        Mul(sinPart1Reg, inPart1Reg, sinPart1Reg, pregLoop);
                        Sub(cosPart2Reg, cosPart2Reg, sinPart1Reg, pregLoop);
                        ops::StoreOneTensorForDtypeT<TDY>(currOutUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::StoreOneTensorForDtypeT<TDY>(currOutUb, cosPart2Reg, pregLoop,
                                                          i * VL_FLOAT32_SIZE + dHalfSize);
                    }
                }
            }
        }
    }
}

// ---- BatchQuarterGradAlignVF ----
// quarter mode (rotary_mode=2): batch version.
// Matches reference in rotary_position_embedding_grad.
template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void BatchQuarterGradAlignVF(__local_mem__ TDY *in, __local_mem__ TCOS *cos, __local_mem__ TCOS *sin,
                                               __local_mem__ TDY *out, uint16_t sLength, uint16_t bLength,
                                               uint16_t nLength, uint32_t dLen, int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t dQuarterSize = dLen / QUARTER_MODE_COEF;
    uint16_t dLoopCount = (dQuarterSize + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;

    int32_t bStepUb = ubFactorN * ubFactorS * dLen;
    int32_t nStepUb = ubFactorS * dLen;
    int32_t nStepCosUb = ubFactorS * dLen;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inPart1Reg, inPart2Reg, inPart3Reg, inPart4Reg;
        MicroAPI::RegTensor<float> cosPart1Reg, cosPart2Reg, cosPart3Reg, cosPart4Reg;
        MicroAPI::RegTensor<float> sinPart1Reg, sinPart2Reg, sinPart3Reg, sinPart4Reg;
        MicroAPI::MaskReg pregLoop;
        __local_mem__ TDY *currInUb, *currOutUb;
        __local_mem__ TCOS *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t count = dQuarterSize;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dLen;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dLen;
                    if constexpr (IsBroadcast) {
                        currCosUb = cos + sIdx * dLen;
                        currSinUb = sin + sIdx * dLen;
                    } else {
                        currCosUb = cos + bIdx * nStepCosUb + sIdx * dLen;
                        currSinUb = sin + bIdx * nStepCosUb + sIdx * dLen;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(count);
                        int32_t off0 = i * VL_FLOAT32_SIZE;
                        int32_t off1 = off0 + dQuarterSize;
                        int32_t off2 = off1 + dQuarterSize;
                        int32_t off3 = off2 + dQuarterSize;

                        // Load dy, sin, cos as 4 quarters
                        ops::LoadTwoTensorForDtypeT<TDY>(currInUb, currInUb, inPart1Reg, inPart2Reg, pregLoop, pregLoop,
                                                         off0, off1);
                        ops::LoadTwoTensorForDtypeT<TDY>(currInUb, currInUb, inPart3Reg, inPart4Reg, pregLoop, pregLoop,
                                                         off2, off3);
                        ops::LoadTwoTensorForDtypeT<TCOS>(currSinUb, currSinUb, sinPart1Reg, sinPart2Reg, pregLoop,
                                                          pregLoop, off0, off1);
                        ops::LoadTwoTensorForDtypeT<TCOS>(currSinUb, currSinUb, sinPart3Reg, sinPart4Reg, pregLoop,
                                                          pregLoop, off2, off3);
                        ops::LoadTwoTensorForDtypeT<TCOS>(currCosUb, currCosUb, cosPart1Reg, cosPart2Reg, pregLoop,
                                                          pregLoop, off0, off1);
                        ops::LoadTwoTensorForDtypeT<TCOS>(currCosUb, currCosUb, cosPart3Reg, cosPart4Reg, pregLoop,
                                                          pregLoop, off2, off3);

                        // q1: cos_q1 * dy_q1 + sin_q2 * dy_q2
                        Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregLoop);
                        Add(cosPart1Reg, cosPart1Reg, sinPart2Reg, pregLoop);
                        // q2: cos_q2 * dy_q2 - sin_q1 * dy_q1
                        Mul(cosPart2Reg, inPart2Reg, cosPart2Reg, pregLoop);
                        Mul(sinPart1Reg, inPart1Reg, sinPart1Reg, pregLoop);
                        Sub(cosPart2Reg, cosPart2Reg, sinPart1Reg, pregLoop);
                        // q3: cos_q3 * dy_q3 + sin_q4 * dy_q4
                        Mul(cosPart3Reg, inPart3Reg, cosPart3Reg, pregLoop);
                        Mul(sinPart4Reg, sinPart4Reg, inPart4Reg, pregLoop);
                        Add(cosPart3Reg, cosPart3Reg, sinPart4Reg, pregLoop);
                        // q4: cos_q4 * dy_q4 - sin_q3 * dy_q3
                        Mul(cosPart4Reg, inPart4Reg, cosPart4Reg, pregLoop);
                        Mul(sinPart3Reg, inPart3Reg, sinPart3Reg, pregLoop);
                        Sub(cosPart4Reg, cosPart4Reg, sinPart3Reg, pregLoop);

                        ops::StoreOneTensorForDtypeT<TDY>(currOutUb, cosPart1Reg, pregLoop, off0);
                        ops::StoreOneTensorForDtypeT<TDY>(currOutUb, cosPart2Reg, pregLoop, off1);
                        ops::StoreOneTensorForDtypeT<TDY>(currOutUb, cosPart3Reg, pregLoop, off2);
                        ops::StoreOneTensorForDtypeT<TDY>(currOutUb, cosPart4Reg, pregLoop, off3);
                    }
                }
            }
        }
    }
}

// ---- BatchInterleaveModeGradVF ----
// interleave mode (rotary_mode=1): batch version.
// Matches reference in rotary_position_embedding_grad.
template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void BatchInterleaveModeGradVF(__local_mem__ TDY *in, __local_mem__ TCOS *cos,
                                                 __local_mem__ TCOS *sin, __local_mem__ TDY *out, uint16_t sLength,
                                                 uint16_t bLength, uint16_t nLength, uint32_t dLen, int64_t ubFactorS,
                                                 int64_t ubFactorN)
{
    uint32_t dAlignLenDy = Ops::Base::CeilAlign<uint32_t>(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(TDY)));
    uint32_t dAlignLenCos = Ops::Base::CeilAlign<uint32_t>(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(TCOS)));

    uint32_t loopSize = 2 * VL_FLOAT32_SIZE;
    uint16_t dLoopCount = (dLen + loopSize - 1) / loopSize;

    uint32_t halfNum = dLen / HALF_INTERLEAVE_COEF;
    uint32_t part1Num = (dLoopCount - 1) * VL_FLOAT32_SIZE;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = dLen - part1Num - part2Num;
    if (tailNum > VL_FLOAT32_SIZE) {
        part1Num += VL_FLOAT32_SIZE;
        part2Num += (tailNum - VL_FLOAT32_SIZE);
    } else {
        part1Num += tailNum;
    }

    int32_t bStepUb = ubFactorN * ubFactorS * dAlignLenDy;
    int32_t nStepUb = ubFactorS * dAlignLenDy;
    int32_t nStepCosUb = ubFactorS * dAlignLenCos;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inPart1Reg, inPart2Reg;
        MicroAPI::RegTensor<float> cosPart1Reg, cosPart2Reg;
        MicroAPI::RegTensor<float> sinPart1Reg, sinPart2Reg;
        MicroAPI::MaskReg pregLoop, pregPart1, pregPart2;
        __local_mem__ TDY *currInUb, *currOutUb;
        __local_mem__ TCOS *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t halfCnt = halfNum;
                    uint32_t part1Cnt = part1Num;
                    uint32_t part2Cnt = part2Num;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignLenDy;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignLenDy;
                    if constexpr (IsBroadcast) {
                        currCosUb = cos + sIdx * dAlignLenCos;
                        currSinUb = sin + sIdx * dAlignLenCos;
                    } else {
                        currCosUb = cos + bIdx * nStepCosUb + sIdx * dAlignLenCos;
                        currSinUb = sin + bIdx * nStepCosUb + sIdx * dAlignLenCos;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                        pregPart1 = MicroAPI::UpdateMask<float>(part1Cnt);
                        pregPart2 = MicroAPI::UpdateMask<float>(part2Cnt);

                        ops::LoadOneTensorForDtypeT<TDY>(currInUb, inPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<TDY>(currInUb, inPart2Reg, pregPart2,
                                                         i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<TCOS>(currCosUb, cosPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<TCOS>(currCosUb, cosPart2Reg, pregPart2,
                                                          i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<TCOS>(currSinUb, sinPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<TCOS>(currSinUb, sinPart2Reg, pregPart2,
                                                          i * loopSize + VL_FLOAT32_SIZE);

                        Mul(cosPart1Reg, cosPart1Reg, inPart1Reg, pregPart1);
                        Mul(cosPart2Reg, cosPart2Reg, inPart2Reg, pregPart2);
                        Mul(sinPart1Reg, sinPart1Reg, inPart1Reg, pregPart1);
                        Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregPart2);

                        MicroAPI::DeInterleave<float>(sinPart1Reg, sinPart2Reg, sinPart1Reg, sinPart2Reg);
                        Muls(sinPart1Reg, sinPart1Reg, float(-1.0), pregLoop);
                        MicroAPI::Interleave<float>(sinPart1Reg, sinPart2Reg, sinPart2Reg, sinPart1Reg);

                        Add(cosPart1Reg, sinPart1Reg, cosPart1Reg, pregPart1);
                        Add(cosPart2Reg, sinPart2Reg, cosPart2Reg, pregPart2);

                        ops::StoreOneTensorForDtypeT<TDY>(currOutUb, cosPart1Reg, pregPart1, i * loopSize);
                        ops::StoreOneTensorForDtypeT<TDY>(currOutUb, cosPart2Reg, pregPart2,
                                                          i * loopSize + VL_FLOAT32_SIZE);
                    }
                }
            }
        }
    }
}

// ---- BatchDeepSeekInterleaveModeGradVF ----
// DeepSeek interleave mode (rotary_mode=3): batch version.
// The output must be interleaved (matching forward DeepSeekInterleave's input format).
// Matches reference BatchDeepSeekInterleaveModeGradVF in rotary_position_embedding_grad.
template <typename TDY, typename TCOS, bool IsBroadcast>
__aicore__ inline void BatchDeepSeekInterleaveModeGradVF(__local_mem__ TDY *in, __local_mem__ TCOS *cos,
                                                         __local_mem__ TCOS *sin, __local_mem__ TDY *out,
                                                         uint16_t sLength, uint16_t bLength, uint16_t nLength,
                                                         uint32_t dLen, int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t dAlignLenDy = Ops::Base::CeilAlign<uint32_t>(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(TDY)));
    uint32_t dAlignLenCos = Ops::Base::CeilAlign<uint32_t>(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(TCOS)));

    uint32_t loopSize = HALF_INTERLEAVE_COEF * VL_FLOAT32_SIZE;
    uint16_t dLoopCount = AscendC::Ceil(dLen, loopSize);
    uint32_t dHalfSize = dLen / HALF_INTERLEAVE_COEF;
    // output mask: part1/part2 for the two interleaved output blocks per iteration
    uint32_t halfNum = dLen / HALF_INTERLEAVE_COEF;
    uint32_t part1Num = (dLoopCount - 1) * VL_FLOAT32_SIZE;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = dLen - part1Num - part2Num;
    if (tailNum > VL_FLOAT32_SIZE) {
        part1Num += VL_FLOAT32_SIZE;
        part2Num += (tailNum - VL_FLOAT32_SIZE);
    } else {
        part1Num += tailNum;
    }

    int32_t bStepUb = ubFactorN * ubFactorS * dAlignLenDy;
    int32_t nStepUb = ubFactorS * dAlignLenDy;
    int32_t nStepCosUb = ubFactorS * dAlignLenCos;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregIn, vregHalfIn, vregCos, vregHalfCos, vregSin, vregHalfSin, vregOut, vregHalfOut;
        MicroAPI::MaskReg pregLoop, pregPart1, pregPart2;
        __local_mem__ TDY *currInUb, *currOutUb;
        __local_mem__ TCOS *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t halfCnt = halfNum;
                    uint32_t part1Cnt = part1Num;
                    uint32_t part2Cnt = part2Num;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignLenDy;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignLenDy;
                    if constexpr (IsBroadcast) {
                        currCosUb = cos + sIdx * dAlignLenCos;
                        currSinUb = sin + sIdx * dAlignLenCos;
                    } else {
                        currCosUb = cos + bIdx * nStepCosUb + sIdx * dAlignLenCos;
                        currSinUb = sin + bIdx * nStepCosUb + sIdx * dAlignLenCos;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                        pregPart1 = MicroAPI::UpdateMask<float>(part1Cnt);
                        pregPart2 = MicroAPI::UpdateMask<float>(part2Cnt);
                        ops::LoadOneTensorForDtypeT<TDY>(currInUb, vregIn, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<TDY>(currInUb, vregHalfIn, pregLoop,
                                                         i * VL_FLOAT32_SIZE + dHalfSize);
                        ops::LoadOneTensorForDtypeT<TCOS>(currCosUb, vregCos, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<TCOS>(currCosUb, vregHalfCos, pregLoop,
                                                          i * VL_FLOAT32_SIZE + dHalfSize);
                        ops::LoadOneTensorForDtypeT<TCOS>(currSinUb, vregSin, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<TCOS>(currSinUb, vregHalfSin, pregLoop,
                                                          i * VL_FLOAT32_SIZE + dHalfSize);

                        // dx_left = cos_left * dy_left + sin_right * dy_right
                        Mul(vregOut, vregCos, vregIn, pregLoop);
                        Mul(vregHalfSin, vregHalfSin, vregHalfIn, pregLoop);
                        Add(vregOut, vregOut, vregHalfSin, pregLoop);
                        // dx_right = cos_right * dy_right - sin_left * dy_left
                        Mul(vregSin, vregSin, vregIn, pregLoop);
                        Mul(vregHalfCos, vregHalfCos, vregHalfIn, pregLoop);
                        Sub(vregHalfOut, vregHalfCos, vregSin, pregLoop);

                        // Interleave: half format → interleaved format (matching forward input x layout)
                        // Reuse vregCos/vregHalfCos (dead after compute) as interleave outputs
                        MicroAPI::Interleave<float>(vregCos, vregHalfCos, vregOut, vregHalfOut);
                        ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregCos, pregPart1, i * loopSize);
                        ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregHalfCos, pregPart2,
                                                          i * loopSize + VL_FLOAT32_SIZE);
                    }
                }
            }
        }
    }
}

} // namespace InplacePartialRotaryMulGrad
#endif
