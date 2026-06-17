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

namespace InplacePartialRotaryMulGrad {
using namespace AscendC;

constexpr uint32_t HALF_INTERLEAVE_COEF = 2;
constexpr uint32_t QUARTER_MODE_COEF = 4;

// mode 常量，与 InplacePartialRotaryMulGradMode 对齐
constexpr int64_t ROTARY_MODE_HALF = 0;
constexpr int64_t ROTARY_MODE_INTERLEAVE = 1;
constexpr int64_t ROTARY_MODE_QUARTER = 2;
constexpr int64_t ROTARY_MODE_INTERLEAVE_HALF = 3;

// ============================================================================
// VF 函数 — 同一套模板适配所有 DTYPE_DY × DTYPE_COS 组合
// cos/sin 以 LocalTensor<DTYPE_COS> 传入，LoadTwoTensorForDtypeT<T> 自动 upcast 到 float
// ============================================================================

// ---- HalfGradAlignVF ----
template <typename DTYPE_DY, typename DTYPE_COS>
__aicore__ inline void HalfGradAlignVF(const LocalTensor<DTYPE_COS> &sinTensor,
                                        const LocalTensor<DTYPE_COS> &cosTensor,
                                        const LocalTensor<DTYPE_DY> &inTensor,
                                        const LocalTensor<DTYPE_DY> &outTensor, uint32_t dLen,
                                        uint32_t dAlignDy, uint16_t currSNum, uint16_t currDNum)
{
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlignDy = AscendC::Ceil(halfD, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_DY)));
    uint32_t halfDAlignCos = AscendC::Ceil(halfD, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_COS)));
    uint16_t repeatTimes = AscendC::Ceil(halfD, static_cast<uint32_t>(VL_FLOAT32_SIZE));

    uint32_t cosRowStride = dLen;

    __local_mem__ DTYPE_COS *sinUb = const_cast<__local_mem__ DTYPE_COS *>(sinTensor.GetPhyAddr());
    __local_mem__ DTYPE_COS *cosUb = const_cast<__local_mem__ DTYPE_COS *>(cosTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *inUb = const_cast<__local_mem__ DTYPE_DY *>(inTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *outUb = const_cast<__local_mem__ DTYPE_DY *>(outTensor.GetPhyAddr());

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> vregIn, vregHalfIn, vregSin, vregHalfSin, vregCos, vregHalfCos, vregOut, vregHalfOut;
        MicroAPI::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            __local_mem__ DTYPE_COS *currSinUb = sinUb + sIdx * cosRowStride;
            __local_mem__ DTYPE_COS *currCosUb = cosUb + sIdx * cosRowStride;
            for (uint16_t row = 0; row < currDNum; row++) {
                __local_mem__ DTYPE_DY *currInUb = inUb + (sIdx * currDNum + row) * dAlignDy;
                __local_mem__ DTYPE_DY *currOutUb = outUb + (sIdx * currDNum + row) * dAlignDy;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = MicroAPI::UpdateMask<float>(updateCnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t halfOffset = offset + halfDAlignDy;
                    int32_t cosHalfOffset = offset + halfDAlignCos;

                    ops::LoadTwoTensorForDtypeT<DTYPE_DY>(currInUb, currInUb, vregIn, vregHalfIn, preg, preg, offset,
                                                           halfOffset);
                    ops::LoadTwoTensorForDtypeT<DTYPE_COS>(currSinUb, currSinUb, vregSin, vregHalfSin, preg, preg,
                                                            offset, cosHalfOffset);
                    ops::LoadTwoTensorForDtypeT<DTYPE_COS>(currCosUb, currCosUb, vregCos, vregHalfCos, preg, preg,
                                                            offset, cosHalfOffset);

                    Mul(vregOut, vregCos, vregIn, preg);
                    Mul(vregHalfSin, vregHalfSin, vregHalfIn, preg);
                    Add(vregOut, vregOut, vregHalfSin, preg);
                    Mul(vregSin, vregSin, vregIn, preg);
                    Mul(vregHalfCos, vregHalfCos, vregHalfIn, preg);
                    Sub(vregHalfOut, vregHalfCos, vregSin, preg);

                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut, preg, offset);
                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregHalfOut, preg, halfOffset);
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
template <typename DTYPE_DY, typename DTYPE_COS>
__aicore__ inline void InterleaveModeGradVF(const LocalTensor<DTYPE_COS> &sinTensor,
                                             const LocalTensor<DTYPE_COS> &cosTensor,
                                             const LocalTensor<DTYPE_DY> &inTensor,
                                             const LocalTensor<DTYPE_DY> &outTensor, uint32_t dLen,
                                             uint16_t currSNum, uint16_t currDNum)
{
    uint32_t dAlignDy = AscendC::Ceil(dLen, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_DY)));
    uint16_t repeatTimes = AscendC::Ceil(dLen, static_cast<uint32_t>(VL_FLOAT32_SIZE));

    uint32_t cosRowStride = dLen;

    __local_mem__ DTYPE_COS *sinUb = const_cast<__local_mem__ DTYPE_COS *>(sinTensor.GetPhyAddr());
    __local_mem__ DTYPE_COS *cosUb = const_cast<__local_mem__ DTYPE_COS *>(cosTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *inUb = const_cast<__local_mem__ DTYPE_DY *>(inTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *outUb = const_cast<__local_mem__ DTYPE_DY *>(outTensor.GetPhyAddr());

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> vregIn, vregCos, vregSin, vregOut;
        MicroAPI::RegTensor<float> vregDyEven, vregDyOdd;
        MicroAPI::RegTensor<float> vregCosEven, vregCosOdd;
        MicroAPI::RegTensor<float> vregSinEven, vregSinOdd;
        MicroAPI::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            __local_mem__ DTYPE_COS *currSinUb = sinUb + sIdx * cosRowStride;
            __local_mem__ DTYPE_COS *currCosUb = cosUb + sIdx * cosRowStride;
            for (uint16_t row = 0; row < currDNum; row++) {
                __local_mem__ DTYPE_DY *currInUb = inUb + (sIdx * currDNum + row) * dAlignDy;
                __local_mem__ DTYPE_DY *currOutUb = outUb + (sIdx * currDNum + row) * dAlignDy;
                uint32_t updateCnt = dLen;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = MicroAPI::UpdateMask<float>(updateCnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;

                    // Load full interleaved dy, cos, sin
                    ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, vregIn, preg, offset);
                    ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, vregCos, preg, offset);
                    ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, vregSin, preg, offset);

                    // Deinterleave to get even/odd elements
                    ops::Deinterleave(vregDyEven, vregDyOdd, vregIn, preg);
                    ops::Deinterleave(vregCosEven, vregCosOdd, vregCos, preg);
                    ops::Deinterleave(vregSinEven, vregSinOdd, vregSin, preg);

                    // cos_even * dy_even, cos_odd * dy_odd
                    Mul(vregCosEven, vregCosEven, vregDyEven, preg);
                    Mul(vregCosOdd, vregCosOdd, vregDyOdd, preg);
                    // sin_even * dy_even, sin_odd * dy_odd
                    Mul(vregSinEven, vregSinEven, vregDyEven, preg);
                    Mul(vregSinOdd, vregSinOdd, vregDyOdd, preg);

                    // dx_even = cos_even*dy_even + sin_odd*dy_odd
                    Add(vregCosEven, vregCosEven, vregSinOdd, preg);
                    // dx_odd  = cos_odd*dy_odd  - sin_even*dy_even
                    Muls(vregSinEven, vregSinEven, float(-1.0), preg);
                    Add(vregCosOdd, vregCosOdd, vregSinEven, preg);

                    // Interleave back to full vector
                    ops::Interleave(vregOut, vregCosEven, vregCosOdd, preg);

                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut, preg, offset);
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
template <typename DTYPE_DY, typename DTYPE_COS>
__aicore__ inline void QuarterGradAlignVF(const LocalTensor<DTYPE_COS> &sinTensor,
                                           const LocalTensor<DTYPE_COS> &cosTensor,
                                           const LocalTensor<DTYPE_DY> &inTensor,
                                           const LocalTensor<DTYPE_DY> &outTensor, uint32_t dLen,
                                           uint32_t dAlignDy, uint16_t currSNum, uint16_t currDNum)
{
    uint32_t quarterD = dLen / QUARTER_MODE_COEF;
    uint32_t quarterDAlignDy =
        AscendC::Ceil(quarterD, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_DY)));
    uint32_t quarterDAlignCos =
        AscendC::Ceil(quarterD, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_COS)));
    uint16_t repeatTimes = AscendC::Ceil(quarterD, static_cast<uint32_t>(VL_FLOAT32_SIZE));

    uint32_t cosRowStride = dLen;

    __local_mem__ DTYPE_COS *sinUb = const_cast<__local_mem__ DTYPE_COS *>(sinTensor.GetPhyAddr());
    __local_mem__ DTYPE_COS *cosUb = const_cast<__local_mem__ DTYPE_COS *>(cosTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *inUb = const_cast<__local_mem__ DTYPE_DY *>(inTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *outUb = const_cast<__local_mem__ DTYPE_DY *>(outTensor.GetPhyAddr());

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> vregIn, vregQ1In, vregQ2In, vregQ3In;
        MicroAPI::RegTensor<float> vregSin, vregQ1Sin, vregQ2Sin, vregQ3Sin;
        MicroAPI::RegTensor<float> vregCos, vregQ1Cos, vregQ2Cos, vregQ3Cos;
        MicroAPI::RegTensor<float> vregOut, vregQ1Out, vregQ2Out, vregQ3Out;
        MicroAPI::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            __local_mem__ DTYPE_COS *currSinUb = sinUb + sIdx * cosRowStride;
            __local_mem__ DTYPE_COS *currCosUb = cosUb + sIdx * cosRowStride;
            for (uint16_t row = 0; row < currDNum; row++) {
                __local_mem__ DTYPE_DY *currInUb = inUb + (sIdx * currDNum + row) * dAlignDy;
                __local_mem__ DTYPE_DY *currOutUb = outUb + (sIdx * currDNum + row) * dAlignDy;
                uint32_t updateCnt = quarterD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = MicroAPI::UpdateMask<float>(updateCnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t dyQ1Off = offset + quarterDAlignDy;
                    int32_t dyQ2Off = dyQ1Off + quarterDAlignDy;
                    int32_t dyQ3Off = dyQ2Off + quarterDAlignDy;
                    int32_t cosQ1Off = offset + quarterDAlignCos;
                    int32_t cosQ2Off = cosQ1Off + quarterDAlignCos;
                    int32_t cosQ3Off = cosQ2Off + quarterDAlignCos;

                    // Load dy as 4 quarters (dy-aligned)
                    ops::LoadTwoTensorForDtypeT<DTYPE_DY>(currInUb, currInUb, vregIn, vregQ1In, preg, preg,
                                                            offset, dyQ1Off);
                    ops::LoadTwoTensorForDtypeT<DTYPE_DY>(currInUb, currInUb, vregQ2In, vregQ3In, preg, preg,
                                                            dyQ2Off, dyQ3Off);
                    // Load sin, cos as 4 quarters (cos-aligned)
                    ops::LoadTwoTensorForDtypeT<DTYPE_COS>(currSinUb, currSinUb, vregSin, vregQ1Sin, preg, preg,
                                                            offset, cosQ1Off);
                    ops::LoadTwoTensorForDtypeT<DTYPE_COS>(currSinUb, currSinUb, vregQ2Sin, vregQ3Sin, preg, preg,
                                                            cosQ2Off, cosQ3Off);
                    ops::LoadTwoTensorForDtypeT<DTYPE_COS>(currCosUb, currCosUb, vregCos, vregQ1Cos, preg, preg,
                                                            offset, cosQ1Off);
                    ops::LoadTwoTensorForDtypeT<DTYPE_COS>(currCosUb, currCosUb, vregQ2Cos, vregQ3Cos, preg, preg,
                                                            cosQ2Off, cosQ3Off);

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

                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut, preg, offset);
                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregQ1Out, preg, dyQ1Off);
                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregQ2Out, preg, dyQ2Off);
                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregQ3Out, preg, dyQ3Off);
                }
            }
        }
    }
}

// ---- InterleaveHalfModeGradVF ----
// interleave-half mode (rotary_mode=3): dx formula is identical to half mode —
//   dx_left  = cos_left  * dy_left  + sin_right * dy_right
//   dx_right = cos_right * dy_right - sin_left  * dy_left
// (the difference from half mode is only in dcos/dsin, not in dx).
// Matches reference DeepSeekInterleaveModeGradVF in rotary_position_embedding_grad.
template <typename DTYPE_DY, typename DTYPE_COS>
__aicore__ inline void InterleaveHalfModeGradVF(const LocalTensor<DTYPE_COS> &sinTensor,
                                                 const LocalTensor<DTYPE_COS> &cosTensor,
                                                 const LocalTensor<DTYPE_DY> &inTensor,
                                                 const LocalTensor<DTYPE_DY> &outTensor, uint32_t dLen,
                                                 uint16_t currSNum, uint16_t currDNum)
{
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t dAlignDy = AscendC::Ceil(dLen, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_DY)));
    uint32_t halfDAlignDy = AscendC::Ceil(halfD, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_DY)));
    uint32_t halfDAlignCos = AscendC::Ceil(halfD, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_COS)));
    uint16_t repeatTimes = AscendC::Ceil(halfD, static_cast<uint32_t>(VL_FLOAT32_SIZE));

    uint32_t cosRowStride = dLen;

    __local_mem__ DTYPE_COS *sinUb = const_cast<__local_mem__ DTYPE_COS *>(sinTensor.GetPhyAddr());
    __local_mem__ DTYPE_COS *cosUb = const_cast<__local_mem__ DTYPE_COS *>(cosTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *inUb = const_cast<__local_mem__ DTYPE_DY *>(inTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *outUb = const_cast<__local_mem__ DTYPE_DY *>(outTensor.GetPhyAddr());

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> vregIn, vregHalfIn, vregSin, vregHalfSin, vregCos, vregHalfCos, vregOut, vregHalfOut;
        MicroAPI::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            __local_mem__ DTYPE_COS *currSinUb = sinUb + sIdx * cosRowStride;
            __local_mem__ DTYPE_COS *currCosUb = cosUb + sIdx * cosRowStride;
            for (uint16_t row = 0; row < currDNum; row++) {
                __local_mem__ DTYPE_DY *currInUb = inUb + (sIdx * currDNum + row) * dAlignDy;
                __local_mem__ DTYPE_DY *currOutUb = outUb + (sIdx * currDNum + row) * dAlignDy;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = MicroAPI::UpdateMask<float>(updateCnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t halfOffset = offset + halfDAlignDy;
                    int32_t cosHalfOffset = offset + halfDAlignCos;

                    ops::LoadTwoTensorForDtypeT<DTYPE_DY>(currInUb, currInUb, vregIn, vregHalfIn, preg, preg,
                                                            offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<DTYPE_COS>(currSinUb, currSinUb, vregSin, vregHalfSin, preg, preg,
                                                            offset, cosHalfOffset);
                    ops::LoadTwoTensorForDtypeT<DTYPE_COS>(currCosUb, currCosUb, vregCos, vregHalfCos, preg, preg,
                                                            offset, cosHalfOffset);

                    // dx_left = cos_left * dy_left + sin_right * dy_right
                    Mul(vregOut, vregCos, vregIn, preg);
                    Mul(vregHalfSin, vregHalfSin, vregHalfIn, preg);
                    Add(vregOut, vregOut, vregHalfSin, preg);
                    // dx_right = cos_right * dy_right - sin_left * dy_left
                    Mul(vregSin, vregSin, vregIn, preg);
                    Mul(vregHalfCos, vregHalfCos, vregHalfIn, preg);
                    Sub(vregHalfOut, vregHalfCos, vregSin, preg);

                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut, preg, offset);
                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregHalfOut, preg, halfOffset);
                }
            }
        }
    }
}

// ============================================================================
// Batch VF 函数 — B→N→S 三维循环内置，用于 AAndB(非广播) 和 ABAAndBA
// ============================================================================

// ---- BatchHalfGradAlignVF ----
template <typename DTYPE_DY, typename DTYPE_COS, bool IsBBoardcast>
__aicore__ inline void BatchHalfGradAlignVF(__local_mem__ DTYPE_DY *in, __local_mem__ DTYPE_COS *cos,
                                             __local_mem__ DTYPE_COS *sin, __local_mem__ DTYPE_DY *out,
                                             uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t dLen,
                                             int64_t dAlignDy, int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t dHalfSize = dLen / HALF_INTERLEAVE_COEF;
    uint16_t dLoopCount = (dHalfSize + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    uint32_t dHalfOffsetDy = dAlignDy / HALF_INTERLEAVE_COEF;
    uint32_t dHalfOffsetCos =
        AscendC::Ceil(dHalfSize, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_COS)));

    int32_t bStepUb = ubFactorN * ubFactorS * dAlignDy;
    int32_t nStepUb = ubFactorS * dAlignDy;
    uint32_t cosRowStride = static_cast<uint32_t>(dLen);
    int32_t nStepCosUb = ubFactorS * cosRowStride;

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> inPart1Reg, inPart2Reg, cosPart1Reg, cosPart2Reg, sinPart1Reg, sinPart2Reg;
        MicroAPI::MaskReg pregLoop;
        __local_mem__ DTYPE_DY *currInUb, *currOutUb;
        __local_mem__ DTYPE_COS *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t count = dHalfSize;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignDy;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignDy;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * cosRowStride;
                        currSinUb = sin + sIdx * cosRowStride;
                    } else {
                        currCosUb = cos + bIdx * nStepCosUb + sIdx * cosRowStride;
                        currSinUb = sin + bIdx * nStepCosUb + sIdx * cosRowStride;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(count);
                        ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, inPart1Reg, pregLoop,
                                                                i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, inPart2Reg, pregLoop,
                                                                i * VL_FLOAT32_SIZE + dHalfOffsetDy);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, cosPart1Reg, pregLoop,
                                                                i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, cosPart2Reg, pregLoop,
                                                                i * VL_FLOAT32_SIZE + dHalfOffsetCos);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, sinPart1Reg, pregLoop,
                                                                i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, sinPart2Reg, pregLoop,
                                                                i * VL_FLOAT32_SIZE + dHalfOffsetCos);
                        Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregLoop);
                        Add(cosPart1Reg, cosPart1Reg, sinPart2Reg, pregLoop);
                        Mul(cosPart2Reg, inPart2Reg, cosPart2Reg, pregLoop);
                        Mul(sinPart1Reg, inPart1Reg, sinPart1Reg, pregLoop);
                        Sub(cosPart2Reg, cosPart2Reg, sinPart1Reg, pregLoop);
                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, cosPart1Reg, pregLoop,
                                                                i * VL_FLOAT32_SIZE);
                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, cosPart2Reg, pregLoop,
                                                                i * VL_FLOAT32_SIZE + dHalfOffsetDy);
                    }
                }
            }
        }
    }
}

// ---- BatchQuarterGradAlignVF ----
// quarter mode (rotary_mode=2): batch version.
// Matches reference in rotary_position_embedding_grad.
template <typename DTYPE_DY, typename DTYPE_COS, bool IsBBoardcast>
__aicore__ inline void BatchQuarterGradAlignVF(
    __local_mem__ DTYPE_DY *in, __local_mem__ DTYPE_COS *cos, __local_mem__ DTYPE_COS *sin,
    __local_mem__ DTYPE_DY *out, uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t dLen,
    int64_t dAlignDy, int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t dQuarterSize = dLen / QUARTER_MODE_COEF;
    uint16_t dLoopCount = (dQuarterSize + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    uint32_t dQuarterOffsetDy = dAlignDy / QUARTER_MODE_COEF;
    uint32_t dHalfOffsetDy = dAlignDy / HALF_INTERLEAVE_COEF;
    uint32_t dThreeQuarterOffsetDy = dQuarterOffsetDy + dHalfOffsetDy;
    uint32_t dQuarterOffsetCos =
        AscendC::Ceil(static_cast<uint32_t>(dQuarterSize),
                      static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_COS)));
    uint32_t dHalfOffsetCos = dQuarterOffsetCos * HALF_INTERLEAVE_COEF;
    uint32_t dThreeQuarterOffsetCos = dQuarterOffsetCos + dHalfOffsetCos;

    int32_t bStepUb = ubFactorN * ubFactorS * dAlignDy;
    int32_t nStepUb = ubFactorS * dAlignDy;
    uint32_t cosRowStride = static_cast<uint32_t>(dLen);
    int32_t nStepCosUb = ubFactorS * cosRowStride;

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> inPart1Reg, inPart2Reg, inPart3Reg, inPart4Reg;
        MicroAPI::RegTensor<float> cosPart1Reg, cosPart2Reg, cosPart3Reg, cosPart4Reg;
        MicroAPI::RegTensor<float> sinPart1Reg, sinPart2Reg, sinPart3Reg, sinPart4Reg;
        MicroAPI::MaskReg pregLoop;
        __local_mem__ DTYPE_DY *currInUb, *currOutUb;
        __local_mem__ DTYPE_COS *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t count = dQuarterSize;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignDy;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignDy;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * cosRowStride;
                        currSinUb = sin + sIdx * cosRowStride;
                    } else {
                        currCosUb = cos + bIdx * nStepCosUb + sIdx * cosRowStride;
                        currSinUb = sin + bIdx * nStepCosUb + sIdx * cosRowStride;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(count);
                        int32_t off0 = i * VL_FLOAT32_SIZE;
                        int32_t dyOff1 = off0 + dQuarterOffsetDy;
                        int32_t dyOff2 = off0 + dHalfOffsetDy;
                        int32_t dyOff3 = off0 + dThreeQuarterOffsetDy;
                        int32_t cosOff0 = off0;
                        int32_t cosOff1 = off0 + dQuarterOffsetCos;
                        int32_t cosOff2 = off0 + 2 * dQuarterOffsetCos;
                        int32_t cosOff3 = off0 + 3 * dQuarterOffsetCos;

                        // Load dy as 4 quarters (dy-aligned)
                        ops::LoadTwoTensorForDtypeT<DTYPE_DY>(currInUb, currInUb, inPart1Reg, inPart2Reg,
                                                                pregLoop, pregLoop, off0, dyOff1);
                        ops::LoadTwoTensorForDtypeT<DTYPE_DY>(currInUb, currInUb, inPart3Reg, inPart4Reg,
                                                                pregLoop, pregLoop, dyOff2, dyOff3);
                        // Load sin, cos as 4 quarters (cos-aligned)
                        ops::LoadTwoTensorForDtypeT<DTYPE_COS>(currSinUb, currSinUb, sinPart1Reg, sinPart2Reg,
                                                                pregLoop, pregLoop, cosOff0, cosOff1);
                        ops::LoadTwoTensorForDtypeT<DTYPE_COS>(currSinUb, currSinUb, sinPart3Reg, sinPart4Reg,
                                                                pregLoop, pregLoop, cosOff2, cosOff3);
                        ops::LoadTwoTensorForDtypeT<DTYPE_COS>(currCosUb, currCosUb, cosPart1Reg, cosPart2Reg,
                                                                pregLoop, pregLoop, cosOff0, cosOff1);
                        ops::LoadTwoTensorForDtypeT<DTYPE_COS>(currCosUb, currCosUb, cosPart3Reg, cosPart4Reg,
                                                                pregLoop, pregLoop, cosOff2, cosOff3);

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

                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, cosPart1Reg, pregLoop, off0);
                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, cosPart2Reg, pregLoop, dyOff1);
                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, cosPart3Reg, pregLoop, dyOff2);
                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, cosPart4Reg, pregLoop, dyOff3);
                    }
                }
            }
        }
    }
}

// ---- BatchInterleaveModeGradVF ----
// interleave mode (rotary_mode=1): batch version.
// Matches reference in rotary_position_embedding_grad.
template <typename DTYPE_DY, typename DTYPE_COS, bool IsBBoardcast>
__aicore__ inline void BatchInterleaveModeGradVF(
    __local_mem__ DTYPE_DY *in, __local_mem__ DTYPE_COS *cos, __local_mem__ DTYPE_COS *sin,
    __local_mem__ DTYPE_DY *out, uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t dLen,
    int64_t dAlignDy, int64_t ubFactorS, int64_t ubFactorN)
{
    uint16_t dLoopCount = (dLen + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;

    int32_t bStepUb = ubFactorN * ubFactorS * dAlignDy;
    int32_t nStepUb = ubFactorS * dAlignDy;
    uint32_t cosRowStride = static_cast<uint32_t>(dLen);
    int32_t nStepCosUb = ubFactorS * cosRowStride;

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> vregIn, vregCos, vregSin, vregOut;
        MicroAPI::RegTensor<float> vregDyEven, vregDyOdd, vregCosEven, vregCosOdd, vregSinEven, vregSinOdd;
        MicroAPI::MaskReg pregLoop;
        __local_mem__ DTYPE_DY *currInUb, *currOutUb;
        __local_mem__ DTYPE_COS *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t count = dLen;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignDy;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignDy;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * cosRowStride;
                        currSinUb = sin + sIdx * cosRowStride;
                    } else {
                        currCosUb = cos + bIdx * nStepCosUb + sIdx * cosRowStride;
                        currSinUb = sin + bIdx * nStepCosUb + sIdx * cosRowStride;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(count);
                        int32_t offset = i * VL_FLOAT32_SIZE;

                        // Load full interleaved dy, cos, sin
                        ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, vregIn, pregLoop, offset);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, vregCos, pregLoop, offset);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, vregSin, pregLoop, offset);

                        // Deinterleave to get even/odd elements
                        ops::Deinterleave(vregDyEven, vregDyOdd, vregIn, pregLoop);
                        ops::Deinterleave(vregCosEven, vregCosOdd, vregCos, pregLoop);
                        ops::Deinterleave(vregSinEven, vregSinOdd, vregSin, pregLoop);

                        // cos_even * dy_even, cos_odd * dy_odd
                        Mul(vregCosEven, vregCosEven, vregDyEven, pregLoop);
                        Mul(vregCosOdd, vregCosOdd, vregDyOdd, pregLoop);
                        // sin_even * dy_even, sin_odd * dy_odd
                        Mul(vregSinEven, vregSinEven, vregDyEven, pregLoop);
                        Mul(vregSinOdd, vregSinOdd, vregDyOdd, pregLoop);

                        // dx_even = cos_even*dy_even + sin_odd*dy_odd
                        Add(vregCosEven, vregCosEven, vregSinOdd, pregLoop);
                        // dx_odd  = cos_odd*dy_odd  - sin_even*dy_even
                        Muls(vregSinEven, vregSinEven, float(-1.0), pregLoop);
                        Add(vregCosOdd, vregCosOdd, vregSinEven, pregLoop);

                        // Interleave back to full vector
                        ops::Interleave(vregOut, vregCosEven, vregCosOdd, pregLoop);

                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut, pregLoop, offset);
                    }
                }
            }
        }
    }
}

// ---- BatchInterleaveHalfModeGradVF ----
// interleave-half mode (rotary_mode=3): batch version.
// dx formula identical to half mode — matches reference DeepSeekInterleaveModeGradVF.
template <typename DTYPE_DY, typename DTYPE_COS, bool IsBBoardcast>
__aicore__ inline void BatchInterleaveHalfModeGradVF(
    __local_mem__ DTYPE_DY *in, __local_mem__ DTYPE_COS *cos, __local_mem__ DTYPE_COS *sin,
    __local_mem__ DTYPE_DY *out, uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t dLen,
    int64_t dAlignDy, int64_t ubFactorS, int64_t ubFactorN)
{
    uint16_t dHalfSize = dLen / HALF_INTERLEAVE_COEF;
    uint16_t dLoopCount = (dHalfSize + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    uint32_t dHalfOffsetDy = dAlignDy / HALF_INTERLEAVE_COEF;
    uint32_t dHalfOffsetCos =
        AscendC::Ceil(static_cast<uint32_t>(dHalfSize),
                      static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_COS)));

    int32_t bStepUb = ubFactorN * ubFactorS * dAlignDy;
    int32_t nStepUb = ubFactorS * dAlignDy;
    uint32_t cosRowStride = static_cast<uint32_t>(dLen);
    int32_t nStepCosUb = ubFactorS * cosRowStride;

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> vregIn, vregHalfIn, vregCos, vregHalfCos, vregSin, vregHalfSin, vregOut, vregHalfOut;
        MicroAPI::MaskReg pregLoop;
        __local_mem__ DTYPE_DY *currInUb, *currOutUb;
        __local_mem__ DTYPE_COS *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t count = dHalfSize;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignDy;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignDy;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * cosRowStride;
                        currSinUb = sin + sIdx * cosRowStride;
                    } else {
                        currCosUb = cos + bIdx * nStepCosUb + sIdx * cosRowStride;
                        currSinUb = sin + bIdx * nStepCosUb + sIdx * cosRowStride;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(count);
                        ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, vregIn, pregLoop,
                                                                i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, vregHalfIn, pregLoop,
                                                                i * VL_FLOAT32_SIZE + dHalfOffsetDy);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, vregCos, pregLoop,
                                                                i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, vregHalfCos, pregLoop,
                                                                i * VL_FLOAT32_SIZE + dHalfOffsetCos);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, vregSin, pregLoop,
                                                                i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, vregHalfSin, pregLoop,
                                                                i * VL_FLOAT32_SIZE + dHalfOffsetCos);

                        // dx_left = cos_left * dy_left + sin_right * dy_right
                        Mul(vregOut, vregCos, vregIn, pregLoop);
                        Mul(vregHalfSin, vregHalfSin, vregHalfIn, pregLoop);
                        Add(vregOut, vregOut, vregHalfSin, pregLoop);
                        // dx_right = cos_right * dy_right - sin_left * dy_left
                        Mul(vregSin, vregSin, vregIn, pregLoop);
                        Mul(vregHalfCos, vregHalfCos, vregHalfIn, pregLoop);
                        Sub(vregHalfOut, vregHalfCos, vregSin, pregLoop);

                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut, pregLoop,
                                                                i * VL_FLOAT32_SIZE);
                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregHalfOut, pregLoop,
                                                                i * VL_FLOAT32_SIZE + dHalfOffsetDy);
                    }
                }
            }
        }
    }
}

} // namespace InplacePartialRotaryMulGrad
#endif
