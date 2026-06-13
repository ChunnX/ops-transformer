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

    uint32_t rowStrideBytes = dAlignDy * sizeof(DTYPE_DY);
    uint32_t cosRowStride = rowStrideBytes / sizeof(DTYPE_COS);

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
template <typename DTYPE_DY, typename DTYPE_COS>
__aicore__ inline void InterleaveModeGradVF(const LocalTensor<DTYPE_COS> &sinTensor,
                                             const LocalTensor<DTYPE_COS> &cosTensor,
                                             const LocalTensor<DTYPE_DY> &inTensor,
                                             const LocalTensor<DTYPE_DY> &outTensor, uint32_t dLen,
                                             uint16_t currSNum, uint16_t currDNum)
{
    uint32_t dAlignDy = AscendC::Ceil(dLen, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_DY)));
    uint16_t repeatTimes = AscendC::Ceil(dLen, static_cast<uint32_t>(VL_FLOAT32_SIZE));

    uint32_t rowStrideBytes = dAlignDy * sizeof(DTYPE_DY);
    uint32_t cosRowStride = rowStrideBytes / sizeof(DTYPE_COS);

    __local_mem__ DTYPE_COS *sinUb = const_cast<__local_mem__ DTYPE_COS *>(sinTensor.GetPhyAddr());
    __local_mem__ DTYPE_COS *cosUb = const_cast<__local_mem__ DTYPE_COS *>(cosTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *inUb = const_cast<__local_mem__ DTYPE_DY *>(inTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *outUb = const_cast<__local_mem__ DTYPE_DY *>(outTensor.GetPhyAddr());

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> vregIn, vregCos, vregSin, vregOut, vregEven, vregOdd;
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

                    ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, vregIn, preg, offset);
                    ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, vregCos, preg, offset);
                    ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, vregSin, preg, offset);

                    ops::Deinterleave(vregEven, vregOdd, vregIn, preg);
                    Mul(vregOut, vregCos, vregIn, preg);
                    Mul(vregSin, vregSin, vregIn, preg);
                    ops::Deinterleave(vregCos, vregIn, vregSin, preg);
                    Sub(vregEven, vregOut, vregIn, preg);
                    Add(vregOdd, vregCos, vregOut, preg);
                    ops::Interleave(vregOut, vregEven, vregOdd, preg);

                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut, preg, offset);
                }
            }
        }
    }
}

// ---- QuarterGradAlignVF ----
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

    uint32_t rowStrideBytes = dAlignDy * sizeof(DTYPE_DY);
    uint32_t cosRowStride = rowStrideBytes / sizeof(DTYPE_COS);

    __local_mem__ DTYPE_COS *sinUb = const_cast<__local_mem__ DTYPE_COS *>(sinTensor.GetPhyAddr());
    __local_mem__ DTYPE_COS *cosUb = const_cast<__local_mem__ DTYPE_COS *>(cosTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *inUb = const_cast<__local_mem__ DTYPE_DY *>(inTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *outUb = const_cast<__local_mem__ DTYPE_DY *>(outTensor.GetPhyAddr());

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> vregIn0, vregIn1, vregIn2, vregIn3;
        MicroAPI::RegTensor<float> vregCos0, vregCos1, vregCos2, vregCos3;
        MicroAPI::RegTensor<float> vregSin0, vregSin1, vregSin2, vregSin3;
        MicroAPI::RegTensor<float> vregOut0, vregOut1, vregOut2, vregOut3;
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
                    int32_t q1Off = offset + quarterDAlignDy;
                    int32_t q2Off = q1Off + quarterDAlignDy;
                    int32_t q3Off = q2Off + quarterDAlignDy;
                    int32_t cosQOff = i * static_cast<int32_t>(quarterDAlignCos);

                    ops::LoadFourTensorForDtypeT<DTYPE_DY>(currInUb, currInUb, currInUb, currInUb, vregIn0, vregIn1,
                                                            vregIn2, vregIn3, preg, preg, preg, preg, offset, q1Off,
                                                            q2Off, q3Off);
                    ops::LoadFourTensorForDtypeT<DTYPE_COS>(currSinUb, currSinUb, currSinUb, currSinUb, vregSin0,
                                                             vregSin1, vregSin2, vregSin3, preg, preg, preg, preg,
                                                             cosQOff, cosQOff + quarterDAlignCos,
                                                             cosQOff + 2 * quarterDAlignCos,
                                                             cosQOff + 3 * quarterDAlignCos);
                    ops::LoadFourTensorForDtypeT<DTYPE_COS>(currCosUb, currCosUb, currCosUb, currCosUb, vregCos0,
                                                             vregCos1, vregCos2, vregCos3, preg, preg, preg, preg,
                                                             cosQOff, cosQOff + quarterDAlignCos,
                                                             cosQOff + 2 * quarterDAlignCos,
                                                             cosQOff + 3 * quarterDAlignCos);

                    Mul(vregOut0, vregCos0, vregIn0, preg);
                    Mul(vregOut1, vregCos1, vregIn3, preg);
                    Add(vregOut0, vregOut0, vregOut1, preg);
                    Mul(vregOut1, vregCos2, vregIn2, preg);
                    Mul(vregOut2, vregCos3, vregIn1, preg);
                    Add(vregOut1, vregOut1, vregOut2, preg);
                    Mul(vregOut2, vregSin0, vregIn0, preg);
                    Mul(vregOut3, vregSin1, vregIn3, preg);
                    Add(vregOut2, vregOut2, vregOut3, preg);
                    Mul(vregOut3, vregSin2, vregIn2, preg);
                    Add(vregOut2, vregOut2, vregOut3, preg);
                    Sub(vregOut1, vregOut1, vregOut2, preg);

                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut0, preg, offset);
                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut1, preg, q1Off);
                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut0, preg, q2Off);
                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut1, preg, q3Off);
                }
            }
        }
    }
}

// ---- InterleaveHalfModeGradVF ----
template <typename DTYPE_DY, typename DTYPE_COS>
__aicore__ inline void InterleaveHalfModeGradVF(const LocalTensor<DTYPE_COS> &sinTensor,
                                                 const LocalTensor<DTYPE_COS> &cosTensor,
                                                 const LocalTensor<DTYPE_DY> &inTensor,
                                                 const LocalTensor<DTYPE_DY> &outTensor, uint32_t dLen,
                                                 uint16_t currSNum, uint16_t currDNum)
{
    uint32_t dAlignDy = AscendC::Ceil(dLen, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_DY)));
    uint16_t repeatTimes = AscendC::Ceil(dLen, static_cast<uint32_t>(VL_FLOAT32_SIZE));

    uint32_t rowStrideBytes = dAlignDy * sizeof(DTYPE_DY);
    uint32_t cosRowStride = rowStrideBytes / sizeof(DTYPE_COS);

    __local_mem__ DTYPE_COS *sinUb = const_cast<__local_mem__ DTYPE_COS *>(sinTensor.GetPhyAddr());
    __local_mem__ DTYPE_COS *cosUb = const_cast<__local_mem__ DTYPE_COS *>(cosTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *inUb = const_cast<__local_mem__ DTYPE_DY *>(inTensor.GetPhyAddr());
    __local_mem__ DTYPE_DY *outUb = const_cast<__local_mem__ DTYPE_DY *>(outTensor.GetPhyAddr());

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> vregIn, vregCos, vregSin, vregOut, vregEven, vregOdd;
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

                    ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, vregIn, preg, offset);
                    ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, vregCos, preg, offset);
                    ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, vregSin, preg, offset);

                    ops::Deinterleave(vregEven, vregOdd, vregIn, preg);
                    Mul(vregCos, vregCos, vregEven, preg);
                    Mul(vregSin, vregSin, vregOdd, preg);
                    ops::Interleave(vregOut, vregCos, vregSin, preg);

                    ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut, preg, offset);
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
    uint32_t cosRowStride = dAlignDy * sizeof(DTYPE_DY) / sizeof(DTYPE_COS);
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
        AscendC::Ceil(dQuarterSize, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_COS)));

    int32_t bStepUb = ubFactorN * ubFactorS * dAlignDy;
    int32_t nStepUb = ubFactorS * dAlignDy;
    uint32_t cosRowStride = dAlignDy * sizeof(DTYPE_DY) / sizeof(DTYPE_COS);
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
                        int32_t off1 = off0 + dQuarterOffsetDy;
                        int32_t off2 = off0 + dHalfOffsetDy;
                        int32_t off3 = off0 + dThreeQuarterOffsetDy;
                        int32_t cosOff0 = i * static_cast<int32_t>(dQuarterOffsetCos);
                        int32_t cosOff1 = cosOff0 + dQuarterOffsetCos;
                        int32_t cosOff2 = cosOff0 + 2 * dQuarterOffsetCos;
                        int32_t cosOff3 = cosOff0 + 3 * dQuarterOffsetCos;

                        ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, inPart1Reg, pregLoop, off0);
                        ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, inPart2Reg, pregLoop, off1);
                        ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, inPart3Reg, pregLoop, off2);
                        ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, inPart4Reg, pregLoop, off3);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, cosPart1Reg, pregLoop, cosOff0);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, cosPart2Reg, pregLoop, cosOff1);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, cosPart3Reg, pregLoop, cosOff2);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, cosPart4Reg, pregLoop, cosOff3);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, sinPart1Reg, pregLoop, cosOff0);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, sinPart2Reg, pregLoop, cosOff1);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, sinPart3Reg, pregLoop, cosOff2);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, sinPart4Reg, pregLoop, cosOff3);

                        Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Mul(cosPart2Reg, inPart4Reg, cosPart2Reg, pregLoop);
                        Add(cosPart2Reg, cosPart2Reg, cosPart1Reg, pregLoop);
                        Mul(cosPart3Reg, inPart3Reg, cosPart3Reg, pregLoop);
                        Mul(sinPart1Reg, inPart2Reg, sinPart1Reg, pregLoop);
                        Add(cosPart3Reg, cosPart3Reg, sinPart1Reg, pregLoop);
                        Mul(sinPart2Reg, inPart1Reg, sinPart2Reg, pregLoop);
                        Mul(sinPart3Reg, inPart4Reg, sinPart3Reg, pregLoop);
                        Add(sinPart2Reg, sinPart2Reg, sinPart3Reg, pregLoop);
                        Sub(sinPart2Reg, cosPart3Reg, sinPart2Reg, pregLoop);
                        Mul(sinPart3Reg, inPart3Reg, sinPart3Reg, pregLoop);
                        Mul(sinPart1Reg, inPart2Reg, sinPart1Reg, pregLoop);
                        Add(sinPart3Reg, sinPart3Reg, sinPart1Reg, pregLoop);
                        Sub(cosPart1Reg, cosPart2Reg, sinPart3Reg, pregLoop);

                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, cosPart1Reg, pregLoop, off0);
                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, sinPart2Reg, pregLoop, off1);
                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, cosPart1Reg, pregLoop, off2);
                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, sinPart2Reg, pregLoop, off3);
                    }
                }
            }
        }
    }
}

// ---- BatchInterleaveModeGradVF ----
template <typename DTYPE_DY, typename DTYPE_COS, bool IsBBoardcast>
__aicore__ inline void BatchInterleaveModeGradVF(
    __local_mem__ DTYPE_DY *in, __local_mem__ DTYPE_COS *cos, __local_mem__ DTYPE_COS *sin,
    __local_mem__ DTYPE_DY *out, uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t dLen,
    int64_t dAlignDy, int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t dAlignDyU32 = dAlignDy;
    uint16_t dLoopCount = (dLen + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;

    int32_t bStepUb = ubFactorN * ubFactorS * dAlignDy;
    int32_t nStepUb = ubFactorS * dAlignDy;
    uint32_t cosRowStride = dAlignDy * sizeof(DTYPE_DY) / sizeof(DTYPE_COS);
    int32_t nStepCosUb = ubFactorS * cosRowStride;

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> vregIn, vregCos, vregSin, vregOut, vregEven, vregOdd;
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

                        ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, vregIn, pregLoop, offset);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, vregCos, pregLoop, offset);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, vregSin, pregLoop, offset);

                        ops::Deinterleave(vregEven, vregOdd, vregIn, pregLoop);
                        Mul(vregOut, vregCos, vregIn, pregLoop);
                        Mul(vregSin, vregSin, vregIn, pregLoop);
                        ops::Deinterleave(vregCos, vregIn, vregSin, pregLoop);
                        Sub(vregEven, vregOut, vregIn, pregLoop);
                        Add(vregOdd, vregCos, vregOut, pregLoop);
                        ops::Interleave(vregOut, vregEven, vregOdd, pregLoop);

                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut, pregLoop, offset);
                    }
                }
            }
        }
    }
}

// ---- BatchInterleaveHalfModeGradVF ----
template <typename DTYPE_DY, typename DTYPE_COS, bool IsBBoardcast>
__aicore__ inline void BatchInterleaveHalfModeGradVF(
    __local_mem__ DTYPE_DY *in, __local_mem__ DTYPE_COS *cos, __local_mem__ DTYPE_COS *sin,
    __local_mem__ DTYPE_DY *out, uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t dLen,
    int64_t dAlignDy, int64_t ubFactorS, int64_t ubFactorN)
{
    uint16_t dLoopCount = (dLen + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;

    int32_t bStepUb = ubFactorN * ubFactorS * dAlignDy;
    int32_t nStepUb = ubFactorS * dAlignDy;
    uint32_t cosRowStride = dAlignDy * sizeof(DTYPE_DY) / sizeof(DTYPE_COS);
    int32_t nStepCosUb = ubFactorS * cosRowStride;

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> vregIn, vregCos, vregSin, vregOut, vregEven, vregOdd;
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

                        ops::LoadOneTensorForDtypeT<DTYPE_DY>(currInUb, vregIn, pregLoop, offset);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currCosUb, vregCos, pregLoop, offset);
                        ops::LoadOneTensorForDtypeT<DTYPE_COS>(currSinUb, vregSin, pregLoop, offset);

                        ops::Deinterleave(vregEven, vregOdd, vregIn, pregLoop);
                        Mul(vregCos, vregCos, vregEven, pregLoop);
                        Mul(vregSin, vregSin, vregOdd, pregLoop);
                        ops::Interleave(vregOut, vregCos, vregSin, pregLoop);

                        ops::StoreOneTensorForDtypeT<DTYPE_DY>(currOutUb, vregOut, pregLoop, offset);
                    }
                }
            }
        }
    }
}

} // namespace InplacePartialRotaryMulGrad
#endif
