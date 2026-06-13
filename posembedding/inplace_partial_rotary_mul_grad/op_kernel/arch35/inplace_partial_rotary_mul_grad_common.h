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

} // namespace InplacePartialRotaryMulGrad
#endif
