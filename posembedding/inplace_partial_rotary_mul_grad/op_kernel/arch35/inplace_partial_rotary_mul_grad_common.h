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
 * \brief Grad vector functions (RegBase / MicroAPI) for InplacePartialRotaryMulGrad.
 *
 * Only the x == nullptr scenario is implemented: dy is rotated in place into dx over the partial slice.
 * The four rotary modes share the same element-wise grad formulas; dy (and the inplace dx output) use
 * dtype Tdy while cos / sin use dtype Tcs (mixed dtype is supported). All compute happens in float32
 * registers; the per-type load / store helpers handle the type conversion.
 *
 * Grad formulas (dLen = sliceLen, per row):
 *   half          (mode 0): dx[0..dLen/2)     = cos1*dy1 + sin2*dy2
 *                          dx[dLen/2..dLen)   = cos2*dy2 - sin1*dy1
 *   interleave    (mode 1): dx[2k]   = cos[2k]*dy[2k]   + sin[2k+1]*dy[2k+1]
 *                          dx[2k+1] = cos[2k+1]*dy[2k+1] - sin[2k]*dy[2k]
 *   quarter       (mode 2): 4-way version of half.
 *   interleave-half(mode 3): like half on the input halves, output stacked (interleaved) back.
 */

#ifndef INPLACE_PARTIAL_ROTARY_MUL_GRAD_COMMON_H
#define INPLACE_PARTIAL_ROTARY_MUL_GRAD_COMMON_H

#include "kernel_operator.h"
// Reuse the strided partial-slice DataCopy helpers (CopyInHalfMode / CopyOutHalfMode / ...) and
// the RegBase constants (VL_FLOAT32_SIZE, BLOCK_TYPE_SIZE, ...) from apply_rotary_pos_emb.
#if __has_include("../../apply_rotary_pos_emb/arch35/apply_rotary_pos_emb_common.h")
#include "../../apply_rotary_pos_emb/arch35/apply_rotary_pos_emb_common.h"
#else
#include "../../apply_rotary_pos_emb/op_kernel/arch35/apply_rotary_pos_emb_common.h"
#endif

using namespace AscendC;

namespace InplacePartialRotaryMulGrad {

/*
 * half / interleave-half grad: split the row into two halves.
 * Inputs (per row): dy halves at [0, halfDAlignDy), cos / sin halves at [0, halfDAlignCs).
 *   dx1 = cos1*dy1 + sin2*dy2 ; dx2 = cos2*dy2 - sin1*dy1
 */
template <typename Tdy, typename Tcs>
__aicore__ inline void GradHalfVF(__local_mem__ Tcs* sinUb, __local_mem__ Tcs* cosUb,
    __local_mem__ Tdy* dyUb, __local_mem__ Tdy* outUb, uint32_t dLen, uint32_t dAlignDy, uint32_t dAlignCs,
    int64_t cosRowStride, int64_t sinRowStride, uint16_t tileRows)
{
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlignDy = dAlignDy / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlignCs = dAlignCs / HALF_INTERLEAVE_COEF;
    uint16_t repeatTimes = Ops::Base::CeilDiv(halfD, VL_FLOAT32_SIZE);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregDy1, vregDy2, vregCos1, vregCos2, vregSin1, vregSin2, vregOut1, vregOut2;
        MicroAPI::MaskReg preg;
        for (uint16_t row = 0; row < tileRows; row++) {
            __local_mem__ Tdy* curDy = dyUb + row * dAlignDy;
            __local_mem__ Tdy* curOut = outUb + row * dAlignDy;
            __local_mem__ Tcs* curCos = cosUb + row * cosRowStride;
            __local_mem__ Tcs* curSin = sinUb + row * sinRowStride;
            uint32_t cnt = halfD;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = MicroAPI::UpdateMask<float>(cnt);
                uint32_t off = i * VL_FLOAT32_SIZE;
                ops::LoadTwoTensorForDtypeT<Tdy>(curDy, curDy, vregDy1, vregDy2, preg, preg, off, off + halfDAlignDy);
                ops::LoadTwoTensorForDtypeT<Tcs>(curCos, curCos, vregCos1, vregCos2, preg, preg, off, off + halfDAlignCs);
                ops::LoadTwoTensorForDtypeT<Tcs>(curSin, curSin, vregSin1, vregSin2, preg, preg, off, off + halfDAlignCs);
                // dx1 = cos1*dy1 + sin2*dy2
                Mul(vregOut1, vregCos1, vregDy1, preg);
                Mul(vregSin2, vregSin2, vregDy2, preg);
                Add(vregOut1, vregOut1, vregSin2, preg);
                // dx2 = cos2*dy2 - sin1*dy1
                Mul(vregOut2, vregCos2, vregDy2, preg);
                Mul(vregSin1, vregSin1, vregDy1, preg);
                Sub(vregOut2, vregOut2, vregSin1, preg);
                ops::StoreOneTensorForDtypeT<Tdy>(curOut, vregOut1, preg, off);
                ops::StoreOneTensorForDtypeT<Tdy>(curOut, vregOut2, preg, off + halfDAlignDy);
            }
        }
    }
}

/*
 * quarter grad: split the row into four quarters.
 *   dx1 = cos1*dy1 + sin2*dy2 ; dx2 = cos2*dy2 - sin1*dy1
 *   dx3 = cos3*dy3 + sin4*dy4 ; dx4 = cos4*dy4 - sin3*dy3
 */
template <typename Tdy, typename Tcs>
__aicore__ inline void GradQuarterVF(__local_mem__ Tcs* sinUb, __local_mem__ Tcs* cosUb,
    __local_mem__ Tdy* dyUb, __local_mem__ Tdy* outUb, uint32_t dLen, uint32_t dAlignDy, uint32_t dAlignCs,
    int64_t cosRowStride, int64_t sinRowStride, uint16_t tileRows)
{
    uint32_t quarterD = dLen / QUARTER_MODE_COEF;
    uint32_t quarterDAlignDy = dAlignDy / QUARTER_MODE_COEF;
    uint32_t quarterDAlignCs = dAlignCs / QUARTER_MODE_COEF;
    uint32_t halfDAlignDy = quarterDAlignDy * 2;
    uint32_t threeQDAlignDy = quarterDAlignDy * 3;
    uint32_t halfDAlignCs = quarterDAlignCs * 2;
    uint32_t threeQDAlignCs = quarterDAlignCs * 3;
    uint16_t repeatTimes = Ops::Base::CeilDiv(quarterD, VL_FLOAT32_SIZE);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vDy1, vDy2, vDy3, vDy4, vCos1, vCos2, vCos3, vCos4, vSin1, vSin2, vSin3, vSin4,
            vOut1, vOut2, vOut3, vOut4;
        MicroAPI::MaskReg preg;
        for (uint16_t row = 0; row < tileRows; row++) {
            __local_mem__ Tdy* curDy = dyUb + row * dAlignDy;
            __local_mem__ Tdy* curOut = outUb + row * dAlignDy;
            __local_mem__ Tcs* curCos = cosUb + row * cosRowStride;
            __local_mem__ Tcs* curSin = sinUb + row * sinRowStride;
            uint32_t cnt = quarterD;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = MicroAPI::UpdateMask<float>(cnt);
                uint32_t off = i * VL_FLOAT32_SIZE;
                ops::LoadTwoTensorForDtypeT<Tdy>(curDy, curDy, vDy1, vDy2, preg, preg, off, off + quarterDAlignDy);
                ops::LoadTwoTensorForDtypeT<Tdy>(curDy, curDy, vDy3, vDy4, preg, preg, off + halfDAlignDy, off + threeQDAlignDy);
                ops::LoadTwoTensorForDtypeT<Tcs>(curCos, curCos, vCos1, vCos2, preg, preg, off, off + quarterDAlignCs);
                ops::LoadTwoTensorForDtypeT<Tcs>(curCos, curCos, vCos3, vCos4, preg, preg, off + halfDAlignCs, off + threeQDAlignCs);
                ops::LoadTwoTensorForDtypeT<Tcs>(curSin, curSin, vSin1, vSin2, preg, preg, off, off + quarterDAlignCs);
                ops::LoadTwoTensorForDtypeT<Tcs>(curSin, curSin, vSin3, vSin4, preg, preg, off + halfDAlignCs, off + threeQDAlignCs);
                Mul(vOut1, vCos1, vDy1, preg);
                Mul(vSin2, vSin2, vDy2, preg);
                Add(vOut1, vOut1, vSin2, preg);
                Mul(vOut2, vCos2, vDy2, preg);
                Mul(vSin1, vSin1, vDy1, preg);
                Sub(vOut2, vOut2, vSin1, preg);
                Mul(vOut3, vCos3, vDy3, preg);
                Mul(vSin4, vSin4, vDy4, preg);
                Add(vOut3, vOut3, vSin4, preg);
                Mul(vOut4, vCos4, vDy4, preg);
                Mul(vSin3, vSin3, vDy3, preg);
                Sub(vOut4, vOut4, vSin3, preg);
                ops::StoreOneTensorForDtypeT<Tdy>(curOut, vOut1, preg, off);
                ops::StoreOneTensorForDtypeT<Tdy>(curOut, vOut2, preg, off + quarterDAlignDy);
                ops::StoreOneTensorForDtypeT<Tdy>(curOut, vOut3, preg, off + halfDAlignDy);
                ops::StoreOneTensorForDtypeT<Tdy>(curOut, vOut4, preg, off + threeQDAlignDy);
            }
        }
    }
}

/*
 * interleave grad: dx[2k] = cos[2k]*dy[2k] + sin[2k+1]*dy[2k+1] ; dx[2k+1] = cos[2k+1]*dy[2k+1] - sin[2k]*dy[2k]
 * Reads dy / cos / sin as contiguous rows (stride dAlignDy / dAlignCs); writes dx contiguous (stride dAlignDy).
 */
template <typename Tdy, typename Tcs>
__aicore__ inline void GradInterleaveVF(__local_mem__ Tcs* sinUb, __local_mem__ Tcs* cosUb,
    __local_mem__ Tdy* dyUb, __local_mem__ Tdy* outUb, uint16_t dLen, uint32_t dAlignDy, uint32_t dAlignCs,
    int64_t cosRowStride, int64_t sinRowStride, uint16_t tileRows)
{
    uint16_t loopSize = 2 * VL_FLOAT32_SIZE;
    uint16_t loopNum = (dLen + loopSize - 1) / loopSize;
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
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vCosE, vCosO, vSinE, vSinO, vDyE, vDyO, vTmpA, vTmpB;
        MicroAPI::MaskReg pregLoop, pregP1, pregP2;
        for (uint16_t row = 0; row < tileRows; row++) {
            __local_mem__ Tdy* curDy = dyUb + row * dAlignDy;
            __local_mem__ Tdy* curOut = outUb + row * dAlignDy;
            __local_mem__ Tcs* curCos = cosUb + row * cosRowStride;
            __local_mem__ Tcs* curSin = sinUb + row * sinRowStride;
            uint32_t halfCnt = halfNum;
            uint32_t p1Cnt = part1Num;
            uint32_t p2Cnt = part2Num;
            for (uint16_t i = 0; i < loopNum; i++) {
                pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                pregP1 = MicroAPI::UpdateMask<float>(p1Cnt);
                pregP2 = MicroAPI::UpdateMask<float>(p2Cnt);
                uint32_t evenOff = (i * 2) * VL_FLOAT32_SIZE;
                uint32_t oddOff = evenOff + VL_FLOAT32_SIZE;
                ops::LoadOneTensorForDtypeT<Tdy>(curDy, vDyE, pregP1, evenOff);
                ops::LoadOneTensorForDtypeT<Tdy>(curDy, vDyO, pregP2, oddOff);
                ops::LoadOneTensorForDtypeT<Tcs>(curCos, vCosE, pregP1, evenOff);
                ops::LoadOneTensorForDtypeT<Tcs>(curCos, vCosO, pregP2, oddOff);
                ops::LoadOneTensorForDtypeT<Tcs>(curSin, vSinE, pregP1, evenOff);
                ops::LoadOneTensorForDtypeT<Tcs>(curSin, vSinO, pregP2, oddOff);
                Mul(vCosE, vCosE, vDyE, pregP1);
                Mul(vCosO, vCosO, vDyO, pregP2);
                Mul(vSinE, vSinE, vDyE, pregP1);
                Mul(vSinO, vSinO, vDyO, pregP2);
                MicroAPI::DeInterleave<float>(vTmpA, vTmpB, vSinE, vSinO);
                Muls(vTmpA, vTmpA, float(-1.0), pregLoop);
                MicroAPI::Interleave<float>(vTmpA, vTmpB, vTmpB, vTmpA);
                Add(vCosE, vTmpA, vCosE, pregP1);
                Add(vCosO, vTmpB, vCosO, pregP2);
                ops::StoreOneTensorForDtypeT<Tdy>(curOut, vCosE, pregP1, evenOff);
                ops::StoreOneTensorForDtypeT<Tdy>(curOut, vCosO, pregP2, oddOff);
            }
        }
    }
}

/*
 * interleave-half grad (mode 3): dy / cos / sin are read as two contiguous halves; the two output halves are
 * interleaved back into the row. Out layout is interleaved-contiguous (stride dAlignOut = CeilAlign(dLen, *)).
 *   tmp1 = cos1*dy1 + sin2*dy2 ; tmp2 = cos2*dy2 - sin1*dy1 ; out = interleave(tmp1, tmp2)
 */
template <typename Tdy, typename Tcs>
__aicore__ inline void GradInterleaveHalfVF(__local_mem__ Tcs* sinUb, __local_mem__ Tcs* cosUb,
    __local_mem__ Tdy* dyUb, __local_mem__ Tdy* outUb, uint32_t dLen, uint32_t dAlignDy, uint32_t dAlignCs,
    uint32_t dAlignOut, int64_t cosRowStride, int64_t sinRowStride, uint16_t tileRows)
{
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlignDy = dAlignDy / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlignCs = dAlignCs / HALF_INTERLEAVE_COEF;
    uint16_t repeatTimes = Ops::Base::CeilDiv(halfD, VL_FLOAT32_SIZE);
    uint32_t part1Num = (repeatTimes - 1) * VL_FLOAT32_SIZE;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = dLen - part1Num - part2Num;
    if (tailNum > VL_FLOAT32_SIZE) {
        part1Num += VL_FLOAT32_SIZE;
        part2Num += (tailNum - VL_FLOAT32_SIZE);
    } else {
        part1Num += tailNum;
    }
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vDy1, vDy2, vCos1, vCos2, vSin1, vSin2, vOut1, vOut2, vInt1, vInt2;
        MicroAPI::MaskReg pregLoop, pregP1, pregP2;
        for (uint16_t row = 0; row < tileRows; row++) {
            __local_mem__ Tdy* curDy = dyUb + row * dAlignDy;
            __local_mem__ Tdy* curOut = outUb + row * dAlignOut;
            __local_mem__ Tcs* curCos = cosUb + row * cosRowStride;
            __local_mem__ Tcs* curSin = sinUb + row * sinRowStride;
            uint32_t halfCnt = halfD;
            uint32_t p1Cnt = part1Num;
            uint32_t p2Cnt = part2Num;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                pregP1 = MicroAPI::UpdateMask<float>(p1Cnt);
                pregP2 = MicroAPI::UpdateMask<float>(p2Cnt);
                uint32_t off = i * VL_FLOAT32_SIZE;
                uint32_t halfOff = off + halfDAlignCs;
                uint32_t halfOffDy = off + halfDAlignDy;
                uint32_t outOff = off * HALF_INTERLEAVE_COEF;
                ops::LoadTwoTensorForDtypeT<Tdy>(curDy, curDy, vDy1, vDy2, pregLoop, pregLoop, off, halfOffDy);
                ops::LoadTwoTensorForDtypeT<Tcs>(curCos, curCos, vCos1, vCos2, pregLoop, pregLoop, off, halfOff);
                ops::LoadTwoTensorForDtypeT<Tcs>(curSin, curSin, vSin1, vSin2, pregLoop, pregLoop, off, halfOff);
                Mul(vOut1, vCos1, vDy1, pregLoop);
                Mul(vSin2, vSin2, vDy2, pregLoop);
                Add(vOut1, vOut1, vSin2, pregLoop);
                Mul(vOut2, vCos2, vDy2, pregLoop);
                Mul(vSin1, vSin1, vDy1, pregLoop);
                Sub(vOut2, vOut2, vSin1, pregLoop);
                MicroAPI::Interleave<float>(vInt1, vInt2, vOut1, vOut2);
                ops::StoreOneTensorForDtypeT<Tdy>(curOut, vInt1, pregP1, outOff);
                ops::StoreOneTensorForDtypeT<Tdy>(curOut, vInt2, pregP2, outOff + VL_FLOAT32_SIZE);
            }
        }
    }
}

} // namespace InplacePartialRotaryMulGrad

#endif // INPLACE_PARTIAL_ROTARY_MUL_GRAD_COMMON_H
