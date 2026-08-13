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
 * \file apply_rotary_pos_emb_grad_common.h
 * \brief
 */

#ifndef APPLY_ROTARY_POS_EMB_GRAD_COMMON_H
#define APPLY_ROTARY_POS_EMB_GRAD_COMMON_H

#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"
#include "op_kernel/math_util.h"

// 复用原有 apply_rotary_pos_emb 常量定义
#if __has_include("../../apply_rotary_pos_emb/arch35/apply_rotary_pos_emb_common.h")
#include "../../apply_rotary_pos_emb/arch35/apply_rotary_pos_emb_common.h"
#else
#include "../../apply_rotary_pos_emb/op_kernel/arch35/apply_rotary_pos_emb_common.h"
#endif

using namespace AscendC;

namespace ApplyRotaryPosEmbGrad {

// 常量复用（从 apply_rotary_pos_emb_common 继承而来，此处重新声明以保证独立可用）
constexpr uint32_t VL_FLOAT32_SIZE = GetVRegSize() / sizeof(float);
constexpr uint32_t BLOCK_TYPE_SIZE = GetUbBlockSize();
constexpr uint32_t HALF_COEF = 2;
constexpr uint32_t DOUBLE_BUFFER = 2;

/*
    grad_out_1 = cos_1 * grad_1 + sin_2 * grad_2
    grad_out_2 = cos_2 * grad_2 - sin_1 * grad_1
*/
template <typename T>
__aicore__ inline void HalfGradAlignVF(
    const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor, const LocalTensor<T>& inTensor,
    const LocalTensor<T>& outTensor, uint32_t dLen, uint32_t dAlign, uint16_t currSNum, uint16_t currDNum)
{
    __local_mem__ T* sinUb = (__local_mem__ T*)sinTensor.GetPhyAddr();
    __local_mem__ T* cosUb = (__local_mem__ T*)cosTensor.GetPhyAddr();
    __local_mem__ T* inUb = (__local_mem__ T*)inTensor.GetPhyAddr();
    __local_mem__ T* outUb = (__local_mem__ T*)outTensor.GetPhyAddr();
    uint32_t halfD = dLen / HALF_COEF;
    uint32_t halfDAlign = Ops::Base::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = Ops::Base::CeilDiv(halfD, VL_FLOAT32_SIZE);
    __local_mem__ T *currInUb, *currOutUb, *currSinUb, *currCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregIn, vregHalfIn, vregSin, vregHalfSin, vregCos, vregHalfCos, vregOut, vregHalfOut;
        MicroAPI::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlign;
            currCosUb = cosUb + sIdx * dAlign;
            for (uint16_t row = 0; row < currDNum; row++) {
                currInUb = inUb + (sIdx * currDNum + row) * dAlign;
                currOutUb = outUb + (sIdx * currDNum + row) * dAlign;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = MicroAPI::UpdateMask<float>(updateCnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t halfOffset = offset + halfDAlign;
                    ops::LoadTwoTensorForDtypeT<T>(
                        currInUb, currInUb, vregIn, vregHalfIn, preg, preg, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currSinUb, currSinUb, vregSin, vregHalfSin, preg, preg, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currCosUb, currCosUb, vregCos, vregHalfCos, preg, preg, offset, halfOffset);

                    Mul(vregOut, vregCos, vregIn, preg);
                    Mul(vregHalfSin, vregHalfSin, vregHalfIn, preg);
                    Add(vregOut, vregOut, vregHalfSin, preg);
                    Mul(vregSin, vregSin, vregIn, preg);
                    Mul(vregHalfCos, vregHalfCos, vregHalfIn, preg);
                    Sub(vregHalfOut, vregHalfCos, vregSin, preg);

                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregOut, preg, offset);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregHalfOut, preg, halfOffset);
                }
            }
        }
    }
}

// ============================================================================
// BatchHalfGradContiguousVF — A 模板 half gradient
// UB 中每行按 dAlign = CeilAlign(D/2, BLOCK/sizeof(T)) * 2 存放（半行分别 32B 对齐落位），
// 后半段偏移为 dAlign/2，保证 D 非 16 对齐（如 D=100）时向量 Load/Store 地址仍然对齐。
// ============================================================================
template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchHalfGradContiguousVF(
    __local_mem__ T* in, __local_mem__ T* cos, __local_mem__ T* sin, __local_mem__ T* out, uint16_t sLength,
    uint16_t bLength, uint16_t nLength, int64_t d, int64_t dAlign, int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t dHalfSize = d / HALF_COEF;
    uint16_t dLoopCount = (dHalfSize + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    uint32_t dHalfOffset = dAlign / HALF_COEF;

    int32_t bStepUb = ubFactorN * ubFactorS * dAlign;
    int32_t nStepUb = ubFactorS * dAlign;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inPart1Reg, inPart2Reg, cosPart1Reg, cosPart2Reg, sinPart1Reg, sinPart2Reg;
        MicroAPI::MaskReg pregLoop;
        __local_mem__ T *currInUb, *currOutUb, *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t count = dHalfSize;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * dAlign;
                        currSinUb = sin + sIdx * dAlign;
                    } else {
                        currCosUb = cos + bIdx * nStepUb + sIdx * dAlign;
                        currSinUb = sin + bIdx * nStepUb + sIdx * dAlign;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(count);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(
                            currInUb, inPart2Reg, pregLoop, i * VL_FLOAT32_SIZE + dHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(
                            currCosUb, cosPart2Reg, pregLoop, i * VL_FLOAT32_SIZE + dHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(
                            currSinUb, sinPart2Reg, pregLoop, i * VL_FLOAT32_SIZE + dHalfOffset);
                        Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregLoop);
                        Add(cosPart1Reg, cosPart1Reg, sinPart2Reg, pregLoop);
                        Mul(cosPart2Reg, inPart2Reg, cosPart2Reg, pregLoop);
                        Mul(sinPart1Reg, inPart1Reg, sinPart1Reg, pregLoop);
                        Sub(cosPart2Reg, cosPart2Reg, sinPart1Reg, pregLoop);
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::StoreOneTensorForDtypeT<T>(
                            currOutUb, cosPart2Reg, pregLoop, i * VL_FLOAT32_SIZE + dHalfOffset);
                    }
                }
            }
        }
    }
}

// ============================================================================
// AccumDcosDsinVF — A 模板融合阶段 grad_cos/grad_sin 部分积累加
// 单路 (grad·x) 的部分积，IsFirst=true 时写入 accumCos/accumSin，
// IsFirst=false 时累加（用于 Q 路写完、K 路累加）。
// UB 行步长为 dAlign（半行各自 32B 对齐），半行偏移为 dAlign/2。
// grad_cos = grad·x（逐元素）
// grad_sin = -(grad1·x2) + grad2·x1（交叉项，前半取负，对应 rotate(x)=cat(-x2,x1)）
// ============================================================================
template <typename T, bool IsFirst>
__aicore__ inline void AccumDcosDsinVF(__local_mem__ T* gradUb, __local_mem__ T* xUb,
                                       __local_mem__ T* accumCosUb, __local_mem__ T* accumSinUb,
                                       uint16_t rowNum, int64_t d, int64_t dAlign)
{
    uint32_t halfD = d / HALF_COEF;
    uint32_t halfDAlign = dAlign / HALF_COEF;
    uint16_t repeatTimes = (halfD + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> g1, g2, x1, x2, c1, c2, s1, s2, t1, t2;
        MicroAPI::MaskReg preg;
        for (uint16_t row = 0; row < rowNum; row++) {
            __local_mem__ T* currGradUb = gradUb + row * dAlign;
            __local_mem__ T* currXUb     = xUb + row * dAlign;
            __local_mem__ T* currCosUb   = accumCosUb + row * dAlign;
            __local_mem__ T* currSinUb   = accumSinUb + row * dAlign;
            uint32_t updateCnt = halfD;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = MicroAPI::UpdateMask<float>(updateCnt);
                int32_t offset = i * VL_FLOAT32_SIZE;
                int32_t halfOffset = offset + halfDAlign;
                ops::LoadTwoTensorForDtypeT<T>(currGradUb, currGradUb, g1, g2, preg, preg, offset, halfOffset);
                ops::LoadTwoTensorForDtypeT<T>(currXUb, currXUb, x1, x2, preg, preg, offset, halfOffset);
                // grad_cos = grad·x
                Mul(c1, g1, x1, preg);
                Mul(c2, g2, x2, preg);
                // grad_sin = -(grad1·x2) + grad2·x1
                Mul(s1, g1, x2, preg);
                Muls(s1, s1, static_cast<float>(-1.0), preg);
                Mul(s2, g2, x1, preg);
                if constexpr (IsFirst) {
                    ops::StoreOneTensorForDtypeT<T>(currCosUb, c1, preg, offset);
                    ops::StoreOneTensorForDtypeT<T>(currCosUb, c2, preg, halfOffset);
                    ops::StoreOneTensorForDtypeT<T>(currSinUb, s1, preg, offset);
                    ops::StoreOneTensorForDtypeT<T>(currSinUb, s2, preg, halfOffset);
                } else {
                    MicroAPI::RegTensor<float> oldC1, oldC2, oldS1, oldS2;
                    ops::LoadOneTensorForDtypeT<T>(currCosUb, oldC1, preg, offset);
                    ops::LoadOneTensorForDtypeT<T>(currCosUb, oldC2, preg, halfOffset);
                    ops::LoadOneTensorForDtypeT<T>(currSinUb, oldS1, preg, offset);
                    ops::LoadOneTensorForDtypeT<T>(currSinUb, oldS2, preg, halfOffset);
                    Add(oldC1, oldC1, c1, preg);
                    Add(oldC2, oldC2, c2, preg);
                    Add(oldS1, oldS1, s1, preg);
                    Add(oldS2, oldS2, s2, preg);
                    ops::StoreOneTensorForDtypeT<T>(currCosUb, oldC1, preg, offset);
                    ops::StoreOneTensorForDtypeT<T>(currCosUb, oldC2, preg, halfOffset);
                    ops::StoreOneTensorForDtypeT<T>(currSinUb, oldS1, preg, offset);
                    ops::StoreOneTensorForDtypeT<T>(currSinUb, oldS2, preg, halfOffset);
                }
            }
        }
    }
}

} // namespace ApplyRotaryPosEmbGrad

#endif // APPLY_ROTARY_POS_EMB_GRAD_COMMON_H
