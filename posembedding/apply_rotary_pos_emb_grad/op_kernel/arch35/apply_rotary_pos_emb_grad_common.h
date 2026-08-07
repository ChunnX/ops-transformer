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
 * \file apply_rotary_pos_emb_grad_common.h
 * \brief apply_rotary_pos_emb_grad A 模板公共常量和 VF 计算函数
 * 复用原有 apply_rotary_pos_emb_common.h 的常量
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
constexpr uint32_t HALF_INTERLEAVE_COEF = 2;
constexpr uint32_t DOUBLE_BUFFER = 2;

// ============================================================================
// BatchHalfGradContiguousVF — A 模板连续 D 布局 half gradient
// UB 中每行按完整 D 连续存放，后半段偏移为 D/2，不再使用 D/2 对齐 padding。
// ============================================================================
template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchHalfGradContiguousVF(
    __local_mem__ T* in, __local_mem__ T* cos, __local_mem__ T* sin, __local_mem__ T* out, uint16_t sLength,
    uint16_t bLength, uint16_t nLength, int64_t d, int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t dHalfSize = d / HALF_INTERLEAVE_COEF;
    uint16_t dLoopCount = (dHalfSize + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    uint32_t dHalfOffset = dHalfSize;

    int32_t bStepUb = ubFactorN * ubFactorS * d;
    int32_t nStepUb = ubFactorS * d;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inPart1Reg, inPart2Reg, cosPart1Reg, cosPart2Reg, sinPart1Reg, sinPart2Reg;
        MicroAPI::MaskReg pregLoop;
        __local_mem__ T *currInUb, *currOutUb, *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t count = dHalfSize;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * d;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * d;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * d;
                        currSinUb = sin + sIdx * d;
                    } else {
                        currCosUb = cos + bIdx * nStepUb + sIdx * d;
                        currSinUb = sin + bIdx * nStepUb + sIdx * d;
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

} // namespace ApplyRotaryPosEmbGrad

#endif // APPLY_ROTARY_POS_EMB_GRAD_COMMON_H
