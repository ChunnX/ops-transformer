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
 * \file swiglu_activation.h
 * \brief SwiGLU 和 SwiGLU-Step 门控激活 Reg VF 实现。
 */

#ifndef MEGA_MOE_ARCH35_SWIGLU_ACTIVATION_H
#define MEGA_MOE_ARCH35_SWIGLU_ACTIVATION_H

#if defined(__DAV_C310__)
#include "activation_common.h"
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"

namespace MegaMoeImpl {
namespace Activation {

// SwiGLU 直接调用接口使用的最小参数集。
struct SwiGluParams {
    float clampLimit = 0.0F;
};

constexpr AscendC::Reg::CastTrait SWIGLU_CAST_INPUT_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait SWIGLU_CAST_FP32_TO_BF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

template <typename InputType>
__simd_callee__ inline void LoadSwiGluInputAsFp32(
    AscendC::Reg::RegTensor<float> &dstReg, AscendC::Reg::RegTensor<InputType> &inputReg,
    __ubuf__ InputType *src, AscendC::Reg::MaskReg computeMask)
{
    AscendC::Reg::LoadAlign<InputType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(inputReg, src);
    AscendC::Reg::Cast<float, InputType, SWIGLU_CAST_INPUT_TO_FP32>(dstReg, inputReg, computeMask);
}

__simd_callee__ inline void StoreSwiGluFp32AsBf16(
    __ubuf__ bfloat16_t *dst, AscendC::Reg::RegTensor<bfloat16_t> &outputReg,
    AscendC::Reg::RegTensor<float> &srcReg, AscendC::Reg::MaskReg computeMask,
    AscendC::Reg::MaskReg storeMask)
{
    AscendC::Reg::Cast<bfloat16_t, float, SWIGLU_CAST_FP32_TO_BF16>(outputReg, srcReg, computeMask);
    AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst, outputReg, storeMask);
}

template <bool IsStep>
__simd_callee__ inline void ComputeSwiGluVector(
    AscendC::Reg::RegTensor<float> &outputFp32Reg, AscendC::Reg::RegTensor<float> &gateFp32Reg,
    AscendC::Reg::RegTensor<float> &upFp32Reg, AscendC::Reg::RegTensor<float> &negReg,
    AscendC::Reg::RegTensor<float> &expReg, AscendC::Reg::RegTensor<float> &denominatorReg,
    AscendC::Reg::RegTensor<float> &sigmoidReg, AscendC::Reg::MaskReg computeMask, float clampLimit)
{
    if constexpr (!IsStep) {
        AscendC::Reg::Mins(gateFp32Reg, gateFp32Reg, clampLimit, computeMask);
    }
    AscendC::Reg::Mins(upFp32Reg, upFp32Reg, clampLimit, computeMask);
    AscendC::Reg::Maxs(upFp32Reg, upFp32Reg, -clampLimit, computeMask);

    AscendC::Reg::Muls(negReg, gateFp32Reg, -1.0F, computeMask);
    AscendC::Reg::Exp(expReg, negReg, computeMask);
    AscendC::Reg::Adds(denominatorReg, expReg, 1.0F, computeMask);
    AscendC::Reg::Div(sigmoidReg, gateFp32Reg, denominatorReg, computeMask);
    if constexpr (IsStep) {
        AscendC::Reg::Mins(sigmoidReg, sigmoidReg, clampLimit, computeMask);
    }
    AscendC::Reg::Mul(outputFp32Reg, sigmoidReg, upFp32Reg, computeMask);
}

template <typename InputType, bool FuseTopkWeight, bool IsStep>
__simd_vf__ inline void SwiGluVf(
    __ubuf__ InputType *gate, __ubuf__ InputType *up, __ubuf__ bfloat16_t *output,
    __ubuf__ float *topkWeights, uint16_t rowLoopCount, uint16_t fullVectorLoopCount,
    uint16_t needTailVectorCompute, uint16_t needAdditionalPaddingStore,
    uint32_t inputRowStrideElements, uint32_t outputRowStrideElements,
    uint32_t tailColumnOffsetElements, uint32_t tailComputeMaskElementCount,
    uint32_t tailStoreMaskElementCount, uint32_t additionalPaddingStoreMaskElementCount,
    float clampLimit)
{
    AscendC::Reg::RegTensor<InputType> gateReg;
    AscendC::Reg::RegTensor<InputType> upReg;
    AscendC::Reg::RegTensor<float> gateFp32Reg;
    AscendC::Reg::RegTensor<float> upFp32Reg;
    AscendC::Reg::RegTensor<float> negReg;
    AscendC::Reg::RegTensor<float> expReg;
    AscendC::Reg::RegTensor<float> denominatorReg;
    AscendC::Reg::RegTensor<float> sigmoidReg;
    AscendC::Reg::RegTensor<float> outputFp32Reg;
    AscendC::Reg::RegTensor<float> weightReg;
    AscendC::Reg::RegTensor<bfloat16_t> outputReg;
    AscendC::Reg::RegTensor<bfloat16_t> zeroReg;

    AscendC::Reg::MaskReg fullMask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    uint32_t tailComputeMaskCount = tailComputeMaskElementCount;
    uint32_t tailStoreMaskCount = tailStoreMaskElementCount;
    uint32_t additionalPaddingStoreMaskCount = additionalPaddingStoreMaskElementCount;
    AscendC::Reg::MaskReg tailComputeMask = AscendC::Reg::UpdateMask<float>(tailComputeMaskCount);
    AscendC::Reg::MaskReg tailStoreMask = AscendC::Reg::UpdateMask<float>(tailStoreMaskCount);
    AscendC::Reg::MaskReg additionalPaddingStoreMask =
        AscendC::Reg::UpdateMask<bfloat16_t>(additionalPaddingStoreMaskCount);

    __ubuf__ InputType *gateTail = gate + tailColumnOffsetElements;
    __ubuf__ InputType *upTail = up + tailColumnOffsetElements;
    __ubuf__ bfloat16_t *outputTail = output + tailColumnOffsetElements;
    __ubuf__ bfloat16_t *additionalPaddingOutput =
        outputTail + needTailVectorCompute * VECTOR_LENGTH_FP32;

    for (uint16_t rowIndex = 0; rowIndex < rowLoopCount; ++rowIndex) {
        if constexpr (FuseTopkWeight) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                weightReg, topkWeights + rowIndex * INT32_PER_256B + WEIGHT_INDEX);
        }

        const uint32_t inputRowOffset = static_cast<uint32_t>(rowIndex) * inputRowStrideElements;
        const uint32_t outputRowOffset = static_cast<uint32_t>(rowIndex) * outputRowStrideElements;
        for (uint16_t vectorIndex = 0; vectorIndex < fullVectorLoopCount; ++vectorIndex) {
            const uint32_t vectorOffset = static_cast<uint32_t>(vectorIndex) * VECTOR_LENGTH_FP32;
            LoadSwiGluInputAsFp32<InputType>(
                gateFp32Reg, gateReg, gate + inputRowOffset + vectorOffset, fullMask);
            LoadSwiGluInputAsFp32<InputType>(
                upFp32Reg, upReg, up + inputRowOffset + vectorOffset, fullMask);
            ComputeSwiGluVector<IsStep>(outputFp32Reg, gateFp32Reg, upFp32Reg, negReg, expReg,
                denominatorReg, sigmoidReg, fullMask, clampLimit);
            if constexpr (FuseTopkWeight) {
                AscendC::Reg::Mul(outputFp32Reg, outputFp32Reg, weightReg, fullMask);
            }
            StoreSwiGluFp32AsBf16(
                output + outputRowOffset + vectorOffset, outputReg, outputFp32Reg, fullMask, fullMask);
        }

        for (uint16_t tailIndex = 0; tailIndex < needTailVectorCompute; ++tailIndex) {
            LoadSwiGluInputAsFp32<InputType>(
                gateFp32Reg, gateReg, gateTail + inputRowOffset, tailComputeMask);
            LoadSwiGluInputAsFp32<InputType>(
                upFp32Reg, upReg, upTail + inputRowOffset, tailComputeMask);
            ComputeSwiGluVector<IsStep>(outputFp32Reg, gateFp32Reg, upFp32Reg, negReg, expReg,
                denominatorReg, sigmoidReg, tailComputeMask, clampLimit);
            if constexpr (FuseTopkWeight) {
                AscendC::Reg::Mul(outputFp32Reg, outputFp32Reg, weightReg, tailComputeMask);
            }
            StoreSwiGluFp32AsBf16(outputTail + outputRowOffset, outputReg, outputFp32Reg,
                tailComputeMask, tailStoreMask);
        }

        for (uint16_t additionalPaddingIndex = 0;
             additionalPaddingIndex < needAdditionalPaddingStore; ++additionalPaddingIndex) {
            AscendC::Reg::Duplicate(zeroReg, static_cast<bfloat16_t>(0));
            AscendC::Reg::StoreAlign<bfloat16_t>(
                additionalPaddingOutput + outputRowOffset, zeroReg, additionalPaddingStoreMask);
        }
    }
}

template <typename InputType, bool FuseTopkWeight, bool IsStep>
__aicore__ inline void RunSwiGLU(const GatedActivationArgs<InputType> &args, const SwiGluParams &params)
{
    const ActivationLoopPlan plan = BuildActivationLoopPlan(args);
    asc_vf_call<SwiGluVf<InputType, FuseTopkWeight, IsStep>>(
        args.gate, args.up, args.output, args.topkWeights,
        static_cast<uint16_t>(plan.validRowCount),
        static_cast<uint16_t>(plan.fullVectorLoopCountPerRow),
        plan.needTailVectorCompute, plan.needAdditionalPaddingStore,
        plan.inputRowStrideElements, plan.outputRowStrideElements,
        plan.tailColumnOffsetElements, plan.tailComputeMaskElementCount,
        plan.tailStoreMaskElementCount, plan.additionalPaddingStoreMaskElementCount,
        params.clampLimit);
}

} // namespace Activation
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_SWIGLU_ACTIVATION_H
