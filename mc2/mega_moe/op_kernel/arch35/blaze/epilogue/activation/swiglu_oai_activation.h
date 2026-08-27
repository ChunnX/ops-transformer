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
 * \file swiglu_oai_activation.h
 * \brief A5（arch35）SwiGLU-OAI 门控激活实现。
 */

#ifndef MEGA_MOE_ARCH35_SWIGLU_OAI_ACTIVATION_H
#define MEGA_MOE_ARCH35_SWIGLU_OAI_ACTIVATION_H

#if defined(__DAV_C310__)
#include "activation_common.h"

namespace MegaMoeImpl {
namespace Activation {

template <typename InputType, bool FuseTopkWeight>
__aicore__ inline void RunSwiGLUOai(const GatedActivationArgs<InputType> &args, const SwiGluOaiParams &params)
{
    const ActivationLoopPlan plan = BuildActivationLoopPlan(args);
    const float scalarOne = 1.0f;
    bfloat16_t zeroValue = 0;

    uint32_t tailComputeMaskElements = 0U;
    uint32_t firstTailStoreMaskElements = 0U;
    uint32_t secondPaddingStoreMaskElements = 0U;
    uint16_t tailLoopCount = 0U;
    uint16_t secondPaddingLoopCount = 0U;
    __ubuf__ InputType *gateTail = args.gate;
    __ubuf__ InputType *upTail = args.up;
    __ubuf__ bfloat16_t *outputTail = args.output;
    __ubuf__ bfloat16_t *secondPaddingOutput = args.output;
    if (plan.tailElementCountPerRow > 0U) {
        tailComputeMaskElements = plan.tailElementCountPerRow;
        tailLoopCount = 1U;
        const uint32_t tailStoreElements =
            plan.tailElementCountPerRow + plan.outputPaddingElementCountPerRow;
        if (tailStoreElements <= VECTOR_LENGTH_FP32) {
            firstTailStoreMaskElements = tailStoreElements;
        } else {
            secondPaddingLoopCount = 1U;
            firstTailStoreMaskElements = VECTOR_LENGTH_FP32;
            secondPaddingStoreMaskElements = tailStoreElements - VECTOR_LENGTH_FP32;
        }
        const uint32_t tailColumnOffset = plan.fullVectorLoopCountPerRow * VECTOR_LENGTH_FP32;
        gateTail = args.gate + tailColumnOffset;
        upTail = args.up + tailColumnOffset;
        outputTail = args.output + tailColumnOffset;
        secondPaddingOutput = outputTail + tailLoopCount * VECTOR_LENGTH_FP32;
    }

    const uint16_t rowLoopCount = static_cast<uint16_t>(plan.validRowCount);
    const uint16_t fullVectorLoopCount = static_cast<uint16_t>(plan.fullVectorLoopCountPerRow);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<InputType> gateReg;
        AscendC::MicroAPI::RegTensor<InputType> upReg;
        AscendC::MicroAPI::RegTensor<float> gateFp32Reg;
        AscendC::MicroAPI::RegTensor<float> upFp32Reg;
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> denominatorReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outputFp32Reg;
        AscendC::MicroAPI::RegTensor<float> weightReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outputReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> zeroReg;
        AscendC::MicroAPI::MaskReg fullMask =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg tailComputeMask =
            AscendC::MicroAPI::UpdateMask<float>(tailComputeMaskElements);
        AscendC::MicroAPI::MaskReg firstTailStoreMask =
            AscendC::MicroAPI::UpdateMask<float>(firstTailStoreMaskElements);
        AscendC::MicroAPI::MaskReg secondPaddingStoreMask =
            AscendC::MicroAPI::UpdateMask<bfloat16_t>(secondPaddingStoreMaskElements);

        for (uint16_t rowIndex = 0; rowIndex < rowLoopCount; rowIndex++) {
            if constexpr (FuseTopkWeight) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    weightReg, args.topkWeights + rowIndex * INT32_PER_256B + WEIGHT_INDEX);
            }
            for (uint16_t vectorIndex = 0; vectorIndex < fullVectorLoopCount; vectorIndex++) {
                AscendC::MicroAPI::AddrReg inputOffset = AscendC::MicroAPI::CreateAddrReg<InputType>(
                    rowIndex, plan.inputRowStrideElements, vectorIndex, VECTOR_LENGTH_FP32);
                AscendC::MicroAPI::DataCopy<InputType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    gateReg, args.gate, inputOffset);
                AscendC::MicroAPI::DataCopy<InputType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    upReg, args.up, inputOffset);
                AscendC::MicroAPI::Cast<float, InputType, CAST_INPUT_TO_FP32>(gateFp32Reg, gateReg, fullMask);
                AscendC::MicroAPI::Cast<float, InputType, CAST_INPUT_TO_FP32>(upFp32Reg, upReg, fullMask);

                AscendC::MicroAPI::Mins(gateFp32Reg, gateFp32Reg, params.clampLimit, fullMask);
                AscendC::MicroAPI::Mins(upFp32Reg, upFp32Reg, params.clampLimit, fullMask);
                AscendC::MicroAPI::Maxs(upFp32Reg, upFp32Reg, -params.clampLimit, fullMask);
                AscendC::MicroAPI::Adds(upFp32Reg, upFp32Reg, params.beta, fullMask);
                AscendC::MicroAPI::Muls(negReg, gateFp32Reg, -params.alpha, fullMask);
                AscendC::MicroAPI::Exp(expReg, negReg, fullMask);
                AscendC::MicroAPI::Adds(denominatorReg, expReg, scalarOne, fullMask);
                AscendC::MicroAPI::Div(sigmoidReg, gateFp32Reg, denominatorReg, fullMask);
                AscendC::MicroAPI::Mul(outputFp32Reg, sigmoidReg, upFp32Reg, fullMask);
                if constexpr (FuseTopkWeight) {
                    AscendC::MicroAPI::Mul(outputFp32Reg, outputFp32Reg, weightReg, fullMask);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(outputReg, outputFp32Reg, fullMask);
                AscendC::MicroAPI::AddrReg outputOffset = AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(
                    rowIndex, plan.outputRowStrideElements, vectorIndex, VECTOR_LENGTH_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    args.output, outputReg, outputOffset, fullMask);
            }

            AscendC::MicroAPI::AddrReg tailInputOffset =
                AscendC::MicroAPI::CreateAddrReg<InputType>(rowIndex, plan.inputRowStrideElements);
            AscendC::MicroAPI::AddrReg tailOutputOffset =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(rowIndex, plan.outputRowStrideElements);
            for (uint16_t tailIndex = 0; tailIndex < tailLoopCount; tailIndex++) {
                AscendC::MicroAPI::DataCopy<InputType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    gateReg, gateTail, tailInputOffset);
                AscendC::MicroAPI::DataCopy<InputType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    upReg, upTail, tailInputOffset);
                AscendC::MicroAPI::Cast<float, InputType, CAST_INPUT_TO_FP32>(
                    gateFp32Reg, gateReg, tailComputeMask);
                AscendC::MicroAPI::Cast<float, InputType, CAST_INPUT_TO_FP32>(upFp32Reg, upReg, tailComputeMask);

                AscendC::MicroAPI::Mins(gateFp32Reg, gateFp32Reg, params.clampLimit, tailComputeMask);
                AscendC::MicroAPI::Mins(upFp32Reg, upFp32Reg, params.clampLimit, tailComputeMask);
                AscendC::MicroAPI::Maxs(upFp32Reg, upFp32Reg, -params.clampLimit, tailComputeMask);
                AscendC::MicroAPI::Adds(upFp32Reg, upFp32Reg, params.beta, tailComputeMask);
                AscendC::MicroAPI::Muls(negReg, gateFp32Reg, -params.alpha, tailComputeMask);
                AscendC::MicroAPI::Exp(expReg, negReg, tailComputeMask);
                AscendC::MicroAPI::Adds(denominatorReg, expReg, scalarOne, tailComputeMask);
                AscendC::MicroAPI::Div(sigmoidReg, gateFp32Reg, denominatorReg, tailComputeMask);
                AscendC::MicroAPI::Mul(outputFp32Reg, sigmoidReg, upFp32Reg, tailComputeMask);
                if constexpr (FuseTopkWeight) {
                    AscendC::MicroAPI::Mul(outputFp32Reg, outputFp32Reg, weightReg, tailComputeMask);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(
                    outputReg, outputFp32Reg, tailComputeMask);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    outputTail, outputReg, tailOutputOffset, firstTailStoreMask);
            }
            for (uint16_t paddingIndex = 0; paddingIndex < secondPaddingLoopCount; paddingIndex++) {
                AscendC::MicroAPI::Duplicate(zeroReg, zeroValue);
                AscendC::MicroAPI::DataCopy<bfloat16_t>(
                    secondPaddingOutput, zeroReg, tailOutputOffset, secondPaddingStoreMask);
            }
        }
    }
}

} // namespace Activation
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_SWIGLU_OAI_ACTIVATION_H
