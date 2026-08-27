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
 * \file gated_activation.h
 * \brief A5（arch35）BlockEpilogueActivationMxQuant 门控激活实现。
 *
 */

#ifndef MEGA_MOE_ARCH35_GATED_ACTIVATION_H
#define MEGA_MOE_ARCH35_GATED_ACTIVATION_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../common/mega_moe_utils.h"
#include "activation/situ_activation.h"
#include "activation/swiglu_activation.h"

namespace MegaMoeImpl {
namespace ActivationImpl {

constexpr uint32_t VF_LEN_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);
static constexpr AscendC::MicroAPI::CastTrait CAST_ZERO = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP16_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename DataTypeInT>
struct ActivationContext {
    __ubuf__ DataTypeInT *firstSrc = nullptr;
    __ubuf__ DataTypeInT *secondSrc = nullptr;
    __ubuf__ bfloat16_t *gluResAddr = nullptr;
    __ubuf__ DataTypeInT *firstTailAddr = nullptr;
    __ubuf__ DataTypeInT *secondTailAddr = nullptr;
    __ubuf__ bfloat16_t *activationTailAddr1 = nullptr;
    __ubuf__ bfloat16_t *activationTailAddr2 = nullptr;
    __ubuf__ float *weightUbAddr = nullptr;

    uint32_t nSrcUbAligned = 0;
    uint32_t nDstUbAligned = 0;

    uint16_t dim0VfTimes = 0;
    uint16_t dim1VfTimes = 0;
    uint32_t dim1Tail = 0;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;

    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
};

struct SwiGLUOaiParams {
    float clampLimit;
    float alpha;
    float beta;
};

/*
 * RunSwiGLUOai：SwiGLU-OAI 激活计算流程
 *   公式：(up + beta) × min(gate, clamp) × sigmoid(alpha × min(gate, clamp))
 *        = (clip(up,±clamp)+beta) × min(gate,clamp) / (1+exp(-alpha×min(gate,clamp)))
 */
template <typename DataTypeIn, bool TopkWeightsPrefetch, bool IsInterleaved>
__aicore__ inline void RunSwiGLUOai(const ActivationContext<DataTypeIn> &ctx, const SwiGLUOaiParams &swiGluOaiParams)
{
    const float scalarOne = 1.0f;
    bfloat16_t numZero = 0;
    uint32_t mask1Num = ctx.mask1Num;
    uint32_t mask2Num = ctx.mask2Num;
    uint32_t mask3Num = ctx.mask3Num;
    uint16_t dim0VfTimes = ctx.dim0VfTimes;
    uint16_t dim1VfTimes = ctx.dim1VfTimes;
    uint16_t dim1TailTimes = ctx.dim1TailTimes;
    uint16_t dim1Tail2 = ctx.dim1Tail2;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX1;
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<float> weightReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> zeroReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(mask3Num);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            if constexpr (TopkWeightsPrefetch) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    weightReg, ctx.weightUbAddr + dim0vfLoopIdx * INT32_PER_256B + WEIGHT_INDEX);
            }
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(
                    dim0vfLoopIdx, ctx.nSrcUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstSrc, srcIdxOffset);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondSrc, srcIdxOffset);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask);

                // 对门控输入作上限裁剪。
                AscendC::MicroAPI::Mins(vregX1F, vregX1F, swiGluOaiParams.clampLimit, mask);
                // 对上投影输入作对称裁剪。
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, swiGluOaiParams.clampLimit, mask);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -swiGluOaiParams.clampLimit, mask);

                // 对上投影输入加 beta。
                AscendC::MicroAPI::Adds(vregX2F, vregX2F, swiGluOaiParams.beta, mask);

                // 计算 gate * sigmoid(alpha * gate)。
                AscendC::MicroAPI::Muls(negReg, vregX1F, -swiGluOaiParams.alpha, mask); // 计算 -alpha * gate
                AscendC::MicroAPI::Exp(expReg, negReg, mask);                           // 计算 exp(-alpha * gate)
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);              // 计算 exp(-alpha * gate) + 1
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask); // 计算 gate * sigmoid(alpha * gate)
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask); // 计算完整激活结果
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(
                    dim0vfLoopIdx, ctx.nDstUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.gluResAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 =
                AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, ctx.nSrcUbAligned);
            AscendC::MicroAPI::AddrReg outOffset1 =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, ctx.nDstUbAligned);
            for (uint16_t tailIdx = 0; tailIdx < dim1TailTimes; tailIdx++) {
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask1);

                AscendC::MicroAPI::Mins(vregX1F, vregX1F, swiGluOaiParams.clampLimit, mask1);
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, swiGluOaiParams.clampLimit, mask1);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -swiGluOaiParams.clampLimit, mask1);

                // 对上投影输入加 beta。
                AscendC::MicroAPI::Adds(vregX2F, vregX2F, swiGluOaiParams.beta, mask1);

                // 计算 SwiGLU-OAI 尾块。
                AscendC::MicroAPI::Muls(negReg, vregX1F, -swiGluOaiParams.alpha, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask1);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask1);
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask1);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.activationTailAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t padIdx = 0; padIdx < dim1Tail2; padIdx++) {
                AscendC::MicroAPI::Duplicate(zeroReg, numZero);
                AscendC::MicroAPI::DataCopy<bfloat16_t>(ctx.activationTailAddr2, zeroReg, outOffset1, mask3);
            }
        }
    }
}

} // namespace ActivationImpl
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_GATED_ACTIVATION_H
