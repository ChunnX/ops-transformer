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
 * \file gmmsq_mx_a8w4_vf.h
 * \brief VF functions for GMMSQ MxA8W4: anti-quantization, SwiGLU, and MX quantization.
 */
#ifndef GMMSQ_MX_A8W4_VF_H
#define GMMSQ_MX_A8W4_VF_H

#include "../common/basic_block_config.h"
#include "../common/basic_block_vf_mx.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

namespace MicroAPI = AscendC::MicroAPI;
using AscendC::BLOCK_CUBE;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

namespace WeightQuantBatchMatmulV2::Arch35 {

__simd_vf__ inline void InitZeroVf(__ubuf__ float *ubAddr, uint16_t loopCount)
{
    MicroAPI::RegTensor<float> zeroVreg;
    MicroAPI::Duplicate(zeroVreg, 0);
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t loopIdx = 0; loopIdx < loopCount; loopIdx++) {
        MicroAPI::AddrReg outAddrReg = MicroAPI::CreateAddrReg<float>(loopIdx, QUADRUPLE_BUFFER_NUM * VEC_MAX_ELEM_B32);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(ubAddr, zeroVreg, outAddrReg, preg);
    }
}

static constexpr MicroAPI::CastTrait CAST_B16_TO_B32_TRAIT = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

static constexpr MicroAPI::CastTrait CAST_FP32_TO_FP8_TRAIT = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

static constexpr MicroAPI::CastTrait CAST_FP16_TO_FP8_TRAIT = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

static constexpr MicroAPI::CastTrait CAST_U32_TO_FP16_TRAIT = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC};

template <typename biasType>
__simd_callee__ inline void GmmsqMxA8W4BiasCompute(uint16_t biasLoopNum, __ubuf__ biasType *biasInUbAddr,
                                                   __ubuf__ biasType *biasOutUbAddr)
{
    static constexpr biasType MX_BIAS_FACTOR = static_cast<biasType>(0.015625f);
    MicroAPI::RegTensor<biasType> biasVreg, biasFactorVreg;
    MicroAPI::MaskReg maskBiasAll = MicroAPI::CreateMask<biasType, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate<biasType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(biasFactorVreg, MX_BIAS_FACTOR,
                                                                             maskBiasAll);
    for (uint16_t loopBiasIdx = 0; loopBiasIdx < biasLoopNum; ++loopBiasIdx) {
        MicroAPI::AddrReg biasAreg = MicroAPI::CreateAddrReg<biasType>(loopBiasIdx, VEC_MAX_ELEM_B16);
        MicroAPI::LoadAlign<biasType, MicroAPI::LoadDist::DIST_NORM>(biasVreg, biasInUbAddr, biasAreg);
        MicroAPI::Mul<biasType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(biasVreg, biasVreg, biasFactorVreg,
                                                                           maskBiasAll);
        MicroAPI::StoreAlign<biasType, MicroAPI::StoreDist::DIST_NORM_B16>(biasOutUbAddr, biasVreg, biasAreg,
                                                                           maskBiasAll);
    }
}

template <typename xType, typename wType, typename biasType, bool hasBias>
__simd_vf__ inline void GmmsqFrAntiQuantMxA8W4NzNkVf(MxA8W4NzParams<xType, wType, biasType> mxA8W4NzParams)
{
    if constexpr (hasBias) {
        GmmsqMxA8W4BiasCompute<biasType>(mxA8W4NzParams.biasLoopNum, mxA8W4NzParams.biasInUbAddr,
                                         mxA8W4NzParams.biasOutUbAddr);
    }
    MicroAPI::RegTensor<int8_t> wShrReg, wShlReg, wAndReg, wLoad, wShl, wShr0, wShr1, wSel, wAnd;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregVsel = MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wShrReg, E2M1_SHIFT_RIGHT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wAndReg, E2M1_AND_MASK, preg);

    for (uint16_t loopKIdx = 0; loopKIdx < mxA8W4NzParams.loopKNum; ++loopKIdx) {
        for (uint16_t innerLoopIdx = 0; innerLoopIdx < mxA8W4NzParams.innerLoopNum; ++innerLoopIdx) {
            MicroAPI::AddrReg aregWeightB8In = MicroAPI::CreateAddrReg<uint8_t>(
                loopKIdx, (C0_SIZE_B8 * mxA8W4NzParams.nRealSizeAlign) >> 1, innerLoopIdx, VECTOR_REG_WIDTH >> 1);
            MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_US_B8>(
                (MicroAPI::RegTensor<uint8_t> &)wLoad, (__ubuf__ uint8_t *&)mxA8W4NzParams.weightLowBitPhyAddr,
                aregWeightB8In);

            MicroAPI::ShiftRight(wShr0, wLoad, wShrReg, preg);
            MicroAPI::ShiftLeft(wShl, wLoad, wShlReg, preg);
            MicroAPI::ShiftRight(wShr1, wShl, wShrReg, preg);
            MicroAPI::Select(wSel, wShr1, wShr0, pregVsel);
            MicroAPI::And(wAnd, wSel, wAndReg, preg);

            MicroAPI::AddrReg aregWeightB8Out = MicroAPI::CreateAddrReg<uint8_t>(
                loopKIdx, mxA8W4NzParams.loopKDstStride, innerLoopIdx, mxA8W4NzParams.innerDstStride);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(
                (__ubuf__ uint8_t *&)mxA8W4NzParams.weightHighBitPhyAddr, (MicroAPI::RegTensor<uint8_t> &)wAnd,
                aregWeightB8Out, preg);
        }
    }
}

__simd_vf__ inline void GmmsqSwiGLUVf(uint16_t mSize, uint16_t nHalfSize, __ubuf__ float *yF32Addr,
                                      __ubuf__ float *gateF32Addr)
{
    (void)mSize;
    (void)nHalfSize;
    (void)yF32Addr;
    (void)gateF32Addr;
}

__simd_vf__ inline void GmmsqComputeMaxExpVf(uint16_t mSize, uint16_t nGroupCount, __ubuf__ float *yF32Addr,
                                             __ubuf__ half *maxB16Addr)
{
    (void)mSize;
    (void)nGroupCount;
    (void)yF32Addr;
    (void)maxB16Addr;
}

__simd_vf__ inline void GmmsqComputeScaleVf(uint16_t mSize, uint16_t nGroupCount, __ubuf__ half *maxB16Addr,
                                            __ubuf__ fp8_e8m0_t *scaleE8M0Addr)
{
    (void)mSize;
    (void)nGroupCount;
    (void)maxB16Addr;
    (void)scaleE8M0Addr;
}

__simd_vf__ inline void GmmsqComputeDataForQuantTargetFp8Vf(uint16_t mSize, uint16_t nHalfSize,
                                                            __ubuf__ float *yF32Addr,
                                                            __ubuf__ fp8_e8m0_t *scaleE8M0Addr,
                                                            __ubuf__ fp8_e4m3fn_t *yFp8Addr)
{
    (void)mSize;
    (void)nHalfSize;
    (void)yF32Addr;
    (void)scaleE8M0Addr;
    (void)yFp8Addr;
}

} // namespace WeightQuantBatchMatmulV2::Arch35
#endif // GMMSQ_MX_A8W4_VF_H
