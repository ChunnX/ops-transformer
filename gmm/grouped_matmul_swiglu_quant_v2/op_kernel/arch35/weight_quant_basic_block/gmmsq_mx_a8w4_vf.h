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

template <typename biasType>
__simd_callee__ inline void FrMxA8W4BiasCompute(uint16_t biasLoopNum, __ubuf__ biasType *biasInUbAddr,
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
__simd_vf__ inline void FrAntiQuantMxA8W4NzNkVf(MxA8W4NzParams<xType, wType, biasType> mxA8W4NzParams)
{
    if constexpr (hasBias) {
        FrMxA8W4BiasCompute<biasType>(mxA8W4NzParams.biasLoopNum, mxA8W4NzParams.biasInUbAddr,
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

__simd_vf__ inline void SwiGLUVf(uint16_t mSize, uint16_t nHalfSize, __ubuf__ float *yF32Addr)
{
    MicroAPI::RegTensor<float> swishVreg, gateVreg, negSwishVreg, expVreg, oneVreg, denomVreg, recipVreg, sigmoidVreg,
        resultVreg;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    MicroAPI::Duplicate(oneVreg, 1.0f, maskAll);

    uint64_t rowStride = nHalfSize * 2;
    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        uint64_t nLoopCnt = nHalfSize / VEC_MAX_ELEM_B32;
        for (uint16_t nLoopIdx = 0; nLoopIdx < nLoopCnt; ++nLoopIdx) {
            __ubuf__ float *swishAddr = yF32Addr + mIdx * rowStride + nLoopIdx * VEC_MAX_ELEM_B32;
            __ubuf__ float *gateAddr = swishAddr + nHalfSize;

            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(swishVreg, swishAddr);
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(gateVreg, gateAddr);

            MicroAPI::Muls(negSwishVreg, swishVreg, -1.0f, maskAll);

            MicroAPI::Exp(expVreg, negSwishVreg, maskAll);

            MicroAPI::Add(denomVreg, oneVreg, expVreg, maskAll);

            MicroAPI::Div(recipVreg, oneVreg, denomVreg, maskAll);

            MicroAPI::Mul(sigmoidVreg, swishVreg, recipVreg, maskAll);

            MicroAPI::Mul(resultVreg, sigmoidVreg, gateVreg, maskAll);

            MicroAPI::AddrReg outAddrReg = MicroAPI::CreateAddrReg<float>(mIdx, rowStride, nLoopIdx, VEC_MAX_ELEM_B32);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(yF32Addr, resultVreg, outAddrReg, maskAll);
        }
    }
}

__simd_vf__ inline void ComputeMaxExpVf(uint16_t mSize, uint16_t nGroupCount, __ubuf__ float *yF32Addr,
                                        __ubuf__ half *maxB16Addr)
{
    MicroAPI::RegTensor<float> dataVreg, absVreg;
    MicroAPI::RegTensor<uint32_t> bitsVreg, expBitsVreg, maxExpVreg;
    MicroAPI::RegTensor<half> maxHalfVreg;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskU32All = MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

    static constexpr uint32_t EXP_MASK = 0x7F800000;
    static constexpr uint32_t EXP_SHIFT = 23;

    MicroAPI::Duplicate<uint32_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(expBitsVreg, EXP_MASK, maskU32All);
    MicroAPI::Duplicate<uint32_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(maxExpVreg, 0, maskU32All);

    uint64_t rowFloatStride = nGroupCount * MX_GROUPSIZE;
    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        __ubuf__ float *rowAddr = yF32Addr + mIdx * rowFloatStride;
        __ubuf__ half *outAddr = maxB16Addr + mIdx * nGroupCount;

        for (uint16_t groupIdx = 0; groupIdx < nGroupCount; ++groupIdx) {
            __ubuf__ float *groupAddr = rowAddr + groupIdx * MX_GROUPSIZE;

            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(dataVreg, groupAddr);

            MicroAPI::Abs(absVreg, dataVreg, maskAll);

            MicroAPI::Cast<uint32_t, float, CAST_B16_TO_B32_TRAIT>(bitsVreg, (MicroAPI::RegTensor<float> &)absVreg, maskU32All);

            MicroAPI::And(expBitsVreg, bitsVreg, expBitsVreg, maskU32All);

            MicroAPI::ShiftRight(maxExpVreg, expBitsVreg, EXP_SHIFT, maskU32All);

            MicroAPI::Cast<half, uint32_t, CAST_B16_TO_B32_TRAIT>(maxHalfVreg, maxExpVreg, maskAll);

            MicroAPI::AddrReg outAddrReg = MicroAPI::CreateAddrReg<half>(mIdx, nGroupCount, groupIdx, 1);
            MicroAPI::StoreAlign<half, MicroAPI::StoreDist::DIST_NORM_B16>(outAddr, maxHalfVreg, outAddrReg, maskAll);
        }
    }
}

__simd_vf__ inline void ComputeScaleVf(uint16_t mSize, uint16_t nGroupCount, __ubuf__ half *maxB16Addr,
                                       __ubuf__ fp8_e8m0_t *scaleE8M0Addr)
{
    MicroAPI::RegTensor<half> maxHalfVreg, scaleHalfVreg, emaxVreg;
    MicroAPI::RegTensor<fp8_e8m0_t> scaleE8M0Vreg;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<half, AscendC::MicroAPI::MaskPattern::ALL>();

    static constexpr half FP8_E4M3FN_EMAX = static_cast<half>(8.0f);
    MicroAPI::Duplicate<half, AscendC::MicroAPI::MaskMergeMode::ZEROING>(emaxVreg, FP8_E4M3FN_EMAX, maskAll);

    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        __ubuf__ half *rowMaxAddr = maxB16Addr + mIdx * nGroupCount;
        __ubuf__ fp8_e8m0_t *rowScaleAddr = scaleE8M0Addr + mIdx * nGroupCount;

        uint64_t loopCnt = nGroupCount / VEC_MAX_ELEM_B16;
        for (uint16_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
            MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_NORM>(maxHalfVreg, rowMaxAddr + loopIdx * VEC_MAX_ELEM_B16);

            MicroAPI::Sub(scaleHalfVreg, maxHalfVreg, emaxVreg, maskAll);

            MicroAPI::Cast<fp8_e8m0_t, half, CAST_B16_TO_B32_TRAIT>(scaleE8M0Vreg, scaleHalfVreg, maskAll);

            MicroAPI::AddrReg outAddrReg =
                MicroAPI::CreateAddrReg<fp8_e8m0_t>(mIdx, nGroupCount, loopIdx, VEC_MAX_ELEM_B16);
            MicroAPI::StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM>(
                rowScaleAddr, scaleE8M0Vreg, outAddrReg, maskAll);
        }
    }
}

__simd_vf__ inline void ComputeDataForQuantTargetFp8Vf(uint16_t mSize, uint16_t nHalfSize, __ubuf__ float *yF32Addr,
                                                       __ubuf__ fp8_e8m0_t *scaleE8M0Addr,
                                                       __ubuf__ fp8_e4m3fn_t *yFp8Addr)
{
    MicroAPI::RegTensor<float> dataVreg, scaleF32Vreg, scaledVreg;
    MicroAPI::RegTensor<fp8_e4m3fn_t> fp8Vreg;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    static constexpr uint16_t MX_GROUPSIZE_U16 = static_cast<uint16_t>(MX_GROUPSIZE);
    uint16_t nGroupCount = (nHalfSize + MX_GROUPSIZE - 1) / MX_GROUPSIZE;

    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        __ubuf__ float *rowDataAddr = yF32Addr + mIdx * nHalfSize;
        __ubuf__ fp8_e8m0_t *rowScaleAddr = scaleE8M0Addr + mIdx * nGroupCount;
        __ubuf__ fp8_e4m3fn_t *rowFp8Addr = yFp8Addr + mIdx * nHalfSize;

        for (uint16_t groupIdx = 0; groupIdx < nGroupCount; ++groupIdx) {
            MicroAPI::LoadAlign<fp8_e8m0_t, MicroAPI::LoadDist::DIST_BRC_B8>(
                (MicroAPI::RegTensor<fp8_e8m0_t> &)scaleF32Vreg, rowScaleAddr + groupIdx);

            __ubuf__ float *groupDataAddr = rowDataAddr + groupIdx * MX_GROUPSIZE;
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(dataVreg, groupDataAddr);

            MicroAPI::Div(scaledVreg, dataVreg, scaleF32Vreg, maskAll);

            MicroAPI::Cast<fp8_e4m3fn_t, float, CAST_B16_TO_B32_TRAIT>(fp8Vreg, scaledVreg, maskAll);

            MicroAPI::AddrReg outAddrReg =
                MicroAPI::CreateAddrReg<fp8_e4m3fn_t>(mIdx, nHalfSize, groupIdx, MX_GROUPSIZE_U16);
            MicroAPI::StoreAlign<fp8_e4m3fn_t, MicroAPI::StoreDist::DIST_NORM>(rowFp8Addr, fp8Vreg, outAddrReg,
                                                                                  maskAll);
        }
    }
}

} // namespace WeightQuantBatchMatmulV2::Arch35
#endif // GMMSQ_MX_A8W4_VF_H
