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
* \file mhc_pre_sinkhorn_backward_deterministic.h
* \brief
*/
#ifndef MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_SIMD_VF_H
#define MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_SIMD_VF_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

struct Process1Info {
    int64_t bsLen = 0;
    int64_t cLen = 0; 
    int64_t n = 0;
    int64_t cLenXTAlign = 0;
    int64_t cLenUAlign = 0;
    int64_t cLenGRADHINTAlign = 0;
    int64_t vRegSize = 0;
};

constexpr static AscendC::MicroAPI::CastTrait castTrait16ToFloat = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::MicroAPI::CastTrait castTraitFloatTo16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};


template <typename PARAM_T>
__aicore__ inline void CopyIn(const LocalTensor<PARAM_T>& dstTensor, const GlobalTensor<PARAM_T>& srcTensor, int64_t repTime, int64_t dataLen, 
                                uint32_t ubStride, uint32_t gmStride)
{
    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(repTime), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(gmStride), static_cast<uint32_t>(ubStride)};
    DataCopyPadExtParams<PARAM_T> padParams = {
        false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<PARAM_T>(0)};
    DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
}

template <typename PARAM_T>
__aicore__ inline void CopyOut(const GlobalTensor<PARAM_T>& dstTensor, const LocalTensor<PARAM_T>& srcTensor, int64_t repTime, int64_t dataLen, 
                                uint32_t ubStride, uint32_t gmStride)
{
    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(repTime), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(gmStride), static_cast<uint32_t>(ubStride)};
    DataCopyPad(dstTensor, srcTensor, copyParams);
}

template<typename U, bool ISPRE>
__aicore__ inline void SigmoidGrad(__local_mem__ U* gradHAddr, __local_mem__ U* zAddr, __local_mem__ U* gradZAddr, 
                                    uint64_t bsLen, uint64_t n, uint64_t vRegSize, float hcEps)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (bsLen * n + vfLen - 1) / vfLen;
    float hcEpsNeg = hcEps * (-1);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> onesReg;
        AscendC::MicroAPI::RegTensor<U> zReg;
        AscendC::MicroAPI::RegTensor<U> gradHReg;
        AscendC::MicroAPI::RegTensor<U> sigmaReg;
        AscendC::MicroAPI::RegTensor<U> dysigmaReg;
        AscendC::MicroAPI::RegTensor<U> dysigma2Reg;
        AscendC::MicroAPI::RegTensor<U> gradZReg;
        AscendC::MicroAPI::MaskReg maskReg;

        for (uint16_t i = 0; i < loopCnt; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
            MicroAPI::Duplicate(onesReg, U(1), maskReg);
            AscendC::MicroAPI::AddrReg zOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::AddrReg gradHOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::AddrReg gradZOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::DataCopy(zReg, zAddr, zOfst);
            AscendC::MicroAPI::DataCopy(gradHReg, gradHAddr, gradHOfst);            
            AscendC::MicroAPI::Neg(zReg, zReg, maskReg);
            AscendC::MicroAPI::Exp(zReg, zReg, maskReg);
            AscendC::MicroAPI::Add(zReg, zReg, onesReg, maskReg);
            AscendC::MicroAPI::Div(sigmaReg, onesReg, zReg, maskReg);
            if constexpr(ISPRE) {
                AscendC::MicroAPI::Adds(sigmaReg, sigmaReg, hcEpsNeg, maskReg);
            } else  {
                AscendC::MicroAPI::Muls(sigmaReg, sigmaReg, U(2),maskReg);
            }
            AscendC::MicroAPI::Mul(dysigmaReg, sigmaReg, gradHReg, maskReg);
            AscendC::MicroAPI::Mul(dysigma2Reg, sigmaReg, dysigmaReg, maskReg);
            AscendC::MicroAPI::Sub(gradZReg, dysigmaReg, dysigma2Reg, maskReg);
            AscendC::MicroAPI::DataCopy(gradZAddr, gradZReg, zPreOfst, maskReg);
        }
    }    
}

template<typename U>
__aicore__ inline void ComputeGradBias(__local_mem__ U* gradBiasAddr, __local_mem__ U* gradZAddr,
                                        uint64_t bsLen, uint64_t colLen) 
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> gradZReg;
        AscendC::MicroAPI::RegTensor<U> gradZSumReg;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg allMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::::MaskPattern::ALL>();
        for (int i = 0; i < loopCnt; i++) {
            auto gradZAddrStart = gradZAddr + i * vfLen;
            maskReg = AscendC::MicroAPI::UpdateMask<U>(bsLen * colLen);
            MicroAPI::Duplicate(gradZPreSumReg, U(0), allMaskReg);
            for (int j = 0; j < bsLen; j++) {
                auto gradBiasStart = gradBiasZAddr + j * colLen;
                AscendC::MicroAPI::AddrReg gradZOfst = AscendC::MicroAPI::CreateAddrReg<X_T>(i, vfLen);
                AscendC::MicroAPI::DataCopy(gradZReg, gradZAddrStart, gradZOfst);
                AscendC::MicroAPI::Add(gradZSumReg, gradZSumReg, gradZReg)
            }
            AscendC::MicroAPI::DataCopy(gradBiasStart, gradZSumReg, gradZOfst);
        }
    }
}

template<typename U>
__aicore__ inline void ComputeGradAlpha(__local_mem__ U* normOutForwardPartAddr, __local_mem__ U* gradZAddr, __local_mem__ U* gradAlphaAddr, 
                                                                            uint64_t bsLen, uint64_t colLen) 
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (bsLen * colLen + vfLen - 1) / vfLen;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> normOutForwardPartReg;
        AscendC::MicroAPI::RegTensor<U> gradZReg;
        AscendC::MicroAPI::RegTensor<U> gradAlphaReg;
        AscendC::MicroAPI::RegTensor<U> gradAlphaLoopSumReg;
        AscendC::MicroAPI::RegTensor<U> gradAlphaSumReg;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg OneMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::::MaskPattern::VL1>();
        MicroAPI::Duplicate(gradAlphaSumReg, U(0), OneMaskReg);
        for (uint16_t i = 0; i < loopCnt; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<U>(bsLen * colLen);
            AscendC::MicroAPI::AddrReg normOutForwardPartOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::AddrReg gradZOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::DataCopy(normOutForwardPartReg, normOutForwardPartAddr, normOutForwardPartOfst);
            AscendC::MicroAPI::DataCopy(gradZReg, gradZAddr, gradZOfst);
            AscendC::MicroAPI::Mul(gradAlphaReg, normOutForwardPartReg, gradZReg, maskReg);
            AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(gradAlphaLoopSumReg, gradAlphaReg, maskReg); //reduce_sum
            AscendC::MicroAPI::Add(gradAlphaSumReg, gradAlphaSumReg, gradAlphaLoopSumReg, OneMaskReg);
        }
        AscendC::MicroAPI::DataCopy(gradAlphaAddr, gradAlphaSumReg, OneMaskReg);
    }
}

#endif  //  MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_SIMD_VF_H