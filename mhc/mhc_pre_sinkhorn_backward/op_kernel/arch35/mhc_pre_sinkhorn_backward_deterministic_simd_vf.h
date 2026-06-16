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

using namespace AscendC;

constexpr uint64_t DOUBLE_BUFFER = 2;

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
        uint32_t maskLen = static_cast<uint32_t>(bsLen * n);
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
            AscendC::MicroAPI::DataCopy(gradZAddr, gradZReg, gradZOfst, maskReg);
        }
    }    
}

template<typename U>
__aicore__ inline void ComputeGradBias(__local_mem__ U* gradBiasAddr, __local_mem__ U* gradZAddr,
                                        uint64_t bsLen, uint64_t colLen, uint64_t vRegSize) 
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> gradZReg;
        AscendC::MicroAPI::RegTensor<U> gradBiasReg;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::MaskReg allMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t maskLen = static_cast<uint32_t>(colLen);
        for (uint16_t i = 0; i < loopCnt; i++) {
            auto gradBiasStart = gradBiasAddr + i * vfLen;
            maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
            MicroAPI::Duplicate(gradBiasReg, U(0), allMaskReg);
            for (uint16_t j = 0; j < bsLen; j++) {
                auto gradZAddrStart = gradZAddr + i * colLen;
                AscendC::MicroAPI::LoadUnAlignPre(u0, gradZAddrStart);
                AscendC::MicroAPI::LoadUnAlign(gradZReg, u0, gradZAddrStart);
                AscendC::MicroAPI::Add(gradBiasReg, gradBiasReg, gradZReg, maskReg);
            }
            AscendC::MicroAPI::DataCopy(gradBiasStart, gradBiasReg, maskReg);
        }
    }
}

template<typename U>
__aicore__ inline void ComputeGradAlpha(__local_mem__ U* normOutForwardPartAddr, __local_mem__ U* gradZAddr, __local_mem__ U* gradAlphaAddr, 
                                                                            uint64_t bsLen, uint64_t colLen, uint64_t vRegSize) 
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
        AscendC::MicroAPI::MaskReg OneMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::VL1>();
        MicroAPI::Duplicate(gradAlphaSumReg, U(0), OneMaskReg);
        uint32_t maskLen = static_cast<uint32_t>(bsLen * colLen);
        for (uint16_t i = 0; i < loopCnt; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
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

template<typename U>
__aicore__ inline void ComputeZAndForwardPart(__local_mem__ U* zAddr, __local_mem__ U* hcBeformNormAddr, __local_mem__ U* invRmsAddr, 
                                            __local_mem__ U* normOutForwardAddr, __local_mem__ U* biasAddr, uint64_t bsLen, uint64_t colLen, 
                                            uint64_t cLenAlign, uint64_t vRegSize, const U& alpha)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> zReg;
        AscendC::MicroAPI::RegTensor<U> hcBeformNormReg;
        AscendC::MicroAPI::RegTensor<U> invRmsReg;
        AscendC::MicroAPI::RegTensor<U> normOutForwardReg;
        AscendC::MicroAPI::RegTensor<U> biasReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        AscendC::MicroAPI::UnalignRegForStore unStoreNormOutForwardReg;
        AscendC::MicroAPI::UnalignRegForStore unStoreZReg;
        uint32_t maskLen = static_cast<uint32_t>(colLen);
        for (uint16_t i = 0; i < loopCnt; i++) {
            AscendC::MicroAPI::UnalignReg u0;
            valueMaskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
            auto biasAddrStart = biasAddr + i * vfLen;
            AscendC::MicroAPI::LoadUnAlignPre(u0, biasAddrStart);
            AscendC::MicroAPI::LoadUnAlign(biasReg, u0, biasAddrStart);
            for (uint16_t j = 0; j < bsLen; i++) {
                AscendC::MicroAPI::UnalignReg u1;
                auto hcBeformNormAddrStart = hcBeformNormAddr + j * cLenAlign;
                auto normOutForwardAddrStart = normOutForwardAddr + j * colLen + i * vfLen;
                auto zAddrStart = zAddr + j * colLen + i * vfLen;
                U invRms = *(invRmsAddr + j);
                AscendC::MicroAPI::AddrReg hcBeformNormOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
                AscendC::MicroAPI::DataCopy(hcBeformNormReg, hcBeformNormAddrStart, hcBeformNormOfst);           
                AscendC::MicroAPI::Muls(normOutForwardReg, hcBeformNormReg, invRms, valueMaskReg);
                AscendC::MicroAPI::Muls(zReg, normOutForwardReg, invRms, valueMaskReg);
                AscendC::MicroAPI::Add(zReg, zReg, biasReg, valueMaskReg);

                MicroAPI::StoreUnAlign<U>(normOutForwardAddrStart, normOutForwardReg, unStoreNormOutForwardReg, vfLen);
                MicroAPI::StoreUnAlignPost<U>(normOutForwardAddrStart, unStoreNormOutForwardReg, 0);
                MicroAPI::StoreUnAlign<U>(zAddrStart, zReg, unStoreZReg, vfLen);
                MicroAPI::StoreUnAlignPost<U>(zAddrStart, unStoreZReg, 0);
            }
        }
    }
}

#endif  //  MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_SIMD_VF_H