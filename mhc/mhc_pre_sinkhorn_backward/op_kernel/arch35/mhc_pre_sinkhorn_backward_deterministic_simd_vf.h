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

constexpr static AscendC::MicroAPI::CastTrait castTrait16ToFloat = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::MicroAPI::CastTrait castTraitFloatTo16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};


template <typename PARAM_T>
__aicore__ inline void CopyIn(const LocalTensor<PARAM_T>& dstTensor, const GlobalTensor<PARAM_T>& srcTensor, int64_t dataLen, 
                                uint32_t ubStride, uint32_t gmStride)
{
    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(gmStride), static_cast<uint32_t>(ubStride)};
    DataCopyPadExtParams<PARAM_T> padParams = {
        false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<PARAM_T>(0)};
    DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
}

template <typename PARAM_T>
__aicore__ inline void CopyOut(const GlobalTensor<PARAM_T>& dstTensor, const LocalTensor<PARAM_T>& srcTensor, int64_t dataLen, 
                                uint32_t ubStride, uint32_t gmStride)
{
    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(gmStride), static_cast<uint32_t>(ubStride)};
    DataCopyPad(dstTensor, srcTensor, copyParams);
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void ComputeProcess1 (LocalTensor<X_T>& xLocal, LocalTensor<X_T>& xCastLocal, LocalTensor<GRADHIN_T>& gradHInLocal, LocalTensor<U>& hPreLocal,
    LocalTensor<U>& gradHPreLocal, LocalTensor<U>& gradXFromHinLocal, LocalTensor<float32>& gradHPostLocal, int64_t bsLen, int64_t cLen, int64_t n)
{
    __local_mem__ X_T* xAddr = (__ubuf__ X_T*)xLocal.GetPhyAddr();
    __local_mem__ U* xCastAddr = (__ubuf__ U*)xCastLocal.GetPhyAddr();
    __local_mem__ GRADHIN_T* gradHInAddr = (__ubuf__ GRADHIN_T*)gradHInLocal.GetPhyAddr();
    __local_mem__ U* gradHPreAddr = (__ubuf__ U*)gradHPreLocal.GetPhyAddr();
    __local_mem__ U* hPreAddr = (__ubuf__ U*)hPreLocal.GetPhyAddr();
    __local_mem__ U* gradXFromHinAddr = (__ubuf__ U*)gradXFromHinLocal.GetPhyAddr();

    uint32_t vfLen = V_REG_SIZE / sizeof(U);
    uint16_t loopCnt = (cLen + vfLen - 1) / vfLen;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<X_T> xReg;
        AscendC::MicroAPI::RegTensor<U> xCastReg;
        AscendC::MicroAPI::RegTensor<GRADHIN_T> gradHInReg;
        AscendC::MicroAPI::RegTensor<U> gradHInCastReg;
        AscendC::MicroAPI::RegTensor<U> gradHPreReg;
        AscendC::MicroAPI::RegTensor<U> gradXFromHinReg;
        uint32_t maskLen = static_cast<uint32_t>(cLen);
        for (uint16_t i = 0; i < bsLen; i++) {
            auto gradHInStart = gradHInAddr + i * cLen;
            for (uint16_t j = 0; j < loopCnt; j++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                AscendC::MicroAPI::AddrReg gradHInAddrOfst = AscendC::MicroAPI::CreateAddrReg<GRADHIN_T>(j, vfLen);
                AscendC::MicroAPI::DataCopy<GRADHIN_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(gradHInReg, gradHInStart, gradHInAddrOfst);
                AscendC::MicroAPI::Cast<U, GRADHIN_T, castTrait16ToFloat>(gradHInCastReg, gradHInReg, valueMaskReg);
                for (uint16_t k = 0; k < n; k++) {
                    U hPre = *(hPreAddr + i * n + k);
                    auto xCastStart = xCastAddr + i * k *cLen;
                    auto xAddrStart = xAddr + i * k * cLen;
                    AscendC::MicroAPI::AddrReg xAddrOfst = AscendC::MicroAPI::CreateAddrReg<X_T>(j, vfLen);
                    AscendC::MicroAPI::AddrReg xCastAddrOfst = AscendC::MicroAPI::CreateAddrReg<U>(j, vfLen);
                    AscendC::MicroAPI::DataCopy<X_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xReg, xAddrStart, xAddrOfst);
                    AscendC::MicroAPI::Cast<U, X_T, castTrait16ToFloat>(xCastReg, xReg, valueMaskReg);
                    AscendC::MicroAPI::Mul(gradHPreReg, xCastReg, gradHInCastReg, valueMaskReg);
                    AscendC::MicroAPI::Muls(gradXFromHinReg, gradHPreReg, hPre, valueMaskReg);
                    
                    AscendC::MicroAPI::DataCopy(xCastStart, xCastReg, xCastAddrOfst, valueMaskReg);
                }
            }
        }
    }
}

#endif  //  MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_SIMD_VF_H