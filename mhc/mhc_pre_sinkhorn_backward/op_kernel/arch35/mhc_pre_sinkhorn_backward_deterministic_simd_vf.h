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
    int64_t c = 0;
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

template <typename T>
__aicore__ inline void ComputeGradNormOutAndGradHc(
    const LocalTensor<T>& srcTensor, const GlobalTensor<T>& GradNormalOut, const GlobalTensor<T>& GradHc, T alpha, T invRms, 
                                uint32_t dataCount, uint32_t offset)
{
    Muls(srcTensor, srcTensor, alpha, dataCount);
    DataCopyExtParams copyParams = {static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(GradNormalOut[offset], srcTensor, copyParams);
    Muls(srcTensor, srcTensor, invRms, dataCount);
    DataCopyPad(GradNormalOut[offset], srcTensor, copyParams);
}

template <typename T>
__aicore__ inline void ComputeGradInvRms(const LocalTensor<T>& gradNormOutLocal, const LocalTensor<T>& hcBeforeNormLocal,
    int64_t count, T& gradInvRms)
{
    Mul(hcBeforeNormLocal, gradNormOutLocal, hcBeforeNormLocal, count);
    ReduceSum<T>(hcBeforeNormLocal, hcBeforeNormLocal, hcBeforeNormLocal, count);
    gradInvRms = hcBeforeNormLocal(0);
}

template <typename T>
__aicore__ inline void ComputeGradXFromRms(const LocalTensor<T>& xFp32Local, T gradInvRms, T invRms, int64_t count)
{
    __local_mem__ T* xFp32Addr = (__local_mem__ T*)xFp32Local.GetPhyAddr();
    uint32_t repeatCount = platform::GetVRegSize() / sizeof(T);
    uint16_t repeatTimes = ops::CeilDiv(static_cast<uint32_t>(count), repeatCount);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> xFp32Reg;
        MicroAPI::RegTensor<T> ncCountReg;
        MicroAPI::MaskReg mask;

        MicroAPI::Duplicate(ncCountReg, static_cast<T>(count));
        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(count);
            MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(i, repeatCount);
            MicroAPI::DataCopy(xFp32Reg, (__local_mem__ T*)xFp32Addr, offset);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, static_cast<T>(-1), mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, gradInvRms, mask);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, invRms, mask);
            MicroAPI::Div(xFp32Reg, ncCountReg, invRms, mask);
            MicroAPI::DataCopy(xFp32Addr, xFp32Reg, mask);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void ComputeGradXFinal(const LocalTensor<T>& xFp32Local, const LocalTensor<T>& gradXFromHinLocal,
    const LocalTensor<T>& gradXFromMatMulLocal, int64_t count)
{
    Mul(xFp32Local, xFp32Local, gradXFromHinLocal, count);
    Mul(xFp32Local, xFp32Local, gradXFromMatMulLocal, count);
    Cast(xFp32Local.ReinterpretCast<U>(), xFp32Local, RoundMode::CAST_RINT, count);
    xFp32Que_.EnQue(xFp32Local);
}

#endif  //  MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_SIMD_VF_H