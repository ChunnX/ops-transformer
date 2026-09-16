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
 * \file vf_mx_finalize_output.h
 * \brief Fuse MX invalid-row handling and FP32-to-BF16 output conversion.
 */
#ifndef VF_MX_FINALIZE_OUTPUT_H
#define VF_MX_FINALIZE_OUTPUT_H

#include "kernel_tensor.h"
#include "vf_basic_block_utils.h"

namespace FaVectorApi {
template <typename OUTPUT_T, uint16_t SRC_D, bool HAS_ATTEN>
__simd_vf__ inline void MxFinalizeOutputVF(__ubuf__ OUTPUT_T *dstUb, __ubuf__ float *srcUb, __ubuf__ float *maxUb,
                                           const uint16_t m)
{
    static_assert(SRC_D == 128U, "MX output finalize only supports D=128");
    static_assert(IsSameType<OUTPUT_T, bfloat16_t>::value || IsSameType<OUTPUT_T, half>::value,
                  "MX output finalize only supports 16-bit floating-point output");

    RegTensor<float> vregEven;
    RegTensor<float> vregOdd;
    RegTensor<float> vregMax;
    RegTensor<float> vregMin;
    RegTensor<float> vregZero;
    RegTensor<OUTPUT_T> vregEvenOut;
    RegTensor<OUTPUT_T> vregOddOut;
    RegTensor<OUTPUT_T> vregPackedOut;

    MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg pregInvalidRow;

    if constexpr (HAS_ATTEN) {
        constexpr uint32_t minBits = 0xFF7FFFFFU;
        const float minValue = *((float *)&minBits);
        Duplicate(vregMin, minValue);
        Duplicate(vregZero, 0.0F);
    }

    for (uint16_t i = 0U; i < m; ++i) {
        LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(vregEven, vregOdd, srcUb + i * SRC_D);
        if constexpr (HAS_ATTEN) {
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vregMax, maxUb + i);
            Compare<float, CMPMODE::EQ>(pregInvalidRow, vregMax, vregMin, pregAll);
            Select(vregEven, vregZero, vregEven, pregInvalidRow);
            Select(vregOdd, vregZero, vregOdd, pregInvalidRow);
        }

        Cast<OUTPUT_T, float, castTraitRoundNoSatZero>(vregEvenOut, vregEven, pregAll);
        Cast<OUTPUT_T, float, castTraitRoundNoSatOne>(vregOddOut, vregOdd, pregAll);
        Or((RegTensor<uint16_t> &)vregPackedOut, (RegTensor<uint16_t> &)vregEvenOut, (RegTensor<uint16_t> &)vregOddOut,
           pregAllB16);
        StoreAlign<OUTPUT_T, Reg::StoreDist::DIST_NORM_B32>(dstUb + i * SRC_D, vregPackedOut, pregAllB16);
    }
}

template <typename OUTPUT_T, uint16_t SRC_D, bool HAS_ATTEN>
__aicore__ inline void MxFinalizeOutput(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<float> &srcTensor,
                                        const LocalTensor<float> &maxTensor, const uint16_t m)
{
    __ubuf__ OUTPUT_T *dstUb = (__ubuf__ OUTPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ float *srcUb = (__ubuf__ float *)srcTensor.GetPhyAddr();
    __ubuf__ float *maxUb = (__ubuf__ float *)maxTensor.GetPhyAddr();
    MxFinalizeOutputVF<OUTPUT_T, SRC_D, HAS_ATTEN>(dstUb, srcUb, maxUb, m);
}
} // namespace FaVectorApi

#endif // VF_MX_FINALIZE_OUTPUT_H
