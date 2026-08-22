/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file vf_rescale_o_dn_mxfp4.h
 * \brief
 */

#ifndef VF_COMPUTE_LSE_H_
#define VF_COMPUTE_LSE_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
namespace NpuArch::Epilogue::Block::Mxfp4VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

// When qS can be copied from UB to GM continuously,
// qS would be stored from reg to UB continuously.
// BNSD
__simd_vf__ inline void compute_lse_qs_axis_continuous(
    __ubuf__ float *gmUb, __ubuf__ float *glUb, __ubuf__ float *lseUb,
    uint32_t row)
{
    // This vf works only when the rowNum in each AIV does not exceed 64
    RegTensor<float> gmVreg;
    RegTensor<float> glVreg;
    RegTensor<float> logGlVreg;
    RegTensor<float> lseVreg;
    MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregTail = UpdateMask<float>(row);

    LoadAlign<float, LoadDist::DIST_NORM>(glVreg, glUb);
    LoadAlign<float, LoadDist::DIST_NORM>(gmVreg, gmUb);
    Ln<float, &LN_MODE>(logGlVreg, glVreg, pregTail);
    Add(lseVreg, logGlVreg, gmVreg, pregTail);
    StoreAlign<float, StoreDist::DIST_NORM_B32>(lseUb, lseVreg, pregTail);
}

// When qS cannot be copied from UB to GM continuously,
// lse would be broadcasted and then stored from reg to UB continuously.
// TND & BSND
__simd_vf__ inline void compute_lse_qs_axis_incontinuous(
    __ubuf__ float *gmUb, __ubuf__ float *glUb, __ubuf__ float *lseUb,
    uint32_t row)
{
    RegTensor<float> gmRowwiseVreg0;
    RegTensor<float> gmRowwiseVreg1;
    RegTensor<float> glRowwiseVreg0;
    RegTensor<float> glRowwiseVreg1;
    RegTensor<float> logGlRowwiseVreg0;
    RegTensor<float> logGlRowwiseVreg1;
    RegTensor<float> lseRowwiseVreg0;
    RegTensor<float> lseRowwiseVreg1;
    UnalignReg rowwiseUreg0;
    UnalignReg rowwiseUreg1;
    MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
    constexpr uint32_t postUpdateStride = 32 / sizeof(float);

    for (uint16_t i = 0; i < row; i+=2) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(glRowwiseVreg0, glUb + i);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(glRowwiseVreg1, glUb + (i + 1));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(gmRowwiseVreg0, gmUb + i);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(gmRowwiseVreg1, gmUb + (i + 1));
        Ln<float, &LN_MODE>(logGlRowwiseVreg0, glRowwiseVreg0, pregFull);
        Ln<float, &LN_MODE>(logGlRowwiseVreg1, glRowwiseVreg1, pregFull);
        Add(lseRowwiseVreg0, logGlRowwiseVreg0, gmRowwiseVreg0, pregFull);
        Add(lseRowwiseVreg1, logGlRowwiseVreg1, gmRowwiseVreg1, pregFull);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(lseUb, lseRowwiseVreg0, rowwiseUreg0, postUpdateStride);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(lseUb, lseRowwiseVreg1, rowwiseUreg1, postUpdateStride);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(lseUb, rowwiseUreg0, postUpdateStride);
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(lseUb, rowwiseUreg1, postUpdateStride);
}

__aicore__ inline void ComputeLseContinuousCallVF(const LocalTensor<float> &gmTensor,
                                                            const LocalTensor<float> &glTensor,
                                                            const LocalTensor<float> &lseTensor,
                                                            uint32_t row)
{
    __ubuf__ float *gmUb = (__ubuf__ float *)gmTensor.GetPhyAddr();
    __ubuf__ float *glUb = (__ubuf__ float *)glTensor.GetPhyAddr();
    __ubuf__ float *lseUb = (__ubuf__ float *)lseTensor.GetPhyAddr();

    compute_lse_qs_axis_continuous(gmUb, glUb, lseUb, row);
}

__aicore__ inline void ComputeLseIncontinuousCallVF(const LocalTensor<float> &gmTensor,
                                                            const LocalTensor<float> &glTensor,
                                                            const LocalTensor<float> &lseTensor,
                                                            uint32_t row)
{
    __ubuf__ float *gmUb = (__ubuf__ float *)gmTensor.GetPhyAddr();
    __ubuf__ float *glUb = (__ubuf__ float *)glTensor.GetPhyAddr();
    __ubuf__ float *lseUb = (__ubuf__ float *)lseTensor.GetPhyAddr();

    compute_lse_qs_axis_incontinuous(gmUb, glUb, lseUb, row);
}

} // namespace NpuArch::Epilogue::Block::Mxfp4VF
#endif // VF_COMPUTE_LSE_H_
