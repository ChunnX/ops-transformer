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
 * \file vf_basic_block_unaligned1024_common.h
 * \brief shared __simd_callee__ helpers for unaligned1024 no_update / update VF kernels
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED1024_COMMON_H
#define VF_BASIC_BLOCK_UNALIGNED1024_COMMON_H

#include "vf_basic_block_utils.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {

__simd_callee__ inline void ComputePseInnerMulAdd16(
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    RegTensor<float> &vreg_pse9, RegTensor<float> &vreg_pse10,
    RegTensor<float> &vreg_pse11, RegTensor<float> &vreg_pse12,
    RegTensor<float> &vreg_pse13, RegTensor<float> &vreg_pse14,
    RegTensor<float> &vreg_pse15, RegTensor<float> &vreg_pse16,
    RegTensor<float> &vreg_alibi1, RegTensor<float> &vreg_alibi2,
    RegTensor<float> &vreg_alibi3, RegTensor<float> &vreg_alibi4,
    RegTensor<float> &vreg_alibi5, RegTensor<float> &vreg_alibi6,
    RegTensor<float> &vreg_alibi7, RegTensor<float> &vreg_alibi8,
    RegTensor<float> &vreg_alibi9, RegTensor<float> &vreg_alibi10,
    RegTensor<float> &vreg_alibi11, RegTensor<float> &vreg_alibi12,
    RegTensor<float> &vreg_alibi13, RegTensor<float> &vreg_alibi14,
    RegTensor<float> &vreg_alibi15, RegTensor<float> &vreg_alibi16,
    const float slopes, MaskReg &preg_all)
{
    Abs(vreg_pse1, vreg_alibi1, preg_all);
    Abs(vreg_pse2, vreg_alibi2, preg_all);
    Abs(vreg_pse3, vreg_alibi3, preg_all);
    Abs(vreg_pse4, vreg_alibi4, preg_all);
    Abs(vreg_pse5, vreg_alibi5, preg_all);
    Abs(vreg_pse6, vreg_alibi6, preg_all);
    Abs(vreg_pse7, vreg_alibi7, preg_all);
    Abs(vreg_pse8, vreg_alibi8, preg_all);
    Abs(vreg_pse9, vreg_alibi9, preg_all);
    Abs(vreg_pse10, vreg_alibi10, preg_all);
    Abs(vreg_pse11, vreg_alibi11, preg_all);
    Abs(vreg_pse12, vreg_alibi12, preg_all);
    Abs(vreg_pse13, vreg_alibi13, preg_all);
    Abs(vreg_pse14, vreg_alibi14, preg_all);
    Abs(vreg_pse15, vreg_alibi15, preg_all);
    Abs(vreg_pse16, vreg_alibi16, preg_all);
    Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
    Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
    Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
    Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
    Muls(vreg_pse5, vreg_pse5, slopes, preg_all);
    Muls(vreg_pse6, vreg_pse6, slopes, preg_all);
    Muls(vreg_pse7, vreg_pse7, slopes, preg_all);
    Muls(vreg_pse8, vreg_pse8, slopes, preg_all);
    Muls(vreg_pse9, vreg_pse9, slopes, preg_all);
    Muls(vreg_pse10, vreg_pse10, slopes, preg_all);
    Muls(vreg_pse11, vreg_pse11, slopes, preg_all);
    Muls(vreg_pse12, vreg_pse12, slopes, preg_all);
    Muls(vreg_pse13, vreg_pse13, slopes, preg_all);
    Muls(vreg_pse14, vreg_pse14, slopes, preg_all);
    Muls(vreg_pse15, vreg_pse15, slopes, preg_all);
    Muls(vreg_pse16, vreg_pse16, slopes, preg_all);
    Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
    Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
    Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
    Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
    Adds(vreg_alibi5, vreg_alibi5, -1.0f, preg_all);
    Adds(vreg_alibi6, vreg_alibi6, -1.0f, preg_all);
    Adds(vreg_alibi7, vreg_alibi7, -1.0f, preg_all);
    Adds(vreg_alibi8, vreg_alibi8, -1.0f, preg_all);
    Adds(vreg_alibi9, vreg_alibi9, -1.0f, preg_all);
    Adds(vreg_alibi10, vreg_alibi10, -1.0f, preg_all);
    Adds(vreg_alibi11, vreg_alibi11, -1.0f, preg_all);
    Adds(vreg_alibi12, vreg_alibi12, -1.0f, preg_all);
    Adds(vreg_alibi13, vreg_alibi13, -1.0f, preg_all);
    Adds(vreg_alibi14, vreg_alibi14, -1.0f, preg_all);
    Adds(vreg_alibi15, vreg_alibi15, -1.0f, preg_all);
    Adds(vreg_alibi16, vreg_alibi16, -1.0f, preg_all);
}

__simd_callee__ inline void ComputePseInnerMulAddSqrt16(
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    RegTensor<float> &vreg_pse9, RegTensor<float> &vreg_pse10,
    RegTensor<float> &vreg_pse11, RegTensor<float> &vreg_pse12,
    RegTensor<float> &vreg_pse13, RegTensor<float> &vreg_pse14,
    RegTensor<float> &vreg_pse15, RegTensor<float> &vreg_pse16,
    RegTensor<float> &vreg_alibi1, RegTensor<float> &vreg_alibi2,
    RegTensor<float> &vreg_alibi3, RegTensor<float> &vreg_alibi4,
    RegTensor<float> &vreg_alibi5, RegTensor<float> &vreg_alibi6,
    RegTensor<float> &vreg_alibi7, RegTensor<float> &vreg_alibi8,
    RegTensor<float> &vreg_alibi9, RegTensor<float> &vreg_alibi10,
    RegTensor<float> &vreg_alibi11, RegTensor<float> &vreg_alibi12,
    RegTensor<float> &vreg_alibi13, RegTensor<float> &vreg_alibi14,
    RegTensor<float> &vreg_alibi15, RegTensor<float> &vreg_alibi16,
    const float slopes, MaskReg &preg_all)
{
    Abs(vreg_pse1, vreg_alibi1, preg_all);
    Abs(vreg_pse2, vreg_alibi2, preg_all);
    Abs(vreg_pse3, vreg_alibi3, preg_all);
    Abs(vreg_pse4, vreg_alibi4, preg_all);
    Abs(vreg_pse5, vreg_alibi5, preg_all);
    Abs(vreg_pse6, vreg_alibi6, preg_all);
    Abs(vreg_pse7, vreg_alibi7, preg_all);
    Abs(vreg_pse8, vreg_alibi8, preg_all);
    Abs(vreg_pse9, vreg_alibi9, preg_all);
    Abs(vreg_pse10, vreg_alibi10, preg_all);
    Abs(vreg_pse11, vreg_alibi11, preg_all);
    Abs(vreg_pse12, vreg_alibi12, preg_all);
    Abs(vreg_pse13, vreg_alibi13, preg_all);
    Abs(vreg_pse14, vreg_alibi14, preg_all);
    Abs(vreg_pse15, vreg_alibi15, preg_all);
    Abs(vreg_pse16, vreg_alibi16, preg_all);
    Sqrt(vreg_pse1, vreg_pse1, preg_all);
    Sqrt(vreg_pse2, vreg_pse2, preg_all);
    Sqrt(vreg_pse3, vreg_pse3, preg_all);
    Sqrt(vreg_pse4, vreg_pse4, preg_all);
    Sqrt(vreg_pse5, vreg_pse5, preg_all);
    Sqrt(vreg_pse6, vreg_pse6, preg_all);
    Sqrt(vreg_pse7, vreg_pse7, preg_all);
    Sqrt(vreg_pse8, vreg_pse8, preg_all);
    Sqrt(vreg_pse9, vreg_pse9, preg_all);
    Sqrt(vreg_pse10, vreg_pse10, preg_all);
    Sqrt(vreg_pse11, vreg_pse11, preg_all);
    Sqrt(vreg_pse12, vreg_pse12, preg_all);
    Sqrt(vreg_pse13, vreg_pse13, preg_all);
    Sqrt(vreg_pse14, vreg_pse14, preg_all);
    Sqrt(vreg_pse15, vreg_pse15, preg_all);
    Sqrt(vreg_pse16, vreg_pse16, preg_all);
    Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
    Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
    Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
    Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
    Muls(vreg_pse5, vreg_pse5, slopes, preg_all);
    Muls(vreg_pse6, vreg_pse6, slopes, preg_all);
    Muls(vreg_pse7, vreg_pse7, slopes, preg_all);
    Muls(vreg_pse8, vreg_pse8, slopes, preg_all);
    Muls(vreg_pse9, vreg_pse9, slopes, preg_all);
    Muls(vreg_pse10, vreg_pse10, slopes, preg_all);
    Muls(vreg_pse11, vreg_pse11, slopes, preg_all);
    Muls(vreg_pse12, vreg_pse12, slopes, preg_all);
    Muls(vreg_pse13, vreg_pse13, slopes, preg_all);
    Muls(vreg_pse14, vreg_pse14, slopes, preg_all);
    Muls(vreg_pse15, vreg_pse15, slopes, preg_all);
    Muls(vreg_pse16, vreg_pse16, slopes, preg_all);
    Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
    Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
    Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
    Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
    Adds(vreg_alibi5, vreg_alibi5, -1.0f, preg_all);
    Adds(vreg_alibi6, vreg_alibi6, -1.0f, preg_all);
    Adds(vreg_alibi7, vreg_alibi7, -1.0f, preg_all);
    Adds(vreg_alibi8, vreg_alibi8, -1.0f, preg_all);
    Adds(vreg_alibi9, vreg_alibi9, -1.0f, preg_all);
    Adds(vreg_alibi10, vreg_alibi10, -1.0f, preg_all);
    Adds(vreg_alibi11, vreg_alibi11, -1.0f, preg_all);
    Adds(vreg_alibi12, vreg_alibi12, -1.0f, preg_all);
    Adds(vreg_alibi13, vreg_alibi13, -1.0f, preg_all);
    Adds(vreg_alibi14, vreg_alibi14, -1.0f, preg_all);
    Adds(vreg_alibi15, vreg_alibi15, -1.0f, preg_all);
    Adds(vreg_alibi16, vreg_alibi16, -1.0f, preg_all);
}

__simd_callee__ inline void LoadCastPseBf16_16(
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    RegTensor<float> &vreg_pse9, RegTensor<float> &vreg_pse10,
    RegTensor<float> &vreg_pse11, RegTensor<float> &vreg_pse12,
    RegTensor<float> &vreg_pse13, RegTensor<float> &vreg_pse14,
    RegTensor<float> &vreg_pse15, RegTensor<float> &vreg_pse16,
    __ubuf__ bfloat16_t *&pseUb, const uint32_t i, const uint32_t pseStride,
    MaskReg &preg_all_b16)
{
    RegTensor<bfloat16_t> vreg_pse_bf16_src1, vreg_pse_bf16_src2;
    RegTensor<bfloat16_t> vreg_pse_bf16_src3, vreg_pse_bf16_src4;
    RegTensor<bfloat16_t> vreg_pse_bf16_src5, vreg_pse_bf16_src6;
    RegTensor<bfloat16_t> vreg_pse_bf16_src7, vreg_pse_bf16_src8;
    RegTensor<bfloat16_t> vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse3_bf16, vreg_pse4_bf16;
    RegTensor<bfloat16_t> vreg_pse5_bf16, vreg_pse6_bf16, vreg_pse7_bf16, vreg_pse8_bf16;
    RegTensor<bfloat16_t> vreg_pse9_bf16, vreg_pse10_bf16, vreg_pse11_bf16, vreg_pse12_bf16;
    RegTensor<bfloat16_t> vreg_pse13_bf16, vreg_pse14_bf16, vreg_pse15_bf16, vreg_pse16_bf16;
    LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
    LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src3, pseUb + floatRepSize * 4 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src4, pseUb + floatRepSize * 6 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src5, pseUb + floatRepSize * 8 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src6, pseUb + floatRepSize * 10 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src7, pseUb + floatRepSize * 12 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src8, pseUb + floatRepSize * 14 + i * pseStride);
    Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
    Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
    Interleave(vreg_pse5_bf16, vreg_pse6_bf16, vreg_pse_bf16_src3, vreg_pse_bf16_src3);
    Interleave(vreg_pse7_bf16, vreg_pse8_bf16, vreg_pse_bf16_src4, vreg_pse_bf16_src4);
    Interleave(vreg_pse9_bf16, vreg_pse10_bf16, vreg_pse_bf16_src5, vreg_pse_bf16_src5);
    Interleave(vreg_pse11_bf16, vreg_pse12_bf16, vreg_pse_bf16_src6, vreg_pse_bf16_src6);
    Interleave(vreg_pse13_bf16, vreg_pse14_bf16, vreg_pse_bf16_src7, vreg_pse_bf16_src7);
    Interleave(vreg_pse15_bf16, vreg_pse16_bf16, vreg_pse_bf16_src8, vreg_pse_bf16_src8);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse5, vreg_pse5_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse6, vreg_pse6_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse7, vreg_pse7_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse8, vreg_pse8_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse9, vreg_pse9_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse10, vreg_pse10_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse11, vreg_pse11_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse12, vreg_pse12_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse13, vreg_pse13_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse14, vreg_pse14_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse15, vreg_pse15_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse16, vreg_pse16_bf16, preg_all_b16);
}

__simd_callee__ inline void LoadCastPseF16_16(
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    RegTensor<float> &vreg_pse9, RegTensor<float> &vreg_pse10,
    RegTensor<float> &vreg_pse11, RegTensor<float> &vreg_pse12,
    RegTensor<float> &vreg_pse13, RegTensor<float> &vreg_pse14,
    RegTensor<float> &vreg_pse15, RegTensor<float> &vreg_pse16,
    __ubuf__ half *&pseUb, const uint32_t i, const uint32_t pseStride,
    MaskReg &preg_all_b16)
{
    RegTensor<half> vreg_pse_f16_src1, vreg_pse_f16_src2;
    RegTensor<half> vreg_pse_f16_src3, vreg_pse_f16_src4;
    RegTensor<half> vreg_pse_f16_src5, vreg_pse_f16_src6;
    RegTensor<half> vreg_pse_f16_src7, vreg_pse_f16_src8;
    RegTensor<half> vreg_pse1_f16, vreg_pse2_f16, vreg_pse3_f16, vreg_pse4_f16;
    RegTensor<half> vreg_pse5_f16, vreg_pse6_f16, vreg_pse7_f16, vreg_pse8_f16;
    RegTensor<half> vreg_pse9_f16, vreg_pse10_f16, vreg_pse11_f16, vreg_pse12_f16;
    RegTensor<half> vreg_pse13_f16, vreg_pse14_f16, vreg_pse15_f16, vreg_pse16_f16;
    LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
    LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
    LoadAlign(vreg_pse_f16_src3, pseUb + floatRepSize * 4 + i * pseStride);
    LoadAlign(vreg_pse_f16_src4, pseUb + floatRepSize * 6 + i * pseStride);
    LoadAlign(vreg_pse_f16_src5, pseUb + floatRepSize * 8 + i * pseStride);
    LoadAlign(vreg_pse_f16_src6, pseUb + floatRepSize * 10 + i * pseStride);
    LoadAlign(vreg_pse_f16_src7, pseUb + floatRepSize * 12 + i * pseStride);
    LoadAlign(vreg_pse_f16_src8, pseUb + floatRepSize * 14 + i * pseStride);
    Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
    Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
    Interleave(vreg_pse5_f16, vreg_pse6_f16, vreg_pse_f16_src3, vreg_pse_f16_src3);
    Interleave(vreg_pse7_f16, vreg_pse8_f16, vreg_pse_f16_src4, vreg_pse_f16_src4);
    Interleave(vreg_pse9_f16, vreg_pse10_f16, vreg_pse_f16_src5, vreg_pse_f16_src5);
    Interleave(vreg_pse11_f16, vreg_pse12_f16, vreg_pse_f16_src6, vreg_pse_f16_src6);
    Interleave(vreg_pse13_f16, vreg_pse14_f16, vreg_pse_f16_src7, vreg_pse_f16_src7);
    Interleave(vreg_pse15_f16, vreg_pse16_f16, vreg_pse_f16_src8, vreg_pse_f16_src8);
    Cast<float, half, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse5, vreg_pse5_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse6, vreg_pse6_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse7, vreg_pse7_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse8, vreg_pse8_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse9, vreg_pse9_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse10, vreg_pse10_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse11, vreg_pse11_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse12, vreg_pse12_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse13, vreg_pse13_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse14, vreg_pse14_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse15, vreg_pse15_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse16, vreg_pse16_f16, preg_all_b16);
}

__simd_callee__ inline void MaxReduce16(
    RegTensor<float> &vreg_input_max,
    RegTensor<float> &vreg_a1, RegTensor<float> &vreg_a2,
    RegTensor<float> &vreg_a3, RegTensor<float> &vreg_a4,
    RegTensor<float> &vreg_a5, RegTensor<float> &vreg_a6,
    RegTensor<float> &vreg_a7, RegTensor<float> &vreg_a8,
    RegTensor<float> &vreg_b1, RegTensor<float> &vreg_b2,
    RegTensor<float> &vreg_b3, RegTensor<float> &vreg_b4,
    RegTensor<float> &vreg_b5, RegTensor<float> &vreg_b6,
    RegTensor<float> &vreg_b7, RegTensor<float> &vreg_b8,
    MaskReg &preg_all)
{
    RegTensor<float> vreg_max_tmp1;
    RegTensor<float> vreg_max_tmp2;
    RegTensor<float> vreg_max_tmp3;
    RegTensor<float> vreg_max_tmp4;
    RegTensor<float> vreg_max_tmp5;
    RegTensor<float> vreg_max_tmp6;
    RegTensor<float> vreg_max_tmp7;
    RegTensor<float> vreg_max_tmp8;
    Max(vreg_max_tmp1, vreg_a1, vreg_a2, preg_all);
    Max(vreg_max_tmp2, vreg_a3, vreg_a4, preg_all);
    Max(vreg_max_tmp3, vreg_a5, vreg_a6, preg_all);
    Max(vreg_max_tmp4, vreg_a7, vreg_a8, preg_all);
    Max(vreg_max_tmp5, vreg_b1, vreg_b2, preg_all);
    Max(vreg_max_tmp6, vreg_b3, vreg_b4, preg_all);
    Max(vreg_max_tmp7, vreg_b5, vreg_b6, preg_all);
    Max(vreg_max_tmp8, vreg_b7, vreg_b8, preg_all);
    Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp2, preg_all);
    Max(vreg_max_tmp3, vreg_max_tmp3, vreg_max_tmp4, preg_all);
    Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp6, preg_all);
    Max(vreg_max_tmp7, vreg_max_tmp7, vreg_max_tmp8, preg_all);
    Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp3, preg_all);
    Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp7, preg_all);
    Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp5, preg_all);
    Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        vreg_input_max, vreg_max_tmp1, preg_all);
}

template<typename T, typename T2>
__simd_callee__ inline void ExpSumReduceStore1024(
    RegTensor<float> &vreg_exp_sum1,
    RegTensor<float> &vreg_exp_even1, RegTensor<float> &vreg_exp_odd1,
    RegTensor<float> &vreg_exp_even2, RegTensor<float> &vreg_exp_odd2,
    RegTensor<float> &vreg_exp_even3, RegTensor<float> &vreg_exp_odd3,
    RegTensor<float> &vreg_exp_even4, RegTensor<float> &vreg_exp_odd4,
    RegTensor<float> &vreg_exp_even5, RegTensor<float> &vreg_exp_odd5,
    RegTensor<float> &vreg_exp_even6, RegTensor<float> &vreg_exp_odd6,
    RegTensor<float> &vreg_exp_even7, RegTensor<float> &vreg_exp_odd7,
    RegTensor<float> &vreg_exp_even8, RegTensor<float> &vreg_exp_odd8,
    UnalignRegForStore &ureg_exp_sum, __ubuf__ T *&expSumUb,
    MaskReg &preg_all)
{
    RegTensor<float> vreg_exp_sum2, vreg_exp_sum3, vreg_exp_sum4;
    RegTensor<float> vreg_exp_sum5, vreg_exp_sum6, vreg_exp_sum7, vreg_exp_sum8;
    Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
    Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
    Add(vreg_exp_sum3, vreg_exp_even3, vreg_exp_odd3, preg_all);
    Add(vreg_exp_sum4, vreg_exp_even4, vreg_exp_odd4, preg_all);
    Add(vreg_exp_sum5, vreg_exp_even5, vreg_exp_odd5, preg_all);
    Add(vreg_exp_sum6, vreg_exp_even6, vreg_exp_odd6, preg_all);
    Add(vreg_exp_sum7, vreg_exp_even7, vreg_exp_odd7, preg_all);
    Add(vreg_exp_sum8, vreg_exp_even8, vreg_exp_odd8, preg_all);
    Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum2, preg_all);
    Add(vreg_exp_sum3, vreg_exp_sum3, vreg_exp_sum4, preg_all);
    Add(vreg_exp_sum5, vreg_exp_sum5, vreg_exp_sum6, preg_all);
    Add(vreg_exp_sum7, vreg_exp_sum7, vreg_exp_sum8, preg_all);
    Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum3, preg_all);
    Add(vreg_exp_sum5, vreg_exp_sum5, vreg_exp_sum7, preg_all);
    Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum5, preg_all);
    Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        vreg_exp_sum1, vreg_exp_sum1, preg_all);
    StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T *&)expSumUb), vreg_exp_sum1, ureg_exp_sum, 1);
}

template<typename T2>
__simd_callee__ inline void CastStoreExpBf16_1024(
    RegTensor<float> &vreg_exp_even1, RegTensor<float> &vreg_exp_odd1,
    RegTensor<float> &vreg_exp_even2, RegTensor<float> &vreg_exp_odd2,
    RegTensor<float> &vreg_exp_even3, RegTensor<float> &vreg_exp_odd3,
    RegTensor<float> &vreg_exp_even4, RegTensor<float> &vreg_exp_odd4,
    RegTensor<float> &vreg_exp_even5, RegTensor<float> &vreg_exp_odd5,
    RegTensor<float> &vreg_exp_even6, RegTensor<float> &vreg_exp_odd6,
    RegTensor<float> &vreg_exp_even7, RegTensor<float> &vreg_exp_odd7,
    RegTensor<float> &vreg_exp_even8, RegTensor<float> &vreg_exp_odd8,
    __ubuf__ T2 *&expUb1, __ubuf__ T2 *&expUb2,
    __ubuf__ T2 *&expUb3, __ubuf__ T2 *&expUb4,
    __ubuf__ T2 *&expUb5, __ubuf__ T2 *&expUb6,
    __ubuf__ T2 *&expUb7, __ubuf__ T2 *&expUb8,
    const uint32_t blockStride, const uint32_t repeatStride,
    MaskReg &preg_all, MaskReg &storeMask)
{
    RegTensor<bfloat16_t> vreg_exp_even1_bf16, vreg_exp_odd1_bf16;
    RegTensor<bfloat16_t> vreg_exp_even2_bf16, vreg_exp_odd2_bf16;
    RegTensor<bfloat16_t> vreg_exp_even3_bf16, vreg_exp_odd3_bf16;
    RegTensor<bfloat16_t> vreg_exp_even4_bf16, vreg_exp_odd4_bf16;
    RegTensor<bfloat16_t> vreg_exp_even5_bf16, vreg_exp_odd5_bf16;
    RegTensor<bfloat16_t> vreg_exp_even6_bf16, vreg_exp_odd6_bf16;
    RegTensor<bfloat16_t> vreg_exp_even7_bf16, vreg_exp_odd7_bf16;
    RegTensor<bfloat16_t> vreg_exp_even8_bf16, vreg_exp_odd8_bf16;
    RegTensor<bfloat16_t> vreg_exp1_bf16, vreg_exp2_bf16, vreg_exp3_bf16, vreg_exp4_bf16;
    RegTensor<bfloat16_t> vreg_exp5_bf16, vreg_exp6_bf16, vreg_exp7_bf16, vreg_exp8_bf16;
    Cast<T2, float, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even3_bf16, vreg_exp_even3, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd3_bf16, vreg_exp_odd3, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even4_bf16, vreg_exp_even4, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd4_bf16, vreg_exp_odd4, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even5_bf16, vreg_exp_even5, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd5_bf16, vreg_exp_odd5, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even6_bf16, vreg_exp_even6, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd6_bf16, vreg_exp_odd6, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even7_bf16, vreg_exp_even7, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd7_bf16, vreg_exp_odd7, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even8_bf16, vreg_exp_even8, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd8_bf16, vreg_exp_odd8, preg_all);
    Or((RegTensor<uint16_t>&)vreg_exp1_bf16, (RegTensor<uint16_t>&)vreg_exp_even1_bf16,
        (RegTensor<uint16_t>&)vreg_exp_odd1_bf16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp2_bf16, (RegTensor<uint16_t>&)vreg_exp_even2_bf16,
        (RegTensor<uint16_t>&)vreg_exp_odd2_bf16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp3_bf16, (RegTensor<uint16_t>&)vreg_exp_even3_bf16,
        (RegTensor<uint16_t>&)vreg_exp_odd3_bf16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp4_bf16, (RegTensor<uint16_t>&)vreg_exp_even4_bf16,
        (RegTensor<uint16_t>&)vreg_exp_odd4_bf16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp5_bf16, (RegTensor<uint16_t>&)vreg_exp_even5_bf16,
        (RegTensor<uint16_t>&)vreg_exp_odd5_bf16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp6_bf16, (RegTensor<uint16_t>&)vreg_exp_even6_bf16,
        (RegTensor<uint16_t>&)vreg_exp_odd6_bf16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp7_bf16, (RegTensor<uint16_t>&)vreg_exp_even7_bf16,
        (RegTensor<uint16_t>&)vreg_exp_odd7_bf16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp8_bf16, (RegTensor<uint16_t>&)vreg_exp_even8_bf16,
        (RegTensor<uint16_t>&)vreg_exp_odd8_bf16, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb3), vreg_exp3_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb4), vreg_exp4_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb5), vreg_exp5_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb6), vreg_exp6_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb7), vreg_exp7_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb8), vreg_exp8_bf16, blockStride, repeatStride, storeMask);
}

template<typename T2>
__simd_callee__ inline void CastStoreExpF16_1024(
    RegTensor<float> &vreg_exp_even1, RegTensor<float> &vreg_exp_odd1,
    RegTensor<float> &vreg_exp_even2, RegTensor<float> &vreg_exp_odd2,
    RegTensor<float> &vreg_exp_even3, RegTensor<float> &vreg_exp_odd3,
    RegTensor<float> &vreg_exp_even4, RegTensor<float> &vreg_exp_odd4,
    RegTensor<float> &vreg_exp_even5, RegTensor<float> &vreg_exp_odd5,
    RegTensor<float> &vreg_exp_even6, RegTensor<float> &vreg_exp_odd6,
    RegTensor<float> &vreg_exp_even7, RegTensor<float> &vreg_exp_odd7,
    RegTensor<float> &vreg_exp_even8, RegTensor<float> &vreg_exp_odd8,
    __ubuf__ T2 *&expUb1, __ubuf__ T2 *&expUb2,
    __ubuf__ T2 *&expUb3, __ubuf__ T2 *&expUb4,
    __ubuf__ T2 *&expUb5, __ubuf__ T2 *&expUb6,
    __ubuf__ T2 *&expUb7, __ubuf__ T2 *&expUb8,
    const uint32_t blockStride, const uint32_t repeatStride,
    MaskReg &preg_all, MaskReg &storeMask)
{
    RegTensor<half> vreg_exp_even1_f16, vreg_exp_odd1_f16;
    RegTensor<half> vreg_exp_even2_f16, vreg_exp_odd2_f16;
    RegTensor<half> vreg_exp_even3_f16, vreg_exp_odd3_f16;
    RegTensor<half> vreg_exp_even4_f16, vreg_exp_odd4_f16;
    RegTensor<half> vreg_exp_even5_f16, vreg_exp_odd5_f16;
    RegTensor<half> vreg_exp_even6_f16, vreg_exp_odd6_f16;
    RegTensor<half> vreg_exp_even7_f16, vreg_exp_odd7_f16;
    RegTensor<half> vreg_exp_even8_f16, vreg_exp_odd8_f16;
    RegTensor<half> vreg_exp1_f16, vreg_exp2_f16, vreg_exp3_f16, vreg_exp4_f16;
    RegTensor<half> vreg_exp5_f16, vreg_exp6_f16, vreg_exp7_f16, vreg_exp8_f16;
    Cast<T2, float, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even3_f16, vreg_exp_even3, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd3_f16, vreg_exp_odd3, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even4_f16, vreg_exp_even4, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd4_f16, vreg_exp_odd4, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even5_f16, vreg_exp_even5, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd5_f16, vreg_exp_odd5, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even6_f16, vreg_exp_even6, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd6_f16, vreg_exp_odd6, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even7_f16, vreg_exp_even7, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd7_f16, vreg_exp_odd7, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even8_f16, vreg_exp_even8, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd8_f16, vreg_exp_odd8, preg_all);
    Or((RegTensor<uint16_t>&)vreg_exp1_f16, (RegTensor<uint16_t>&)vreg_exp_even1_f16,
        (RegTensor<uint16_t>&)vreg_exp_odd1_f16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp2_f16, (RegTensor<uint16_t>&)vreg_exp_even2_f16,
        (RegTensor<uint16_t>&)vreg_exp_odd2_f16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp3_f16, (RegTensor<uint16_t>&)vreg_exp_even3_f16,
        (RegTensor<uint16_t>&)vreg_exp_odd3_f16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp4_f16, (RegTensor<uint16_t>&)vreg_exp_even4_f16,
        (RegTensor<uint16_t>&)vreg_exp_odd4_f16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp5_f16, (RegTensor<uint16_t>&)vreg_exp_even5_f16,
        (RegTensor<uint16_t>&)vreg_exp_odd5_f16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp6_f16, (RegTensor<uint16_t>&)vreg_exp_even6_f16,
        (RegTensor<uint16_t>&)vreg_exp_odd6_f16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp7_f16, (RegTensor<uint16_t>&)vreg_exp_even7_f16,
        (RegTensor<uint16_t>&)vreg_exp_odd7_f16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp8_f16, (RegTensor<uint16_t>&)vreg_exp_even8_f16,
        (RegTensor<uint16_t>&)vreg_exp_odd8_f16, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb3), vreg_exp3_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb4), vreg_exp4_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb5), vreg_exp5_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb6), vreg_exp6_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb7), vreg_exp7_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb8), vreg_exp8_f16, blockStride, repeatStride, storeMask);
}

} // namespace FaVectorApi

#endif // VF_BASIC_BLOCK_UNALIGNED1024_COMMON_H
