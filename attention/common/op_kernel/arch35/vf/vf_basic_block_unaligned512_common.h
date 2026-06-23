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
 * \file vf_basic_block_unaligned512_common.h
 * \brief shared __simd_callee__ helpers for unaligned512 no_update / update VF kernels
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED512_COMMON_H
#define VF_BASIC_BLOCK_UNALIGNED512_COMMON_H

#include "vf_basic_block_utils.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {

#define VREG_FLOAT_DECL_8(name) \
    RegTensor<float> name##1; RegTensor<float> name##2; \
    RegTensor<float> name##3; RegTensor<float> name##4; \
    RegTensor<float> name##5; RegTensor<float> name##6; \
    RegTensor<float> name##7; RegTensor<float> name##8

#define VREG_FLOAT_DECL_EXP_PAIRS_4(name) \
    RegTensor<float> name##_even1; RegTensor<float> name##_odd1; \
    RegTensor<float> name##_even2; RegTensor<float> name##_odd2; \
    RegTensor<float> name##_even3; RegTensor<float> name##_odd3; \
    RegTensor<float> name##_even4; RegTensor<float> name##_odd4

#define VREG_PREG_COMPARE_DECL_8() \
    MaskReg preg_compare1; MaskReg preg_compare2; \
    MaskReg preg_compare3; MaskReg preg_compare4; \
    MaskReg preg_compare5; MaskReg preg_compare6; \
    MaskReg preg_compare7; MaskReg preg_compare8

#define VREG_FLOAT_DECL_6(name) \
    RegTensor<float> name##1; RegTensor<float> name##2; \
    RegTensor<float> name##3; RegTensor<float> name##4; \
    RegTensor<float> name##5; RegTensor<float> name##6

#define VREG_PREG_COMPARE_DECL_6() \
    MaskReg preg_compare1; MaskReg preg_compare2; \
    MaskReg preg_compare3; MaskReg preg_compare4; \
    MaskReg preg_compare5; MaskReg preg_compare6

__simd_callee__ inline void ComputePseInnerMulAdd8(
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    RegTensor<float> &vreg_alibi1, RegTensor<float> &vreg_alibi2,
    RegTensor<float> &vreg_alibi3, RegTensor<float> &vreg_alibi4,
    RegTensor<float> &vreg_alibi5, RegTensor<float> &vreg_alibi6,
    RegTensor<float> &vreg_alibi7, RegTensor<float> &vreg_alibi8,
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
    Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
    Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
    Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
    Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
    Muls(vreg_pse5, vreg_pse5, slopes, preg_all);
    Muls(vreg_pse6, vreg_pse6, slopes, preg_all);
    Muls(vreg_pse7, vreg_pse7, slopes, preg_all);
    Muls(vreg_pse8, vreg_pse8, slopes, preg_all);
    Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
    Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
    Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
    Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
    Adds(vreg_alibi5, vreg_alibi5, -1.0f, preg_all);
    Adds(vreg_alibi6, vreg_alibi6, -1.0f, preg_all);
    Adds(vreg_alibi7, vreg_alibi7, -1.0f, preg_all);
    Adds(vreg_alibi8, vreg_alibi8, -1.0f, preg_all);
}

__simd_callee__ inline void ComputePseInnerMulAddSqrt8(
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    RegTensor<float> &vreg_alibi1, RegTensor<float> &vreg_alibi2,
    RegTensor<float> &vreg_alibi3, RegTensor<float> &vreg_alibi4,
    RegTensor<float> &vreg_alibi5, RegTensor<float> &vreg_alibi6,
    RegTensor<float> &vreg_alibi7, RegTensor<float> &vreg_alibi8,
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
    Sqrt(vreg_pse1, vreg_pse1, preg_all);
    Sqrt(vreg_pse2, vreg_pse2, preg_all);
    Sqrt(vreg_pse3, vreg_pse3, preg_all);
    Sqrt(vreg_pse4, vreg_pse4, preg_all);
    Sqrt(vreg_pse5, vreg_pse5, preg_all);
    Sqrt(vreg_pse6, vreg_pse6, preg_all);
    Sqrt(vreg_pse7, vreg_pse7, preg_all);
    Sqrt(vreg_pse8, vreg_pse8, preg_all);
    Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
    Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
    Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
    Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
    Muls(vreg_pse5, vreg_pse5, slopes, preg_all);
    Muls(vreg_pse6, vreg_pse6, slopes, preg_all);
    Muls(vreg_pse7, vreg_pse7, slopes, preg_all);
    Muls(vreg_pse8, vreg_pse8, slopes, preg_all);
    Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
    Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
    Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
    Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
    Adds(vreg_alibi5, vreg_alibi5, -1.0f, preg_all);
    Adds(vreg_alibi6, vreg_alibi6, -1.0f, preg_all);
    Adds(vreg_alibi7, vreg_alibi7, -1.0f, preg_all);
    Adds(vreg_alibi8, vreg_alibi8, -1.0f, preg_all);
}

__simd_callee__ inline void ComputePseInnerMulAdd6(
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    RegTensor<float> &vreg_alibi1, RegTensor<float> &vreg_alibi2,
    RegTensor<float> &vreg_alibi3, RegTensor<float> &vreg_alibi4,
    RegTensor<float> &vreg_alibi5, RegTensor<float> &vreg_alibi6,
    const float slopes, MaskReg &preg_all)
{
    Abs(vreg_pse1, vreg_alibi1, preg_all);
    Abs(vreg_pse2, vreg_alibi2, preg_all);
    Abs(vreg_pse3, vreg_alibi3, preg_all);
    Abs(vreg_pse4, vreg_alibi4, preg_all);
    Abs(vreg_pse5, vreg_alibi5, preg_all);
    Abs(vreg_pse6, vreg_alibi6, preg_all);
    Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
    Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
    Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
    Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
    Muls(vreg_pse5, vreg_pse5, slopes, preg_all);
    Muls(vreg_pse6, vreg_pse6, slopes, preg_all);
    Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
    Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
    Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
    Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
    Adds(vreg_alibi5, vreg_alibi5, -1.0f, preg_all);
    Adds(vreg_alibi6, vreg_alibi6, -1.0f, preg_all);
}

__simd_callee__ inline void ComputePseInnerMulAddSqrt6(
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    RegTensor<float> &vreg_alibi1, RegTensor<float> &vreg_alibi2,
    RegTensor<float> &vreg_alibi3, RegTensor<float> &vreg_alibi4,
    RegTensor<float> &vreg_alibi5, RegTensor<float> &vreg_alibi6,
    const float slopes, MaskReg &preg_all)
{
    Abs(vreg_pse1, vreg_alibi1, preg_all);
    Abs(vreg_pse2, vreg_alibi2, preg_all);
    Abs(vreg_pse3, vreg_alibi3, preg_all);
    Abs(vreg_pse4, vreg_alibi4, preg_all);
    Abs(vreg_pse5, vreg_alibi5, preg_all);
    Abs(vreg_pse6, vreg_alibi6, preg_all);
    Sqrt(vreg_pse1, vreg_pse1, preg_all);
    Sqrt(vreg_pse2, vreg_pse2, preg_all);
    Sqrt(vreg_pse3, vreg_pse3, preg_all);
    Sqrt(vreg_pse4, vreg_pse4, preg_all);
    Sqrt(vreg_pse5, vreg_pse5, preg_all);
    Sqrt(vreg_pse6, vreg_pse6, preg_all);
    Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
    Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
    Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
    Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
    Muls(vreg_pse5, vreg_pse5, slopes, preg_all);
    Muls(vreg_pse6, vreg_pse6, slopes, preg_all);
    Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
    Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
    Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
    Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
    Adds(vreg_alibi5, vreg_alibi5, -1.0f, preg_all);
    Adds(vreg_alibi6, vreg_alibi6, -1.0f, preg_all);
}

__simd_callee__ inline void ComputePseInnerMulAdd2(
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    RegTensor<float> &vreg_alibi7, RegTensor<float> &vreg_alibi8,
    const float slopes, MaskReg &preg_all)
{
    Abs(vreg_pse7, vreg_alibi7, preg_all);
    Abs(vreg_pse8, vreg_alibi8, preg_all);
    Muls(vreg_pse7, vreg_pse7, slopes, preg_all);
    Muls(vreg_pse8, vreg_pse8, slopes, preg_all);
    Adds(vreg_alibi7, vreg_alibi7, -1.0f, preg_all);
    Adds(vreg_alibi8, vreg_alibi8, -1.0f, preg_all);
}

__simd_callee__ inline void ComputePseInnerMulAddSqrt2(
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    RegTensor<float> &vreg_alibi7, RegTensor<float> &vreg_alibi8,
    const float slopes, MaskReg &preg_all)
{
    Abs(vreg_pse7, vreg_alibi7, preg_all);
    Abs(vreg_pse8, vreg_alibi8, preg_all);
    Sqrt(vreg_pse7, vreg_pse7, preg_all);
    Sqrt(vreg_pse8, vreg_pse8, preg_all);
    Muls(vreg_pse7, vreg_pse7, slopes, preg_all);
    Muls(vreg_pse8, vreg_pse8, slopes, preg_all);
    Adds(vreg_alibi7, vreg_alibi7, -1.0f, preg_all);
    Adds(vreg_alibi8, vreg_alibi8, -1.0f, preg_all);
}

__simd_callee__ inline void LoadCastPseBf16_8(
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    __ubuf__ bfloat16_t *&pseUb, const uint32_t i, const uint32_t pseStride,
    MaskReg &preg_all_b16)
{
    RegTensor<bfloat16_t> vreg_pse_bf16_src1;
    RegTensor<bfloat16_t> vreg_pse_bf16_src2;
    RegTensor<bfloat16_t> vreg_pse_bf16_src3;
    RegTensor<bfloat16_t> vreg_pse_bf16_src4;
    RegTensor<bfloat16_t> vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse3_bf16, vreg_pse4_bf16;
    RegTensor<bfloat16_t> vreg_pse5_bf16, vreg_pse6_bf16, vreg_pse7_bf16, vreg_pse8_bf16;
    LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
    LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src3, pseUb + floatRepSize * 4 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src4, pseUb + floatRepSize * 6 + i * pseStride);
    Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
    Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
    Interleave(vreg_pse5_bf16, vreg_pse6_bf16, vreg_pse_bf16_src3, vreg_pse_bf16_src3);
    Interleave(vreg_pse7_bf16, vreg_pse8_bf16, vreg_pse_bf16_src4, vreg_pse_bf16_src4);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse5, vreg_pse5_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse6, vreg_pse6_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse7, vreg_pse7_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse8, vreg_pse8_bf16, preg_all_b16);
}

__simd_callee__ inline void LoadCastPseF16_8(
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    __ubuf__ half *&pseUb, const uint32_t i, const uint32_t pseStride,
    MaskReg &preg_all_b16)
{
    RegTensor<half> vreg_pse_f16_src1;
    RegTensor<half> vreg_pse_f16_src2;
    RegTensor<half> vreg_pse_f16_src3;
    RegTensor<half> vreg_pse_f16_src4;
    RegTensor<half> vreg_pse1_f16, vreg_pse2_f16, vreg_pse3_f16, vreg_pse4_f16;
    RegTensor<half> vreg_pse5_f16, vreg_pse6_f16, vreg_pse7_f16, vreg_pse8_f16;
    LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
    LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
    LoadAlign(vreg_pse_f16_src3, pseUb + floatRepSize * 4 + i * pseStride);
    LoadAlign(vreg_pse_f16_src4, pseUb + floatRepSize * 6 + i * pseStride);
    Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
    Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
    Interleave(vreg_pse5_f16, vreg_pse6_f16, vreg_pse_f16_src3, vreg_pse_f16_src3);
    Interleave(vreg_pse7_f16, vreg_pse8_f16, vreg_pse_f16_src4, vreg_pse_f16_src4);
    Cast<float, half, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse5, vreg_pse5_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse6, vreg_pse6_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse7, vreg_pse7_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse8, vreg_pse8_f16, preg_all_b16);
}

__simd_callee__ inline void LoadCastPseBf16_6(
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    __ubuf__ bfloat16_t *&pseUb, const uint32_t i, const uint32_t pseStride,
    MaskReg &preg_all_b16)
{
    RegTensor<bfloat16_t> vreg_pse_bf16_src1;
    RegTensor<bfloat16_t> vreg_pse_bf16_src2;
    RegTensor<bfloat16_t> vreg_pse_bf16_src3;
    RegTensor<bfloat16_t> vreg_pse1_bf16, vreg_pse2_bf16;
    RegTensor<bfloat16_t> vreg_pse3_bf16, vreg_pse4_bf16;
    RegTensor<bfloat16_t> vreg_pse5_bf16, vreg_pse6_bf16;
    LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
    LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src3, pseUb + floatRepSize * 4 + i * pseStride);
    Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
    Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
    Interleave(vreg_pse5_bf16, vreg_pse6_bf16, vreg_pse_bf16_src3, vreg_pse_bf16_src3);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse5, vreg_pse5_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse6, vreg_pse6_bf16, preg_all_b16);
}

__simd_callee__ inline void LoadCastPseF16_6(
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    __ubuf__ half *&pseUb, const uint32_t i, const uint32_t pseStride,
    MaskReg &preg_all_b16)
{
    RegTensor<half> vreg_pse_f16_src1;
    RegTensor<half> vreg_pse_f16_src2;
    RegTensor<half> vreg_pse_f16_src3;
    RegTensor<half> vreg_pse1_f16, vreg_pse2_f16;
    RegTensor<half> vreg_pse3_f16, vreg_pse4_f16;
    RegTensor<half> vreg_pse5_f16, vreg_pse6_f16;
    LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
    LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
    LoadAlign(vreg_pse_f16_src3, pseUb + floatRepSize * 4 + i * pseStride);
    Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
    Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
    Interleave(vreg_pse5_f16, vreg_pse6_f16, vreg_pse_f16_src3, vreg_pse_f16_src3);
    Cast<float, half, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse5, vreg_pse5_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse6, vreg_pse6_f16, preg_all_b16);
}

__simd_callee__ inline void LoadCastPseBf16_2(
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    __ubuf__ bfloat16_t *&pseUb, const uint32_t offset,
    const uint32_t i, const uint32_t pseStride,
    MaskReg &preg_all_b16)
{
    RegTensor<bfloat16_t> vreg_pse_bf16_src4;
    RegTensor<bfloat16_t> vreg_pse7_bf16, vreg_pse8_bf16;
    LoadAlign(vreg_pse_bf16_src4, pseUb + offset + i * pseStride);
    Interleave(vreg_pse7_bf16, vreg_pse8_bf16, vreg_pse_bf16_src4, vreg_pse_bf16_src4);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse7, vreg_pse7_bf16, preg_all_b16);
    Cast<float, bfloat16_t, castTraitZero>(vreg_pse8, vreg_pse8_bf16, preg_all_b16);
}

__simd_callee__ inline void LoadCastPseF16_2(
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    __ubuf__ half *&pseUb, const uint32_t offset,
    const uint32_t i, const uint32_t pseStride,
    MaskReg &preg_all_b16)
{
    RegTensor<half> vreg_pse_f16_src4;
    RegTensor<half> vreg_pse7_f16, vreg_pse8_f16;
    LoadAlign(vreg_pse_f16_src4, pseUb + offset + i * pseStride);
    Interleave(vreg_pse7_f16, vreg_pse8_f16, vreg_pse_f16_src4, vreg_pse_f16_src4);
    Cast<float, half, castTraitZero>(vreg_pse7, vreg_pse7_f16, preg_all_b16);
    Cast<float, half, castTraitZero>(vreg_pse8, vreg_pse8_f16, preg_all_b16);
}

template<typename T, typename T2>
__simd_callee__ inline void ExpSumReduceStore512(
    RegTensor<float> &vreg_exp_sum3,
    RegTensor<float> &vreg_exp_even1, RegTensor<float> &vreg_exp_odd1,
    RegTensor<float> &vreg_exp_even2, RegTensor<float> &vreg_exp_odd2,
    RegTensor<float> &vreg_exp_even3, RegTensor<float> &vreg_exp_odd3,
    RegTensor<float> &vreg_exp_even4, RegTensor<float> &vreg_exp_odd4,
    UnalignRegForStore &ureg_exp_sum, __ubuf__ T *&expSumUb,
    MaskReg &preg_all)
{
    RegTensor<float> vreg_exp_sum1, vreg_exp_sum2, vreg_exp_sum4;
    Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
    Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
    Add(vreg_exp_sum3, vreg_exp_even3, vreg_exp_odd3, preg_all);
    Add(vreg_exp_sum4, vreg_exp_even4, vreg_exp_odd4, preg_all);
    Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum2, preg_all);
    Add(vreg_exp_sum3, vreg_exp_sum3, vreg_exp_sum4, preg_all);
    Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum3, preg_all);
    Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        vreg_exp_sum3, vreg_exp_sum3, preg_all);
    StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T *&)expSumUb), vreg_exp_sum3, ureg_exp_sum, 1);
}

template<typename T2>
__simd_callee__ inline void CastStoreExpBf16_512(
    RegTensor<float> &vreg_exp_even1, RegTensor<float> &vreg_exp_odd1,
    RegTensor<float> &vreg_exp_even2, RegTensor<float> &vreg_exp_odd2,
    RegTensor<float> &vreg_exp_even3, RegTensor<float> &vreg_exp_odd3,
    RegTensor<float> &vreg_exp_even4, RegTensor<float> &vreg_exp_odd4,
    __ubuf__ T2 *&expUb1, __ubuf__ T2 *&expUb2,
    __ubuf__ T2 *&expUb3, __ubuf__ T2 *&expUb4,
    const uint32_t blockStride, const uint32_t repeatStride,
    MaskReg &preg_all, MaskReg &storeMask)
{
    RegTensor<bfloat16_t> vreg_exp_even1_bf16, vreg_exp_odd1_bf16;
    RegTensor<bfloat16_t> vreg_exp_even2_bf16, vreg_exp_odd2_bf16;
    RegTensor<bfloat16_t> vreg_exp_even3_bf16, vreg_exp_odd3_bf16;
    RegTensor<bfloat16_t> vreg_exp_even4_bf16, vreg_exp_odd4_bf16;
    RegTensor<bfloat16_t> vreg_exp1_bf16, vreg_exp2_bf16, vreg_exp3_bf16, vreg_exp4_bf16;
    Cast<T2, float, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even3_bf16, vreg_exp_even3, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd3_bf16, vreg_exp_odd3, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even4_bf16, vreg_exp_even4, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd4_bf16, vreg_exp_odd4, preg_all);
    Or((RegTensor<uint16_t>&)vreg_exp1_bf16, (RegTensor<uint16_t>&)vreg_exp_even1_bf16,
        (RegTensor<uint16_t>&)vreg_exp_odd1_bf16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp2_bf16, (RegTensor<uint16_t>&)vreg_exp_even2_bf16,
        (RegTensor<uint16_t>&)vreg_exp_odd2_bf16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp3_bf16, (RegTensor<uint16_t>&)vreg_exp_even3_bf16,
        (RegTensor<uint16_t>&)vreg_exp_odd3_bf16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp4_bf16, (RegTensor<uint16_t>&)vreg_exp_even4_bf16,
        (RegTensor<uint16_t>&)vreg_exp_odd4_bf16, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb3), vreg_exp3_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb4), vreg_exp4_bf16, blockStride, repeatStride, storeMask);
}

template<typename T2>
__simd_callee__ inline void CastStoreExpF16_512(
    RegTensor<float> &vreg_exp_even1, RegTensor<float> &vreg_exp_odd1,
    RegTensor<float> &vreg_exp_even2, RegTensor<float> &vreg_exp_odd2,
    RegTensor<float> &vreg_exp_even3, RegTensor<float> &vreg_exp_odd3,
    RegTensor<float> &vreg_exp_even4, RegTensor<float> &vreg_exp_odd4,
    __ubuf__ T2 *&expUb1, __ubuf__ T2 *&expUb2,
    __ubuf__ T2 *&expUb3, __ubuf__ T2 *&expUb4,
    const uint32_t blockStride, const uint32_t repeatStride,
    MaskReg &preg_all, MaskReg &storeMask)
{
    RegTensor<half> vreg_exp_even1_f16, vreg_exp_odd1_f16;
    RegTensor<half> vreg_exp_even2_f16, vreg_exp_odd2_f16;
    RegTensor<half> vreg_exp_even3_f16, vreg_exp_odd3_f16;
    RegTensor<half> vreg_exp_even4_f16, vreg_exp_odd4_f16;
    RegTensor<half> vreg_exp1_f16, vreg_exp2_f16, vreg_exp3_f16, vreg_exp4_f16;
    Cast<T2, float, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even3_f16, vreg_exp_even3, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd3_f16, vreg_exp_odd3, preg_all);
    Cast<T2, float, castTraitZero>(vreg_exp_even4_f16, vreg_exp_even4, preg_all);
    Cast<T2, float, castTraitOne>(vreg_exp_odd4_f16, vreg_exp_odd4, preg_all);
    Or((RegTensor<uint16_t>&)vreg_exp1_f16, (RegTensor<uint16_t>&)vreg_exp_even1_f16,
        (RegTensor<uint16_t>&)vreg_exp_odd1_f16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp2_f16, (RegTensor<uint16_t>&)vreg_exp_even2_f16,
        (RegTensor<uint16_t>&)vreg_exp_odd2_f16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp3_f16, (RegTensor<uint16_t>&)vreg_exp_even3_f16,
        (RegTensor<uint16_t>&)vreg_exp_odd3_f16, storeMask);
    Or((RegTensor<uint16_t>&)vreg_exp4_f16, (RegTensor<uint16_t>&)vreg_exp_even4_f16,
        (RegTensor<uint16_t>&)vreg_exp_odd4_f16, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb3), vreg_exp3_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb4), vreg_exp4_f16, blockStride, repeatStride, storeMask);
}

__simd_callee__ inline void MaxReduce4(
    RegTensor<float> &vreg_input_max,
    RegTensor<float> &vreg_a1, RegTensor<float> &vreg_a2,
    RegTensor<float> &vreg_a3, RegTensor<float> &vreg_a4,
    RegTensor<float> &vreg_b1, RegTensor<float> &vreg_b2,
    RegTensor<float> &vreg_b3, RegTensor<float> &vreg_b4,
    MaskReg &preg_all)
{
    RegTensor<float> vreg_max_tmp1;
    RegTensor<float> vreg_max_tmp2;
    RegTensor<float> vreg_max_tmp3;
    RegTensor<float> vreg_max_tmp4;
    Max(vreg_max_tmp1, vreg_a1, vreg_a2, preg_all);
    Max(vreg_max_tmp2, vreg_a3, vreg_a4, preg_all);
    Max(vreg_max_tmp3, vreg_b1, vreg_b2, preg_all);
    Max(vreg_max_tmp4, vreg_b3, vreg_b4, preg_all);
    Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp2, preg_all);
    Max(vreg_max_tmp3, vreg_max_tmp3, vreg_max_tmp4, preg_all);
    Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp3, preg_all);
    Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        vreg_input_max, vreg_max_tmp3, preg_all);
}

__simd_callee__ inline void InitAlibiOffsets8(
    RegTensor<float> &vreg_alibi1, RegTensor<float> &vreg_alibi2,
    RegTensor<float> &vreg_alibi3, RegTensor<float> &vreg_alibi4,
    RegTensor<float> &vreg_alibi5, RegTensor<float> &vreg_alibi6,
    RegTensor<float> &vreg_alibi7, RegTensor<float> &vreg_alibi8,
    const float posShift, MaskReg &preg_all)
{
    Arange(vreg_alibi1, posShift);
    Adds(vreg_alibi2, vreg_alibi1, floatRepSize, preg_all);
    Adds(vreg_alibi3, vreg_alibi2, floatRepSize, preg_all);
    Adds(vreg_alibi4, vreg_alibi3, floatRepSize, preg_all);
    Adds(vreg_alibi5, vreg_alibi4, floatRepSize, preg_all);
    Adds(vreg_alibi6, vreg_alibi5, floatRepSize, preg_all);
    Adds(vreg_alibi7, vreg_alibi6, floatRepSize, preg_all);
    Adds(vreg_alibi8, vreg_alibi7, floatRepSize, preg_all);
}

__simd_callee__ inline void InitAlibiOffsets6(
    RegTensor<float> &vreg_alibi1, RegTensor<float> &vreg_alibi2,
    RegTensor<float> &vreg_alibi3, RegTensor<float> &vreg_alibi4,
    RegTensor<float> &vreg_alibi5, RegTensor<float> &vreg_alibi6,
    const float posShift, MaskReg &preg_all)
{
    Arange(vreg_alibi1, posShift);
    Adds(vreg_alibi2, vreg_alibi1, floatRepSize, preg_all);
    Adds(vreg_alibi3, vreg_alibi2, floatRepSize, preg_all);
    Adds(vreg_alibi4, vreg_alibi3, floatRepSize, preg_all);
    Adds(vreg_alibi5, vreg_alibi4, floatRepSize, preg_all);
    Adds(vreg_alibi6, vreg_alibi5, floatRepSize, preg_all);
}

template<typename T>
__simd_callee__ inline void LoadInput8(
    RegTensor<float> &vreg_input_x1, RegTensor<float> &vreg_input_x2,
    RegTensor<float> &vreg_input_x3, RegTensor<float> &vreg_input_x4,
    RegTensor<float> &vreg_input_x5, RegTensor<float> &vreg_input_x6,
    RegTensor<float> &vreg_input_x7, RegTensor<float> &vreg_input_x8,
    __ubuf__ T *&srcUb, const uint16_t i, const uint32_t s2BaseSize)
{
    LoadAlign(vreg_input_x1, srcUb + i * s2BaseSize);
    LoadAlign(vreg_input_x2, srcUb + floatRepSize + i * s2BaseSize);
    LoadAlign(vreg_input_x3, srcUb + floatRepSize * 2 + i * s2BaseSize);
    LoadAlign(vreg_input_x4, srcUb + floatRepSize * 3 + i * s2BaseSize);
    LoadAlign(vreg_input_x5, srcUb + floatRepSize * 4 + i * s2BaseSize);
    LoadAlign(vreg_input_x6, srcUb + floatRepSize * 5 + i * s2BaseSize);
    LoadAlign(vreg_input_x7, srcUb + floatRepSize * 6 + i * s2BaseSize);
    LoadAlign(vreg_input_x8, srcUb + floatRepSize * 7 + i * s2BaseSize);
}

template<typename T>
__simd_callee__ inline void LoadInput6(
    RegTensor<float> &vreg_input_x1, RegTensor<float> &vreg_input_x2,
    RegTensor<float> &vreg_input_x3, RegTensor<float> &vreg_input_x4,
    RegTensor<float> &vreg_input_x5, RegTensor<float> &vreg_input_x6,
    __ubuf__ T *&srcUb, const uint16_t i, const uint32_t s2BaseSize)
{
    LoadAlign(vreg_input_x1, srcUb + i * s2BaseSize);
    LoadAlign(vreg_input_x2, srcUb + floatRepSize + i * s2BaseSize);
    LoadAlign(vreg_input_x3, srcUb + floatRepSize * 2 + i * s2BaseSize);
    LoadAlign(vreg_input_x4, srcUb + floatRepSize * 3 + i * s2BaseSize);
    LoadAlign(vreg_input_x5, srcUb + floatRepSize * 4 + i * s2BaseSize);
    LoadAlign(vreg_input_x6, srcUb + floatRepSize * 5 + i * s2BaseSize);
}

__simd_callee__ inline void ScaleInput4x4(
    RegTensor<float> &v1, RegTensor<float> &v2,
    RegTensor<float> &v3, RegTensor<float> &v4,
    RegTensor<float> &v5, RegTensor<float> &v6,
    RegTensor<float> &v7, RegTensor<float> &v8,
    const float scale, MaskReg &preg_all,
    MaskReg &m1, MaskReg &m2, MaskReg &m3, MaskReg &m4)
{
    Muls(v1, v1, scale, preg_all);
    Muls(v2, v2, scale, preg_all);
    Muls(v3, v3, scale, preg_all);
    Muls(v4, v4, scale, preg_all);
    Muls(v5, v5, scale, m1);
    Muls(v6, v6, scale, m2);
    Muls(v7, v7, scale, m3);
    Muls(v8, v8, scale, m4);
}

__simd_callee__ inline void ScaleInput4x2(
    RegTensor<float> &v1, RegTensor<float> &v2,
    RegTensor<float> &v3, RegTensor<float> &v4,
    RegTensor<float> &v5, RegTensor<float> &v6,
    const float scale, MaskReg &preg_all,
    MaskReg &m1, MaskReg &m2)
{
    Muls(v1, v1, scale, preg_all);
    Muls(v2, v2, scale, preg_all);
    Muls(v3, v3, scale, preg_all);
    Muls(v4, v4, scale, preg_all);
    Muls(v5, v5, scale, m1);
    Muls(v6, v6, scale, m2);
}

__simd_callee__ inline void AddPseToInput4x4(
    RegTensor<float> &x1, RegTensor<float> &x2,
    RegTensor<float> &x3, RegTensor<float> &x4,
    RegTensor<float> &x5, RegTensor<float> &x6,
    RegTensor<float> &x7, RegTensor<float> &x8,
    RegTensor<float> &p1, RegTensor<float> &p2,
    RegTensor<float> &p3, RegTensor<float> &p4,
    RegTensor<float> &p5, RegTensor<float> &p6,
    RegTensor<float> &p7, RegTensor<float> &p8,
    MaskReg &preg_all, MaskReg &m1, MaskReg &m2, MaskReg &m3, MaskReg &m4)
{
    Add(x1, x1, p1, preg_all);
    Add(x2, x2, p2, preg_all);
    Add(x3, x3, p3, preg_all);
    Add(x4, x4, p4, preg_all);
    Add(x5, x5, p5, m1);
    Add(x6, x6, p6, m2);
    Add(x7, x7, p7, m3);
    Add(x8, x8, p8, m4);
}

__simd_callee__ inline void AddPseToInput4x2(
    RegTensor<float> &x1, RegTensor<float> &x2,
    RegTensor<float> &x3, RegTensor<float> &x4,
    RegTensor<float> &x5, RegTensor<float> &x6,
    RegTensor<float> &p1, RegTensor<float> &p2,
    RegTensor<float> &p3, RegTensor<float> &p4,
    RegTensor<float> &p5, RegTensor<float> &p6,
    MaskReg &preg_all, MaskReg &m1, MaskReg &m2)
{
    Add(x1, x1, p1, preg_all);
    Add(x2, x2, p2, preg_all);
    Add(x3, x3, p3, preg_all);
    Add(x4, x4, p4, preg_all);
    Add(x5, x5, p5, m1);
    Add(x6, x6, p6, m2);
}

__simd_callee__ inline void LoadAttenMask8(
    MaskReg &preg_compare1, MaskReg &preg_compare2,
    MaskReg &preg_compare3, MaskReg &preg_compare4,
    MaskReg &preg_compare5, MaskReg &preg_compare6,
    MaskReg &preg_compare7, MaskReg &preg_compare8,
    __ubuf__ uint32_t *&maskUb1, __ubuf__ uint32_t *&maskUb2,
    __ubuf__ uint32_t *&maskUb3, __ubuf__ uint32_t *&maskUb4,
    __ubuf__ uint32_t *&maskUb5, __ubuf__ uint32_t *&maskUb6,
    __ubuf__ uint32_t *&maskUb7, __ubuf__ uint32_t *&maskUb8,
    const uint32_t nPadding)
{
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare1, maskUb1, nPadding);
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare2, maskUb2, nPadding);
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare3, maskUb3, nPadding);
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare4, maskUb4, nPadding);
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare5, maskUb5, nPadding);
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare6, maskUb6, nPadding);
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare7, maskUb7, nPadding);
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare8, maskUb8, nPadding);
}

__simd_callee__ inline void LoadAttenMask6(
    MaskReg &preg_compare1, MaskReg &preg_compare2,
    MaskReg &preg_compare3, MaskReg &preg_compare4,
    MaskReg &preg_compare5, MaskReg &preg_compare6,
    __ubuf__ uint32_t *&maskUb1, __ubuf__ uint32_t *&maskUb2,
    __ubuf__ uint32_t *&maskUb3, __ubuf__ uint32_t *&maskUb4,
    __ubuf__ uint32_t *&maskUb5, __ubuf__ uint32_t *&maskUb6,
    const uint32_t nPadding)
{
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare1, maskUb1, nPadding);
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare2, maskUb2, nPadding);
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare3, maskUb3, nPadding);
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare4, maskUb4, nPadding);
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare5, maskUb5, nPadding);
    LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
        preg_compare6, maskUb6, nPadding);
}

__simd_callee__ inline void ApplyAttenMaskSelect8(
    RegTensor<float> &s1, RegTensor<float> &s2,
    RegTensor<float> &s3, RegTensor<float> &s4,
    RegTensor<float> &s5, RegTensor<float> &s6,
    RegTensor<float> &s7, RegTensor<float> &s8,
    RegTensor<float> &vreg_min,
    RegTensor<float> &x1, RegTensor<float> &x2,
    RegTensor<float> &x3, RegTensor<float> &x4,
    RegTensor<float> &x5, RegTensor<float> &x6,
    RegTensor<float> &x7, RegTensor<float> &x8,
    MaskReg &c1, MaskReg &c2, MaskReg &c3, MaskReg &c4,
    MaskReg &c5, MaskReg &c6, MaskReg &c7, MaskReg &c8)
{
    Select(s1, vreg_min, x1, c1);
    Select(s2, vreg_min, x2, c2);
    Select(s3, vreg_min, x3, c3);
    Select(s4, vreg_min, x4, c4);
    Select(s5, vreg_min, x5, c5);
    Select(s6, vreg_min, x6, c6);
    Select(s7, vreg_min, x7, c7);
    Select(s8, vreg_min, x8, c8);
}

__simd_callee__ inline void ApplyAttenMaskSelect6(
    RegTensor<float> &s1, RegTensor<float> &s2,
    RegTensor<float> &s3, RegTensor<float> &s4,
    RegTensor<float> &s5, RegTensor<float> &s6,
    RegTensor<float> &vreg_min,
    RegTensor<float> &x1, RegTensor<float> &x2,
    RegTensor<float> &x3, RegTensor<float> &x4,
    RegTensor<float> &x5, RegTensor<float> &x6,
    MaskReg &c1, MaskReg &c2, MaskReg &c3,
    MaskReg &c4, MaskReg &c5, MaskReg &c6)
{
    Select(s1, vreg_min, x1, c1);
    Select(s2, vreg_min, x2, c2);
    Select(s3, vreg_min, x3, c3);
    Select(s4, vreg_min, x4, c4);
    Select(s5, vreg_min, x5, c5);
    Select(s6, vreg_min, x6, c6);
}

template<typename T>
__simd_callee__ inline void StoreAlign8(
    RegTensor<float> &v1, RegTensor<float> &v2,
    RegTensor<float> &v3, RegTensor<float> &v4,
    RegTensor<float> &v5, RegTensor<float> &v6,
    RegTensor<float> &v7, RegTensor<float> &v8,
    __ubuf__ T *&srcUb, const uint16_t i, const uint32_t s2BaseSize,
    MaskReg &preg_all)
{
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + i * s2BaseSize, v1, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, v2, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, v3, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, v4, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize * 4 + i * s2BaseSize, v5, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize * 5 + i * s2BaseSize, v6, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize * 6 + i * s2BaseSize, v7, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize * 7 + i * s2BaseSize, v8, preg_all);
}

template<typename T>
__simd_callee__ inline void StoreAlign6(
    RegTensor<float> &v1, RegTensor<float> &v2,
    RegTensor<float> &v3, RegTensor<float> &v4,
    RegTensor<float> &v5, RegTensor<float> &v6,
    __ubuf__ T *&srcUb, const uint16_t i, const uint32_t s2BaseSize,
    MaskReg &preg_all)
{
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + i * s2BaseSize, v1, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, v2, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, v3, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, v4, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize * 4 + i * s2BaseSize, v5, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)srcUb + floatRepSize * 5 + i * s2BaseSize, v6, preg_all);
}

template<typename T>
__simd_callee__ inline void LoadInputDinterleave4(
    RegTensor<float> &x1, RegTensor<float> &x2,
    RegTensor<float> &x3, RegTensor<float> &x4,
    RegTensor<float> &x5, RegTensor<float> &x6,
    RegTensor<float> &x7, RegTensor<float> &x8,
    __ubuf__ T *&srcUb, const uint16_t i, const uint32_t s2BaseSize)
{
    LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
        x1, x2, srcUb + i * s2BaseSize);
    LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
        x3, x4, srcUb + floatRepSize * 2 + i * s2BaseSize);
    LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
        x5, x6, srcUb + floatRepSize * 4 + i * s2BaseSize);
    LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
        x7, x8, srcUb + floatRepSize * 6 + i * s2BaseSize);
}

__simd_callee__ inline void ExpSub8(
    RegTensor<float> &e1, RegTensor<float> &o1,
    RegTensor<float> &e2, RegTensor<float> &o2,
    RegTensor<float> &e3, RegTensor<float> &o3,
    RegTensor<float> &e4, RegTensor<float> &o4,
    RegTensor<float> &x1, RegTensor<float> &x2,
    RegTensor<float> &x3, RegTensor<float> &x4,
    RegTensor<float> &x5, RegTensor<float> &x6,
    RegTensor<float> &x7, RegTensor<float> &x8,
    RegTensor<float> &vreg_max, MaskReg &preg_all)
{
    ExpSub(e1, x1, vreg_max, preg_all);
    ExpSub(o1, x2, vreg_max, preg_all);
    ExpSub(e2, x3, vreg_max, preg_all);
    ExpSub(o2, x4, vreg_max, preg_all);
    ExpSub(e3, x5, vreg_max, preg_all);
    ExpSub(o3, x6, vreg_max, preg_all);
    ExpSub(e4, x7, vreg_max, preg_all);
    ExpSub(o4, x8, vreg_max, preg_all);
}

__simd_callee__ inline void SelectOriTail4(
    RegTensor<float> &o1, RegTensor<float> &o2,
    RegTensor<float> &o3, RegTensor<float> &o4,
    RegTensor<float> &i1, RegTensor<float> &i2,
    RegTensor<float> &i3, RegTensor<float> &i4,
    RegTensor<float> &vreg_min,
    MaskReg &m1, MaskReg &m2, MaskReg &m3, MaskReg &m4)
{
    Select(o1, i1, vreg_min, m1);
    Select(o2, i2, vreg_min, m2);
    Select(o3, i3, vreg_min, m3);
    Select(o4, i4, vreg_min, m4);
}

template <PseTypeEnum pseMode, typename T2>
__simd_callee__ inline void ComputeOrLoadPse512(
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    RegTensor<float> &vreg_alibi1, RegTensor<float> &vreg_alibi2,
    RegTensor<float> &vreg_alibi3, RegTensor<float> &vreg_alibi4,
    RegTensor<float> &vreg_alibi5, RegTensor<float> &vreg_alibi6,
    RegTensor<float> &vreg_alibi7, RegTensor<float> &vreg_alibi8,
    const float slopes, MaskReg &preg_all, MaskReg &preg_all_b16,
    __ubuf__ T2 *pseUb, const uint16_t i, const uint32_t pseStride)
{
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                  pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            ComputePseInnerMulAddSqrt8(vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4,
                vreg_pse5, vreg_pse6, vreg_pse7, vreg_pse8,
                vreg_alibi1, vreg_alibi2, vreg_alibi3, vreg_alibi4,
                vreg_alibi5, vreg_alibi6, vreg_alibi7, vreg_alibi8,
                slopes, preg_all);
        } else {
            ComputePseInnerMulAdd8(vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4,
                vreg_pse5, vreg_pse6, vreg_pse7, vreg_pse8,
                vreg_alibi1, vreg_alibi2, vreg_alibi3, vreg_alibi4,
                vreg_alibi5, vreg_alibi6, vreg_alibi7, vreg_alibi8,
                slopes, preg_all);
        }
    } else {
        if constexpr (IsSameType<T2, bfloat16_t>::value) {
            LoadCastPseBf16_8(vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4,
                vreg_pse5, vreg_pse6, vreg_pse7, vreg_pse8,
                pseUb, i, pseStride, preg_all_b16);
        } else if constexpr (IsSameType<T2, half>::value) {
            LoadCastPseF16_8(vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4,
                vreg_pse5, vreg_pse6, vreg_pse7, vreg_pse8,
                pseUb, i, pseStride, preg_all_b16);
        }
    }
}

template <PseTypeEnum pseMode>
__simd_callee__ inline void ScaleAndAddPse512(
    RegTensor<float> &x1, RegTensor<float> &x2,
    RegTensor<float> &x3, RegTensor<float> &x4,
    RegTensor<float> &x5, RegTensor<float> &x6,
    RegTensor<float> &x7, RegTensor<float> &x8,
    RegTensor<float> &p1, RegTensor<float> &p2,
    RegTensor<float> &p3, RegTensor<float> &p4,
    RegTensor<float> &p5, RegTensor<float> &p6,
    RegTensor<float> &p7, RegTensor<float> &p8,
    const float scale, MaskReg &preg_all,
    MaskReg &m1, MaskReg &m2, MaskReg &m3, MaskReg &m4)
{
    if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
        ScaleInput4x4(x1, x2, x3, x4, x5, x6, x7, x8,
            scale, preg_all, m1, m2, m3, m4);
    }
    if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
        AddPseToInput4x4(x1, x2, x3, x4, x5, x6, x7, x8,
            p1, p2, p3, p4, p5, p6, p7, p8,
            preg_all, m1, m2, m3, m4);
    }
    if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
        ScaleInput4x4(x1, x2, x3, x4, x5, x6, x7, x8,
            scale, preg_all, m1, m2, m3, m4);
    }
}

template <typename T>
__simd_callee__ inline void ProcessAtten512(
    RegTensor<float> &s1, RegTensor<float> &s2,
    RegTensor<float> &s3, RegTensor<float> &s4,
    RegTensor<float> &s5, RegTensor<float> &s6,
    RegTensor<float> &s7, RegTensor<float> &s8,
    RegTensor<float> &s5_new, RegTensor<float> &s6_new,
    RegTensor<float> &s7_new, RegTensor<float> &s8_new,
    RegTensor<float> &vreg_min,
    RegTensor<float> &x1, RegTensor<float> &x2,
    RegTensor<float> &x3, RegTensor<float> &x4,
    RegTensor<float> &x5, RegTensor<float> &x6,
    RegTensor<float> &x7, RegTensor<float> &x8,
    RegTensor<float> &vreg_input_max,
    MaskReg &c1, MaskReg &c2, MaskReg &c3, MaskReg &c4,
    MaskReg &c5, MaskReg &c6, MaskReg &c7, MaskReg &c8,
    __ubuf__ uint32_t *&mUb1, __ubuf__ uint32_t *&mUb2,
    __ubuf__ uint32_t *&mUb3, __ubuf__ uint32_t *&mUb4,
    __ubuf__ uint32_t *&mUb5, __ubuf__ uint32_t *&mUb6,
    __ubuf__ uint32_t *&mUb7, __ubuf__ uint32_t *&mUb8,
    const uint32_t nPadding, __ubuf__ T *&srcUb,
    const uint16_t i, const uint32_t s2BaseSize,
    MaskReg &preg_all,
    MaskReg &m1, MaskReg &m2, MaskReg &m3, MaskReg &m4)
{
    LoadAttenMask8(c1, c2, c3, c4, c5, c6, c7, c8,
        mUb1, mUb2, mUb3, mUb4, mUb5, mUb6, mUb7, mUb8, nPadding);

    ApplyAttenMaskSelect8(s1, s2, s3, s4, s5, s6, s7, s8,
        vreg_min,
        x1, x2, x3, x4, x5, x6, x7, x8,
        c1, c2, c3, c4, c5, c6, c7, c8);

    SelectOriTail4(s5_new, s6_new, s7_new, s8_new,
        s5, s6, s7, s8, vreg_min, m1, m2, m3, m4);

    StoreAlign8<T>(s1, s2, s3, s4, s5_new, s6_new, s7_new, s8_new,
        srcUb, i, s2BaseSize, preg_all);

    MaxReduce4(vreg_input_max,
        s1, s2, s3, s4, s5_new, s6_new, s7_new, s8_new, preg_all);
}

template <typename T>
__simd_callee__ inline void ProcessTail512(
    RegTensor<float> &x1, RegTensor<float> &x2,
    RegTensor<float> &x3, RegTensor<float> &x4,
    RegTensor<float> &x5, RegTensor<float> &x6,
    RegTensor<float> &x7, RegTensor<float> &x8,
    RegTensor<float> &x5_new, RegTensor<float> &x6_new,
    RegTensor<float> &x7_new, RegTensor<float> &x8_new,
    RegTensor<float> &vreg_min,
    RegTensor<float> &vreg_input_max,
    __ubuf__ T *&srcUb, const uint16_t i, const uint32_t s2BaseSize,
    MaskReg &preg_all,
    MaskReg &m1, MaskReg &m2, MaskReg &m3, MaskReg &m4)
{
    SelectOriTail4(x5_new, x6_new, x7_new, x8_new,
        x5, x6, x7, x8, vreg_min, m1, m2, m3, m4);

    StoreAlign8<T>(x1, x2, x3, x4, x5_new, x6_new, x7_new, x8_new,
        srcUb, i, s2BaseSize, preg_all);

    MaxReduce4(vreg_input_max,
        x1, x2, x3, x4, x5_new, x6_new, x7_new, x8_new, preg_all);
}

template <typename T, typename T2>
__simd_callee__ inline void ProcessSecondLoop512(
    RegTensor<float> &vreg_max,
    RegTensor<float> &x1, RegTensor<float> &x2,
    RegTensor<float> &x3, RegTensor<float> &x4,
    RegTensor<float> &x5, RegTensor<float> &x6,
    RegTensor<float> &x7, RegTensor<float> &x8,
    RegTensor<float> &exp_even1, RegTensor<float> &exp_odd1,
    RegTensor<float> &exp_even2, RegTensor<float> &exp_odd2,
    RegTensor<float> &exp_even3, RegTensor<float> &exp_odd3,
    RegTensor<float> &exp_even4, RegTensor<float> &exp_odd4,
    RegTensor<float> &vreg_exp_sum3,
    __ubuf__ T *maxSrc, __ubuf__ T *&srcUb,
    const uint16_t i, const uint32_t s2BaseSize,
    __ubuf__ T *&expSumUb, UnalignRegForStore &ureg_exp_sum,
    __ubuf__ T2 *&expUb1, __ubuf__ T2 *&expUb2,
    __ubuf__ T2 *&expUb3, __ubuf__ T2 *&expUb4,
    const uint32_t blockStride, const uint32_t repeatStride,
    MaskReg &preg_all, MaskReg &storeMask)
{
    LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(
        vreg_max, maxSrc);

    LoadInputDinterleave4<T>(x1, x2, x3, x4, x5, x6, x7, x8,
        srcUb, i, s2BaseSize);

    ExpSub8(exp_even1, exp_odd1, exp_even2, exp_odd2,
        exp_even3, exp_odd3, exp_even4, exp_odd4,
        x1, x2, x3, x4, x5, x6, x7, x8,
        vreg_max, preg_all);

    ExpSumReduceStore512<float, T2>(vreg_exp_sum3,
        exp_even1, exp_odd1, exp_even2, exp_odd2,
        exp_even3, exp_odd3, exp_even4, exp_odd4,
        ureg_exp_sum, expSumUb, preg_all);

    if constexpr (IsSameType<T2, bfloat16_t>::value) {
        CastStoreExpBf16_512<T2>(exp_even1, exp_odd1,
            exp_even2, exp_odd2, exp_even3, exp_odd3,
            exp_even4, exp_odd4,
            expUb1, expUb2, expUb3, expUb4,
            blockStride, repeatStride, preg_all, storeMask);
    } else if constexpr (IsSameType<T2, half>::value) {
        CastStoreExpF16_512<T2>(exp_even1, exp_odd1,
            exp_even2, exp_odd2, exp_even3, exp_odd3,
            exp_even4, exp_odd4,
            expUb1, expUb2, expUb3, expUb4,
            blockStride, repeatStride, preg_all, storeMask);
    }
}

template <typename T>
__simd_callee__ inline void MergeMaxUpdate512(
    RegTensor<float> &vreg_input_max,
    RegTensor<float> &vreg_max_new,
    RegTensor<float> &vreg_in_max,
    RegTensor<float> &vreg_input_max_tmp,
    __ubuf__ T *&tmpMaxUb, __ubuf__ T *&inMaxUb,
    __ubuf__ T *&tmpMaxUb2,
    UnalignRegForStore &ureg_max,
    MaskReg &preg_all)
{
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);
    LoadAlign(vreg_in_max, inMaxUb);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_input_max_tmp, tmpMaxUb2);
    Max(vreg_max_new, vreg_input_max_tmp, vreg_in_max, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <PseTypeEnum pseMode, typename T, typename T2>
__simd_callee__ inline void ProcessStage1Pse512(
    RegTensor<float> &vreg_input_x7, RegTensor<float> &vreg_input_x8,
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    RegTensor<float> &vreg_alibi7, RegTensor<float> &vreg_alibi8,
    const float scale, const float slopes,
    MaskReg &preg_all, MaskReg &preg_all_b16,
    __ubuf__ T *&srcUb, __ubuf__ T2 *pseUb,
    const uint16_t i, const uint32_t s2BaseSize, const uint32_t pseStride,
    MaskReg &preg_ori_tail_n3, MaskReg &preg_ori_tail_n4)
{
    LoadAlign(vreg_input_x7, srcUb + floatRepSize * 6 + i * s2BaseSize);
    LoadAlign(vreg_input_x8, srcUb + floatRepSize * 7 + i * s2BaseSize);

    if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
        Muls(vreg_input_x7, vreg_input_x7, scale, preg_ori_tail_n3);
        Muls(vreg_input_x8, vreg_input_x8, scale, preg_ori_tail_n4);
    }
    if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
        if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                      pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                ComputePseInnerMulAddSqrt2(vreg_pse7, vreg_pse8,
                    vreg_alibi7, vreg_alibi8, slopes, preg_all);
            } else {
                ComputePseInnerMulAdd2(vreg_pse7, vreg_pse8,
                    vreg_alibi7, vreg_alibi8, slopes, preg_all);
            }
        } else {
            if constexpr (IsSameType<T2, bfloat16_t>::value) {
                LoadCastPseBf16_2(vreg_pse7, vreg_pse8,
                    pseUb, floatRepSize * 6, i, pseStride, preg_all_b16);
            } else if constexpr (IsSameType<T2, half>::value) {
                LoadCastPseF16_2(vreg_pse7, vreg_pse8,
                    pseUb, floatRepSize * 6, i, pseStride, preg_all_b16);
            }
        }
        Add(vreg_input_x7, vreg_input_x7, vreg_pse7, preg_ori_tail_n3);
        Add(vreg_input_x8, vreg_input_x8, vreg_pse8, preg_ori_tail_n4);
    }
    if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
        Muls(vreg_input_x7, vreg_input_x7, scale, preg_ori_tail_n3);
        Muls(vreg_input_x8, vreg_input_x8, scale, preg_ori_tail_n4);
    }
}

template <PseTypeEnum pseMode, typename T, typename T2>
__simd_callee__ inline void ProcessStage2Pse512(
    RegTensor<float> &x1, RegTensor<float> &x2,
    RegTensor<float> &x3, RegTensor<float> &x4,
    RegTensor<float> &x5, RegTensor<float> &x6,
    RegTensor<float> &p1, RegTensor<float> &p2,
    RegTensor<float> &p3, RegTensor<float> &p4,
    RegTensor<float> &p5, RegTensor<float> &p6,
    RegTensor<float> &a1, RegTensor<float> &a2,
    RegTensor<float> &a3, RegTensor<float> &a4,
    RegTensor<float> &a5, RegTensor<float> &a6,
    const float scale, const float slopes,
    MaskReg &preg_all, MaskReg &preg_all_b16,
    __ubuf__ T *&srcUb, __ubuf__ T2 *pseUb,
    const uint16_t i, const uint32_t s2BaseSize, const uint32_t pseStride,
    MaskReg &m1, MaskReg &m2)
{
    LoadInput6<T>(x1, x2, x3, x4, x5, x6, srcUb, i, s2BaseSize);

    if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
        ScaleInput4x2(x1, x2, x3, x4, x5, x6,
            scale, preg_all, m1, m2);
    }
    if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
        if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                      pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                ComputePseInnerMulAddSqrt6(p1, p2, p3, p4, p5, p6,
                    a1, a2, a3, a4, a5, a6, slopes, preg_all);
            } else {
                ComputePseInnerMulAdd6(p1, p2, p3, p4, p5, p6,
                    a1, a2, a3, a4, a5, a6, slopes, preg_all);
            }
        } else {
            if constexpr (IsSameType<T2, bfloat16_t>::value) {
                LoadCastPseBf16_6(p1, p2, p3, p4, p5, p6,
                    pseUb, i, pseStride, preg_all_b16);
            } else if constexpr (IsSameType<T2, half>::value) {
                LoadCastPseF16_6(p1, p2, p3, p4, p5, p6,
                    pseUb, i, pseStride, preg_all_b16);
            }
        }
        AddPseToInput4x2(x1, x2, x3, x4, x5, x6,
            p1, p2, p3, p4, p5, p6, preg_all, m1, m2);
    }
    if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
        ScaleInput4x2(x1, x2, x3, x4, x5, x6,
            scale, preg_all, m1, m2);
    }
}

template <bool hasAtten, typename T>
__simd_callee__ inline void ProcessStage2AttenTail512(
    RegTensor<float> &x1, RegTensor<float> &x2,
    RegTensor<float> &x3, RegTensor<float> &x4,
    RegTensor<float> &x5, RegTensor<float> &x6,
    RegTensor<float> &s1, RegTensor<float> &s2,
    RegTensor<float> &s3, RegTensor<float> &s4,
    RegTensor<float> &s5, RegTensor<float> &s6,
    RegTensor<float> &s5_new, RegTensor<float> &s6_new,
    RegTensor<float> &s7_new, RegTensor<float> &s8_new,
    RegTensor<float> &x5_new, RegTensor<float> &x6_new,
    RegTensor<float> &x7_new, RegTensor<float> &x8_new,
    RegTensor<float> &vreg_min,
    RegTensor<float> &vreg_input_max,
    MaskReg &preg_all,
    MaskReg &m1, MaskReg &m2,
    MaskReg &c1, MaskReg &c2, MaskReg &c3,
    MaskReg &c4, MaskReg &c5, MaskReg &c6,
    __ubuf__ uint32_t *&mUb1, __ubuf__ uint32_t *&mUb2,
    __ubuf__ uint32_t *&mUb3, __ubuf__ uint32_t *&mUb4,
    __ubuf__ uint32_t *&mUb5, __ubuf__ uint32_t *&mUb6,
    const uint32_t nPadding,
    __ubuf__ T *&srcUb, const uint16_t i, const uint32_t s2BaseSize)
{
    if constexpr (hasAtten) {
        LoadAttenMask6(c1, c2, c3, c4, c5, c6,
            mUb1, mUb2, mUb3, mUb4, mUb5, mUb6, nPadding);
        ApplyAttenMaskSelect6(s1, s2, s3, s4, s5, s6,
            vreg_min, x1, x2, x3, x4, x5, x6,
            c1, c2, c3, c4, c5, c6);
        Select(s5_new, s5, vreg_min, m1);
        Select(s6_new, s6, vreg_min, m2);
        LoadAlign(s7_new, srcUb + floatRepSize * 6 + i * s2BaseSize);
        LoadAlign(s8_new, srcUb + floatRepSize * 7 + i * s2BaseSize);
        StoreAlign6<T>(s1, s2, s3, s4, s5_new, s6_new,
            srcUb, i, s2BaseSize, preg_all);
        MaxReduce4(vreg_input_max,
            s1, s2, s3, s4, s5_new, s6_new, s7_new, s8_new, preg_all);
    } else {
        Select(x5_new, x5, vreg_min, m1);
        Select(x6_new, x6, vreg_min, m2);
        LoadAlign(x7_new, srcUb + floatRepSize * 6 + i * s2BaseSize);
        LoadAlign(x8_new, srcUb + floatRepSize * 7 + i * s2BaseSize);
        StoreAlign6<T>(x1, x2, x3, x4, x5_new, x6_new,
            srcUb, i, s2BaseSize, preg_all);
        MaxReduce4(vreg_input_max,
            x1, x2, x3, x4, x5_new, x6_new, x7_new, x8_new, preg_all);
    }
}

} // namespace FaVectorApi

#endif // VF_BASIC_BLOCK_UNALIGNED512_COMMON_H
