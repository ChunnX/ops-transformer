/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fag_custom_matmul_policy_s1s2_bn2gs1s2_sab.h
 * \brief
 */

#ifndef FAG_CUSTOM_MATMUL_POLICY_S1S2_BN2GS1S2_SAB_KERNEL_H_
#define FAG_CUSTOM_MATMUL_POLICY_S1S2_BN2GS1S2_SAB_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../fag_data_common.h"
#include "../copy_in_buffer/fag_copy_in_buffer_512_512_64.h"
#include "../copy_in_buffer/fag_copy_in_buffer_base.h"
#include "../copy_cube_in/fag_copy_cube_in_base.h"
#include "../copy_cube_in/fag_s1s2_bn2gs1s2_sab/fag_copy_cube_in_mm_dq.h"
#include "../copy_cube_in/fag_s1s2_bn2gs1s2_sab/fag_copy_cube_in_mm_dkv.h"
#include "../copy_cube_in/fag_s1s2_bn2gs1s2_sab/fag_copy_cube_in_mm_512_512_192.h"
#include "../copy_cube_in/fag_s1s2_bn2gs1s2_sab/fag_copy_cube_in_mm_512_512_64.h"
namespace AscendC {
    
namespace Impl {
namespace Detail {

// ---------------------------------------------------policys-----------------------------------------------------------

// ---------------------------------------------------D<=128 singleS1/S2=512--------------------------------------------
/*
 * support: D<=128; singleS1/S2=512
 * limit: mm-dk/dv matmul baseM/N/K相同
 */
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyMM12 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using CopyCubeInA = FagCopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeIn<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContext<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyMMDq : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using CopyCubeInA = FagCopyCubeInMMDq<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInMMDq<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContext<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyMMDkv : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using CopyCubeInA = FagCopyCubeInMMDkv<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInMMDkv<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContext<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

// -------------------------------------------D=64---------------------------------------------------------------------

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyHeadDim64 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using CopyCubeInA = FagCopyCubeInHeadDim64<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInHeadDim64<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBufferHeadDim64<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBufferHeadDim64<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContext<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

// -------------------------------------------D=192---------------------------------------------------------------------
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyMM1HeadDim192 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using CopyCubeInA = FagCopyCubeInMM1HeadDim192<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInMM1HeadDim192<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContext<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicyMM2HeadDim192 : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using CopyCubeInA = FagCopyCubeInMM2HeadDim192<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FagCopyCubeInMM2HeadDim192<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContext<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

// ------------------------------------------------selectors------------------------------------------------------------
/////////////////////////////////////////////////////
// mm-dyv/qk L1 custom policy selector
/////////////////////////////////////////////////////
template <DTemplateType dTemplateType> struct MM12ConstPolicySelector {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

template <> struct MM12ConstPolicySelector<DTemplateType::Aligned128> {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyMM12<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

template <> struct MM12ConstPolicySelector<DTemplateType::Aligned192> {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyMM1HeadDim192<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

template <> struct MM12ConstPolicySelector<DTemplateType::Aligned64> {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyHeadDim64<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

/////////////////////////////////////////////////////
// mm-dq L1 custom policy selector
/////////////////////////////////////////////////////
template <DTemplateType dTemplateType> struct MMDqConstPolicySelector {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

template <> struct MMDqConstPolicySelector<DTemplateType::Aligned128> {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyMMDq<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

template <> struct MMDqConstPolicySelector<DTemplateType::Aligned192> {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyMM2HeadDim192<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

template <> struct MMDqConstPolicySelector<DTemplateType::Aligned64> {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyHeadDim64<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};
/////////////////////////////////////////////////////
// mm-dkv L1 custom policy selector
/////////////////////////////////////////////////////
template <DTemplateType dTemplateType> struct MMDkvConstPolicySelector {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

template <> struct MMDkvConstPolicySelector<DTemplateType::Aligned128> {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyMMDkv<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

template <> struct MMDkvConstPolicySelector<DTemplateType::Aligned192> {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyMM2HeadDim192<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};

template <> struct MMDkvConstPolicySelector<DTemplateType::Aligned64> {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FagMatmulPolicyHeadDim64<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
};
}
}
}
#endif // FAG_CUSTOM_MATMUL_POLICY_S1S2_BN2GS1S2_SAB_KERNEL_H_
