/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_grouped_mat_mul.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../allto_allv_grouped_mat_mul_tiling.h"
#include "../allto_allv_grouped_mat_mul_tiling_key.h"
#if __has_include("../../allto_allv_quant_grouped_mat_mul/mc2_templates/mc2_templates.h")
#include "../../allto_allv_quant_grouped_mat_mul/mc2_templates/mc2_templates.h"
#else
#include "../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/mc2_templates.h"
#endif

using namespace AscendC;
using namespace MC2KernelTemplate;
using namespace Mc2GroupedMatmulTilingData;

#if defined(CONST_TILING)
#define GET_NESTED_TILING_DATA_MEMBER_ADDR(outerType, innerType, outerMember, innerMember, var, tiling) \
    const outerType *outerPtr##var = (const outerType *)(tiling);                                       \
    const innerType *innerPtr##var = &(outerPtr##var->outerMember);                                     \
    const int32_t *(var) = (const int32_t)((const uint8_t *)&(innerPtr##var->innerMember));
#else
#define GET_NESTED_TILING_DATA_MEMBER_ADDR(outerType, innerType, outerMember, innerMember, var, tiling) \
    size_t outerOffset##var = (size_t)(&((outerType *)0)->outerMember);                                 \
    size_t innerOffset##var = (size_t)(&((innerType *)0)->innerMember);                                 \
    __gm__ int32_t *(var) = (__gm__ int32_t *)((__gm__ uint8_t *)(tiling) + outerOffset##var + innerOffset##var);
#endif

#if defined(CONST_TILING)
#define TILING_TYPE const int32_t
#else
#define TILING_TYPE __gm__ int32_t
#endif

template <
    bool TILINGKEY_GMM_WEIGHT_TRANSPOSE,
    bool TILINGKEY_MM_WEIGHT_TRANSPOSE,
    int TILINGKEY_COMM_MODE>
__global__ __aicore__ void allto_allv_grouped_mat_mul(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM,
    GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM,
    GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR userWorkspace = GetUserWorkspace(workspaceGM);
    TPipe pipe;
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();

    REGISTER_TILING_DEFAULT(AlltoAllvGmmTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    if (!tilingData.isFp16) {
        using ComputeOpType = QuantGroupedMatmul<AlltoAllvGmmTilingData, GMMQuantTilingData, bfloat16_t,
            bfloat16_t, float, bfloat16_t, CubeFormat::ND, false, TILINGKEY_GMM_WEIGHT_TRANSPOSE,
            false, true>;
        using LocalComputeOpType = QuantGroupedMatmul<AlltoAllvGmmTilingData, GMMQuantTilingData, bfloat16_t,
            bfloat16_t, float, bfloat16_t, CubeFormat::ND, false, TILINGKEY_MM_WEIGHT_TRANSPOSE,
            true, true>;
        A2avGmmScheduler<HcclA2avOp<bfloat16_t, true, TILINGKEY_COMM_MODE>, ComputeOpType, LocalComputeOpType,
            AlltoAllvGmmTilingData, GMMQuantTilingData, TILING_TYPE> a2avGmmScheduler;

        GET_NESTED_TILING_DATA_MEMBER_ADDR(AlltoAllvGmmTilingData, GMMQuantTilingData, gmmQuantTilingData,
            gmmArray, gmmArrayAddr_, tilingGM);
        GET_NESTED_TILING_DATA_MEMBER_ADDR(AlltoAllvGmmTilingData, GMMQuantTilingData, mmQuantTilingData,
            gmmArray, mmArrayAddr_, tilingGM);

        a2avGmmScheduler.Init(gmmxGM, gmmweightGM, mmxOptionalGM, mmweightOptionalGM,
            nullptr, nullptr, nullptr, nullptr,
            gmmyGM, mmyOptionalGM, permuteOutOptionalGM, userWorkspace,
            tilingGM, gmmArrayAddr_, mmArrayAddr_, &pipe, true);
        a2avGmmScheduler.Process();
        return;
    }

    using ComputeOpType = QuantGroupedMatmul<AlltoAllvGmmTilingData, GMMQuantTilingData, half,
        half, float, half, CubeFormat::ND, false, TILINGKEY_GMM_WEIGHT_TRANSPOSE,
        false, true>;
    using LocalComputeOpType = QuantGroupedMatmul<AlltoAllvGmmTilingData, GMMQuantTilingData, half,
        half, float, half, CubeFormat::ND, false, TILINGKEY_MM_WEIGHT_TRANSPOSE,
        true, true>;
    A2avGmmScheduler<HcclA2avOp<half, true, TILINGKEY_COMM_MODE>, ComputeOpType, LocalComputeOpType,
        AlltoAllvGmmTilingData, GMMQuantTilingData, TILING_TYPE> a2avGmmScheduler;

    GET_NESTED_TILING_DATA_MEMBER_ADDR(AlltoAllvGmmTilingData, GMMQuantTilingData, gmmQuantTilingData,
        gmmArray, gmmArrayAddr_, tilingGM);
    GET_NESTED_TILING_DATA_MEMBER_ADDR(AlltoAllvGmmTilingData, GMMQuantTilingData, mmQuantTilingData,
        gmmArray, mmArrayAddr_, tilingGM);

    a2avGmmScheduler.Init(gmmxGM, gmmweightGM, mmxOptionalGM, mmweightOptionalGM, nullptr, nullptr, nullptr, nullptr,
        gmmyGM, mmyOptionalGM, permuteOutOptionalGM, userWorkspace, tilingGM, gmmArrayAddr_, mmArrayAddr_, &pipe, true);
    a2avGmmScheduler.Process();
}