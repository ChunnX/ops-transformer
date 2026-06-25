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
#include "allto_allv_grouped_mat_mul_tiling.h"
#include "allto_allv_grouped_mat_mul_tiling_key.h"
#if __has_include("../allto_allv_quant_grouped_mat_mul/mc2_templates/mc2_templates.h")
#include "../allto_allv_quant_grouped_mat_mul/mc2_templates/mc2_templates.h"
#else
#include "../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/mc2_templates.h"
#endif

using namespace AscendC;
using namespace MC2KernelTemplate;
using namespace Mc2GroupedMatmulTilingData;

#if defined(CONST_TILING)
#define GET_NESTED_TILING_DATA_MEMBER_ADDR(outerType, innerType, outerMember, innerMember, var, tiling)                \
    const outerType *outerPtr##var = (const outerType *)(tiling);                                                      \
    const innerType *innerPtr##var = &(outerPtr##var->outerPtr##var);                                                  \
    const int32_t *(var) = (const int32_t *)((const uint8_t *)&(innerPtr##var->innerMember))
#else
#define GET_NESTED_TILING_DATA_MEMBER_ADDR(outerType, innerType, outerMember, innerMember, var, tiling)                \
    size_t outerOffset##var = (size_t)(&((outerType *)0)->outerMember);                                                \
    size_t innerOffset##var = (size_t)(&((innerType *)0)->innerMember);                                                \
    __gm__ int32_t *(var) = (__gm__ int32_t *)((__gm__ uint8_t *)(tiling) + outerOffset##var + innerOffset##var)
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
    if (workspaceGM == nullptr) {
        return;
    }
    SetSysWorkspace(workspaceGM);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspaceGM);
    if (userWorkspace == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    REGISTER_TILING_DEFAULT(AlltoAllvGmmTilingData);
    // tilingData
    GET_TILING_DATA(tilingData, tilingGM);
    const AlltoAllvGmmTilingData *tilingData_ = &tilingData;
    const void *hcclInitTiling = &(tilingData_->hcclA2avTilingInfo.hcclInitTiling);
    uint64_t hcclCcTilingOffset = offsetof(AlltoAllvGmmTilingData, hcclA2avTilingInfo) +
                                  offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling);
    constexpr CubeFormat W_FORMAT = CubeFormat::ND;
    // workspace
    uint64_t workspaceOffset = 0;
    bool isNeedPermute = tilingData_->taskTilingInfo.expertNum > 1;
    GM_ADDR commXOutGm = workspaceGM + workspaceOffset;
    commXOutGm = (!isNeedPermute && tilingData_->isPermuteOut) ?
        permuteOutOptionalGM : commXOutGm;
    workspaceOffset += tilingData_->workspaceInfo.commXOutSize;
    GM_ADDR gmmXPermuteGM = workspaceGM + workspaceOffset;
    gmmXPermuteGM = (!tilingData_->isPermuteOut && isNeedPermute) ?
        gmmXPermuteGM : permuteOutOptionalGM;
    workspaceOffset += tilingData_->workspaceInfo.permuteXOutSize;
    GM_ADDR gmmXComputeGM = workspaceGM + workspaceOffset;
    GM_ADDR computeGmmXGM = commXOutGm;
    // A2avGmmScheduler
    using HcclOpType = HcclA2avOp<DTYPE_GMM_X, true, TILINGKEY_COMM_MODE>;
    using ComputeOpType = QuantGroupedMatmul<AlltoAllvGmmTilingData, GMMQuantTilingData, DTYPE_GMM_X,
        DTYPE_GMM_WEIGHT, float, DTYPE_GMM_Y, CubeFormat::ND, false, TILINGKEY_GMM_WEIGHT_TRANSPOSE,
        false, true>;
    using LocalComputeOpType = QuantGroupedMatmul<AlltoAllvGmmTilingData, GMMQuantTilingData, DTYPE_MM_X,
        DTYPE_MM_WEIGHT, float, DTYPE_MM_Y, CubeFormat::ND, false, TILINGKEY_MM_WEIGHT_TRANSPOSE,
        true, true>;
    using A2avGmmSchedulerType = A2avGmmScheduler<HcclOpType, ComputeOpType, LocalComputeOpType>;
    // comm
    HcclOpType commOp;
    commOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData_->taskTilingInfo, gmmxGM,
        commXOutGm, gmmXPermuteGM, static_cast<uint32_t>(tilingData_->taskTilingInfo.aivCoreNum));
    computeGmmXGM = isNeedPermute ? gmmXPermuteGM : commXOutGm;
    // compute
    GET_NESTED_TILING_DATA_MEMBER_ADDR(AlltoAllvGmmTilingData, GMMQuantTilingData, gmmQuantTilingData,
        gmmArray, gmmArrayAddr_, tilingGM);
    GET_NESTED_TILING_DATA_MEMBER_ADDR(AlltoAllvGmmTilingData, GMMQuantTilingData, mmQuantTilingData,
        gmmArray, mmArrayAddr_, tilingGM);
    ComputeOpType computeOp;
    computeOp.Init(computeGmmXGM, gmmweightGM, nullptr, nullptr, gmmyGM, gmmXComputeGM,
                   tilingData_, &tilingData_->gmmQuantTilingData, gmmArrayAddr_, &pipe, true);
    LocalComputeOpType localComputeOp;
    localComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, nullptr, nullptr, mmyOptionalGM, gmmXComputeGM,
                    tilingData_, &tilingData_->mmQuantTilingData, mmArrayAddr_, &pipe, true);
    // scheduler
    A2avGmmSchedulerType a2avGmmScheduler(commOp, computeOp, localComputeOp, &tilingData_->taskTilingInfo,
                                        tilingData_->isNeedMM);
    a2avGmmScheduler.Process();
}