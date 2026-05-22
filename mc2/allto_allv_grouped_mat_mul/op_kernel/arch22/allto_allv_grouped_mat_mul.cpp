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
 * \brief AlltoAllV Grouped MatMul kernel entry - 复用quant版mc2_templates
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../allto_allv_grouped_mat_mul_tiling.h"
#include "../allto_allv_grouped_mat_mul_tiling_key.h"
#if __has_include("../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/mc2_templates.h")
#include "../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/mc2_templates.h"
#elif __has_include("../allto_allv_quant_grouped_mat_mul/mc2_templates/mc2_templates.h")
#include "../allto_allv_quant_grouped_mat_mul/mc2_templates/mc2_templates.h"
#else
#include "../../allto_allv_quant_grouped_mat_mul/mc2_templates/mc2_templates.h"
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
    AscendC::printf("[LBH] allto_allv_grouped_mat_mul entry blockIdx=%d\n", GetBlockIdx());
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR userWorkspace = GetUserWorkspace(workspaceGM);
    if (userWorkspace == nullptr) {
        return;
    }

    TPipe pipe;
    REGISTER_TILING_DEFAULT(AlltoAllvGmmTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    const AlltoAllvGmmTilingData *tilingData_ = &tilingData;

    AscendC::printf("[LBH-DIAG] tiling: isNeedMM=%d isPermuteOut=%d A=%llu H1=%llu H2=%llu N1=%llu N2=%llu BS=%llu e=%d\n",
        tilingData_->isNeedMM ? 1 : 0, tilingData_->isPermuteOut ? 1 : 0,
        tilingData_->taskTilingInfo.A, tilingData_->taskTilingInfo.H1, tilingData_->taskTilingInfo.H2,
        tilingData_->taskTilingInfo.N1, tilingData_->taskTilingInfo.N2,
        tilingData_->taskTilingInfo.BS, tilingData_->taskTilingInfo.e);

    // Compute workspace layout
    // workspace布局：[permuteOut区域] + [groupList + ptrTable + gmmASWKernel workspace]
    // groupList和workspace共享同一片GM区域：groupList在开头，ptrTable紧跟其后，剩余给gmmASWKernel内部缓冲
    uint64_t workspaceOffset = 0;
    if (!tilingData_->isPermuteOut) {
        uint64_t commOutLen = Align(CeilDiv(tilingData_->taskTilingInfo.A * tilingData_->taskTilingInfo.H1, PACK_FACTOR)
            * X_TYPE_SIZE, TENSOR_LIST_SIZE);
        workspaceOffset += commOutLen;
    }
    GM_ADDR computeWorkspaceGm = userWorkspace + workspaceOffset;
    AscendC::printf("[LBH-DIAG] workspace: userWorkspace=%llu workspaceOffset=%llu computeWorkspaceGm=%llu mmx=%llu mmw=%llu mmy=%llu\n",
        reinterpret_cast<uint64_t>(userWorkspace), workspaceOffset,
        reinterpret_cast<uint64_t>(computeWorkspaceGm),
        reinterpret_cast<uint64_t>(mmxOptionalGM), reinterpret_cast<uint64_t>(mmweightOptionalGM),
        reinterpret_cast<uint64_t>(mmyOptionalGM));
    // groupListGM和workspaceGM共享同一片区域（groupList在开头，ptrTable紧跟，剩余给gmmASWKernel）
    GM_ADDR gmmGroupListGm = computeWorkspaceGm;
    GM_ADDR gmmWorkspaceGm = computeWorkspaceGm;
    // localComputeOp和computeOp串行执行，共享同一份groupList/workspace区域
    GM_ADDR mmGroupListGm = computeWorkspaceGm;
    GM_ADDR mmWorkspaceGm = computeWorkspaceGm;

    // === Shared Expert (local MM) ===
    using SharedGmmExpertOpType = QuantGroupedMatmul<AlltoAllvGmmTilingData, GMMQuantTilingData, DTYPE_GMM_X,
        DTYPE_GMM_WEIGHT, float, DTYPE_GMM_Y, CubeFormat::ND, false, TILINGKEY_MM_WEIGHT_TRANSPOSE, true, true>;

    GET_NESTED_TILING_DATA_MEMBER_ADDR(AlltoAllvGmmTilingData, GMMQuantTilingData, mmQuantTilingData,
        gmmArray, mmArrayAddr_, tilingGM);

    SharedGmmExpertOpType shareComputeOp;
    if (tilingData_->isNeedMM != 0) {
        shareComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, nullptr, nullptr, mmyOptionalGM,
                            mmGroupListGm, mmWorkspaceGm, tilingData_, &tilingData_->mmQuantTilingData, mmArrayAddr_, &pipe,
                            false);
        shareComputeOp.Process(0);
    }

    // === Routed Expert (GMM compute + communication) ===
    // TODO: 启用通信和路由专家计算时，使用GmmA2avScheduler编排通信+计算流水线
    // using HcclOpType = HcclA2avOp<DTYPE_GMM_X, true, TILINGKEY_COMM_MODE>;
    // using ComputeOpType = QuantGroupedMatmul<AlltoAllvGmmTilingData, GMMQuantTilingData, DTYPE_GMM_X,
    //     DTYPE_GMM_WEIGHT, float, DTYPE_GMM_Y, CubeFormat::ND, false, TILINGKEY_GMM_WEIGHT_TRANSPOSE, false, true>;
    // using GmmA2avSchedulerType = GmmA2avScheduler<HcclOpType, ComputeOpType, SharedGmmExpertOpType,
    //     TILINGKEY_COMPUTE_MATMUL>;
    // HcclOpType hcclOp;
    // hcclOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData.taskTilingInfo, gmmxGM, commOutGm);
    // ComputeOpType computeOp;
    // GET_NESTED_TILING_DATA_MEMBER_ADDR(AlltoAllvGmmTilingData, GMMQuantTilingData, gmmQuantTilingData,
    //     gmmArray, gmmArrayAddr_, tilingGM);
    // computeOp.Init(commOutGm, gmmweightGM, scalePlaceholderGM, scalePlaceholderGM, gmmyGM,
    //     gmmGroupListGm, gmmWorkspaceGm, tilingData_, &tilingData_->gmmQuantTilingData, gmmArrayAddr_, &pipe, false);
    // GmmA2avSchedulerType gmmA2avScheduler(hcclOp, computeOp, shareComputeOp, &tilingData.taskTilingInfo);
    // gmmA2avScheduler.Process();

    if (tilingData_->isNeedMM != 0) {
        shareComputeOp.End();
    }
}