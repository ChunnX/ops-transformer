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
 * \brief AlltoAllV Grouped MatMul kernel entry - debug: only shared expert (localComputeOp)
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
    const innerType *innerPtr##var = &(outerPtr##var->outerPtr##var);                                   \
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
    if (userWorkspace == nullptr) {
        return;
    }

    TPipe pipe;
    REGISTER_TILING_DEFAULT(AlltoAllvGmmTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    const AlltoAllvGmmTilingData *tilingData_ = &tilingData;

    AscendC::printf("[LBH-V4] entry blk=%d needMM=%d A=%llu H1=%llu N2=%llu BS=%llu e=%d\n",
        GetBlockIdx(), tilingData_->isNeedMM ? 1 : 0,
        tilingData_->taskTilingInfo.A, tilingData_->taskTilingInfo.H1,
        tilingData_->taskTilingInfo.N2, tilingData_->taskTilingInfo.BS,
        tilingData_->taskTilingInfo.e);

    // === Shared Expert (local MM) only - no HCCL, no routed expert ===
    using SharedGmmExpertOpType = QuantGroupedMatmul<AlltoAllvGmmTilingData, GMMQuantTilingData, DTYPE_GMM_X,
        DTYPE_GMM_WEIGHT, float, DTYPE_GMM_Y, CubeFormat::ND, false, TILINGKEY_MM_WEIGHT_TRANSPOSE, true, true>;

    GET_NESTED_TILING_DATA_MEMBER_ADDR(AlltoAllvGmmTilingData, GMMQuantTilingData, mmQuantTilingData,
        gmmArray, mmArrayAddr_, tilingGM);

    // workspace layout: groupList + ptrTable + compute workspace (shared area)
    uint64_t workspaceOffset = 0;
    if (!tilingData_->isPermuteOut) {
        uint64_t commOutLen = Align(CeilDiv(tilingData_->taskTilingInfo.A * tilingData_->taskTilingInfo.H1, PACK_FACTOR)
            * X_TYPE_SIZE, TENSOR_LIST_SIZE);
        workspaceOffset += commOutLen;
    }
    GM_ADDR computeWorkspaceGm = userWorkspace + workspaceOffset;

    SharedGmmExpertOpType shareComputeOp;
    if (tilingData_->isNeedMM != 0) {
        shareComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, nullptr, nullptr, mmyOptionalGM,
                            computeWorkspaceGm, computeWorkspaceGm, tilingData_, &tilingData_->mmQuantTilingData, mmArrayAddr_, &pipe,
                            false);
        shareComputeOp.Process(0);
        shareComputeOp.End();
    }
}