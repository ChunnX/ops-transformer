/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file quant_grouped_mat_mul_allto_allv.cpp
 * \brief
 */
#include "basic_api/kernel_basic_intf.h"
#include "quant_grouped_mat_mul_allto_allv_tiling.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_key.h"

#include "../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/mc2_templates.h"
#include "../../common/op_kernel/apace/kernel/fusions/quant_grouped_mat_mul_allto_allv/quant_grouped_mat_mul_allto_allv_impl.h"
#include "../../common/op_kernel/apace/kernel/fusions/quant_grouped_mat_mul_allto_allv/quant_grouped_mat_mul_allto_allv_hcomm_impl.h"

using namespace AscendC;
using namespace MC2KernelTemplate;
using namespace Mc2GroupedMatmulTilingData;

template <typename X_T, const bool IS_OPT_MM, const bool IS_GMM_WEIGHT_TRANS, const bool IS_OPT_WEIGHT_TRANS>
struct GMMATAVType { // Grouped_Mat_Mul_All_To_Allv_Type
    using xType = X_T;
    static constexpr bool isOptionalMm = IS_OPT_MM;
    static constexpr bool isGmmWeightTrans = IS_GMM_WEIGHT_TRANS;
    static constexpr bool isOptWeightTrans = IS_OPT_WEIGHT_TRANS;
};

template <bool TILINGKEY_COMPUTE_MATMUL, bool TILINGKEY_GROUPED_MATMUL_TRANS, bool TILINGKEY_MATMUL_TRANS,
          int TILINGKEY_COMM_MODE>
__global__ __aicore__ void quant_grouped_mat_mul_allto_allv(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM,
    GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM,
    GM_ADDR mmweightOptionalGM, GM_ADDR mmxScaleGM, GM_ADDR mmWeightScaleGM, GM_ADDR commQuantScaleGM, GM_ADDR yGM,
    GM_ADDR mmyOptionalGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspaceGM == nullptr) {
        return;
    }
    SetSysWorkspace(workspaceGM);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspaceGM);
    if (userWorkspace == nullptr) {
        return;
    }

    TPipe pipe;
    REGISTER_TILING_DEFAULT(QuantGmmA2avTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    const QuantGmmA2avTilingData *tilingData_ = &tilingData;

    // workspace布局 (后通信模式):
    //   [0, wsGmmOutputSize)                 → 路由专家 GMM 输出 (每轮从偏移0开始写入)
    //   [+, wsGmmComputeWorkspaceSize)       → GmmComputeOp 内部空间
    //   [+, wsSharedGmmComputeWorkspaceSize) → SharedGmmComputeOp 内部空间
    uint64_t wsOffset = 0;
    GM_ADDR gmmOutputGM = userWorkspace + wsOffset;
    wsOffset += tilingData_->workspaceInfo.wsGmmOutputSize;
    GM_ADDR gmmComputeWSGM = userWorkspace + wsOffset;
    wsOffset += tilingData_->workspaceInfo.wsGmmComputeWorkspaceSize;
    GM_ADDR sharedGmmComputeWSGM = userWorkspace + wsOffset;
    wsOffset += tilingData_->workspaceInfo.wsSharedGmmComputeWorkspaceSize;

#if MX_QUANT_MODE
    // MX(FP4/FP8) 量化路径：切换 apace 模板 GroupedMatmulMxKernel 计算 GMM，
    // 通信采用 AIC 核内 AlltoAllV（参考 PR 10588 hcomm 通算融合模式），
    // 规避 Blaze 与 AIV/AICPU KFC 通信引擎的兼容问题。
    using SharedGmmExpertOpType =
        Apace::QuantGroupedMatmulMx<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT,
                                    DTYPE_MM_X_SCALE, DTYPE_Y, CubeFormat::ND, false, TILINGKEY_MATMUL_TRANS, true,
                                    false>;
    using HcommImpl =
        Apace::QuantGmmA2avHcommImpl<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT,
                                     DTYPE_GMM_X_SCALE, DTYPE_Y, CubeFormat::ND, false,
                                     TILINGKEY_GROUPED_MATMUL_TRANS, false, TILINGKEY_COMM_MODE>;
    if ASCEND_IS_AIC {
        GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, gmmBaseTiling, gmmArray,
                                           gmmArrayAddr_, tilingGM);
        HcommImpl hcommImpl;
        hcommImpl.Init(gmmxGM, gmmweightGM, gmmxScaleGM, gmmWeightScaleGM, gmmOutputGM, yGM, gmmComputeWSGM, tilingData_,
                       &tilingData_->gmmBaseTiling, gmmArrayAddr_, &pipe);
        uint32_t e = tilingData_->taskTilingInfo.e;
        uint32_t expertNum = tilingData_->taskTilingInfo.expertNum;
        for (uint32_t start = 0; start < e; start += expertNum) {
            uint32_t actualExpertNum = (start + expertNum > e) ? (e - start) : expertNum;
            hcommImpl.Process(start, actualExpertNum);
        }
        hcommImpl.End();
        // 共享专家（独立输出，不走通信）
        if constexpr (TILINGKEY_COMPUTE_MATMUL) {
            GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, sharedGmmTiling, gmmArray,
                                               mmArrayAddr_, tilingGM);
            SharedGmmExpertOpType shareComputeOp;
            shareComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
                                sharedGmmComputeWSGM, tilingData_, &tilingData_->sharedGmmTiling, mmArrayAddr_, &pipe);
            shareComputeOp.Process(0, 1);
        }
    }
    if ASCEND_IS_AIV {
        // 通信在 AIC 核完成，AIV 不参与
    }
#else
    using HcclOpType = HcclA2avOp<DTYPE_Y, false, TILINGKEY_COMM_MODE>;
    using ComputeOpType =
        QuantGroupedMatmul<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT, DTYPE_GMM_X_SCALE,
                           DTYPE_Y, CubeFormat::ND, false, TILINGKEY_GROUPED_MATMUL_TRANS, false, false>;
    using SharedGmmExpertOpType =
        QuantGroupedMatmul<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT, DTYPE_MM_X_SCALE,
                           DTYPE_Y, CubeFormat::ND, false, TILINGKEY_MATMUL_TRANS, true, false>;
    using GmmA2avSchedulerType =
        GmmA2avScheduler<HcclOpType, ComputeOpType, SharedGmmExpertOpType, TILINGKEY_COMPUTE_MATMUL>;
    const void *hcclInitTiling = &(tilingData_->hcclA2avTiling.hcclInitTiling);
    uint64_t hcclCcTilingOffset = offsetof(QuantGmmA2avTilingData, hcclA2avTiling) +
                                  offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling);
    // hccl: sendBuffer=GMM输出缓冲 (rank-first布局, 无需permute重排)
    HcclOpType hcclOp;
    hcclOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData.taskTilingInfo, gmmOutputGM, yGM,
                static_cast<uint32_t>(tilingData_->taskTilingInfo.aivCoreNum));
    if ASCEND_IS_AIC {
        GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, gmmBaseTiling, gmmArray,
                                           gmmArrayAddr_, tilingGM);
        ComputeOpType computeOp;
        computeOp.Init(gmmxGM, gmmweightGM, gmmxScaleGM, gmmWeightScaleGM, gmmOutputGM, gmmComputeWSGM, tilingData_,
                       &tilingData_->gmmBaseTiling, gmmArrayAddr_, &pipe);
        GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, sharedGmmTiling, gmmArray,
                                           mmArrayAddr_, tilingGM);
        SharedGmmExpertOpType shareComputeOp;
        shareComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
                            sharedGmmComputeWSGM, tilingData_, &tilingData_->sharedGmmTiling, mmArrayAddr_, &pipe);
        GmmA2avSchedulerType::ProcessAic(computeOp, shareComputeOp, &tilingData.taskTilingInfo);
    }
    if ASCEND_IS_AIV {
        GmmA2avSchedulerType::ProcessAiv(hcclOp, &tilingData.taskTilingInfo);
    }
#endif
}
