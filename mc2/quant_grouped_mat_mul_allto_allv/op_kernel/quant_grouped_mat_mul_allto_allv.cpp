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

// apace hcomm 融合通路（MX 量化场景，tiling key 选路）：
// MX tiling 类编码 TILINGKEY_USING_APACE_IMPL=1，GMM 计算切换 apace GroupedMatmulMxKernel，
// AllToAllV 通信在 AIC 核内经 Hccl 提交（Blaze 与 AIV 侧 KFC 通信引擎同 kernel 共存会致
// AIV 通信任务参数校验失败）。该组合经 tiling key 的 ASCENDC_TPL_KERNEL_TYPE_SEL 声明
// AIC_ONLY 任务类型（无 AIV 参与，无需核型守卫）；TT 量化实例由 TT tiling 类编码 USING=0，
// 走原通路（MIX_AIC_1_2，AIC 计算 + AIV 通信）。任务类型全部由 TPL SEL 按组合声明，
// 不再使用 KERNEL_TASK_TYPE_DEFAULT（编译单元级机制，无法按组合区分）。
#include "../../common/op_kernel/apace/kernel/fusions/quant_grouped_matmul_all_to_all_v/quant_grouped_matmul_all_to_all_v_hcomm_impl.h"

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
          int TILINGKEY_COMM_MODE, bool TILINGKEY_USING_APACE_IMPL>
__global__ __aicore__ void quant_grouped_mat_mul_allto_allv(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM,
    GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM,
    GM_ADDR mmweightOptionalGM, GM_ADDR mmxScaleGM, GM_ADDR mmWeightScaleGM, GM_ADDR commQuantScaleGM, GM_ADDR yGM,
    GM_ADDR mmyOptionalGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
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
    if constexpr (TILINGKEY_USING_APACE_IMPL) {
#if MX_QUANT_MODE
        // MX(FP4/FP8) 量化路径（AIC_ONLY 任务类型由 tiling key KERNEL_TYPE_SEL 声明，
        // 无 AIV 参与，无需核型守卫）：GMM 计算切换 apace 模板 GroupedMatmulMxKernel，
        // 通信采用 AIC 核内 AllToAllV（Hccl），规避 Blaze 与 AIV 通信引擎的共存问题
        using SharedGmmExpertOpType =
            Apace::QuantGroupedMatmulMx<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT,
                                        DTYPE_MM_X_SCALE, DTYPE_Y, CubeFormat::ND, false, TILINGKEY_MATMUL_TRANS, true,
                                        false>;
        using HcommImpl =
            Apace::QuantGmmA2avHcommImpl<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT,
                                         DTYPE_GMM_X_SCALE, DTYPE_Y, CubeFormat::ND, false,
                                         TILINGKEY_GROUPED_MATMUL_TRANS, false, TILINGKEY_COMM_MODE>;
        // 部署自检：block0 先写 y[0]=0xA5A5A5A5，可判定"新二进制在跑 + apace 分支生效"
        // （通信成功后该值被 recv 数据覆盖，不影响结果）
        if (AscendC::GetBlockIdx() == 0) {
            *reinterpret_cast<__gm__ uint32_t *>(yGM) = 0xA5A5A5A5U;
        }
        GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, gmmBaseTiling, gmmArray,
                                           gmmArrayAddr_, tilingGM);
        HcommImpl hcommImpl;
        hcommImpl.Init(gmmxGM, gmmweightGM, gmmxScaleGM, gmmWeightScaleGM, gmmOutputGM, yGM, gmmComputeWSGM,
                       tilingData_, &tilingData_->gmmBaseTiling, gmmArrayAddr_, &pipe);
        const uint32_t e = static_cast<uint32_t>(tilingData_->taskTilingInfo.e);
        const uint32_t expertNum = tilingData_->taskTilingInfo.expertNum;
        for (uint32_t start = 0; start < e; start += expertNum) {
            const uint32_t actualExpertNum = (start + expertNum > e) ? (e - start) : expertNum;
            hcommImpl.Process(start, actualExpertNum);
        }
        // 共享专家（独立输出，不走通信）：必须在 Hccl Finalize 之前执行——全部已验证
        // 形态的 Blaze 计算均位于 Finalize 之前，且与末批 AllToAllV 通信重叠
        // （通信由引擎异步执行，不占用 AIC 核）；Finalize 之后执行 Blaze 会挂死（真机实证）
        if constexpr (TILINGKEY_COMPUTE_MATMUL) {
            GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, sharedGmmTiling, gmmArray,
                                               mmArrayAddr_, tilingGM);
            SharedGmmExpertOpType shareComputeOp;
            shareComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
                                sharedGmmComputeWSGM, tilingData_, &tilingData_->sharedGmmTiling, mmArrayAddr_, &pipe);
            shareComputeOp.Process(0, 1);
        }
        hcommImpl.End();
        return;
    }

    // 原通路（TT 量化实例走此路径；MIX_AIC_1_2 任务类型由 KERNEL_TYPE_SEL 声明，
    // AIC 计算 + AIV 通信，历史行为不变）
    constexpr CubeFormat W_FORMAT = CubeFormat::ND;
    constexpr bool USE_SEND_COUNTS = true;
    constexpr bool IS_SHARED_EXPERT = true;
    constexpr bool IS_NOT_SHARED_EXPERT = false;
    using HcclOpType = HcclA2avOp<DTYPE_Y, false, TILINGKEY_COMM_MODE>;
    using GmmASWKernelType =
        Mc2GroupedMatmul::Mc2GmmASWKernel<DTYPE_GMM_X, DTYPE_GMM_WEIGHT, float, DTYPE_GMM_X_SCALE, DTYPE_Y, W_FORMAT,
                                          TILINGKEY_GROUPED_MATMUL_TRANS, TILINGKEY_MATMUL_TRANS>;
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
    GM_ADDR sendBufferAddr = gmmOutputGM;
    HcclOpType hcclOp;
    hcclOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData.taskTilingInfo, sendBufferAddr, yGM,
                static_cast<uint32_t>(tilingData_->taskTilingInfo.aivCoreNum));
    // gmm: yGM=GMM输出缓冲(通信发送源), workspaceGM=groupList+ptrTable区域
    GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, gmmBaseTiling, gmmArray,
                                       gmmArrayAddr_, tilingGM);
    ComputeOpType computeOp;
    computeOp.Init(gmmxGM, gmmweightGM, gmmxScaleGM, gmmWeightScaleGM, gmmOutputGM, gmmComputeWSGM, tilingData_,
                   &tilingData_->gmmBaseTiling, gmmArrayAddr_, &pipe);
    // sharedmm
    GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, sharedGmmTiling, gmmArray,
                                       mmArrayAddr_, tilingGM);
    SharedGmmExpertOpType shareComputeOp;
    shareComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
                        sharedGmmComputeWSGM, tilingData_, &tilingData_->sharedGmmTiling, mmArrayAddr_, &pipe);
    GmmA2avSchedulerType gmmA2avScheduler(hcclOp, computeOp, shareComputeOp, &tilingData.taskTilingInfo);
    gmmA2avScheduler.Process();
}
