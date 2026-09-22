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

// apace hcomm 通路依赖 tensor_api/blaze 头（opc 流程按构建开关引入，与 mega_moe 的 ENABLE_TENSOR_API 惯例一致）。
// 打开方式：构建时环境变量 ENABLE_APACE_QUANT_GMM_A2AV=1（op_host/CMakeLists 同步下发本宏至 kernel 与 tiling）。
#ifdef ENABLE_APACE_QUANT_GMM_A2AV
#include "apace/kernel/fusions/quant_grouped_matmul_all_to_all_v/quant_grouped_matmul_all_to_all_v_tiling_data.h"
#include "apace/kernel/fusions/quant_grouped_matmul_all_to_all_v/mx_quant_grouped_matmul_all_to_all_v_hcomm_impl.h"

// apace 通路支持的 MX 量化 dtype 组合（与算子 README 类型约束表一致，FP8 家族内 X/weight 可混）：
//   FP8：X ∈ {E4M3FN, E5M2} 且 W ∈ {E4M3FN, E5M2}（含 E4M3FN+E5M2 / E5M2+E4M3FN 混合）
//   FP4：X 与 W 均为 E2M1（FP4 不与其他位宽混用）
// 其余 dtype 二进制中 APACE tiling key 实例退化为兜底空实现（tiling 侧 IsCapable 同步拦截）
#define APACE_GMM_A2AV_MX_DTYPE_OK() \
    ((((ORIG_DTYPE_GMM_X == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_GMM_X == DT_FLOAT8_E5M2)) && \
      ((ORIG_DTYPE_GMM_WEIGHT == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_GMM_WEIGHT == DT_FLOAT8_E5M2))) || \
     ((ORIG_DTYPE_GMM_X == DT_FLOAT4_E2M1) && (ORIG_DTYPE_GMM_WEIGHT == DT_FLOAT4_E2M1)))
#endif

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

    // 原 QuantGmmA2avTilingData 保留为默认 tiling 结构；apace hcomm 通路按 tiling key 条件注册
    // （opc 约束：单 kernel 仅允许一个 default tiling 结构，其余须按 tiling key 表达式注册）
    REGISTER_TILING_DEFAULT(QuantGmmA2avTilingData);
#ifdef ENABLE_APACE_QUANT_GMM_A2AV
    REGISTER_TILING_FOR_TILINGKEY("TILINGKEY_COMM_MODE == TILINGKEY_TPL_APACE",
                                  Apace::hcommQuantGmmAlltoAllvTilingData);
#endif

    if constexpr (TILINGKEY_USING_APACE_IMPL) {
        // apace hcomm 融合通路（MIX_AIC_1_2，与原通路同款双核模型）：
        // AIC 走 Blaze GroupedMatmulMxKernel，AIV block0 经 Hccl 提交/等待 AllToAllV，
        // 其余 AIV 参与 Wait/Finalize 多核协议；
        // 内核任务类型由 tiling key 的 ASCENDC_TPL_KERNEL_TYPE_SEL(MIX_AIC_1_2) 声明
        // （MC2 通信域辅助 task 依赖 AIV，不可声明 AIC_ONLY——真机验证 AIC_ONLY 下
        //   kernel 仅 Init+Finalize 也卡死，与 kernel 内容无关）
        // 部署自检 magic：block0 首核先写 y[0]=0xA5A5A5A5，host 侧卡死后读 y 前 4 字节
        // 即可判定"新二进制确实在跑 + 确实走 apace 分支"（排除部署/缓存/通路选择问题）
        *reinterpret_cast<__gm__ uint32_t *>(yGM) = 0xA5A5A5A5U;
#if defined(ENABLE_APACE_QUANT_GMM_A2AV)
#if APACE_GMM_A2AV_MX_DTYPE_OK()
        GET_TILING_DATA_WITH_STRUCT(Apace::hcommQuantGmmAlltoAllvTilingData, apaceTilingData, tilingGM);
        using ApaceImplType =
            Apace::MxQuantGroupedMatmulAllToAllvHcommImpl<DTYPE_GMM_X, DTYPE_GMM_WEIGHT, DTYPE_Y,
                                                          AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU,
                                                          TILINGKEY_GROUPED_MATMUL_TRANS, TILINGKEY_MATMUL_TRANS>;
        ApaceImplType apaceImpl(&apaceTilingData);
        apaceImpl.Init(gmmxGM, gmmweightGM, gmmxScaleGM, gmmWeightScaleGM, mmxOptionalGM, mmweightOptionalGM,
                       mmxScaleGM, mmWeightScaleGM, yGM, mmyOptionalGM, userWorkspace, tilingGM);
        apaceImpl.Run();
        return;
#else
        // 非 MX 量化 dtype 二进制：tiling 侧不会下发该 tiling key，此处兜底直接返回
        return;
#endif
#else
        // 未开启 apace 构建开关（默认）：tiling 侧 IsCapable 同步关闭，不会下发该 tiling key；此处兜底直接返回
        return;
#endif
    }

    // 原通路（CCU/AICPU）：MIX_AIC_1_2 任务类型（AIC 计算 + AIV 通信，历史行为不变）
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    GET_TILING_DATA(tilingData, tilingGM);
    const QuantGmmA2avTilingData *tilingData_ = &tilingData;
    const void *hcclInitTiling = &(tilingData_->hcclA2avTiling.hcclInitTiling);
    uint64_t hcclCcTilingOffset = offsetof(QuantGmmA2avTilingData, hcclA2avTiling) +
                                  offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling);
    constexpr CubeFormat W_FORMAT = CubeFormat::ND;
    constexpr bool USE_SEND_COUNTS = true;
    constexpr bool IS_SHARED_EXPERT = true;
    constexpr bool IS_NOT_SHARED_EXPERT = false;

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
