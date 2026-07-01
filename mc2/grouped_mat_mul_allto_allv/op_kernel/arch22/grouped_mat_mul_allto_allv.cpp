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
 * \file grouped_mat_mul_allto_allv.cpp
 * \brief GroupedMatMulAlltoAllv 算子 kernel 实现（arch22/Atlas A3 平台）
 *
 * 整体数据流与执行流程：
 *   1. AlltoAllv 通信：将本卡的 gmmX token 按路由结果发送到目标卡，同时接收其他卡发来的 token
 *   2. GroupedMatMul：以接收到的 token 为左矩阵、gmmWeight 为右矩阵，进行路由专家分组矩阵乘
 *   3. Unpermute + AlltoAllv：将计算结果按原顺序恢复并回送到各卡
 *   4. 共享专家 MatMul：mmX @ mmWeight，与上述 1~3 流水并行执行，共享通信带宽空闲时段
 *
 * 关键设计：通过 GmmA2avScheduler 调度器将通信与计算重叠（overlap），以隐藏通信延迟。
 * 通信算子 HcclA2avOp 负责数据搬运，计算算子 QuantGroupedMatmul 负责矩阵乘，
 * 调度器按 token 分块驱动通信-计算流水线交替执行。
 */
#include "grouped_mat_mul_allto_allv_tiling.h"
#include "../grouped_mat_mul_allto_allv_tiling_key.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

// MC2 模板库提供通信算子、计算算子与调度器的通用实现，本算子通过模板参数特化复用
#if __has_include("../../allto_allv_quant_grouped_mat_mul/mc2_templates/mc2_templates.h")
#include "../../allto_allv_quant_grouped_mat_mul/mc2_templates/mc2_templates.h"
#elif __has_include("../../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/mc2_templates.h")
#include "../../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/mc2_templates.h"
#elif __has_include("../allto_allv_quant_grouped_mat_mul/mc2_templates/mc2_templates.h")
#include "../allto_allv_quant_grouped_mat_mul/mc2_templates/mc2_templates.h"
#else
#include "../../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/mc2_templates.h"
#endif

/*
 * 宏：获取嵌套 tiling 数据成员的地址
 *
 * tiling 数据结构为多层嵌套（QuantGmmA2avTilingData -> GMMQuantTilingData -> gmmArray），
 * 该宏根据外层类型、内层类型及各自成员名，计算出最内层成员的地址。
 *
 * 两种场景：
 *   - CONST_TILING（常量 tiling）：tiling 数据在编译期已知，直接通过指针解引用获取成员地址
 *   - 非 CONST_TILING：tiling 数据在 device 侧 __gm__ 全局内存中，通过偏移量求和计算地址
 *
 * 参数说明：
 *   outerType/innerType - 外层/内层结构体类型
 *   outerMember/innerMember - 外层/内层中目标成员名
 *   var - 输出变量名
 *   tiling - tiling 数据起始地址
 */
#if defined(CONST_TILING)
#define GET_NESTED_TILING_DATA_MEMBER_ADDR(outerType, innerType, outerMember, innerMember, var, tiling)                \
    const outerType *outerPtr##var = (const outerType *)(tiling);                                                      \
    const innerType *innerPtr##var = &(outerPtr##var->outerMember);                                                    \
    const int32_t *(var) = (const int32_t *)((const uint8_t *)&(innerPtr##var->innerMember))
#else
#define GET_NESTED_TILING_DATA_MEMBER_ADDR(outerType, innerType, outerMember, innerMember, var, tiling)                \
    size_t outerOffset##var = (size_t)(&((outerType *)0)->outerMember);                                                \
    size_t innerOffset##var = (size_t)(&((innerType *)0)->innerMember);                                                \
    __gm__ int32_t *(var) = (__gm__ int32_t *)((__gm__ uint8_t *)(tiling) + outerOffset##var + innerOffset##var)
#endif

using namespace AscendC;
using namespace MC2KernelTemplate;
using namespace Mc2GroupedMatmulTilingData;

#if defined(CONST_TILING)
#define TILING_TYPE const int32_t
#else
#define TILING_TYPE __gm__ int32_t
#endif

/*
 * kernel 模板函数：通过 4 个模板参数在编译期实例化不同的 kernel 二进制，
 * 运行时根据 tiling key 选择对应的 kernel 执行，避免分支判断开销。
 *
 * 模板参数含义：
 *   TILINGKEY_COMPUTE_MATMUL         - 是否执行 MatMul 计算（0=仅通信, 1=通信+计算）
 *   TILINGKEY_GMM_WEIGHT_TRANS       - 路由专家 gmm_weight 是否转置（影响矩阵乘的 K/N 维度映射）
 *   TILINGKEY_SHARED_MM_WEIGHT_TRANS - 共享专家 mm_weight 是否转置
 *   TILINGKEY_COMM_MODE             - 通信模式（CCU=0, AICPU=1），决定底层通信引擎
 *
 * 以上 4 个参数共组合出 2*2*2*2=16 种 kernel 实例，由 tiling key 映射选择。
 */
template <bool TILINGKEY_COMPUTE_MATMUL, bool TILINGKEY_GMM_WEIGHT_TRANS, bool TILINGKEY_SHARED_MM_WEIGHT_TRANS,
          int TILINGKEY_COMM_MODE>
__global__ __aicore__ void
grouped_mat_mul_allto_allv(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM,
                           GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM,
                           GM_ADDR yGM, GM_ADDR mmyOptionalGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // 设置 kernel 任务类型为 MIX_AIC_1_2：1 个 AIC（Cube 计算核）配 2 个 AIV（Vector 计算核），
    // 适配通信+矩阵乘混合负载场景
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // workspace 为空表示无可执行任务（如所有 token 数为 0 的极端场景），直接返回
    if (workspaceGM == nullptr) {
        return;
    }
    // workspace 前部为框架保留区域，GetUserWorkspace 跳过该区域获取用户可用空间起始地址
    GM_ADDR userWorkspace = GetUserWorkspace(workspaceGM);
    if (userWorkspace == nullptr) {
        return;
    }

    TPipe pipe;
    REGISTER_TILING_DEFAULT(QuantGmmA2avTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    const QuantGmmA2avTilingData *tilingData_ = &tilingData;

    // ---- 解析 HCCL 通信相关 tiling 数据 ----
    // hcclInitTiling 包含通信域初始化信息（rank、world_size 等）
    const void *hcclInitTiling = &(tilingData_->hcclA2avTiling.hcclInitTiling);
    // a2avCcTiling 包含 AlltoAllv 通信的控制信息（每卡发送/接收 token 数、地址偏移等），
    // 通过两层 offsetof 计算其在 tiling 结构中的偏移量，供通信算子运行时按地址读取
    uint64_t hcclCcTilingOffset = offsetof(QuantGmmA2avTilingData, hcclA2avTiling) +
                                  offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling);

    constexpr CubeFormat W_FORMAT = CubeFormat::ND;

    // 非量化场景没有 scale 张量，但计算算子接口需要传入 scale 地址参数，
    // 这里用 userWorkspace 作为占位地址（实际不会被读取）
    GM_ADDR scalePlaceholderGM = userWorkspace;

    // ---- 定义算子类型 ----
    // 通信算子：AlltoAllv 通信，输出数据类型为 DTYPE_Y，非量化，通信模式由模板参数 TILINGKEY_COMM_MODE 决定
    using HcclOpType = HcclA2avOp<DTYPE_Y, false, TILINGKEY_COMM_MODE>;

    // GMM ASW kernel：路由专家矩阵乘的地址计算与 workspace 管理辅助类型，
    // 转置标志通过模板参数传入，影响 K/N 维度的寻址方式
    using GmmASWKernelType =
        Mc2GroupedMatmul::Mc2GmmASWKernel<DTYPE_GMM_X, DTYPE_GMM_WEIGHT, float, DTYPE_GMM_X, DTYPE_Y, W_FORMAT,
                                          TILINGKEY_GMM_WEIGHT_TRANS, TILINGKEY_SHARED_MM_WEIGHT_TRANS>;

    // 路由专家计算算子：QuantGroupedMatmul 模板的非量化特化
    //   第 7 个模板参数 false = 非量化
    //   第 8 个参数 TILINGKEY_GMM_WEIGHT_TRANS = gmm_weight 转置标志
    //   第 9 个参数 false = 非共享专家
    //   第 10 个参数 false = 无 per-token scale
    using ComputeOpType =
        QuantGroupedMatmul<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT, DTYPE_GMM_X,
                           DTYPE_Y, CubeFormat::ND, false, TILINGKEY_GMM_WEIGHT_TRANS, false, false>;

    // 共享专家计算算子：与路由专家算子同为 QuantGroupedMatmul 模板，但第 9 个参数为 true（标记为共享专家），
    // 第 8 个参数使用 TILINGKEY_SHARED_MM_WEIGHT_TRANS（共享专家权重转置标志），
    // 共享专家与路由专家共享 workspace 管理逻辑但独立计算
    using SharedGmmExpertOpType =
        QuantGroupedMatmul<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT, DTYPE_GMM_X,
                           DTYPE_Y, CubeFormat::ND, false, TILINGKEY_SHARED_MM_WEIGHT_TRANS, true, false>;

    // 调度器：核心组件，负责将通信算子与计算算子编排为流水线
    //   TILINGKEY_COMPUTE_MATMUL 控制是否在通信的同时执行矩阵乘（为 false 时仅做通信数据搬运）
    //   调度器内部按 token 分块，交替发起通信与计算，实现通信-计算 overlap
    using GmmA2avSchedulerType =
        GmmA2avScheduler<HcclOpType, ComputeOpType, SharedGmmExpertOpType, TILINGKEY_COMPUTE_MATMUL>;

    // ---- 初始化通信算子 ----
    // 传入：通信初始化 tiling、CC tiling 偏移、任务分块信息、workspace、输出地址 yGM
    // 通信算子将 AlltoAllv 接收的数据直接写入 yGM（通信结果即计算输入，原地复用）
    HcclOpType hcclOp;
    hcclOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData.taskTilingInfo, userWorkspace, yGM);

    // ---- 初始化路由专家计算算子 ----
    // gmmArray 是每个专家的 token 分组列表（记录每个专家处理多少 token），从嵌套 tiling 结构中提取地址
    GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, gmmBaseTiling, gmmArray,
                                       gmmArrayAddr_, tilingGM);
    ComputeOpType computeOp;
    // workspace 内存布局：[gmm 输出区 | gmm 计算工作区 | mm groupList 区 | ...]
    // gmmGroupList 紧随 gmm 输出区之后，用于存放路由专家的分组索引
    auto gmmGroupListGM = userWorkspace + tilingData_->workspaceInfo.wsGmmOutputSize;
    // 初始化计算算子：左矩阵 gmmxGM、右矩阵 gmmweightGM、scale 占位、workspace、groupList 地址、
    // tiling 数据、pipe 队列管理器、最后一个 false 表示不使用 per-token 量化
    computeOp.Init(gmmxGM, gmmweightGM, scalePlaceholderGM, scalePlaceholderGM, userWorkspace, gmmGroupListGM,
                   tilingData_, &tilingData_->gmmBaseTiling, gmmArrayAddr_, &pipe, false);

    // ---- 初始化共享专家计算算子 ----
    // mmArray 是共享专家的 token 分组列表，结构与路由专家一致但使用独立的 tiling 成员
    GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, sharedGmmTiling, gmmArray,
                                       mmArrayAddr_, tilingGM);
    SharedGmmExpertOpType shareComputeOp;
    // mmGroupList 紧随 gmm 输出区与 gmm 计算工作区之后
    auto mmGroupListGM = userWorkspace + tilingData_->workspaceInfo.wsGmmOutputSize +
                         tilingData_->workspaceInfo.wsGmmComputeWorkspaceSize;
    // 共享专家的输入/输出为独立的 mmxOptionalGM/mmweightOptionalGM/mmyOptionalGM，
    // 与路由专家计算完全独立，仅在调度器层面与通信流水并行
    shareComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, scalePlaceholderGM, scalePlaceholderGM, mmyOptionalGM,
                        mmGroupListGM, tilingData_, &tilingData_->sharedGmmTiling, mmArrayAddr_, &pipe, false);

    // ---- 执行调度 ----
    // 调度器 Process() 是算子执行的核心入口：
    //   1. 按 taskTilingInfo 中的分块策略，将 token 划分为多个 chunk
    //   2. 对每个 chunk 交替发起 AlltoAllv 通信与 GroupedMatMul 计算
    //   3. 当 TILINGKEY_COMPUTE_MATMUL=true 时，通信与计算 overlap 执行
    //   4. 共享专家 MatMul 在通信空闲时段穿插执行，最大化硬件利用率
    GmmA2avSchedulerType gmmA2avScheduler(hcclOp, computeOp, shareComputeOp, &tilingData.taskTilingInfo);
    gmmA2avScheduler.Process();
}
