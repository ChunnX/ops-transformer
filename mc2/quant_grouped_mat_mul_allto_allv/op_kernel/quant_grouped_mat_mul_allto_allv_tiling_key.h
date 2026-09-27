/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_mat_mul_allto_allv_tiling_key.h
 * \brief Quant Grouped MatMul AlltoAllV TilingKey 定义
 */
#ifndef QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_KEY_H__
#define QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// 通信模式
#define TILINGKEY_TPL_CCU 0
#define TILINGKEY_TPL_AICPU 1

// 量化模式宏定义
#define QUANT_MODE_NONE 0 // 无量化
#define QUANT_MODE_TT 1   // TT
#define QUANT_MODE_MX 6   // MX量化

// 通信量化模式宏定义
#define COMM_QUANT_MODE_NONE 0 // 不量化

// 是否有共享专家 MM 计算
#define NO_COMPUTE_MATMUL 0  // 无共享专家
#define HAS_COMPUTE_MATMUL 1 // 有共享专家

// 权重转置标志
#define WEIGHT_NO_TRANS 0 // 不转置
#define WEIGHT_TRANS 1    // 转置

/**
 * TilingKey 定义
 * 使用 ASCENDC_TPL_ARGS_DECL 宏实现 TilingKey 到模板参数的自动转换
 *
 * 参数说明：
 * - TILINGKEY_COMPUTE_MATMUL: 是否进行共享专家 MM 计算 (0=无, 1=有)
 * - TILINGKEY_GMM_WEIGHT_TRANS: GMM 计算权重转置场景 (0=不转置, 1=转置)
 * - TILINGKEY_SHARED_MM_WEIGHT_TRANS: 共享专家 MM 计算转置场景 (0=不转置, 1=转置)
 * - TILINGKEY_COMM_MODE: 通信引擎
 * - TILINGKEY_USING_APACE_IMPL: 是否走 apace 融合通路（MX 量化场景按 tiling 侧
 *   CheckApaceImplCapable 动态编码；USING=1 组合经 ASCENDC_TPL_KERNEL_TYPE_SEL 声明
 *   AIC_ONLY 任务类型，USING=0 组合保持 MIX_AIC_1_2 原通路）
 */
ASCENDC_TPL_ARGS_DECL(QuantGroupedMatMulAlltoAllv,

                      // 是否进行MM计算
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_COMPUTE_MATMUL, 0, 1),

                      // GMM 计算转置场景
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_GMM_WEIGHT_TRANS, 0, 1),

                      // 共享专家 MM 计算转置场景
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, 0, 1),

                      // 通信引擎
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_COMM_MODE, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_TPL_CCU, TILINGKEY_TPL_AICPU),

                      // 是否走 apace 融合通路（KERNEL_TYPE_SEL 按此参数声明任务类型）
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_USING_APACE_IMPL, 0, 1), );

/**
 * TilingKey 选择器定义（对齐镜像算子 allto_allv_quant_grouped_mat_mul 的 SEL 结构）：
 * 每组 ARGS_SEL 以 ASCENDC_TPL_KERNEL_TYPE_SEL 首参声明任务类型，BOOL_SEL 以
 * (0, 1) 值对展开覆盖全部组合，无需逐组合枚举：
 * - USING=0（原通路）：KERNEL_TYPE_MIX_AIC_1_2（AIC 计算 + AIV 通信，历史行为不变）
 * - USING=1（apace 融合通路）：KERNEL_TYPE_AIC_ONLY——GMM 计算与 AllToAllV 通信均在
 *   AIC 核内完成，无 AIV 参与
 */
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_COMPUTE_MATMUL, NO_COMPUTE_MATMUL, HAS_COMPUTE_MATMUL),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANS, WEIGHT_NO_TRANS, WEIGHT_TRANS),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, WEIGHT_NO_TRANS, WEIGHT_TRANS),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU,
                                              TILINGKEY_TPL_AICPU),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_USING_APACE_IMPL, 0)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_COMPUTE_MATMUL, NO_COMPUTE_MATMUL, HAS_COMPUTE_MATMUL),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANS, WEIGHT_NO_TRANS, WEIGHT_TRANS),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, WEIGHT_NO_TRANS, WEIGHT_TRANS),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU,
                                              TILINGKEY_TPL_AICPU),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_USING_APACE_IMPL, 1)), );

#endif // QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_KEY_H__
