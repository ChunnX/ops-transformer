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
 * \file allto_allv_grouped_mat_mul_tiling_key.h
 * \brief tiling_key
 */
#ifndef __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_KEY_H__
#define __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// 通信模式
#ifndef TILINGKEY_TPL_CCU
#define TILINGKEY_TPL_CCU 0
#endif
#ifndef TILINGKEY_TPL_AICPU
#define TILINGKEY_TPL_AICPU 1
#endif

// 模板参数范围声明
ASCENDC_TPL_ARGS_DECL(AlltoAllvGroupedMatMul,
                      ASCENDC_TPL_BOOL_DECL( // gmm计算转置场景
                          TILINGKEY_GMM_WEIGHT_TRANSPOSE, 0, 1),
                      ASCENDC_TPL_BOOL_DECL( // mm计算转置场景
                          TILINGKEY_MM_WEIGHT_TRANSPOSE, 0, 1),
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_COMM_MODE, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_TPL_CCU, TILINGKEY_TPL_AICPU),
                      ASCENDC_TPL_BOOL_DECL( // counts 来源：0=attr数组, 1=GM tensor
                          TILINGKEY_COUNTS_FROM_TENSOR, 0, 1), );

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
// 全组合：gmmTrans(0,1) x mmTrans(0,1) x commMode(CCU,AICPU) x countsFromTensor(0,1) = 16种
ASCENDC_TPL_SEL(
    // CCU 通信模式：gmmTrans(0,1) x mmTrans(0,1) x countsFromTensor(0,1) = 8种组合
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANSPOSE, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_MM_WEIGHT_TRANSPOSE, 0, 1),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_COUNTS_FROM_TENSOR, 0, 1)),
    // AICPU 通信模式：gmmTrans(0,1) x mmTrans(0,1) x countsFromTensor(0,1) = 8种组合
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANSPOSE, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_MM_WEIGHT_TRANSPOSE, 0, 1),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_AICPU),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_COUNTS_FROM_TENSOR, 0, 1)),
); // AlltoAllvGroupedMatMul

#endif // __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_KEY_H__
