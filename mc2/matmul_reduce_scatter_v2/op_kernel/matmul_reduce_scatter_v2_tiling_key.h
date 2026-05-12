/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_reduce_scatter_v2_tiling_key.h
 * \brief
 */

#ifndef __OP_KERNEL_MATMUL_REDUCE_SCATTER_V2_TILING_KEY_H__
#define __OP_KERNEL_MATMUL_REDUCE_SCATTER_V2_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_COMM_MODE_CCU 0
#define TPL_COMM_MODE_AICPU 1

namespace Mc2Tiling{
// 模板参数
ASCENDC_TPL_ARGS_DECL(
    Mc2MatmulReduceScatterV2, // 算子OpType
    ASCENDC_TPL_BOOL_DECL(TPL_ISBIAS, 0, 1),
    ASCENDC_TPL_BOOL_DECL(TPL_IS_TRANSPOSE_A, 0, 1),
    ASCENDC_TPL_BOOL_DECL(TPL_IS_TRANSPOSE_B, 0, 1),
    ASCENDC_TPL_BOOL_DECL(TPL_IS_SMALLM, 0, 1),
    ASCENDC_TPL_UINT_DECL(TPL_COMM_MODE, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST,
                          TPL_COMM_MODE_CCU, TPL_COMM_MODE_AICPU),
);

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    // aiv_mode_tiling
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(TPL_ISBIAS, 0),
        ASCENDC_TPL_BOOL_SEL(TPL_IS_TRANSPOSE_A, 0),
        ASCENDC_TPL_BOOL_SEL(TPL_IS_TRANSPOSE_B, 0, 1),
        ASCENDC_TPL_BOOL_SEL(TPL_IS_SMALLM, 0, 1)),
        ASCENDC_TPL_UINT_SEL(TPL_COMM_MODE, TPL_COMM_MODE_CCU, TPL_COMM_MODE_AICPU),
);
} // matmul_reduce_scatter_v2_tiling_key

#endif // __OP_KERNEL_MATMUL_REDUCE_SCATTER_V2_TILING_KEY_H__