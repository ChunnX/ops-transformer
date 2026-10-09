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
 * \file causal_conv1d_backward_apt.cpp
 * \brief CausalConv1dBackward kernel 入口（arch35 架构）
 *
 * 模板参数 D_T（数据类型）由 causal_conv1d_backward_tiling_key.h 的
 * ASCENDC_TPL_DATATYPE_DECL(D_T, ...) 定义，Tiling 侧通过
 * ASCENDC_TPL_SEL_PARAM 选择 float16/bfloat16。
 */

#include "arch35/causal_conv1d_backward.h"

template <typename D_T>
__global__ __aicore__ void causal_conv1d_backward(
    GM_ADDR grad_y, GM_ADDR x, GM_ADDR weight, GM_ADDR query_start_loc,
    GM_ADDR grad_x, GM_ADDR grad_weight,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CausalConv1dBackwardTilingData);
    GET_TILING_DATA_WITH_STRUCT(CausalConv1dBackwardTilingData, tilingData, tiling);
    NsCausalConv1dBackward::CausalConv1dBackwardKernel<D_T> op;
    op.Init(grad_y, x, weight, query_start_loc, grad_x, grad_weight, &tilingData);
    op.Process();
}
