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
 * \file causal_conv1d_grad_arch35.cpp
 * \brief CausalConv1dGrad kernel 入口（arch35 架构）
 *
 * 模板参数 D_T（数据类型）由 causal_conv1d_grad_tiling_key.h 的
 * ASCENDC_TPL_DATATYPE_DECL(D_T, ...) 定义，Tiling 侧通过
 * ASCENDC_TPL_SEL_PARAM 选择 float16/bfloat16。
 */

#include "arch35/causal_conv1d_grad.h"

template <typename D_T>
__global__ __aicore__ void causal_conv1d_grad(
    GM_ADDR grad_y, GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR dx, GM_ADDR dweight, GM_ADDR dbias,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CausalConv1dGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(CausalConv1dGradTilingData, tilingData, tiling);
    NsCausalConv1dGrad::CausalConv1dGradKernel<D_T> op;
    op.Init(grad_y, x, weight, bias, dx, dweight, dbias, &tilingData);
    op.Process();
}
