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
 * \file causal_conv1d_grad_tiling_key.h
 * \brief TilingKey 模板参数定义（编译期模板分发）
 *
 * ✅ 使用 ASCENDC_TPL_ARGS_DECL 模板编程方式
 * ❌ 禁止使用废弃的 TILING_KEY_IS 宏
 *
 * 本算子 dtype 支持 float16 / bfloat16，kW/activation/inputMode 钉死为常量，
 * 因此 TilingKey 仅由 dtype 一个维度决定（2 个模板实例）。
 */

#ifndef __CAUSAL_CONV1D_GRAD_TILING_KEY_H__
#define __CAUSAL_CONV1D_GRAD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(CausalConv1dGrad,
    ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0))
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_BF16)
    ),
);

#endif
