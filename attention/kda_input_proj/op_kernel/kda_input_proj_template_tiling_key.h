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
 * \file kda_input_proj_template_tiling_key.h
 * \brief
 */

#ifndef KDA_INPUT_PROJ_TEMPLATE_TILING_KEY_H
#define KDA_INPUT_PROJ_TEMPLATE_TILING_KEY_H

#ifndef ORIG_DTYPE_X
#define ORIG_DTYPE_X (-1)
#endif

#ifndef ORIG_DTYPE_WEIGHT_QKV
#define ORIG_DTYPE_WEIGHT_QKV (-1)
#endif

#ifndef ORIG_DTYPE_QKV
#define ORIG_DTYPE_QKV (-1)
#endif

#include "ascendc/host_api/tiling/template_argument.h"
#include "kda_input_proj_tiling_data.h"

// 当前无通过 tilingkey 配置的模板参数；dtype/format 由编译框架 ORIG_DTYPE_* / DTYPE_* / FORMAT_* 注入
ASCENDC_TPL_ARGS_DECL(KdaInputProj, );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_TILING_STRUCT_SEL(optiling::KdaInputProjTilingData)));

#endif // KDA_INPUT_PROJ_TEMPLATE_TILING_KEY_H
