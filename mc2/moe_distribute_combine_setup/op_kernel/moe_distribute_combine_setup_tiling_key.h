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
 * \file moe_distribute_combine_setup_tiling_key.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_SETUP_TILING_KEY_H
#define MOE_DISTRIBUTE_COMBINE_SETUP_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

//由于DECL和SEL只有TILINGKEY_TP一种模板参数，且它只有一个值，所以删除了DECL和SEL
// 模板参数

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法

#endif // MOE_DISTRIBUTE_COMBINE_SETUP_TILING_KEY_H