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
 * \file aot_tiling_config.h
 * \brief AOT Tiling 配置定义 - 预编译的高频使用配置
 */

#ifndef AOT_TILING_CONFIG_H
#define AOT_TILING_CONFIG_H

#include "op_kernel/moe_distribute_combine_v2_tiling.h"
#include "../../moe_distribute_dispatch_v2/ascend910_93/aot_framework.h"

namespace AOT {

static constexpr uint8_t AOT_MoeDistributeCombineV2_runtime_Value[] = {
16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 32, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 128, 4, 0, 0, 72, 0, 0, 0,
6, 0, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 48, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 2, 0, 0, 0, 0, 0,
0, 0, 0, 25, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0
};

using AOT_MoeDistributeCombineV2_runtime_Type = AOTHolder<MoeDistributeCombineV2Info, AOT_MoeDistributeCombineV2_runtime_Value>;

// AOT 注册表
using MoeDistributeCombineV2AOTRegistry = AOTRegistry<
    AOT_MoeDistributeCombineV2_runtime_Type
>;

} // namespace AOT

#endif // AOT_TILING_CONFIG_H
