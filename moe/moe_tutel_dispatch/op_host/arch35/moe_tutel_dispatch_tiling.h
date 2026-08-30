/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tutel_dispatch_tiling.h
 * \brief Ascend 950 host tiling declarations.
 */

#ifndef MOE_TUTEL_DISPATCH_TILING_H_
#define MOE_TUTEL_DISPATCH_TILING_H_

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "../../op_kernel/arch35/moe_tutel_dispatch_tiling_data.h"
#include "../../op_kernel/arch35/moe_tutel_dispatch_tiling_key.h"

namespace optiling {
struct MoeTutelDispatchCompileInfo {
    int64_t aivCoreLimit = 0;
    int64_t ubSize = 0;
};
} // namespace optiling

#endif // MOE_TUTEL_DISPATCH_TILING_H_
