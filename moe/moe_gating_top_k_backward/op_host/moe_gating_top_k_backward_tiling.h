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
 * \file moe_gating_top_k_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_H

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "log/log.h"

#include "platform/platform_infos_def.h"
#include "util/math_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeGatingTopKBackwardTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum); // 所需要的总核数
TILING_DATA_FIELD_DEF(int64_t, perCoreRows); // 头核需要处理的token数
TILING_DATA_FIELD_DEF(int64_t, lastCoreRows); // 尾核需要处理的token数
TILING_DATA_FIELD_DEF(int64_t, baseRows); // 头核、尾核的头loop一次搬运的token数，同时也是一个UB可以处理的最多token数
TILING_DATA_FIELD_DEF(int64_t, perLoopTimes); // 头核需要loop的次数
TILING_DATA_FIELD_DEF(int64_t, perTailRows); // 头核尾loop一次搬运的token数
TILING_DATA_FIELD_DEF(int64_t, lastLoopTimes); // 尾核需要loop的次数
TILING_DATA_FIELD_DEF(int64_t, lastTailRows); // 尾核尾loop一次搬运的token数
TILING_DATA_FIELD_DEF(int64_t, expertCount); // x shape [M,N]中的N
TILING_DATA_FIELD_DEF(int64_t, k); // x shape [M,K]中的K
TILING_DATA_FIELD_DEF(int64_t, gradYDtype); // gradY的数据类型
TILING_DATA_FIELD_DEF(int64_t, renorm);
TILING_DATA_FIELD_DEF(int64_t, normType);
TILING_DATA_FIELD_DEF(float, routedScalingFactor);
TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeGatingTopKBackward, MoeGatingTopKBackwardTilingData)

struct MoeGatingTopKBackwardCompileInfo {};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_H
