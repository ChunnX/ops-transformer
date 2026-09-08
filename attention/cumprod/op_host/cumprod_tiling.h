/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 */

/*!
 * \file cumprod_tiling.h
 * \brief Cumprod tiling data 注册（布局与 op_kernel/cumprod.py 的 CumprodTiling 一一对应）
 */

#ifndef OPS_TRANSFORMER_CUMPROD_TILING_H_
#define OPS_TRANSFORMER_CUMPROD_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(CumprodTilingData)
TILING_DATA_FIELD_DEF(int64_t, outer)
TILING_DATA_FIELD_DEF(int64_t, scan)
TILING_DATA_FIELD_DEF(int64_t, inner)
TILING_DATA_FIELD_DEF(int64_t, seg_count)
TILING_DATA_FIELD_DEF(int64_t, seg_len)
TILING_DATA_FIELD_DEF(bool, exclusive)
TILING_DATA_FIELD_DEF(bool, reverse)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(Cumprod, CumprodTilingData)

struct CumprodCompileInfo {};

} // namespace optiling

#endif // OPS_TRANSFORMER_CUMPROD_TILING_H_
