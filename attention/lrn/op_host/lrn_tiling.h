/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lrn_tiling.h
 * \brief LRN host tiling data definition.
 */

#ifndef LRN_TILING_H_
#define LRN_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

// Layout-identical twin of the PyPTO dataclass LRNTiling
// (op_kernel/lrn.py); the pypto-generated struct itself is
// force-included into the tiling sources at build time. The tile
// geometry (hw_tile / c_chunk / halo tier) rides the TilingKey
// (LRNTilingKey in op_kernel/lrn.py) instead of TilingData.
BEGIN_TILING_DATA_DEF(LRNTilingData)
TILING_DATA_FIELD_DEF(int64_t, n)        // batch N
TILING_DATA_FIELD_DEF(int64_t, c)        // channels C
TILING_DATA_FIELD_DEF(int64_t, hw)       // H*W merged columns
TILING_DATA_FIELD_DEF(int64_t, r)        // depth_radius (exact runtime value)
TILING_DATA_FIELD_DEF(float, bias)
TILING_DATA_FIELD_DEF(float, alpha)
TILING_DATA_FIELD_DEF(float, beta)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(LRN, LRNTilingData)

struct LRNCompileInfo {
    uint32_t aivNum{0};
    uint32_t aicNum{0};
    uint64_t ubSize{0};
    uint64_t l1Size{0};
    NpuArch npuArch{};
};

} // namespace optiling

#endif // LRN_TILING_H_
