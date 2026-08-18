/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_TILING_H_
#define FLASH_ATTN_GRAD_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FlashAttnGradTilingDataTmp)
TILING_DATA_FIELD_DEF(int64_t, b)
TILING_DATA_FIELD_DEF(int64_t, s1)
TILING_DATA_FIELD_DEF(int64_t, s2)
TILING_DATA_FIELD_DEF(int64_t, n1)
TILING_DATA_FIELD_DEF(int64_t, n2)
TILING_DATA_FIELD_DEF(int64_t, d)
TILING_DATA_FIELD_DEF(int64_t, dv)
TILING_DATA_FIELD_DEF(float, scaleValue)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FlashAttnGrad, FlashAttnGradTilingDataTmp)

struct FlashAttnGradCompileInfo {
    static ge::graphStatus ParamCheck(gert::TilingContext *context) { return ge::GRAPH_SUCCESS; }
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_H_
