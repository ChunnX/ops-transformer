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
// Layout view parameters. Both BSND and BNSD are expressed as ONE 4D compact
// view [viewD0, S, viewD2, D] so the kernel needs a single pl.load form with a
// fixed order= -- the layout difference reduces to these numbers, evaluated at
// runtime with no branch and no extra tilingkey bit.
//   BSND [B,S,N,D] -> view [B,   S, N, D], index [b,     s, n, 0]
//   BNSD [B,N,S,D] -> view [B*N, S, 1, D], index [b*N+n, s, 0, 0]
// index0 = b * coefB0 + n * coefN0 ; index2 = n * coefN2
//   BSND: coefB0=1, coefN0=0, coefN2=1
//   BNSD: coefB0=N, coefN0=1, coefN2=0
// Note: a plain stride remap does NOT work -- make_tensor silently ignores
// non-compact strides and reads with the compact ones instead.
TILING_DATA_FIELD_DEF(int64_t, viewD0)
TILING_DATA_FIELD_DEF(int64_t, viewD2)
TILING_DATA_FIELD_DEF(int64_t, coefB0)
TILING_DATA_FIELD_DEF(int64_t, coefN0)
TILING_DATA_FIELD_DEF(int64_t, coefN2)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FlashAttnGrad, FlashAttnGradTilingDataTmp)

struct FlashAttnGradCompileInfo {
    static ge::graphStatus ParamCheck(gert::TilingContext *context) { return ge::GRAPH_SUCCESS; }
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_H_
