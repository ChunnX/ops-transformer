/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef KV_CACHE_TURBO_QUANT_TILING_H_
#define KV_CACHE_TURBO_QUANT_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(KvCacheTurboQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, numTokens);
TILING_DATA_FIELD_DEF(int64_t, numKvHeads);
TILING_DATA_FIELD_DEF(int64_t, headDim);      // 首版固定 128
TILING_DATA_FIELD_DEF(int64_t, mseBits);      // 2/3/4
TILING_DATA_FIELD_DEF(int64_t, qjlDim);       // 首版 = headDim
TILING_DATA_FIELD_DEF(int64_t, idxBytes);     // ceil(headDim * mseBits / 8)
TILING_DATA_FIELD_DEF(int64_t, qjlBytes);     // ceil(qjlDim / 8)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvCacheTurboQuant, KvCacheTurboQuantTilingData)

} // namespace optiling

#endif // KV_CACHE_TURBO_QUANT_TILING_H_
