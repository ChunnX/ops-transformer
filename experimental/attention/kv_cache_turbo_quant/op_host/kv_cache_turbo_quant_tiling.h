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
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(KvCacheTurboQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, numTokens);
TILING_DATA_FIELD_DEF(int64_t, numKvHeads);
TILING_DATA_FIELD_DEF(int64_t, headDim);
TILING_DATA_FIELD_DEF(int64_t, mseBits);
TILING_DATA_FIELD_DEF(int64_t, qjlDim);
TILING_DATA_FIELD_DEF(int64_t, idxBytes);
TILING_DATA_FIELD_DEF(int64_t, qjlBytes);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);     // AIC 核数（SetBlockDim）
TILING_DATA_FIELD_DEF(uint32_t, totalRows);   // numTokens * numKvHeads
TILING_DATA_FIELD_DEF(uint32_t, coreRows);    // 每 AIV 分片行数
TILING_DATA_FIELD_DEF(uint32_t, vecCoreNum);  // AIV 逻辑核 = AIC*2
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvCacheTurboQuant, KvCacheTurboQuantTilingData)

} // namespace optiling

#endif // KV_CACHE_TURBO_QUANT_TILING_H_
