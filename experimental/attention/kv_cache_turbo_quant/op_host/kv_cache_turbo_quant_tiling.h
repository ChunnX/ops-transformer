/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file kv_cache_turbo_quant_tiling.h
 */
#ifndef KV_CACHE_TURBO_QUANT_TILING_H_
#define KV_CACHE_TURBO_QUANT_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(KvCacheTurboQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalVec); // num_tokens * num_kv_heads
TILING_DATA_FIELD_DEF(int64_t, qjlDim);   // rows of qjl_matrix
TILING_DATA_FIELD_DEF(int64_t, mseBits);  // MSE primary quantizer bit width: 2/3/4
TILING_DATA_FIELD_DEF(uint32_t, coreNum); // used logical core num (AIV num for key 0, tri-core groups for key 1)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTiling); // cube tiling for the matmul-lib z/p kernels
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvCacheTurboQuant, KvCacheTurboQuantTilingData)

} // namespace optiling

#endif // KV_CACHE_TURBO_QUANT_TILING_H_
