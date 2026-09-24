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
 * \file kv_cache_turbo_quant_tiling_data.h
 * \brief Tiling data shared between host-side tiling and device-side kernel (plain POD, layout must match).
 */

#ifndef KV_CACHE_TURBO_QUANT_TILING_DATA_H
#define KV_CACHE_TURBO_QUANT_TILING_DATA_H

#include <cstdint>

namespace KvCacheTurboQuant {

struct KvCacheTurboQuantTilingData {
    int64_t numVectors = 0;     // num_tokens * num_kv_heads, 向量总数
    uint32_t usedCoreNum = 0;   // 实际启用核数
    uint32_t formerCoreNum = 0; // 前 formerCoreNum 个核多处理 1 个向量
    uint32_t vecsPerCore = 0;   // 每核基础向量数
    uint32_t mseBits = 3;       // 主量化位宽 2/3/4（冗余存一份，tilingkey 为主）
    uint32_t matDtype = 0;      // rotation/qjl_matrix 元素类型：0=float32，1=bfloat16
    uint32_t outDtype = 0;      // quant_norm/quant_gamma 元素类型：0=bfloat16，1=float16
};

} // namespace KvCacheTurboQuant

#endif // KV_CACHE_TURBO_QUANT_TILING_DATA_H
