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
 * \file kv_cache_turbo_quant_tiling.h
 * \brief Function-style tiling declaration for KvCacheTurboQuant.
 */

#ifndef KV_CACHE_TURBO_QUANT_TILING_H
#define KV_CACHE_TURBO_QUANT_TILING_H

#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>
#include "../op_kernel/kv_cache_turbo_quant_tiling_data.h"

namespace optiling {

struct KvCacheTurboQuantCompileInfo {
    uint32_t coreNum;
    uint64_t ubSizePlatform;
    uint32_t aicNum; // AIC 核数（TilingPrepare 阶段填充，tiling 阶段只读，避免函数级 static 的多线程竞争）
};

ge::graphStatus KvCacheTurboQuantTilingFunc(gert::TilingContext *context);

ge::graphStatus TilingPrepareForKvCacheTurboQuant(gert::TilingParseContext *context);

} // namespace optiling

#endif // KV_CACHE_TURBO_QUANT_TILING_H
