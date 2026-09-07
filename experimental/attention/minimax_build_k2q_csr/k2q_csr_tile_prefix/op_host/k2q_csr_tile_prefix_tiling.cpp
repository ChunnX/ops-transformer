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
 * K2qCsrTilePrefix Host tiling.
 */
#include "../../k2q_csr_common/op_host/k2q_csr_tiling_common.h"

namespace optiling {
struct K2qCsrTilePrefixCompileInfo {};

static ge::graphStatus K2qCsrTilePrefixTilingFunc(gert::TilingContext *context)
{
    return k2q_csr_tiling::FillTiling(context, k2q_csr_tiling::Stage::TilePrefix);
}

static ge::graphStatus TilingParseForK2qCsrTilePrefix([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(K2qCsrTilePrefix)
    .Tiling(K2qCsrTilePrefixTilingFunc)
    .TilingParse<K2qCsrTilePrefixCompileInfo>(TilingParseForK2qCsrTilePrefix);
} // namespace optiling
