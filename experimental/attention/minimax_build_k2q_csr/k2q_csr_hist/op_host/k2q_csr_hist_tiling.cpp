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
 * K2qCsrHist Host tiling.
 */
#include "../../k2q_csr_common/op_host/k2q_csr_tiling_common.h"

namespace optiling {
struct K2qCsrHistCompileInfo {};

static ge::graphStatus K2qCsrHistTilingFunc(gert::TilingContext *context)
{
    return k2q_csr_tiling::FillTiling(context, k2q_csr_tiling::Stage::Hist);
}

static ge::graphStatus TilingParseForK2qCsrHist([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(K2qCsrHist).Tiling(K2qCsrHistTilingFunc).TilingParse<K2qCsrHistCompileInfo>(TilingParseForK2qCsrHist);
} // namespace optiling
