/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file turbo_quant_attention_score_tiling.h
 * \brief Function-style tiling declaration for TurboQuantAttentionScore.
 */

#ifndef TURBO_QUANT_ATTENTION_SCORE_TILING_H
#define TURBO_QUANT_ATTENTION_SCORE_TILING_H

#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>
#include "../op_kernel/turbo_quant_attention_score_tiling_data.h"

namespace optiling {

struct TurboQuantAttentionScoreCompileInfo {
    uint32_t coreNum;
    uint64_t ubSizePlatform;
};

ge::graphStatus TurboQuantAttentionScoreTilingFunc(gert::TilingContext *context);

ge::graphStatus TilingPrepareForTurboQuantAttentionScore(gert::TilingParseContext *context);

} // namespace optiling

#endif // TURBO_QUANT_ATTENTION_SCORE_TILING_H
