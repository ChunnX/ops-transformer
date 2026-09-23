/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef TURBO_QUANT_ATTENTION_SCORE_TILING_H
#define TURBO_QUANT_ATTENTION_SCORE_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(TurboQuantAttentionScoreTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, tokens);
    TILING_DATA_FIELD_DEF(uint32_t, qHeads);
    TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, blocks);
    TILING_DATA_FIELD_DEF(uint32_t, tableWidth);
    TILING_DATA_FIELD_DEF(uint32_t, maxLen);
    TILING_DATA_FIELD_DEF(uint32_t, bits);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TurboQuantAttentionScore, TurboQuantAttentionScoreTilingData)
} // namespace optiling
#endif
