/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TURBO_QUANT_ATTENTION_SCORE_TILING_H
#define TURBO_QUANT_ATTENTION_SCORE_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
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
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cube);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TurboQuantAttentionScore, TurboQuantAttentionScoreTilingData)
} // namespace optiling
#endif
