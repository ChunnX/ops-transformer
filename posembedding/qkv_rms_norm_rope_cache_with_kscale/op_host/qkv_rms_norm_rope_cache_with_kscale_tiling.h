/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_KSCALE_TILING_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_KSCALE_TILING_H_

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(QkvRmsNormRopeCacheWithKScaleTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalTokens);
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, numQHeads);
TILING_DATA_FIELD_DEF(uint32_t, numKHeads);
TILING_DATA_FIELD_DEF(uint32_t, numVHeads);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, qkvStrideT);
TILING_DATA_FIELD_DEF(uint32_t, qkvStrideH);
TILING_DATA_FIELD_DEF(uint32_t, cacheStrideBlock);
TILING_DATA_FIELD_DEF(uint32_t, cacheStrideHead);
TILING_DATA_FIELD_DEF(uint32_t, cacheStrideToken);
TILING_DATA_FIELD_DEF(uint32_t, tokenTile);
TILING_DATA_FIELD_DEF(uint32_t, tokenTilePerAiv);
TILING_DATA_FIELD_DEF(uint32_t, rowTile);
TILING_DATA_FIELD_DEF(uint32_t, rowTileAligned);
TILING_DATA_FIELD_DEF(uint32_t, qkHeadTile);
TILING_DATA_FIELD_DEF(uint32_t, vHeadTile);
TILING_DATA_FIELD_DEF(uint32_t, dimTile);
TILING_DATA_FIELD_DEF(uint32_t, flags);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QkvRmsNormRopeCacheWithKScale, QkvRmsNormRopeCacheWithKScaleTilingData)

struct QkvRmsNormRopeCacheWithKScaleCompileInfo {
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
};

enum QkvRrcFlags : uint32_t {
    QKV_RRC_FLAG_D128_FAST = 1U << 0,
    QKV_RRC_FLAG_ROTATION_L1_RESIDENT = 1U << 1,
    QKV_RRC_FLAG_V_CACHE_WRITE_AFTER_A_READY = 1U << 2,
    QKV_RRC_FLAG_TOKEN_TILE_EVEN = 1U << 3,
    QKV_RRC_FLAG_L0C_DOUBLE_BUFFER = 1U << 4
};

ge::graphStatus Tiling4QkvRmsNormRopeCacheWithKScale(gert::TilingContext *context);
ge::graphStatus TilingPrepare4QkvRmsNormRopeCacheWithKScale(gert::TilingParseContext *context);

} // namespace optiling

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_KSCALE_TILING_H_
