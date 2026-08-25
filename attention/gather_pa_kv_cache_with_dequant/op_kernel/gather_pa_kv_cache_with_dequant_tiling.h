/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef GATHER_PA_KV_CACHE_WITH_DEQUANT_TILING_H
#define GATHER_PA_KV_CACHE_WITH_DEQUANT_TILING_H

#include <string>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GatherPaKvCacheWithDequantTilingData)
TILING_DATA_FIELD_DEF(int32_t, blockSize);
TILING_DATA_FIELD_DEF(int32_t, numTokens);
TILING_DATA_FIELD_DEF(int32_t, numblkTabCol);
TILING_DATA_FIELD_DEF(int32_t, tokenSizeK);       // numHeads * headDim
TILING_DATA_FIELD_DEF(int32_t, tokenSizeV);
TILING_DATA_FIELD_DEF(int32_t, numHeads);          // number of attention heads
TILING_DATA_FIELD_DEF(int32_t, headDim);            // head dimension
TILING_DATA_FIELD_DEF(int32_t, typeByte);           // key cache type byte (1 for int8)
TILING_DATA_FIELD_DEF(int32_t, hasSeqStarts);
TILING_DATA_FIELD_DEF(int32_t, isSeqLensCumsum);
TILING_DATA_FIELD_DEF(int64_t, kCacheBlockStride);
TILING_DATA_FIELD_DEF(int64_t, vCacheBlockStride);
TILING_DATA_FIELD_DEF(int64_t, kScaleBlockStride);  // blockSize * numHeads
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherPaKvCacheWithDequant, GatherPaKvCacheWithDequantTilingData);

ge::graphStatus Tiling4GatherPaKvCacheWithDequant(gert::TilingContext *context);
ge::graphStatus TilingPrepare4GatherPaKvCacheWithDequant(gert::TilingParseContext *context);

struct Tiling4GatherPaKvCacheWithDequantCompileInfo {
    uint32_t coreNum;
    uint64_t ubSize;
    uint32_t sysWorkspaceSize;
};

} // namespace optiling

#endif // GATHER_PA_KV_CACHE_WITH_DEQUANT_TILING_H
