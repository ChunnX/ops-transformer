#ifndef CROSS_V2_TILING_H
#define CROSS_V2_TILING_H
#include "register/tilingdata_base.h"
#include "common/tiling_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CrossV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, stepSize);
TILING_DATA_FIELD_DEF(int64_t, tileNum);
TILING_DATA_FIELD_DEF(int64_t, tileNumPerBatch);
TILING_DATA_FIELD_DEF(int64_t, tileDataNum);
TILING_DATA_FIELD_DEF(int64_t, tailDataNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CrossV2, CrossV2TilingData)
} // namespace optiling
#endif // CROSS_V2_TILING_H