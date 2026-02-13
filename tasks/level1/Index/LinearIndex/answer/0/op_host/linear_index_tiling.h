#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_LINEAR_INDEX_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_LINEAR_INDEX_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LinearIndexTilingData)
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, eachCount);
TILING_DATA_FIELD_DEF(uint64_t, lastCount);
TILING_DATA_FIELD_DEF(uint64_t, indicesCount);
TILING_DATA_FIELD_DEF(uint64_t, indicesAlign);
TILING_DATA_FIELD_DEF(uint64_t, maxSize);
TILING_DATA_FIELD_DEF(uint64_t, eachNum);
TILING_DATA_FIELD_DEF(uint64_t, eachLoop);
TILING_DATA_FIELD_DEF(uint64_t, eachTail);
TILING_DATA_FIELD_DEF(uint64_t, lastNum);
TILING_DATA_FIELD_DEF(uint64_t, lastLoop);
TILING_DATA_FIELD_DEF(uint64_t, lastTail);
TILING_DATA_FIELD_DEF(uint64_t, target);
TILING_DATA_FIELD_DEF(uint64_t, selfStride);
TILING_DATA_FIELD_DEF(uint64_t, indicesStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LinearIndex, LinearIndexTilingData)
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_LINEAR_INDEX_H