
#ifndef CHUNK_LOCAL_CUMSUM_TILING_H_
#define CHUNK_LOCAL_CUMSUM_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ChunkLocalCumsumTilingData)

TILING_DATA_FIELD_DEF(int, coreNum);
TILING_DATA_FIELD_DEF(int, tokenNum);
TILING_DATA_FIELD_DEF(int, headNum);
TILING_DATA_FIELD_DEF(int, chunkNum);
TILING_DATA_FIELD_DEF(int, chunkSize);
TILING_DATA_FIELD_DEF_ARR(int, 128, curChunkNumDict);  // 128 is max batch

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkLocalCumsum, ChunkLocalCumsumTilingData)
}  // namespace optiling

#endif  // CHUNK_LOCAL_CUMSUM_TILING_H_
