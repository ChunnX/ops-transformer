/*!
 * \file pows_tiling.h
 * \brief pows_tiling
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_POWS_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_POWS_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PowsTilingData)
TILING_DATA_FIELD_DEF(int64_t, mainCoreLoopNum);
TILING_DATA_FIELD_DEF(int64_t, mainCoreTailLength);
TILING_DATA_FIELD_DEF(int64_t, tailCoreLoopNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreTailLength);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numPerCore);
TILING_DATA_FIELD_DEF(int64_t, dataLength);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, bufSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Pows, PowsTilingData)

struct PowsCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

struct TilingParam {
    int64_t x;
    int64_t coreNum;
    int64_t blockSize;
    int64_t bufSize;
    int64_t ubSize;
};

enum class PowsTilingKey : int64_t
{
    TILINGKEY_101 = 101,
    TILINGKEY_201 = 201,
    TILINGKEY_301 = 301,
};
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_POWS_H_