#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_LINSPACE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_LINSPACE_H_

#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "common/tiling_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LinSpaceTilingData)
TILING_DATA_FIELD_DEF(float, start);
TILING_DATA_FIELD_DEF(float, stop);
TILING_DATA_FIELD_DEF(float, scalar);
TILING_DATA_FIELD_DEF(int64_t, num);
TILING_DATA_FIELD_DEF(int64_t, matrixLen);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numPerCore);
TILING_DATA_FIELD_DEF(int64_t, tailNum);
TILING_DATA_FIELD_DEF(int64_t, innerLoopNum);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTail);
TILING_DATA_FIELD_DEF(int64_t, innerTailLoopNum);
TILING_DATA_FIELD_DEF(int64_t, innerTailLoopTail);
TILING_DATA_FIELD_DEF(int64_t, outerLoopNum);
TILING_DATA_FIELD_DEF(int64_t, outerLoopNumTail);
TILING_DATA_FIELD_DEF(int64_t, outerTailLoopNum);
TILING_DATA_FIELD_DEF(int64_t, outerTailLoopNumTail);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LinSpace, LinSpaceTilingData)

struct LinSpaceCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    NpuArch npuArch = NpuArch::DAV_2201;
};

enum class LinSpaceTilingKey : int64_t
{
    TILINGKEY_302 = 302,
    TILINGKEY_402 = 402,
    TILINGKEY_502 = 502,
    TILINGKEY_602 = 602,
    TILINGKEY_702 = 702
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_LINSPACE_H_