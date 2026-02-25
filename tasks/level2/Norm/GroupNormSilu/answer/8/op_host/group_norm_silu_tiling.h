#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SILU_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SILU_TILING_H_

#include <cstdint>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "common/tiling_base.h"
#include "common/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "common/tiling_templates_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GroupNormSiluTilingData)
TILING_DATA_FIELD_DEF(int64_t, numGroups);
TILING_DATA_FIELD_DEF(int64_t, hwNum);
TILING_DATA_FIELD_DEF(int64_t, elemNum);
TILING_DATA_FIELD_DEF(int64_t, shapeC);
TILING_DATA_FIELD_DEF(int64_t, shapeD);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numPerCore);
TILING_DATA_FIELD_DEF(int64_t, numLastCore);
TILING_DATA_FIELD_DEF(int64_t, processSize);
TILING_DATA_FIELD_DEF(int64_t, loopNum);
TILING_DATA_FIELD_DEF(int64_t, loopTail);
TILING_DATA_FIELD_DEF(int64_t, innerLoopNum);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTail);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int64_t, activateSilu);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormSilu, GroupNormSiluTilingData)

struct GroupNormSiluCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int32_t is310P = 0;
    int32_t isRegbase = 0;
    uint32_t vectorLength = 0;
    uint32_t blockSizePlatform = 0;
};

enum class GroupNormSiluTilingKey : int64_t
{
    TILINGKEY_HW_ONE_B32 = 108,             // H*W is 1 and dtype is b32
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SILU_TILING_H_