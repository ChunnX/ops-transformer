#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ANGLE_V2_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ANGLE_V2_H
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AngleV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, totalLength);
TILING_DATA_FIELD_DEF(int64_t, formerNum);
TILING_DATA_FIELD_DEF(int64_t, tailNum);
TILING_DATA_FIELD_DEF(int64_t, formerLength);
TILING_DATA_FIELD_DEF(int64_t, tailLength);
TILING_DATA_FIELD_DEF(int64_t, alignNum);
TILING_DATA_FIELD_DEF(int64_t, totalLengthAligned);
TILING_DATA_FIELD_DEF(int64_t, tileLength);
TILING_DATA_FIELD_DEF(int64_t, dataPerRepeat);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AngleV2, AngleV2TilingData)
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ANGLE_V2_H