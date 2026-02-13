#ifndef CTC_LOSS_V3_H_
#define CTC_LOSS_V3_H_
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CTCLossV3TilingData)
TILING_DATA_FIELD_DEF(int64_t, sliceLength);
TILING_DATA_FIELD_DEF(int64_t, sliceLengthTail);
TILING_DATA_FIELD_DEF(int64_t, probSliceNum);
TILING_DATA_FIELD_DEF(int64_t, maxTargetLength);
TILING_DATA_FIELD_DEF(int64_t, timeStep);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, symbleSet);
TILING_DATA_FIELD_DEF(int64_t, targetsDimNum);
TILING_DATA_FIELD_DEF(int64_t, targetsDimLength);
TILING_DATA_FIELD_DEF(int64_t, targetsNum);
TILING_DATA_FIELD_DEF(int64_t, taskPerCore);
TILING_DATA_FIELD_DEF(int64_t, taskTailCore);
TILING_DATA_FIELD_DEF(int64_t, blank);
TILING_DATA_FIELD_DEF(int64_t, zeroInfinity);
END_TILING_DATA_DEF;

struct CTCLossV3CompileInfo {
    uint32_t coreNum = 0;
    uint64_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
};

REGISTER_TILING_DATA_CLASS(CTCLossV3, CTCLossV3TilingData)
} // namespace optiling

#endif // CTC_LOSS_V3_H_