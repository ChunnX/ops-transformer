#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ChamferDistanceGrad_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ChamferDistanceGrad_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ChamferDistanceGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch_size);     // 0
TILING_DATA_FIELD_DEF(uint32_t, num);            // 1
TILING_DATA_FIELD_DEF(uint64_t, ub_size);        // 2
TILING_DATA_FIELD_DEF(uint32_t, task_per_core);  // 3
TILING_DATA_FIELD_DEF(uint32_t, core_used);      // 4
TILING_DATA_FIELD_DEF(uint32_t, task_tail_core); // 5
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChamferDistanceGrad, ChamferDistanceGradTilingData)

struct ChamferDistanceGradCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ChamferDistanceGrad_H