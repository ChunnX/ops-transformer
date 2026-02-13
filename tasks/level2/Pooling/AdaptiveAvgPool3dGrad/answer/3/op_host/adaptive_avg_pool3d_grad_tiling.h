#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_GRAD_H_
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_GRAD_H_
#include "common/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "common/tiling_base.h"
#include "common/tiling_templates_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AdaptiveAvgPool3dGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, ncNum);

TILING_DATA_FIELD_DEF(int64_t, dIn);
TILING_DATA_FIELD_DEF(int64_t, hIn);
TILING_DATA_FIELD_DEF(int64_t, wIn);

TILING_DATA_FIELD_DEF(int64_t, dOut);
TILING_DATA_FIELD_DEF(int64_t, hOut);
TILING_DATA_FIELD_DEF(int64_t, wOut);

TILING_DATA_FIELD_DEF(int64_t, taskCoreUsed);
TILING_DATA_FIELD_DEF(int64_t, taskNumPerCore);
TILING_DATA_FIELD_DEF(int64_t, taskNumLastCore);
TILING_DATA_FIELD_DEF(int64_t, yNumPerCalc);

TILING_DATA_FIELD_DEF(int64_t, ncSliceNum);
TILING_DATA_FIELD_DEF(int64_t, ncAlignSliceLength);
TILING_DATA_FIELD_DEF(int64_t, ncAlignSliceTail);

TILING_DATA_FIELD_DEF(int64_t, isAtomicAdd);
TILING_DATA_FIELD_DEF(int64_t, deterministicFlag);
END_TILING_DATA_DEF;

struct AdaptiveAvgPool3dGradCompileInfo {
    uint32_t coreNum = 0;
    uint64_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
};

REGISTER_TILING_DATA_CLASS(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradTilingData)
} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_GRAD_H_