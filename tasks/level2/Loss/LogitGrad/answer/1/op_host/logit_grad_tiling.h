#ifndef LOGIT_GRAD_TILING_DEF_H
#define LOGIT_GRAD_TILING_DEF_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
struct LogitGradCompileInfo {
};

BEGIN_TILING_DATA_DEF(LogitGradTilingData)

TILING_DATA_FIELD_DEF(int64_t, elementNum);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(uint64_t, needCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LogitGrad, LogitGradTilingData)
} // namespace optiling

#endif