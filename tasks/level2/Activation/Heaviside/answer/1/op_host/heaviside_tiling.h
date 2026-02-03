#ifndef HEAVISIDE_TILING_H
#define HEAVISIDE_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
struct HeavisideCompileInfo {
};

BEGIN_TILING_DATA_DEF(HeavisideTilingData)
  TILING_DATA_FIELD_DEF(int64_t, elementNum);
  TILING_DATA_FIELD_DEF(bool, valuesType);
  TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Heaviside, HeavisideTilingData)
} // namespace optiling

#endif // HEAVISIDE_TILING_H