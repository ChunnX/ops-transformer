#ifndef FATRELU_MUL_TILING_DEF_H
#define FATRELU_MUL_TILING_DEF_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
struct FatreluMulCompileInfo {
};

BEGIN_TILING_DATA_DEF(FatreluMulTilingData)

TILING_DATA_FIELD_DEF(int64_t, lastDimSize);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FatreluMul, FatreluMulTilingData)
} // namespace optiling

#endif // FATRELU_MUL_TILING_DEF_H