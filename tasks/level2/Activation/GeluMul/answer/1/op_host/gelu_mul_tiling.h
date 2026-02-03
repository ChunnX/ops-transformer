/*!
 * \file gelu_mul_tiling.h
 * \brief gelu_mul_tiling
 */

#ifndef GELU_MUL_TILING_H
#define GELU_MUL_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
struct GeluMulCompileInfo {};

BEGIN_TILING_DATA_DEF(GeluMulTilingData)
  TILING_DATA_FIELD_DEF(int32_t, lastDimSize);
  TILING_DATA_FIELD_DEF(int32_t, batchSize);
  TILING_DATA_FIELD_DEF(int32_t, approximateMode);
  TILING_DATA_FIELD_DEF(int32_t, PPMaxCalNum);
  TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluMul, GeluMulTilingData)
}  // namespace optiling

#endif // GELU_MUL_TILING_H