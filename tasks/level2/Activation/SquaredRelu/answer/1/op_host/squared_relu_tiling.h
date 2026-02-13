/*!
 * \file squared_relu_tiling.h
 * \brief squared_relu_tiling
 */

#ifndef SQUARED_RELU_TILING_DEF_H
#define SQUARED_RELU_TILING_DEF_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
struct SquaredReluCompileInfo {};

BEGIN_TILING_DATA_DEF(SquaredReluTilingData)

TILING_DATA_FIELD_DEF(int64_t, elementNum);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SquaredRelu, SquaredReluTilingData)
}  // namespace optiling

#endif