/*!
 * \file dua_quantize_add_layer_norm_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_DUA_QUANTIZE_ADD_LAYER_NORM_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_DUA_QUANTIZE_ADD_LAYER_NORM_H

#include "register/tilingdata_base.h"
#include "common/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DuaQuantizeAddLayerNormTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numCore);
TILING_DATA_FIELD_DEF(uint32_t, numLastDim);
TILING_DATA_FIELD_DEF(uint32_t, numRow);
TILING_DATA_FIELD_DEF(uint32_t, nlFirstDimPerCore);
TILING_DATA_FIELD_DEF(uint32_t, lFirstDimPerCore);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerTime);
TILING_DATA_FIELD_DEF(uint32_t, lastDimPerTime);
TILING_DATA_FIELD_DEF(float, aveNum);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(uint32_t, colMoveCnt);
TILING_DATA_FIELD_DEF(uint32_t, colTail);
TILING_DATA_FIELD_DEF(uint32_t, isZeroPoint1Exist);
TILING_DATA_FIELD_DEF(uint32_t, isZeroPoint2Exist);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DuaQuantizeAddLayerNorm, DuaQuantizeAddLayerNormTilingData)
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_DUA_QUANTIZE_ADD_LAYER_NORM_H