/*!
 * \file quantize_add_layer_norm_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_QUANTIZE_ADD_LAYER_NORM_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_QUANTIZE_ADD_LAYER_NORM_H

#include "register/tilingdata_base.h"
#include "common/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(QuantizeAddLayerNormTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numCore);
TILING_DATA_FIELD_DEF(uint32_t, numLastDim);
TILING_DATA_FIELD_DEF(uint32_t, numFirstDim);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerCore);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerCoreTail);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerTime);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, aveFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm, QuantizeAddLayerNormTilingData)
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_QUANTIZE_ADD_LAYER_NORM_H