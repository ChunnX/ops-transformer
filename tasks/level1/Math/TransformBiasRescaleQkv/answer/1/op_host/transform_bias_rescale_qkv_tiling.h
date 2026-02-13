/*!
 * \file transform_bias_rescale_qkv_tiling.h
 * \brief transform_bias_rescale_qkv_tiling_def
 */
#ifndef TRANSFORM_BIAS_RESCALE_QKV_TILING_DEF_H
#define TRANSFORM_BIAS_RESCALE_QKV_TILING_DEF_H

#include "register/tilingdata_base.h"

namespace optiling {
struct TransformBiasRescaleQkvCompileInfo {
};

BEGIN_TILING_DATA_DEF(TransformBiasRescaleQkvTilingData)

TILING_DATA_FIELD_DEF(int64_t, qkvShapeSize);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, token);
TILING_DATA_FIELD_DEF(int64_t, dimension);
TILING_DATA_FIELD_DEF(int64_t, numHeads);
TILING_DATA_FIELD_DEF(int64_t, dimPerHead);
TILING_DATA_FIELD_DEF(int64_t, maxEleNumUB);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TransformBiasRescaleQkv, TransformBiasRescaleQkvTilingData)
} // namespace optiling

#endif