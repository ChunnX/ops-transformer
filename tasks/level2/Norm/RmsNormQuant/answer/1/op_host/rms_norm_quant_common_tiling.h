/*!
 * \file rms_norm_quant_common_tiling.h
 * \brief
 */

#ifndef ASCEND_OPS_RMS_NORM_ATB_COMMON_H
#define ASCEND_OPS_RMS_NORM_ATB_COMMON_H

#include <register/tilingdata_base.h>

namespace optiling {

enum class NormInputIndex : int32_t
{
    X = 0,
    GAMMA = 1,
    BETA = 2,
    RES_IN = 3
};

enum class NormOutputIndex : int32_t
{
    Y = 0,
    RES_OUT
};

enum class NormAttrIndex : int32_t
{
    EPSILON = 0,
    HIGH_PRECISION_MODE = 1,
    GEMMA_MODE = 2,
    DST_TYPE = 3
};

struct NormCommonCompileInfo {
    uint32_t coreNumAiv = 0;
    uint32_t ubSize = 0;
    uint32_t sysWorkspaceSize = 0;
};

BEGIN_TILING_DATA_DEF(NormCommonTilingData1)
TILING_DATA_FIELD_DEF(uint32_t, numCore);
TILING_DATA_FIELD_DEF(uint32_t, numCol);
TILING_DATA_FIELD_DEF(uint32_t, numRow);
TILING_DATA_FIELD_DEF(float, avgFactor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(uint32_t, sliceSize);
TILING_DATA_FIELD_DEF(uint32_t, sliceNum);
TILING_DATA_FIELD_DEF(uint32_t, tailSliceSize);
TILING_DATA_FIELD_DEF(int32_t, quantMin);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(int32_t, offset);
TILING_DATA_FIELD_DEF(bool, highPrecisionMode);
TILING_DATA_FIELD_DEF(bool, gemmaMode);
TILING_DATA_FIELD_DEF(int32_t, dstType);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNormQuant, NormCommonTilingData1)

} // namespace optiling

#endif