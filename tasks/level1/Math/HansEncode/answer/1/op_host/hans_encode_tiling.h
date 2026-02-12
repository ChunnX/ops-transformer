/*!
 * \file hans_encode_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_HANS_ENCODE_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_HANS_ENCODE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
struct HansEncodeCompileInfo {
};

BEGIN_TILING_DATA_DEF(HansEncodeTilingData)
TILING_DATA_FIELD_DEF(int64_t, processCoreDim);
TILING_DATA_FIELD_DEF(int64_t, processLoopPerCore);
TILING_DATA_FIELD_DEF(int64_t, processLoopLastCore);
TILING_DATA_FIELD_DEF(int64_t, fixedLengthPerCore);
TILING_DATA_FIELD_DEF(int64_t, fixedLengthLastCore);
TILING_DATA_FIELD_DEF(int64_t, varLength);
TILING_DATA_FIELD_DEF(bool, statistic);
TILING_DATA_FIELD_DEF(bool, reshuff);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HansEncode, HansEncodeTilingData)
} // namespace optiling

#endif