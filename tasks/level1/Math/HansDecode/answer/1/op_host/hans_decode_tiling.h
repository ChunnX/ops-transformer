/*!
 * \file hans_decode_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_HANS_DECODED_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_HANS_DECODED_H
#include "register/tilingdata_base.h"

namespace optiling {
struct HansDecodeCompileInfo {
};

BEGIN_TILING_DATA_DEF(HansDecodeTilingData)
TILING_DATA_FIELD_DEF(int64_t, mantissaByteSize);
TILING_DATA_FIELD_DEF(int64_t, fixedByteSize);
TILING_DATA_FIELD_DEF(int64_t, recoverExpByteSize);
TILING_DATA_FIELD_DEF(int64_t, recoverByteSize);
TILING_DATA_FIELD_DEF(bool, reshuff);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HansDecode, HansDecodeTilingData)
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_HANS_DECODED_H