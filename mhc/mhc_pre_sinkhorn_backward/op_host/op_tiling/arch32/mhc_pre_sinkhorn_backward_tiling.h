/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef MHC_GRAD_TILING_H
#define MHC_GRAD_TILING_H
#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "err/ops_err.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MhcPreSinkhornBackwardTilingData)
TILING_DATA_FIELD_DEF(int64_t, batchSize)
TILING_DATA_FIELD_DEF(int64_t, seqLength)
TILING_DATA_FIELD_DEF(int64_t, c)
TILING_DATA_FIELD_DEF(int64_t, n)
TILING_DATA_FIELD_DEF(int64_t, c0)      // tile for c
TILING_DATA_FIELD_DEF(int64_t, c1)      // tile count of c
TILING_DATA_FIELD_DEF(int64_t, cTail)      // tail of c
TILING_DATA_FIELD_DEF(int64_t, aivNum)      // tile count of c
TILING_DATA_FIELD_DEF(int64_t, tileGradY)
TILING_DATA_FIELD_DEF(int64_t, tileHHat2)
TILING_DATA_FIELD_DEF(int64_t, tileSize)
TILING_DATA_FIELD_DEF(int64_t, skIterCount)
TILING_DATA_FIELD_DEF(float, eps)

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(MhcPreSinkhornBackward, MhcPreSinkhornBackwardTilingData)
} // namespace optiling

#endif // MHC_GRAD_TILING_H