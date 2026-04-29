/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre_sinkhorn_backward_tiling.h
 * \brief MhcPreSinkhornBackward operator tiling data definition
 */
#ifndef OP_HOST_OP_TILING_ARCH32_MHC_PRE_SINKHORN_BACKWARD_TILING_H
#define OP_HOST_OP_TILING_ARCH32_MHC_PRE_SINKHORN_BACKWARD_TILING_H
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
TILING_DATA_FIELD_DEF(int64_t, ubSize)
TILING_DATA_FIELD_DEF(float, eps)

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(MhcPreSinkhornBackward, MhcPreSinkhornBackwardTilingData)
} // namespace optiling

#endif // OP_HOST_OP_TILING_ARCH32_MHC_PRE_SINKHORN_BACKWARD_TILING_H