/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_v3_tiling.h
 */

#ifndef __OP_HOST_MATMUL_V3_TILING_H__
#define __OP_HOST_MATMUL_V3_TILING_H__
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Mc2L2cacheUseInfo)
  TILING_DATA_FIELD_DEF(uint32_t, l2CacheFlag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Mc2L2cacheUseInfoOp, Mc2L2cacheUseInfo);

BEGIN_TILING_DATA_DEF(Mc2L2cacheTilePara)
  TILING_DATA_FIELD_DEF(uint32_t, mTileCntL2);
  TILING_DATA_FIELD_DEF(uint32_t, nTileCntL2);
  TILING_DATA_FIELD_DEF(uint32_t, mTileBlock);
  TILING_DATA_FIELD_DEF(uint32_t, nTileBlock);
  TILING_DATA_FIELD_DEF(uint32_t, calOrder);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Mc2L2cacheTileParaOp, Mc2L2cacheTilePara)

BEGIN_TILING_DATA_DEF(Mc2MatMulRunInfo)
  TILING_DATA_FIELD_DEF(uint32_t, transA);
  TILING_DATA_FIELD_DEF(uint32_t, transB);
  TILING_DATA_FIELD_DEF(uint32_t, nd2nzA);
  TILING_DATA_FIELD_DEF(uint32_t, nd2nzB);
  TILING_DATA_FIELD_DEF(uint32_t, isNzA);
  TILING_DATA_FIELD_DEF(uint32_t, isNzB);
  TILING_DATA_FIELD_DEF(uint32_t, isHf32);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Mc2MatMulRunInfoOp, Mc2MatMulRunInfo)

BEGIN_TILING_DATA_DEF(Mc2MatmulV3TilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
  TILING_DATA_FIELD_DEF_STRUCT(Mc2L2cacheTilePara, tileL2cacheTiling);
  TILING_DATA_FIELD_DEF_STRUCT(Mc2MatMulRunInfo, matmulRunInfo);
  TILING_DATA_FIELD_DEF_STRUCT(Mc2L2cacheUseInfo, l2cacheUseInfo);
  TILING_DATA_FIELD_DEF(uint32_t, baseAN);
  TILING_DATA_FIELD_DEF(uint32_t, baseAD);
  TILING_DATA_FIELD_DEF(uint32_t, baseBN);
  TILING_DATA_FIELD_DEF(uint32_t, baseBD);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Mc2MatMulV3, Mc2MatmulV3TilingData)
REGISTER_TILING_DATA_CLASS(Mc2MatmulV3TilingDataOp, Mc2MatmulV3TilingData)
}
#endif // __OP_HOST_MATMUL_V3_TILING_H__