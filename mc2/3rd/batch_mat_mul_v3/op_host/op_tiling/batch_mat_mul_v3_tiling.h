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
 * \file batch_mat_mul_v3_tiling.h
 * \brief
 */
#ifndef __OP_HOST_BATCH_MAT_MUL_V3_TILING_H__
#define __OP_HOST_BATCH_MAT_MUL_V3_TILING_H__
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../../mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.h"

// ！！！！！！！！！ATTENTION ！！！！！！！！！！！！
// 当前canndev仓mc2编译时用的是原canndev仓结构体
// nn仓bmmv3用的是此文件定义的结构体
// ascendc对同名结构体会检测，行号不一致打屏warning触发冒烟打断
// ！！！！！！！！千万不要动本文件行号！！！！！！！！！

namespace optiling {

BEGIN_TILING_DATA_DEF(Mc2MultiBatchInfo)
  TILING_DATA_FIELD_DEF(uint32_t, batchUsedCoreNum);

  TILING_DATA_FIELD_DEF(uint32_t, aBatchDimAll);
  TILING_DATA_FIELD_DEF(uint32_t, bBatchDimAll);
  TILING_DATA_FIELD_DEF(uint32_t, cBatchDimAll);

  TILING_DATA_FIELD_DEF(uint32_t, aBatchDim0);
  TILING_DATA_FIELD_DEF(uint32_t, bBatchDim0);
  TILING_DATA_FIELD_DEF(uint32_t, cBatchDim0);

  TILING_DATA_FIELD_DEF(uint32_t, aBatchDim1);
  TILING_DATA_FIELD_DEF(uint32_t, bBatchDim1);
  TILING_DATA_FIELD_DEF(uint32_t, cBatchDim1);

  TILING_DATA_FIELD_DEF(uint32_t, aBatchDim2);
  TILING_DATA_FIELD_DEF(uint32_t, bBatchDim2);
  TILING_DATA_FIELD_DEF(uint32_t, cBatchDim2);

  TILING_DATA_FIELD_DEF(uint32_t, aBatchDim3);
  TILING_DATA_FIELD_DEF(uint32_t, bBatchDim3);
  TILING_DATA_FIELD_DEF(uint32_t, cBatchDim3);

  TILING_DATA_FIELD_DEF(uint32_t, iterBatch);
  TILING_DATA_FIELD_DEF(uint32_t, biasWithBatch);
  TILING_DATA_FIELD_DEF(uint32_t, mOri);
  TILING_DATA_FIELD_DEF(uint32_t, batchTileBlock);

  TILING_DATA_FIELD_DEF(uint32_t, aBatch);
  TILING_DATA_FIELD_DEF(uint32_t, bBatch);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Mc2MultiBatchInfoOp, Mc2MultiBatchInfo)

BEGIN_TILING_DATA_DEF(Mc2BatchMatmulTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(Mc2MatmulV3TilingData, matmulTiling);
  TILING_DATA_FIELD_DEF_STRUCT(Mc2MultiBatchInfo, Mc2multiBatchInfo);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Mc2BatchMatMulV3, Mc2BatchMatmulTilingData)
REGISTER_TILING_DATA_CLASS(Mc2BatchMatmulTilingDataOp, Mc2BatchMatmulTilingData)
}
#endif // __OP_HOST_BATCH_MAT_MUL_V3_TILING_H__