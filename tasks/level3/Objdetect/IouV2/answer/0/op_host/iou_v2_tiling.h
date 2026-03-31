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
 * \file iou_v2_tiling.h
 * \brief
 * 
 * 
 * 
 * 
 * 
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_IOU_V2_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_IOU_V2_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(IouV2TilingData)
  TILING_DATA_FIELD_DEF(uint64_t, bBoxLength);
  TILING_DATA_FIELD_DEF(uint64_t, gtBoxLength);
  TILING_DATA_FIELD_DEF(uint64_t, frontCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, loopNum);
  TILING_DATA_FIELD_DEF(uint64_t, tileLength);
  TILING_DATA_FIELD_DEF(uint64_t, subTileLen);
  TILING_DATA_FIELD_DEF(float, eps);
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(IouV2, IouV2TilingData)

  struct IouV2CompileInfo
  {
  };

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_IOU_V2_H