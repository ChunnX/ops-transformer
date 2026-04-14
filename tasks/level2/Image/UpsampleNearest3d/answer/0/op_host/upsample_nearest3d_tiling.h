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
 * \file upsample_nearest3d_tiling.h
 * \brief
 */
#ifndef UPSAMPLE_NEAREST3D_TILING_H
#define UPSAMPLE_NEAREST3D_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(UpsampleNearest3dTilingData)
TILING_DATA_FIELD_DEF(uint8_t, dataType);
TILING_DATA_FIELD_DEF(int64_t, batches);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, inputShapes);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, outputShapes);

TILING_DATA_FIELD_DEF(float, scaleW);
TILING_DATA_FIELD_DEF(float, scaleH);
TILING_DATA_FIELD_DEF(float, scaleD);
TILING_DATA_FIELD_DEF(int64_t, slideSizeW);
TILING_DATA_FIELD_DEF(int64_t, tensorSizeW);
TILING_DATA_FIELD_DEF(int64_t, tensorSizeH);
TILING_DATA_FIELD_DEF(int64_t, tensorSizeD);

TILING_DATA_FIELD_DEF(int64_t, slideNumH);
TILING_DATA_FIELD_DEF(int64_t, slideNumD);
TILING_DATA_FIELD_DEF(int64_t, eachCoreSlideNum);
TILING_DATA_FIELD_DEF(int64_t, remainder);
TILING_DATA_FIELD_DEF(int64_t, tailStartSlideNum);
TILING_DATA_FIELD_DEF(int64_t, groupCoreNum);
TILING_DATA_FIELD_DEF(int64_t, inputRow);
TILING_DATA_FIELD_DEF(int64_t, tailAvergingRow);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(bool, isView1DAndSmallW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpsampleNearest3d, UpsampleNearest3dTilingData)

}  // namespace optiling

#endif  // UPSAMPLE_NEAREST3D_TILING_H
