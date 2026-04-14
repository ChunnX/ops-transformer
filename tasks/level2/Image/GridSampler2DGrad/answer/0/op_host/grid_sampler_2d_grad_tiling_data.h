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
 * \file grid_sampler_2d_grad_tiling_data.h
 * \brief
 */
#ifndef GRID_SAMPLER_2_D_GRAD_TILINGDATA_H_
#define GRID_SAMPLER_2_D_GRAD_TILINGDATA_H_

#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GridSampler2DGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, pNumPerCore);
TILING_DATA_FIELD_DEF(uint32_t, tailPNum);
TILING_DATA_FIELD_DEF(uint32_t, channel);
TILING_DATA_FIELD_DEF(uint32_t, height);
TILING_DATA_FIELD_DEF(uint32_t, width);
TILING_DATA_FIELD_DEF(uint32_t, gridH);
TILING_DATA_FIELD_DEF(uint32_t, gridW);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
TILING_DATA_FIELD_DEF(uint32_t, interpolation);
TILING_DATA_FIELD_DEF(uint32_t, padding);
TILING_DATA_FIELD_DEF(uint32_t, alignCorners);
TILING_DATA_FIELD_DEF(uint32_t, group);
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNumCast);
TILING_DATA_FIELD_DEF(uint32_t, pNumPerCoreCast);
TILING_DATA_FIELD_DEF(uint32_t, tailPNumCast);
TILING_DATA_FIELD_DEF(uint32_t, castElement);
TILING_DATA_FIELD_DEF(uint32_t, isDeterministic);
TILING_DATA_FIELD_DEF(uint32_t, tailBNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GridSampler2DGrad, GridSampler2DGradTilingData)

struct InputParamsInfo {
    uint32_t batch = 0;
    uint32_t channel = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t gridH = 0;
    uint32_t gridW = 0;
    uint32_t tilingKey = 0;
    int interpolation = 0;
    int padding = 0;
    int alignCorners = 0;
};
} // namespace optiling
#endif // __GRID_SAMPLER_2D_GRAD_TILINGDATA_H__
