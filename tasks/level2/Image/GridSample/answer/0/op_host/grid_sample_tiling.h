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
 * \file grid_sample_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_GRID_SAMPLE_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_GRID_SAMPLE_H
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GridSampleTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNumVar);
TILING_DATA_FIELD_DEF(int64_t, inN);
TILING_DATA_FIELD_DEF(int64_t, inC);
TILING_DATA_FIELD_DEF(int64_t, inD);
TILING_DATA_FIELD_DEF(int64_t, inH);
TILING_DATA_FIELD_DEF(int64_t, inW);
TILING_DATA_FIELD_DEF(int64_t, outD);
TILING_DATA_FIELD_DEF(int64_t, outH);
TILING_DATA_FIELD_DEF(int64_t, outW);
TILING_DATA_FIELD_DEF(int64_t, interpolationMode);
TILING_DATA_FIELD_DEF(int64_t, paddingMode);
TILING_DATA_FIELD_DEF(int64_t, alignCorners);
TILING_DATA_FIELD_DEF(int64_t, channelLast);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, preCoreNum);
TILING_DATA_FIELD_DEF(int64_t, preNumPerCore);
TILING_DATA_FIELD_DEF(int64_t, postNumPerCore);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GridSample, GridSampleTilingData)

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_GRID_SAMPLE_H
