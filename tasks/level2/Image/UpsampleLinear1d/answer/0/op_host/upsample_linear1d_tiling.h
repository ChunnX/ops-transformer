/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_linear1d_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_LINEAR1D_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_LINEAR1D_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(UpsampleLinear1dTilingData)
TILING_DATA_FIELD_DEF(bool, align_corners);
TILING_DATA_FIELD_DEF(int64_t, slide_size_w);
TILING_DATA_FIELD_DEF(float, scale_w);
TILING_DATA_FIELD_DEF(uint32_t, radio_matrix_size_w);
TILING_DATA_FIELD_DEF(uint32_t, need_core_num_w);
TILING_DATA_FIELD_DEF(int64_t, eachCoreSlideNumW);
TILING_DATA_FIELD_DEF(int64_t, tailStartSlideNumW);
TILING_DATA_FIELD_DEF(int64_t, slideNumW);
TILING_DATA_FIELD_DEF(int64_t, groupCoreNumW);
TILING_DATA_FIELD_DEF(int64_t, tailAvergingRowsW);
TILING_DATA_FIELD_DEF(int64_t, remainderW);
TILING_DATA_FIELD_DEF(uint64_t, mPerTime);
TILING_DATA_FIELD_DEF(uint64_t, loopTimes0);
TILING_DATA_FIELD_DEF(uint64_t, loopTimes1);
TILING_DATA_FIELD_DEF(uint64_t, loopTail0);
TILING_DATA_FIELD_DEF(uint64_t, loopTail1);
TILING_DATA_FIELD_DEF(uint64_t, inputUbSize);
TILING_DATA_FIELD_DEF(uint64_t, outputUbSize);
TILING_DATA_FIELD_DEF(uint64_t, loopTailTimes0);
TILING_DATA_FIELD_DEF(uint64_t, loopTailTimes1);
TILING_DATA_FIELD_DEF(uint64_t, loopTailTail0);
TILING_DATA_FIELD_DEF(uint64_t, loopTailTail1);
TILING_DATA_FIELD_DEF(uint64_t, matmulLoopTimes);
TILING_DATA_FIELD_DEF(uint64_t, matmulBlockTail);
TILING_DATA_FIELD_DEF(uint64_t, matmulBlockTail0);
TILING_DATA_FIELD_DEF(uint64_t, remainderMatmulLoopTimes);
TILING_DATA_FIELD_DEF(uint64_t, remainderMatmulBlockTail);
TILING_DATA_FIELD_DEF(uint64_t, remainderMatmulBlockTail0);
TILING_DATA_FIELD_DEF(uint64_t, remainderLoopTailTimes0);
TILING_DATA_FIELD_DEF(uint64_t, remainderLoopTailTimes1);
TILING_DATA_FIELD_DEF(uint64_t, remainderLoopTailTail0);
TILING_DATA_FIELD_DEF(uint64_t, remainderLoopTailTail1);
TILING_DATA_FIELD_DEF(uint64_t, matmulBlockPerTime);
TILING_DATA_FIELD_DEF(uint64_t, matmulBlockPerTime0);
TILING_DATA_FIELD_DEF(uint64_t, inputH);
TILING_DATA_FIELD_DEF(uint64_t, mmInputNum);
TILING_DATA_FIELD_DEF(uint64_t, mmtotalPerCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, blockSizeNum);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, input_shapes);
TILING_DATA_FIELD_DEF_ARR(int64_t, 3, output_shapes);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling_w);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpsampleLinear1d, UpsampleLinear1dTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_LINEAR_1D_TILING_H
