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
 * \file upsample_nearest_exact2d_grad_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_NEAREST_EXACT2D_GRAD_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_NEAREST_EXACT2D_GRAD_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr uint16_t MAX_CORE_CONT = 50;
constexpr int8_t SHAPE_SIZE = 4;
constexpr int8_t N_INDEX = 0;
constexpr int8_t C_INDEX = 1;
constexpr int8_t H_INDEX = 2;
constexpr int8_t W_INDEX = 3;

constexpr float ZERO_FLOAT = 0.0f;
constexpr uint8_t SCHEDULE_MODE = 1;
const std::string EXACT_2D_GRAD_TYPE = "UpsampleNearestExact2dGrad";

BEGIN_TILING_DATA_DEF(UpsampleNearestExact2dGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, slide_size);
TILING_DATA_FIELD_DEF(int64_t, dataType);
TILING_DATA_FIELD_DEF(float, scale_w);
TILING_DATA_FIELD_DEF(float, scale_h);
TILING_DATA_FIELD_DEF(float, invscale_w);
TILING_DATA_FIELD_DEF(float, invscale_h);

TILING_DATA_FIELD_DEF(float, support_w);
TILING_DATA_FIELD_DEF(int16_t, max_interp_size_w);
TILING_DATA_FIELD_DEF(float, support_h);
TILING_DATA_FIELD_DEF(int16_t, max_interp_size_h);
TILING_DATA_FIELD_DEF(uint64_t, intermediate_matrix_size);
TILING_DATA_FIELD_DEF(uint32_t, radio_matrix_size);
TILING_DATA_FIELD_DEF(uint32_t, radio_matrix_size_h);
TILING_DATA_FIELD_DEF_ARR(int64_t, 4, input_shapes);
TILING_DATA_FIELD_DEF_ARR(int64_t, 4, output_shapes);
TILING_DATA_FIELD_DEF(uint32_t, need_core_num_w);
TILING_DATA_FIELD_DEF(uint32_t, need_core_num_h);

TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, slideStartList_w);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, slideEndList_w);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailSlideStartList_w);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailSlideEndList_w);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailRowStartList_w);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailRowEndList_w);

TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, slideStartList_h);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, slideEndList_h);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailSlideStartList_h);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailSlideEndList_h);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailRowStartList_h);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tailRowEndList_h);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, tailBatchStartListH);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, tailBatchEndListH);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling_w);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling_h);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpsampleNearestExact2dGrad, UpsampleNearestExact2dGradTilingData);

BEGIN_TILING_DATA_DEF(UpsampleNearestExact2dGradTransposeTilingData)
TILING_DATA_FIELD_DEF(int64_t, slideSize);
TILING_DATA_FIELD_DEF(int64_t, slideSizeH);
TILING_DATA_FIELD_DEF(int64_t, slideSizeW);
TILING_DATA_FIELD_DEF(float, scaleH);
TILING_DATA_FIELD_DEF(float, scaleW);
TILING_DATA_FIELD_DEF(float, realScaleH);
TILING_DATA_FIELD_DEF(float, realScaleW);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, batches);
TILING_DATA_FIELD_DEF(bool, isHResizeSmall);
TILING_DATA_FIELD_DEF(bool, isWResizeSmall);
TILING_DATA_FIELD_DEF(bool, isHAlign);
TILING_DATA_FIELD_DEF(bool, isWAlign);

TILING_DATA_FIELD_DEF_ARR(int64_t, 4, input_shapes);
TILING_DATA_FIELD_DEF_ARR(int64_t, 4, output_shapes);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, startW);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, endW);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, startH);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, endH);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, startBatches);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, endBatches);
END_TILING_DATA_DEF;

ge::graphStatus tiling4UpsampleNearestExact2dGradTransposeTiling(gert::TilingContext* context);

inline float compute_scale_value(int64_t in_size, int64_t out_size, const float* scale)
{
    if (scale != nullptr && *scale > 0) {
        return static_cast<float>(*scale);
    } else {
        if (out_size > 0) {
            return static_cast<float>(in_size) / out_size;
        }
    }
    return ZERO_FLOAT;
}

inline bool FloatEqual(float a, float b)
{
    float closeTo0 = float(1e-6);
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
}
} // namespace optiling
#endif
