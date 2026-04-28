/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_TILING_CHECK_H
#define QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_TILING_CHECK_H
#include "tiling/mc2_tiling_utils.h"
namespace MC2Tiling {
constexpr size_t X1_INDEX = 0;
constexpr size_t X2_INDEX = 1;
constexpr size_t Y_INDEX = 2;
constexpr size_t GAMMA_INDEX = 3;
constexpr size_t SCALE_INDEX = 4;
constexpr size_t BIAS_INDEX = 5;
constexpr size_t PER_TOKEN_SCALE_INDEX = 6;
constexpr size_t Y1_INDEX = 0;
constexpr size_t Y2_INDEX = 1;
constexpr size_t X_INDEX = 2;
constexpr size_t OUTPUT_INDEX = 0;
constexpr size_t GROUP_INDEX = 0;
constexpr size_t RANK_SIZE_INDEX = 1;
constexpr size_t TRANSPOSE_X2_INDEX = 2;
constexpr size_t OUT_PUT_DTYPE_INDEX = 3;
constexpr size_t EPSILON_INDEX = 4;
constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;
constexpr size_t NUM_THREE = 3;
constexpr size_t TP_NUMBER = 4;
constexpr size_t ONE_DIM = 1;
constexpr size_t TWO_DIMS = 2;
constexpr size_t FOUR_DIMS = 4;
class QbmmReduceScatterAddRmsNormCastCheckTiling {
public:
    static bool CheckAttrs(const gert::TilingContext *context);
    static bool CheckTensorDataType(const gert::TilingContext *context);
    static bool CheckTensorDim(const gert::TilingContext *context);
    static bool CheckTensorFormat(const gert::TilingContext *context);
    static bool CheckWindowSize(const gert::TilingContext *context);
    static ge::graphStatus TilingCheckQbmmReduceScatterAddRmsNormCast(const gert::TilingContext *context);
};
}; // namespace MC2Tiling
#endif //__QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_TILING_CHECK_H__