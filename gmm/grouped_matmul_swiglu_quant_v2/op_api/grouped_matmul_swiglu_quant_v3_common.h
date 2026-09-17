/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_GROUPED_MATMUL_SWIGLU_QUANT_V3_COMMON_H_
#define OP_API_GROUPED_MATMUL_SWIGLU_QUANT_V3_COMMON_H_

#include <cstddef>
#include <cstdint>

namespace gmm_swiglu_quant_v3 {
constexpr size_t SINGLE_TENSOR_SIZE = 1UL;
constexpr size_t FIRST_TENSOR_INDEX = 0UL;
constexpr size_t X_DIM_NUM = 2UL;
constexpr size_t X_SCALE_DIM_NUM = 3UL;
constexpr size_t WEIGHT_VIEW_DIM_NUM = 3UL;
constexpr size_t WEIGHT_SCALE_DIM_NUM = 4UL;
constexpr size_t GROUP_LIST_DIM_NUM = 1UL;
constexpr size_t OUTPUT_DIM_NUM = 2UL;
constexpr size_t OUTPUT_SCALE_DIM_NUM = 3UL;
constexpr size_t WEIGHT_STORAGE_DIM_NUM = 5UL;

constexpr int64_t MX_GROUP_SIZE = 64L;
constexpr int64_t MX_SCALE_PAIR = 2L;
constexpr int64_t NZ_INNER_SIZE = 16L;
constexpr int64_t NZ_C0_SIZE = 32L;
constexpr int64_t SWIGLU_SPLIT_FACTOR = 2L;
constexpr int64_t MXFP8_N_ALIGN = 64L;
constexpr int64_t MAX_GROUP_NUM = 1024L;
constexpr int64_t QUANT_MODE_MX = 2L;
constexpr int64_t DEQUANT_DTYPE_FLOAT = 0L;
constexpr int64_t SCALE_ALG_OCP = 0L;
constexpr int64_t SCALE_ALG_CUBLAS = 1L;
constexpr double DEFAULT_DST_TYPE_MAX = 0.0;
constexpr char DEFAULT_ROUND_MODE[] = "rint";
} // namespace gmm_swiglu_quant_v3

#endif // OP_API_GROUPED_MATMUL_SWIGLU_QUANT_V3_COMMON_H_
