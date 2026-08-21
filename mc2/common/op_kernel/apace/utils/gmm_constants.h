/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gmm_constants.h
 * \brief Shared constants and enums for GMM block-layer components.
 */
#ifndef APACE_UTILS_GMM_CONSTANTS_H
#define APACE_UTILS_GMM_CONSTANTS_H

#include <cstdint>

namespace Apace {
namespace Gemm {

constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t INT32_PER_256B = 8U;
constexpr uint32_t ACTIVATION_N_HALF = 2U;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr uint32_t WEIGHT_INDEX = 3U;

enum class ActMode : uint8_t {
    SWIGLU = 0U,
    SITU = 1U,
    SWIGLU_STEP = 2U,
    SWIGLU_OAI = 3U,
};

enum class ActSubMode : uint8_t {
    DEFAULT = 0U,
    LINEAR = 1U,
};

} // namespace Gemm
} // namespace Apace

#endif // APACE_UTILS_GMM_CONSTANTS_H
