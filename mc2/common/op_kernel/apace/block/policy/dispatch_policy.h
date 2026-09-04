/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_policy.h
 * \brief Dispatch policy and pipeline constants for the MX FP8/FP4 dynamic KL1 tail-resplit path.
 */
#pragma once

namespace Blaze {
namespace Gemm {

// MX FP8FP4 pipeline constants
constexpr uint64_t MX_FP8FP4_L1_K_CONFIG_256 = 256UL;
constexpr uint64_t MX_FP8FP4_L1_K_CONFIG_512 = 512UL;
constexpr uint64_t MX_FP8FP4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128UL;
constexpr uint64_t MX_FP8FP4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD = 256UL;
constexpr uint64_t MX_FP8FP4_SCALE_K_L1_SIZE = 4096UL;
constexpr uint64_t DOUBLE_BUFFER = 2UL;
constexpr uint64_t SYNC_MODE4 = 4;

/**
 * @struct MatmulMxFp8Fp4DynamicKL1TailResplit
 * @brief Dispatch policy for the MX FP8/FP4 dynamic KL1 tail-resplit path.
 */
struct MatmulMxFp8Fp4DynamicKL1TailResplit {};

} // namespace Gemm
} // namespace Blaze
