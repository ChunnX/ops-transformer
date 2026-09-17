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
 * \file common_utils_mega_moe.h
 * \brief MXFP8FP4 grouped-matmul pipeline constants extracted from blaze for mega_moe.
 */
#pragma once

namespace Blaze {
namespace Gemm {

constexpr uint64_t MX_FP8FP4_L1_K_CONFIG_256 = 256UL;
constexpr uint64_t MX_FP8FP4_L1_K_CONFIG_512 = 512UL;
constexpr uint64_t MX_FP8FP4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128UL;
constexpr uint64_t MX_FP8FP4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD = 256UL;
constexpr uint64_t MX_FP8FP4_SCALE_K_L1_SIZE = 4096UL;
constexpr uint64_t DOUBLE_BUFFER = 2UL;
constexpr uint64_t SYNC_MODE4 = 4;

// MoE A8W4 uses A2/B4; shared experts, Layered and other callers retain upstream A2/B2.
// Both AIC and AIV0 select the same layout at compile time.
template <bool UseL1B4 = false>
struct MegaMoeMxL1Layout {
    static constexpr uint64_t KB = 1024;
    static constexpr uint64_t A_BUFFER_NUM = 2;
    static constexpr uint64_t B_BUFFER_NUM = UseL1B4 ? 4 : 2;
    static constexpr uint64_t A_SLOT_BYTES = (UseL1B4 ? 64 : 128) * KB;
    static constexpr uint64_t B_SLOT_BYTES = 64 * KB;
    static constexpr uint64_t SCALE_SLOT_BYTES = 32 * KB;
    static constexpr uint64_t A_BUFFER_STRIDE_BYTES = (UseL1B4 ? 256 : 128) * KB;
    static constexpr uint64_t B_SCALE_BUFFER_STRIDE_BYTES = (UseL1B4 ? 256 : 384) * KB;
    static constexpr uint64_t A_OFFSET = 128 * KB;
    static constexpr uint64_t SCALE_A_OFFSET = (UseL1B4 ? 192 : 64) * KB;
    static constexpr uint64_t SCALE_B_OFFSET = (UseL1B4 ? 224 : 96) * KB;

    // B4 map (KiB): B={0,256,64,320}, A={128,384}, scA={192,448}, scB={224,480}.
    // B2 map (KiB): B={0,384}, A={128,256}, scA={64,448}, scB={96,480}.
    __aicore__ static inline uint64_t BOffset(uint64_t slot)
    {
        if constexpr (UseL1B4) {
            return (slot & 1U) * B_SCALE_BUFFER_STRIDE_BYTES + (slot >> 1U) * B_SLOT_BYTES;
        } else {
            return slot * B_SCALE_BUFFER_STRIDE_BYTES;
        }
    }
};

} // namespace Gemm
} // namespace Blaze
