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

// Shared by the AIC consumer and AIV0 producer. Slots stay fixed across all
// persistent GMM calls, including tails; only the tensor's valid shape changes.
struct MegaMoeMxL1Layout {
    static constexpr uint64_t KB = 1024;
    static constexpr uint64_t A_STAGES = 2;
    static constexpr uint64_t B_STAGES = 4;
    static constexpr uint64_t A_SLOT_BYTES = 64 * KB;
    static constexpr uint64_t B_SLOT_BYTES = 64 * KB;
    static constexpr uint64_t SCALE_SLOT_BYTES = 32 * KB;
    static_assert(A_STAGES * A_SLOT_BYTES + B_STAGES * B_SLOT_BYTES + 4 * SCALE_SLOT_BYTES == 512 * KB,
                  "B4 L1 allocation must fit 512 KiB");

    // B4 map, in KiB:
    // [0,64) B0, [64,128) B2, [128,192) A0, [192,224) scA0, [224,256) scB0
    // [256,320) B1, [320,384) B3, [384,448) A1, [448,480) scA1, [480,512) scB1
    __aicore__ static inline uint64_t AOffset(uint64_t slot)
    {
        return 128 * KB + slot * 256 * KB;
    }

    __aicore__ static inline uint64_t BOffset(uint64_t slot)
    {
        return (slot & 1U) * 256 * KB + (slot >> 1U) * B_SLOT_BYTES;
    }

    __aicore__ static inline uint64_t ScaleAOffset(uint64_t slot)
    {
        return 192 * KB + slot * 256 * KB;
    }

    __aicore__ static inline uint64_t ScaleBOffset(uint64_t slot)
    {
        return 224 * KB + slot * 256 * KB;
    }
};

} // namespace Gemm
} // namespace Blaze
