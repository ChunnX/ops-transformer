/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_PRE_BACKWARD_MM_CONFIG_H
#define MHC_PRE_BACKWARD_MM_CONFIG_H

#include <cstdint>

namespace MhcPreBackward {

// 与接入前实现完全等价的低阶MM配置；后续参数消融以新增配置或独立提交进行。
struct MhcPreBackwardMmConfig0 {
    static constexpr uint32_t KEY = 0U;

    static constexpr uint32_t C0_BASE_M = 128U;
    static constexpr uint32_t C0_BASE_N = 128U;
    static constexpr uint32_t C0_BASE_K = 32U;
    static constexpr uint32_t C0_L1_K = 64U;

    static constexpr uint32_t C1_BASE_M = 128U;
    static constexpr uint32_t C1_BASE_N = 128U;
    static constexpr uint32_t C1_BASE_K = 32U;
    static constexpr uint32_t C1_L1_K = 64U;

    static constexpr uint32_t L1_A_BUFFER_DEPTH = 2U;
    static constexpr uint32_t L1_B_BUFFER_DEPTH = 2U;
    static constexpr uint32_t L0A_BUFFER_DEPTH = 2U;
    static constexpr uint32_t L0B_BUFFER_DEPTH = 2U;
    static constexpr uint32_t L0C_BUFFER_DEPTH = 2U;
    static constexpr uint32_t C0_B1_FULL_LOAD = 1U;

    static constexpr uint32_t L1_SLOT_BYTES = 128U * 1024U;
    static constexpr uint32_t L0A_SLOT_BYTES = 32U * 1024U;
    static constexpr uint32_t L0B_SLOT_BYTES = 32U * 1024U;
    static constexpr uint32_t L0C_SLOT_BYTES = 128U * 1024U;

    static constexpr uint32_t C0_M_BLOCK = 512U;
    static constexpr uint32_t C0_N_BLOCK = 128U;
    static constexpr uint32_t C1_K_BLOCK = 512U;
    static constexpr uint32_t C0_TO_V2_ROWS = 128U;
    static constexpr uint32_t V2_TO_C1_ROWS = 512U;
    static constexpr uint32_t V2_CHUNK_ROWS = 128U;
};

} // namespace MhcPreBackward

#endif // MHC_PRE_BACKWARD_MM_CONFIG_H
