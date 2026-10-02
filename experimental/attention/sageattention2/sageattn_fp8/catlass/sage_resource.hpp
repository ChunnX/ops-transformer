/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_RESOURCE_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_RESOURCE_HPP

#include <cstdint>
#include <cstddef>
#include "../ascendc/tiling.h"
#include "sage_tile.hpp"

namespace Catlass::SageAttn {

/**
 * 1C2V 硬件资源配额契约（Atlas 950PR 架构白皮书表 4-2）
 */
struct HardwareResourceQuota {
    // 物理硬件上限
    static constexpr uint32_t L1_TOTAL_BYTES   = sage_attn::L1_TOTAL_SIZE;        // 512 KB
    static constexpr uint32_t L0A_TOTAL_BYTES  = sage_attn::L0A_SIZE;             // 64 KB
    static constexpr uint32_t L0B_TOTAL_BYTES  = sage_attn::L0B_SIZE;             // 64 KB
    static constexpr uint32_t L0C_TOTAL_BYTES  = sage_attn::L0C_SIZE;             // 256 KB
    static constexpr uint32_t UB_PER_V_BYTES   = sage_attn::UB_TOTAL_SIZE_PER_V;  // 256 KB
    static constexpr uint32_t UB_TOTAL_BYTES   = sage_attn::UB_TOTAL_SIZE;        // 512 KB
};

/**
 * Vector 核心 UB 严格解耦划分契约 (task/01_design.md 3.2 节)
 */
struct Vector0UbBudget {
    // V0（量化 prologue 核）：负责 GM 输入暂存、INT8/FP8 量化、ND2NZ 摆放
    static constexpr uint32_t GM_INPUT_PING_OFFSET     = 0;
    static constexpr uint32_t GM_INPUT_PING_BYTES      = 48 * 1024;  // 48 KB

    static constexpr uint32_t GM_INPUT_PONG_OFFSET     = 48 * 1024;
    static constexpr uint32_t GM_INPUT_PONG_BYTES      = 48 * 1024;  // 48 KB

    static constexpr uint32_t QUANT_WORKSPACE_OFFSET   = 96 * 1024;
    static constexpr uint32_t QUANT_WORKSPACE_BYTES    = 48 * 1024;  // 48 KB

    static constexpr uint32_t NZ_OUTPUT_OFFSET         = 144 * 1024;
    static constexpr uint32_t NZ_OUTPUT_BYTES          = 96 * 1024;  // 96 KB (含每行 32B padding 防 bank conflict)

    static constexpr uint32_t RESERVED_OFFSET          = 240 * 1024;
    static constexpr uint32_t RESERVED_BYTES           = 16 * 1024;  // 16 KB 保留

    static_assert(RESERVED_OFFSET + RESERVED_BYTES <= HardwareResourceQuota::UB_PER_V_BYTES,
                  "Vector0 UB allocation exceeds 256KB physical limit");
};

struct Vector1UbBudget {
    // V1（softmax 核）：负责 Online Softmax、O 主累加器、两级累加归约
    static constexpr uint32_t O_ACCUM_OFFSET           = 0;
    static constexpr uint32_t O_ACCUM_BYTES            = 64 * 1024;  // 64 KB (FP32 常驻主累加器)

    static constexpr uint32_t FIXPIPE_RECV_OFFSET      = 64 * 1024;
    static constexpr uint32_t FIXPIPE_RECV_BYTES       = 64 * 1024;  // 64 KB (P 矩阵落地与 S 中间量)

    static constexpr uint32_t TWO_LEVEL_REDUCE_OFFSET  = 128 * 1024;
    static constexpr uint32_t TWO_LEVEL_REDUCE_BYTES   = 64 * 1024;  // 64 KB (两级累加归约缓冲)

    static constexpr uint32_t LSE_WORKSPACE_OFFSET     = 192 * 1024;
    static constexpr uint32_t LSE_WORKSPACE_BYTES      = 32 * 1024;  // 32 KB (m_run, d_run, lse 向量)

    static constexpr uint32_t RESERVED_OFFSET          = 224 * 1024;
    static constexpr uint32_t RESERVED_BYTES           = 32 * 1024;  // 32 KB 保留

    static_assert(RESERVED_OFFSET + RESERVED_BYTES <= HardwareResourceQuota::UB_PER_V_BYTES,
                  "Vector1 UB allocation exceeds 256KB physical limit");
};

/**
 * 资源分配校验器
 */
class SageResourceManager {
public:
    /**
     * 静态校验分块配置是否满足片上容量硬约束
     */
    static constexpr bool ValidateBudget(uint32_t m_per_core, uint32_t block_n, uint32_t head_dim) {
        // Q 矩阵大小 (INT8)
        uint32_t q_bytes = m_per_core * head_dim * sizeof(int8_t);
        if (q_bytes > L1DoubleBankContract::BANK0_SIZE) {
            return false;
        }

        // KV 单块大小 (INT8 或 FP8)
        uint32_t kv_single_block_bytes = block_n * head_dim * sizeof(int8_t);
        // 双缓冲 Ping-Pong 需要 2 个 KV 块
        if (kv_single_block_bytes * 2U > L1DoubleBankContract::BANK1_SIZE) {
            return false;
        }

        // L0C 矩阵乘中间累加大小 (INT32 或 FP32)
        uint32_t l0c_bytes = m_per_core * block_n * sizeof(float);
        if (l0c_bytes > HardwareResourceQuota::L0C_TOTAL_BYTES) {
            return false;
        }

        return true;
    }
};

} // namespace Catlass::SageAttn

#endif // SAGE_ATTN_FP8_CATLASS_SAGE_RESOURCE_HPP
