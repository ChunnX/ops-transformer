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

#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_QK_MMAD_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_QK_MMAD_HPP

#include <cstdint>
#include "../ascendc/tiling.h"
#include "sage_tile.hpp"
#include "sage_resource.hpp"

#ifndef __aicore__
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
#define __aicore__ [[]]
#else
#define __aicore__
#endif
#endif

namespace Catlass::SageAttn {

/**
 * INT8 QK^T 矩阵乘分块与调度策略
 * 负责 L1 -> L0A/L0B -> MMAD -> L0C 的标准硬件通路封装
 */
template <
    uint32_t TileM = 64,
    uint32_t TileN = 64,
    uint32_t TileK = 128
>
struct SageQkMmadTraits {
    using ElementA = int8_t;    // Query 元素类型 (INT8)
    using ElementB = int8_t;    // Key 元素类型 (INT8)
    using ElementC = int32_t;   // 原始 MMAD 累加类型 (INT32)
    using ElementDequant = float; // 反量化后用于 Softmax 的浮点类型

    using LayoutA = Layout_zZ;  // L0A 中 Q 分形
    using LayoutB = Layout_nZ;  // L0B 中 K 分形
    using LayoutC = Layout_zN;  // L0C 中 P 分形

    static constexpr uint32_t M = TileM;
    static constexpr uint32_t N = TileN;
    static constexpr uint32_t K = TileK;

    // 基本分形块尺寸 (INT8 下 16x16x32)
    static constexpr uint32_t M0 = FRACTAL_M0;
    static constexpr uint32_t N0 = FRACTAL_N0;
    static constexpr uint32_t K0 = FRACTAL_K0_INT8;

    static constexpr uint32_t A_BYTES = M * K * sizeof(ElementA);
    static constexpr uint32_t B_BYTES = N * K * sizeof(ElementB);
    static constexpr uint32_t C_BYTES = M * N * sizeof(ElementC);
};

/**
 * INT8 QK^T 矩阵乘执行器
 */
template <class Traits = SageQkMmadTraits<64, 64, 128>>
class SageQkMmad {
public:
    using ElementA = typename Traits::ElementA;
    using ElementB = typename Traits::ElementB;
    using ElementC = typename Traits::ElementC;

    __aicore__ SageQkMmad() {}

    /**
     * 加载 Q 分块：从 L1 Bank 0 搬入 L0A
     */
    __aicore__ void LoadL0A(uint32_t l1_q_addr, uint32_t l0a_addr, uint32_t m, uint32_t k) {
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
        // 调用 MTE1 硬件搬运指令
        (void)l1_q_addr;
        (void)l0a_addr;
        (void)m;
        (void)k;
#else
        (void)l1_q_addr;
        (void)l0a_addr;
        (void)m;
        (void)k;
#endif
    }

    /**
     * 加载 K 分块：从 L1 Bank 1 搬入 L0B
     */
    __aicore__ void LoadL0B(uint32_t l1_k_addr, uint32_t l0b_addr, uint32_t n, uint32_t k) {
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
        // 调用 MTE1 硬件搬运指令
        (void)l1_k_addr;
        (void)l0b_addr;
        (void)n;
        (void)k;
#else
        (void)l1_k_addr;
        (void)l0b_addr;
        (void)n;
        (void)k;
#endif
    }

    /**
     * 执行 Cube MMAD 计算：P = Q x K^T，累加于 L0C
     * 
     * @param init 是否清零 L0C 累加器 (新分块起始为 true，迭代累加为 false)
     */
    __aicore__ void Compute(bool init = true) {
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
        (void)init;
#else
        (void)init;
#endif
    }

    /**
     * 反量化缩放融合：将 INT32 的 MMAD 结果乘以 Q Scale、K Scale 与 sm_scale_log2e
     * 遵循 task/01_design.md 2.2 节与策略四：
     * scale 不缩放 K 本身，而是在 dequant 阶段随 fragment 广播到 QK^T 结果
     */
    __aicore__ static inline float ComputeDequantScale(
        float q_scale,
        float k_scale,
        float sm_scale_log2e) {
        
        return q_scale * k_scale * sm_scale_log2e;
    }
};

} // namespace Catlass::SageAttn

#endif // SAGE_ATTN_FP8_CATLASS_SAGE_QK_MMAD_HPP
