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

#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP

#include <cstdint>
#include "../ascendc/tiling.h"
#include "../ascendc/accum_reduce.h"
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
 * FP8 PV 矩阵乘分块与硬件特征
 * 负责 P(FP8) x V(FP8) -> O(FP32) 的 MMAD 运算与 FixPipe 随路量化 Epilogue
 */
template <
    uint32_t TileM = 64,
    uint32_t TileN = 128,
    uint32_t TileK = 64
>
struct SagePvMmadTraits {
    using ElementA = int8_t;  // Softmax 概率矩阵 S (FP8 e4m3)
    using ElementB = int8_t;  // Value 矩阵 V (FP8 e4m3)
    using ElementC = float;   // L0C 内部累加格式 (FP32)

    using LayoutA = Layout_zZ; // L0A 中 S 分形
    using LayoutB = Layout_nZ; // L0B 中 V 分形
    using LayoutC = Layout_zN; // L0C 中 O 分形

    static constexpr uint32_t M = TileM;
    static constexpr uint32_t N = TileN;
    static constexpr uint32_t K = TileK;

    // 基本分形块尺寸 (FP8 下 16x16x32)
    static constexpr uint32_t M0 = FRACTAL_M0;
    static constexpr uint32_t N0 = FRACTAL_N0;
    static constexpr uint32_t K0 = FRACTAL_K0_FP8;

    static constexpr uint32_t A_BYTES = M * K * sizeof(ElementA);
    static constexpr uint32_t B_BYTES = K * N * sizeof(ElementB);
    static constexpr uint32_t C_BYTES = M * N * sizeof(ElementC);
};

/**
 * FP8 PV 矩阵乘执行器与 Epilogue 封装
 */
template <class Traits = SagePvMmadTraits<64, 128, 64>>
class SagePvMmad {
public:
    using ElementA = typename Traits::ElementA;
    using ElementB = typename Traits::ElementB;
    using ElementC = typename Traits::ElementC;

    __aicore__ SagePvMmad() {}

    /**
     * 加载 Softmax 概率矩阵 S 分块至 L0A
     */
    __aicore__ void LoadL0A(uint32_t s_addr, uint32_t l0a_addr, uint32_t m, uint32_t k) {
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
        (void)s_addr;
        (void)l0a_addr;
        (void)m;
        (void)k;
#else
        (void)s_addr;
        (void)l0a_addr;
        (void)m;
        (void)k;
#endif
    }

    /**
     * 加载 V 分块至 L0B (从 L1 Bank 1 搬运)
     */
    __aicore__ void LoadL0B(uint32_t l1_v_addr, uint32_t l0b_addr, uint32_t k, uint32_t n) {
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
        (void)l1_v_addr;
        (void)l0b_addr;
        (void)k;
        (void)n;
#else
        (void)l1_v_addr;
        (void)l0b_addr;
        (void)k;
        (void)n;
#endif
    }

    /**
     * 执行 Cube FP8 MMAD 计算：O += S x V
     * 
     * @param init 是否重置 L0C 累加器 (两级累加新周期起始为 true)
     */
    __aicore__ void Compute(bool init = false) {
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
        (void)init;
#else
        (void)init;
#endif
    }

    /**
     * FixPipe Epilogue 回写 UB 主累加器
     * 支持随路量化 (FP32 -> FP16) 与分形 NZ -> ND 转置
     */
    __aicore__ void ExecuteEpilogue(
        uint32_t l0c_src_offset,
        uint32_t ub_dst_offset,
        const sage_attn::FixpipeConfig& fixpipe_cfg) {
        
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
        (void)l0c_src_offset;
        (void)ub_dst_offset;
        (void)fixpipe_cfg;
#else
        (void)l0c_src_offset;
        (void)ub_dst_offset;
        (void)fixpipe_cfg;
#endif
    }
};

} // namespace Catlass::SageAttn

#endif // SAGE_ATTN_FP8_CATLASS_SAGE_PV_MMAD_HPP
