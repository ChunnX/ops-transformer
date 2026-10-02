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

#ifndef SAGE_ATTN_FP8_ASCENDC_ONLINE_SOFTMAX_H
#define SAGE_ATTN_FP8_ASCENDC_ONLINE_SOFTMAX_H

#include <cmath>
#include <cstdint>
#include <cstring>

#include "tiling.h"

/**
 * bisheng -xasc 设备编译标记（dav-3510 预定义 __DAV_C310_CUBE__/__DAV_C310_VEC__/__CCE__）；
 * 宿主 g++/clang++（含 __CCE_KT_TEST__ 单测）均不定义 __CCE__，走 std 数学函数分支。
 * 实测依据：设备侧 C 库仅 fabsf 等少数符号有实现，exp2f/log2f 链接期报 undefined symbol，
 * 故设备端 exp2/log2 必须按位构造自实现（见 FastExp2/FastLog2）。
 * 与 sync_guard.h 的定义点共用同一 #ifndef 守卫，保证任意包含顺序下仅定义一次。
 */
#ifndef SAGE_ATTN_DEVICE_COMPILE
#if defined(__CCE__) && !defined(__CCE_KT_TEST__)
#define SAGE_ATTN_DEVICE_COMPILE 1
#else
#define SAGE_ATTN_DEVICE_COMPILE 0
#endif
#endif

#ifndef __aicore__
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
#define __aicore__ [[]]
#else
#define __aicore__
#endif
#endif

namespace sage_attn {

/**
 * 快速绝对值：device 侧 <cmath> 无 std::fabs(float) 可用重载（bisheng 报 no matching function），
 * 统一用符号位判断，host/device 语义一致（对 -0.0 返回 -0.0，不影响本算子比较逻辑）。
 */
__aicore__ static inline float SageFastAbs(float x) {
    return (x < 0.0f) ? -x : x;
}

/**
 * 数学常量定义（exp2 域在线 Softmax）
 */
// LOG2E / INV_LOG2E / SKIP_SCALE_TAU 的唯一定义位于 tiling.h（本文件经 include 引用）
constexpr float EXP2_MIN_CLAMP = -87.33f;           // exp2 下溢极值，低于此值截断为 0
constexpr float MIN_FLOAT_VAL = -3.402823466e+38F;  // -FLT_MAX，float 最小有限值（不再经宏定义）

/**
 * exp2 域 Online Softmax 状态描述符
 */
struct OnlineSoftmaxState {
    // NSDMI 缺省初始化（收敛轮 B/STYLE-8）：设备分支 row_states_ 数组构造后
    // 不经 Reset 也无未定值读取；Reset 语义与缺省值一致
    float m_run = MIN_FLOAT_VAL;       // running max（exp2 域当前全局最大值）
    float d_run = 0.0f;                // running denominator（exp2 域分母累加和）
    float m_cur = MIN_FLOAT_VAL;       // 当前分块的局部最大值
    float d_cur = 0.0f;                // 当前分块的局部 exp2 和
    float m_scale = MIN_FLOAT_VAL;     // 上一次执行 O 累加器重缩放的锚点最大值
    float lse = MIN_FLOAT_VAL;         // 当前维护的 exp2 域 LSE (log-sum-exp)
    bool need_rescale = true;          // 本轮是否需要对 O 主累加器执行 rescale

    __aicore__ void Reset() {
        m_run = MIN_FLOAT_VAL;
        d_run = 0.0f;
        m_cur = MIN_FLOAT_VAL;
        d_cur = 0.0f;
        m_scale = MIN_FLOAT_VAL;
        lse = MIN_FLOAT_VAL;
        need_rescale = true;
    }
};

/**
 * exp2 域 online softmax 状态转移器
 * 核心技术依据：FlashMLA 报告 skip-scale、FlashAttention-3 延迟一拍流水以及 task/01_design.md 2.6/4.3
 */
class OnlineSoftmaxTransformer {
public:
    /**
     * 快速 exp2 实现
     * - 宿主（含 KT 单测）：std::exp2；
     * - bisheng dav-3510 设备端：C 库 exp2f 无设备实现（链接期 undefined symbol），
     *   改为"最近整数拆分 + fdlibm [-0.5,0.5) 极小极大多项式 + 2^n 指数位构造"自实现，
     *   float 精度 < 1 ulp。调用域约定：本算子所有调用点 x <= 0（max 对齐差），
     *   EXP2_MIN_CLAMP 哨兵保证 n >= -88，不会落入非规格化域；仍保留上/下溢防护兜底。
     */
    __aicore__ static inline float FastExp2(float x) {
        if (x < EXP2_MIN_CLAMP) {
            return 0.0f;
        }
#if SAGE_ATTN_DEVICE_COMPILE
        // x = n + f，n 为最近整数，f ∈ [-0.5, 0.5)
        const int32_t n = static_cast<int32_t>(x + (x >= 0.0f ? 0.5f : -0.5f));
        const float f = x - static_cast<float>(n);
        // 2^f 多项式，系数为 ln2^k / k!（与 fdlibm exp 归约多项式一致）
        float p = 1.5252733806584667e-05f;
        p = p * f + 1.5403530393381610e-04f;
        p = p * f + 1.3333558146428443e-03f;
        p = p * f + 9.6181291076284772e-03f;
        p = p * f + 5.5504108664821580e-02f;
        p = p * f + 2.4022650695910071e-01f;
        p = p * f + 6.9314718055994531e-01f;
        p = p * f + 1.0f;
        // 2^n 位构造：float 位型 (n + 127) << 23
        const int32_t e = n + 127;
        uint32_t bits;
        if (e <= 0) {
            bits = 0u;                  // 下溢为 0（哨兵之上不会到达）
        } else if (e >= 255) {
            bits = 0x7F800000u;         // 上溢为 +inf（哨兵之上不会到达）
        } else {
            bits = static_cast<uint32_t>(e) << 23;
        }
        float scale;
        std::memcpy(&scale, &bits, 4);
        return p * scale;
#else
        return std::exp2(x);
#endif
    }

    /**
     * 快速 log2 实现
     * - 宿主（含 KT 单测）：std::log2；
     * - bisheng dav-3510 设备端：C 库 log2f 无设备实现（链接期 undefined symbol），
     *   改为"指数位提取 + 折半归约 [√0.5,√2) + atanh 级数"自实现，float 精度 < 1 ulp。
     */
    __aicore__ static inline float FastLog2(float x) {
        if (x <= 0.0f) {
            return MIN_FLOAT_VAL;
        }
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t bits;
        std::memcpy(&bits, &x, 4);
        int32_t e = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127;
        // 尾数规格化回 [1,2)：非规格化输入（指数域 0）视作 1.m × 2^-127，公式仍精确
        const uint32_t mant_bits = (bits & 0x007FFFFFu) | 0x3F800000u;
        float m;
        std::memcpy(&m, &mant_bits, 4);
        // 折半归约到 [√0.5, √2)：使 |t| ≤ 0.1717，级数收敛最优
        if (m > 1.41421356237309515f) {
            m *= 0.5f;
            ++e;
        }
        const float t = (m - 1.0f) / (m + 1.0f);
        const float t2 = t * t;
        // ln(m) = 2·atanh(t)，偶多项式 Horner（截断项 ≈ 1e-11，远低于 float 精度）
        float s = 1.0f / 11.0f;
        s = s * t2 + 1.0f / 9.0f;
        s = s * t2 + 1.0f / 7.0f;
        s = s * t2 + 1.0f / 5.0f;
        s = s * t2 + 1.0f / 3.0f;
        s = s * t2 + 1.0f;
        s = s * t;
        // log2(x) = e + ln(m)/ln(2) = e + 2·atanh(t)·log2(e)
        return static_cast<float>(e) + (2.0f * s) * LOG2E;
#else
        return std::log2(x);
#endif
    }

    /**
     * 单步状态转移（处理一个到来的 KV 分块极值与局部指数和）
     *
     * @param state 待更新的 Softmax 状态引用
     * @param cur_max 当前 KV 块的局部最大值 (已融合 sm_scale_log2e)
     * @param cur_sum 当前 KV 块按 (logits - cur_max) 计算出的 exp2 之和
     * @param is_force_rescale 是否强制重缩放 (如在循环结束的 drain 步)
     * @return 返回需要施加于 O 主累加器的缩放系数（若跳步缩放则为 1.0f）
     */
    __aicore__ static inline float Step(
        OnlineSoftmaxState& state,
        float cur_max,
        float cur_sum,
        bool is_force_rescale = false) {

        state.m_cur = cur_max;
        state.d_cur = cur_sum;

        if (state.m_run <= MIN_FLOAT_VAL) {
            // 首个有效分块初始化
            state.m_run = cur_max;
            state.d_run = cur_sum;
            state.m_scale = cur_max;
            state.lse = cur_max + FastLog2(cur_sum);
            state.need_rescale = false;
            return 1.0f;
        }

        float m_new = (state.m_run > cur_max) ? state.m_run : cur_max;

        // 判定 skip-scale 条件：
        // 若 (cur_max - m_scale) < tau (6.0f) 且非强制刷新，则跳过对 O 的即时重缩放。
        // O 锚点纪律（2026-10-03 修复）：O 累加器始终以 m_scale 为唯一锚点——
        // 调用方计算 S 分块必须以 state.m_scale（而非 m_run）为基准，
        // 2. S 值域因此放宽到 (0, 2^tau]（tau=6 时 ≤64，e4m3/FP16/FP32 均可精确表示）；
        // 2. d_run 始终以 m_run 为锚维护，最终由 GetFinalInvDenominator 以
        //    2^(m_scale - m_run) 一次性补偿——分子分母锚点各自全局一致，代数严格无漂移。
        //    （修复前的实现允许 skip 期间的 S 以各自 m_run 锚点进入 O，末尾单次补偿
        //     无法纠正逐块锚差，最长块间权重被压低至 2^-tau。）
        bool skip = (!is_force_rescale) && ((cur_max - state.m_scale) < SKIP_SCALE_TAU);

        float rescale_o_factor = 1.0f;

        if (skip) {
            state.need_rescale = false;
            // 延迟重缩放模式：分母仍以全局 m_new 为基准更新；O 锚点 m_scale 不动
            float alpha_old = FastExp2(state.m_run - m_new);
            float alpha_cur = FastExp2(cur_max - m_new);
            state.d_run = state.d_run * alpha_old + cur_sum * alpha_cur;
            state.m_run = m_new;
            rescale_o_factor = 1.0f;
        } else {
            state.need_rescale = true;
            // 触发 O 主累加器更新：将 O 从旧锚点 m_scale 转换到新锚点 m_new
            rescale_o_factor = FastExp2(state.m_scale - m_new);

            float alpha_old = FastExp2(state.m_run - m_new);
            float alpha_cur = FastExp2(cur_max - m_new);
            state.d_run = state.d_run * alpha_old + cur_sum * alpha_cur;

            state.m_run = m_new;
            state.m_scale = m_new;
        }

        // 更新 exp2 域 LSE
        state.lse = state.m_run + FastLog2(state.d_run);

        return rescale_o_factor;
    }

    /**
     * 最终 LSE 换算：将核内 exp2 域 LSE 转换为 Host/PyTorch 消费的自然对数域 LSE
     * 遵循 task/01_design.md 2.6 节与 2.4 节 S_FP8_OFFSET 语义：
     *   P 以 m' = m - s_fp8_offset 平移后量化（= 448·softmax 权重），分母同域累加，
     *   故核内维护的 exp2 域 LSE = 真实 exp2 域 LSE + s_fp8_offset；
     *   出口先补扣 s_fp8_offset，再 /1.44269504 换回自然对数域：
     *   lse_out = (lse_exp2 - s_fp8_offset) / log2(e) + (smooth_k ? lse_correction * sm_scale : 0)
     *
     * @param s_fp8_offset P 矩阵 e4m3 放大偏置（tiling 携带，缺省 0 兼容未启用偏置的调用方）
     */
    __aicore__ static inline float ConvertLseToHost(
        float lse_exp2,
        float sm_scale,
        float lse_correction,
        bool is_smooth_k,
        float s_fp8_offset = 0.0f) {

        float lse_true = (lse_exp2 - s_fp8_offset) * INV_LOG2E;
        if (is_smooth_k) {
            lse_true += lse_correction * sm_scale;
        }
        return lse_true;
    }

    /**
     * 输出归一化系数计算：获取最终除以分母的倒数乘数
     */
    __aicore__ static inline float GetFinalInvDenominator(const OnlineSoftmaxState& state) {
        if (state.d_run > 0.0f) {
            // 需要补偿 m_scale 到 m_run 的残余指数偏差
            float scale_bias = FastExp2(state.m_scale - state.m_run);
            return scale_bias / state.d_run;
        }
        return 0.0f;
    }
};

// 清理记录（2026-10-03 R5 收敛清扫）：原 SoftmaxMode 枚举、DEFAULT_DR_DELTA /
// FALLBACK_OVERFLOW_MARGIN 常量、DeclaredReferenceSoftmaxState 状态与
// DeclaredReferenceSoftmax（FoldAttention DR-Softmax，含 ComputeAdaptiveDelta /
// Merge 等全部静态接口）整块删除——全仓库（内核/宿主/单测/基准）零调用，
// 属未接线的预留实现；Split-KV 归并落地时按需重入并配单测。

} // namespace sage_attn

#endif // SAGE_ATTN_FP8_ASCENDC_ONLINE_SOFTMAX_H
