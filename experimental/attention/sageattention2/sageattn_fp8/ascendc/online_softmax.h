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
#include <algorithm>

#ifndef __aicore__
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
#define __aicore__ [[]]
#else
#define __aicore__
#endif
#endif

namespace sage_attn {

/**
 * 数学常量定义（exp2 域在线 Softmax）
 */
constexpr float LOG2E = 1.4426950408889634f;        // log2(e) 缩放常数
constexpr float INV_LOG2E = 0.6931471805599453f;    // 1.0f / log2(e) = ln(2)
constexpr float SKIP_SCALE_TAU = 6.0f;              // FlashMLA 判定的跳步缩放阈值 tau = 6.0
constexpr float EXP2_MIN_CLAMP = -87.33f;           // exp2 下溢极值，低于此值截断为 0
constexpr float MIN_FLOAT_VAL = -1.0e30f;

/**
 * exp2 域 Online Softmax 状态描述符
 */
struct OnlineSoftmaxState {
    float m_run;       // running max（exp2 域当前全局最大值）
    float d_run;       // running denominator（exp2 域分母累加和）
    float m_cur;       // 当前分块的局部最大值
    float d_cur;       // 当前分块的局部 exp2 和
    float m_scale;     // 上一次执行 O 累加器重缩放的锚点最大值
    float lse;         // 当前维护的 exp2 域 LSE (log-sum-exp)
    bool need_rescale; // 本轮是否需要对 O 主累加器执行 rescale

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
     * 快速 exp2 实现（在 Ascend NPU 上映射到向量 exp2 指令或快速多项式）
     */
    __aicore__ static inline float FastExp2(float x) {
        if (x < EXP2_MIN_CLAMP) {
            return 0.0f;
        }
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
        return ::exp2f(x);
#else
        return std::exp2(x);
#endif
    }

    /**
     * 快速 log2 实现
     */
    __aicore__ static inline float FastLog2(float x) {
        if (x <= 0.0f) {
            return MIN_FLOAT_VAL;
        }
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
        return ::log2f(x);
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
        // 若 (cur_max - m_scale) < tau (6.0f) 且非强制刷新，则跳过对 O 的即时重缩放
        bool skip = (!is_force_rescale) && ((cur_max - state.m_scale) < SKIP_SCALE_TAU);

        float rescale_o_factor = 1.0f;

        if (skip) {
            state.need_rescale = false;
            // 延迟重缩放模式：分母仍以全局 m_new 为基准更新
            float alpha_old = FastExp2(state.m_run - m_new);
            float alpha_cur = FastExp2(cur_max - m_new);
            state.d_run = state.d_run * alpha_old + cur_sum * alpha_cur;
            state.m_run = m_new;
            // O 累加器暂不乘以系数，保留在当前缩放基准
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
     * 在线归并两个独立的 Softmax 状态（用于 Split-KV 多核合并与 Ring-Attention）
     */
    __aicore__ static inline OnlineSoftmaxState Merge(
        const OnlineSoftmaxState& s1,
        const OnlineSoftmaxState& s2,
        float& rescale_o1,
        float& rescale_o2) {
        
        OnlineSoftmaxState res;
        if (s1.m_run <= MIN_FLOAT_VAL) {
            rescale_o1 = 0.0f;
            rescale_o2 = 1.0f;
            return s2;
        }
        if (s2.m_run <= MIN_FLOAT_VAL) {
            rescale_o1 = 1.0f;
            rescale_o2 = 0.0f;
            return s1;
        }

        float m_max = (s1.m_run > s2.m_run) ? s1.m_run : s2.m_run;
        
        // 计算两个部分和各自对齐到 m_max 的缩放系数
        rescale_o1 = FastExp2(s1.m_scale - m_max);
        rescale_o2 = FastExp2(s2.m_scale - m_max);

        float d1 = s1.d_run * FastExp2(s1.m_run - m_max);
        float d2 = s2.d_run * FastExp2(s2.m_run - m_max);

        res.m_run = m_max;
        res.d_run = d1 + d2;
        res.m_scale = m_max;
        res.m_cur = m_max;
        res.d_cur = 0.0f;
        res.lse = m_max + FastLog2(res.d_run);
        res.need_rescale = true;

        return res;
    }

    /**
     * 最终 LSE 换算：将核内 exp2 域 LSE 转换为 Host/PyTorch 消费的自然对数域 LSE
     * 遵循 task/01_design.md 2.6 节公式：
     * lse_out = lse_exp2 / 1.44269504 + (smooth_k ? (lse_correction * sm_scale) : 0)
     */
    __aicore__ static inline float ConvertLseToHost(
        float lse_exp2,
        float sm_scale,
        float lse_correction,
        bool is_smooth_k) {
        
        float lse_true = lse_exp2 * INV_LOG2E;
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

} // namespace sage_attn

#endif // SAGE_ATTN_FP8_ASCENDC_ONLINE_SOFTMAX_H
