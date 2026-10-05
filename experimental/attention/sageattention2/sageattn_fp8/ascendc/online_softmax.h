#ifndef SAGE_ATTN_FP8_ASCENDC_ONLINE_SOFTMAX_H
#define SAGE_ATTN_FP8_ASCENDC_ONLINE_SOFTMAX_H
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "tiling.h"

/**
 * bisheng -xasc 设备编译标记（dav-3510 预定义 __DAV_C310_CUBE__/__DAV_C310_VEC__/__CCE__）；
 * BiSheng 宿主模式不定义 __CCE__，走 std 数学函数分支。
 * 实测依据：设备侧 C 库仅 fabsf 等少数符号有实现，exp2f/log2f 链接期报 undefined symbol，
 * 故设备端 exp2/log2 必须按位构造自实现（见 FastExp2/FastLog2）。
 */


namespace sage_attn {

/**
 * 快速绝对值：device 侧 <cmath> 无 std::fabs(float) 可用重载（bisheng 报 no matching function），
 * 统一用符号位判断，host/device 语义一致（对 -0.0 返回 -0.0，不影响本算子比较逻辑）。
 */
[[nodiscard]] __aicore__ inline __attribute__((always_inline)) float SageFastAbs(float x) {
    return (x < 0.0f) ? -x : x;
}

/**
 * 数学常量定义（exp2 域在线 Softmax）
 */
// LOG2E / INV_LOG2E 的唯一定义位于 tiling.h（本文件经 include 引用）
constexpr float EXP2_MIN_CLAMP = -87.33f;           // exp2 下溢极值，低于此值截断为 0
constexpr float MIN_FLOAT_VAL = -3.402823466e+38F;  // -FLT_MAX，float 最小有限值（不再经宏定义）
// causal/尾块掩码哨兵（唯一定义点）：必须 > -FLT_MAX，使 "cur_max <= SAGE_MASK_SENTINEL"
// 能区分"整块被掩码"与"状态未初始化"（后者为 MIN_FLOAT_VAL）
constexpr float SAGE_MASK_SENTINEL = -1.0e30f;

/**
 * exp2 域 Online Softmax 状态描述符
 */
struct OnlineSoftmaxState {
    // 结构体成员缺省初始化，确保状态确定性（块内局部极值/局部和为 Step 入参，
    // 不驻留状态——原 m_cur/d_cur 死成员已随 2026-10-03 死代码清扫删除）
    float m_run = MIN_FLOAT_VAL;       // running max（exp2 域当前全局最大值）
    float d_run = 0.0f;                // running denominator（exp2 域分母累加和）
    float m_scale = MIN_FLOAT_VAL;     // 上一次执行 O 累加器重缩放的锚点最大值
    float lse = MIN_FLOAT_VAL;         // 当前维护的 exp2 域 LSE (log-sum-exp)
    bool need_rescale = true;          // 本轮是否需要对 O 主累加器执行 rescale

    __aicore__ inline __attribute__((always_inline)) void Reset() {
        m_run = MIN_FLOAT_VAL;
        d_run = 0.0f;
        m_scale = MIN_FLOAT_VAL;
        lse = MIN_FLOAT_VAL;
        need_rescale = true;
    }
};

/**
 * exp2 域 online softmax 状态转移器
 * 核心技术依据：FlashAttention-3 延迟一拍流水以及 task/01_design.md 2.6/4.3；
 * 缩放策略为恒逐块强锚定重缩放（每块必触发 O 重缩放，m_scale 恒等于 m_run）——
 * golden 的逐块锚定轨迹依赖该纪律（e4m3 量化网格依赖锚点，锚点滞后会偏离网格）。
 */
class OnlineSoftmaxTransformer {
public:
    /**
     * 快速 exp2 实现
     * - 宿主（BiSheng 宿主模式）：std::exp2；
     * - bisheng dav-3510 设备端：C 库 exp2f 无设备实现（链接期 undefined symbol），
     *   改为"最近整数拆分 + fdlibm [-0.5,0.5) 极小极大多项式 + 2^n 指数位构造"自实现，
     *   float 精度 < 1 ulp。调用域约定：本算子所有调用点 x <= 0（max 对齐差），
     *   EXP2_MIN_CLAMP 哨兵保证 n >= -88，不会落入非规格化域；仍保留上/下溢防护兜底。
     */
    [[nodiscard]] __aicore__ inline __attribute__((always_inline)) static float FastExp2(float x) {
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
     * - 宿主（BiSheng 宿主模式）：std::log2；
     * - bisheng dav-3510 设备端：C 库 log2f 无设备实现（链接期 undefined symbol），
     *   改为"指数位提取 + 折半归约 [√0.5,√2) + atanh 级数"自实现，float 精度 < 1 ulp。
     */
    [[nodiscard]] __aicore__ inline __attribute__((always_inline)) static float FastLog2(float x) {
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
     * 单步状态转移（处理一个到来的 KV 分块极值与局部指数和）。
     * 恒逐块强锚定重缩放：除首个有效分块（初始化，返回 1.0 且 need_rescale=false）外，
     * 每块均将 O 锚点 m_scale 重锚到 m_new 并返回 O 重缩放系数——不变量 m_scale == m_run
     * （本文件类注 + test_tiling_struct 锁值）下该系数与分母维持系数 alpha_old 同参同值，
     * 单次 FastExp2 双消费，不另设第二趟指数。
     *
     * @param state 待更新的 Softmax 状态引用
     * @param cur_max 当前 KV 块的局部最大值 (已融合 sm_scale_log2e)
     * @param cur_sum 当前 KV 块按 (logits - cur_max) 计算出的 exp2 之和
     * @return 返回需要施加于 O 主累加器的缩放系数（首块初始化为 1.0f）
     */
    [[nodiscard]] __aicore__ inline __attribute__((always_inline)) static float Step(
        OnlineSoftmaxState& state,
        float cur_max,
        float cur_sum) {

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

        // O 锚点纪律：O 累加器始终以 m_scale 为唯一锚点——
        // 调用方计算 S 分块必须以 state.m_scale（而非 m_run）为基准。
        // 不变量 m_scale == m_run（进函数时成立）⇒ FastExp2(m_scale - m_new) 与
        // FastExp2(m_run - m_new) 同参同值，单算 alpha_old 同时充当 O 重缩放系数与
        // 分母维持系数（原两趟 FastExp2 归并，位级等价：同参纯函数同返回值）。
        const float alpha_old = FastExp2(state.m_run - m_new);
        const float alpha_cur = FastExp2(cur_max - m_new);
        state.d_run = state.d_run * alpha_old + cur_sum * alpha_cur;

        state.m_run = m_new;
        state.m_scale = m_new;
        state.need_rescale = true;

        // 更新 exp2 域 LSE
        state.lse = state.m_run + FastLog2(state.d_run);

        // O 主累加器重缩放系数 = 旧锚 m_scale → m_new（== alpha_old，见上）
        return alpha_old;
    }

    /**
     * 最终 LSE 换算：将核内 exp2 域 LSE 转换为 Host/PyTorch 消费的自然对数域 LSE
     * 遵循 task/01_design.md 2.6 节与 2.4 节 S_FP8_OFFSET 语义：
     *   P 以 m' = m - s_fp8_offset 平移后量化（= 448·softmax 权重），分母同域累加，
     *   故核内维护的 exp2 域 LSE = 真实 exp2 域 LSE + s_fp8_offset；
     *   出口先补扣 s_fp8_offset，再 /1.44269504 换回自然对数域：
     *   lse_out = (lse_exp2 - s_fp8_offset) / log2(e) + (smooth_k ? lse_correction * sm_scale : 0)
     *
     * @param s_fp8_offset P 矩阵 e4m3 放大偏置（S_FP8_OFFSET 常量，唯一定义 tiling.h）
     */
    [[nodiscard]] __aicore__ inline __attribute__((always_inline)) static float ConvertLseToHost(
        float lse_exp2,
        float sm_scale,
        float lse_correction,
        bool is_smooth_k,
        float s_fp8_offset) {

        float lse_true = (lse_exp2 - s_fp8_offset) * INV_LOG2E;
        if (is_smooth_k) {
            lse_true += lse_correction * sm_scale;
        }
        return lse_true;
    }

    /**
     * 输出归一化系数计算：获取最终除以分母的倒数乘数
     */
    [[nodiscard]] __aicore__ inline __attribute__((always_inline)) static float GetFinalInvDenominator(const OnlineSoftmaxState& state) {
        if (state.d_run > 0.0f) {
            // 恒逐块强锚定不变量下 m_scale == m_run，m_scale→m_run 残余指数补偿因子
            // 恒为 FastExp2(0)=1.0f（设备位型构造与宿主 std::exp2 对 0 均精确回 1.0f），
            // 直取倒数，不再保留恒等补偿乘式。
            return 1.0f / state.d_run;
        }
        return 0.0f;
    }
};

} // namespace sage_attn

#endif // SAGE_ATTN_FP8_ASCENDC_ONLINE_SOFTMAX_H
