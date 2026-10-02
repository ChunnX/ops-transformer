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

#ifndef SAGE_ATTN_FP8_ASCENDC_TWIN_REFERENCE_H
#define SAGE_ATTN_FP8_ASCENDC_TWIN_REFERENCE_H

/**
 * \file twin_reference.h
 * \brief SageAttention2 全流水 CPU 孪生参考实现（task/04_local_test.md CPU 孪生调试）。
 *
 * 语义单一定义纪律：
 * 1. Q/K/V 量化直接调用 quant 层的唯一定义（QuantizeQInt8Ascend / QuantizeKInt8Ascend /
 *    QuantizeVFp8Ascend，含 half-away 与 RNE 舍入、per_thread/per_warp 分组、e4m3 编码、
 *    16-token permute、scale_max 分档）——本文件零量化逻辑复制；
 * 2. Online Softmax 状态机复用 online_softmax.h 的 OnlineSoftmaxTransformer（锚点纪律见其
 *    Step 注释：S 与 O 同锚 m_scale，分母 d_run 以 m_run 维护、出口一次补偿）；
 * 3. QK^T 以 INT32 精确累加后一次性乘 (q_scale·k_scale·sm_scale_log2e)，与设备核
 *    INT8 MMAD + 反量化 scale 广播的舍入轨迹一致（优于逐元素反量化相乘的参考路径）；
 * 4. P 矩阵 e4m3 域默认即 448· 放大域：SoftmaxProbCode/OnlineBlockExp2Sum 乘
 *    FP8_E4M3_MAX，LSE 出口 ConvertLseToHost 扣 S_FP8_OFFSET（与已对齐 golden 的
 *    Python 原型口径一致）；放大域与 skip-scale tau=6 存在值域冲突（2^tau·448 溢出
 *    e4m3 上限），故放大域下强制逐块重缩放（Step 恒 force=true，skip 分支不启用，
 *    tau 机制仅留给无放大域路线复用）。S_FP8_OFFSET 由 TilingData.s_fp8_offset 携带，
 *    设备核同口径（P 量化锚点 = m_scale - s_fp8_offset）。
 */

#include "tiling.h"
#include "online_softmax.h"

#include "../../quant/quant_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace sage_attn {
namespace twin {

/**
 * 输入/输出载体：q/k/v/out 以 fp16 或 bf16 位型传入（dtype_mode 1=fp16, 2=bf16），
 * lse 为 float32 [b*hq*sq]。布局 HND/NHD 与 quant 层共用同一 LAYOUT_* 编码。
 */
struct TwinIo {
    const void* q;
    const void* k;
    const void* v;
    void* out;
    float* lse;             // 可为 nullptr（等价 return_lse=False）
    const float* key_mean;  // 预计算的 km [b*hkv*d_pad]，smooth_k=0 时可为 nullptr

    uint32_t b;
    uint32_t hq;
    uint32_t hkv;
    uint32_t sq;
    uint32_t sk;
    uint32_t d_og;

    uint32_t tensor_layout;   // LAYOUT_HND / LAYOUT_NHD
    uint32_t qk_quant_gran;   // QUANT_GRAN_PER_WARP / QUANT_GRAN_PER_THREAD
    uint32_t pv_accum_dtype;  // PV_ACCUM_* 三档
    uint32_t is_causal;
    uint32_t smooth_k;
    int dtype_mode;           // 1=fp16, 2=bf16
    float sm_scale;           // 调用方保证非空（host 已做 1/sqrt(d_og) 缺省折叠）
};

/**
 * 量化中间态 dump（golden 位级比对用；须在 RunSequential 之后调用，下次调用失效）
 */
struct TwinQuantDump {
    const int8_t* q_int8 = nullptr;   // [b, hq, sq, d_pad]
    const int8_t* k_int8 = nullptr;   // [b, hkv, sk, d_pad]
    const uint8_t* v_fp8 = nullptr;   // [b, hkv, d_pad, pad_sk_permuted] 与 golden v_fp8 同构
    const float* q_scale = nullptr;
    const float* k_scale = nullptr;
    const float* v_scale = nullptr;   // [b, hkv, d_pad]
    uint32_t d_pad = 0;
    uint32_t pad_sk_v = 0;
};

// 最近一次 RunSequential 的量化中间态（供 C 导出，golden 位级比对用）
inline TwinQuantDump g_last_dump;

namespace detail {

// 输入按 dtype 读取（与 quant 层 Reader 同语义，但 twin 需要独立索引寻址，故经此单点）
inline float ReadInput(const void* base, size_t idx, int dtype_mode) noexcept {
    const auto* bits = static_cast<const uint16_t*>(base) + idx;
    return (dtype_mode == 2) ? quant::Bf16ToFp32(*bits) : quant::Fp16ToFp32(*bits);
}

inline void WriteOutput(void* base, size_t idx, int dtype_mode, float v) noexcept {
    auto* bits = static_cast<uint16_t*>(base) + idx;
    *bits = (dtype_mode == 2) ? quant::Fp32ToBf16(v) : quant::Fp32ToFp16(v);
}

// 四维 (b, h, s, d) 与 (b, s, h, d) 的统一寻址：s/d 两维顺序由布局决定
inline size_t ElemOffset(uint32_t b_idx, uint32_t h_idx, uint32_t s_idx, uint32_t d_idx,
                         uint32_t b, uint32_t h, uint32_t s, uint32_t d, bool is_hnd) noexcept {
    if (is_hnd) {
        return ((static_cast<size_t>(b_idx) * h + h_idx) * s + s_idx) * d + d_idx;
    }
    return ((static_cast<size_t>(b_idx) * s + s_idx) * h + h_idx) * d + d_idx;
}

}  // namespace detail

// -------------------------------------------------------------------------
// 全流水共享语义单点（RunSequential 与设备核 KT 标量基线 include 复用，禁止两处复制）
// -------------------------------------------------------------------------

/**
 * QK^T INT32 点积的反量化 logit（exp2 域，单点定义）：
 * logit = dot · q_scale · k_scale · sm_scale_log2e（乘法顺序固定为左结合，
 * 与设备核 INT8 MMAD + 单点 scale 乘轨迹一致）
 */
inline float DequantQkLogit(int32_t dot, float q_scale, float k_scale,
                            float sm_scale_log2e) noexcept {
    return static_cast<float>(dot) * q_scale * k_scale * sm_scale_log2e;
}

/**
 * P 矩阵标量量化单点（twin 5.4 与设备核 ComputeSoftmaxProbability 共享）：
 * P = FastExp2(logit_exp2 - m') · FP8_E4M3_MAX —— 448· 放大域经乘法实现（twin/golden
 * 轨迹），块行 max 元素恰为 448，填满 e4m3 动态范围后饱和截断 + RNE 转 e4m3 码。
 * m' 为锚点（基线 = 行锚 m_scale，强制逐块重缩放下恒等于 m_run）。
 * 指数偏置 S_FP8_OFFSET 变体属非默认可选：启用须同时关闭 skip-scale（tau=0），
 * 由调用方把偏置折叠进 m' 后传入。
 */
inline uint8_t SoftmaxProbCode(float logit_exp2, float m_prime) noexcept {
    float p = OnlineSoftmaxTransformer::FastExp2(logit_exp2 - m_prime) * FP8_E4M3_MAX;
    p = std::clamp(p, -FP8_E4M3_MAX, FP8_E4M3_MAX);
    return quant::FloatToE4m3fn(p);
}

/** SoftmaxProbCode 的反量化视图（e4m3 码 → float，供标量 PV 累加） */
inline float SoftmaxProbDequant(float logit_exp2, float m_prime) noexcept {
    return quant::E4m3fnToFloat(SoftmaxProbCode(logit_exp2, m_prime));
}

/**
 * 块内 exp2 和（twin 5.2 单点）：Σ_未掩码j FastExp2(s_row[j] - cur_max) · FP8_E4M3_MAX，
 * 与 P 量化同域（448·），供 Step/ConvertLseToHost 同域相消；掩码列不进分母。
 * s_row 以绝对 token 序号减 s_base 寻址（twin 传全行缓冲 s_base=0）。
 */
inline float OnlineBlockExp2Sum(const float* s_row, uint32_t s_base,
                                uint32_t j_beg, uint32_t j_end,
                                bool is_causal, uint32_t token_q, float cur_max) noexcept {
    float cur_sum = 0.0f;
    for (uint32_t j = j_beg; j < j_end; ++j) {
        if (is_causal && static_cast<int64_t>(j) > static_cast<int64_t>(token_q)) {
            continue;
        }
        cur_sum += OnlineSoftmaxTransformer::FastExp2(s_row[j - s_base] - cur_max) * FP8_E4M3_MAX;
    }
    return cur_sum;
}

/**
 * 单行·单块 P 量化 + PV 累加（twin 5.4 的唯一定义；设备核 KT 标量基线经 include
 * 复用本轨迹，内核内禁止复制）。
 * P 以 m'（锚点 - 可选指数偏置）量化为 e4m3 后反量化累加：
 * - fp32 档：逐 token 直累 o_row[c] += p8 · v(j,c)（p8==0 跳过）；
 * - fp32+fp16 档：按 k_step 子块在 FP16 域累加（乘数双侧舍入 FP16），子块和舍入
 *   FP16 后并入 fp32 主累加器（对齐 FixPipe 随路量化轨迹）。
 * v_at(绝对token j, 通道 c) 返回反量化 V 值（e4m3 解码 · 通道 scale）。
 */
template <typename VAt>
inline void QuantizePAndAccumulatePvRow(
    float* o_row,
    uint32_t d_pad,
    const float* s_row,
    uint32_t s_base,
    uint32_t j_beg,
    uint32_t j_end,
    bool is_causal,
    uint32_t token_q,
    float m_prime,
    bool fp16_accum,
    uint32_t k_step,
    const VAt& v_at) {
    if (fp16_accum) {
        for (uint32_t ks_beg = j_beg; ks_beg < j_end; ks_beg += k_step) {
            const uint32_t ks_fin = (ks_beg + k_step < j_end) ? ks_beg + k_step : j_end;
            for (uint32_t c = 0; c < d_pad; ++c) {
                float sub_sum = 0.0f;
                for (uint32_t j = ks_beg; j < ks_fin; ++j) {
                    if (is_causal && static_cast<int64_t>(j) > static_cast<int64_t>(token_q)) {
                        continue;
                    }
                    const float p8 = SoftmaxProbDequant(s_row[j - s_base], m_prime);
                    sub_sum += quant::Fp16ToFp32(quant::Fp32ToFp16(p8)) *
                               quant::Fp16ToFp32(quant::Fp32ToFp16(v_at(j, c)));
                }
                o_row[c] += quant::Fp16ToFp32(quant::Fp32ToFp16(sub_sum));
            }
        }
    } else {
        for (uint32_t j = j_beg; j < j_end; ++j) {
            if (is_causal && static_cast<int64_t>(j) > static_cast<int64_t>(token_q)) {
                continue;
            }
            const float p8 = SoftmaxProbDequant(s_row[j - s_base], m_prime);
            if (p8 == 0.0f) continue;
            for (uint32_t c = 0; c < d_pad; ++c) {
                o_row[c] += p8 * v_at(j, c);
            }
        }
    }
}

/**
 * 顺序执行全流水：pad → (km) → 量化 → QK^T → online softmax → FP8 PV → 归一化与 LSE。
 * 返回 0 成功；非 0 为不支持的配置（当前拒绝 D>128）。
 */
inline int RunSequential(const TwinIo& io, TwinQuantDump* dump = nullptr) {
    using quant::E4m3fnToFloat;
    using quant::FloatToE4m3fn;

    if (io.d_og == 0 || io.d_og > 128) {
        return 1;
    }
    const uint32_t d_pad = (io.d_og <= 64) ? 64 : 128;
    const bool is_hnd = io.tensor_layout == LAYOUT_HND;
    const bool per_thread = io.qk_quant_gran == QUANT_GRAN_PER_THREAD;
    const bool smooth_k = io.smooth_k != 0;
    const bool fp16_accum = io.pv_accum_dtype == PV_ACCUM_FP32_FP16;
    const size_t total_q = static_cast<size_t>(io.b) * io.hq * io.sq * d_pad;
    const size_t total_kv = static_cast<size_t>(io.b) * io.hkv * io.sk * d_pad;

    // 1. D 维补零（对齐 prototype 的 F.pad(q/k/v, (0, pad_d))）
    std::vector<uint16_t> q_pad(total_q), k_pad(total_kv), v_pad(total_kv);
    for (uint32_t bi = 0; bi < io.b; ++bi) {
        for (uint32_t hi = 0; hi < io.hq; ++hi) {
            for (uint32_t si = 0; si < io.sq; ++si) {
                for (uint32_t di = 0; di < d_pad; ++di) {
                    const size_t src = detail::ElemOffset(bi, hi, si, di, io.b, io.hq, io.sq, io.d_og, is_hnd);
                    const size_t dst = detail::ElemOffset(bi, hi, si, di, io.b, io.hq, io.sq, d_pad, true);
                    q_pad[dst] = (di < io.d_og)
                        ? static_cast<const uint16_t*>(io.q)[src]
                        : static_cast<uint16_t>(0);
                }
            }
        }
        for (uint32_t hi = 0; hi < io.hkv; ++hi) {
            for (uint32_t si = 0; si < io.sk; ++si) {
                for (uint32_t di = 0; di < d_pad; ++di) {
                    const size_t src = detail::ElemOffset(bi, hi, si, di, io.b, io.hkv, io.sk, io.d_og, is_hnd);
                    const size_t dst_k = detail::ElemOffset(bi, hi, si, di, io.b, io.hkv, io.sk, d_pad, true);
                    k_pad[dst_k] = (di < io.d_og)
                        ? static_cast<const uint16_t*>(io.k)[src]
                        : static_cast<uint16_t>(0);
                    v_pad[dst_k] = (di < io.d_og)
                        ? static_cast<const uint16_t*>(io.v)[src]
                        : static_cast<uint16_t>(0);
                }
            }
        }
    }

    // 2. smooth_k 的 km（double 累加求均值后舍入回输入 dtype，对齐 torch.mean 行为）
    std::vector<float> km;
    if (smooth_k) {
        if (io.key_mean != nullptr) {
            km.assign(io.key_mean, io.key_mean + static_cast<size_t>(io.b) * io.hkv * d_pad);
        } else {
            km.resize(static_cast<size_t>(io.b) * io.hkv * d_pad);
            for (uint32_t bi = 0; bi < io.b; ++bi) {
                for (uint32_t hi = 0; hi < io.hkv; ++hi) {
                    for (uint32_t di = 0; di < d_pad; ++di) {
                        double sum = 0.0;
                        for (uint32_t si = 0; si < io.sk; ++si) {
                            sum += detail::ReadInput(
                                k_pad.data(),
                                detail::ElemOffset(bi, hi, si, di, io.b, io.hkv, io.sk, d_pad, true),
                                io.dtype_mode);
                        }
                        const float mean = (io.sk > 0) ? static_cast<float>(sum / io.sk) : 0.0f;
                        const uint16_t rounded = (io.dtype_mode == 2)
                            ? quant::Fp32ToBf16(mean)
                            : quant::Fp32ToFp16(mean);
                        km[(static_cast<size_t>(bi) * io.hkv + hi) * d_pad + di] =
                            (io.dtype_mode == 2) ? quant::Bf16ToFp32(rounded) : quant::Fp16ToFp32(rounded);
                    }
                }
            }
        }
    }

    // 3. 量化（quant 层唯一定义；内部自带 128/64 块的 scale 分组与补零语义）
    const uint32_t q_scales = quant::QScaleCount(io.sq, per_thread);
    const uint32_t k_scales = quant::KScaleCount(io.sk, per_thread);
    const uint32_t pad_sk_v = ((io.sk + 63) / 64) * 64;
    // static：SageAttnTwinQuantDump 导出的指针在下一次调用前保持有效
    static std::vector<int8_t> q_int8, k_int8;
    static std::vector<uint8_t> v_fp8;
    static std::vector<float> q_scale, k_scale, v_scale;
    q_int8.assign(total_q, 0);
    k_int8.assign(total_kv, 0);
    v_fp8.assign(static_cast<size_t>(io.b) * io.hkv * d_pad * pad_sk_v, 0);
    q_scale.assign(static_cast<size_t>(io.b) * io.hq * q_scales, 0.0f);
    k_scale.assign(static_cast<size_t>(io.b) * io.hkv * k_scales, 0.0f);
    v_scale.assign(static_cast<size_t>(io.b) * io.hkv * d_pad, 0.0f);

    QuantizeQInt8Ascend(q_pad.data(), q_int8.data(), q_scale.data(), io.b, io.hq, io.sq, d_pad,
                        io.qk_quant_gran, LAYOUT_HND, io.dtype_mode);
    QuantizeKInt8Ascend(k_pad.data(), smooth_k ? km.data() : nullptr, k_int8.data(), k_scale.data(),
                        io.b, io.hkv, io.sk, d_pad, io.qk_quant_gran, smooth_k ? 1 : 0,
                        LAYOUT_HND, io.dtype_mode);
    QuantizeVFp8Ascend(v_pad.data(), v_fp8.data(), v_scale.data(), nullptr, io.b, io.hkv, io.sk,
                       d_pad, io.pv_accum_dtype, 0, LAYOUT_HND, io.dtype_mode);

    // 4. V 反量化（e4m3 位型转 float 供 PV 使用）
    //    v_fp8 物理布局 [b, hkv, d_pad, pad_sk_v]：段内物理槽 p 存放 token V_PERM_PATTERN[p]
    auto v_deq = [&](uint32_t bi, uint32_t hi, uint32_t di, uint32_t j) -> float {
        const uint32_t slot = (j / 16) * 16 + quant::V_PERM_INV_PATTERN[j % 16];
        const size_t off = ((static_cast<size_t>(bi) * io.hkv + hi) * d_pad + di) * pad_sk_v + slot;
        return E4m3fnToFloat(v_fp8[off]) * v_scale[(static_cast<size_t>(bi) * io.hkv + hi) * d_pad + di];
    };

    // 5. 主循环：逐 (b, hq) 头，QK^T → online softmax（锚点纪律）→ FP8 PV → 归一化
    //    online softmax 状态为行级（每个 query 行独立维护 max/denominator/O 行）
    const float sm_scale_log2e = io.sm_scale * LOG2E;
    const uint32_t gqa = io.hq / io.hkv;
    std::vector<float> o_acc(io.sq * d_pad);
    std::vector<float> s_row(io.sk);
    std::vector<float> corr_row(io.sq);
    std::vector<float> inv_row(io.sq);

    for (uint32_t bi = 0; bi < io.b; ++bi) {
        for (uint32_t hi = 0; hi < io.hq; ++hi) {
            const uint32_t kh = hi / gqa;
            const int8_t* q_head = q_int8.data() + (static_cast<size_t>(bi) * io.hq + hi) * io.sq * d_pad;
            const float* qs_head = q_scale.data() + (static_cast<size_t>(bi) * io.hq + hi) * q_scales;
            const int8_t* k_head = k_int8.data() + (static_cast<size_t>(bi) * io.hkv + kh) * io.sk * d_pad;
            const float* ks_head = k_scale.data() + (static_cast<size_t>(bi) * io.hkv + kh) * k_scales;

            // smooth_k 的 LSE 校正项（行级 lse_correction = q_i · km，积和按输入 dtype 舍入）
            if (smooth_k) {
                const float* km_head = km.data() + (static_cast<size_t>(bi) * io.hkv + kh) * d_pad;
                for (uint32_t i = 0; i < io.sq; ++i) {
                    float corr = 0.0f;
                    for (uint32_t c = 0; c < d_pad; ++c) {
                        corr += detail::ReadInput(
                            q_pad.data(), detail::ElemOffset(bi, hi, i, c, io.b, io.hq, io.sq, d_pad, true),
                            io.dtype_mode) * km_head[c];
                    }
                    corr_row[i] = corr;
                }
            }

            std::fill(o_acc.begin(), o_acc.end(), 0.0f);

            // cta_k = 64：P 的 e4m3 量化以"块级 running max"为锚，量化轨迹依赖分块
            // 结构（golden/generator 语义，亦即设备核 KV 流水分块语义）；故孪生必须
            // 逐 64-token 块在线推进，不能整行单步归一。
            constexpr uint32_t CTA_K = 64;
            const uint32_t num_cta = (io.sk + CTA_K - 1) / CTA_K;

            // 常量唯一定义纪律：S_FP8_OFFSET（P 放大偏置 log2(448)）、FP8_E4M3_MAX
            // （P 放大后饱和的 e4m3 满量程，即 2^S_FP8_OFFSET）、DEFAULT_K_STEP（fp16
            // 子块归约周期）均引用 tiling.h 唯一定义点，消灭本地多重别名与冗余定义。
            for (uint32_t i = 0; i < io.sq; ++i) {
                const uint32_t q_grp = quant::QScaleIndex(i, per_thread);
                float* o_row = o_acc.data() + static_cast<size_t>(i) * d_pad;
                OnlineSoftmaxState state;
                state.Reset();

                for (uint32_t blk = 0; blk < num_cta; ++blk) {
                    const uint32_t j_beg = blk * CTA_K;
                    const uint32_t j_end = (j_beg + CTA_K < io.sk) ? j_beg + CTA_K : io.sk;

                    // 5.1 块内 logits（INT32 精确点积 + 单点反量化；与设备核 INT8 MMAD 同轨迹）
                    float cur_max = -1.0e30f;
                    for (uint32_t j = j_beg; j < j_end; ++j) {
                        if (io.is_causal && static_cast<int64_t>(j) > static_cast<int64_t>(i)) {
                            s_row[j] = -1.0e30f;  // 掩码位置记负无穷且不参与 max/sum
                            continue;
                        }
                        int32_t dot = 0;
                        const int8_t* qr = q_head + static_cast<size_t>(i) * d_pad;
                        const int8_t* kr = k_head + static_cast<size_t>(j) * d_pad;
                        for (uint32_t c = 0; c < d_pad; ++c) {
                            dot += qr[c] * kr[c];
                        }
                        s_row[j] = DequantQkLogit(dot, qs_head[q_grp],
                                                  ks_head[quant::KScaleIndex(j, per_thread)],
                                                  sm_scale_log2e);
                        if (s_row[j] > cur_max) cur_max = s_row[j];
                    }
                    if (cur_max <= -1.0e30f) {
                        continue;  // 整块被因果掩码（前缀行），无贡献
                    }

                    // 5.2 块内 exp2 和（相对块内 max；掩码列不进分母；448· 放大域）——单点
                    const float cur_sum =
                        OnlineBlockExp2Sum(s_row.data(), 0, j_beg, j_end, io.is_causal != 0, i, cur_max);

                    // 5.3 在线状态步进（强制重缩放：对齐 golden 的逐块锚定轨迹）
                    const float rescale =
                        OnlineSoftmaxTransformer::Step(state, cur_max, cur_sum, true);
                    if (state.need_rescale && std::fabs(rescale - 1.0f) > SCALE_ONE_TOLERANCE) {
                        for (uint32_t c = 0; c < d_pad; ++c) {
                            o_row[c] *= rescale;
                        }
                    }

                    // 5.4 P 量化（e4m3，448· 放大域）+ PV 累加——单点
                    //     fp32+fp16 档：按 k_step (16) 子块在 FP16 域累加后舍入回 FP16 并入主累加器
                    QuantizePAndAccumulatePvRow(
                        o_row, d_pad, s_row.data(), 0, j_beg, j_end, io.is_causal != 0, i,
                        state.m_scale, fp16_accum, DEFAULT_K_STEP,
                        [&](uint32_t j, uint32_t c) { return v_deq(bi, kh, c, j); });
                }

                // 5.5 行级分母倒数与 LSE
                inv_row[i] = OnlineSoftmaxTransformer::GetFinalInvDenominator(state);
                if (io.lse != nullptr) {
                    io.lse[(static_cast<size_t>(bi) * io.hq + hi) * io.sq + i] =
                        OnlineSoftmaxTransformer::ConvertLseToHost(
                            state.lse, io.sm_scale, corr_row[i], smooth_k, S_FP8_OFFSET);
                }            }

            // 5.6 归一化（行级 1/d）、裁剪回 d_og、写回输出
            for (uint32_t i = 0; i < io.sq; ++i) {
                const float* o_row = o_acc.data() + static_cast<size_t>(i) * d_pad;
                for (uint32_t c = 0; c < io.d_og; ++c) {
                    detail::WriteOutput(io.out,
                                        detail::ElemOffset(bi, hi, i, c, io.b, io.hq, io.sq,
                                                           io.d_og, is_hnd),
                                        io.dtype_mode, o_row[c] * inv_row[i]);
                }
            }
        }
    }

    // 静态导出（SageAttnTwinQuantDump 读取；向量本体为 static，指针长期有效）
    g_last_dump.q_int8 = q_int8.data();
    g_last_dump.k_int8 = k_int8.data();
    g_last_dump.v_fp8 = v_fp8.data();
    g_last_dump.q_scale = q_scale.data();
    g_last_dump.k_scale = k_scale.data();
    g_last_dump.v_scale = v_scale.data();
    g_last_dump.d_pad = d_pad;
    g_last_dump.pad_sk_v = pad_sk_v;
    if (dump != nullptr) {
        *dump = g_last_dump;
        dump->k_int8 = k_int8.data();
        dump->v_fp8 = v_fp8.data();
        dump->q_scale = q_scale.data();
        dump->k_scale = k_scale.data();
        dump->v_scale = v_scale.data();
        dump->d_pad = d_pad;
        dump->pad_sk_v = pad_sk_v;
    }
    return 0;
}


/**
 * 导出最近一次 RunSequential 的量化中间态（静态缓冲，golden 位级比对用）
 */
inline void ExportQuantDump(const signed char** q8, const signed char** k8,
                            const unsigned char** v8, const float** qs,
                            const float** ks, const float** vs,
                            unsigned* d_pad, unsigned* pad_sk_v) noexcept {
    // 静态缓冲由 RunSequential 维护；此处经 TwinQuantDump 读取
    *q8 = reinterpret_cast<const signed char*>(g_last_dump.q_int8);
    *k8 = reinterpret_cast<const signed char*>(g_last_dump.k_int8);
    *v8 = g_last_dump.v_fp8;
    *qs = g_last_dump.q_scale;
    *ks = g_last_dump.k_scale;
    *vs = g_last_dump.v_scale;
    *d_pad = g_last_dump.d_pad;
    *pad_sk_v = g_last_dump.pad_sk_v;
}

}  // namespace twin
}  // namespace sage_attn

#endif  // SAGE_ATTN_FP8_ASCENDC_TWIN_REFERENCE_H
