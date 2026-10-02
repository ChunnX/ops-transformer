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

#ifndef SAGE_ATTN_QUANT_HOST_H
#define SAGE_ATTN_QUANT_HOST_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#include "../sageattn_fp8/ascendc/tiling.h"

namespace sage_attn {
namespace quant {

/**
 * 数学与硬件常量定义（对齐 task/01_design.md 2.4-2.7 节）
 */
constexpr float LOG2E = 1.4426950408889634f;                 // log2(e) 缩放常数
constexpr float INV_LOG2E = 0.6931471805599453f;             // 1.0f / log2(e) = ln(2)
constexpr float S_FP8_OFFSET = 8.807354922057604f;           // log2(448.0f) P 矩阵 e4m3 偏置
constexpr float SCALE_MAX_FP16_ACCUM = 2.25f;                // fp32+fp16 档 V 量化 scale_max
constexpr float SCALE_MAX_FP32_ACCUM = 448.0f;               // 其余档位 V 量化 scale_max
constexpr float INT8_QUANT_MAX = 127.0f;                     // INT8 对称量化上限
constexpr float FP8_E4M3_MAX = 448.0f;                       // FP8 e4m3fn 最大有限值
constexpr float SCALE_EPSILON = 1e-7f;                       // INT8 量化 scale 兜底微量
constexpr uint32_t ALIGN_BYTES_32 = 32;                      // MTE2 / UB 32 字节对齐基准

/**
 * 1. sm_scale_log2e 常量折叠与默认 sm_scale 计算
 * 将 1.0f / sqrt(d) 与 log2(e) 在 Host 侧预先乘算折叠，消除核内反复乘法开销
 */
inline float GetDefaultSmScale(uint32_t head_dim_og) {
    if (head_dim_og == 0) return 1.0f;
    return 1.0f / std::sqrt(static_cast<float>(head_dim_og));
}

inline float FoldSmScaleLog2e(float sm_scale) {
    return sm_scale * LOG2E;
}

inline float GetVScaleMax(uint32_t pv_accum_dtype) {
    return (pv_accum_dtype == PV_ACCUM_FP32_FP16) ? SCALE_MAX_FP16_ACCUM : SCALE_MAX_FP32_ACCUM;
}

inline uint32_t GetPaddedHeadDim(uint32_t head_dim_og) {
    if (head_dim_og <= 64) {
        return 64;
    } else if (head_dim_og <= 128) {
        return 128;
    }
    return ((head_dim_og + 63) / 64) * 64;
}

/**
 * 2. 量化 Scale 数量与内存大小计算
 */
inline uint32_t GetQScaleCount(uint32_t sq, uint32_t qk_quant_gran) {
    uint32_t num_blkq = (sq + 127) / 128;
    return (qk_quant_gran == QUANT_GRAN_PER_THREAD) ? (num_blkq * 32) : (num_blkq * 4);
}

inline uint32_t GetKScaleCount(uint32_t sk, uint32_t qk_quant_gran) {
    uint32_t num_blkk = (sk + 63) / 64;
    return (qk_quant_gran == QUANT_GRAN_PER_THREAD) ? (num_blkk * 4) : (num_blkk * 1);
}

inline uint32_t GetVScaleCount(uint32_t d) {
    return d;
}

inline uint32_t Align32(uint32_t size) {
    return ((size + ALIGN_BYTES_32 - 1) / ALIGN_BYTES_32) * ALIGN_BYTES_32;
}

/**
 * 3. Data + Scale 相邻紧凑排布布局计算（MTE2 合并搬运请求契约）
 * KV 数据块在连续内存后直接跟 FP32 scale，尾部 32B 对齐
 */
inline uint32_t GetCompactKvBlockBytes(uint32_t block_k, uint32_t d, uint32_t qk_quant_gran) {
    uint32_t data_bytes = block_k * d * sizeof(int8_t);
    uint32_t num_scales = (qk_quant_gran == QUANT_GRAN_PER_THREAD) ? 4 : 1;
    uint32_t scale_bytes = num_scales * sizeof(float);
    return Align32(data_bytes + scale_bytes);
}

inline uint32_t GetCompactVBlockBytes(uint32_t block_k, uint32_t d) {
    // NZ 分形中每行末尾 pad 32B 防 UB bank 冲突
    uint32_t row_bytes = Align32(d * sizeof(uint8_t));
    uint32_t data_bytes = block_k * row_bytes;
    uint32_t scale_bytes = d * sizeof(float);
    return Align32(data_bytes + scale_bytes);
}

/**
 * 4. smooth_k 均值计算 (ComputeKm)
 * 计算 Key 沿序列维度的均值 km = mean(k, dim=seq_dim)
 * 支持 HND [b, hkv, sk, d] 与 NHD [b, sk, hkv, d] 两种布局
 * km_out 形状: [b, hkv, d]
 */
template <typename T>
void ComputeKm(
    const T* k_ptr,
    float* km_out,
    uint32_t b,
    uint32_t hkv,
    uint32_t sk,
    uint32_t d,
    uint32_t tensor_layout
) {
    if (sk == 0) return;
    float inv_sk = 1.0f / static_cast<float>(sk);

    for (uint32_t b_idx = 0; b_idx < b; ++b_idx) {
        for (uint32_t h_idx = 0; h_idx < hkv; ++h_idx) {
            float* km_head = km_out + (b_idx * hkv + h_idx) * d;
            std::fill(km_head, km_head + d, 0.0f);

            for (uint32_t s_idx = 0; s_idx < sk; ++s_idx) {
                const T* k_row = nullptr;
                if (tensor_layout == LAYOUT_HND) {
                    k_row = k_ptr + ((b_idx * hkv + h_idx) * sk + s_idx) * d;
                } else {
                    k_row = k_ptr + ((b_idx * sk + s_idx) * hkv + h_idx) * d;
                }

                for (uint32_t c = 0; c < d; ++c) {
                    km_head[c] += static_cast<float>(k_row[c]);
                }
            }

            for (uint32_t c = 0; c < d; ++c) {
                km_head[c] *= inv_sk;
            }
        }
    }
}

/**
 * 5. lse_correction 计算 (ComputeLseCorrection)
 * 公式: lse_correction = Q * km_broadcast^T (沿通道维求内积)
 * 支持 GQA 广播: rep = hq / hkv, km_broadcast 按 head 广播
 * lse_corr_out 形状: [b, hq, sq]
 */
template <typename T>
void ComputeLseCorrection(
    const T* q_ptr,
    const float* km_ptr,
    float* lse_corr_out,
    uint32_t b,
    uint32_t hq,
    uint32_t hkv,
    uint32_t sq,
    uint32_t d,
    uint32_t tensor_layout
) {
    uint32_t rep = (hkv > 0) ? (hq / hkv) : 1;

    for (uint32_t b_idx = 0; b_idx < b; ++b_idx) {
        for (uint32_t hq_idx = 0; hq_idx < hq; ++hq_idx) {
            uint32_t hkv_idx = hq_idx / rep;
            const float* km_vec = km_ptr + (b_idx * hkv + hkv_idx) * d;
            float* lse_head = lse_corr_out + (b_idx * hq + hq_idx) * sq;

            for (uint32_t s_idx = 0; s_idx < sq; ++s_idx) {
                const T* q_row = nullptr;
                if (tensor_layout == LAYOUT_HND) {
                    q_row = q_ptr + ((b_idx * hq + hq_idx) * sq + s_idx) * d;
                } else {
                    q_row = q_ptr + ((b_idx * sq + s_idx) * hq + hq_idx) * d;
                }

                float dot = 0.0f;
                for (uint32_t c = 0; c < d; ++c) {
                    dot += static_cast<float>(q_row[c]) * km_vec[c];
                }
                lse_head[s_idx] = dot;
            }
        }
    }
}

/**
 * 6. LSE 对外返回前换算 (ConvertLseExp2ToOut)
 * 公式: lse_out = lse_exp2 / 1.44269504 + (smooth_k ? lse_correction * sm_scale : 0.0)
 */
inline void ConvertLseExp2ToOut(
    const float* lse_exp2_ptr,
    const float* lse_corr_ptr,
    float* lse_out,
    uint32_t total_elements,
    float sm_scale,
    bool smooth_k
) {
    for (uint32_t i = 0; i < total_elements; ++i) {
        float base_lse = lse_exp2_ptr[i] * INV_LOG2E;
        if (smooth_k && lse_corr_ptr != nullptr) {
            base_lse += lse_corr_ptr[i] * sm_scale;
        }
        lse_out[i] = base_lse;
    }
}

} // namespace quant
} // namespace sage_attn

#endif // SAGE_ATTN_QUANT_HOST_H
