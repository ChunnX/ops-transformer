#include "quant_host.h"
#include "quant_common.h"

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

namespace sage_attn {
namespace quant {

// 舍入原语 RoundHalfAwayFromZero / RoundNearestEven 的唯一定义位于 quant_common.h

/**
 * AscendC Vector Core 独立 INT8 量化算子实现类 (V0 核心量化 Prologue)
 */
class AscendCQuantInt8Kernel {
public:
    __aicore__ AscendCQuantInt8Kernel() {}

    /**
     * Q 矩阵 INT8 量化主函数
     * 支持 per_warp (BLKQ=128, WARPQ=32, 4 scales/块) 与
     *      per_thread (BLKQ=128, 4 子块 x 8 组交织 token, 32 scales/块)
     */
    template <typename T, typename Reader = ElementReader<T>>
    void ProcessQ(
        const T* q_in,
        int8_t* q_out,
        float* q_scale_out,
        uint32_t b,
        uint32_t hq,
        uint32_t sq,
        uint32_t d,
        uint32_t qk_quant_gran,
        uint32_t tensor_layout
    ) {
        constexpr uint32_t BLKQ = 128;
        uint32_t num_blkq = (sq + BLKQ - 1) / BLKQ;

        for (uint32_t b_idx = 0; b_idx < b; ++b_idx) {
            for (uint32_t h_idx = 0; h_idx < hq; ++h_idx) {
                float* cur_scale_head = q_scale_out + (static_cast<size_t>(b_idx) * hq + h_idx) * QScaleCount(sq, qk_quant_gran == QUANT_GRAN_PER_THREAD);

                for (uint32_t blk_idx = 0; blk_idx < num_blkq; ++blk_idx) {
                    uint32_t token_blk_start = blk_idx * BLKQ;

                    if (qk_quant_gran == QUANT_GRAN_PER_THREAD) {
                        // per_thread: 128 token 分 4 个 32 token 子块，每个子块分 8 组步长 8 交织 token
                        for (uint32_t sub_idx = 0; sub_idx < 4; ++sub_idx) {
                            uint32_t sub_start = token_blk_start + sub_idx * 32;

                            for (uint32_t g = 0; g < 8; ++g) {
                                float amax = 0.0f;

                                // 遍历该组的 4 个交织 token: sub_start + p * 8 + g (p = 0..3)
                                for (uint32_t p = 0; p < 4; ++p) {
                                    uint32_t t = sub_start + p * 8 + g;
                                    if (t < sq) {
                                        size_t base_offset = (tensor_layout == LAYOUT_HND) ?
                                            ((static_cast<size_t>(b_idx) * hq + h_idx) * sq + t) * d :
                                            ((static_cast<size_t>(b_idx) * sq + t) * hq + h_idx) * d;

                                        for (uint32_t c = 0; c < d; ++c) {
                                            float abs_v = std::abs(Reader::Read(q_in, base_offset + c));
                                            if (abs_v > amax) amax = abs_v;
                                        }
                                    }
                                }

                                float scale = FinalizeQkScale(amax, true);
                                uint32_t scale_idx = QScaleIndex(sub_start + g, true);
                                cur_scale_head[scale_idx] = scale;

                                // 量化并执行 half-away-from-zero 舍入 (保持除法运算以对齐 GPU ULP 舍入轨迹)
                                for (uint32_t p = 0; p < 4; ++p) {
                                    uint32_t t = sub_start + p * 8 + g;
                                    if (t < sq) {
                                        size_t base_offset = (tensor_layout == LAYOUT_HND) ?
                                            ((static_cast<size_t>(b_idx) * hq + h_idx) * sq + t) * d :
                                            ((static_cast<size_t>(b_idx) * sq + t) * hq + h_idx) * d;

                                        int8_t* row_out = (tensor_layout == LAYOUT_HND) ?
                                            (q_out + ((static_cast<size_t>(b_idx) * hq + h_idx) * sq + t) * d) :
                                            (q_out + ((static_cast<size_t>(b_idx) * sq + t) * hq + h_idx) * d);

                                        for (uint32_t c = 0; c < d; ++c) {
                                            float scaled_v = Reader::Read(q_in, base_offset + c) / scale;
                                            row_out[c] = RoundHalfAwayFromZero(scaled_v);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // per_warp: 128 token 分 4 个 32 token 子块，每 32 token 一个 scale
                        for (uint32_t sub_idx = 0; sub_idx < 4; ++sub_idx) {
                            uint32_t sub_start = token_blk_start + sub_idx * 32;
                            float amax = 0.0f;

                            for (uint32_t i = 0; i < 32; ++i) {
                                uint32_t t = sub_start + i;
                                if (t < sq) {
                                    size_t base_offset = (tensor_layout == LAYOUT_HND) ?
                                        ((static_cast<size_t>(b_idx) * hq + h_idx) * sq + t) * d :
                                        ((static_cast<size_t>(b_idx) * sq + t) * hq + h_idx) * d;

                                    for (uint32_t c = 0; c < d; ++c) {
                                        float abs_v = std::abs(Reader::Read(q_in, base_offset + c));
                                        if (abs_v > amax) amax = abs_v;
                                    }
                                }
                            }

                            float scale = FinalizeQkScale(amax, false);
                            uint32_t scale_idx = QScaleIndex(sub_start, false);
                            cur_scale_head[scale_idx] = scale;

                            // 量化并执行 RNE 舍入
                            for (uint32_t i = 0; i < 32; ++i) {
                                uint32_t t = sub_start + i;
                                if (t < sq) {
                                    size_t base_offset = (tensor_layout == LAYOUT_HND) ?
                                        ((static_cast<size_t>(b_idx) * hq + h_idx) * sq + t) * d :
                                        ((static_cast<size_t>(b_idx) * sq + t) * hq + h_idx) * d;

                                    int8_t* row_out = (tensor_layout == LAYOUT_HND) ?
                                        (q_out + ((static_cast<size_t>(b_idx) * hq + h_idx) * sq + t) * d) :
                                        (q_out + ((static_cast<size_t>(b_idx) * sq + t) * hq + h_idx) * d);

                                    for (uint32_t c = 0; c < d; ++c) {
                                        float scaled_v = Reader::Read(q_in, base_offset + c) / scale;
                                        row_out[c] = RoundNearestEven(scaled_v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * K 矩阵 INT8 量化主函数 (融合 smooth_k 减均值)
     * 支持 per_warp (BLKK=64, 1 scale/块) 与
     *      per_thread (BLKK=64, 4 组相邻 2 列 token, 4 scales/块)
     */
    template <typename T, typename Reader = ElementReader<T>>
    void ProcessK(
        const T* k_in,
        const float* km_in,
        int8_t* k_out,
        float* k_scale_out,
        uint32_t b,
        uint32_t hkv,
        uint32_t sk,
        uint32_t d,
        uint32_t qk_quant_gran,
        bool smooth_k,
        uint32_t tensor_layout
    ) {
        constexpr uint32_t BLKK = 64;
        uint32_t num_blkk = (sk + BLKK - 1) / BLKK;

        for (uint32_t b_idx = 0; b_idx < b; ++b_idx) {
            for (uint32_t h_idx = 0; h_idx < hkv; ++h_idx) {
                const float* km_vec = (smooth_k && km_in != nullptr) ? (km_in + (static_cast<size_t>(b_idx) * hkv + h_idx) * d) : nullptr;
                float* cur_scale_head = k_scale_out + (static_cast<size_t>(b_idx) * hkv + h_idx) * KScaleCount(sk, qk_quant_gran == QUANT_GRAN_PER_THREAD);

                for (uint32_t blk_idx = 0; blk_idx < num_blkk; ++blk_idx) {
                    uint32_t token_blk_start = blk_idx * BLKK;

                    if (qk_quant_gran == QUANT_GRAN_PER_THREAD) {
                        // per_thread: 64 token 分 4 组，组 g 含 8 对相邻 token (n, n+1)，其中 n mod 8 == 2g
                        for (uint32_t g = 0; g < 4; ++g) {
                            float amax = 0.0f;

                            // 扫描该组的 8 对相邻 token (共 16 个 token)
                            for (uint32_t p = 0; p < 8; ++p) {
                                for (uint32_t pair_elem = 0; pair_elem < 2; ++pair_elem) {
                                    uint32_t t = token_blk_start + (p * 8 + 2 * g + pair_elem);
                                    if (t < sk) {
                                        size_t base_offset = (tensor_layout == LAYOUT_HND) ?
                                            ((static_cast<size_t>(b_idx) * hkv + h_idx) * sk + t) * d :
                                            ((static_cast<size_t>(b_idx) * sk + t) * hkv + h_idx) * d;

                                        for (uint32_t c = 0; c < d; ++c) {
                                            float raw_val = Reader::Read(k_in, base_offset + c);
                                            float val = (km_vec != nullptr) ? Reader::Sub(raw_val, km_vec[c]) : raw_val;
                                            float abs_v = std::abs(val);
                                            if (abs_v > amax) amax = abs_v;
                                        }
                                    }
                                }
                            }

                            float scale = FinalizeQkScale(amax, true);
                            uint32_t scale_idx = KScaleIndex(token_blk_start + 2 * g, true);
                            cur_scale_head[scale_idx] = scale;

                            // 量化并执行 half-away-from-zero 舍入
                            for (uint32_t p = 0; p < 8; ++p) {
                                for (uint32_t pair_elem = 0; pair_elem < 2; ++pair_elem) {
                                    uint32_t t = token_blk_start + (p * 8 + 2 * g + pair_elem);
                                    if (t < sk) {
                                        size_t base_offset = (tensor_layout == LAYOUT_HND) ?
                                            ((static_cast<size_t>(b_idx) * hkv + h_idx) * sk + t) * d :
                                            ((static_cast<size_t>(b_idx) * sk + t) * hkv + h_idx) * d;

                                        int8_t* row_out = (tensor_layout == LAYOUT_HND) ?
                                            (k_out + ((static_cast<size_t>(b_idx) * hkv + h_idx) * sk + t) * d) :
                                            (k_out + ((static_cast<size_t>(b_idx) * sk + t) * hkv + h_idx) * d);

                                        for (uint32_t c = 0; c < d; ++c) {
                                            float raw_val = Reader::Read(k_in, base_offset + c);
                                            float val = (km_vec != nullptr) ? Reader::Sub(raw_val, km_vec[c]) : raw_val;
                                            float scaled_v = val / scale;
                                            row_out[c] = RoundHalfAwayFromZero(scaled_v);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // per_warp (per_block): 64 token 块共享 1 个 scale
                        float amax = 0.0f;
                        for (uint32_t i = 0; i < 64; ++i) {
                            uint32_t t = token_blk_start + i;
                            if (t < sk) {
                                size_t base_offset = (tensor_layout == LAYOUT_HND) ?
                                    ((static_cast<size_t>(b_idx) * hkv + h_idx) * sk + t) * d :
                                    ((static_cast<size_t>(b_idx) * sk + t) * hkv + h_idx) * d;

                                for (uint32_t c = 0; c < d; ++c) {
                                    float raw_val = Reader::Read(k_in, base_offset + c);
                                    float val = (km_vec != nullptr) ? Reader::Sub(raw_val, km_vec[c]) : raw_val;
                                    float abs_v = std::abs(val);
                                    if (abs_v > amax) amax = abs_v;
                                }
                            }
                        }

                        float scale = FinalizeQkScale(amax, false);
                        uint32_t scale_idx = KScaleIndex(token_blk_start, false);
                        cur_scale_head[scale_idx] = scale;

                        // 量化并执行 RNE 舍入
                        for (uint32_t i = 0; i < 64; ++i) {
                            uint32_t t = token_blk_start + i;
                            if (t < sk) {
                                size_t base_offset = (tensor_layout == LAYOUT_HND) ?
                                    ((static_cast<size_t>(b_idx) * hkv + h_idx) * sk + t) * d :
                                    ((static_cast<size_t>(b_idx) * sk + t) * hkv + h_idx) * d;

                                int8_t* row_out = (tensor_layout == LAYOUT_HND) ?
                                    (k_out + ((static_cast<size_t>(b_idx) * hkv + h_idx) * sk + t) * d) :
                                    (k_out + ((static_cast<size_t>(b_idx) * sk + t) * hkv + h_idx) * d);

                                for (uint32_t c = 0; c < d; ++c) {
                                    float raw_val = Reader::Read(k_in, base_offset + c);
                                    float val = (km_vec != nullptr) ? Reader::Sub(raw_val, km_vec[c]) : raw_val;
                                    float scaled_v = val / scale;
                                    row_out[c] = RoundNearestEven(scaled_v);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

} // namespace quant
} // namespace sage_attn

/**
 * C-ABI 导出接口
 * dtype_mode: 0 = float32, 1 = float16, 2 = bfloat16
 */
extern "C" {

int QuantizeQInt8Ascend(
    const void* q_in,
    void* q_out,
    float* q_scale_out,
    uint32_t b,
    uint32_t hq,
    uint32_t sq,
    uint32_t d,
    uint32_t qk_quant_gran,
    uint32_t tensor_layout,
    int dtype_mode
) noexcept {
    sage_attn::quant::AscendCQuantInt8Kernel kernel;
    if (dtype_mode == 1) {
        kernel.ProcessQ<uint16_t, sage_attn::quant::HalfReader>(
            reinterpret_cast<const uint16_t*>(q_in),
            reinterpret_cast<int8_t*>(q_out),
            q_scale_out, b, hq, sq, d, qk_quant_gran, tensor_layout);
    } else if (dtype_mode == 2) {
        kernel.ProcessQ<uint16_t, sage_attn::quant::Bf16Reader>(
            reinterpret_cast<const uint16_t*>(q_in),
            reinterpret_cast<int8_t*>(q_out),
            q_scale_out, b, hq, sq, d, qk_quant_gran, tensor_layout);
    } else {
        kernel.ProcessQ<float, sage_attn::quant::ElementReader<float>>(
            reinterpret_cast<const float*>(q_in),
            reinterpret_cast<int8_t*>(q_out),
            q_scale_out, b, hq, sq, d, qk_quant_gran, tensor_layout);
    }
    return 0;
}

int QuantizeKInt8Ascend(
    const void* k_in,
    const float* km_in,
    void* k_out,
    float* k_scale_out,
    uint32_t b,
    uint32_t hkv,
    uint32_t sk,
    uint32_t d,
    uint32_t qk_quant_gran,
    int smooth_k,
    uint32_t tensor_layout,
    int dtype_mode
) noexcept {
    sage_attn::quant::AscendCQuantInt8Kernel kernel;
    if (dtype_mode == 1) {
        kernel.ProcessK<uint16_t, sage_attn::quant::HalfReader>(
            reinterpret_cast<const uint16_t*>(k_in),
            km_in,
            reinterpret_cast<int8_t*>(k_out),
            k_scale_out, b, hkv, sk, d, qk_quant_gran, smooth_k != 0, tensor_layout);
    } else if (dtype_mode == 2) {
        kernel.ProcessK<uint16_t, sage_attn::quant::Bf16Reader>(
            reinterpret_cast<const uint16_t*>(k_in),
            km_in,
            reinterpret_cast<int8_t*>(k_out),
            k_scale_out, b, hkv, sk, d, qk_quant_gran, smooth_k != 0, tensor_layout);
    } else {
        kernel.ProcessK<float, sage_attn::quant::ElementReader<float>>(
            reinterpret_cast<const float*>(k_in),
            km_in,
            reinterpret_cast<int8_t*>(k_out),
            k_scale_out, b, hkv, sk, d, qk_quant_gran, smooth_k != 0, tensor_layout);
    }
    return 0;
}

} // extern "C"
