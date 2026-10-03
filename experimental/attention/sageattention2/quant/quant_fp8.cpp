#include "quant_host.h"
#include "quant_common.h"
#include "catlass/detail/alignment.hpp"

#include <cstdint>
#include <cmath>

namespace sage_attn {
namespace quant {

// FloatToE4m3fn / E4m3fnToFloat / V_PERM_PATTERN 的唯一定义位于 quant_common.h

/**
 * AscendC Vector Core 独立 FP8 量化算子实现类 (V0 核心量化 Prologue)
 */
class AscendCQuantFp8Kernel {
public:
    __aicore__ AscendCQuantFp8Kernel() {}

    /**
     * V 矩阵 per-channel FP8(e4m3) 量化主逻辑
     * 1. 沿序列维 pad 到 64 的倍数 (CTA_SIZE = 64)
     * 2. 16-token 重排 (或 ND2NZ 随路排布)
     * 3. 沿通道维 (head_dim) 计算 scale = amax / scale_max
     *    注意：根据 task/01_design.md 2.7 节，FP8 路径 amax 无 1e-7 兜底
     * 4. 饱和截断 [-scale_max, scale_max] 并 RNE 舍入为 e4m3
     * 5. 若 smooth_v 为 true (仅 fp32 档)，沿序列维减 vm 均值
     */
    template <typename T, typename Reader = ElementReader<T>>
    void ProcessV(
        const T* v_in,
        uint8_t* v_fp8_out,
        float* v_scale_out,
        float* vm_out,
        uint32_t b,
        uint32_t hkv,
        uint32_t sk,
        uint32_t d,
        uint32_t pv_accum_dtype,
        bool smooth_v,
        uint32_t tensor_layout
    ) {
        float scale_max = GetVScaleMax(pv_accum_dtype);
        uint32_t pad_sk_v = ::RoundUp<64u>(sk);
        uint32_t num_16 = pad_sk_v / 16;
        uint32_t fp8_padded_tokens = ::RoundUp<16u>(sk);

        for (uint32_t b_idx = 0; b_idx < b; ++b_idx) {
            for (uint32_t h_idx = 0; h_idx < hkv; ++h_idx) {
                float* cur_scale_head = v_scale_out + (static_cast<size_t>(b_idx) * hkv + h_idx) * d;
                float* cur_vm_head = (smooth_v && vm_out != nullptr) ? (vm_out + (static_cast<size_t>(b_idx) * hkv + h_idx) * d) : nullptr;

                for (uint32_t c = 0; c < d; ++c) {
                    // 1. 均值 vm 计算 (若启用 smooth_v)
                    float vm = 0.0f;
                    if (smooth_v) {
                        float sum_v = 0.0f;
                        for (uint32_t t = 0; t < sk; ++t) {
                            size_t base_offset = (tensor_layout == LAYOUT_HND) ?
                                ((static_cast<size_t>(b_idx) * hkv + h_idx) * sk + t) * d + c :
                                ((static_cast<size_t>(b_idx) * sk + t) * hkv + h_idx) * d + c;
                            sum_v += Reader::Read(v_in, base_offset);
                        }
                        vm = sum_v / static_cast<float>(fp8_padded_tokens);
                        if (cur_vm_head != nullptr) {
                            cur_vm_head[c] = vm;
                        }
                    }

                    // 2. amax 计算 (仅在参与归约的 fp8_padded_tokens 范围内扫描)
                    float amax = 0.0f;
                    for (uint32_t t = 0; t < fp8_padded_tokens; ++t) {
                        float val = 0.0f;
                        if (t < sk) {
                            size_t base_offset = (tensor_layout == LAYOUT_HND) ?
                                ((static_cast<size_t>(b_idx) * hkv + h_idx) * sk + t) * d + c :
                                ((static_cast<size_t>(b_idx) * sk + t) * hkv + h_idx) * d + c;
                            val = Reader::Read(v_in, base_offset);
                        }
                        if (smooth_v) {
                            val -= vm;
                        }
                        float abs_v = std::abs(val);
                        if (abs_v > amax) amax = abs_v;
                    }

                    // 3. scale 计算 (FP8 路径无 1e-7 兜底，保持全零通道生成 NaN 容错)
                    float scale = amax / scale_max;
                    cur_scale_head[c] = scale;

                    // 4. 重排并执行量化写入
                    // 物理输出排布: [b, hkv, d, pad_sk_v]
                    size_t out_head_base = ((static_cast<size_t>(b_idx) * hkv + h_idx) * d + c) * pad_sk_v;

                    for (uint32_t chunk_idx = 0; chunk_idx < num_16; ++chunk_idx) {
                        uint32_t chunk_token_base = chunk_idx * 16;

                        for (uint32_t p = 0; p < 16; ++p) {
                            uint32_t permuted_offset = V_PERM_PATTERN[p];
                            uint32_t orig_token = chunk_token_base + permuted_offset;

                            float val = 0.0f;
                            if (orig_token < sk) {
                                size_t in_offset = (tensor_layout == LAYOUT_HND) ?
                                    ((static_cast<size_t>(b_idx) * hkv + h_idx) * sk + orig_token) * d + c :
                                    ((static_cast<size_t>(b_idx) * sk + orig_token) * hkv + h_idx) * d + c;
                                val = Reader::Read(v_in, in_offset);
                            }

                            if (smooth_v) {
                                val -= vm;
                            }

                            // 直接除以 scale（全零通道 scale=0 时产生 NaN，经 FloatToE4m3fn 转为 0x7f，与 GPU / Golden 严格一致）
                            float scaled_v = val / scale;
                            if (scaled_v > FP8_E4M3_MAX) scaled_v = FP8_E4M3_MAX;
                            if (scaled_v < -FP8_E4M3_MAX) scaled_v = -FP8_E4M3_MAX;

                            uint8_t e4m3_byte = FloatToE4m3fn(scaled_v);
                            v_fp8_out[out_head_base + chunk_token_base + p] = e4m3_byte;
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

int QuantizeVFp8Ascend(
    const void* v_in,
    void* v_fp8_out,
    float* v_scale_out,
    float* vm_out,
    uint32_t b,
    uint32_t hkv,
    uint32_t sk,
    uint32_t d,
    uint32_t pv_accum_dtype,
    int smooth_v,
    uint32_t tensor_layout,
    int dtype_mode
) noexcept {
    sage_attn::quant::AscendCQuantFp8Kernel kernel;
    if (dtype_mode == 1) {
        kernel.ProcessV<uint16_t, sage_attn::quant::HalfReader>(
            reinterpret_cast<const uint16_t*>(v_in),
            reinterpret_cast<uint8_t*>(v_fp8_out),
            v_scale_out, vm_out, b, hkv, sk, d, pv_accum_dtype, smooth_v != 0, tensor_layout);
    } else if (dtype_mode == 2) {
        kernel.ProcessV<uint16_t, sage_attn::quant::Bf16Reader>(
            reinterpret_cast<const uint16_t*>(v_in),
            reinterpret_cast<uint8_t*>(v_fp8_out),
            v_scale_out, vm_out, b, hkv, sk, d, pv_accum_dtype, smooth_v != 0, tensor_layout);
    } else {
        kernel.ProcessV<float, sage_attn::quant::ElementReader<float>>(
            reinterpret_cast<const float*>(v_in),
            reinterpret_cast<uint8_t*>(v_fp8_out),
            v_scale_out, vm_out, b, hkv, sk, d, pv_accum_dtype, smooth_v != 0, tensor_layout);
    }
    return 0;
}

} // extern "C"
