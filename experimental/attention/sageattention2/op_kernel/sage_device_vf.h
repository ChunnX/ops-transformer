/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#pragma once

#include <cstdint>

#ifndef SAGE_ATTN_DEVICE_COMPILE
#if defined(__CCE__) && !defined(__CCE_KT_TEST__)
#define SAGE_ATTN_DEVICE_COMPILE 1
#else
#define SAGE_ATTN_DEVICE_COMPILE 0
#endif
#endif

#if SAGE_ATTN_DEVICE_COMPILE
#include "kernel_operator.h"

namespace sage_attn {
namespace dev_vf {

/**
 * 1. Q 向量量化 (INT8)：使用 950 MicroAPI 对齐加载、求 amax、计算缩放倒数并饱和写出。
 */
__simd_vf__ inline void QuantizeRowInt8VF(__ubuf__ half* in_ptr, __ubuf__ int8_t* out_ptr, uint32_t d) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toInt8{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    RegTensor<float> amax_reg;
    Duplicate(amax_reg, 0.0f);
    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<half> raw;
        RegTensor<float> val;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(raw, in_ptr + c);
        Cast<float, half, toFp32>(val, raw, full);
        Abs(val, val, full);
        Max(amax_reg, amax_reg, val, full);
    }
    ReduceMax(amax_reg, amax_reg, full);

    RegTensor<float> scale_reg, inv_scale_reg, div127_reg, one_reg;
    Duplicate(one_reg, 1.0f);
    Duplicate(div127_reg, 127.0f);
    Div(scale_reg, amax_reg, div127_reg, full);
    Maxs(scale_reg, scale_reg, 1e-6f, full);
    Div(inv_scale_reg, one_reg, scale_reg, full);

    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<half> raw;
        RegTensor<float> val;
        RegTensor<int32_t> packed;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(raw, in_ptr + c);
        Cast<float, half, toFp32>(val, raw, full);
        Mul(val, val, inv_scale_reg, full);
        Cast<int32_t, float, toInt8>(packed, val, full);
        StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out_ptr + c, reinterpret_cast<RegTensor<int8_t>&>(packed), full);
    }
}

/**
 * 2. K 向量量化 (INT8)：支持 smooth_k 减均值、求 amax、计算缩放倒数并写出。
 */
__simd_vf__ inline void QuantizeKeyRowInt8VF(__ubuf__ half* in_ptr, __ubuf__ float* km_ptr, __ubuf__ int8_t* out_ptr, uint32_t d, uint32_t has_km) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toInt8{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    RegTensor<float> amax_reg;
    Duplicate(amax_reg, 0.0f);
    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<half> raw;
        RegTensor<float> val, km_val;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(raw, in_ptr + c);
        Cast<float, half, toFp32>(val, raw, full);
        if (has_km) {
            LoadAlign(km_val, km_ptr + c);
            Sub(val, val, km_val, full);
        }
        Abs(val, val, full);
        Max(amax_reg, amax_reg, val, full);
    }
    ReduceMax(amax_reg, amax_reg, full);

    RegTensor<float> scale_reg, inv_scale_reg, div127_reg, one_reg;
    Duplicate(one_reg, 1.0f);
    Duplicate(div127_reg, 127.0f);
    Div(scale_reg, amax_reg, div127_reg, full);
    Maxs(scale_reg, scale_reg, 1e-6f, full);
    Div(inv_scale_reg, one_reg, scale_reg, full);

    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<half> raw;
        RegTensor<float> val, km_val;
        RegTensor<int32_t> packed;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(raw, in_ptr + c);
        Cast<float, half, toFp32>(val, raw, full);
        if (has_km) {
            LoadAlign(km_val, km_ptr + c);
            Sub(val, val, km_val, full);
        }
        Mul(val, val, inv_scale_reg, full);
        Cast<int32_t, float, toInt8>(packed, val, full);
        StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out_ptr + c, reinterpret_cast<RegTensor<int8_t>&>(packed), full);
    }
}

/**
 * 3. V 向量量化 (FP8 e4m3fn)：直接利用 MicroAPI 向量 Cast 饱和转换。
 */
__simd_vf__ inline void QuantizeValueRowFp8VF(__ubuf__ half* in_ptr, __ubuf__ float8_e4m3_t* out_ptr, uint32_t d) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toFp8{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<half> raw;
        RegTensor<float> val;
        RegTensor<float8_e4m3_t> packed;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(raw, in_ptr + c);
        Cast<float, half, toFp32>(val, raw, full);
        Cast<float8_e4m3_t, float, toFp8>(packed, val, full);
        StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out_ptr + c, packed, full);
    }
}

/**
 * 4. QK 块极值与局部指数和：通过 MicroAPI 的 ReduceMax 与 ExpSub 硬件原语求和。
 */
__simd_vf__ inline void BlockMaxAndSumVF(__ubuf__ float* logits, uint32_t block_n, __ubuf__ float* out_max_sum) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> max_val, sum_val;
    Duplicate(max_val, -1.0e30f);
    Duplicate(sum_val, 0.0f);

    for (uint32_t col = 0; col < block_n; col += 64) {
        RegTensor<float> x;
        LoadAlign(x, logits + col);
        Max(max_val, max_val, x, full);
    }
    ReduceMax(max_val, max_val, full);

    for (uint32_t col = 0; col < block_n; col += 64) {
        RegTensor<float> x, diff, p;
        LoadAlign(x, logits + col);
        Sub(diff, x, max_val, full);
        Exp(p, diff, full);
        Add(sum_val, sum_val, p, full);
    }
    ReduceSum(sum_val, sum_val, full);

    StoreAlign(out_max_sum, max_val, full);
    StoreAlign(out_max_sum + 64, sum_val, full);
}

/**
 * 5. O 累加器单行原地缩放。
 */
__simd_vf__ inline void RescaleAccumRowVF(__ubuf__ float* o_row, float factor, uint32_t d) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> factor_reg;
    Duplicate(factor_reg, factor);
    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> x;
        LoadAlign(x, o_row + c);
        Mul(x, x, factor_reg, full);
        StoreAlign(o_row + c, x, full);
    }
}

/**
 * 6. Softmax 概率矩阵生成：将 448 倍放大偏移融合进指数指数项 (m_prime - ln(448))，
 *    消除循环内的向量浮点乘法指令，并饱和转换至 FP8 e4m3fn。
 */
__simd_vf__ inline void SoftmaxProbRowVF(__ubuf__ float* logits, __ubuf__ float8_e4m3_t* out_p, float m_prime, uint32_t block_n) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp8{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<float> m_reg;
    // 数学恒等：exp(x - m_prime) * 448 = exp(x - (m_prime - ln(448)))，ln(448) ≈ 6.104793232414985f
    // 硬件级融合：循环内消灭 Mul 向量乘法，节约 1 拍向量 ALU 计算
    Duplicate(m_reg, m_prime - 6.104793232414985f);
    for (uint32_t j = 0; j < block_n; j += 64) {
        RegTensor<float> logit_val, diff, exp_val;
        RegTensor<float8_e4m3_t> packed;
        LoadAlign(logit_val, logits + j);
        Sub(diff, logit_val, m_reg, full);
        Exp(exp_val, diff, full);
        Cast<float8_e4m3_t, float, toFp8>(packed, exp_val, full);
        StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out_p + j, packed, full);
    }
}

/**
 * 7. 最终归一化与 FP16 输出转换。
 */
__simd_vf__ inline void FinalizeRowVF(__ubuf__ float* o_row, __ubuf__ half* out_row, float inv_d, uint32_t d) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toHalf{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<float> inv_reg;
    Duplicate(inv_reg, inv_d);
    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> x;
        RegTensor<half> h;
        LoadAlign(x, o_row + c);
        Mul(x, x, inv_reg, full);
        Cast<half, float, toHalf>(h, x, full);
        StoreAlign<half, StoreDist::DIST_PACK_B32>(out_row + c, h, full);
    }
}

} // namespace dev_vf
} // namespace sage_attn
#endif
