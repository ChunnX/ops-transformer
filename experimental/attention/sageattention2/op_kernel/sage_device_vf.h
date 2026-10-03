#pragma once

#include <cstdint>
#include "../sageattn_fp8/ascendc/tiling.h"

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
 * 1a. 行绝对值最大（DEFECT-F-1 组粒度量化的行内阶段）：
 * 可选 kMean 减法（带输入位型舍回），Abs 后 ReduceMax，行 max 写 1 元素槽。
 * 组间归并与 scale 定档由调用方按 quant::QScaleIndex/KScaleIndex 单点完成
 * （与宿主串行路径逐步一致）。
 */
__simd_vf__ inline void RowAbsMaxVF(__ubuf__ half* in_ptr, __ubuf__ float* out_slot,
                                    __ubuf__ float* km_ptr, uint32_t d,
                                    uint32_t has_km, uint32_t dtype_mode) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toHalf{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toBf16{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    RegTensor<float> amax_reg;
    Duplicate(amax_reg, 0.0f);
    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> val;
        if (dtype_mode == 2u) {
            RegTensor<bfloat16_t> raw;
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(raw, reinterpret_cast<__ubuf__ bfloat16_t*>(in_ptr) + c);
            Cast<float, bfloat16_t, toFp32>(val, raw, full);
        } else {
            RegTensor<half> raw;
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(raw, in_ptr + c);
            Cast<float, half, toFp32>(val, raw, full);
        }
        if (has_km) {
            RegTensor<float> km_val;
            LoadAlign(km_val, km_ptr + c);
            Sub(val, val, km_val, full);
            // 差值位型舍回（对齐孪生 Half/Bf16Reader::Sub 口径）
            if (dtype_mode == 2u) {
                RegTensor<bfloat16_t> rt;
                Cast<bfloat16_t, float, toBf16>(rt, val, full);
                Cast<float, bfloat16_t, toFp32>(val, rt, full);
            } else {
                RegTensor<half> rt;
                Cast<half, float, toHalf>(rt, val, full);
                Cast<float, half, toFp32>(val, rt, full);
            }
        }
        Abs(val, val, full);
        Max(amax_reg, amax_reg, val, full);
    }
    ReduceMax(amax_reg, amax_reg, full);
    StoreAlign(out_slot, amax_reg, full);
}

/**
 * 1b. 行 INT8 量化（DEFECT-F-1 除法轨迹 + 舍入分档）：
 * - 除法轨迹：val / scale（禁 inv-scale 乘法）；
 * - per_thread：half-away-from-zero——x 加 0.5·sign(x) 后 CAST_TRUNC（对齐
 *   quant::RoundHalfAwayFromZero 的 floor(|x|+0.5)·sign）；
 * - per_warp：CAST_RINT（对齐 quant::RoundNearestEven / cvt.rni）；
 * - 饱和 ±127（与两个 Round* 的 clamp 一致）。
 */
__simd_vf__ inline void QuantRowInt8VF(__ubuf__ half* in_ptr, __ubuf__ int8_t* out_ptr,
                                       float scale, uint32_t d,
                                       uint32_t per_thread, uint32_t has_km,
                                       __ubuf__ float* km_ptr, uint32_t dtype_mode) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toHalf{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toBf16{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toI32Trunc{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    static constexpr CastTrait toI32Rint{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toI8Sat{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

    RegTensor<float> scale_reg, neg127, pos127;
    Duplicate(scale_reg, scale);
    // 饱和界 INT8_QUANT_MAX 唯一定义位于 tiling.h（quant_common.h 同源纪律）
    Duplicate(neg127, -INT8_QUANT_MAX);
    Duplicate(pos127, INT8_QUANT_MAX);
    RegTensor<float> zero, half_reg, neg_half_reg;
    Duplicate(zero, 0.0f);
    Duplicate(half_reg, 0.5f);
    Duplicate(neg_half_reg, -0.5f);

    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> val;
        if (dtype_mode == 2u) {
            RegTensor<bfloat16_t> raw;
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(raw, reinterpret_cast<__ubuf__ bfloat16_t*>(in_ptr) + c);
            Cast<float, bfloat16_t, toFp32>(val, raw, full);
        } else {
            RegTensor<half> raw;
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(raw, in_ptr + c);
            Cast<float, half, toFp32>(val, raw, full);
        }
        if (has_km) {
            RegTensor<float> km_val;
            LoadAlign(km_val, km_ptr + c);
            Sub(val, val, km_val, full);
            if (dtype_mode == 2u) {
                RegTensor<bfloat16_t> rt;
                Cast<bfloat16_t, float, toBf16>(rt, val, full);
                Cast<float, bfloat16_t, toFp32>(val, rt, full);
            } else {
                RegTensor<half> rt;
                Cast<half, float, toHalf>(rt, val, full);
                Cast<float, half, toFp32>(val, rt, full);
            }
        }
        Div(val, val, scale_reg, full);

        RegTensor<int32_t> packed_i;
        if (per_thread) {
            // half-away：x + 0.5·sign(x) 后截断（对 -0 边界与宿主 floor(|x|+0.5)·sign 同值）。
            // MicroAPI 语义（dav-3510 头与 CANN 内置实现核对）：Compare<LT>(a,b) = a<b，
            // Select(dst,s0,s1,mask) = mask ? s0 : s1——neg(val<0) 须取 neg_half_reg(-0.5)
            // 才是 +0.5·sign(x)；反接会退化为 round-half-toward-zero（R16 审查修正）
            RegTensor<float> bias, shifted;
            MaskReg neg;
            Compare<float, CMPMODE::LT>(neg, val, zero, full);
            Select(bias, neg_half_reg, half_reg, neg);
            Add(shifted, val, bias, full);
            Cast<int32_t, float, toI32Trunc>(packed_i, shifted, full);
        } else {
            Cast<int32_t, float, toI32Rint>(packed_i, val, full);
        }
        Cast<float, int32_t, toFp32>(val, packed_i, full);
        Max(val, val, neg127, full);
        Min(val, val, pos127, full);
        // 硬件 vconv 无 float→int8 直转（dav-3510 static assert）：int32 中转后
        // DIST_PACK4_B32 打包窄化（±127 已在 float 域 clamp，窄化无损）
        Cast<int32_t, float, toI8Sat>(packed_i, val, full);
        StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out_ptr + c,
            reinterpret_cast<RegTensor<int8_t>&>(packed_i), full);
    }
}

/**
 * 3. V 行 FP8 量化（DEFECT-F-1：per-channel scale 分档）：
 * - scale 向量由调用方按全序列 per-channel amax / v_scale_max（2.25 或 448 分档）预写；
 * - 除法轨迹：scaled = val / scale（对齐宿主/孪生除法）；
 * - 全零通道 scale=0 → 0/0=NaN → e4m3 编码 0x7f（对齐 quant::FloatToE4m3fn 的
 *   NaN 保真与 golden）；有限通道饱和 ±scale_max 后 RNE 转 e4m3。
 */
__simd_vf__ inline void QuantVRowFp8VF(__ubuf__ half* in_ptr, __ubuf__ float8_e4m3_t* out_ptr,
                                       __ubuf__ float* scale_ptr, uint32_t d,
                                       float scale_max, uint32_t dtype_mode) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toBf16{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toFp8{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    RegTensor<float> scale_max_reg, neg_scale_max_reg;
    Duplicate(scale_max_reg, scale_max);
    Duplicate(neg_scale_max_reg, -scale_max);
    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> val, scale_reg, scaled;
        RegTensor<float8_e4m3_t> packed;
        if (dtype_mode == 2u) {
            RegTensor<bfloat16_t> raw;
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(raw, reinterpret_cast<__ubuf__ bfloat16_t*>(in_ptr) + c);
            Cast<float, bfloat16_t, toFp32>(val, raw, full);
        } else {
            RegTensor<half> raw;
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(raw, in_ptr + c);
            Cast<float, half, toFp32>(val, raw, full);
        }
        LoadAlign(scale_reg, scale_ptr + c);
        Div(scaled, val, scale_reg, full);
        Min(scaled, scaled, scale_max_reg, full);
        Max(scaled, scaled, neg_scale_max_reg, full);
        Cast<float8_e4m3_t, float, toFp8>(packed, scaled, full);
        StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out_ptr + c, packed, full);
    }
}

/**
 * 4a. A7 V1 消费原语（R17-1）：dot32 区单行 INT32 QK^T 反量化 + causal/尾块掩码。
 * - 数值唯一依据 twin::DequantQkLogit 左结合乘序：(float)dot · q_scale · k_scale[j] ·
 *   sm_scale_log2e——int32→fp32 cast 在 |dot| ≤ 2^21 整数域内精确（01_design 2.9：
 *   127·127·128 ≈ 2^20.97），三次标量/向量乘按宿主同一结合序逐位同式；
 *   注意：台账所记"×2^-11"定点折算若按字面串入乘链，将给结果附加恰 2^-11 的精确
 *   幂偏置、破坏与宿主的逐位一致性，故不采用（2 的整数次幂乘法精确、不可消去）；
 * - 掩码谓词与宿主 ComputeBlockMaxAndSum 同式：causal（token_k > token_q）与尾块
 *   pad token（token_k ≥ sk_）写 -1e30 哨兵（不参与行 max/448· 域 exp2 和）；
 * - 列绝对 token 序号经 Arange 硬件原语生成，GT/GE 两级比较出掩码后 Select 落哨兵。
 * 行 max 与 exp2 和由调用方紧随调用 BlockMaxAndSumVF 完成（组合即宿主同构语义）。
 */
__simd_vf__ inline void DequantMaskLogitsRowVF(__ubuf__ int32_t* dot_row, __ubuf__ float* logits_row,
                                               float q_scale, __ubuf__ float* k_scale_row,
                                               float sm_scale_log2e, uint32_t token_q,
                                               uint32_t k_start, uint32_t sk, uint32_t is_causal,
                                               uint32_t block_n) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    RegTensor<float> qs_reg, sm_reg, neg_inf;
    Duplicate(qs_reg, q_scale);
    Duplicate(sm_reg, sm_scale_log2e);
    Duplicate(neg_inf, -1.0e30f);

    for (uint32_t j = 0; j < block_n; j += 64) {
        RegTensor<float> val, ks_vec, masked_q, masked_k;
        RegTensor<int32_t> dot_i;
        LoadAlign(dot_i, dot_row + j);
        Cast<float, int32_t, toFp32>(val, dot_i, full);   // INT32→FP32 精确 cast（≤2^21 整数域）
        Mul(val, val, qs_reg, full);                      // · q_scale（行解析，先乘）
        LoadAlign(ks_vec, k_scale_row + j);
        Mul(val, val, ks_vec, full);                      // · k_scale[j]（块解析向量）
        Mul(val, val, sm_reg, full);                      // · sm_scale_log2e（左结合同宿主）

        // 掩码谓词与宿主 ComputeBlockMaxAndSum 同式，比较界经标量预折算到列序域：
        // token_k = k_start + j + lane，causal: token_k > token_q ⟺ lane > token_q − 段基址
        //（非 causal 置 INT32_MAX 界使 GT 恒假）；尾块 pad: token_k ≥ sk ⟺ lane ≥ sk − 段基址。
        // 列序经 Arange 硬件原语生成——dav-3510 Arange 仅支持 s8/s16/s32/f32/f16/s64，
        // 列序取 int32 域（float 域 Arange/Adds 触发 AIV 后端 unsupported scalar
        // instruction，bisheng 门实测捕获）。
        RegTensor<int32_t> idx, tok_q_bound, sk_bound;
        Arange<int32_t>(idx, 0);
        Duplicate(tok_q_bound, (is_causal != 0u)
                                   ? static_cast<int32_t>(token_q) - static_cast<int32_t>(k_start + j)
                                   : 2147483647);
        Duplicate(sk_bound, static_cast<int32_t>(sk) - static_cast<int32_t>(k_start + j));
        MaskReg gt_tok_q, ge_sk;
        Compare<int32_t, CMPMODE::GT>(gt_tok_q, idx, tok_q_bound, full);
        Select(masked_q, neg_inf, val, gt_tok_q);         // causal：token_k > token_q → -1e30
        Compare<int32_t, CMPMODE::GE>(ge_sk, idx, sk_bound, full);
        Select(masked_k, neg_inf, masked_q, ge_sk);       // 尾块 pad：token_k ≥ sk → -1e30
        StoreAlign(logits_row + j, masked_k, full);
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

    // exp2 域 + 448· 放大（twin OnlineBlockExp2Sum 同式，收敛轮 F/DEFECT-F-2）：
    // cur_sum = Σ 2^(x-m)·448 = 448·Σ Exp((x-m)·ln2)；掩码哨兵 -1e30 经 Exp 下溢自然为 0
    for (uint32_t col = 0; col < block_n; col += 64) {
        RegTensor<float> x, diff, nat;
        LoadAlign(x, logits + col);
        Sub(diff, x, max_val, full);
        Muls(diff, diff, INV_LOG2E, full);
        Exp(nat, diff, full);
        Muls(nat, nat, FP8_E4M3_MAX, full);
        Add(sum_val, sum_val, nat, full);
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
 * 6. Softmax 概率矩阵生成：exp2 域恒等 P = 2^(x-m')·448（logits 已折 sm_scale·log2e，
 *    经 ln2 缩放进自然域），448· 放大经乘法在 Exp 之后实现（twin::SoftmaxProbCode
 *    同式；DEFECT-R11-1：ln448 融合进锚点的以加代乘轨迹与孪生差 1 ulp，经 e4m3
 *    量化放大为位翻转，同 N7 裁定），并饱和转换至 FP8 e4m3fn。
 */
__simd_vf__ inline void SoftmaxProbRowVF(__ubuf__ float* logits, __ubuf__ float8_e4m3_t* out_p, float m_prime, uint32_t block_n) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp8{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    // exp2 域恒等（twin::SoftmaxProbCode 同式）：P = 2^(x-m')·448，乘法在 Exp 之后。
    // logits 为 exp2 域量（已折 sm_scale·log2e），经 ln2 缩放进硬件自然指数域——
    // ln2 缩放为设备本征域换算（真机 L-A 容差域，A7 族登记）；
    // 448· 放大使行 max 恰为 448 填满 e4m3 动态范围。
    RegTensor<float> m_ln_reg, ln2_reg;
    Duplicate(m_ln_reg, m_prime * INV_LOG2E);
    Duplicate(ln2_reg, INV_LOG2E);
    for (uint32_t j = 0; j < block_n; j += 64) {
        RegTensor<float> logit_val, scaled, diff, exp_val;
        RegTensor<float8_e4m3_t> packed;
        LoadAlign(logit_val, logits + j);
        Mul(scaled, logit_val, ln2_reg, full);
        Sub(diff, scaled, m_ln_reg, full);
        Exp(exp_val, diff, full);
        Muls(exp_val, exp_val, FP8_E4M3_MAX, full);
        Cast<float8_e4m3_t, float, toFp8>(packed, exp_val, full);
        StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out_p + j, packed, full);
    }
}

/**
 * A7-1（task/12 第五节第 3 条 V0 侧）V per-channel 全序列 amax 预扫·通道累加步：amax[c] = max(amax[c], |val[c]|)
 * （输入位型 cast + 可选 kMean 域外无——V 无 kMean），逐 token 行调用一次，
 * 由调用方先充零 amax 缓冲（twin/quant_fp8 per-channel 口径）。
 */
__simd_vf__ inline void ChannelAmaxAccumVF(__ubuf__ half* in_ptr, __ubuf__ float* amax_ptr,
                                           uint32_t d, uint32_t dtype_mode) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> val, amax;
        if (dtype_mode == 2u) {
            RegTensor<bfloat16_t> raw;
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(raw, reinterpret_cast<__ubuf__ bfloat16_t*>(in_ptr) + c);
            Cast<float, bfloat16_t, toFp32>(val, raw, full);
        } else {
            RegTensor<half> raw;
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(raw, in_ptr + c);
            Cast<float, half, toFp32>(val, raw, full);
        }
        Abs(val, val, full);
        LoadAlign(amax, amax_ptr + c);
        Max(amax, amax, val, full);
        StoreAlign(amax_ptr + c, amax, full);
    }
}

/**
 * A7-2 V per-channel 定标：scale[c] = amax[c] / scale_max（除法轨迹，对齐宿主
 * quant_fp8 / 孪生；全零通道 amax=0 → scale=0，经 QuantVRowFp8VF 的 0/0=NaN →
 * e4m3 0x7f 容错）。可原地（out_scale_ptr == amax_ptr）。
 */
__simd_vf__ inline void FinalizeVScaleVF(__ubuf__ float* amax_ptr, __ubuf__ float* out_scale_ptr,
                                         float scale_max, uint32_t d) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> smax_reg;
    Duplicate(smax_reg, scale_max);
    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> amax;
        LoadAlign(amax, amax_ptr + c);
        Div(amax, amax, smax_reg, full);
        StoreAlign(out_scale_ptr + c, amax, full);
    }
}

/**
 * 7. 最终归一化与输出位型转换（dtype_mode：2=bf16，其余按 fp16）。
 */
__simd_vf__ inline void FinalizeRowVF(__ubuf__ float* o_row, __ubuf__ uint16_t* out_row, float inv_d, uint32_t d, uint32_t dtype_mode) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toHalf{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toBf16{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<float> inv_reg;
    Duplicate(inv_reg, inv_d);
    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> x;
        LoadAlign(x, o_row + c);
        Mul(x, x, inv_reg, full);
        if (dtype_mode == 2u) {
            // bf16 输出（收敛轮 F/DEFECT-F-3）：与宿主 WriteOutputBits 的 Fp32ToBf16 口径一致
            RegTensor<bfloat16_t> b;
            Cast<bfloat16_t, float, toBf16>(b, x, full);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(reinterpret_cast<__ubuf__ bfloat16_t*>(out_row) + c, b, full);
        } else {
            RegTensor<half> h;
            Cast<half, float, toHalf>(h, x, full);
            StoreAlign<half, StoreDist::DIST_PACK_B32>(reinterpret_cast<__ubuf__ half*>(out_row) + c, h, full);
        }
    }
}

} // namespace dev_vf
} // namespace sage_attn
#endif
