#pragma once

#include <cstdint>
#include <limits>
#include "../sageattn_fp8/ascendc/tiling.h"


#if SAGE_ATTN_DEVICE_COMPILE
#include "kernel_operator.h"
// 官方 CATLASS 设备函数宏（catlass/detail/macros.hpp）：CATLASS_DEVICE_SIMD_VF =
// __simd_vf__ static（CATLASS_ARCH=3510），与官方 950 FA softmax epilogue 同款标记，
// 取代本文件原裸 __simd_vf__ inline 书写（官方演进单点跟随；设备编译全路径已注入
// -DCATLASS_ARCH=3510，见 run.sh kernel / build_opp.sh）。
#include "catlass/detail/macros.hpp"

// BiSheng 编译器缺陷约束（2026-10-04 真机编译崩溃定位，禁"优化"回旧写法）：
// 本文件（含全部 dev_vf 原语）禁止 64 位操作数的变量移位（64-bit value 按非常量
// 移位会触发 BiSheng LLVM SelectionDAG 编译期崩溃）。行寻址等索引算术一律留在
// uint32_t 域——本算子行数 ≤ MAX_PACK_ROWS=128、行宽 ≤ 128 元素，32 位乘积无溢出；
// 行级位图拆 4 个 uint32_t 字、以 `& 31` 掩码移位取位（RescaleAccumRowsVF 与
// sageattention2.cpp 的 bm 组装同款）。改回 uint64_t 偏移/位图会原样复现崩溃。

namespace sage_attn {
namespace dev_vf {

/**
 * M5 编译期位型特化约定：凡按输入位型（1=fp16, 2=bf16，编码唯一定义
 * tiling.h input_dtype / sage_attention2_template_tiling_key.h）分流的 VF 原语，
 * 档位经非类型模板参数 DTYPE_MODE 承接（调用方传 SageAttnKernel 的编译期
 * 常量 kDtype），位型支路一律 if constexpr——单一源定义，实例化只保留所取
 * 支路的指令序列（原运行时 dtype_mode 形参 + 运行时 if 的零分支化，逐位算术不变）。
 */

/**
 * 1a. 行批绝对值最大（DEFECT-F-1 组粒度量化的行内阶段，R4 行批化）：
 * 可选 kMean 减法（带输入位型舍回），Abs 后 ReduceMax。
 * 针对昇腾 DaVinci 硬件 VEC 访 UB 必须 32 字节对齐约束（彻底根除逐行 +4 字节非对齐
 * StoreAlign 触发硬件 340 异常）：
 * 采用 CANN MicroAPI 标准 StoreUnAlign / StoreUnAlignPost 流式非对齐写原语，
 * 在寄存器中缓冲逐行单元素归约值，满 32 字节（8 float）对齐写出，循环后统一刷回 UB。
 * 组间归并与 scale 定档由调用方按 quant::QScaleIndex/KScaleIndex 单点完成
 * （与宿主串行路径逐步一致）。
 */
template <uint32_t DTYPE_MODE>
CATLASS_DEVICE_SIMD_VF void RowAbsMaxRowsVF(__ubuf__ half* in_base, __ubuf__ float* out_rowmax,
                                            __ubuf__ float* km_ptr, uint32_t rows, uint32_t d,
                                            uint32_t has_km) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    if (rows == 0u) {
        return;
    }
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toHalf{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toBf16{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    __ubuf__ float* dst = out_rowmax;
    UnalignReg compact;

    for (uint32_t r = 0; r < rows; ++r) {
        __ubuf__ half* in_ptr = in_base + r * d;
        RegTensor<float> amax_reg;
        Duplicate(amax_reg, 0.0f);
        for (uint32_t c = 0; c < d; c += 64) {
            RegTensor<float> val;
            if constexpr (DTYPE_MODE == 2u) {
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
                if constexpr (DTYPE_MODE == 2u) {
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
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, amax_reg, compact, 1);
    }
    StoreUnAlignPost(dst, compact, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

/**
 * 1b. 行 INT8 量化（DEFECT-F-1 除法轨迹 + 舍入分档）：
 * - 除法轨迹：val / scale（禁 inv-scale 乘法）；
 * - per_thread：half-away-from-zero——x 加 0.5·sign(x) 后 CAST_TRUNC（对齐
 *   quant::RoundHalfAwayFromZero 的 floor(|x|+0.5)·sign）；
 * - per_warp：CAST_RINT（对齐 quant::RoundNearestEven / cvt.rni）；
 * - 饱和 ±127（与两个 Round* 的 clamp 一致）。
 */
template <uint32_t DTYPE_MODE>
CATLASS_DEVICE_SIMD_VF void QuantRowInt8VF(__ubuf__ half* in_ptr, __ubuf__ int8_t* out_ptr,
                                       float scale, uint32_t d,
                                       uint32_t per_thread, uint32_t has_km,
                                       __ubuf__ float* km_ptr) {
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
        if constexpr (DTYPE_MODE == 2u) {
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
            if constexpr (DTYPE_MODE == 2u) {
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
 * - 可选 smooth_v 的 vm 减法（float 域直减、无位型舍回，对齐 quant_fp8 ProcessV 口径）；
 * - 除法轨迹：scaled = val / scale（对齐宿主/孪生除法）；
 * - 全零通道 scale=0 → 0/0=NaN → e4m3 编码 0x7f（对齐 quant::FloatToE4m3fn 的
 *   NaN 保真与 golden）；有限通道饱和 ±scale_max 后 RNE 转 e4m3。
 */
template <uint32_t DTYPE_MODE>
CATLASS_DEVICE_SIMD_VF void QuantVRowFp8VF(__ubuf__ half* in_ptr, __ubuf__ float8_e4m3_t* out_ptr,
                                       __ubuf__ float* scale_ptr, __ubuf__ float* vm_ptr,
                                       uint32_t has_vm, uint32_t d,
                                       float scale_max) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toFp8{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    RegTensor<float> scale_max_reg, neg_scale_max_reg, zero_reg, nan_reg;
    Duplicate(scale_max_reg, scale_max);
    Duplicate(neg_scale_max_reg, -scale_max);
    Duplicate(zero_reg, 0.0f);
    Duplicate(nan_reg, std::numeric_limits<float>::quiet_NaN());
    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> val, scale_reg, scaled;
        RegTensor<float8_e4m3_t> packed;
        if constexpr (DTYPE_MODE == 2u) {
            RegTensor<bfloat16_t> raw;
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(raw, reinterpret_cast<__ubuf__ bfloat16_t*>(in_ptr) + c);
            Cast<float, bfloat16_t, toFp32>(val, raw, full);
        } else {
            RegTensor<half> raw;
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(raw, in_ptr + c);
            Cast<float, half, toFp32>(val, raw, full);
        }
        if (has_vm != 0u) {
            RegTensor<float> vm_val;
            LoadAlign(vm_val, vm_ptr + c);
            Sub(val, val, vm_val, full);
        }
        LoadAlign(scale_reg, scale_ptr + c);
        Div(scaled, val, scale_reg, full);
        // 全零通道 NaN 守卫（收敛轮 1/R1-1）：不依赖硬件 Min/Max NaN 语义——
        // scale=0 → Div 产 NaN → SAT cast 保留 e4m3 0x7f（对齐 golden）
        MaskReg zero_scale;
        Compare<float, CMPMODE::EQ>(zero_scale, scale_reg, zero_reg, full);
        Select(scaled, nan_reg, scaled, zero_scale);
        Min(scaled, scaled, scale_max_reg, full);
        Max(scaled, scaled, neg_scale_max_reg, full);
        Cast<float8_e4m3_t, float, toFp8>(packed, scaled, full);
        StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out_ptr + c, packed, full);
    }
}

/**
 * 4a. A7 V1 消费原语（R17-1，R4 行批化）：dot32 区整窗 INT32 QK^T 反量化 + causal/尾块掩码。
 * - 原地轨迹：float logits 按 [row][block_n] 布局覆写同一 dot32 缓冲（int32→float 元素
 *   等宽）——同 chunk 地址 Load 先于 Store（寄存器 RAW 序，各 chunk 地址互斥）；dot32
 *   为单缓冲读保护交接槽，本区覆写发生在 wait QK_P_READY 之后、V_S 排空 + set DOT_FREE
 *   之前，Cube 下一拍覆写被 FLAG_DOT_FREE 挡在 V_S 之后，无竞争窗口；
 * - 数值唯一依据 twin::DequantQkLogit 左结合乘序：(float)dot · q_scale · k_scale[j] ·
 *   sm_scale_log2e——int32→fp32 cast 在 |dot| ≤ 2^21 整数域内精确（01_design 2.9：
 *   127·127·128 ≈ 2^20.97），三次标量/向量乘按宿主同一结合序逐位同式；
 *   注意：台账所记"×2^-11"定点折算若按字面串入乘链，将给结果附加恰 2^-11 的精确
 *   幂偏置、破坏与宿主的逐位一致性，故不采用（2 的整数次幂乘法精确、不可消去）；
 * - 行批等价论证（R4）：行间独立，跨行在途交织不改各行 FP 轨迹。每行的行解析量取法
 *   位保：token_q 由标量实参算术（与原调用方同式）；q_scale 经 DIST_BRC_B32 单元素
 *   广播 Load（读同一 ub_qs 槽位的同一 float，取代原 Duplicate 入参——搬运非 FP 运算）；
 *   每行三态掩码快速路径判定与 chunk 分流逐分支保持原单行函数形态；
 * - 三态掩码快速路径（Tri-State Masking Fast Path，逐行独立判定）：
 *   ① 全活跃态（is_pure_active / chunk_pure_active）：完全在因果下三角与 sk 尾界内，
 *      跳过全部 Arange、bounds Duplicate、Compare 与 Select，反量化值直存；
 *   ② 全掩码态（is_pure_masked / chunk_pure_masked）：完全在因果上三角或超 sk 尾界，
 *      跳过点积读取、Cast 与三次乘法，直接落 -1e30 哨兵；
 *   ③ 混合边界态：仅对触碰因果对角线或尾块 pad 界的 chunk 执行硬件 int32 Arange 与 GT/GE 掩码 Select。
 * 行 max 与 exp2 和由调用方紧随调用 BlockMaxAndSumRowsVF 完成（组合即宿主同构语义）。
 */
CATLASS_DEVICE_SIMD_VF void DequantMaskLogitsRowsVF(__ubuf__ float* logits_dot_base, uint32_t rows,
                                               __ubuf__ float* qs_row, __ubuf__ float* k_scale_row,
                                               float sm_scale_log2e, uint32_t cur_q_start,
                                               uint32_t cur_q_len, uint32_t k_start, uint32_t sk,
                                               uint32_t is_causal, uint32_t block_n) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    __ubuf__ int32_t* dot_base = reinterpret_cast<__ubuf__ int32_t*>(logits_dot_base);

    RegTensor<float> sm_reg;
    Duplicate(sm_reg, sm_scale_log2e);

    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t token_q = cur_q_start + (r % cur_q_len);
        __ubuf__ int32_t* dot_row = dot_base + r * block_n;
        __ubuf__ float* logits_row = logits_dot_base + r * block_n;

        // 行级快速路径 1：全掩码行（整行处于因果上三角或超 sk 尾界），直接写 -1e30 哨兵
        const bool is_pure_masked = (is_causal != 0u && token_q < k_start) || (k_start >= sk);
        if (is_pure_masked) {
            RegTensor<float> neg_inf;
            Duplicate(neg_inf, SAGE_MASK_SENTINEL);
            for (uint32_t j = 0; j < block_n; j += 64) {
                StoreAlign(logits_row + j, neg_inf, full);
            }
            continue;
        }

        // 行解析 q_scale：DIST_BRC_B32 单元素广播 Load（读 ub_qs[r] 原 float，位保搬运）
        RegTensor<float> qs_reg;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(qs_reg, qs_row + r);

        // 行级快速路径 2：全活跃行（整行完全在因果下三角与 sk 尾界内），免除全部掩码计算
        const bool is_pure_active = (is_causal == 0u || token_q >= k_start + block_n - 1) && (k_start + block_n <= sk);
        if (is_pure_active) {
            for (uint32_t j = 0; j < block_n; j += 64) {
                RegTensor<float> val, ks_vec;
                RegTensor<int32_t> dot_i;
                LoadAlign(dot_i, dot_row + j);
                Cast<float, int32_t, toFp32>(val, dot_i, full);   // INT32→FP32 精确 cast（≤2^21 整数域）
                Mul(val, val, qs_reg, full);                      // · q_scale（行解析，先乘）
                LoadAlign(ks_vec, k_scale_row + j);
                Mul(val, val, ks_vec, full);                      // · k_scale[j]（块解析向量）
                Mul(val, val, sm_reg, full);                      // · sm_scale_log2e（左结合同宿主）
                StoreAlign(logits_row + j, val, full);
            }
            continue;
        }

        // 行级路径 3：混合行（触碰因果斜线或 sk 尾界），按 64-element chunk 细粒度三态分流
        RegTensor<float> neg_inf;
        Duplicate(neg_inf, SAGE_MASK_SENTINEL);

        for (uint32_t j = 0; j < block_n; j += 64) {
            const uint32_t chunk_start = k_start + j;
            const uint32_t chunk_end = chunk_start + 63u;

            // Chunk 级态 1：全掩码（因果上三角或超 sk 尾界），跳过点积读取与乘法直接落哨兵
            if ((is_causal != 0u && token_q < chunk_start) || (chunk_start >= sk)) {
                StoreAlign(logits_row + j, neg_inf, full);
                continue;
            }

            RegTensor<float> val, ks_vec;
            RegTensor<int32_t> dot_i;
            LoadAlign(dot_i, dot_row + j);
            Cast<float, int32_t, toFp32>(val, dot_i, full);
            Mul(val, val, qs_reg, full);
            LoadAlign(ks_vec, k_scale_row + j);
            Mul(val, val, ks_vec, full);
            Mul(val, val, sm_reg, full);

            // Chunk 级态 2：全活跃（完全在因果下三角与 sk 尾界内），直接写反量化结果
            if ((is_causal == 0u || token_q >= chunk_end) && (chunk_start + 64u <= sk)) {
                StoreAlign(logits_row + j, val, full);
                continue;
            }

            // Chunk 级态 3：混合边界，调用硬件 Arange/Compare/Select 精确掩码
            RegTensor<int32_t> idx, tok_q_bound, sk_bound;
            Arange<int32_t>(idx, 0);
            Duplicate(tok_q_bound, (is_causal != 0u)
                                       ? static_cast<int32_t>(token_q) - static_cast<int32_t>(chunk_start)
                                       : 2147483647);
            Duplicate(sk_bound, static_cast<int32_t>(sk) - static_cast<int32_t>(chunk_start));
            MaskReg gt_tok_q, ge_sk;
            Compare<int32_t, CMPMODE::GT>(gt_tok_q, idx, tok_q_bound, full);
            RegTensor<float> masked_q, masked_k;
            Select(masked_q, neg_inf, val, gt_tok_q);         // causal：token_k > token_q → -1e30
            Compare<int32_t, CMPMODE::GE>(ge_sk, idx, sk_bound, full);
            Select(masked_k, neg_inf, masked_q, ge_sk);       // 尾块 pad：token_k ≥ sk → -1e30
            StoreAlign(logits_row + j, masked_k, full);
        }
    }
}

/**
 * 4. QK 块极值与局部指数和（R4 行批化）：逐行原 ReduceMax/Exp 链不变。
 *    针对 DaVinci 硬件 VEC 访 UB 必须 32 字节对齐约束（根除 r>0 时 out+r 的 +4 字节
 *    非对齐 StoreAlign 触发硬件 340 异常）：
 *    采用 CANN MicroAPI 标准 StoreUnAlign / StoreUnAlignPost 流式非对齐写原语，
 *    在寄存器中缓冲逐行单元素归约值，满 32 字节（8 float）对齐写出，循环后统一刷回 UB。
 *    行内两级 chunk 循环、ReduceMax/ReduceSum 归约树、掩码哨兵 -1e30 的 Exp 下溢行为逐位同原单行版本。
 */
CATLASS_DEVICE_SIMD_VF void BlockMaxAndSumRowsVF(__ubuf__ float* logits_base, uint32_t rows, uint32_t block_n,
                                                 __ubuf__ float* out_max_row, __ubuf__ float* out_sum_row) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    if (rows == 0u) {
        return;
    }
    auto full = CreateMask<float, MaskPattern::ALL>();
    __ubuf__ float* max_ptr = out_max_row;
    __ubuf__ float* sum_ptr = out_sum_row;
    UnalignReg max_store;
    UnalignReg sum_store;

    for (uint32_t r = 0; r < rows; ++r) {
        __ubuf__ float* logits = logits_base + r * block_n;
        RegTensor<float> max_val, sum_val;
        Duplicate(max_val, SAGE_MASK_SENTINEL);
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

        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(max_ptr, max_val, max_store, 1);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sum_ptr, sum_val, sum_store, 1);
    }
    StoreUnAlignPost(max_ptr, max_store, 0);
    StoreUnAlignPost(sum_ptr, sum_store, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

/**
 * 5. O 累加器行批原地缩放（R4）：逐行位图许可（bm 位 r 置位方执行该行，与原标量
 *    need_rescale && |factor-1|>tol 判定逐行同集）；系数经 DIST_BRC_B32 广播 Load
 *    ub_factor[r]（原 Duplicate(rescale_factor) 的同一 float，位保搬运），行内
 *    Load→Mul→Store 链与原 RescaleAccumRowVF 逐位同式。未置位行零访存零运算
 *    （不做 ×1.0f 恒等乘，保持原跳过语义的零开销）。
 */
CATLASS_DEVICE_SIMD_VF void RescaleAccumRowsVF(__ubuf__ float* o_base, __ubuf__ float* factor_row,
                                               uint32_t rows, uint32_t d,
                                               uint32_t bm0, uint32_t bm1, uint32_t bm2, uint32_t bm3) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t r = 0; r < rows; ++r) {
        // 位图字选择为三元链（c06f1d69c 复审：VF 原语内不放局部数组，
        // 原 uint32_t bm[4] 就地展开）；位提取恒为 uint32_t 掩码移位
        // （本文件头 BiSheng 64 位变量移位禁令）
        const uint32_t w = (r < 32u) ? bm0 : ((r < 64u) ? bm1 : ((r < 96u) ? bm2 : bm3));
        const uint32_t fire = (w >> (r & 31u)) & 1u;
        if (fire == 0u) {
            continue;
        }
        RegTensor<float> factor_reg;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(factor_reg, factor_row + r);
        __ubuf__ float* o_row = o_base + r * d;
        for (uint32_t c = 0; c < d; c += 64) {
            RegTensor<float> x;
            LoadAlign(x, o_row + c);
            Mul(x, x, factor_reg, full);
            StoreAlign(o_row + c, x, full);
        }
    }
}

/**
 * 6. Softmax 概率矩阵生成（R4 行批化）：exp2 域恒等 P = 2^(x-m')·448（logits 已折
 *    sm_scale·log2e，经 ln2 缩放进自然域），448· 放大经乘法在 Exp 之后实现
 *    （twin::SoftmaxProbCode 同式；DEFECT-R11-1：ln448 融合进锚点的以加代乘轨迹与
 *    孪生差 1 ulp，经 e4m3 量化放大为位翻转，同 N7 裁定），并饱和转换至 FP8 e4m3fn。
 *    行锚 m'·ln2 由调用方标量域预乘写入 m_ln_row[r]（与原 Duplicate(m_prime*INV_LOG2E)
 *    的同一标量乘式同值），VF 经 DIST_BRC_B32 广播 Load 回读——行内
 *    Mul(ln2)→Sub(m_ln)→Exp→Muls(448)→Cast 链逐位同原单行版本；掩码行经 exp2
 *    下溢自然落到 e4m3 码 0。
 */
CATLASS_DEVICE_SIMD_VF void SoftmaxProbRowsVF(__ubuf__ float* logits_base, __ubuf__ float8_e4m3_t* out_p_base,
                                              __ubuf__ float* m_ln_row, uint32_t rows, uint32_t block_n) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp8{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    // exp2 域恒等（twin::SoftmaxProbCode 同式）：P = 2^(x-m')·448，乘法在 Exp 之后。
    // logits 为 exp2 域量（已折 sm_scale·log2e），经 ln2 缩放进硬件自然指数域——
    // ln2 缩放为设备本征域换算（真机 L-A 容差域，A7 族登记）；
    // 448· 放大使行 max 恰为 448 填满 e4m3 动态范围。
    RegTensor<float> ln2_reg;
    Duplicate(ln2_reg, INV_LOG2E);
    for (uint32_t r = 0; r < rows; ++r) {
        RegTensor<float> m_ln_reg;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(m_ln_reg, m_ln_row + r);
        __ubuf__ float* logits = logits_base + r * block_n;
        __ubuf__ float8_e4m3_t* out_p = out_p_base + r * block_n;
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
}

/**
 * A7（task/12 D2，R4 行批化）pv_partial 交接槽整窗合并：partial[r][c] → O[r][c]。
 * 轨迹 = 宿主 twin::AccumulatePvRow（AccumulatePvBlockTwin 调用）的设备段实例：
 * o_row[c] += partial[r][c] · v_scale[c]，其中 partial 为 Cube PV MMAD 的 FP32
 * 部分和（P8 码 × V8 码，P 侧 448· 放大域与 v_scale 的 V 通道域拆分同宿主 v_at 分解）。
 * - fp32+fp16 档（fp16_round=1，档位判据由调用方直读 tiling.pv_accum_dtype）：
 *   并入前做 FP16 RNE 舍回（与宿主 quant::Fp32ToFp16/Fp16ToFp32 同 RNE 口径——
 *   块内求和已由 PV MMAD 承担（设备段粒度替代 twin k_step 子块，L-A 容差域，task/12）；
 * - fp32 / fp32+fp32 档：FP32 直通并入。
 * 行批等价论证（R4）：行间独立，跨行交织不改各行 Load→Mul(→FP16 RNE 舍回)→Add→Store
 * 链；v_scale 通道向量全行共享（LoadAlign 同址同值）。
 * 调用方须保证 O 行经 S 槽 / dot32 同款握手串行化（FLAG_PV_PARTIAL_READY/FREE）。
 */
CATLASS_DEVICE_SIMD_VF void MergePvPartialRowsVF(__ubuf__ float* partial_base, __ubuf__ float* o_base,
                                                 __ubuf__ float* v_scale_ptr, uint32_t rows, uint32_t d,
                                                 uint32_t fp16_round) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp16{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint32_t r = 0; r < rows; ++r) {
        __ubuf__ float* partial_row = partial_base + r * d;
        __ubuf__ float* o_row = o_base + r * d;
        for (uint32_t c = 0; c < d; c += 64) {
            RegTensor<float> partial, vs, o;
            LoadAlign(partial, partial_row + c);
            LoadAlign(vs, v_scale_ptr + c);
            Mul(partial, partial, vs, full);
            if (fp16_round != 0u) {
                RegTensor<half> rounded;
                Cast<half, float, toFp16>(rounded, partial, full);
                Cast<float, half, toFp32>(partial, rounded, full);
            }
            LoadAlign(o, o_row + c);
            Add(o, o, partial, full);
            StoreAlign(o_row + c, o, full);
        }
    }
}

/**
 * A7-1（task/12 第五节第 3 条 V0 侧）V per-channel 全序列 amax 预扫·通道累加步：amax[c] = max(amax[c], |val[c]|)
 * （输入位型 cast + 可选 kMean 域外无——V 无 kMean），逐 token 行调用一次，
 * 由调用方先充零 amax 缓冲（twin/quant_fp8 per-channel 口径）。
 */
template <uint32_t DTYPE_MODE>
CATLASS_DEVICE_SIMD_VF void ChannelAmaxAccumVF(__ubuf__ half* in_ptr, __ubuf__ float* amax_ptr,
                                           uint32_t d) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> val, amax;
        if constexpr (DTYPE_MODE == 2u) {
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
CATLASS_DEVICE_SIMD_VF void FinalizeVScaleVF(__ubuf__ float* amax_ptr, __ubuf__ float* out_scale_ptr,
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
 * A7-3 smooth_v 预扫·通道 max/min/sum 累加步（smooth_v 生效时替代 ChannelAmaxAccumVF，
 * 单趟读出三统计量；调用方按 max=-FLT_MAX / min=+FLT_MAX / sum=0 充初值）。
 */
template <uint32_t DTYPE_MODE>
CATLASS_DEVICE_SIMD_VF void ChannelStatsAccumVF(__ubuf__ half* in_ptr, __ubuf__ float* vmax_ptr,
                                            __ubuf__ float* vmin_ptr, __ubuf__ float* vsum_ptr,
                                            uint32_t d) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> val, vmax, vmin, vsum;
        if constexpr (DTYPE_MODE == 2u) {
            RegTensor<bfloat16_t> raw;
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(raw, reinterpret_cast<__ubuf__ bfloat16_t*>(in_ptr) + c);
            Cast<float, bfloat16_t, toFp32>(val, raw, full);
        } else {
            RegTensor<half> raw;
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(raw, in_ptr + c);
            Cast<float, half, toFp32>(val, raw, full);
        }
        LoadAlign(vmax, vmax_ptr + c);
        LoadAlign(vmin, vmin_ptr + c);
        LoadAlign(vsum, vsum_ptr + c);
        Max(vmax, vmax, val, full);
        Min(vmin, vmin, val, full);
        Add(vsum, vsum, val, full);
        StoreAlign(vmax_ptr + c, vmax, full);
        StoreAlign(vmin_ptr + c, vmin, full);
        StoreAlign(vsum_ptr + c, vsum, full);
    }
}

/**
 * A7-4 smooth_v 定标单点（dispatch._quant_v_fp8 smooth_v 支路 / quant_fp8 ProcessV 同式）：
 * vm[c] = sum[c] / fp8_padded；amax[c] = max(|max-vm|, |min-vm|, 有 pad 时 |0-vm|=|vm|)；
 * scale[c] = amax[c] / scale_max（除法轨迹；全零通道 amax=0 → scale=0 的 NaN 容错
 * 与非 smooth 路径一致）。vm 原地写回 vsum_ptr，scale 写 scale_out_ptr。
 */
CATLASS_DEVICE_SIMD_VF void FinalizeSmoothVScaleVF(__ubuf__ float* vmax_ptr, __ubuf__ float* vmin_ptr,
                                               __ubuf__ float* vsum_ptr, __ubuf__ float* scale_out_ptr,
                                               uint32_t fp8_padded, uint32_t has_pad,
                                               float scale_max, uint32_t d) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> pad_reg, smax_reg, zero_reg;
    Duplicate(pad_reg, static_cast<float>(fp8_padded));
    Duplicate(smax_reg, scale_max);
    Duplicate(zero_reg, 0.0f);
    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> vmax, vmin, vm, hi, lo, amax;
        LoadAlign(vmax, vmax_ptr + c);
        LoadAlign(vmin, vmin_ptr + c);
        LoadAlign(vm, vsum_ptr + c);
        Div(vm, vm, pad_reg, full);            // vm = sum / fp8_padded
        Sub(hi, vmax, vm, full);
        Abs(hi, hi, full);
        Sub(lo, vmin, vm, full);
        Abs(lo, lo, full);
        Max(amax, hi, lo, full);
        if (has_pad != 0u) {
            RegTensor<float> vm_abs;
            Abs(vm_abs, vm, full);             // pad token (val=0) 贡献 |0 - vm|
            Max(amax, amax, vm_abs, full);
        }
        Div(amax, amax, smax_reg, full);       // scale = amax / scale_max
        StoreAlign(scale_out_ptr + c, amax, full);
        StoreAlign(vsum_ptr + c, vm, full);    // vm 原地写回
    }
}

/**
 * 7. 最终归一化与输出位型转换（DTYPE_MODE：2=bf16，其余按 fp16）。
 * 可选 smooth_v 的 vm 偏置：V 减均值量化下 PV 分子分母的 vm·Σp 项在归一化中
 * 恰好还原为 vm（不参与归一化），故出口直接逐通道加回（o·inv + vm）。
 */
template <uint32_t DTYPE_MODE>
CATLASS_DEVICE_SIMD_VF void FinalizeRowVF(__ubuf__ float* o_row, __ubuf__ uint16_t* out_row, float inv_d,
                                      __ubuf__ float* vm_ptr, uint32_t d) {
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
        if (vm_ptr != nullptr) {
            RegTensor<float> vm_val;
            LoadAlign(vm_val, vm_ptr + c);
            Add(x, x, vm_val, full);
        }
        if constexpr (DTYPE_MODE == 2u) {
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

/**
 * 8. UB float 区广播填充（P2-8 标量化收尾：scale 组清零 / STAGE 条带零填充等整区
 * 同值写的单点）：Duplicate 常量后按 64-lane chunk StoreAlign 写对齐段；
 * 尾部 ≥8 float 的 2 的幂段（32/16/8 float，均为 32 字节整数倍）以对齐掩码 StoreAlign 写出；
 * 尾部 <8 float（1~7 float，非 32 字节对齐）严禁使用 StoreAlign，转由 StoreUnAlign /
 * StoreUnAlignPost 写入，彻底根除非 32 字节对齐硬件 340 异常。
 */
CATLASS_DEVICE_SIMD_VF void FillF32VF(__ubuf__ float* dst, float value, uint32_t count) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    if (count == 0u) {
        return;
    }
    RegTensor<float> vreg;
    Duplicate(vreg, value);
    auto full = CreateMask<float, MaskPattern::ALL>();
    uint32_t c = 0;
    for (; c + 64u <= count; c += 64u) {
        StoreAlign(dst + c, vreg, full);
    }
    uint32_t rem = count - c;
    uint32_t off = c;
    if (rem & 32u) { auto m = CreateMask<float, MaskPattern::VL32>(); StoreAlign(dst + off, vreg, m); off += 32u; }
    if (rem & 16u) { auto m = CreateMask<float, MaskPattern::VL16>(); StoreAlign(dst + off, vreg, m); off += 16u; }
    if (rem & 8u)  { auto m = CreateMask<float, MaskPattern::VL8>();  StoreAlign(dst + off, vreg, m); off += 8u;  }
    uint32_t tail = rem & 7u;
    if (tail > 0u) {
        UnalignReg tail_store;
        __ubuf__ float* tail_ptr = dst + off;
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(tail_ptr, vreg, tail_store, tail);
        StoreUnAlignPost(tail_ptr, tail_store, 0);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

} // namespace dev_vf
} // namespace sage_attn
#endif
