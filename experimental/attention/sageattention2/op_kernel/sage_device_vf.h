#pragma once

#include <cstdint>
#include <limits>
#include "../sageattn_fp8/ascendc/tiling.h"
#include "../sageattn_fp8/ascendc/online_softmax.h"


#if SAGE_ATTN_DEVICE_COMPILE
#include "kernel_operator.h"
// 官方 CATLASS 设备函数宏（catlass/detail/macros.hpp）：CATLASS_DEVICE_SIMD_VF =
// __simd_vf__ static（CATLASS_ARCH=3510），与官方 950 FA softmax epilogue 同款标记，
// 设备核内所有 VF 原语显式标注 __simd_vf__ __aicore__ inline __attribute__((always_inline))，
// 保持绝对零函数调用栈帧开销。
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
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void RowAbsMaxRowsVF(__ubuf__ half* in_base, __ubuf__ float* out_rowmax,
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
 * - CATLASS 950 模式：直接由 scale_ptr 经 LoadAlign<float, LoadDist::DIST_BRC_B32>
 *   随路完成 UB 标量加载与 64 通道全局广播，彻底消除标量单元 GetValue 与独立的向量 Duplicate 指令；
 * - 指针对齐与微架构安全：scale_ptr 继承 UB Slot 32 字节对齐基址与 float（4 字节）步长偏移，
 *   严格满足 LoadDist::DIST_BRC_B32 在 dav-3510 下的 32 位（4 字节）对齐约束，微架构执行零风险；
 * - per_thread：half-away-from-zero——x 加 0.5·sign(x) 后 CAST_TRUNC（对齐
 *   quant::RoundHalfAwayFromZero 的 floor(|x|+0.5)·sign）；
 * - per_warp：CAST_RINT（对齐 quant::RoundNearestEven / cvt.rni）；
 * - 饱和 ±127（与两个 Round* 的 clamp 一致）；
 * - 打包与写出：微架构 StoreDist::DIST_PACK4_B32 硬件压包直写 UB。
 */
template <uint32_t DTYPE_MODE>
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void QuantRowInt8VF(__ubuf__ half* in_ptr, __ubuf__ int8_t* out_ptr,
                                       __ubuf__ float* scale_ptr, uint32_t d,
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
    // CATLASS 950 模式：LoadAlign 随路完成 UB 标量加载与 64 通道全局广播
    LoadAlign<float, LoadDist::DIST_BRC_B32>(scale_reg, scale_ptr);
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
            RegTensor<float> bias;
            MaskReg neg;
            Compare<float, CMPMODE::LT>(neg, val, zero, full);
            Select(bias, neg_half_reg, half_reg, neg);
            Add(val, val, bias, full);
            Cast<int32_t, float, toI32Trunc>(packed_i, val, full);
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
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void QuantVRowFp8VF(__ubuf__ half* in_ptr, __ubuf__ float8_e4m3_t* out_ptr,
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
        RegTensor<float> val, scale_reg;
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
        Div(val, val, scale_reg, full);
        // 全零通道 NaN 守卫（收敛轮 1/R1-1）：不依赖硬件 Min/Max NaN 语义——
        // scale=0 → Div 产 NaN → SAT cast 保留 e4m3 0x7f（对齐 golden）
        MaskReg zero_scale;
        Compare<float, CMPMODE::EQ>(zero_scale, scale_reg, zero_reg, full);
        Select(val, nan_reg, val, zero_scale);
        Min(val, val, scale_max_reg, full);
        Max(val, val, neg_scale_max_reg, full);
        Cast<float8_e4m3_t, float, toFp8>(packed, val, full);
        StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out_ptr + c, packed, full);
    }
}

/**
 * 4a. Logits 单 chunk 反量化分片助手（4 融合原语的行内编译期展开单元）：
 * 执行与融合前完全一致的指令轨迹——dot32 读取、int32→fp32 cast、twin::DequantQkLogit
 * 左结合三乘链、三态掩码快速路径——并将结果驻留 out_val 寄存器瓦片返回给调用方
 * （StoreAlign 位保写出 UB：寄存器值 ≡ UB 回读值，供行内 Max/ExpSub 直用免二次加载）。
 * 硬件 ABI 约束（dav-3510 前端检查"simd_vf function can only call simd_callee function"
 * 与"simd_vf 函数指针形参必须位于 __ubuf__ 地址空间"）：跨函数传递寄存器瓦片的用户
 * 助手须以官方 MicroAPI 内在函数同款 __simd_callee__ 标记，且形参类型于 dev_vf 命名
 * 空间作用域以全限定 AscendC::Reg::RegTensor 承接（MicroAPI 即 Reg 的命名空间别名，
 * kernel_macros.h L129；函数体内 using 指令不作用于签名解析）；MaskReg 谓词形参禁用，
 * full 由助手内部 CreateMask 现场生成（编译期全真，零运行时开销）。always_inline
 * 内联后与单函数内联展开同构、零调用栈帧。
 * 注意：台账所记"×2^-11"定点折算若按字面串入乘链，将给结果附加恰 2^-11 的精确
 * 幂偏置、破坏与宿主的逐位一致性，故不采用（2 的整数次幂乘法精确、不可消去）。
 */
__simd_callee__ __aicore__ inline __attribute__((always_inline)) void DequantStoreLogitChunkVF(
    __ubuf__ int32_t* dot_row, __ubuf__ float* logits_row, uint32_t j,
    AscendC::Reg::RegTensor<float>& qs_reg, __ubuf__ float* k_scale_row,
    AscendC::Reg::RegTensor<float>& sm_reg, AscendC::Reg::RegTensor<float>& neg_inf,
    uint32_t token_q, uint32_t k_start, uint32_t sk,
    uint32_t is_causal, bool row_pure_active, AscendC::Reg::RegTensor<float>& out_val) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    static constexpr CastTrait toFp32{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    auto full = CreateMask<float, MaskPattern::ALL>();

    const uint32_t chunk_start = k_start + j;
    uint32_t bound_k = sk;
    if (is_causal != 0u && token_q + 1u < bound_k) {
        bound_k = token_q + 1u;
    }

    // Chunk 级态 1：全掩码（因果上三角或超 sk 尾界），跳过点积读取、Cast 与三次乘法，
    // 直接落 -1e30 哨兵（哨兵同步驻留 out_val 寄存器瓦片，行内 Max/ExpSub 链与 UB 回读同值）
    if (!row_pure_active && chunk_start >= bound_k) {
        out_val = neg_inf;
        StoreAlign(logits_row + j, neg_inf, full);
        return;
    }

    RegTensor<float> ks_vec;
    RegTensor<int32_t> dot_i;
    LoadAlign(dot_i, dot_row + j);
    Cast<float, int32_t, toFp32>(out_val, dot_i, full);   // INT32→FP32 精确 cast（≤2^21 整数域）
    Mul(out_val, out_val, qs_reg, full);                  // · q_scale（行解析，先乘）
    LoadAlign(ks_vec, k_scale_row + j);
    Mul(out_val, out_val, ks_vec, full);                  // · k_scale[j]（块解析向量）
    Mul(out_val, out_val, sm_reg, full);                  // · sm_scale_log2e（左结合同宿主）

    // Chunk 级态 2：全活跃（完全在因果下三角与 sk 尾界内），直接写反量化结果
    if (row_pure_active || chunk_start + 64u <= bound_k) {
        StoreAlign(logits_row + j, out_val, full);
        return;
    }

    // Chunk 级态 3：混合边界，复用 CATLASS CreateMask 与 UpdateMask 动态尾块硬件掩码生成
    uint32_t tailN = bound_k - chunk_start;
    MaskReg pregTailN = UpdateMask<float>(tailN);
    Select(out_val, out_val, neg_inf, pregTailN);
    StoreAlign(logits_row + j, out_val, full);
}

/**
 * 4b. 寄存器瓦片 ExpSub 指数 + 448· 放大跨 chunk 向量累加单步助手（4 融合原语单元）：
 * exp2 域 + 448· 放大（twin OnlineBlockExp2Sum 同式，收敛轮 F/DEFECT-F-2）：
 * cur_sum = Σ 2^(x-m)·448 = 448·Σ Exp((x-m)·ln2)。
 * CATLASS 950 MicroAPI ExpSub + MulDstAdd 指令级下沉（M6-1 已核准语义）：
 * 1. max_nat 为自然域行最大，ExpSub 熔断减法与指数本征指令；
 * 2. MulDstAdd 硬件语义 (dst = dst * src0 + src1): nat (dst) * e4m3_max_reg (src0) + sum_val (src1) -> nat，
 *    sum_val = nat 保持跨 chunk 64-lane 向量累加器严格同序。
 * 同 4a：__simd_callee__ 标记 + AscendC::Reg::RegTensor 全限定形参（ABI 合法形态）。
 */
__simd_callee__ __aicore__ inline __attribute__((always_inline)) void ExpSubAccumChunkVF(
    AscendC::Reg::RegTensor<float>& val, AscendC::Reg::RegTensor<float>& max_nat,
    AscendC::Reg::RegTensor<float>& e4m3_max_reg, AscendC::Reg::RegTensor<float>& sum_val) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> nat;
    Muls(val, val, INV_LOG2E, full);
    ExpSub(nat, val, max_nat, full);
    MulDstAdd(nat, e4m3_max_reg, sum_val, full);
    sum_val = nat;
}

/**
 * 4. QK 块 Logits 瓦片寄存器流式融合（M6-3「挖掘 Softmax 进化与 CATLASS 复用 3」，
 *    R4 行批化）：借鉴 CATLASS 950 FA Softmax Epilogue 的 ComputeMaskandScale /
 *    ComputeExpSubSum 行流水组织模式（block_epilogue_fa_softmax_ascend950.hpp），
 *    融合原 4a（DequantMaskLogitsRowsVF）与原 4（BlockMaxAndSumRowsVF）两趟结构。
 *    原结构中同一份 float Logits 在 UB 往返 4 次（反量化 1 写 + Max 趟 1 读 +
 *    Exp/Sum 趟 1 读 + 下游 SoftmaxProb 1 读，共 3 读 1 写），严重挤占 UB 读写端口；
 *    本原语将反量化、三态掩码、行内 Max 与 ExpSub 指数求和熔断为单遍行流水：
 * - 逐 64 元素 chunk 从 dot32 交接槽做唯一一次流式读取，int32→FP32 反量化与
 *   因果/尾块三态掩码后，chunk 值全程驻留向量寄存器（RegTensor）；A7 dot32 台账
 *   封顶 block_n ≤ 128（host 强制 n ∈ {64,128}，sage_attn_host_tiling.h 越界拒绝），
 *   即每行至多 2 个 64-lane 瓦片，按 NCHUNK 编译期展开、零动态寄存器索引；
 * - 行内最大值由寄存器 Max 链 + ReduceMax 直接得出（消除原 Max 趟 UB 回读）；
 * - 随即在寄存器瓦片上原位执行 Muls(ln2)→ExpSub→MulDstAdd(448·) 累加链（4b 助手，
 *   消除原 Exp/Sum 趟 UB 回读）；Logits 仅 1 次流式写出 UB 供下游 SoftmaxProbRowsVF
 *   按 Online 更新后的 running max 生成 FP8 P——该次读受位级黄金语义约束必需
 *   running max（P = 2^(x-m')·448，m' 为 Online 步进后的行状态，非本块局部 max，
 *   若以块局部 max 铸 FP8 将在 e4m3 次正规域引入舍入差、破坏 matched_ratio=1.0），
 *   按契约保留；对 UB Logits 的 3 次重复加载由此消除为 1 次流式读取，
 *   Max/Sum 中间状态完全驻留向量寄存器；
 * - 原地轨迹：float logits 按 [row][block_n] 布局覆写同一 dot32 缓冲（int32→float 元素
 *   等宽）——同 chunk 地址 Load 先于 Store（寄存器 RAW 序，各 chunk 地址互斥）；dot32
 *   为单缓冲读保护交接槽，本区覆写发生在 wait QK_P_READY 之后、V_S 排空 + set DOT_FREE
 *   之前，Cube 下一拍覆写被 FLAG_DOT_FREE 挡在 V_S 之后，无竞争窗口；
 * - 数值唯一依据 twin::DequantQkLogit 左结合乘序：(float)dot · q_scale · k_scale[j] ·
 *   sm_scale_log2e——int32→fp32 cast 在 |dot| ≤ 2^21 整数域内精确（01_design 2.9：
 *   127·127·128 ≈ 2^20.97），三次标量/向量乘按宿主同一结合序逐位同式；
 * - 行批等价与位级一致性论证（R4/M6-3）：行间独立，跨行在途交织不改各行 FP 轨迹。
 *   StoreAlign 位保，寄存器瓦片直用 ≡ 原两趟 UB 回读同位型；chunk 顺序 Max 链与
 *   MulDstAdd 跨 chunk 向量累加链与原两趟严格同序；掩码哨兵 -1e30 全掩码行
 *   （x ≡ max，ExpSub(0)=1、448· 逐 chunk 累加 = 448·block_n）与 Online 步进中
 *   alpha_cur 下溢清零的联合行为同原实现逐位一致；
 * - 行解析 q_scale 经 DIST_BRC_B32 单元素广播 Load（读同一 ub_qs 槽位的同一 float，
 *   搬运非 FP 运算）；三态掩码快速路径（逐行独立判定）：① 全活跃态免除全部掩码
 *   计算；② 全掩码态跳过点积读取与乘法直接落哨兵；③ 混合边界态仅对触碰因果对角线
 *   或尾块 pad 界的 chunk 执行 UpdateMask 掩码 Select——逐分支保持原形态；
 * - 行 max/sum 采用 CANN MicroAPI 标准 StoreUnAlign / StoreUnAlignPost 流式非对齐
 *   写原语满 32 字节（8 float）对齐刷写（DaVinci VEC 访 UB 32 字节对齐约束，根除
 *   硬件 340 异常），收尾 LocalMemBar 保证后继 Online 步进与 SoftmaxProb 的 UB 读
 *   可见本函数全部写出。
 */
template <uint32_t NCHUNK>
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void DequantMaskMaxSumRowsVFImpl(
    __ubuf__ float* logits_dot_base, uint32_t rows,
    __ubuf__ float* qs_row, __ubuf__ float* k_scale_row,
    float sm_scale_log2e, uint32_t cur_q_start, uint32_t cur_q_len,
    uint32_t k_start, uint32_t sk, uint32_t is_causal, uint32_t block_n,
    __ubuf__ float* out_max_row, __ubuf__ float* out_sum_row) {
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    __ubuf__ int32_t* dot_base = reinterpret_cast<__ubuf__ int32_t*>(logits_dot_base);
    __ubuf__ float* max_ptr = out_max_row;
    __ubuf__ float* sum_ptr = out_sum_row;
    UnalignReg max_store;
    UnalignReg sum_store;

    RegTensor<float> sm_reg;
    Duplicate(sm_reg, sm_scale_log2e);
    RegTensor<float> neg_inf;
    Duplicate(neg_inf, SAGE_MASK_SENTINEL);
    RegTensor<float> e4m3_max_reg;
    Duplicate(e4m3_max_reg, FP8_E4M3_MAX);

    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t token_q = cur_q_start + (r % cur_q_len);
        __ubuf__ int32_t* dot_row = dot_base + r * block_n;
        __ubuf__ float* logits_row = logits_dot_base + r * block_n;

        // 行级快速路径 1：全掩码行（整行处于因果上三角或超 sk 尾界），哨兵直落 UB 与寄存器瓦片
        const bool is_pure_masked = (is_causal != 0u && token_q < k_start) || (k_start >= sk);
        // 行级快速路径 2：全活跃行（整行完全在因果下三角与 sk 尾界内），免除全部掩码计算
        const bool is_pure_active = (is_causal == 0u || token_q >= k_start + block_n - 1) && (k_start + block_n <= sk);

        RegTensor<float> max_val, sum_val;
        Duplicate(max_val, SAGE_MASK_SENTINEL);
        Duplicate(sum_val, 0.0f);

        RegTensor<float> val0;
        RegTensor<float> val1;
        if (is_pure_masked) {
            val0 = neg_inf;
            StoreAlign(logits_row, neg_inf, full);
            if constexpr (NCHUNK > 1u) {
                val1 = neg_inf;
                StoreAlign(logits_row + 64u, neg_inf, full);
            }
        } else {
            // 行解析 q_scale：DIST_BRC_B32 单元素广播 Load（读 ub_qs[r] 原 float，位保搬运）
            RegTensor<float> qs_reg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(qs_reg, qs_row + r);
            DequantStoreLogitChunkVF(dot_row, logits_row, 0u, qs_reg, k_scale_row, sm_reg, neg_inf,
                                     token_q, k_start, sk, is_causal, is_pure_active, val0);
            Max(max_val, max_val, val0, full);
            if constexpr (NCHUNK > 1u) {
                DequantStoreLogitChunkVF(dot_row, logits_row, 64u, qs_reg, k_scale_row, sm_reg, neg_inf,
                                         token_q, k_start, sk, is_causal, is_pure_active, val1);
                Max(max_val, max_val, val1, full);
            }
        }

        ReduceMax(max_val, max_val, full);

        RegTensor<float> max_nat;
        Muls(max_nat, max_val, INV_LOG2E, full);

        ExpSubAccumChunkVF(val0, max_nat, e4m3_max_reg, sum_val);
        if constexpr (NCHUNK > 1u) {
            ExpSubAccumChunkVF(val1, max_nat, e4m3_max_reg, sum_val);
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
 * 4. Logits 瓦片寄存器流式融合原语入口：按 block_n 台账取值域（64/128）编译期分派
 * 瓦片数 NCHUNK（每行 1 或 2 个 64-lane chunk），block_n > 128 由 host tiling 越界拒绝封顶。
 * 入口保持普通 __aicore__ inline（非 __simd_vf__）——dav-3510 前端约束"simd_vf 只可
 * 调用 simd_callee"，simd_vf→simd_vf 跨瓦片数分派调用不合法；普通函数调用 simd_vf 合法。
 */
__aicore__ inline __attribute__((always_inline)) void DequantMaskMaxSumRowsVF(
    __ubuf__ float* logits_dot_base, uint32_t rows,
    __ubuf__ float* qs_row, __ubuf__ float* k_scale_row,
    float sm_scale_log2e, uint32_t cur_q_start, uint32_t cur_q_len,
    uint32_t k_start, uint32_t sk, uint32_t is_causal, uint32_t block_n,
    __ubuf__ float* out_max_row, __ubuf__ float* out_sum_row) {
    if (rows == 0u) {
        return;
    }
    if (block_n <= 64u) {
        DequantMaskMaxSumRowsVFImpl<1u>(logits_dot_base, rows, qs_row, k_scale_row, sm_scale_log2e,
                                        cur_q_start, cur_q_len, k_start, sk, is_causal, block_n,
                                        out_max_row, out_sum_row);
    } else {
        DequantMaskMaxSumRowsVFImpl<2u>(logits_dot_base, rows, qs_row, k_scale_row, sm_scale_log2e,
                                        cur_q_start, cur_q_len, k_start, sk, is_causal, block_n,
                                        out_max_row, out_sum_row);
    }
}

/**
 * 4.5. 全量纯向量化 Online Softmax 状态步进（R4 行批化）：
 * 消除全部标量循环与 V_S / S_V 跨引擎同步气泡。
 * 纯向量 SIMD 指令更新 running_max 与 running_sum，并计算 O 累加器的 rescale 因子：
 * - m_new = max(m_old, m_cur)
 * - alpha_old = exp((m_old - m_new) * ln2)
 * - alpha_cur = exp((m_cur - m_new) * ln2)
 * - d_new = d_old * alpha_old + d_cur * alpha_cur
 * - ub_factor = alpha_old (用于 O 主累加器原地重缩放)
 */
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void UpdateOnlineSoftmaxRowsVF(
    __ubuf__ float* running_max,
    __ubuf__ float* running_sum,
    __ubuf__ float* block_max,
    __ubuf__ float* block_sum,
    __ubuf__ float* ub_factor,
    uint32_t rows,
    uint32_t k)
{
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();

    for (uint32_t r = 0; r < rows; r += 64) {
        RegTensor<float> m_cur, d_cur;
        LoadAlign(m_cur, block_max + r);
        LoadAlign(d_cur, block_sum + r);

        RegTensor<float> m_new, d_new, alpha_old;
        if (k == 0) {
            m_new = m_cur;
            d_new = d_cur;
            Duplicate(alpha_old, 1.0f);
        } else {
            RegTensor<float> m_old, d_old;
            LoadAlign(m_old, running_max + r);
            LoadAlign(d_old, running_sum + r);

            // m_new = max(m_old, m_cur)
            Max(m_new, m_old, m_cur, full);

            // alpha_old = exp2(m_old - m_new) = exp((m_old - m_new) * ln2) (ExpSub 熔断)
            // alpha_cur = exp2(m_cur - m_new) = exp((m_cur - m_new) * ln2) (ExpSub 熔断)
            // 原地缩放 m_old 与 m_cur 至自然域，消除冗余寄存器分配
            RegTensor<float> m_new_nat, alpha_cur;
            Muls(m_old, m_old, INV_LOG2E, full);
            Muls(m_cur, m_cur, INV_LOG2E, full);
            Muls(m_new_nat, m_new, INV_LOG2E, full);
            ExpSub(alpha_old, m_old, m_new_nat, full);
            ExpSub(alpha_cur, m_cur, m_new_nat, full);

            // 先行将 alpha_old 刷入 ub_factor（用于主累加器 O 的原地重缩放因子）
            StoreAlign(ub_factor + r, alpha_old, full);

            // d_new = d_old * alpha_old + d_cur * alpha_cur (CATLASS MicroAPI MulDstAdd 硬件熔断)
            // MulDstAdd(dst, src0, src1): dst = dst * src0 + src1
            // 此处 alpha_old (dst) * d_old (src0) + (d_cur * alpha_cur) (src1) -> alpha_old (即 d_new)
            Mul(alpha_cur, d_cur, alpha_cur, full);
            MulDstAdd(alpha_old, d_old, alpha_cur, full);
            d_new = alpha_old;
        }

        StoreAlign(running_max + r, m_new, full);
        StoreAlign(running_sum + r, d_new, full);
        if (k == 0) {
            StoreAlign(ub_factor + r, alpha_old, full);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

/**
 * 5. O 累加器行批原地缩放（R4）：纯向量 SIMD 广播缩放
 */
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void RescaleAccumRowsVF(
    __ubuf__ float* o_base,
    __ubuf__ float* factor_row,
    uint32_t rows,
    uint32_t d)
{
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t r = 0; r < rows; ++r) {
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
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

/**
 * 6. Softmax 概率矩阵生成（R4 行批化）：exp2 域恒等 P = 2^(x-m')·448，并饱和转换至 FP8 e4m3fn
 */
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void SoftmaxProbRowsVF(
    __ubuf__ float* logits_base,
    __ubuf__ float8_e4m3_t* out_p_base,
    __ubuf__ float* running_max,
    uint32_t rows,
    uint32_t block_n)
{
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFp8{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint32_t r = 0; r < rows; ++r) {
        RegTensor<float> m_nat;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(m_nat, running_max + r);
        Muls(m_nat, m_nat, INV_LOG2E, full);
        __ubuf__ float* logits = logits_base + r * block_n;
        __ubuf__ float8_e4m3_t* out_p = out_p_base + r * block_n;
        for (uint32_t j = 0; j < block_n; j += 64) {
            RegTensor<float> logit_val, exp_val;
            RegTensor<float8_e4m3_t> packed;
            LoadAlign(logit_val, logits + j);
            Muls(logit_val, logit_val, INV_LOG2E, full);
            ExpSub(exp_val, logit_val, m_nat, full);
            Muls(exp_val, exp_val, FP8_E4M3_MAX, full);
            Cast<float8_e4m3_t, float, toFp8>(packed, exp_val, full);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out_p + j, packed, full);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
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
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void MergePvPartialRowsVF(__ubuf__ float* partial_base, __ubuf__ float* o_base,
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
            LoadAlign(o, o_row + c);
            if (fp16_round != 0u) {
                Mul(partial, partial, vs, full);
                RegTensor<half> rounded;
                Cast<half, float, toFp16>(rounded, partial, full);
                Cast<float, half, toFp32>(partial, rounded, full);
                Add(o, o, partial, full);
            } else {
                // CATLASS MicroAPI MulDstAdd: dst = dst * src0 + src1
                // 此处 partial (dst) * vs (src0) + o (src1) -> partial (累加和)
                MulDstAdd(partial, vs, o, full);
                o = partial;
            }
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
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void ChannelAmaxAccumVF(__ubuf__ half* in_ptr, __ubuf__ float* amax_ptr,
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
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void FinalizeVScaleVF(__ubuf__ float* amax_ptr, __ubuf__ float* out_scale_ptr,
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
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void ChannelStatsAccumVF(__ubuf__ half* in_ptr, __ubuf__ float* vmax_ptr,
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
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void FinalizeSmoothVScaleVF(__ubuf__ float* vmax_ptr, __ubuf__ float* vmin_ptr,
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
 * 7. 最终行归一化与输出位型转换（DTYPE_MODE：2=bf16，其余按 fp16）。
 * - 借鉴 CATLASS 950 官方源码模式（FlashUpdateLastNew 与 LastDivNew）：
 *   行首直接通过 LoadAlign<float, LoadDist::DIST_BRC_B32> 从 d_run_ptr + row_idx 随路加载
 *   当前行的 running sum 分母值并广播到 64 通道；
 * - 寄存器内部就地计算归一化系数 inv_reg = (d > 0.0f) ? (1.0f / d) : 0.0f，
 *   彻底消灭原独立的 ComputeInvDenominatorVF 全局遍历、UB 暂存区往返读写与 LocalMemBar 屏障；
 * - 可选 smooth_v 的 vm 偏置：V 减均值量化下 PV 分子分母的 vm·Σp 项在归一化中
 *   恰好还原为 vm（不参与归一化），故出口直接逐通道加回（MulDstAdd 熔断 o·inv + vm）。
 */
template <uint32_t DTYPE_MODE>
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void FinalizeRowVF(
    __ubuf__ float* o_row,
    __ubuf__ uint16_t* out_row,
    __ubuf__ float* d_run_ptr,
    uint32_t row_idx,
    __ubuf__ float* vm_ptr,
    uint32_t d)
{
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toHalf{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toBf16{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    // CATLASS 950 模式（FlashUpdateLastNew / LastDivNew）：随路加载分母并在寄存器内求倒数
    // 微架构安全：d_run_ptr 继承 SlotV1RowSum 32B 对齐基址，row_idx 步长 4B，严格满足 DIST_BRC_B32 32 位对齐约束；
    // 寄存器分配打磨：Duplicate(inv_reg, 1.0f) 后 Div(inv_reg, inv_reg, d_reg) 原地求倒，彻底消除 one_reg 冗余寄存器；
    // 乱序与流水线掩盖：在 Div 发射前就绪 pos_mask，当 d <= 0.0f 时 Select 填 0.0f 防护非正常行数值溢出。
    RegTensor<float> d_reg, inv_reg, zero_reg;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(d_reg, d_run_ptr + row_idx);
    Duplicate(zero_reg, 0.0f);
    MaskReg pos_mask;
    Compare<float, CMPMODE::GT>(pos_mask, d_reg, zero_reg, full);
    Duplicate(inv_reg, 1.0f);
    Div(inv_reg, inv_reg, d_reg, full);
    Select(inv_reg, inv_reg, zero_reg, pos_mask);

    for (uint32_t c = 0; c < d; c += 64) {
        RegTensor<float> x;
        LoadAlign(x, o_row + c);
        if (vm_ptr != nullptr) {
            RegTensor<float> vm_val;
            LoadAlign(vm_val, vm_ptr + c);
            // CATLASS MicroAPI MulDstAdd: dst = dst * src0 + src1
            // x (dst) * inv_reg (src0) + vm_val (src1) -> x
            MulDstAdd(x, inv_reg, vm_val, full);
        } else {
            Mul(x, x, inv_reg, full);
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
 * 尾部 ≥8 float 且满足 32 字节对齐段通过 CATLASS UpdateMask<float>(rem_aligned) 动态硬件掩码 StoreAlign 写出；
 * 尾部 <8 float（1~7 float，非 32 字节对齐）转由 MicroAPI StoreUnAlign / StoreUnAlignPost 写入，
 * 彻底根除非 32 字节对齐硬件 340 异常，且零遗漏任意尾部元素。
 */
__simd_vf__ __aicore__ inline __attribute__((always_inline)) void FillF32VF(__ubuf__ float* dst, float value, uint32_t count) {
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
    uint32_t rem_aligned = rem & ~7u;
    if (rem_aligned > 0u) {
        MaskReg m = UpdateMask<float>(rem_aligned);
        StoreAlign(dst + off, vreg, m);
        off += rem_aligned;
    }
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
