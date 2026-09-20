/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_mul_sel_softmaxflashv2_cast_nz_dn_mxfp8_v2.h
 * \brief MXFP8场景DN布局softmax VF: 单V核/FP16 softmax/无row_sum/无softmaxscale版本
 *
 * 相比 vf_mul_sel_softmaxflashv2_cast_nz_dn.h 中的 MXFP8 实现, 本文件按以下设计重写:
 * 1. 基本块仍为128*512, 但V核从两个变为一个: m=128, 单个V核处理完整的128行S1,
 *    一行S1(128个FP16=256B)恰好占一个寄存器;
 * 2. 保持C1V1C1V1C2V2流水, 保留subLoop概念, 每次VF调用处理一个子环的ubN=256列;
 * 3. softmax数据类型从FP32改为FP16(T=half): S输入/行max/exp均在FP16域计算;
 * 4. 不计算row_sum: C2(MM2)阶段V矩阵d维多一行, p*V直接得到row_sum,
 *    因此VF侧删除全部sum寄存器与sum状态tensor(expSum/preLoopSum/firstLoopSum);
 * 5. 不乘softmaxscale(含Q/K descale): C1阶段FIXPIPE随路完成, S输入已完成缩放。
 *
 * 调用侧需同步适配的契约:
 * - m必须为128(寄存器一次放下一行S1), 输入S为FP16, DN列主序: 第c列起始地址为 input + c*m;
 * - softmaxMax/softmaxExp/preLoopMax状态tensor改为FP16, 每个m=128个元素;
 * - P输出FP8打包从4-way(4列x64B)改为2-way(2列x128B): 每个pack寄存器=[列A|列B]各128B,
 *   vselrIndexesBuf的DN_INDEX表需按新打包重排;
 * - P store的blockStride/repeatStride沿用旧布局取值(ubN>>2|1, 2), pack间指针间隔沿用+32B
 *   约定(x_exp_k = x_exp + 32*k), m=128的新L1排布需在调用侧重新对齐(可直接调VF接口传新值);
 * - pScaleSubLoop0(E8M0)由"64值x2份复制"改为每行一个共m=128个值, L1侧按s2分组复制需适配。
 */
#ifndef MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_MXFP8_V2_H_
#define MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_MXFP8_V2_H_
#include "kernel_tensor.h"
#include "vf_basic_block_utils.h"
#include "../util_regbase.h"

namespace FaVectorApi {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Reg;
using namespace regbaseutil;

// half -> bfloat16: 1:1满寄存器数值转换(供E8M0指数提取), 与indexer_quant_cache写法一致
static constexpr Reg::CastTrait castTraitHalf2Bf16Mx = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

/*
 * NoUpdate: 首个mLoop(isFirstS2Loop), 无历史全局max状态, 仅维护new_global_max。
 * 子环0: 计算本块行max(ln2对齐)后存new_global_max, 以对齐max为基数量化P;
 * 子环1: 与子环0的max合并, 冻结的P0通过pScaleSubLoop0(E8M0)做scale修正。
 */
template <typename T, typename T2, uint16_t ubN = 256>
__simd_vf__ inline void ProcessVec1DnNoUpdateMxfp8V2VF(__ubuf__ T2 *x_exp, __ubuf__ T *input_x_local_UB,
                                                       __ubuf__ T *new_global_max, __ubuf__ uint8_t *indexesUb,
                                                       __ubuf__ fp8_e8m0_t *pScaleSubLoop0, const uint32_t m,
                                                       const uint32_t originN, const float pScale, const T minValue,
                                                       const uint32_t blockStride, const uint32_t repeatStride,
                                                       int32_t subLoop)
{
    static_assert(ubN % 8 == 0, "ProcessVec1DnNoUpdateMxfp8V2VF ubN must be multiple of 8");
    RegTensor<T> src0, src1, src2, src3;
    RegTensor<T> max0, max1, max2, max3;
    RegTensor<T> vreg_x_max_b; // 子环1: 子环0存储的max及exp(旧max-新max)修正因子
    RegTensor<T> vreg_min;
    RegTensor<T> vreg_x_0, vreg_x_1, vreg_x_2, vreg_x_3, vreg_x_4, vreg_x_5, vreg_x_6, vreg_x_7;
    RegTensor<T> vreg_x_exp_0, vreg_x_exp_1, vreg_x_exp_2, vreg_x_exp_3;
    RegTensor<T> vreg_x_exp_4, vreg_x_exp_5, vreg_x_exp_6, vreg_x_exp_7;
    RegTensor<T> vreg_ln_p_scale; // ln(pScale)广播为整寄存器FP16
    RegTensor<T> vreg_ln_p_scale_lo, vreg_ln_p_scale_hi;
    RegTensor<float> vreg_p_scale_f;
    RegTensor<float> vreg_ln_p_scale_f;
    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_fp8_1, vreg_x_exp_fp8_2, vreg_x_exp_fp8_3;
    RegTensor<T2> vreg_x_exp_f8_pack_0, vreg_x_exp_f8_pack_1, vreg_x_exp_f8_pack_2, vreg_x_exp_f8_pack_3;
    RegTensor<uint8_t> idx_nd2nz;
    RegTensor<bfloat16_t> vreg_p_scale_bf16;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;

    MaskReg preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg preg_135 = CreateMask<T, MaskPattern::ALL>();
    MaskReg preg_f = CreateMask<float, MaskPattern::ALL>();

    // 4个pack寄存器store起始指针, 沿用原实现pack间+32(T2元素)的间隔约定
    __ubuf__ T2 *x_exp_1 = x_exp + 32;
    __ubuf__ T2 *x_exp_2 = x_exp + 64;
    __ubuf__ T2 *x_exp_3 = x_exp + 96;

    __ubuf__ T *src_ub0 = input_x_local_UB;
    __ubuf__ T *src_ub1 = src_ub0 + m;
    __ubuf__ T *src_ub2 = src_ub0 + m * 2;
    __ubuf__ T *src_ub3 = src_ub0 + m * 3;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    // ln(pScale)在float域计算, 再广播成整寄存器FP16
    Duplicate(vreg_p_scale_f, pScale);
    Ln(vreg_ln_p_scale_f, vreg_p_scale_f, preg_f);
    Cast<T, float, castTraitRintZero>(vreg_ln_p_scale_lo, vreg_ln_p_scale_f, preg_f);
    Cast<T, float, castTraitRintOne>(vreg_ln_p_scale_hi, vreg_ln_p_scale_f, preg_f);
    Or((RegTensor<uint16_t> &)vreg_ln_p_scale, (RegTensor<uint16_t> &)vreg_ln_p_scale_lo,
       (RegTensor<uint16_t> &)vreg_ln_p_scale_hi, preg_108);

    // S2尾部列填充minValue: 不影响行max, 对应P为0(对p*V无贡献)
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
        LoadAlign(src0, src_ub0 + iter_m * m * 4);
        LoadAlign(src1, src_ub1 + iter_m * m * 4);
        LoadAlign(src2, src_ub2 + iter_m * m * 4);
        LoadAlign(src3, src_ub3 + iter_m * m * 4);
        Max(max0, max0, src0, preg_108);
        Max(max1, max1, src1, preg_108);
        Max(max2, max2, src2, preg_108);
        Max(max3, max3, src3, preg_108);
    }

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    // ln2对齐: 保证exp(旧max-新max)为2的幂, E8M0 scale修正仅需改指数位
    // softmaxscale已由C1阶段FIXPIPE随路乘, 此处不再乘dScale
    Muls(max0, max0, static_cast<T>(INV_LN2), preg_108);
    Truncate<T, RoundMode::CAST_CEIL>(max0, max0, preg_108);
    Muls(max0, max0, static_cast<T>(LN2), preg_108);

    if (subLoop == 1) {
        // 与子环0存储的max合并; vreg_x_max_b = exp(旧max - 新max), 即P0的scale修正因子
        LoadAlign(vreg_x_max_b, new_global_max);
        Max(max0, max0, vreg_x_max_b, preg_108);
        FusedExpSub(vreg_x_max_b, vreg_x_max_b, max0, preg_134);
    }

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);
    // exp基值 = 对齐max - ln(pScale), 即P量化时预留pScale倍余量
    Sub(max0, max0, vreg_ln_p_scale, preg_108);

    LoadAlign(idx_nd2nz, indexesUb);
    uint16_t loopNum = ubN / 8;

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        // 每次迭代处理8列, 4组交错取列: 组g覆盖列[64g, 64g+64), 组内取列2*i0与2*i0+1;
        // exp_0~3为4组的偶数列(2*i0), exp_4~7为4组的奇数列(2*i0+1), 与原实现取列顺序一致
        LoadAlign(vreg_x_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_4, input_x_local_UB + i0 * m * 2 + m);
        LoadAlign(vreg_x_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + m);
        LoadAlign(vreg_x_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
        LoadAlign(vreg_x_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + m);
        LoadAlign(vreg_x_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + m);

        FusedExpSub(vreg_x_exp_0, vreg_x_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_3, max0, preg_134);

        // 2-way打包(half->fp8, 2:1): pack_0 = [组0偶数列 | 组1偶数列], 每列128B
        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_0, preg_134);
        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

        // pack_1 = [组2偶数列 | 组3偶数列]
        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_2, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_2, preg_134);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);

        FusedExpSub(vreg_x_exp_4, vreg_x_4, max0, preg_134);
        FusedExpSub(vreg_x_exp_5, vreg_x_5, max0, preg_134);
        FusedExpSub(vreg_x_exp_6, vreg_x_6, max0, preg_134);
        FusedExpSub(vreg_x_exp_7, vreg_x_7, max0, preg_134);

        // pack_2 = [组0奇数列 | 组1奇数列]
        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_2, vreg_x_exp_4, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_2, (RegTensor<uint8_t> &)vreg_x_exp_fp8_2,
           (RegTensor<uint8_t> &)vreg_x_exp_4, preg_134);
        Gather(vreg_x_exp_f8_pack_2, vreg_x_exp_fp8_2, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_2), vreg_x_exp_f8_pack_2, blockStride, repeatStride, preg_134);

        // pack_3 = [组2奇数列 | 组3奇数列]
        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_3, vreg_x_exp_6, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_3, (RegTensor<uint8_t> &)vreg_x_exp_fp8_3,
           (RegTensor<uint8_t> &)vreg_x_exp_6, preg_134);
        Gather(vreg_x_exp_f8_pack_3, vreg_x_exp_fp8_3, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_3), vreg_x_exp_f8_pack_3, blockStride, repeatStride, preg_134);
    }

    if (subLoop == 1) {
        // P0的scale修正因子exp(子环0 max - 新max)转为E8M0: half -> bf16(1:1) -> e8m0,
        // 每行一个共m=128个值, MM2读P0时通过MXFP8 scale随路乘
        Cast<bfloat16_t, T, castTraitHalf2Bf16Mx>(vreg_p_scale_bf16, vreg_x_max_b, preg_108);
        Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0, vreg_p_scale_bf16, preg_108);
        StoreAlign<fp8_e8m0_t, Reg::StoreDist::DIST_PACK_B16>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
                                                              vreg_p_scale_f8e8m0, preg_108);
    }
}

template <typename T, typename T2, uint16_t ubN = 256>
__aicore__ inline void ProcessVec1DnNoUpdateMxfp8V2(const LocalTensor<T2> &dstTensor, const LocalTensor<T> &maxTensor,
                                                    const LocalTensor<T> &srcTensor,
                                                    const LocalTensor<uint8_t> &vselrIndexesBuf, const uint32_t m,
                                                    const uint32_t originN, const float pScale, const T minValue,
                                                    int32_t subLoop,
                                                    const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T *input_x_local_UB = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ T *new_global_max = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0Tensor.GetPhyAddr();
    __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)vselrIndexesBuf.GetPhyAddr();
    // 沿用原DN布局FP8输出的store参数, m=128的新L1排布如需调整可直接调用VF接口传新值
    const uint32_t blockStride = ubN >> 2 | 0x1;
    const uint32_t repeatStride = 2;

    ProcessVec1DnNoUpdateMxfp8V2VF<T, T2, ubN>(x_exp, input_x_local_UB, new_global_max, indexesUb, pScaleSubLoop0Ub, m,
                                               originN, pScale, minValue, blockStride, repeatStride, subLoop);
}

/*
 * Update: 非首个mLoop, 行max需与历史全局max合并, 并维护三类修正状态:
 * - pre_loop_max: 子环0时保存历史全局max, 供子环1计算O重缩放因子;
 * - expMax: O重缩放因子exp(历史max - 新max), V2阶段重缩放已累积的O与row_sum;
 * - pScaleSubLoop0(E8M0): 本mLoop冻结P0的scale修正因子exp(子环0合并max - 新max),
 *   MM2读P0时随路乘。
 * 子环0: P0以max(本块, 历史)为基数量化; 子环1: P1以max(本块, 子环0, 历史)为基数量化。
 */
template <typename T, typename T2, uint16_t ubN = 256>
__simd_vf__ inline void ProcessVec1DnUpdateMxfp8V2VF(__ubuf__ T2 *x_exp, __ubuf__ T *input_x_local_UB,
                                                     __ubuf__ T *exp_max, __ubuf__ T *new_global_max,
                                                     __ubuf__ uint8_t *indexesUb, __ubuf__ fp8_e8m0_t *pScaleSubLoop0,
                                                     const uint32_t m, const uint32_t originN, const float pScale,
                                                     const T minValue, const uint32_t blockStride,
                                                     const uint32_t repeatStride, __ubuf__ T *pre_loop_max,
                                                     int32_t subLoop)
{
    static_assert(ubN % 8 == 0, "ProcessVec1DnUpdateMxfp8V2VF ubN must be multiple of 8");
    RegTensor<T> src0, src1, src2, src3;
    RegTensor<T> max0, max1, max2, max3;
    RegTensor<T> vreg_x_max_b;        // 历史max/exp修正因子
    RegTensor<T> vreg_subloop_update; // 子环1: P0的scale修正因子exp(子环0合并max - 新max)
    RegTensor<T> vreg_min;
    RegTensor<T> vreg_x_0, vreg_x_1, vreg_x_2, vreg_x_3, vreg_x_4, vreg_x_5, vreg_x_6, vreg_x_7;
    RegTensor<T> vreg_x_exp_0, vreg_x_exp_1, vreg_x_exp_2, vreg_x_exp_3;
    RegTensor<T> vreg_x_exp_4, vreg_x_exp_5, vreg_x_exp_6, vreg_x_exp_7;
    RegTensor<T> vreg_ln_p_scale; // ln(pScale)广播为整寄存器FP16
    RegTensor<T> vreg_ln_p_scale_lo, vreg_ln_p_scale_hi;
    RegTensor<float> vreg_p_scale_f;
    RegTensor<float> vreg_ln_p_scale_f;
    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_fp8_1, vreg_x_exp_fp8_2, vreg_x_exp_fp8_3;
    RegTensor<T2> vreg_x_exp_f8_pack_0, vreg_x_exp_f8_pack_1, vreg_x_exp_f8_pack_2, vreg_x_exp_f8_pack_3;
    RegTensor<uint8_t> idx_nd2nz;
    RegTensor<bfloat16_t> vreg_p_scale_bf16;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;

    MaskReg preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg preg_135 = CreateMask<T, MaskPattern::ALL>();
    MaskReg preg_f = CreateMask<float, MaskPattern::ALL>();

    // 4个pack寄存器store起始指针, 沿用原实现pack间+32(T2元素)的间隔约定
    __ubuf__ T2 *x_exp_1 = x_exp + 32;
    __ubuf__ T2 *x_exp_2 = x_exp + 64;
    __ubuf__ T2 *x_exp_3 = x_exp + 96;

    __ubuf__ T *src_ub0 = input_x_local_UB;
    __ubuf__ T *src_ub1 = src_ub0 + m;
    __ubuf__ T *src_ub2 = src_ub0 + m * 2;
    __ubuf__ T *src_ub3 = src_ub0 + m * 3;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    // ln(pScale)在float域计算, 再广播成整寄存器FP16
    Duplicate(vreg_p_scale_f, pScale);
    Ln(vreg_ln_p_scale_f, vreg_p_scale_f, preg_f);
    Cast<T, float, castTraitRintZero>(vreg_ln_p_scale_lo, vreg_ln_p_scale_f, preg_f);
    Cast<T, float, castTraitRintOne>(vreg_ln_p_scale_hi, vreg_ln_p_scale_f, preg_f);
    Or((RegTensor<uint16_t> &)vreg_ln_p_scale, (RegTensor<uint16_t> &)vreg_ln_p_scale_lo,
       (RegTensor<uint16_t> &)vreg_ln_p_scale_hi, preg_108);

    // S2尾部列填充minValue: 不影响行max, 对应P为0(对p*V无贡献)
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
        LoadAlign(src0, src_ub0 + iter_m * m * 4);
        LoadAlign(src1, src_ub1 + iter_m * m * 4);
        LoadAlign(src2, src_ub2 + iter_m * m * 4);
        LoadAlign(src3, src_ub3 + iter_m * m * 4);
        Max(max0, max0, src0, preg_108);
        Max(max1, max1, src1, preg_108);
        Max(max2, max2, src2, preg_108);
        Max(max3, max3, src3, preg_108);
    }

    LoadAlign(vreg_x_max_b, new_global_max); // 历史全局max(此前mLoop累积)
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    // ln2对齐: 保证exp(旧max-新max)为2的幂, E8M0 scale修正仅需改指数位
    // softmaxscale已由C1阶段FIXPIPE随路乘, 此处不再乘dScale
    Muls(max0, max0, static_cast<T>(INV_LN2), preg_108);
    Truncate<T, RoundMode::CAST_CEIL>(max0, max0, preg_108);
    Muls(max0, max0, static_cast<T>(LN2), preg_108);
    // 与历史max合并(两者均为ln2对齐, 合并后仍对齐)
    Max(max0, max0, vreg_x_max_b, preg_108);

    if (subLoop == 0) {
        // 保存历史max, 供子环1计算O重缩放因子
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)pre_loop_max, vreg_x_max_b, preg_108);
        FusedExpSub(vreg_x_max_b, vreg_x_max_b, max0, preg_134); // exp(历史max - 合并max)
    } else {
        // exp(子环0合并max - 新max): 本mLoop冻结P0的scale修正因子
        FusedExpSub(vreg_subloop_update, vreg_x_max_b, max0, preg_134);
        LoadAlign(vreg_x_max_b, pre_loop_max);
        FusedExpSub(vreg_x_max_b, vreg_x_max_b, max0, preg_134); // exp(历史max - 新max): O重缩放因子
    }

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);
    // exp基值 = 对齐max - ln(pScale), 即P量化时预留pScale倍余量
    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)exp_max, vreg_x_max_b, preg_108);

    LoadAlign(idx_nd2nz, indexesUb);
    uint16_t loopNum = ubN / 8;

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        // 每次迭代处理8列, 4组交错取列: 组g覆盖列[64g, 64g+64), 组内取列2*i0与2*i0+1
        LoadAlign(vreg_x_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_4, input_x_local_UB + i0 * m * 2 + m);
        LoadAlign(vreg_x_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + m);
        LoadAlign(vreg_x_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
        LoadAlign(vreg_x_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + m);
        LoadAlign(vreg_x_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + m);

        FusedExpSub(vreg_x_exp_0, vreg_x_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_3, max0, preg_134);

        // 2-way打包(half->fp8, 2:1): pack_0 = [组0偶数列 | 组1偶数列], 每列128B
        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_0, (RegTensor<uint8_t> &)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t> &)vreg_x_exp_0, preg_134);
        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

        // pack_1 = [组2偶数列 | 组3偶数列]
        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_2, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_1, (RegTensor<uint8_t> &)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t> &)vreg_x_exp_2, preg_134);
        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);

        FusedExpSub(vreg_x_exp_4, vreg_x_4, max0, preg_134);
        FusedExpSub(vreg_x_exp_5, vreg_x_5, max0, preg_134);
        FusedExpSub(vreg_x_exp_6, vreg_x_6, max0, preg_134);
        FusedExpSub(vreg_x_exp_7, vreg_x_7, max0, preg_134);

        // pack_2 = [组0奇数列 | 组1奇数列]
        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_2, vreg_x_exp_4, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_2, (RegTensor<uint8_t> &)vreg_x_exp_fp8_2,
           (RegTensor<uint8_t> &)vreg_x_exp_4, preg_134);
        Gather(vreg_x_exp_f8_pack_2, vreg_x_exp_fp8_2, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_2), vreg_x_exp_f8_pack_2, blockStride, repeatStride, preg_134);

        // pack_3 = [组2奇数列 | 组3奇数列]
        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_3, vreg_x_exp_6, preg_135);
        Cast<T2, T, castTraitRintOne>((RegTensor<T2> &)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        Or((RegTensor<uint8_t> &)vreg_x_exp_fp8_3, (RegTensor<uint8_t> &)vreg_x_exp_fp8_3,
           (RegTensor<uint8_t> &)vreg_x_exp_6, preg_134);
        Gather(vreg_x_exp_f8_pack_3, vreg_x_exp_fp8_3, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2 *&)x_exp_3), vreg_x_exp_f8_pack_3, blockStride, repeatStride, preg_134);
    }

    if (subLoop == 1) {
        // P0的scale修正因子exp(子环0合并max - 新max)转为E8M0: half -> bf16(1:1) -> e8m0,
        // 每行一个共m=128个值, MM2读P0时通过MXFP8 scale随路乘
        Cast<bfloat16_t, T, castTraitHalf2Bf16Mx>(vreg_p_scale_bf16, vreg_subloop_update, preg_108);
        Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0, vreg_p_scale_bf16, preg_108);
        StoreAlign<fp8_e8m0_t, Reg::StoreDist::DIST_PACK_B16>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
                                                              vreg_p_scale_f8e8m0, preg_108);
    }
}

template <typename T, typename T2, uint16_t ubN = 256>
__aicore__ inline void ProcessVec1DnUpdateMxfp8V2(const LocalTensor<T2> &dstTensor, const LocalTensor<T> &maxTensor,
                                                  const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor,
                                                  const LocalTensor<uint8_t> &vselrIndexesBuf, const uint32_t m,
                                                  const uint32_t originN, const float pScale, const T minValue,
                                                  const LocalTensor<T> &preLoopMaxTensor, int32_t subLoop,
                                                  const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T *input_x_local_UB = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ T *exp_max = (__ubuf__ T *)expMaxTensor.GetPhyAddr();
    __ubuf__ T *new_global_max = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *pre_loop_max = (__ubuf__ T *)preLoopMaxTensor.GetPhyAddr();
    __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0Tensor.GetPhyAddr();
    __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)vselrIndexesBuf.GetPhyAddr();
    // 沿用原DN布局FP8输出的store参数, m=128的新L1排布如需调整可直接调用VF接口传新值
    const uint32_t blockStride = ubN >> 2 | 0x1;
    const uint32_t repeatStride = 2;

    ProcessVec1DnUpdateMxfp8V2VF<T, T2, ubN>(x_exp, input_x_local_UB, exp_max, new_global_max, indexesUb,
                                             pScaleSubLoop0Ub, m, originN, pScale, minValue, blockStride, repeatStride,
                                             pre_loop_max, subLoop);
}

/*
 * @ingroup ProcessVec1VfDnMxfp8V2
 * @brief MXFP8场景DN布局单V核softmax VF入口: S(FP16) -> P(FP8) + max状态 + E8M0 scale
 * @param [out] dstTensor, P输出LocalTensor(FP8, L1)
 * @param [out] maxTensor, new_global_max(本子环合并后的ln2对齐行max)
 * @param [in] srcTensor, 输入S LocalTensor(FP16, DN列主序, 第c列起始地址为 + c*m)
 * @param [out] expMaxTensor, O重缩放因子exp(历史max - 新max), 仅Update使用
 * @param [in] vselrIndexesBuf, DN->NZ Gather索引表(需按2-way新打包重排)
 * @param [out] pScaleSubLoop0Tensor, P0的E8M0 scale修正因子, 仅subLoop==1写入
 * @param [in] m, 输入行数, 必须为128(一行S1恰好占满一个寄存器)
 * @param [in] originN, 输入原始列数, 0 < originN <= ubN
 * @param [in] pScale, P量化的scale余量
 * @param [in] preLoopMaxTensor, 历史max快照, 仅Update使用
 * @param [in] subLoop, 子环编号(0/1), 对应C1V1C1V1C2V2中的两次V1
 */
template <typename T, typename T2, bool isUpdate = false, uint16_t ubN = 256>
__aicore__ inline void ProcessVec1VfDnMxfp8V2(const LocalTensor<T2> &dstTensor, const LocalTensor<T> &maxTensor,
                                              const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor,
                                              TBuf<> *vselrIndexesBuf,
                                              const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor, const uint32_t m,
                                              const uint32_t originN, const float pScale, const T minValue,
                                              const LocalTensor<T> &preLoopMaxTensor, int32_t subLoop)
{
    static_assert(IsSameType<T, half>::value, "ProcessVec1VfDnMxfp8V2, T must be half(FP16 softmax)");
    static_assert(IsSameType<T2, fp8_e4m3fn_t>::value, "ProcessVec1VfDnMxfp8V2, T2 must be fp8_e4m3fn(MXFP8)");
    LocalTensor<uint8_t> indexesTensor;
    indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
    if constexpr (!isUpdate) {
        ProcessVec1DnNoUpdateMxfp8V2<T, T2, ubN>(dstTensor, maxTensor, srcTensor, indexesTensor, m, originN, pScale,
                                                 minValue, subLoop, pScaleSubLoop0Tensor);
    } else {
        ProcessVec1DnUpdateMxfp8V2<T, T2, ubN>(dstTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, m, originN,
                                               pScale, minValue, preLoopMaxTensor, subLoop, pScaleSubLoop0Tensor);
    }
}
} // namespace FaVectorApi

#endif // MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_MXFP8_V2_H_
