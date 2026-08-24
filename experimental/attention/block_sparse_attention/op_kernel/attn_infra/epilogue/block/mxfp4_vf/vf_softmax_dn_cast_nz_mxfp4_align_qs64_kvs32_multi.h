/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You can obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file vf_softmax_dn_cast_nz_mxfp4_align_qs64_kvs32_multi.h
 * \brief softmax + mxfp4 quant for qs=64, kvs padding 32 multi (kvsActBaseTileAlign32 >= 64, runtime variable)
 */

#ifndef VF_SOFTMAX_DN_CAST_NZ_MXFP4_ALIGN_QS64_KVS32_MULTI_H_
#define VF_SOFTMAX_DN_CAST_NZ_MXFP4_ALIGN_QS64_KVS32_MULTI_H_
#include "vf_common_def.h"
#include "../../bsa_epilogue_dispatch_policy.hpp"

namespace NpuArch::Epilogue::Block::Mxfp4VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;


// ============================================================================
// 公共代码：处理 128 行（blockshapeY=128）的一个大块（chunk 0 = 行 [0,128), chunk 1 = 行 [128,256)）。
// 与 kvs256 的 chunk 结构完全一致：块内拆 4 个组（每组 32 行），
//   组 [0, validGroups)   -- 按原有流水计算（softmax + mxfp4 量化）
//   组 [validGroups, 4)   -- padding：写 MIN_VALUE 使 pscale=0 + pDest 填 0
// hasChunk=false 时整块按 padding 处理（无第二个 128 行块的场景）。
// qs=64：向量行宽仍 128 half(256B)，但仅前 64 个元素有效(actQs=64)，
// 计算类指令统一用 preg_VL64_16bit；preg_all_16bit 保留 ALL 仅用于存储。
// ============================================================================
template <MXQuantMode MX_QUANT_MODE = MXQuantMode::OCP, bool clear_gmax, typename T, typename T2,
          uint16_t QsBase = 128>
__simd_callee__ inline void softmax_align_qs64_kvs32_multi_chunk_vf(__ubuf__ T2 *p_dest, __ubuf__ T *s,
                                                  __ubuf__ T *local_group_max, __ubuf__ T *global_max,
                                                  __ubuf__ uint8_t *indexes_ub, const T NEG_LOG2_CX,
                                                  const uint16_t chunkY, const uint16_t effY, const bool hasChunk,
                                                  const uint16_t rows)
{
    // ====================== 寄存器定义 ======================
    RegTensor<half> src_c0, src_c1, src_c2, src_c3;
    RegTensor<half> src_n0, src_n1, src_n2, src_n3;
    RegTensor<half> curr_group_max;
    RegTensor<half> next_group_max;
    RegTensor<half> group_gmax;
    RegTensor<half> min_val_reg;
    RegTensor<uint8_t> idx_nd2nz;
    RegTensor<uint8_t> zero8;

    // 量化专用寄存器
    RegTensor<bfloat16_t> src_bf16_0, src_bf16_1, src_bf16_2, src_bf16_3;
    RegTensor<float4_e2m1x2_t> quant_0, quant_1, quant_2, quant_3;

    // ====================== 分块常量 ======================
    const uint16_t ROWS_PER_GROUP = 32;
    const uint16_t ROW_SUB_LOOP = 4;
    const uint16_t ITER_PER_GROUP = ROWS_PER_GROUP / ROW_SUB_LOOP;
    const uint16_t GROUPS_PER_CHUNK = 128 / ROWS_PER_GROUP;
    // blockEffRows(attenMask): 本 128 行大块的有效行数(hasChunk=false 时整块无有效行)。
    // 截断到真实行数 rows(尾块不满 128 行时, 超出的组走 padding 分支, 不读 s 越界)。
    const uint16_t effClamped = hasChunk ? (effY < rows ? effY : rows) : 0;
    const uint16_t validGroups = hasChunk ? (effClamped + ROWS_PER_GROUP - 1) / ROWS_PER_GROUP : 0;  // 有效组数向上取整
    const uint16_t groupBase = chunkY * GROUPS_PER_CHUNK; // 当前大块的全局组起始下标
    uint32_t MID_VALID_CNT = QsBase * (validGroups - 1);  // validGroups==0 时不会被使用

    // ====================== 掩码定义 ======================
    MaskReg preg_all_16bit = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_VL64_16bit = CreateMask<uint16_t, MaskPattern::VL64>();
    MaskReg preg_all_8bit = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg preg_vl128 = CreateMask<uint8_t, MaskPattern::VL128>();

    MaskReg preg_invalid_max;

    LoadAlign(group_gmax, global_max);

    LoadAlign(idx_nd2nz, indexes_ub);

    Duplicate(min_val_reg, MIN_VALUE);
    Duplicate(zero8, static_cast<uint8_t>(0));

    // ====================== 超出行置 MIN_VALUE（行级精确）======================
    // effClamped 非 32 倍数时，最后一个有效组内行 [effClamped%32, 32) 同样置 -inf：
    // 把本大块行 [effClamped, min((validGroups+1)*32, rows)) 写 MIN_VALUE，覆盖：
    //   1) 最后一个有效组(组 validGroups-1)的超出行 —— 组内计算读到 MIN_VALUE 后
    //      exp(MIN_VALUE - max) 下溢为 0（等价 -inf），且 max 统计不受影响；
    //   2) 组 validGroups 整组 —— 该组会被组循环最后一轮的 next 预取(next_group_max)
    //      读到，若为 MM1 真实数据会污染 group_gmax/global_max（跨 chunk online
    //      rescale 的 alpha），故同样置 MIN_VALUE。
    // 组 [validGroups+1, 4) 由 padding 分支处理（写 MIN_VALUE + pDest 填 0），不读 s，无需写。
    // validGroups==0 时整块无有效行：该循环不执行（避免 MIN_VALUE 污染预计算 -> scale 255/NaN）。
    const uint16_t writeEnd = ((validGroups + 1) * ROWS_PER_GROUP) < rows
                                  ? ((validGroups + 1) * ROWS_PER_GROUP)
                                  : rows;
    for (uint16_t r = effClamped; validGroups > 0 && r < writeEnd; ++r) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
            s + ((chunkY * 128 + r) * QsBase) * 2, min_val_reg, preg_VL64_16bit);
    }

    // ====================== 预计算：本大块第一个分组的最大值 ======================
    // 前提: 调用方保证进入本 VF 的所有大块 validEffRows>0（validGroups>=1），故预计算无条件执行。
    Duplicate(curr_group_max, MIN_VALUE);
    for (uint16_t iter = 0; iter < ITER_PER_GROUP; ++iter) {
        LoadAlign(src_c0, s + ((chunkY * 128 + iter * ROW_SUB_LOOP) * QsBase + 0 * QsBase) * 2);
        LoadAlign(src_c1, s + ((chunkY * 128 + iter * ROW_SUB_LOOP) * QsBase + 1 * QsBase) * 2);
        LoadAlign(src_c2, s + ((chunkY * 128 + iter * ROW_SUB_LOOP) * QsBase + 2 * QsBase) * 2);
        LoadAlign(src_c3, s + ((chunkY * 128 + iter * ROW_SUB_LOOP) * QsBase + 3 * QsBase) * 2);

        Max(src_c0, src_c0, src_c1, preg_VL64_16bit);
        Max(src_c2, src_c2, src_c3, preg_VL64_16bit);
        Max(curr_group_max, curr_group_max, src_c0, preg_VL64_16bit);
        Max(curr_group_max, curr_group_max, src_c2, preg_VL64_16bit);
    }

    Muls(curr_group_max, curr_group_max, INV_LN2, preg_VL64_16bit);

    if constexpr (MX_QUANT_MODE == MXQuantMode::OCP) {
        Truncate<T, RoundMode::CAST_FLOOR>(curr_group_max, curr_group_max, preg_VL64_16bit);
    } else {
        // CX: M'_(local,FP16) = ceil(Smax/ln2 - log2(CX))·ln2, 其中 -log2(CX) = NEG_LOG2_CX。
        Adds(curr_group_max, curr_group_max, NEG_LOG2_CX, preg_VL64_16bit);
        Truncate<T, RoundMode::CAST_CEIL>(curr_group_max, curr_group_max, preg_VL64_16bit);
    }

    Max(group_gmax, group_gmax, curr_group_max, preg_VL64_16bit);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max + groupBase * QsBase, curr_group_max,
                                                      preg_VL64_16bit);

    if constexpr (MX_QUANT_MODE == MXQuantMode::OCP) {
        Adds(curr_group_max, curr_group_max, NEG_TWO_VALE, preg_VL64_16bit);
    }
    Muls(curr_group_max, curr_group_max, LN2, preg_VL64_16bit);


    // ====================== 核心：双块流水分组循环 ======================
    for (uint16_t g = 0; g < validGroups; ++g) {
        MaskReg preg_valid_max = UpdateMask<half>(MID_VALID_CNT);
        MaskNot(preg_invalid_max, preg_valid_max, preg_all_16bit);

        // 初始化下一个块最大值
        Duplicate(next_group_max, MIN_VALUE);

        // ========== 第一个内循环：处理偶数子块 j = 0,2,4,6 ==========
        for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
            uint16_t rowOffset_next = (groupBase + g + 1) * ROWS_PER_GROUP + j * ROW_SUB_LOOP;
            LoadAlign(src_n0, s + (rowOffset_next * QsBase + 0 * QsBase) * 2);
            LoadAlign(src_n1, s + (rowOffset_next * QsBase + 1 * QsBase) * 2);
            LoadAlign(src_n2, s + (rowOffset_next * QsBase + 2 * QsBase) * 2);
            LoadAlign(src_n3, s + (rowOffset_next * QsBase + 3 * QsBase) * 2);
            Max(src_n0, src_n0, src_n1, preg_valid_max);
            Max(src_n2, src_n2, src_n3, preg_valid_max);
            Max(next_group_max, next_group_max, src_n0, preg_valid_max);
            Max(next_group_max, next_group_max, src_n2, preg_valid_max);

            // 当前块计算：Sub + Exp
            uint16_t rowOffset_cur = (groupBase + g) * ROWS_PER_GROUP + j * ROW_SUB_LOOP;
            LoadAlign(src_c0, s + (rowOffset_cur * QsBase + 0 * QsBase) * 2);
            LoadAlign(src_c1, s + (rowOffset_cur * QsBase + 1 * QsBase) * 2);
            LoadAlign(src_c2, s + (rowOffset_cur * QsBase + 2 * QsBase) * 2);
            LoadAlign(src_c3, s + (rowOffset_cur * QsBase + 3 * QsBase) * 2);

            Sub(src_c0, src_c0, curr_group_max, preg_VL64_16bit);
            Sub(src_c1, src_c1, curr_group_max, preg_VL64_16bit);
            Sub(src_c2, src_c2, curr_group_max, preg_VL64_16bit);
            Sub(src_c3, src_c3, curr_group_max, preg_VL64_16bit);

            Exp(src_c0, src_c0, preg_VL64_16bit);
            Exp(src_c1, src_c1, preg_VL64_16bit);
            Exp(src_c2, src_c2, preg_VL64_16bit);
            Exp(src_c3, src_c3, preg_VL64_16bit);

            // 4bit 量化压缩
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_0, src_c0, preg_VL64_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_1, src_c1, preg_VL64_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_2, src_c2, preg_VL64_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_3, src_c3, preg_VL64_16bit);

            Cast<float4_e2m1x2_t, bfloat16_t, castTraitZero>(quant_0, src_bf16_0, preg_VL64_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitOne>(quant_1, src_bf16_1, preg_VL64_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitTwo>(quant_2, src_bf16_2, preg_VL64_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitThree>(quant_3, src_bf16_3, preg_VL64_16bit);

            // 数据合并
            Or((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_1,
               preg_all_8bit);
            Or((RegTensor<uint8_t> &)quant_2, (RegTensor<uint8_t> &)quant_2, (RegTensor<uint8_t> &)quant_3,
               preg_all_8bit);
            Or((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_2,
               preg_all_8bit);
            Gather((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, idx_nd2nz);

            // 写入内存（偶数块：无+128偏移）
            StoreAlign(((__ubuf__ uint8_t *&)p_dest) + (groupBase + g) * 2048 + j * 256,
                       (RegTensor<uint8_t> &)quant_0,
                       preg_vl128  // 低128字节掩码
            );
        }

        // ========== 第二个内循环：处理奇数子块 j+1 = 1,3,5,7 ==========
        for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
            // 预计算：下一个块 (g+1) 最大值
            uint16_t rowOffset_next = (groupBase + g + 1) * ROWS_PER_GROUP + (j + 1) * ROW_SUB_LOOP;
            LoadAlign(src_n0, s + (rowOffset_next * QsBase + 0 * QsBase) * 2);
            LoadAlign(src_n1, s + (rowOffset_next * QsBase + 1 * QsBase) * 2);
            LoadAlign(src_n2, s + (rowOffset_next * QsBase + 2 * QsBase) * 2);
            LoadAlign(src_n3, s + (rowOffset_next * QsBase + 3 * QsBase) * 2);

            Max(src_n0, src_n0, src_n1, preg_valid_max);
            Max(src_n2, src_n2, src_n3, preg_valid_max);
            Max(next_group_max, next_group_max, src_n0, preg_valid_max);
            Max(next_group_max, next_group_max, src_n2, preg_valid_max);

            // 当前块计算：Sub + Exp
            uint16_t rowOffset_cur = (groupBase + g) * ROWS_PER_GROUP + (j + 1) * ROW_SUB_LOOP;
            LoadAlign(src_c0, s + (rowOffset_cur * QsBase + 0 * QsBase) * 2);
            LoadAlign(src_c1, s + (rowOffset_cur * QsBase + 1 * QsBase) * 2);
            LoadAlign(src_c2, s + (rowOffset_cur * QsBase + 2 * QsBase) * 2);
            LoadAlign(src_c3, s + (rowOffset_cur * QsBase + 3 * QsBase) * 2);

            Sub(src_c0, src_c0, curr_group_max, preg_VL64_16bit);
            Sub(src_c1, src_c1, curr_group_max, preg_VL64_16bit);
            Sub(src_c2, src_c2, curr_group_max, preg_VL64_16bit);
            Sub(src_c3, src_c3, curr_group_max, preg_VL64_16bit);

            Exp(src_c0, src_c0, preg_VL64_16bit);
            Exp(src_c1, src_c1, preg_VL64_16bit);
            Exp(src_c2, src_c2, preg_VL64_16bit);
            Exp(src_c3, src_c3, preg_VL64_16bit);

            // 4bit 量化压缩
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_0, src_c0, preg_VL64_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_1, src_c1, preg_VL64_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_2, src_c2, preg_VL64_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_3, src_c3, preg_VL64_16bit);

            Cast<float4_e2m1x2_t, bfloat16_t, castTraitZero>(quant_0, src_bf16_0, preg_VL64_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitOne>(quant_1, src_bf16_1, preg_VL64_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitTwo>(quant_2, src_bf16_2, preg_VL64_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitThree>(quant_3, src_bf16_3, preg_VL64_16bit);

            // 数据合并
            Or((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_1,
               preg_all_8bit);
            Or((RegTensor<uint8_t> &)quant_2, (RegTensor<uint8_t> &)quant_2, (RegTensor<uint8_t> &)quant_3,
               preg_all_8bit);
            Or((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_2,
               preg_all_8bit);
            Gather((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, idx_nd2nz);

            // 奇数块：低128字节写入（基础地址 +128 偏移）
            StoreAlign(((__ubuf__ uint8_t *&)p_dest) + (groupBase + g) * 2048 + j * 256 + 128,
                       (RegTensor<uint8_t> &)quant_0,
                       preg_vl128);
        }

        // ====================== 全局/局部最大值更新 ======================
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(global_max, group_gmax, preg_invalid_max);

        // 下一块最大值归一化
        Muls(next_group_max, next_group_max, INV_LN2, preg_valid_max);
        if constexpr (MX_QUANT_MODE == MXQuantMode::OCP) {
            Truncate<T, RoundMode::CAST_FLOOR>(next_group_max, next_group_max, preg_valid_max);
        } else {
            // CX: 与当前块一致, ceil(Smax/ln2 - log2(CX))
            Adds(next_group_max, next_group_max, NEG_LOG2_CX, preg_valid_max);
            Truncate<T, RoundMode::CAST_CEIL>(next_group_max, next_group_max, preg_valid_max);
        }
        Max(group_gmax, group_gmax, next_group_max, preg_valid_max);

        // 存储下一块最大值到 ulmax (groupBase+g+1)*128 偏移
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max + ((groupBase + g + 1) * QsBase),
                                                          next_group_max, preg_valid_max);

        // 更新当前块最大值，用于下一次循环
        if constexpr (MX_QUANT_MODE == MXQuantMode::OCP) {
            Adds(next_group_max, next_group_max, NEG_TWO_VALE, preg_all_16bit);
        }
        Muls(curr_group_max, next_group_max, LN2, preg_valid_max);
    }

    // ====================== padding 组：直接更新 max + pDest 填 0 ======================
    for (uint16_t g = validGroups; g < GROUPS_PER_CHUNK; ++g) {
        // 写 MIN_VALUE，使下游 computePscale 得到 scale=0
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max + (groupBase + g) * QsBase, min_val_reg,
                                                          preg_all_16bit);

        // 偶数子块清零
        for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
            StoreAlign(((__ubuf__ uint8_t *&)p_dest) + (groupBase + g) * 2048 + j * 256, zero8, preg_vl128);
        }

        // 奇数子块清零
        for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
            StoreAlign(((__ubuf__ uint8_t *&)p_dest) + (groupBase + g) * 2048 + j * 256 + 128, zero8, preg_vl128);
        }
    }
}


// ============================================================================
// 大循环：actKvsTile 行（33~255），拆成 chunk0（行 [0,128)）+ chunk1（行 [128,256)）两段下发。
// hasChunk1 = actKvsTile > 128（无第二个 128 行块时 chunk1 整块按 padding 处理）。
// 与 kvs256 的 Y1/Y2 双 chunk 结构完全一致。
// ============================================================================
template <MXQuantMode MX_QUANT_MODE = MXQuantMode::OCP, bool clear_gmax, typename T, typename T2,
          uint16_t QsBase = 128>
__simd_vf__ inline void softmax_with_group_max_align_qs64_kvs32_multi_vf(__ubuf__ T2 *p_dest, __ubuf__ T *s,
                                                  __ubuf__ T *local_group_max, __ubuf__ T *global_max,
                                                  __ubuf__ uint8_t *indexes_ub, const T NEG_LOG2_CX,
                                                  uint16_t KvsBaseAlign32, uint16_t KvsBaseAlign64,
                                                  const uint16_t effY1, const uint16_t effY2, const uint16_t actKvsTile)
{
    const bool hasChunk1 = actKvsTile > 128;
    const uint16_t rows0 = actKvsTile < 128 ? actKvsTile : 128;
    const uint16_t rows1 = hasChunk1 ? (actKvsTile - 128) : 0;

    // ====================== chunk 0：第 0 ~ 127 行 ======================
    softmax_align_qs64_kvs32_multi_chunk_vf<MX_QUANT_MODE, clear_gmax, T, T2, QsBase>(
        p_dest, s, local_group_max, global_max, indexes_ub, NEG_LOG2_CX, 0, effY1, true, rows0);

    // ====================== chunk 1：第 128 ~ 255 行 ======================
    softmax_align_qs64_kvs32_multi_chunk_vf<MX_QUANT_MODE, clear_gmax, T, T2, QsBase>(
        p_dest, s, local_group_max, global_max, indexes_ub, NEG_LOG2_CX, 1, effY2, hasChunk1, rows1);

    // padding group_max 到 KvsBaseAlign64 (multi -inf), 使对应的 pscale = 0
    // chunk 内已覆盖组 [0, 8)（两个 128 行块），这里补 [8, KvsBaseAlign64/32) 的组。
    (void)KvsBaseAlign32;
    MaskReg preg_all_16bit = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<half> min_val_reg;
    Duplicate(min_val_reg, MIN_VALUE);
    const uint16_t PAD_START = 8;  // 两个 128 行 chunk 各 4 组
    const uint16_t PAD_END = KvsBaseAlign64 / 32;
    for (uint16_t i = PAD_START; i < PAD_END; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max + i * QsBase, min_val_reg,
                                                          preg_all_16bit);
    }
}

template <MXQuantMode MX_QUANT_MODE = MXQuantMode::OCP, bool clear_gmax, typename T, typename T2,
          uint16_t QsBase = 128>
__aicore__ inline void SoftmaxWithGroupMaxAlignQs64Kvs32MultiCallVF(const LocalTensor<T2> &dstTensor,
                                                  const LocalTensor<T> &srcTensor,
                                                  const LocalTensor<T> &localGroupMax,
                                                  const LocalTensor<T> &globalMax,
                                                  const LocalTensor<uint8_t> &indexesBuf,
                                                  const T NEG_LOG2_CX,
                                                  uint16_t KvsBaseAlign32, uint16_t KvsBaseAlign64,
                                                  const uint16_t effY1, const uint16_t effY2, const uint16_t actKvsTile)
{
    __ubuf__ T2 *p_dest = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ T *s_ub = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ T *local_group_max = (__ubuf__ T*)localGroupMax.GetPhyAddr();
    __ubuf__ T *global_max = (__ubuf__ T*)globalMax.GetPhyAddr();
    __ubuf__ uint8_t * indexes_ub = (__ubuf__ uint8_t*)indexesBuf.GetPhyAddr();

    softmax_with_group_max_align_qs64_kvs32_multi_vf<MX_QUANT_MODE, clear_gmax, T, T2, QsBase>(
        p_dest, s_ub, local_group_max, global_max, indexes_ub, NEG_LOG2_CX,
        KvsBaseAlign32, KvsBaseAlign64, effY1, effY2, actKvsTile);
}

}
#endif  // VF_SOFTMAX_DN_CAST_NZ_MXFP4_ALIGN_QS64_KVS32_MULTI_H_
