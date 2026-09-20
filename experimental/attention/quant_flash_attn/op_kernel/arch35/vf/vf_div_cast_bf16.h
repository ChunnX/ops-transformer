/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, and you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You should have received a copy of the License along with this program.
 * If not, see <https://www.hiascend.com/legal/cannlicense>.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
 * PURPOSE.
 */

/*!
 * \file vf_div_cast_bf16.h
 * \brief V2 末块：逐元素除 rowsum + cast bf16，[DV,S1] 段加载
 */

#ifndef VF_DIV_CAST_BF16_H_
#define VF_DIV_CAST_BF16_H_

#include "kernel_tensor.h"
#include "vf_common_def.h"

namespace QFA_KERNEL {
namespace QfaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

/*
 * @ingroup VfaVectorApi
 * @brief V2 末块：dst = bf16(src / rowsum)，[DV,S1] 段加载
 *
 * rowsum 沿 S1 列方向逐元素（[DV,S1] 行主序，与数据行形状对齐，无需广播）。
 * rowsum 分 2 regs（前 64 / 后 64），载入一次全程复用。
 * fp32→bf16 是 2:1 窄化，但源只有 64 元素/寄存器 → 单 Cast 即可（无需双 Cast+Or，
 * 区别于 half→uint8 的 128 元素需双 Cast）。
 *
 * @param [out] dst     divOut（[DV,S1] bf16）
 * @param [in]  src     stage2OutUB（[DV,S1] fp32，V2 末块累积分子）
 * @param [in]  rowsum  accRowsumUB（[128] fp32，per S1，累积分母）
 */
__simd_vf__ inline void VfDivAndCastVF(__ubuf__ bfloat16_t *dstT, __ubuf__ float *src,
                                         __ubuf__ float *rowsum)
{
    // ===== 融合转置写（设计 §7.5，DN processOutVF 镜像）=====
    // src [DV,S1] fp32 → dstT 转置带 pad [S1,DV] bf16（mm2Res 别名区）
    RegTensor<float> vreg_src_1, vreg_src_2, vreg_src_3, vreg_src_4;      // 16 寄存器块
    RegTensor<float> vreg_src_5, vreg_src_6, vreg_src_7, vreg_src_8;      // = 16 行 × 64 列
    RegTensor<float> vreg_src_9, vreg_src_10, vreg_src_11, vreg_src_12;   // （倒数乘后复用为
    RegTensor<float> vreg_src_13, vreg_src_14, vreg_src_15, vreg_src_16;  //   Cast/转置载体）
    RegTensor<float> vreg_recip;   // 当前一半列的 1/rowsum（先算倒数再乘，替代逐行 Div）
    RegTensor<float> vreg_ones;    // 全 1（倒数分子）
    MaskReg preg_f32 = CreateMask<float, MaskPattern::ALL>();   // 64 lanes fp32

    Duplicate(vreg_ones, 1.0f);

    // ===== 按列分成前后两半处理（每半 64 列；DN 由调用方调用两次，我们单函数内循环）=====
    for (uint32_t colHalf = 0; colHalf < 2; colHalf++) {
        // 倒数：recip = 1/rowsum[本半的 64 个分母]（DN L36-38 同款；每半算一次，各行复用）
        LoadAlign(vreg_recip, rowsum + colHalf * 64);
        Div(vreg_recip, vreg_ones, vreg_recip, preg_f32);

        // ===== 按行分成 8 块（每块 16 行，覆盖 128 个头维度行）=====
        for (uint32_t block = 0; block < 8; block++) {
            uint32_t blockBase = block * 16 * 128 + colHalf * 64;  // 行距 128（DN 为 64）+ 半区列偏移
            // 装载 16 行 × 本半 64 列（行距 128，DN 为 64——两处刻意差异之一），乘倒数
            LoadAlign(vreg_src_1, src + blockBase + 0 * 128);
            LoadAlign(vreg_src_2, src + blockBase + 1 * 128);
            LoadAlign(vreg_src_3, src + blockBase + 2 * 128);
            LoadAlign(vreg_src_4, src + blockBase + 3 * 128);
            LoadAlign(vreg_src_5, src + blockBase + 4 * 128);
            LoadAlign(vreg_src_6, src + blockBase + 5 * 128);
            LoadAlign(vreg_src_7, src + blockBase + 6 * 128);
            LoadAlign(vreg_src_8, src + blockBase + 7 * 128);
            LoadAlign(vreg_src_9, src + blockBase + 8 * 128);
            LoadAlign(vreg_src_10, src + blockBase + 9 * 128);
            LoadAlign(vreg_src_11, src + blockBase + 10 * 128);
            LoadAlign(vreg_src_12, src + blockBase + 11 * 128);
            LoadAlign(vreg_src_13, src + blockBase + 12 * 128);
            LoadAlign(vreg_src_14, src + blockBase + 13 * 128);
            LoadAlign(vreg_src_15, src + blockBase + 14 * 128);
            LoadAlign(vreg_src_16, src + blockBase + 15 * 128);
            Mul(vreg_src_1, vreg_src_1, vreg_recip, preg_f32);
            Mul(vreg_src_2, vreg_src_2, vreg_recip, preg_f32);
            Mul(vreg_src_3, vreg_src_3, vreg_recip, preg_f32);
            Mul(vreg_src_4, vreg_src_4, vreg_recip, preg_f32);
            Mul(vreg_src_5, vreg_src_5, vreg_recip, preg_f32);
            Mul(vreg_src_6, vreg_src_6, vreg_recip, preg_f32);
            Mul(vreg_src_7, vreg_src_7, vreg_recip, preg_f32);
            Mul(vreg_src_8, vreg_src_8, vreg_recip, preg_f32);
            Mul(vreg_src_9, vreg_src_9, vreg_recip, preg_f32);
            Mul(vreg_src_10, vreg_src_10, vreg_recip, preg_f32);
            Mul(vreg_src_11, vreg_src_11, vreg_recip, preg_f32);
            Mul(vreg_src_12, vreg_src_12, vreg_recip, preg_f32);
            Mul(vreg_src_13, vreg_src_13, vreg_recip, preg_f32);
            Mul(vreg_src_14, vreg_src_14, vreg_recip, preg_f32);
            Mul(vreg_src_15, vreg_src_15, vreg_recip, preg_f32);
            Mul(vreg_src_16, vreg_src_16, vreg_recip, preg_f32);
            // TODO(4.1c-2): 8 对跨行 Cast+Or(uint16) 造垂直对
            // TODO(4.1c-3): 12 条 Interleave 级联 + 8 条 StoreAlign(8+1) 散布
            (void)dstT;
        }
    }
}

} // namespace QfaVectorApi
} // namespace QFA_KERNEL

#endif // VF_DIV_CAST_BF16_H_
