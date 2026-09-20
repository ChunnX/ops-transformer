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
__simd_vf__ inline void VfDivAndCast(__ubuf__ bfloat16_t *dst, __ubuf__ float *src,
                                     __ubuf__ float *rowsum)
{
    RegTensor<float> vreg_rowsum_a;
    RegTensor<float> vreg_rowsum_b;
    RegTensor<float> vreg_src_0, vreg_src_1, vreg_src_2, vreg_src_3;
    RegTensor<bfloat16_t> vreg_cast_0, vreg_cast_1, vreg_cast_2, vreg_cast_3;
    MaskReg preg_f32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_bf16 = CreateMask<bfloat16_t, MaskPattern::ALL>();

    // ① rowsum 载入（2 regs，全程复用）
    LoadAlign(vreg_rowsum_a, rowsum);
    LoadAlign(vreg_rowsum_b, rowsum + 64);

    // ② 128 行 × 2 段（前 64 列 rowsumA / 后 64 列 rowsumB），4 路行展开
    //    src 和 dst 的行偏移数值相同（均为 128 元素/行），指针类型自动处理字节宽度差异
    for (uint32_t row = 0; row < 128; row += 4) {
        uint32_t base0 = row * 128;
        uint32_t base1 = (row + 1) * 128;
        uint32_t base2 = (row + 2) * 128;
        uint32_t base3 = (row + 3) * 128;

        // ===== 前 64 列（rowsumA）=====
        LoadAlign(vreg_src_0, src + base0);
        LoadAlign(vreg_src_1, src + base1);
        LoadAlign(vreg_src_2, src + base2);
        LoadAlign(vreg_src_3, src + base3);
        Div(vreg_src_0, vreg_src_0, vreg_rowsum_a, preg_f32);
        Div(vreg_src_1, vreg_src_1, vreg_rowsum_a, preg_f32);
        Div(vreg_src_2, vreg_src_2, vreg_rowsum_a, preg_f32);
        Div(vreg_src_3, vreg_src_3, vreg_rowsum_a, preg_f32);
        Cast<bfloat16_t, float, castTraitZero>(vreg_cast_0, vreg_src_0, preg_f32);
        Cast<bfloat16_t, float, castTraitZero>(vreg_cast_1, vreg_src_1, preg_f32);
        Cast<bfloat16_t, float, castTraitZero>(vreg_cast_2, vreg_src_2, preg_f32);
        Cast<bfloat16_t, float, castTraitZero>(vreg_cast_3, vreg_src_3, preg_f32);
        StoreAlign(dst + base0, vreg_cast_0, preg_bf16);
        StoreAlign(dst + base1, vreg_cast_1, preg_bf16);
        StoreAlign(dst + base2, vreg_cast_2, preg_bf16);
        StoreAlign(dst + base3, vreg_cast_3, preg_bf16);

        // ===== 后 64 列（rowsumB）=====
        LoadAlign(vreg_src_0, src + base0 + 64);
        LoadAlign(vreg_src_1, src + base1 + 64);
        LoadAlign(vreg_src_2, src + base2 + 64);
        LoadAlign(vreg_src_3, src + base3 + 64);
        Div(vreg_src_0, vreg_src_0, vreg_rowsum_b, preg_f32);
        Div(vreg_src_1, vreg_src_1, vreg_rowsum_b, preg_f32);
        Div(vreg_src_2, vreg_src_2, vreg_rowsum_b, preg_f32);
        Div(vreg_src_3, vreg_src_3, vreg_rowsum_b, preg_f32);
        Cast<bfloat16_t, float, castTraitZero>(vreg_cast_0, vreg_src_0, preg_f32);
        Cast<bfloat16_t, float, castTraitZero>(vreg_cast_1, vreg_src_1, preg_f32);
        Cast<bfloat16_t, float, castTraitZero>(vreg_cast_2, vreg_src_2, preg_f32);
        Cast<bfloat16_t, float, castTraitZero>(vreg_cast_3, vreg_src_3, preg_f32);
        StoreAlign(dst + base0 + 64, vreg_cast_0, preg_bf16);
        StoreAlign(dst + base1 + 64, vreg_cast_1, preg_bf16);
        StoreAlign(dst + base2 + 64, vreg_cast_2, preg_bf16);
        StoreAlign(dst + base3 + 64, vreg_cast_3, preg_bf16);
    }
}

} // namespace QfaVectorApi
} // namespace QFA_KERNEL

#endif // VF_DIV_CAST_BF16_H_
