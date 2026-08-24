/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ffn_layout.h
 * \brief FFN 权重布局识别的唯一公共实现（infershape / aclnn / tiling / op_api / binding 共用）。
 *        canonical [K,N]（arch22 原生）与 linear [N,K]（PyTorch Linear (out,in)）。
 */

#ifndef OPS_TRANSFORMER_FFN_LAYOUT_H_
#define OPS_TRANSFORMER_FFN_LAYOUT_H_

#include <cstdint>

namespace ffnlayout {

enum class FfnLayout {
    LINEAR,    // w1 [H,K] / swiglu [2H,K]，w2 [N,H]（PyTorch Linear）
    CANONICAL, // w1 [K,H] / swiglu [K,2H]，w2 [H,N]（arch22 原生）
    INVALID    // w1 与 xK 完全不匹配
};

// 布局识别（入参一律传 weight 的最后两维；isSwiglu 由调用方按 ActiveType 枚举算好传入）：
//  1) w1 非方阵：w1d1==xK → LINEAR；w1d0==xK → CANONICAL；
//  2) w1 方阵（形状不可区分）：用 w2 消歧——swiglu 的隐藏宽是 w1 边长的一半，
//     其余激活隐藏宽=w1 边长；
//  3) 全方阵（H==K==N）或消歧失败：真歧义，与 arch22 一致默认 CANONICAL。
inline FfnLayout FfnDetectLayout(int64_t w1d0, int64_t w1d1, int64_t w2d0, int64_t w2d1,
                                 int64_t xK, bool isSwiglu)
{
    const int64_t hiddenW = isSwiglu ? w1d0 / 2 : w1d0;
    const int64_t hiddenW2 = isSwiglu ? w1d1 / 2 : w1d1;
    if (w1d1 == xK && w1d0 != xK) {
        return FfnLayout::LINEAR;
    }
    if (w1d0 == xK && w1d1 != xK) {
        return FfnLayout::CANONICAL;
    }
    if (w1d0 == xK && w1d1 == xK) {
        if (w2d1 == hiddenW && w2d0 != hiddenW2) {
            return FfnLayout::LINEAR;
        }
        if (w2d0 == hiddenW2 && w2d1 != hiddenW) {
            return FfnLayout::CANONICAL;
        }
        return FfnLayout::CANONICAL; // 全方阵真歧义：默认 canonical
    }
    return FfnLayout::INVALID;
}

} // namespace ffnlayout

#endif // OPS_TRANSFORMER_FFN_LAYOUT_H_
