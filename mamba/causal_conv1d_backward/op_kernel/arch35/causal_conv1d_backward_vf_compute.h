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
 * \file causal_conv1d_backward_vf_compute.h
 * \brief RegBase VF body —— CausalConv1dBackward（按 ascendc-regbase-best-practice 规范：
 *        `__simd_vf__` 入口 + `AscendC::Reg::*` 全限定 + 外层 `asc_vf_call` 调用；
 *        遵守 VF 硬约束：循环变量 uint16_t / VF 内禁止数组与运行时 if
 *        （K 以模板参数传入，tap 用 if constexpr 展开）。
 *
 * 分层：B16→fp32 扩位由 UB 层 membase Cast 完成（dav_3510 的 Reg::Cast 不支持
 * fp32→B16 下行），VF body 只消费 fp32 平面（行紧排 D0，每平面尾部 64-lane
 * slack 承接分组装载的尾组越界读）。
 *
 * 数据流（UB 域，行距 = D0，由调度层准备）：
 *   xF32   : [(hist + rows) × D0] float —— 段内 x（行 [0, hist) 为冷启动 pad 0）
 *   gyF32  : [rows × D0]            float —— grad_y
 *   wF32   : [K × D0]               float —— weight（weight[i] 配 x[t-(K-1-i)]）
 *   gradZ  : [(hist + rows + K) × D0] float —— grad_z（[0,hist)=tail 交接区，
 *            [hist,...)=当前区；[hist+rows, hist+rows+K-1)=段末扩展行，由外层清零）
 *   dwAcc  : [K × D0]               float —— grad_weight 跨 chunk 累加器
 *   dxF32  : [rows × D0]            float —— grad_x（CopyOut 前由 UB 层 cast 回 T）
 *
 * GradZVf（PassA，融合链）：d0 分组循环内逐 token 行——
 *   x 窗口行自 UB fp32 平面装载（行 r+i）→ z = Σ_i w[i]·x[t-(K-1-i)]
 *   （Mul + MulAddDst 融合）→ σ/silu' 全寄存器完成（z 不落 UB）
 *   → grad_z 存 UB（跨 chunk tail 交接 + PassB 消费）
 *   → grad_weight 部分和寄存器内按行累加（初值取自 dwAcc，chunk 末回存，无乘积平面/树归并）
 * GradXVf（PassB）：dx[s] = Σ_i w[i]·grad_z[s+i]（Mul + MulAddDst 链；段末扩展行
 *   已由外层清零，全 K tap 无分支）
 * CopyGzTailVf    ：gradZ 尾部 hist 行搬到行 [0, hist)
 */

#ifndef CAUSAL_CONV1D_BACKWARD_VF_COMPUTE_H
#define CAUSAL_CONV1D_BACKWARD_VF_COMPUTE_H

#include "kernel_operator.h"

namespace NsCausalConv1dBackwardVF {

constexpr uint32_t V_LENGTH = 64;  // 256B / sizeof(float)

// PassA：z 重算 + silu'(z)（kAct=true）/ 直通（kAct=false）+ grad_z + grad_weight 寄存器累加
template <typename DType, int32_t K, bool kAct>
__simd_vf__ inline void ConvBwdGradZVfImpl(__ubuf__ float *xF32, __ubuf__ float *gyF32,
                                           __ubuf__ float *wF32, __ubuf__ float *gradZ,
                                           __ubuf__ float *dwAcc, uint16_t rows,
                                           uint16_t groupCnt, uint32_t d0, int32_t hist)
{
    AscendC::Reg::RegTensor<float> w0;
    AscendC::Reg::RegTensor<float> w1;
    AscendC::Reg::RegTensor<float> w2;
    AscendC::Reg::RegTensor<float> w3;
    AscendC::Reg::RegTensor<float> dw0;
    AscendC::Reg::RegTensor<float> dw1;
    AscendC::Reg::RegTensor<float> dw2;
    AscendC::Reg::RegTensor<float> dw3;
    AscendC::Reg::RegTensor<float> x0;
    AscendC::Reg::RegTensor<float> x1;
    AscendC::Reg::RegTensor<float> x2;
    AscendC::Reg::RegTensor<float> x3;
    AscendC::Reg::RegTensor<float> prod0;
    AscendC::Reg::RegTensor<float> prod1;
    AscendC::Reg::RegTensor<float> prod2;
    AscendC::Reg::RegTensor<float> prod3;

    for (uint16_t g = 0; g < groupCnt; ++g) {
        uint32_t off = static_cast<uint32_t>(g) * V_LENGTH;
        uint32_t valid = d0 - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(valid);

        AscendC::Reg::LoadAlign(w0, wF32 + off);
        AscendC::Reg::LoadAlign(dw0, dwAcc + off);
        if constexpr (K >= 2) {
            AscendC::Reg::LoadAlign(w1, wF32 + d0 + off);
            AscendC::Reg::LoadAlign(dw1, dwAcc + d0 + off);
        }
        if constexpr (K >= 3) {
            AscendC::Reg::LoadAlign(w2, wF32 + 2 * d0 + off);
            AscendC::Reg::LoadAlign(dw2, dwAcc + 2 * d0 + off);
        }
        if constexpr (K >= 4) {
            AscendC::Reg::LoadAlign(w3, wF32 + 3 * d0 + off);
            AscendC::Reg::LoadAlign(dw3, dwAcc + 3 * d0 + off);
        }

        for (uint16_t r = 0; r < rows; ++r) {
            // x 窗口行：x[t-(K-1-i)] = 行 (hist + r - (K-1-i)) = 行 (r + i)
            // （kAct=false 时 z 不重算，但 dw 累加仍需 x 窗口）
            AscendC::Reg::LoadAlign(x0, xF32 + r * d0 + off);
            if constexpr (K >= 2) {
                AscendC::Reg::LoadAlign(x1, xF32 + (r + 1) * d0 + off);
            }
            if constexpr (K >= 3) {
                AscendC::Reg::LoadAlign(x2, xF32 + (r + 2) * d0 + off);
            }
            if constexpr (K >= 4) {
                AscendC::Reg::LoadAlign(x3, xF32 + (r + 3) * d0 + off);
            }

            // grad_y 行
            AscendC::Reg::RegTensor<float> gy;
            AscendC::Reg::LoadAlign(gy, gyF32 + r * d0 + off);

            // grad_z：kAct=true 时 = gy·silu'(z)（z 全寄存器完成）；false 时 = gy
            AscendC::Reg::RegTensor<float> gz;
            if constexpr (kAct) {
                // z = Σ_i w[i]·x[t-(K-1-i)]
                AscendC::Reg::RegTensor<float> z;
                AscendC::Reg::Mul(z, x0, w0, mask);
                if constexpr (K >= 2) {
                    AscendC::Reg::MulAddDst(z, x1, w1, mask);
                }
                if constexpr (K >= 3) {
                    AscendC::Reg::MulAddDst(z, x2, w2, mask);
                }
                if constexpr (K >= 4) {
                    AscendC::Reg::MulAddDst(z, x3, w3, mask);
                }

                // σ(z) = 1/(1+exp(-z))
                AscendC::Reg::RegTensor<float> neg;
                AscendC::Reg::Muls(neg, z, -1.0f, mask);
                AscendC::Reg::Exp(neg, neg, mask);
                AscendC::Reg::Adds(neg, neg, 1.0f, mask);
                AscendC::Reg::RegTensor<float> sig;
                AscendC::Reg::Duplicate(sig, 1.0f);
                AscendC::Reg::Div(sig, sig, neg, mask);

                // silu'(z) = σ·(1 + z·(1-σ))
                AscendC::Reg::RegTensor<float> siluDeriv;
                AscendC::Reg::Muls(siluDeriv, sig, -1.0f, mask);
                AscendC::Reg::Adds(siluDeriv, siluDeriv, 1.0f, mask);
                AscendC::Reg::Mul(siluDeriv, siluDeriv, z, mask);
                AscendC::Reg::Adds(siluDeriv, siluDeriv, 1.0f, mask);
                AscendC::Reg::Mul(siluDeriv, sig, siluDeriv, mask);

                // grad_z = gy·silu'(z)
                AscendC::Reg::Mul(gz, gy, siluDeriv, mask);
            } else {
                // activation=none：Act' ≡ 1，grad_z = gy（x 装载被 if constexpr 裁剪）
                gz = gy;
            }
            AscendC::Reg::StoreAlign(gradZ + (static_cast<uint32_t>(hist) + r) * d0 + off, gz, mask);

            // grad_weight 部分和：dw[i] += x[t-(K-1-i)]·grad_z（寄存器内按行累加）
            AscendC::Reg::Mul(prod0, x0, gz, mask);
            AscendC::Reg::Add(dw0, dw0, prod0, mask);
            if constexpr (K >= 2) {
                AscendC::Reg::Mul(prod1, x1, gz, mask);
                AscendC::Reg::Add(dw1, dw1, prod1, mask);
            }
            if constexpr (K >= 3) {
                AscendC::Reg::Mul(prod2, x2, gz, mask);
                AscendC::Reg::Add(dw2, dw2, prod2, mask);
            }
            if constexpr (K >= 4) {
                AscendC::Reg::Mul(prod3, x3, gz, mask);
                AscendC::Reg::Add(dw3, dw3, prod3, mask);
            }
        }

        // chunk 部分和回存跨 chunk 累加器
        AscendC::Reg::StoreAlign(dwAcc + off, dw0, mask);
        if constexpr (K >= 2) {
            AscendC::Reg::StoreAlign(dwAcc + d0 + off, dw1, mask);
        }
        if constexpr (K >= 3) {
            AscendC::Reg::StoreAlign(dwAcc + 2 * d0 + off, dw2, mask);
        }
        if constexpr (K >= 4) {
            AscendC::Reg::StoreAlign(dwAcc + 3 * d0 + off, dw3, mask);
        }
    }
}

// PassB：dx[s] = Σ_i w[i]·grad_z[s+i]（段末扩展行已由外层清零，全 K tap 无分支）
template <typename DType, int32_t K>
__simd_vf__ inline void ConvBwdGradXVfImpl(__ubuf__ float *gradZ, __ubuf__ float *wF32,
                                           __ubuf__ float *dxF32, int32_t srcBase,
                                           uint16_t rows, uint16_t groupCnt, uint32_t d0)
{
    AscendC::Reg::RegTensor<float> w0;
    AscendC::Reg::RegTensor<float> w1;
    AscendC::Reg::RegTensor<float> w2;
    AscendC::Reg::RegTensor<float> w3;
    AscendC::Reg::RegTensor<float> gz0;
    AscendC::Reg::RegTensor<float> gz1;
    AscendC::Reg::RegTensor<float> gz2;
    AscendC::Reg::RegTensor<float> gz3;
    AscendC::Reg::RegTensor<float> acc;

    for (uint16_t g = 0; g < groupCnt; ++g) {
        uint32_t off = static_cast<uint32_t>(g) * V_LENGTH;
        uint32_t valid = d0 - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(valid);

        AscendC::Reg::LoadAlign(w0, wF32 + (K - 1) * d0 + off);
        if constexpr (K >= 2) {
            AscendC::Reg::LoadAlign(w1, wF32 + (K - 2) * d0 + off);
        }
        if constexpr (K >= 3) {
            AscendC::Reg::LoadAlign(w2, wF32 + (K - 3) * d0 + off);
        }
        if constexpr (K >= 4) {
            AscendC::Reg::LoadAlign(w3, wF32 + 0 * d0 + off);
        }

        for (uint16_t r = 0; r < rows; ++r) {
            uint32_t base = (static_cast<uint32_t>(srcBase) + r) * d0 + off;
            AscendC::Reg::LoadAlign(gz0, gradZ + base);
            AscendC::Reg::Mul(acc, gz0, w0, mask);
            if constexpr (K >= 2) {
                AscendC::Reg::LoadAlign(gz1, gradZ + base + d0);
                AscendC::Reg::MulAddDst(acc, gz1, w1, mask);
            }
            if constexpr (K >= 3) {
                AscendC::Reg::LoadAlign(gz2, gradZ + base + 2 * d0);
                AscendC::Reg::MulAddDst(acc, gz2, w2, mask);
            }
            if constexpr (K >= 4) {
                AscendC::Reg::LoadAlign(gz3, gradZ + base + 3 * d0);
                AscendC::Reg::MulAddDst(acc, gz3, w3, mask);
            }
            AscendC::Reg::StoreAlign(dxF32 + r * d0 + off, acc, mask);
        }
    }
}

// Tail：gradZ 尾部 hist 行搬到行 [0, hist)（fp32 行拷贝，供下一 chunk 补算读取）
__simd_vf__ inline void ConvBwdCopyGzTailVfImpl(__ubuf__ float *gradZ, uint16_t rows,
                                                uint16_t hist, uint32_t d0,
                                                uint16_t groupCnt)
{
    AscendC::Reg::RegTensor<float> v;
    for (uint16_t g = 0; g < groupCnt; ++g) {
        uint32_t off = static_cast<uint32_t>(g) * V_LENGTH;
        uint32_t valid = d0 - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(valid);
        for (uint16_t r = 0; r < hist; ++r) {
            AscendC::Reg::LoadAlign(v, gradZ + (rows + r) * d0 + off);
            AscendC::Reg::StoreAlign(gradZ + r * d0 + off, v, mask);
        }
    }
}

}  // namespace NsCausalConv1dBackwardVF

#endif  // CAUSAL_CONV1D_BACKWARD_VF_COMPUTE_H
