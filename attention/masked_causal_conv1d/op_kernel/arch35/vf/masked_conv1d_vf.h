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
 * \file masked_conv1d_vf.h
 * \brief VF function for MaskedCausalConv1d kernel-width-3 causal convolution.
 *
 * ioQueue layout (unified prefix + x buffer):
 *   [bUbCur][sCur+2][H_REG]
 *   Row s   of batch b  =  x[s0 + s - 2]   (s=0,1 are prefix rows loaded from GM)
 *   Row s+1 of batch b  =  x[s0 + s - 1]
 *   Row s+2 of batch b  =  x[s0 + s]       (current token)
 *
 * In-place write: y[s] is stored back to row s (overwrites x[s0+s-2], already consumed).
 * After VF: rows 0..sCur-1 hold y[0..sCur-1]; rows sCur, sCur+1 still hold
 *           x[s0+sCur-2], x[s0+sCur-1] (untouched — not needed by CopyOut).
 *
 * Mask is applied at __aicore__ level (in Compute) to avoid b8 SLD.
 */

#ifndef MASKED_CONV1D_VF_H
#define MASKED_CONV1D_VF_H

#include "kernel_operator.h"

using namespace AscendC;

namespace MaskedCausalConv1dNs {

constexpr MicroAPI::CastTrait castTraitB162B32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr MicroAPI::CastTrait castTraitB322B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

// H_REG: VF FP32 register width in elements (256B / 4B = 64)
constexpr uint32_t H_REG = 64;

// ============================================================================
// MaskedConv1dVF: 3-tap causal convolution with unified prefix+x buffer.
//
// ioBase   : [bUbCur][sCur+2][H_REG] T  — row s = x[s0+s-2]; in-place output
// wBase    : [3][H_REG] T               — weight[0..2]
// sCur     : number of S tokens in this tile
// bUbCur   : number of B elements in this tile
// ============================================================================
template <typename T>
__simd_vf__ void MaskedConv1dVF(
    __ubuf__ T* ioBase,
    __ubuf__ T* wBase,
    uint32_t    sCur,
    uint32_t    bUbCur)
{
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    // Load weight once — shared across all (b, s)
    MicroAPI::RegTensor<T>     w0B16, w1B16, w2B16;
    MicroAPI::RegTensor<float> w0F32, w1F32, w2F32;
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w0B16, wBase + 0 * H_REG);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w1B16, wBase + 1 * H_REG);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w2B16, wBase + 2 * H_REG);
    MicroAPI::Cast<float, T, castTraitB162B32>(w0F32, w0B16, maskFull);
    MicroAPI::Cast<float, T, castTraitB162B32>(w1F32, w1B16, maskFull);
    MicroAPI::Cast<float, T, castTraitB162B32>(w2F32, w2B16, maskFull);

    for (uint32_t b = 0; b < bUbCur; ++b) {
        // Base element offset for batch b in the [bUbCur][sCur+2][H_REG] buffer
        uint32_t bBase = b * (sCur + 2) * H_REG;

        for (uint32_t s = 0; s < sCur; ++s) {
            // Row s   → x[s0+s-2]  (prefix row 0 or 1, or earlier x)
            // Row s+1 → x[s0+s-1]
            // Row s+2 → x[s0+s]    (current token)
            __ubuf__ T* addrPrev2 = ioBase + bBase + s       * H_REG;
            __ubuf__ T* addrPrev1 = ioBase + bBase + (s + 1) * H_REG;
            __ubuf__ T* addrCurr  = ioBase + bBase + (s + 2) * H_REG;

            MicroAPI::RegTensor<T>     xp2B16, xp1B16, xcB16;
            MicroAPI::RegTensor<float> xp2F32, xp1F32, xcF32, yF32, tmpF32;

            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xp2B16, addrPrev2);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xp1B16, addrPrev1);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xcB16,  addrCurr);
            MicroAPI::Cast<float, T, castTraitB162B32>(xp2F32, xp2B16, maskFull);
            MicroAPI::Cast<float, T, castTraitB162B32>(xp1F32, xp1B16, maskFull);
            MicroAPI::Cast<float, T, castTraitB162B32>(xcF32,  xcB16,  maskFull);

            // y = x[-2]*w[0] + x[-1]*w[1] + x[0]*w[2]  (FP32 accumulation)
            MicroAPI::Mul(yF32,   xp2F32, w0F32, maskFull);
            MicroAPI::Mul(tmpF32, xp1F32, w1F32, maskFull);
            MicroAPI::Add(yF32,   yF32,   tmpF32, maskFull);
            MicroAPI::Mul(tmpF32, xcF32,  w2F32, maskFull);
            MicroAPI::Add(yF32,   yF32,   tmpF32, maskFull);

            MicroAPI::RegTensor<T> yB16;
            MicroAPI::Cast<T, float, castTraitB322B16>(yB16, yF32, maskFull);

            // In-place write: y[s] → row s (x[s0+s-2] already consumed, safe to overwrite)
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                ioBase + bBase + s * H_REG, yB16, maskFull);
        }
        // Rows sCur and sCur+1 remain untouched: x[s0+sCur-2], x[s0+sCur-1]
        // (not needed by CopyOut; no special handling required)
    }
}

// ============================================================================
// CallMaskedConv1dVF: __aicore__ wrapper converting LocalTensor to raw ptr
// ============================================================================
template <typename T>
__aicore__ inline void CallMaskedConv1dVF(
    LocalTensor<T> ioBuf,
    LocalTensor<T> wBuf,
    uint32_t       sCur,
    uint32_t       bUbCur)
{
    __ubuf__ T* ioBase = (__ubuf__ T*)ioBuf.GetPhyAddr();
    __ubuf__ T* wBase  = (__ubuf__ T*)wBuf.GetPhyAddr();
    MaskedConv1dVF<T>(ioBase, wBase, sCur, bUbCur);
}

} // namespace MaskedCausalConv1dNs

#endif // MASKED_CONV1D_VF_H
