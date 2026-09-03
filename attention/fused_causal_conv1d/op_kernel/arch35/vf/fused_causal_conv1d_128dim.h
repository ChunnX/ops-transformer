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
 * \file compute.h
 * \brief
 */

#ifndef FUSED_CAUSAL_CONV1D_128DIM_H
#define FUSED_CAUSAL_CONV1D_128DIM_H

#include "kernel_operator.h"
#include "fused_causal_conv1d_state_single_tail.h"

// The a5 headers are included directly, not via <pto/pto-inst.hpp>: that aggregate pulls
// TTrans.hpp, whose file-scope `using namespace pto;` makes TPipe and GlobalTensor ambiguous.
#include <pto/common/type.hpp>
#include <pto/common/memory.hpp>
#include <pto/common/arch_macro.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/npu/a5/TCvt.hpp>

namespace FusedCausalConv1dPtoVf {

constexpr int32_t PTO_LANES_B32 = 64;      // fp32 lanes in one 256-byte vector register
constexpr uint32_t PTO_ROT = 3;            // output rows per rotated iteration
constexpr uint32_t PTO_SHORT_RUN_ROWS = 8; // decode passes seqLen - (K-1), at most 7

// SiLU in registers: x / (1 + exp(-x)).
__aicore__ inline void SiluInReg(pto::RegTensor<float> &acc, pto::RegTensor<float> &tmp, pto::MaskReg &preg)
{
    using namespace pto;
    vmuls(tmp, acc, (float)-1, preg, MODE_ZEROING);
    vexp(tmp, tmp, preg, MODE_ZEROING);
    vadds(tmp, tmp, (float)1, preg, MODE_ZEROING);
    vdiv(acc, acc, tmp, preg, MODE_ZEROING);
}

#define CC1D_LOAD(reg, stage, base, off) \
    do { \
        vlds((stage), (base), (off), UNPK_B16); \
        vcvt((reg), (stage), pg16, PART_EVEN); \
    } while (0)

#define CC1D_STORE(acc, xr, off) \
    do { \
        if constexpr (HASBIAS) { \
            vadd((acc), (acc), bias, preg, MODE_ZEROING); \
        } \
        if constexpr (HASACT) { \
            SiluInReg((acc), tmp, preg); \
        } \
        if constexpr (RESIDUAL) { \
            vadd((acc), (acc), (xr), preg, MODE_ZEROING); \
        } \
        /* The kernels replaced zero their accumulator and add into it, so an all -0.0 row gives \
           +0.0 there; these start from the multiply and give -0.0. +0.0 normalises just that. */ \
        vadds((acc), (acc), (float)0.0, preg, MODE_ZEROING); \
        vcvt(y16, (acc), preg, ROUND_R, RS_DISABLE, PART_EVEN); \
        vsts(y16, yAddr, (off), PK_B32, preg); \
    } while (0)

template <typename T, bool RESIDUAL, bool HASBIAS = false, bool HASACT = false>
__aicore__ inline void Conv1dNoNeedStateShortRunPtoVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
                                                      uint32_t xSLen, uint32_t dimLen, __ubuf__ T *biasAddr = nullptr)
{
    using namespace pto;
    const int32_t rowStride = (int32_t)dimLen;
    const int32_t stride2 = 2 * rowStride;
    const uint16_t groups = (uint16_t)(dimLen / (uint32_t)PTO_LANES_B32);
    const uint16_t rows = (uint16_t)xSLen;

    __VEC_SCOPE__
    {
        RegTensor<float> w0, w1, w2, x0, x1, x2, acc, bias, tmp;
        RegTensor<T> s0, s1, s2, y16;
        MaskReg preg = PSetWithType<float>(PAT_ALL);
        uint32_t len16 = ELE_CNT_B16;
        MaskReg pg16 = CreatePredicate<half>(len16);

        int32_t go = 0;
        for (uint16_t g = 0; g < groups; ++g) {
            CC1D_LOAD(w0, s0, weightAddr, go);
            CC1D_LOAD(w1, s1, weightAddr, go + rowStride);
            CC1D_LOAD(w2, s2, weightAddr, go + stride2);
            if constexpr (HASBIAS) {
                CC1D_LOAD(bias, s0, biasAddr, go);
            }

            int32_t ro = go;
            for (uint16_t r = 0; r < rows; ++r) {
                CC1D_LOAD(x0, s0, xAddr, ro);
                CC1D_LOAD(x1, s1, xAddr, ro + rowStride);
                CC1D_LOAD(x2, s2, xAddr, ro + stride2);
                vmul(acc, x0, w0, preg, MODE_ZEROING);
                vmula(acc, x1, w1, preg, MODE_ZEROING);
                vmula(acc, x2, w2, preg, MODE_ZEROING);
                CC1D_STORE(acc, x2, ro);
                ro += rowStride;
            }
            go += PTO_LANES_B32;
        }
    }
}

template <typename T, bool RESIDUAL, bool HASBIAS = false, bool HASACT = false>
__aicore__ inline void Conv1dNoNeedStateLongRunPtoVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
                                                     uint32_t xSLen, uint32_t dimLen, __ubuf__ T *biasAddr = nullptr)
{
    using namespace pto;
    const int32_t rowStride = (int32_t)dimLen;
    const int32_t stride2 = 2 * rowStride;
    const int32_t stride3 = 3 * rowStride;
    const uint16_t groups = (uint16_t)(dimLen / (uint32_t)PTO_LANES_B32);
    const uint16_t steps = (uint16_t)(xSLen / PTO_ROT);
    const uint16_t tail = (uint16_t)(xSLen % PTO_ROT);

    __VEC_SCOPE__
    {
        RegTensor<float> w0, w1, w2, a0, a1, a2, x0, x1, x2, bias, tmp;
        RegTensor<T> s0, s1, s2, y16;
        MaskReg preg = PSetWithType<float>(PAT_ALL);
        uint32_t len16 = ELE_CNT_B16;
        MaskReg pg16 = CreatePredicate<half>(len16);

// Input row xr completes output (old): store it, then open the two later outputs. Accumulator
// and staging names rotate with the body.
#define CC1D_STEP(fresh, mid, old, xr, stage, inOff, outOff) \
    do { \
        CC1D_LOAD((xr), (stage), xAddr, (inOff)); \
        vmula((old), (xr), w2, preg, MODE_ZEROING); \
        CC1D_STORE((old), (xr), (outOff)); \
        vmul((fresh), (xr), w0, preg, MODE_ZEROING); \
        vmula((mid), (xr), w1, preg, MODE_ZEROING); \
    } while (0)

        int32_t go = 0;
        for (uint16_t g = 0; g < groups; ++g) {
            CC1D_LOAD(w0, s0, weightAddr, go);
            CC1D_LOAD(w1, s1, weightAddr, go + rowStride);
            CC1D_LOAD(w2, s2, weightAddr, go + stride2);
            if constexpr (HASBIAS) {
                CC1D_LOAD(bias, s0, biasAddr, go);
            }

            CC1D_LOAD(x0, s0, xAddr, go);
            vmul(a0, x0, w0, preg, MODE_ZEROING);
            CC1D_LOAD(x1, s1, xAddr, go + rowStride);
            vmul(a1, x1, w0, preg, MODE_ZEROING);
            vmula(a0, x1, w1, preg, MODE_ZEROING);

            int32_t in = go + stride2;
            int32_t out = go;
            for (uint16_t t = 0; t < steps; ++t) {
                CC1D_STEP(a2, a1, a0, x2, s2, in, out);
                CC1D_STEP(a0, a2, a1, x0, s0, in + rowStride, out + rowStride);
                CC1D_STEP(a1, a0, a2, x1, s1, in + stride2, out + stride2);
                in += stride3;
                out += stride3;
            }
            if (tail >= 1) {
                CC1D_STEP(a2, a1, a0, x2, s2, in, out);
            }
            if (tail >= 2) {
                CC1D_STEP(a0, a2, a1, x0, s0, in + rowStride, out + rowStride);
            }
            go += PTO_LANES_B32;
        }
#undef CC1D_STEP
    }
}

/*
 * One output row whose K-wide window straddles the state buffer: STATE_ROWS rows from stateAddr,
 * the rest from xAddr, tap k applying to window position k either way. STATE_ROWS is a template
 * parameter so the vector scope stays branch-free. Matches Conv1dNeedState*VF.
 */
template <typename T, bool RESIDUAL, uint32_t STATE_ROWS, bool HASBIAS = false, bool HASACT = false>
__aicore__ inline void Conv1dNeedStatePtoVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                            __ubuf__ T *yAddr, uint32_t dimLen, __ubuf__ T *biasAddr = nullptr)
{
    using namespace pto;
    static_assert(STATE_ROWS >= 1 && STATE_ROWS <= 2, "K = 3 leaves 1 or 2 rows in the state buffer");
    const int32_t rowStride = (int32_t)dimLen;
    const uint16_t groups = (uint16_t)(dimLen / (uint32_t)PTO_LANES_B32);

    __VEC_SCOPE__
    {
        RegTensor<float> w0, w1, w2, d0, d1, d2, acc, bias, tmp;
        RegTensor<T> s0, s1, s2, y16;
        MaskReg preg = PSetWithType<float>(PAT_ALL);
        uint32_t len16 = ELE_CNT_B16;
        MaskReg pg16 = CreatePredicate<half>(len16);

        int32_t go = 0;
        for (uint16_t g = 0; g < groups; ++g) {
            CC1D_LOAD(w0, s0, weightAddr, go);
            CC1D_LOAD(w1, s1, weightAddr, go + rowStride);
            CC1D_LOAD(w2, s2, weightAddr, go + 2 * rowStride);
            if constexpr (HASBIAS) {
                CC1D_LOAD(bias, s0, biasAddr, go);
            }
            if constexpr (STATE_ROWS == 1) {
                CC1D_LOAD(d0, s0, stateAddr, go);
                CC1D_LOAD(d1, s1, xAddr, go);
                CC1D_LOAD(d2, s2, xAddr, go + rowStride);
            } else {
                CC1D_LOAD(d0, s0, stateAddr, go);
                CC1D_LOAD(d1, s1, stateAddr, go + rowStride);
                CC1D_LOAD(d2, s2, xAddr, go);
            }
            vmul(acc, d0, w0, preg, MODE_ZEROING);
            vmula(acc, d1, w1, preg, MODE_ZEROING);
            vmula(acc, d2, w2, preg, MODE_ZEROING);
            CC1D_STORE(acc, d2, go);
            go += PTO_LANES_B32;
        }
    }
}

/*
 * Decode with a full K-1 state: two output rows share one pass over four loads.
 * Matches Conv1dNeedState*BHVF, selected when convStateLen == 2.
 *   y0 = s0*w0 + s1*w1 + x0*w2      y1 = s1*w0 + x0*w1 + x1*w2
 */
template <typename T, bool RESIDUAL, bool HASBIAS = false, bool HASACT = false>
__aicore__ inline void Conv1dNeedStateBHPtoVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                              __ubuf__ T *yAddr, uint32_t dimLen, __ubuf__ T *biasAddr = nullptr)
{
    using namespace pto;
    const int32_t rowStride = (int32_t)dimLen;
    const uint16_t groups = (uint16_t)(dimLen / (uint32_t)PTO_LANES_B32);

    __VEC_SCOPE__
    {
        RegTensor<float> w0, w1, w2, st0, st1, xr0, xr1, acc, bias, tmp;
        RegTensor<T> s0, s1, s2, y16;
        MaskReg preg = PSetWithType<float>(PAT_ALL);
        uint32_t len16 = ELE_CNT_B16;
        MaskReg pg16 = CreatePredicate<half>(len16);

        int32_t go = 0;
        for (uint16_t g = 0; g < groups; ++g) {
            CC1D_LOAD(w0, s0, weightAddr, go);
            CC1D_LOAD(w1, s1, weightAddr, go + rowStride);
            CC1D_LOAD(w2, s2, weightAddr, go + 2 * rowStride);
            if constexpr (HASBIAS) {
                CC1D_LOAD(bias, s0, biasAddr, go);
            }
            CC1D_LOAD(st0, s0, stateAddr, go);
            CC1D_LOAD(st1, s1, stateAddr, go + rowStride);
            CC1D_LOAD(xr0, s2, xAddr, go);
            CC1D_LOAD(xr1, s0, xAddr, go + rowStride);

            vmul(acc, st0, w0, preg, MODE_ZEROING);
            vmula(acc, st1, w1, preg, MODE_ZEROING);
            vmula(acc, xr0, w2, preg, MODE_ZEROING);
            CC1D_STORE(acc, xr0, go);

            vmul(acc, st1, w0, preg, MODE_ZEROING);
            vmula(acc, xr0, w1, preg, MODE_ZEROING);
            vmula(acc, xr1, w2, preg, MODE_ZEROING);
            CC1D_STORE(acc, xr1, go + rowStride);
            go += PTO_LANES_B32;
        }
    }
}

#undef CC1D_LOAD
#undef CC1D_STORE

} // namespace FusedCausalConv1dPtoVf

template <typename T>
__aicore__ inline void Conv1dNeedState(LocalTensor<T> &xUb, LocalTensor<T> &weightUb, LocalTensor<T> &stateUb,
                                       LocalTensor<T> &yUb, uint8_t stateSLen, uint32_t xSLen, uint32_t dimLen,
                                       int32_t isResidualConnection)
{
    __ubuf__ T *xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T *weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T *stateAddr = (__ubuf__ T *)stateUb.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)yUb.GetPhyAddr();
    // stateSLen is 1 or 2 for K = 3; selecting it here keeps the vector scope branch-free.
    if (stateSLen == 1) {
        if (isResidualConnection == 1) {
            FusedCausalConv1dPtoVf::Conv1dNeedStatePtoVF<T, true, 1>(xAddr, weightAddr, stateAddr, yAddr, dimLen);
        } else {
            FusedCausalConv1dPtoVf::Conv1dNeedStatePtoVF<T, false, 1>(xAddr, weightAddr, stateAddr, yAddr, dimLen);
        }
    } else if (isResidualConnection == 1) {
        FusedCausalConv1dPtoVf::Conv1dNeedStatePtoVF<T, true, 2>(xAddr, weightAddr, stateAddr, yAddr, dimLen);
    } else {
        FusedCausalConv1dPtoVf::Conv1dNeedStatePtoVF<T, false, 2>(xAddr, weightAddr, stateAddr, yAddr, dimLen);
    }
}

template <typename T>
__aicore__ inline void Conv1dNeedStateBH(LocalTensor<T> &xUb, LocalTensor<T> &weightUb, LocalTensor<T> &stateUb,
                                         LocalTensor<T> &yUb, uint32_t convStateLen, uint32_t dimLen,
                                         int32_t isResidualConnection)
{
    __ubuf__ T *xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T *weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T *stateAddr = (__ubuf__ T *)stateUb.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)yUb.GetPhyAddr();
    if (isResidualConnection == 1) {
        if (convStateLen == 2) {
            FusedCausalConv1dPtoVf::Conv1dNeedStateBHPtoVF<T, true>(xAddr, weightAddr, stateAddr, yAddr, dimLen);
        } else {
            FusedCausalConv1dPtoVf::Conv1dNeedStatePtoVF<T, true, 2>(xAddr, weightAddr, stateAddr, yAddr, dimLen);
        }
    } else {
        if (convStateLen == 2) {
            FusedCausalConv1dPtoVf::Conv1dNeedStateBHPtoVF<T, false>(xAddr, weightAddr, stateAddr, yAddr, dimLen);
        } else {
            FusedCausalConv1dPtoVf::Conv1dNeedStatePtoVF<T, false, 2>(xAddr, weightAddr, stateAddr, yAddr, dimLen);
        }
    }
}

template <typename T>
__aicore__ inline void Conv1dNoNeedState(LocalTensor<T> &xUb, LocalTensor<T> &weightUb, LocalTensor<T> &yUb,
                                         uint32_t xSLen, uint32_t dimLen, int32_t isResidualConnection)
{
    __ubuf__ T *xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T *weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)yUb.GetPhyAddr();
    // Below this row count the rotated body's priming prologue costs more than it saves. The split
    // is on the scalar side so neither kernel carries a branch inside its vector scope.
    if (xSLen < FusedCausalConv1dPtoVf::PTO_SHORT_RUN_ROWS) {
        if (isResidualConnection == 1) {
            FusedCausalConv1dPtoVf::Conv1dNoNeedStateShortRunPtoVF<T, true>(xAddr, weightAddr, yAddr, xSLen, dimLen);
        } else {
            FusedCausalConv1dPtoVf::Conv1dNoNeedStateShortRunPtoVF<T, false>(xAddr, weightAddr, yAddr, xSLen, dimLen);
        }
    } else if (isResidualConnection == 1) {
        FusedCausalConv1dPtoVf::Conv1dNoNeedStateLongRunPtoVF<T, true>(xAddr, weightAddr, yAddr, xSLen, dimLen);
    } else {
        FusedCausalConv1dPtoVf::Conv1dNoNeedStateLongRunPtoVF<T, false>(xAddr, weightAddr, yAddr, xSLen, dimLen);
    }
}

#endif
