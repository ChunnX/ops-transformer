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
 * \file fa_block_cube.h
 * \brief Cube-side compute — static tensor, no TPipe, template block size.
 */
#ifndef FA_BLOCK_CUBE_H
#define FA_BLOCK_CUBE_H

#ifdef __DAV_C310_CUBE__

#include "../memcopy/offset_calculator.h"
#include "../matmul/matmul.h"
#include "../memcopy/memory_copy.h"
#include "../fa_kernel_public.h"
#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace fa_base_matmul;

// ---- Fixed shape ----
static constexpr uint32_t FIXED_B   = 1;
static constexpr uint32_t FIXED_N2  = 1;
static constexpr uint32_t FIXED_G   = 32;
static constexpr uint32_t FIXED_S1  = 8192;
static constexpr uint32_t FIXED_S2  = 8192;

static constexpr uint32_t NUM_BN2          = FIXED_B * FIXED_N2;                    // 1
static constexpr uint32_t PRELOAD_N        = 2;
static constexpr uint32_t UB_BMM1_ELEMS    = 64 * 128;                              // M_BASE / 2 × N_BASE: 8192
static constexpr uint32_t UB_BMM2_ELEMS    = 64 * 128;                              // M_BASE / 2 × D_SIZE: 8192
static constexpr uint32_t L0_AB_ELEMS      = 128 * 128;                             // M_BASE * D_SIZE = N_BASE * D_SIZE
static constexpr uint32_t L0C_ELEMS        = 128 * 128;                             // M_BASE * N_BASE

using IN_T  = bfloat16_t;
using T     = float;
using OUT_T = bfloat16_t;

// ---- inner-core event IDs (get_buf/rls_buf, IDs 6,7 reserved) ----
#define E_L1Q(s)  (0 + (s))
#define E_L1K(s)  (2 + (s))
#define E_L1V(s)  (4 + (s))
// 6,7 reserved
#define E_L0C(s)  (8 + (s))
#define E_L0AB(s) (12 + (s))

// ---- standalone Fixpipe helpers ----
static constexpr FixpipeConfig COMMON_FP_CFG = {CO2Layout::ROW_MAJOR, true};

template <uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void FixpipeMm1(const LocalTensor<float> &dst, const LocalTensor<float> &l0C)
{
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> p;
    p.nSize = (N_BASE + 7) >> 3 << 3;
    p.mSize = ((N_BASE > D_SIZE ? N_BASE : D_SIZE) + 1) >> 1 << 1;
    p.srcStride = ((p.mSize + 15) / 16) * 16;
    p.dstStride = N_BASE;
    p.dualDstCtl = 1;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<float, float, COMMON_FP_CFG>(dst, l0C, p);
}

template <uint32_t M_BASE, uint32_t D_SIZE>
__aicore__ inline void FixpipeMm2(const LocalTensor<float> &dst, const LocalTensor<float> &l0C)
{
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> p;
    p.nSize = ((M_BASE > D_SIZE ? M_BASE : D_SIZE) + 7) >> 3 << 3;
    p.mSize = (M_BASE + 1) >> 1 << 1;
    p.srcStride = ((p.mSize + 15) / 16) * 16;
    p.dstStride = ((D_SIZE + 15) >> 4 << 4);
    p.dualDstCtl = 1;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<float, float, COMMON_FP_CFG>(dst, l0C, p);
}

// ---- BMM1: Q @ K^T ----

template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void ComputeMm1(
    LocalTensor<IN_T> &l1Q, LocalTensor<IN_T> &l1K,
    LocalTensor<IN_T> (&l0A)[2], LocalTensor<IN_T> (&l0B)[2],
    uint32_t &l0aIdx, uint32_t &l0bIdx,
    LocalTensor<float> &l0C,
    LocalTensor<float> &ubBmm1, uint32_t bmm1Idx,
    uint32_t l1QIdx, uint32_t l1KIdx, uint32_t l0CIdx,
    FaGmTensor<IN_T, GmFormat::BSNGD> &qGm,
    FaGmTensor<IN_T, GmFormat::BSND>  &kGm,
    uint32_t bIdx, uint32_t n2Idx, uint32_t gS1Idx, uint32_t s2Idx,
    bool isFirstS2Loop, bool isLastS2Loop)
{
    CrossCoreWaitFlag<4, PIPE_FIX>(bmm1Idx);
    CrossCoreWaitFlag<4, PIPE_FIX>(bmm1Idx + 16);

    if (isFirstS2Loop) {
        get_buf(PIPE_MTE2, E_L1Q(l1QIdx), false);
        rls_buf(PIPE_MTE2, E_L1Q(l1QIdx), false);
        FaL1Tensor<IN_T> qL1{.tensor = l1Q, .rowCount = (M_BASE + 15) >> 4 << 4};
        GmCoord qCoord{bIdx, n2Idx, gS1Idx, 0, M_BASE, D_SIZE};
        CopyQueryGmToL1<IN_T, GmFormat::BSNGD>(qL1, qGm, qCoord);
        get_buf(PIPE_MTE2, E_L1Q(l1QIdx), true);
        rls_buf(PIPE_MTE2, E_L1Q(l1QIdx), true);
    }

    get_buf(PIPE_MTE2, E_L1K(l1KIdx), false);
    rls_buf(PIPE_MTE2, E_L1K(l1KIdx), false);
    {
        FaL1Tensor<IN_T> kL1{.tensor = l1K, .rowCount = (N_BASE + 15) >> 4 << 4};
        GmKvCoord kCoord{bIdx, n2Idx, s2Idx, 0, N_BASE, D_SIZE};
        CopyKvGmToL1<IN_T, GmFormat::BSND>(kL1, kGm, kCoord);
    }
    get_buf(PIPE_MTE2, E_L1K(l1KIdx), true);
    rls_buf(PIPE_MTE2, E_L1K(l1KIdx), true);

    get_buf(PIPE_MTE1, E_L1Q(l1QIdx), false);
    rls_buf(PIPE_MTE1, E_L1Q(l1QIdx), false);
    get_buf(PIPE_MTE1, E_L1K(l1KIdx), false);
    rls_buf(PIPE_MTE1, E_L1K(l1KIdx), false);

    get_buf(PIPE_M, E_L0C(l0CIdx), false);
    rls_buf(PIPE_M, E_L0C(l0CIdx), false);

    {
        LocalTensor<IN_T> aL0[2] = {l0A[0], l0A[1]};
        LocalTensor<IN_T> bL0[2] = {l0B[0], l0B[1]};
        MMParam param = MakeMMParam(M_BASE, N_BASE, D_SIZE, false, true);
        MatmulFull<IN_T, IN_T, float, static_cast<uint32_t>(128), static_cast<uint32_t>(128), static_cast<uint32_t>(128), ABLayout::MK, ABLayout::KN>(
            l1Q, l1K, aL0, l0aIdx, bL0, l0bIdx, l0C, param, E_L0AB(l0aIdx));
    }
    if (isLastS2Loop) {
        get_buf(PIPE_MTE1, E_L1Q(l1QIdx), true);
        rls_buf(PIPE_MTE1, E_L1Q(l1QIdx), true);
    }
    get_buf(PIPE_MTE1, E_L1K(l1KIdx), true);
    rls_buf(PIPE_MTE1, E_L1K(l1KIdx), true);

    get_buf(PIPE_M, E_L0C(l0CIdx), true);
    rls_buf(PIPE_M, E_L0C(l0CIdx), true);
    get_buf(PIPE_FIX, E_L0C(l0CIdx), false);
    rls_buf(PIPE_FIX, E_L0C(l0CIdx), false);

    FixpipeMm1<N_BASE, D_SIZE>(ubBmm1, l0C);

    get_buf(PIPE_FIX, E_L0C(l0CIdx), true);
    rls_buf(PIPE_FIX, E_L0C(l0CIdx), true);

    CrossCoreSetFlag<4, PIPE_FIX>(bmm1Idx);
    CrossCoreSetFlag<4, PIPE_FIX>(bmm1Idx + 16);
}

// ---- BMM2: P @ V ----

template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void ComputeMm2(
    LocalTensor<IN_T> &l1P, LocalTensor<IN_T> &l1V,
    LocalTensor<IN_T> (&l0A)[2], LocalTensor<IN_T> (&l0B)[2],
    uint32_t &l0aIdx, uint32_t &l0bIdx,
    LocalTensor<float> &l0C,
    LocalTensor<float> &ubBmm2, uint32_t bmm2Idx, uint32_t l1PIdx,
    uint32_t l1VIdx, uint32_t l0CIdx,
    FaGmTensor<IN_T, GmFormat::BSND> &vGm,
    uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx)
{
    CrossCoreWaitFlag<4, PIPE_FIX>(3 + bmm2Idx);
    CrossCoreWaitFlag<4, PIPE_FIX>(3 + bmm2Idx + 16);
    CrossCoreWaitFlag<4, PIPE_MTE1>(5 + l1PIdx);
    CrossCoreWaitFlag<4, PIPE_MTE1>(5 + l1PIdx + 16);

    get_buf(PIPE_MTE2, E_L1V(l1VIdx), false);
    rls_buf(PIPE_MTE2, E_L1V(l1VIdx), false);
    {
        FaL1Tensor<IN_T> vL1{.tensor = l1V, .rowCount = AlignUp(N_BASE, 16U)};
        GmKvCoord vCoord{bIdx, n2Idx, s2Idx, 0, N_BASE, D_SIZE};
        CopyKvGmToL1<IN_T, GmFormat::BSND>(vL1, vGm, vCoord);
    }
    get_buf(PIPE_MTE2, E_L1V(l1VIdx), true);
    rls_buf(PIPE_MTE2, E_L1V(l1VIdx), true);

    get_buf(PIPE_MTE1, E_L1V(l1VIdx), false);
    rls_buf(PIPE_MTE1, E_L1V(l1VIdx), false);
    get_buf(PIPE_M, E_L0C(l0CIdx), false);
    rls_buf(PIPE_M, E_L0C(l0CIdx), false);

    {
        LocalTensor<IN_T> aL0[2] = {l0A[0], l0A[1]};
        LocalTensor<IN_T> bL0[2] = {l0B[0], l0B[1]};
        MMParam param = {M_BASE, D_SIZE, N_BASE, false, false};
        param.realM = M_BASE;
        MatmulFull<IN_T, IN_T, float, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
            l1P, l1V, aL0, l0aIdx, bL0, l0bIdx, l0C, param, E_L0AB(l0aIdx));
    }

    get_buf(PIPE_M, E_L0C(l0CIdx), true);
    rls_buf(PIPE_M, E_L0C(l0CIdx), true);
    get_buf(PIPE_FIX, E_L0C(l0CIdx), false);
    rls_buf(PIPE_FIX, E_L0C(l0CIdx), false);

    FixpipeMm2<M_BASE, D_SIZE>(ubBmm2, l0C);

    get_buf(PIPE_FIX, E_L0C(l0CIdx), true);
    rls_buf(PIPE_FIX, E_L0C(l0CIdx), true);
    get_buf(PIPE_MTE1, E_L1V(l1VIdx), true);
    rls_buf(PIPE_MTE1, E_L1V(l1VIdx), true);

    CrossCoreSetFlag<4, PIPE_FIX>(3 + bmm2Idx);
    CrossCoreSetFlag<4, PIPE_FIX>(3 + bmm2Idx + 16);
}

// ---- Main Cube entry ----

template <uint32_t M_BASE = 128, uint32_t N_BASE = 128, uint32_t D_SIZE = 128>
__aicore__ inline void Cube(__gm__ uint8_t *query, __gm__ uint8_t *key,
                            __gm__ uint8_t *value, __gm__ uint8_t *workspace,
                            __gm__ uint8_t *tiling, uint32_t blockDim)
{
    static constexpr uint32_t NUM_GS1_BLOCKS = FIXED_S1 * FIXED_G / M_BASE;
    static constexpr uint32_t NUM_S2_BLOCKS = FIXED_S2 / N_BASE;
    static constexpr uint32_t L1_P_ELEMS   = M_BASE * N_BASE;
    static constexpr uint32_t L1_Q_ELEMS   = M_BASE * D_SIZE;
    static constexpr uint32_t L1_KV_ELEMS  = N_BASE * D_SIZE;

    // === GM tensor setup ===
    FaGmTensor<IN_T, GmFormat::BSNGD> qGm((__gm__ IN_T *)query, FIXED_B, FIXED_N2, FIXED_G, FIXED_S1, D_SIZE);
    FaGmTensor<IN_T, GmFormat::BSND>  kGm((__gm__ IN_T *)key,   FIXED_B, FIXED_N2, FIXED_S2, D_SIZE, 0);
    FaGmTensor<IN_T, GmFormat::BSND>  vGm((__gm__ IN_T *)value, FIXED_B, FIXED_N2, FIXED_S2, D_SIZE, 0);

    // === L1 buffers — linear allocation from address 0 ===
    uint32_t l1Addr = 0;
    LocalTensor<IN_T> l1P[3] = {
        LocalTensor<IN_T>(TPosition::A1, l1Addr, L1_P_ELEMS),
        LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_P_ELEMS * sizeof(IN_T)), L1_P_ELEMS),
        LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_P_ELEMS * sizeof(IN_T)), L1_P_ELEMS)
    };
    LocalTensor<IN_T> l1Q[2] = {
        LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_P_ELEMS * sizeof(IN_T)), L1_Q_ELEMS),
        LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_Q_ELEMS * sizeof(IN_T)), L1_Q_ELEMS)
    };
    LocalTensor<IN_T> l1K[2] = {
        LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_Q_ELEMS * sizeof(IN_T)), L1_KV_ELEMS),
        LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_KV_ELEMS * sizeof(IN_T)), L1_KV_ELEMS)
    };
    LocalTensor<IN_T> l1V[2] = {
        LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_KV_ELEMS * sizeof(IN_T)), L1_KV_ELEMS),
        LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_KV_ELEMS * sizeof(IN_T)), L1_KV_ELEMS)
    };

    // === L0A/L0B double-buffer ===
    LocalTensor<IN_T> l0A[2] = {
        LocalTensor<IN_T>(TPosition::A2, 0, L0_AB_ELEMS),
        LocalTensor<IN_T>(TPosition::A2, (L0_AB_ELEMS * sizeof(IN_T)), L0_AB_ELEMS)
    };
    LocalTensor<IN_T> l0B[2] = {
        LocalTensor<IN_T>(TPosition::B2, 0, L0_AB_ELEMS),
        LocalTensor<IN_T>(TPosition::B2, (L0_AB_ELEMS * sizeof(IN_T)), L0_AB_ELEMS)
    };

    // === L0C quad-buffer ===
    LocalTensor<float> l0C[4] = {
        LocalTensor<float>(TPosition::CO1, 0, L0C_ELEMS),
        LocalTensor<float>(TPosition::CO1, (1 * L0C_ELEMS * sizeof(T)), L0C_ELEMS),
        LocalTensor<float>(TPosition::CO1, (2 * L0C_ELEMS * sizeof(T)), L0C_ELEMS),
        LocalTensor<float>(TPosition::CO1, (3 * L0C_ELEMS * sizeof(T)), L0C_ELEMS)
    };

    // === UB buffers ===
    uint32_t ubAddr = 0;
    LocalTensor<float> ubBmm1[PRELOAD_N + 1] = {
        LocalTensor<float>(TPosition::VECIN, ubAddr, UB_BMM1_ELEMS),
        LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_BMM1_ELEMS * sizeof(float)), UB_BMM1_ELEMS),
        LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_BMM1_ELEMS * sizeof(float)), UB_BMM1_ELEMS)
    };
    LocalTensor<float> ubBmm2[2] = {
        LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_BMM1_ELEMS * sizeof(float)), UB_BMM2_ELEMS),
        LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_BMM2_ELEMS * sizeof(float)), UB_BMM2_ELEMS)
    };

    // === Main loop ===
    uint32_t aicIdx     = GetBlockIdx();
    uint32_t gs1PerCore = NUM_GS1_BLOCKS / blockDim;
    uint32_t gs1Start   = aicIdx * gs1PerCore;
    uint32_t gs1End     = gs1Start + gs1PerCore;
    uint32_t bIdx = 0, n2Idx = 0;

    uint32_t i = 0;
    uint32_t p2S2 = 0;
    uint32_t l0aIdx = 0, l0bIdx = 0;
    for (uint32_t gs1Block = gs1Start; gs1Block < gs1End; gs1Block++) {
        uint32_t gS1Idx = gs1Block * M_BASE;
        uint32_t l1QIdx = gs1Block & 1;
        bool isQFirst = true;

        for (uint32_t s2Block = 0; s2Block < NUM_S2_BLOCKS; s2Block++) {
            uint32_t bmm1Idx = i % (PRELOAD_N + 1);
            uint32_t l1KIdx = i & 1;
            uint32_t l0CIdx = i % 4;
            uint32_t s2Idx = s2Block * N_BASE;

            ComputeMm1<M_BASE, N_BASE, D_SIZE>(l1Q[l1QIdx], l1K[l1KIdx], l0A, l0B, l0aIdx, l0bIdx,
                       l0C[l0CIdx], ubBmm1[bmm1Idx], bmm1Idx,
                       l1QIdx, l1KIdx, l0CIdx,
                       qGm, kGm, bIdx, n2Idx, gS1Idx, s2Idx, isQFirst, (s2Block + 1) == NUM_S2_BLOCKS);
            isQFirst = false;

            if (i >= PRELOAD_N) {
                uint32_t j = i - PRELOAD_N;
                uint32_t bmm2Idx = j & 1;
                uint32_t l1PIdx = j % 3;
                uint32_t l1VIdx = j & 1;
                uint32_t l0CIdx2 = j % 4;
                uint32_t prevS2 = p2S2 * N_BASE;

                ComputeMm2<M_BASE, N_BASE, D_SIZE>(l1P[l1PIdx], l1V[l1VIdx], l0A, l0B, l0aIdx, l0bIdx,
                           l0C[l0CIdx2], ubBmm2[bmm2Idx], bmm2Idx, l1PIdx,
                           l1VIdx, l0CIdx2, vGm, bIdx, n2Idx, prevS2);
                if (++p2S2 == NUM_S2_BLOCKS) {
                    p2S2 = 0;
                }
            }
            i++;
        }
    }

    uint32_t total = i;
    for (uint32_t k = 0; k < PRELOAD_N; k++) {
        uint32_t j = total - PRELOAD_N + k;
        uint32_t bmm2Idx = j & 1;
        uint32_t l1PIdx  = j % 3;
        uint32_t l1VIdx  = j & 1;
        uint32_t l0CIdx2 = j % 4;
        uint32_t prevS2 = p2S2 * N_BASE;
        ComputeMm2<M_BASE, N_BASE, D_SIZE>(l1P[l1PIdx], l1V[l1VIdx], l0A, l0B, l0aIdx, l0bIdx,
                   l0C[l0CIdx2], ubBmm2[bmm2Idx], bmm2Idx, l1PIdx,
                   l1VIdx, l0CIdx2, vGm, 0, 0, prevS2);
        if (++p2S2 == NUM_S2_BLOCKS) {
            p2S2 = 0;
        }
    }
    CrossCoreWaitFlag<4, PIPE_FIX>(0);
    CrossCoreWaitFlag<4, PIPE_FIX>(1);
    CrossCoreWaitFlag<4, PIPE_FIX>(2);
    CrossCoreWaitFlag<4, PIPE_FIX>(3);
    CrossCoreWaitFlag<4, PIPE_FIX>(4);
    CrossCoreWaitFlag<4, PIPE_FIX>(0 + 16);
    CrossCoreWaitFlag<4, PIPE_FIX>(1 + 16);
    CrossCoreWaitFlag<4, PIPE_FIX>(2 + 16);
    CrossCoreWaitFlag<4, PIPE_FIX>(3 + 16);
    CrossCoreWaitFlag<4, PIPE_FIX>(4 + 16);
}
#endif // __DAV_C310_CUBE__
#endif // FA_BLOCK_CUBE_H