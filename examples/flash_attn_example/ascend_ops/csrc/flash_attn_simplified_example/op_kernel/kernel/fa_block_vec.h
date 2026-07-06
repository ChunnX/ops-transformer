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
 * \file fa_block_vec.h
 * \brief Vector-side compute — static tensor, no TPipe, template block size.
 */
#ifndef FA_BLOCK_VEC_H
#define FA_BLOCK_VEC_H

#ifndef __DAV_C310_CUBE__

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "adv_api/activation/softmax.h"
#include "../vector/vf/vf_softmax.h"
#include "../vector/vf/vf_rescale.h"
#include "../fa_kernel_public.h"
#include "../memcopy/memory_copy.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;

static constexpr uint32_t BLOCK_BYTES = 32;
static constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF;

// ---- Fixed shape (same as Cube) ----
static constexpr uint32_t FIXED_B   = 1;
static constexpr uint32_t FIXED_N2  = 1;
static constexpr uint32_t FIXED_G   = 32;
static constexpr uint32_t FIXED_S1  = 8192;
static constexpr uint32_t FIXED_S2  = 8192;

static constexpr uint32_t NUM_BN2         = FIXED_B * FIXED_N2;                    // 1
static constexpr uint32_t PRELOAD_N       = 2;

static constexpr uint32_t UB_SOFTMAX_BUF  = 256;
static constexpr uint32_t UB_SOFTMAX_ELEMS = UB_SOFTMAX_BUF / sizeof(float);        // 64
static constexpr uint32_t UB_COMMON_BUF   = 512;
static constexpr uint32_t UB_COMMON_ELEMS  = UB_COMMON_BUF / sizeof(uint8_t);       // 512

using IN_T  = bfloat16_t;
using T     = float;
using OUT_T = bfloat16_t;

// ---- AIV inner-core event IDs ----
// stage1Buf uses ids [0, PRELOAD_N]; E_VEC2_ID follows to avoid overlap.
#define E_VEC2_ID  (PRELOAD_N + 1)

// ---- ComputeSoftmax: softmax computation ----
template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void ComputeSoftmax(
    LocalTensor<float> &bmm1Ub,
    LocalTensor<IN_T>  &stage1Out,
    LocalTensor<float> (&sumBuf)[PRELOAD_N + 1],
    LocalTensor<float> (&maxBuf)[PRELOAD_N + 1],
    LocalTensor<float> (&expBuf)[PRELOAD_N + 1],
    LocalTensor<uint8_t> &commonBuf,
    uint32_t bmm1Idx, uint32_t stage1BufIdx, uint32_t mloop, uint32_t loop,
    bool isFirstS2Loop, bool isUpdate,
    float softmaxScale, float negMin)
{
    CrossCoreWaitFlag<4, PIPE_V>(bmm1Idx);

    LocalTensor<float> sumU = sumBuf[mloop % (PRELOAD_N + 1)];
    LocalTensor<float> maxU = maxBuf[mloop % (PRELOAD_N + 1)];
    LocalTensor<float> expU = expBuf[loop % (PRELOAD_N + 1)];
    LocalTensor<uint8_t> tmp = commonBuf;
    LocalTensor<uint8_t> dummyMask;

    get_buf(PIPE_V, stage1BufIdx, false);
    rls_buf(PIPE_V, stage1BufIdx, false);
    if (!isUpdate) {
        ProcessVec1Vf<float, IN_T, false, M_BASE, N_BASE, EQ_128, false>(
            stage1Out, nullptr, sumU, maxU, bmm1Ub, expU, sumU, maxU, dummyMask,
            tmp, M_BASE / 2, N_BASE, softmaxScale, negMin);
    } else {
        ProcessVec1Vf<float, IN_T, true, M_BASE, N_BASE, EQ_128, false>(
            stage1Out, nullptr, sumU, maxU, bmm1Ub, expU, sumU, maxU, dummyMask,
            tmp, M_BASE / 2, N_BASE, softmaxScale, negMin);
    }

    if (!isFirstS2Loop) {
        UpdateExpSumAndExpMax<float>(sumU, maxU, expU, sumU, maxU, tmp, M_BASE / 2);
    }
    get_buf(PIPE_V, stage1BufIdx, true);
    rls_buf(PIPE_V, stage1BufIdx, true);
}

// ---- CopySoftmaxToL1: copy P to L1 ----
template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void CopySoftmaxToL1(
    LocalTensor<IN_T> &stage1Out,
    LocalTensor<IN_T> &l1P,
    uint32_t bmm1Idx, uint32_t stage1BufIdx, uint32_t l1PIdx,
    uint32_t subBlockIdx)
{
    get_buf(PIPE_MTE3, stage1BufIdx, false);
    rls_buf(PIPE_MTE3, stage1BufIdx, false);
    constexpr uint32_t halfM = M_BASE / 2;
    DataCopy(l1P[subBlockIdx * (BLOCK_BYTES / sizeof(IN_T)) * (M_BASE - halfM)],
             stage1Out,
             {N_BASE / 16, (uint16_t)halfM,
              (uint16_t)((M_BASE / 2) + 1 - halfM),
              (uint16_t)(M_BASE - halfM)});
    get_buf(PIPE_MTE3, stage1BufIdx, true);
    rls_buf(PIPE_MTE3, stage1BufIdx, true);

    CrossCoreSetFlag<4, PIPE_V>(bmm1Idx);
    CrossCoreSetFlag<4, PIPE_MTE3>(5 + l1PIdx);
}

// ---- ComputeRescale: flash update + last div + cast ----
template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void ComputeRescale(
    LocalTensor<float> &bmm2Ub,
    LocalTensor<float> &stage2Buf,
    LocalTensor<float> (&sumBuf)[PRELOAD_N + 1],
    LocalTensor<float> (&expBuf)[PRELOAD_N + 1],
    uint32_t bmm2Idx, uint32_t mloop, uint32_t loop,
    bool isFirstS2Loop, bool isLastS2Loop)
{
    CrossCoreWaitFlag<4, PIPE_V>(3 + bmm2Idx);
    constexpr uint32_t halfM = M_BASE / 2;

    if (isFirstS2Loop) {
        get_buf(PIPE_V, E_VEC2_ID, false);
        rls_buf(PIPE_V, E_VEC2_ID, false);
        DataCopy(stage2Buf, bmm2Ub, halfM * D_SIZE);
    } else {
        LocalTensor<float> expU = expBuf[loop % (PRELOAD_N + 1)];
        if (!isLastS2Loop) {
            FlashUpdateNew<float, IN_T, OUT_T, D_SIZE, false>(
                stage2Buf, bmm2Ub, stage2Buf, expU, stage2Buf,
                halfM, D_SIZE, 1.0f, 1.0f);
        } else {
            LocalTensor<float> sumU = sumBuf[mloop % (PRELOAD_N + 1)];
            FlashUpdateLastNew<float, IN_T, OUT_T, D_SIZE, false>(
                stage2Buf, bmm2Ub, stage2Buf, expU, stage2Buf, sumU,
                halfM, D_SIZE, 1.0f, 1.0f);
        }
    }
    CrossCoreSetFlag<4, PIPE_V>(3 + bmm2Idx);
}

// ---- CopyAttnOutToGm: output copy to GM ----
template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void CopyAttnOutToGm(
    LocalTensor<float> &stage2Buf, LocalTensor<OUT_T> &castResult,
    FaGmTensor<OUT_T, GmFormat::BSNGD> &outGmTensor, LocalTensor<float> (&sumBuf)[PRELOAD_N + 1],
    uint32_t bmm2Idx, uint32_t mloop, uint32_t bIdx, uint32_t n2Idx,
    uint32_t gS1Idx, uint32_t vecMStart, bool isFirstS2Loop)
{
    constexpr uint32_t halfM = M_BASE / 2;
    if (isFirstS2Loop) {
        LocalTensor<float> sumU = sumBuf[mloop % (PRELOAD_N + 1)];
        LastDivNew<float, IN_T, OUT_T, D_SIZE>(stage2Buf, stage2Buf, sumU, halfM, D_SIZE, 0.0f);
    }

    Cast(castResult, stage2Buf, RoundMode::CAST_ROUND, halfM * D_SIZE);
    get_buf(PIPE_V, E_VEC2_ID, true);
    rls_buf(PIPE_V, E_VEC2_ID, true);
    get_buf(PIPE_MTE3, E_VEC2_ID, false);
    rls_buf(PIPE_MTE3, E_VEC2_ID, false);

    FaUbTensor<OUT_T> ubTensor{.tensor = castResult,
                               .rowCount = M_BASE / 2,
                               .colCount = D_SIZE};
    GmCoord gmCoord{.bIdx = bIdx,
                    .n2Idx = n2Idx,
                    .gS1Idx = gS1Idx + vecMStart,
                    .dIdx = 0,
                    .gS1DealSize = M_BASE / 2,
                    .dDealSize = D_SIZE};

    CopyAttnOutUbToGm<OUT_T, GmFormat::BSNGD>(outGmTensor, ubTensor, gmCoord);

    get_buf(PIPE_MTE3, E_VEC2_ID, true);
    rls_buf(PIPE_MTE3, E_VEC2_ID, true);
}

// ---- Main Vector entry ----
template <uint32_t M_BASE = 128, uint32_t N_BASE = 128, uint32_t D_SIZE = 128>
__aicore__ inline void Vector(__gm__ uint8_t *query, __gm__ uint8_t *key,
                              __gm__ uint8_t *value, __gm__ uint8_t *attentionOut,
                              __gm__ uint8_t *workspace, __gm__ uint8_t *tiling,
                              uint32_t blockDim, float softmaxScale)
{
    static constexpr uint32_t NUM_GS1_BLOCKS  = FIXED_S1 * FIXED_G / M_BASE;
    static constexpr uint32_t NUM_S2_BLOCKS   = FIXED_S2 / N_BASE;
    static constexpr uint32_t UB_BMM1_ELEMS   = M_BASE / 2 * N_BASE;
    static constexpr uint32_t UB_BMM2_ELEMS   = M_BASE / 2 * D_SIZE;
    static constexpr uint32_t UB_STAGE1_ELEMS = (M_BASE / 2 + 2) * N_BASE;
    static constexpr uint32_t UB_STAGE2_ELEMS = M_BASE / 2 * D_SIZE;
    static constexpr uint32_t L1_P_ELEMS      = M_BASE * N_BASE;

    // === GM output tensor setup ===
    FaGmTensor<OUT_T, GmFormat::BSNGD> outGmTensor((__gm__ OUT_T *)attentionOut, FIXED_B, FIXED_N2, FIXED_G, FIXED_S1, D_SIZE);

    // === UB buffers (shared — must match Cube addresses) ===
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

    // === L1 P buffers (shared — must match Cube addresses) ===
    uint32_t l1Addr = 0;
    LocalTensor<IN_T> l1P[3] = {
        LocalTensor<IN_T>(TPosition::A1, l1Addr, L1_P_ELEMS),
        LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_P_ELEMS * sizeof(IN_T)), L1_P_ELEMS),
        LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_P_ELEMS * sizeof(IN_T)), L1_P_ELEMS)
    };

    // === UB AIV-private buffers ===
    LocalTensor<float> softmaxSum[PRELOAD_N + 1] = {
        LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_BMM2_ELEMS * sizeof(float)), UB_SOFTMAX_ELEMS),
        LocalTensor<float>(TPosition::VECIN, ubAddr += UB_SOFTMAX_ELEMS * sizeof(float), UB_SOFTMAX_ELEMS),
        LocalTensor<float>(TPosition::VECIN, ubAddr += UB_SOFTMAX_ELEMS * sizeof(float), UB_SOFTMAX_ELEMS)
    };
    LocalTensor<float> softmaxMax[PRELOAD_N + 1] = {
        LocalTensor<float>(TPosition::VECIN, ubAddr += UB_SOFTMAX_ELEMS * sizeof(float), UB_SOFTMAX_ELEMS),
        LocalTensor<float>(TPosition::VECIN, ubAddr += UB_SOFTMAX_ELEMS * sizeof(float), UB_SOFTMAX_ELEMS),
        LocalTensor<float>(TPosition::VECIN, ubAddr += UB_SOFTMAX_ELEMS * sizeof(float), UB_SOFTMAX_ELEMS)
    };
    LocalTensor<float> softmaxExp[PRELOAD_N + 1] = {
        LocalTensor<float>(TPosition::VECIN, ubAddr += UB_SOFTMAX_ELEMS * sizeof(float), UB_SOFTMAX_ELEMS),
        LocalTensor<float>(TPosition::VECIN, ubAddr += UB_SOFTMAX_ELEMS * sizeof(float), UB_SOFTMAX_ELEMS),
        LocalTensor<float>(TPosition::VECIN, ubAddr += UB_SOFTMAX_ELEMS * sizeof(float), UB_SOFTMAX_ELEMS)
    };
    LocalTensor<uint8_t> commonBuf(TPosition::VECIN, ubAddr += UB_SOFTMAX_ELEMS * sizeof(float), UB_COMMON_ELEMS);

    LocalTensor<IN_T> stage1Buf[PRELOAD_N + 1] = {
        LocalTensor<IN_T>(TPosition::VECIN, ubAddr += UB_COMMON_ELEMS * sizeof(uint8_t), UB_STAGE1_ELEMS),
        LocalTensor<IN_T>(TPosition::VECIN, ubAddr += UB_STAGE1_ELEMS * sizeof(IN_T), UB_STAGE1_ELEMS),
        LocalTensor<IN_T>(TPosition::VECIN, ubAddr += UB_STAGE1_ELEMS * sizeof(IN_T), UB_STAGE1_ELEMS)
    };

    LocalTensor<float> stage2Buf(
        TPosition::VECIN, ubAddr += UB_STAGE1_ELEMS * sizeof(IN_T), UB_STAGE2_ELEMS);
    LocalTensor<IN_T> castBuf(
        TPosition::VECIN, ubAddr, UB_STAGE2_ELEMS);

    // === Cross-core init ===
    CrossCoreSetFlag<4, PIPE_V>(0);
    CrossCoreSetFlag<4, PIPE_V>(1);
    CrossCoreSetFlag<4, PIPE_V>(2);
    CrossCoreSetFlag<4, PIPE_V>(3);
    CrossCoreSetFlag<4, PIPE_V>(4);

    // === Main loop ===
    uint32_t aivIdx     = GetBlockIdx();
    uint32_t aicIdx     = aivIdx / 2;
    uint32_t subBlockIdx = GetSubBlockIdx();
    uint32_t gs1PerCore = NUM_GS1_BLOCKS / blockDim;
    uint32_t gs1Start   = aicIdx * gs1PerCore;
    uint32_t gs1End     = gs1Start + gs1PerCore;
    uint32_t bIdx = 0, n2Idx = 0;

    uint32_t tmpNeg = NEGATIVE_MIN_VALUE_FP32;
    float negMin = *((float *)&tmpNeg);

    uint32_t i = 0, mloop = 0;
    uint32_t p2S2 = 0, p2Row = 0;
    for (uint32_t gs1Block = gs1Start; gs1Block < gs1End; gs1Block++) {
        for (uint32_t s2Block = 0; s2Block < NUM_S2_BLOCKS; s2Block++) {
            uint32_t bmm1Idx = i % (PRELOAD_N + 1);
            uint32_t l1PIdx    = i % 3;
            uint32_t stage1BufIdx = i % (PRELOAD_N + 1);
            bool isFirstS2Loop = (s2Block == 0);

            if (isFirstS2Loop) {
                mloop++;
            }

            ComputeSoftmax<M_BASE, N_BASE, D_SIZE>(ubBmm1[bmm1Idx], stage1Buf[stage1BufIdx],
                           softmaxSum, softmaxMax, softmaxExp, commonBuf,
                           bmm1Idx, stage1BufIdx, mloop - 1, i, isFirstS2Loop, !isFirstS2Loop,
                           softmaxScale, negMin);

            CopySoftmaxToL1<M_BASE, N_BASE, D_SIZE>(stage1Buf[stage1BufIdx], l1P[l1PIdx],
                            bmm1Idx, stage1BufIdx, l1PIdx, subBlockIdx);

            if (i >= PRELOAD_N) {
                uint32_t j          = i - PRELOAD_N;
                uint32_t bmm2Idx    = j & 1;
                bool jIsFirst       = (p2S2 == 0);
                bool jIsLast        = (p2S2 == NUM_S2_BLOCKS - 1);
                uint32_t gS1Idx_j   = (gs1Start + p2Row) * M_BASE;

                ComputeRescale<M_BASE, N_BASE, D_SIZE>(ubBmm2[bmm2Idx], stage2Buf,
                               softmaxSum, softmaxExp, bmm2Idx, p2Row, j, jIsFirst, jIsLast);

                if (jIsLast) {
                    CopyAttnOutToGm<M_BASE, N_BASE, D_SIZE>(stage2Buf, castBuf, outGmTensor, softmaxSum,
                                    bmm2Idx, p2Row, bIdx, n2Idx, gS1Idx_j,
                                    subBlockIdx * (M_BASE / 2), jIsFirst);
                }
                if (++p2S2 == NUM_S2_BLOCKS) {
                    p2S2 = 0;
                    p2Row++;
                }
            }
            i++;
        }
    }

    uint32_t total = i;
    for (uint32_t k = 0; k < PRELOAD_N; k++) {
        uint32_t j          = total - PRELOAD_N + k;
        uint32_t bmm2Idx    = j & 1;
        bool jIsFirst       = (p2S2 == 0);
        bool jIsLast        = (p2S2 == NUM_S2_BLOCKS - 1);
        uint32_t gS1Idx_j   = (gs1Start + p2Row) * M_BASE;

        ComputeRescale<M_BASE, N_BASE, D_SIZE>(ubBmm2[bmm2Idx], stage2Buf,
                       softmaxSum, softmaxExp, bmm2Idx, p2Row, j, jIsFirst, jIsLast);

        if (jIsLast) {
            CopyAttnOutToGm<M_BASE, N_BASE, D_SIZE>(stage2Buf, castBuf, outGmTensor, softmaxSum,
                            bmm2Idx, p2Row, 0, 0, gS1Idx_j,
                            subBlockIdx * (M_BASE / 2), jIsFirst);
        }
        if (++p2S2 == NUM_S2_BLOCKS) {
            p2S2 = 0;
            p2Row++;
        }
    }
}
#endif // __DAV_C310_CUBE__
#endif // FA_BLOCK_VEC_H