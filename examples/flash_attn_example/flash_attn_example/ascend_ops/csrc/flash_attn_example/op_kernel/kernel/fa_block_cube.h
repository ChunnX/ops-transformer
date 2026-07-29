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
 * \brief Cube-side compute — standalone function, static tensor + bufId, BN1→S1.
 *        Uses BSND format for Q (flat BN1 index), BSND for KV.
 *        PRELOAD_N=2 pipelining: BMM1/BMM2 overlapped via multi-buffering.
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
using namespace fa_kernel::util;
using namespace fa_base_matmul;

namespace FaCube {

static constexpr uint32_t PRELOAD_N       = 2;
static constexpr uint32_t UB_BMM1_ELEMS   = 64 * 128;
static constexpr uint32_t UB_BMM2_ELEMS   = 64 * 128;
static constexpr uint32_t L0_AB_ELEMS     = 128 * 128;
static constexpr uint32_t L0C_ELEMS       = 128 * 128;

using IN_T  = bfloat16_t;
using T     = float;
using OUT_T = bfloat16_t;

#define E_L1Q(s)  (0 + (s))
#define E_L1K(s)  (2 + (s))
#define E_L1V(s)  (4 + (s))
#define E_L0C(s)  (8 + (s))
#define E_L0AB(s) (12 + (s))

#define C_BMM1(s)     (s)
#define C_BMM1_D(s)   (16 + (s))
#define C_BMM2(s)     (3 + (s))
#define C_BMM2_D(s)   (19 + (s))
#define C_P(s)        (5 + (s))
#define C_P_D(s)      (21 + (s))

static constexpr FixpipeConfig COMMON_FP_CFG = {CO2Layout::ROW_MAJOR, true};

template <uint32_t N_BASE>
__aicore__ inline void FixpipeMm1(const LocalTensor<float> &dst, const LocalTensor<float> &l0C,
    uint32_t mActual, uint32_t nActual)
{
    uint32_t nSize = (nActual + 7) >> 3 << 3;
    uint32_t mSize = (mActual + 1) >> 1 << 1;
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> p;
    p.nSize = nSize;
    p.mSize = mSize;
    p.srcStride = ((mSize + 15) / 16) * 16;
    p.dstStride = N_BASE;
    p.dualDstCtl = 1;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<float, float, COMMON_FP_CFG>(dst, l0C, p);
}

template <uint32_t M_BASE, uint32_t D_SIZE>
__aicore__ inline void FixpipeMm2(const LocalTensor<float> &dst, const LocalTensor<float> &l0C,
    uint32_t mActual, uint32_t nActual)
{
    uint32_t nSize = (D_SIZE + 7) >> 3 << 3;
    uint32_t mSize = (mActual + 1) >> 1 << 1;
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> p;
    p.nSize = nSize;
    p.mSize = mSize;
    p.srcStride = ((mSize + 15) / 16) * 16;
    p.dstStride = ((D_SIZE + 15) >> 4 << 4);
    p.dualDstCtl = 1;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<float, float, COMMON_FP_CFG>(dst, l0C, p);
}

template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void ComputeMm1(
    Buffer<IN_T>  &q,    Buffer<IN_T>  &k,
    RingBuffer<IN_T> &l0A, RingBuffer<IN_T> &l0B,
    RingBuffer<float> &l0C,
    Buffer<float> &bmm1,
    FaGmTensor<IN_T, GmFormat::BSND> &qGm,
    FaGmTensor<IN_T, GmFormat::BSND> &kGm,
    uint32_t bIdx, uint32_t n1Head, uint32_t n2Head,
    uint32_t s1BlockBase, uint32_t s2BlockBase,
    uint32_t mActual, uint32_t nActual,
    bool isFirstS2Loop, bool isLastS2Loop)
{
    CrossCoreWaitFlag<4, PIPE_FIX>(C_BMM1(bmm1.idx));
    CrossCoreWaitFlag<4, PIPE_FIX>(C_BMM1_D(bmm1.idx));

    if (isFirstS2Loop) {
        Mutex::Lock<PIPE_MTE2>(E_L1Q(q.idx));
        FaL1Tensor<IN_T> qL1{.tensor = q.tensor, .rowCount = (((uint32_t)mActual + 15) >> 4 << 4)};
        CopyMatrixGmToL1<IN_T, GmFormat::BSND>(qL1, qGm, bIdx, n1Head, s1BlockBase, mActual);
        Mutex::Unlock<PIPE_MTE2>(E_L1Q(q.idx));
        Mutex::Lock<PIPE_MTE1>(E_L1Q(q.idx));
    }

    Mutex::Lock<PIPE_MTE2>(E_L1K(k.idx));
    {
        FaL1Tensor<IN_T> kL1{.tensor = k.tensor, .rowCount = (((uint32_t)nActual + 15) >> 4 << 4)};
        CopyMatrixGmToL1<IN_T, GmFormat::BSND>(kL1, kGm, bIdx, n2Head, s2BlockBase, nActual);
    }
    Mutex::Unlock<PIPE_MTE2>(E_L1K(k.idx));

    Mutex::Lock<PIPE_MTE1>(E_L1K(k.idx));
    Buffer<float> cBuf = l0C.GetNext();
    Mutex::Lock<PIPE_M>(E_L0C(cBuf.idx));
    {
        MMParam param = MakeMMParam(mActual, nActual, (uint32_t)D_SIZE, false, true);
        MatmulFull<IN_T, IN_T, float, static_cast<uint32_t>(128), static_cast<uint32_t>(128), static_cast<uint32_t>(128), ABLayout::MK, ABLayout::KN>(
            q.tensor, k.tensor, l0A, l0B, cBuf.tensor, param);
    }
    Mutex::Unlock<PIPE_M>(E_L0C(cBuf.idx));
    Mutex::Unlock<PIPE_MTE1>(E_L1K(k.idx));

    if (isLastS2Loop) {
        Mutex::Unlock<PIPE_MTE1>(E_L1Q(q.idx));
    }

    Mutex::Lock<PIPE_FIX>(E_L0C(cBuf.idx));
    FixpipeMm1<N_BASE>(bmm1.tensor, cBuf.tensor, mActual, nActual);
    Mutex::Unlock<PIPE_FIX>(E_L0C(cBuf.idx));

    CrossCoreSetFlag<4, PIPE_FIX>(C_BMM1(bmm1.idx));
    CrossCoreSetFlag<4, PIPE_FIX>(C_BMM1_D(bmm1.idx));
}

template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void ComputeMm2(
    Buffer<IN_T>  &p,    Buffer<IN_T>  &v,
    RingBuffer<IN_T> &l0A, RingBuffer<IN_T> &l0B,
    RingBuffer<float> &l0C,
    Buffer<float> &bmm2,
    FaGmTensor<IN_T, GmFormat::BSND> &vGm,
    uint32_t bIdx, uint32_t n2Head, uint32_t s2BlockBase,
    uint32_t mActual, uint32_t nActual)
{
    CrossCoreWaitFlag<4, PIPE_FIX>(C_BMM2(bmm2.idx));
    CrossCoreWaitFlag<4, PIPE_FIX>(C_BMM2_D(bmm2.idx));
    CrossCoreWaitFlag<4, PIPE_MTE1>(C_P(p.idx));
    CrossCoreWaitFlag<4, PIPE_MTE1>(C_P_D(p.idx));

    Mutex::Lock<PIPE_MTE2>(E_L1V(v.idx));
    {
        FaL1Tensor<IN_T> vL1{.tensor = v.tensor, .rowCount = AlignUp(nActual, 16U)};
        CopyMatrixGmToL1<IN_T, GmFormat::BSND>(vL1, vGm, bIdx, n2Head, s2BlockBase, nActual);
    }
    Mutex::Unlock<PIPE_MTE2>(E_L1V(v.idx));

    Mutex::Lock<PIPE_MTE1>(E_L1V(v.idx));
    Buffer<float> cBuf = l0C.GetNext();
    Mutex::Lock<PIPE_M>(E_L0C(cBuf.idx));
    {
        MMParam param = MakeMMParam(M_BASE, (uint32_t)D_SIZE, nActual, false, false);
        param.realM = mActual;
        MatmulFull<IN_T, IN_T, float, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
            p.tensor, v.tensor, l0A, l0B, cBuf.tensor, param);
    }
    Mutex::Unlock<PIPE_M>(E_L0C(cBuf.idx));
    Mutex::Unlock<PIPE_MTE1>(E_L1V(v.idx));

    Mutex::Lock<PIPE_FIX>(E_L0C(cBuf.idx));
    FixpipeMm2<M_BASE, D_SIZE>(bmm2.tensor, cBuf.tensor, mActual, nActual);
    Mutex::Unlock<PIPE_FIX>(E_L0C(cBuf.idx));

    CrossCoreSetFlag<4, PIPE_FIX>(C_BMM2(bmm2.idx));
    CrossCoreSetFlag<4, PIPE_FIX>(C_BMM2_D(bmm2.idx));
}

__aicore__ inline void CalcCurS2Range(uint32_t &s2StartBlock, uint32_t &s2EndBlock,
    uint32_t s1BlockBase, uint32_t mBaseSize,
    uint64_t actS1Size, uint64_t actS2Size,
    int64_t preTokens, int64_t nextTokens,
    bool hasAttnMask, uint32_t s2BaseSize)
{
    if (!hasAttnMask) {
        s2StartBlock = 0U;
        s2EndBlock = static_cast<uint32_t>((actS2Size + s2BaseSize - 1) / s2BaseSize);
        return;
    }

    int64_t s1FirstToken = static_cast<int64_t>(s1BlockBase);
    int64_t s1LastToken = Min(s1FirstToken + static_cast<int64_t>(mBaseSize),
                              static_cast<int64_t>(actS1Size)) - 1;

    int64_t s2FirstToken = s1FirstToken - preTokens;
    int64_t s2LastToken = s1LastToken + nextTokens;

    if (s2FirstToken >= static_cast<int64_t>(actS2Size) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
        s2StartBlock = 0U;
        s2EndBlock = 0U;
        return;
    }

    s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actS2Size - 1));
    s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actS2Size - 1));
    s2StartBlock = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
    s2EndBlock = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;
}

static constexpr uint32_t CTX_RING_SIZE = PRELOAD_N + 1;

struct Bmm2Ctx {
    uint32_t bIdx;
    uint32_t n2Head;
    uint32_t s2BlockBase;
    uint32_t mActual;
    uint32_t nActual;
};

template <uint32_t M_BASE = 128, uint32_t N_BASE = 128, uint32_t D_SIZE = 128>
__aicore__ inline void CubeFunc(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                __gm__ uint8_t *value, __gm__ uint8_t *attnMask,
                                __gm__ uint8_t *tilingGM,
                                const optiling::FaBaseParams &faBaseParams,
                                const optiling::FaAttnMaskParams &faAttnMaskParams,
                                const optiling::FaMetaData &faMetaData,
                                uint32_t aicIdx)
{
    static constexpr uint32_t L1_P_ELEMS   = M_BASE * N_BASE;
    static constexpr uint32_t L1_Q_ELEMS   = M_BASE * D_SIZE;
    static constexpr uint32_t L1_KV_ELEMS  = N_BASE * D_SIZE;

    uint32_t l1Addr = 0;
    Buffer<IN_T> l1P[3] = {
        {LocalTensor<IN_T>(TPosition::A1, l1Addr, L1_P_ELEMS), 0},
        {LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_P_ELEMS * sizeof(IN_T)), L1_P_ELEMS), 1},
        {LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_P_ELEMS * sizeof(IN_T)), L1_P_ELEMS), 2}
    };
    Buffer<IN_T> l1Q[2] = {
        {LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_P_ELEMS * sizeof(IN_T)), L1_Q_ELEMS), 0},
        {LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_Q_ELEMS * sizeof(IN_T)), L1_Q_ELEMS), 1}
    };
    Buffer<IN_T> l1K[2] = {
        {LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_Q_ELEMS * sizeof(IN_T)), L1_KV_ELEMS), 0},
        {LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_KV_ELEMS * sizeof(IN_T)), L1_KV_ELEMS), 1}
    };
    Buffer<IN_T> l1V[2] = {
        {LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_KV_ELEMS * sizeof(IN_T)), L1_KV_ELEMS), 0},
        {LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_KV_ELEMS * sizeof(IN_T)), L1_KV_ELEMS), 1}
    };

    Buffer<IN_T> l0ABufs[2] = {
        {LocalTensor<IN_T>(TPosition::A2, 0, L0_AB_ELEMS), 0},
        {LocalTensor<IN_T>(TPosition::A2, (L0_AB_ELEMS * sizeof(IN_T)), L0_AB_ELEMS), 1}
    };
    Buffer<IN_T> l0BBufs[2] = {
        {LocalTensor<IN_T>(TPosition::B2, 0, L0_AB_ELEMS), 0},
        {LocalTensor<IN_T>(TPosition::B2, (L0_AB_ELEMS * sizeof(IN_T)), L0_AB_ELEMS), 1}
    };

    Buffer<float> l0CBufs[4] = {
        {LocalTensor<float>(TPosition::CO1, 0, L0C_ELEMS), 0},
        {LocalTensor<float>(TPosition::CO1, (1 * L0C_ELEMS * sizeof(T)), L0C_ELEMS), 1},
        {LocalTensor<float>(TPosition::CO1, (2 * L0C_ELEMS * sizeof(T)), L0C_ELEMS), 2},
        {LocalTensor<float>(TPosition::CO1, (3 * L0C_ELEMS * sizeof(T)), L0C_ELEMS), 3}
    };

    RingBuffer<IN_T>  l0A(l0ABufs, 2);
    RingBuffer<IN_T>  l0B(l0BBufs, 2);
    RingBuffer<float> l0C(l0CBufs, 4);

    uint32_t ubAddr = 0;
    Buffer<float> ubBmm1[PRELOAD_N + 1] = {
        {LocalTensor<float>(TPosition::VECIN, ubAddr, UB_BMM1_ELEMS), 0},
        {LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_BMM1_ELEMS * sizeof(float)), UB_BMM1_ELEMS), 1},
        {LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_BMM1_ELEMS * sizeof(float)), UB_BMM1_ELEMS), 2}
    };
    Buffer<float> ubBmm2[2] = {
        {LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_BMM1_ELEMS * sizeof(float)), UB_BMM2_ELEMS), 0},
        {LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_BMM2_ELEMS * sizeof(float)), UB_BMM2_ELEMS), 1}
    };

    uint32_t B   = faBaseParams.bSize;
    uint32_t N1  = faBaseParams.n1Size;
    uint32_t N2  = faBaseParams.n2Size;
    uint32_t S1  = faBaseParams.s1Size;
    uint32_t S2  = faBaseParams.s2Size;

    bool hasAttnMask = (faAttnMaskParams.sparseMode != 0);
    int64_t preTokens = faAttnMaskParams.preTokens;
    int64_t nextTokens = faAttnMaskParams.nextTokens;

    uint32_t bn1Start = faMetaData.FAMetadata[aicIdx][optiling::FA_BN1_START_INDEX];
    uint32_t s1OStart  = faMetaData.FAMetadata[aicIdx][optiling::FA_S1_START_INDEX];
    uint32_t bn1End   = faMetaData.FAMetadata[aicIdx][optiling::FA_BN1_END_INDEX];
    uint32_t s1OEnd   = faMetaData.FAMetadata[aicIdx][optiling::FA_S1_END_INDEX];
    uint32_t coreEnable = faMetaData.FAMetadata[aicIdx][optiling::FA_CORE_ENABLE_INDEX];
    if (!coreEnable) {
        return;
    }

    uint32_t numS1Blocks = (S1 + M_BASE - 1) / M_BASE;

    FaGmTensor<IN_T, GmFormat::BSND> qGm((__gm__ IN_T *)query, B, N1, S1, (uint32_t)D_SIZE);
    FaGmTensor<IN_T, GmFormat::BSND> kGm((__gm__ IN_T *)key,   B, N2, S2, (uint32_t)D_SIZE);
    FaGmTensor<IN_T, GmFormat::BSND> vGm((__gm__ IN_T *)value, B, N2, S2, (uint32_t)D_SIZE);

    Bmm2Ctx bmm2CtxRing[CTX_RING_SIZE];

    uint32_t effectiveBn1End = bn1End;
    if (s1OEnd > 0) {
        effectiveBn1End++;
    }

    uint32_t i = 0;
    uint32_t s1SeqIdx = 0;

    for (uint32_t bn1 = bn1Start; bn1 < effectiveBn1End; bn1++) {
        uint32_t bIdx   = bn1 / N1;
        uint32_t n1Head = bn1 % N1;
        uint32_t n2Head = n1Head * N2 / N1;

        uint32_t s1Start = (bn1 == bn1Start) ? s1OStart : 0;
        uint32_t s1Stop;
        if (bn1 + 1 == effectiveBn1End) {
            s1Stop = (s1OEnd > 0) ? s1OEnd : numS1Blocks;
        } else {
            s1Stop = numS1Blocks;
        }
        if (s1Start >= s1Stop) {
            continue;
        }

        for (uint32_t s1Block = s1Start; s1Block < s1Stop; s1Block++) {
            uint32_t s1BlockBase = s1Block * M_BASE;
            uint32_t mActual = s1BlockBase + M_BASE > S1 ? S1 - s1BlockBase : (uint32_t)M_BASE;

            uint32_t s2StartBlock = 0, s2EndBlock = 0;
            CalcCurS2Range(s2StartBlock, s2EndBlock, s1BlockBase, M_BASE,
                S1, S2, preTokens, nextTokens, hasAttnMask, N_BASE);

            if (s2StartBlock >= s2EndBlock) {
                continue;
            }

            uint32_t l1QIdx = s1SeqIdx & 1;
            bool isQFirst = true;

            for (uint32_t s2Block = s2StartBlock; s2Block < s2EndBlock; s2Block++) {
                uint32_t s2BlockBase = s2Block * N_BASE;
                uint32_t nActual = s2BlockBase + N_BASE > S2 ? S2 - s2BlockBase : (uint32_t)N_BASE;
                bool isLastS2Loop = ((s2Block + 1) == s2EndBlock);

                uint32_t bmm1Idx = i % (PRELOAD_N + 1);
                uint32_t l1KIdx  = i & 1;
                uint32_t ringSlot = i % CTX_RING_SIZE;

                bmm2CtxRing[ringSlot].bIdx = bIdx;
                bmm2CtxRing[ringSlot].n2Head = n2Head;
                bmm2CtxRing[ringSlot].s2BlockBase = s2BlockBase;
                bmm2CtxRing[ringSlot].mActual = mActual;
                bmm2CtxRing[ringSlot].nActual = nActual;

                ComputeMm1<M_BASE, N_BASE, D_SIZE>(l1Q[l1QIdx], l1K[l1KIdx], l0A, l0B, l0C,
                    ubBmm1[bmm1Idx],
                    qGm, kGm, bIdx, n1Head, n2Head,
                    s1BlockBase, s2BlockBase, mActual, nActual,
                    isQFirst, isLastS2Loop);
                isQFirst = false;

                if (i >= PRELOAD_N) {
                    uint32_t j = i - PRELOAD_N;
                    uint32_t bmm2Idx = j & 1;
                    uint32_t l1PIdx  = j % 3;
                    uint32_t l1VIdx  = j & 1;
                    Bmm2Ctx &ctx = bmm2CtxRing[j % CTX_RING_SIZE];

                    ComputeMm2<M_BASE, N_BASE, D_SIZE>(l1P[l1PIdx], l1V[l1VIdx], l0A, l0B, l0C,
                        ubBmm2[bmm2Idx],
                        vGm, ctx.bIdx, ctx.n2Head, ctx.s2BlockBase,
                        ctx.mActual, ctx.nActual);
                }
                i++;
            }
            s1SeqIdx++;
        }
    }

    uint32_t total = i;
    uint32_t drainCount = (total > PRELOAD_N) ? PRELOAD_N : total;
    for (uint32_t k = 0; k < drainCount; k++) {
        uint32_t j = total - drainCount + k;
        uint32_t bmm2Idx = j & 1;
        uint32_t l1PIdx  = j % 3;
        uint32_t l1VIdx  = j & 1;
        Bmm2Ctx &ctx = bmm2CtxRing[j % CTX_RING_SIZE];

        ComputeMm2<M_BASE, N_BASE, D_SIZE>(l1P[l1PIdx], l1V[l1VIdx], l0A, l0B, l0C,
            ubBmm2[bmm2Idx],
            vGm, ctx.bIdx, ctx.n2Head, ctx.s2BlockBase,
            ctx.mActual, ctx.nActual);
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

} // namespace FaCube

#endif // __DAV_C310_CUBE__
#endif // FA_BLOCK_CUBE_H
