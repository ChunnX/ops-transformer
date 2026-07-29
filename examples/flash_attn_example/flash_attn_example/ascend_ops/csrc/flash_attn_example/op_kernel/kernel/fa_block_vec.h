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
 * \brief Vector-side compute — standalone function, static tensor + bufId, BN1→S1.
 *        Uses BSND format for output.
 *        PRELOAD_N=2 pipelining: Softmax/Copy/BMM2 overlapped via multi-buffering.
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
#include "../vector/vector.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace fa_kernel::util;
using namespace fa_base_vector;

namespace FaVector {

static constexpr uint32_t BLOCK_BYTES = 32;
static constexpr uint32_t CV_RATIO = 2;
static constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF;

static constexpr uint32_t PRELOAD_N = 2;

static constexpr uint32_t UB_SOFTMAX_BUF  = 256;
static constexpr uint32_t UB_SOFTMAX_ELEMS = UB_SOFTMAX_BUF / sizeof(float);
static constexpr uint32_t UB_COMMON_BUF   = 512;
static constexpr uint32_t UB_COMMON_ELEMS  = UB_COMMON_BUF / sizeof(uint8_t);

#define E_STAGE1(s)    (s)
#define E_STAGE2(s)    (3 + (s))
#define E_ATTN_MASK(s) (4 + (s))

#define C_BMM1(s)  (s)
#define C_BMM2(s)  (3 + (s))
#define C_P(s)     (5 + (s))

using IN_T  = bfloat16_t;
using T     = float;
using OUT_T = bfloat16_t;

template <typename ElemT>
struct Buffer {
    LocalTensor<ElemT> tensor;
    uint32_t idx;
};

struct VecBmm2Ctx {
    uint32_t bIdx;
    uint32_t n1Head;
    uint32_t s1BlockBase;
    bool     isFirstS2Loop;
    bool     isLastS2Loop;
    uint32_t mloopIdx;
    uint32_t mActual;
};

struct AttnMaskContext {
    __gm__ uint8_t *gmPtr;
    const optiling::FaAttnMaskParams *params;
    uint32_t s1BlockBase;
    uint32_t subBlockIdx;
    uint32_t s2BlockBase;
    uint32_t s1;
    uint32_t s2;
    int64_t preTokens;
    int64_t nextTokens;
    uint32_t maskValue;
    uint32_t mActual;
};

template <uint32_t M_BASE = 128, uint32_t N_BASE = 128>
__aicore__ inline void LoadAttnMask(
    Buffer<uint8_t> &attnMaskUb,
    const AttnMaskContext &ctx,
    uint32_t mHalf,
    uint32_t nActual)
{
    MaskInfo maskInfo;
    uint32_t gmRowOffset = (ctx.subBlockIdx == 0) ? 0 : ((ctx.mActual + 1) / 2);
    maskInfo.s1StartIdx  = ctx.s1BlockBase + gmRowOffset;
    maskInfo.s1dealNum   = mHalf;
    maskInfo.s1Size       = ctx.s1;
    maskInfo.s2StartIdx   = ctx.s2BlockBase;
    maskInfo.s2dealNum    = nActual;
    maskInfo.s2Size       = ctx.s2;
    maskInfo.nBaseSize    = N_BASE;
    maskInfo.preToken     = ctx.preTokens;
    maskInfo.nextToken    = ctx.nextTokens;
    maskInfo.attnMaskBatchStride = ctx.params->attnMaskS1Size * ctx.params->attnMaskS2Size;
    maskInfo.attnMaskS1Stride    = ctx.params->attnMaskS2Size;
    maskInfo.attnMaskDstStride   = (N_BASE - AlignUp(nActual, 32U)) / 32;
    maskInfo.maskValue    = ctx.maskValue;
    maskInfo.layout       = SG;
    maskInfo.attnMaskType = MASK_BOOL;
    maskInfo.sparseMode   = static_cast<SparseMode>(ctx.params->sparseMode);

    bool IsSkipMask = IsSkipAttentionmask(maskInfo);
    if (!IsSkipMask) {
        GlobalTensor<uint8_t> attnMaskGm;
        attnMaskGm.SetGlobalBuffer(ctx.gmPtr,
            ctx.params->attnMaskS1Size * ctx.params->attnMaskS2Size);
        AttentionmaskCopyIn<uint8_t, SG, true, N_BASE>(attnMaskUb.tensor, attnMaskGm, maskInfo);
    } else {
        Duplicate(attnMaskUb.tensor, static_cast<uint8_t>(0U),
                  maskInfo.s1dealNum * AlignUp(maskInfo.s2dealNum, 32U));
    }
}

template <uint32_t M_BASE = 128, uint32_t N_BASE = 128, uint32_t D_SIZE = 128, bool HAS_MASK = false>
__aicore__ inline void ComputeSoftmax(
    Buffer<float> &bmm1,
    Buffer<IN_T>  &stage1,
    Buffer<float> (&sumBuf)[PRELOAD_N + 1],
    Buffer<float> (&maxBuf)[PRELOAD_N + 1],
    Buffer<float> (&expBuf)[PRELOAD_N + 1],
    Buffer<uint8_t> &commonBuf,
    Buffer<uint8_t> &attnMaskUb,
    const AttnMaskContext *maskCtx,
    uint32_t mloop, uint32_t loop,
    bool isFirstS2Loop, bool isUpdate,
    float softmaxScale, float negMin,
    uint32_t mActualHalf, uint32_t nActual)
{
    CrossCoreWaitFlag<4, PIPE_V>(C_BMM1(bmm1.idx));

    LocalTensor<float> sumU = sumBuf[mloop % (PRELOAD_N + 1)].tensor;
    LocalTensor<float> maxU = maxBuf[mloop % (PRELOAD_N + 1)].tensor;
    LocalTensor<float> expU = expBuf[loop % (PRELOAD_N + 1)].tensor;
    LocalTensor<uint8_t> tmp = commonBuf.tensor;

    if constexpr (HAS_MASK) {
        Mutex::Lock<PIPE_MTE2>(E_ATTN_MASK(attnMaskUb.idx));
        LoadAttnMask<M_BASE, N_BASE>(attnMaskUb, *maskCtx, mActualHalf, nActual);
        Mutex::Unlock<PIPE_MTE2>(E_ATTN_MASK(attnMaskUb.idx));
    }

    Mutex::Lock<PIPE_V>(E_STAGE1(stage1.idx));
    Mutex::Lock<PIPE_V>(E_ATTN_MASK(attnMaskUb.idx));
    if (!isUpdate) {
        if (likely(nActual == N_BASE)) {
            ProcessVec1Vf<float, IN_T, false, M_BASE, N_BASE, EQ_128, HAS_MASK>(
                stage1.tensor, nullptr, sumU, maxU, bmm1.tensor, expU, sumU, maxU, attnMaskUb.tensor,
                tmp, mActualHalf, nActual, softmaxScale, negMin);
        } else if (nActual <= 64) {
            ProcessVec1Vf<float, IN_T, false, M_BASE, N_BASE, GT_0_AND_LTE_64, HAS_MASK>(
                stage1.tensor, nullptr, sumU, maxU, bmm1.tensor, expU, sumU, maxU, attnMaskUb.tensor,
                tmp, mActualHalf, nActual, softmaxScale, negMin);
        } else {
            ProcessVec1Vf<float, IN_T, false, M_BASE, N_BASE, GT_64_AND_LTE_128, HAS_MASK>(
                stage1.tensor, nullptr, sumU, maxU, bmm1.tensor, expU, sumU, maxU, attnMaskUb.tensor,
                tmp, mActualHalf, nActual, softmaxScale, negMin);
        }
    } else {
        if (likely(nActual == N_BASE)) {
            ProcessVec1Vf<float, IN_T, true, M_BASE, N_BASE, EQ_128, HAS_MASK>(
                stage1.tensor, nullptr, sumU, maxU, bmm1.tensor, expU, sumU, maxU, attnMaskUb.tensor,
                tmp, mActualHalf, nActual, softmaxScale, negMin);
        } else if (nActual <= 64) {
            ProcessVec1Vf<float, IN_T, true, M_BASE, N_BASE, GT_0_AND_LTE_64, HAS_MASK>(
                stage1.tensor, nullptr, sumU, maxU, bmm1.tensor, expU, sumU, maxU, attnMaskUb.tensor,
                tmp, mActualHalf, nActual, softmaxScale, negMin);
        } else {
            ProcessVec1Vf<float, IN_T, true, M_BASE, N_BASE, GT_64_AND_LTE_128, HAS_MASK>(
                stage1.tensor, nullptr, sumU, maxU, bmm1.tensor, expU, sumU, maxU, attnMaskUb.tensor,
                tmp, mActualHalf, nActual, softmaxScale, negMin);
        }
    }
    Mutex::Unlock<PIPE_V>(E_ATTN_MASK(attnMaskUb.idx));

    if (!isFirstS2Loop) {
        UpdateExpSumAndExpMax<float>(sumU, maxU, expU, sumU, maxU, tmp, mActualHalf);
    }
    Mutex::Unlock<PIPE_V>(E_STAGE1(stage1.idx));
}

template <uint32_t M_BASE = 128, uint32_t N_BASE = 128, uint32_t D_SIZE = 128>
__aicore__ inline void CopySoftmaxToL1(
    Buffer<IN_T> &stage1,
    Buffer<IN_T> &l1pBuf,
    Buffer<float> &bmm1,
    uint32_t subBlockIdx, uint32_t mActual, uint32_t nActual)
{
    Mutex::Lock<PIPE_MTE3>(E_STAGE1(stage1.idx));
    uint32_t myHalfM = (subBlockIdx == 0) ? (mActual + 1) / 2 : mActual / 2;
    if (myHalfM > 0) {
        DataCopy(l1pBuf.tensor[subBlockIdx * (BLOCK_BYTES / sizeof(IN_T)) * (mActual - myHalfM)],
                 stage1.tensor,
                 {static_cast<uint16_t>(N_BASE / 16), static_cast<uint16_t>(myHalfM),
                 static_cast<uint16_t>(M_BASE / 2 + 1 - myHalfM),
                 static_cast<uint16_t>(M_BASE - myHalfM)});
    }
    Mutex::Unlock<PIPE_MTE3>(E_STAGE1(stage1.idx));

    CrossCoreSetFlag<4, PIPE_V>(C_BMM1(bmm1.idx));
    CrossCoreSetFlag<4, PIPE_MTE3>(C_P(l1pBuf.idx));
}

template <uint32_t M_BASE = 128, uint32_t N_BASE = 128, uint32_t D_SIZE = 128>
__aicore__ inline void ComputeRescale(
    Buffer<float> &bmm2,
    Buffer<float> &stage2Buf,
    Buffer<float> (&sumBuf)[PRELOAD_N + 1],
    Buffer<float> (&expBuf)[PRELOAD_N + 1],
    uint32_t mloop, uint32_t loop,
    bool isFirstS2Loop, bool isLastS2Loop,
    uint32_t halfM)
{
    CrossCoreWaitFlag<4, PIPE_V>(C_BMM2(bmm2.idx));

    if (unlikely(isFirstS2Loop)) {
        Mutex::Lock<PIPE_V>(E_STAGE2(stage2Buf.idx));
        DataCopy(stage2Buf.tensor, bmm2.tensor, halfM * (uint32_t)D_SIZE);
    } else {
        LocalTensor<float> expU = expBuf[loop % (PRELOAD_N + 1)].tensor;
        if (!isLastS2Loop) {
            FlashUpdate<float, IN_T, OUT_T, D_SIZE, false>(
                stage2Buf.tensor, bmm2.tensor, stage2Buf.tensor, expU, stage2Buf.tensor,
                halfM, (uint32_t)D_SIZE, 1.0f, 1.0f);
        } else {
            LocalTensor<float> sumU = sumBuf[mloop % (PRELOAD_N + 1)].tensor;
            FlashUpdateLast<float, IN_T, OUT_T, D_SIZE, false>(
                stage2Buf.tensor, bmm2.tensor, stage2Buf.tensor, expU, stage2Buf.tensor, sumU,
                halfM, (uint32_t)D_SIZE, 1.0f, 1.0f);
        }
    }
    CrossCoreSetFlag<4, PIPE_V>(C_BMM2(bmm2.idx));
}

template <uint32_t M_BASE = 128, uint32_t N_BASE = 128, uint32_t D_SIZE = 128>
__aicore__ inline void CopyAttnOutToGm(
    Buffer<float> &stage2Buf, Buffer<OUT_T> &castBuf,
    FaGmTensor<OUT_T, GmFormat::BSND> &outGm,
    Buffer<float> (&sumBuf)[PRELOAD_N + 1],
    uint32_t mloop,
    uint32_t bIdx, uint32_t n1Head,
    uint32_t s1BlockBase, uint32_t subBlockIdx,
    bool isFirstS2Loop, uint32_t mActual)
{
    uint32_t halfM = (subBlockIdx == 0) ? (mActual + 1) / 2 : mActual / 2;
    uint32_t gmRowBase = (subBlockIdx == 0) ? 0 : (mActual + 1) / 2;
    if (isFirstS2Loop) {
        LocalTensor<float> sumU = sumBuf[mloop % (PRELOAD_N + 1)].tensor;
        LastDiv<float, IN_T, OUT_T, (uint32_t)D_SIZE>(stage2Buf.tensor, stage2Buf.tensor, sumU, halfM, (uint32_t)D_SIZE, 0.0f);
    }

    Cast(castBuf.tensor, stage2Buf.tensor, RoundMode::CAST_ROUND, halfM * (uint32_t)D_SIZE);
    Mutex::Unlock<PIPE_V>(E_STAGE2(stage2Buf.idx));
    Mutex::Lock<PIPE_MTE3>(E_STAGE2(stage2Buf.idx));

    FaUbTensor<OUT_T> ubTensor{.tensor = castBuf.tensor,
                               .rowCount = halfM,
                               .colCount = (uint32_t)D_SIZE};
    CopyAttnOutBN1UbToGm<OUT_T>(outGm, ubTensor, bIdx, n1Head,
                                s1BlockBase + gmRowBase);

    Mutex::Unlock<PIPE_MTE3>(E_STAGE2(stage2Buf.idx));
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

static constexpr uint32_t VEC_CTX_RING_SIZE = PRELOAD_N + 1;

template <uint32_t M_BASE = 128, uint32_t N_BASE = 128, uint32_t D_SIZE = 128>
__aicore__ inline void VectorFunc(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                  __gm__ uint8_t *value, __gm__ uint8_t *attnMask,
                                  __gm__ uint8_t *attentionOut,
                                  __gm__ uint8_t *tilingGM,
                                  const optiling::FaBaseParams &faBaseParams,
                                  const optiling::FaAttnMaskParams &faAttnMaskParams,
                                  const optiling::FaMetaData &faMetaData,
                                  uint32_t aivIdx)
{
    static constexpr uint32_t UB_BMM1_ELEMS   = M_BASE / 2 * N_BASE;
    static constexpr uint32_t UB_BMM2_ELEMS   = M_BASE / 2 * D_SIZE;
    static constexpr uint32_t UB_STAGE1_ELEMS = (M_BASE / 2 + 2) * N_BASE;
    static constexpr uint32_t UB_STAGE2_ELEMS = M_BASE / 2 * D_SIZE;
    static constexpr uint32_t UB_ATTN_MASK_ELEMS = M_BASE / 2 * N_BASE;
    static constexpr uint32_t L1_P_ELEMS      = M_BASE * N_BASE;

    uint32_t B   = faBaseParams.bSize;
    uint32_t N1  = faBaseParams.n1Size;
    uint32_t N2  = faBaseParams.n2Size;
    uint32_t S1  = faBaseParams.s1Size;
    uint32_t S2  = faBaseParams.s2Size;
    float softmaxScale = faBaseParams.softmaxScale;

    bool hasAttnMask = (faAttnMaskParams.sparseMode != 0);
    int64_t preTokens = faAttnMaskParams.preTokens;
    int64_t nextTokens = faAttnMaskParams.nextTokens;

    uint32_t aicIdx = aivIdx / CV_RATIO;
    uint32_t subBlockIdx = GetSubBlockIdx();

    uint32_t bn1Start = faMetaData.FAMetadata[aicIdx][optiling::FA_BN1_START_INDEX];
    uint32_t s1OStart  = faMetaData.FAMetadata[aicIdx][optiling::FA_S1_START_INDEX];
    uint32_t bn1End   = faMetaData.FAMetadata[aicIdx][optiling::FA_BN1_END_INDEX];
    uint32_t s1OEnd   = faMetaData.FAMetadata[aicIdx][optiling::FA_S1_END_INDEX];
    uint32_t coreEnable = faMetaData.FAMetadata[aicIdx][optiling::FA_CORE_ENABLE_INDEX];
    if (!coreEnable) {
        return;
    }

    uint32_t numS1Blocks = (S1 + M_BASE - 1) / M_BASE;

    FaGmTensor<OUT_T, GmFormat::BSND> outGm((__gm__ OUT_T *)attentionOut, B, N1, S1, (uint32_t)D_SIZE);

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

    uint32_t l1Addr = 0;
    Buffer<IN_T> l1P[3] = {
        {LocalTensor<IN_T>(TPosition::A1, l1Addr, L1_P_ELEMS), 0},
        {LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_P_ELEMS * sizeof(IN_T)), L1_P_ELEMS), 1},
        {LocalTensor<IN_T>(TPosition::A1, l1Addr += (L1_P_ELEMS * sizeof(IN_T)), L1_P_ELEMS), 2}
    };

    Buffer<float> softmaxSum[PRELOAD_N + 1] = {
        {LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_BMM2_ELEMS * sizeof(float)), UB_SOFTMAX_ELEMS), 0},
        {LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_SOFTMAX_ELEMS * sizeof(float)), UB_SOFTMAX_ELEMS), 1},
        {LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_SOFTMAX_ELEMS * sizeof(float)), UB_SOFTMAX_ELEMS), 2}
    };
    Buffer<float> softmaxMax[PRELOAD_N + 1] = {
        {LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_SOFTMAX_ELEMS * sizeof(float)), UB_SOFTMAX_ELEMS), 0},
        {LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_SOFTMAX_ELEMS * sizeof(float)), UB_SOFTMAX_ELEMS), 1},
        {LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_SOFTMAX_ELEMS * sizeof(float)), UB_SOFTMAX_ELEMS), 2}
    };
    Buffer<float> softmaxExp[PRELOAD_N + 1] = {
        {LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_SOFTMAX_ELEMS * sizeof(float)), UB_SOFTMAX_ELEMS), 0},
        {LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_SOFTMAX_ELEMS * sizeof(float)), UB_SOFTMAX_ELEMS), 1},
        {LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_SOFTMAX_ELEMS * sizeof(float)), UB_SOFTMAX_ELEMS), 2}
    };
    Buffer<uint8_t> commonBuf = {
        LocalTensor<uint8_t>(TPosition::VECIN, ubAddr += (UB_SOFTMAX_ELEMS * sizeof(float)), UB_COMMON_ELEMS),
        0
    };

    Buffer<IN_T> stage1Buf[PRELOAD_N + 1] = {
        {LocalTensor<IN_T>(TPosition::VECIN, ubAddr += (UB_COMMON_ELEMS * sizeof(uint8_t)), UB_STAGE1_ELEMS), 0},
        {LocalTensor<IN_T>(TPosition::VECIN, ubAddr += (UB_STAGE1_ELEMS * sizeof(IN_T)), UB_STAGE1_ELEMS), 1},
        {LocalTensor<IN_T>(TPosition::VECIN, ubAddr += (UB_STAGE1_ELEMS * sizeof(IN_T)), UB_STAGE1_ELEMS), 2}
    };

    Buffer<float> stage2Buf = {
        LocalTensor<float>(TPosition::VECIN, ubAddr += (UB_STAGE1_ELEMS * sizeof(IN_T)), UB_STAGE2_ELEMS),
        0
    };
    Buffer<IN_T> castBuf = {
        LocalTensor<IN_T>(TPosition::VECIN, ubAddr, UB_STAGE2_ELEMS),
        0
    };
    Buffer<uint8_t> attnMaskUb = {
        LocalTensor<uint8_t>(TPosition::VECIN, ubAddr + (UB_STAGE2_ELEMS * sizeof(float)), UB_ATTN_MASK_ELEMS),
        0
    };

    CrossCoreSetFlag<4, PIPE_V>(0);
    CrossCoreSetFlag<4, PIPE_V>(1);
    CrossCoreSetFlag<4, PIPE_V>(2);
    CrossCoreSetFlag<4, PIPE_V>(3);
    CrossCoreSetFlag<4, PIPE_V>(4);

    VecBmm2Ctx vecCtxRing[VEC_CTX_RING_SIZE];

    uint32_t tmpNeg = NEGATIVE_MIN_VALUE_FP32;
    float negMin = *((float *)&tmpNeg);

    uint32_t effectiveBn1End = bn1End;
    if (s1OEnd > 0) {
        effectiveBn1End++;
    }

    uint32_t i = 0;
    uint32_t mloop = 0;

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
            uint32_t myHalfM = (subBlockIdx == 0) ? (mActual + 1) / 2 : mActual / 2;

            uint32_t s2StartBlock = 0, s2EndBlock = 0;
            CalcCurS2Range(s2StartBlock, s2EndBlock, s1BlockBase, M_BASE,
                S1, S2, preTokens, nextTokens, hasAttnMask, N_BASE);

            if (s2StartBlock >= s2EndBlock) {
                continue;
            }

            for (uint32_t s2Block = s2StartBlock; s2Block < s2EndBlock; s2Block++) {
                uint32_t s2BlockBase = s2Block * N_BASE;
                uint32_t nActual = s2BlockBase + N_BASE > S2 ? S2 - s2BlockBase : (uint32_t)N_BASE;
                bool isFirstS2Loop = (s2Block == s2StartBlock);
                bool isLastS2Loop = (s2Block + 1 == s2EndBlock);

                uint32_t bmm1Idx       = i % (PRELOAD_N + 1);
                uint32_t stage1BufIdx  = i % (PRELOAD_N + 1);
                uint32_t l1PIdx        = i % 3;
                uint32_t ringSlot      = i % VEC_CTX_RING_SIZE;

                if (isFirstS2Loop) {
                    mloop++;
                }

                vecCtxRing[ringSlot].bIdx = bIdx;
                vecCtxRing[ringSlot].n1Head = n1Head;
                vecCtxRing[ringSlot].s1BlockBase = s1BlockBase;
                vecCtxRing[ringSlot].isFirstS2Loop = isFirstS2Loop;
                vecCtxRing[ringSlot].isLastS2Loop = isLastS2Loop;
                vecCtxRing[ringSlot].mloopIdx = mloop - 1;
                vecCtxRing[ringSlot].mActual = mActual;

                if (hasAttnMask) {
                    AttnMaskContext maskCtx;
                    maskCtx.gmPtr       = attnMask;
                    maskCtx.params      = &faAttnMaskParams;
                    maskCtx.s1BlockBase = s1BlockBase;
                    maskCtx.subBlockIdx = subBlockIdx;
                    maskCtx.s2BlockBase = s2BlockBase;
                    maskCtx.s1          = S1;
                    maskCtx.s2          = S2;
                    maskCtx.preTokens   = preTokens;
                    maskCtx.nextTokens  = nextTokens;
                    maskCtx.maskValue   = tmpNeg;
                    maskCtx.mActual     = mActual;

                    ComputeSoftmax<M_BASE, N_BASE, D_SIZE, true>(ubBmm1[bmm1Idx], stage1Buf[stage1BufIdx],
                        softmaxSum, softmaxMax, softmaxExp, commonBuf, attnMaskUb, &maskCtx,
                        mloop - 1, i,
                        isFirstS2Loop, !isFirstS2Loop,
                        softmaxScale, negMin, myHalfM, nActual);
                } else {
                    ComputeSoftmax<M_BASE, N_BASE, D_SIZE, false>(ubBmm1[bmm1Idx], stage1Buf[stage1BufIdx],
                        softmaxSum, softmaxMax, softmaxExp, commonBuf, attnMaskUb, nullptr,
                        mloop - 1, i,
                        isFirstS2Loop, !isFirstS2Loop,
                        softmaxScale, negMin, myHalfM, nActual);
                }

                CopySoftmaxToL1<M_BASE, N_BASE, D_SIZE>(stage1Buf[stage1BufIdx], l1P[l1PIdx],
                    ubBmm1[bmm1Idx],
                    subBlockIdx, mActual, nActual);

                if (i >= PRELOAD_N) {
                    uint32_t j = i - PRELOAD_N;
                    uint32_t bmm2Idx = j & 1;
                    VecBmm2Ctx &ctx = vecCtxRing[j % VEC_CTX_RING_SIZE];
                    uint32_t ctxHalfM = (subBlockIdx == 0) ? (ctx.mActual + 1) / 2 : ctx.mActual / 2;

                    ComputeRescale<M_BASE, N_BASE, D_SIZE>(ubBmm2[bmm2Idx], stage2Buf,
                        softmaxSum, softmaxExp,
                        ctx.mloopIdx, j,
                        ctx.isFirstS2Loop, ctx.isLastS2Loop,
                        ctxHalfM);

                    if (ctx.isLastS2Loop) {
                        CopyAttnOutToGm<M_BASE, N_BASE, D_SIZE>(stage2Buf, castBuf, outGm,
                            softmaxSum, ctx.mloopIdx,
                            ctx.bIdx, ctx.n1Head,
                            ctx.s1BlockBase, subBlockIdx,
                            ctx.isFirstS2Loop, ctx.mActual);
                    }
                }
                i++;
            }
        }
    }

    uint32_t total = i;
    uint32_t drainCount = (total > PRELOAD_N) ? PRELOAD_N : total;
    for (uint32_t k = 0; k < drainCount; k++) {
        uint32_t j = total - drainCount + k;
        uint32_t bmm2Idx = j & 1;
        VecBmm2Ctx &ctx = vecCtxRing[j % VEC_CTX_RING_SIZE];
        uint32_t ctxHalfM = (subBlockIdx == 0) ? (ctx.mActual + 1) / 2 : ctx.mActual / 2;

        ComputeRescale<M_BASE, N_BASE, D_SIZE>(ubBmm2[bmm2Idx], stage2Buf,
            softmaxSum, softmaxExp,
            ctx.mloopIdx, j,
            ctx.isFirstS2Loop, ctx.isLastS2Loop,
            ctxHalfM);

        if (ctx.isLastS2Loop) {
            CopyAttnOutToGm<M_BASE, N_BASE, D_SIZE>(stage2Buf, castBuf, outGm,
                softmaxSum, ctx.mloopIdx,
                ctx.bIdx, ctx.n1Head,
                ctx.s1BlockBase, subBlockIdx,
                ctx.isFirstS2Loop, ctx.mActual);
        }
    }
}

} // namespace FaVector

#endif // __DAV_C310_CUBE__
#endif // FA_BLOCK_VEC_H
