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
 * \file flash_attention_kernel_noquant_mla.h
 * \brief
 */

#ifndef FLASH_ATTENTION_KERNEL_NOQUANT_MLA_H_
#define FLASH_ATTENTION_KERNEL_NOQUANT_MLA_H_

#include "attenmask.h"
#include "pse.h"
#include "flash_attention_block_cube_noquant_mla.h"
#include "flash_attention_noquant_block_vec_infer.h"

using namespace fa_base_matmul;

template<bool isInfer, bool hasRope>
struct FAContext : public ConstInfo<isInfer, hasRope> {
    AttenMaskInfo attenMask;
    PseInfo pse;
};

template <typename CubeBlockType, typename VecBlockType>
class FAKernelNoquantMla {
public:
    ARGS_TRAITS;
    __aicore__ inline FAKernelNoquantMla() {};
    __aicore__ inline void Run(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                               __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                               __gm__ uint8_t *blockTable, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
                               __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                               __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                               const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe);

private:
    __aicore__ inline void InitInput(FAContext<isInfer, hasRope> &ctx,
                                     __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                     __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                                     __gm__ uint8_t *blockTable, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
                                     __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                     __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace);
    __aicore__ inline void InitSyncFlags();
    __aicore__ inline void ComputeConstexpr(FAContext<isInfer, hasRope> &ctx);
    __aicore__ inline void DoProcess(FAContext<isInfer, hasRope> &ctx);
    __aicore__ inline void SetRunInfo(FAContext<isInfer, hasRope> &ctx, RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(FAContext<isInfer, hasRope> &ctx, int64_t bnIndx, int64_t gS1Index, int64_t &multiCoreInnerIdx, RunParamStr<isInfer>& runParam);
    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvLen);
    __aicore__ inline void ComputeBmm1Tail(FAContext<isInfer, hasRope> &ctx, RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam);
    __aicore__ inline bool IsLastBN(FAContext<isInfer, hasRope> &ctx, uint32_t bnStartIdx, uint32_t bnEndIdx);

    // Buffer access: 3-rotation for mm12Right L1
    __aicore__ inline LocalTensor<uint8_t> GetMm12RightL1() {
        uint8_t idx = mm12rIdx_;
        mm12rIdx_ = (mm12rIdx_ + 1 >= 3) ? 0 : mm12rIdx_ + 1;
        return l1Base_[L1_MM12R[idx]];
    }
    __aicore__ inline LocalTensor<uint8_t> GetMm12RightL1Reused() {
        uint8_t idx = mm12rReusedIdx_;
        mm12rReusedIdx_ = (mm12rReusedIdx_ + 1 >= 3) ? 0 : mm12rReusedIdx_ + 1;
        return l1Base_[L1_MM12R[idx]];
    }
    __aicore__ inline uint32_t GetMm12RightCCId() {
        uint8_t prev = (mm12rIdx_ == 0) ? 2 : mm12rIdx_ - 1;
        return CC_L1[prev];
    }
    __aicore__ inline uint32_t GetMm12RightReusedCCId() {
        uint8_t prev = (mm12rReusedIdx_ == 0) ? 2 : mm12rReusedIdx_ - 1;
        return CC_L1[prev];
    }
    __aicore__ inline TEventID GetMm12RightP2CId() {
        uint8_t prev = (mm12rIdx_ == 0) ? 2 : mm12rIdx_ - 1;
        return L1_EV_P2C[prev];
    }
    __aicore__ inline TEventID GetMm12RightC2PId() {
        uint8_t prev = (mm12rIdx_ == 0) ? 2 : mm12rIdx_ - 1;
        return L1_EV_C2P[prev];
    }

    __aicore__ inline LocalTensor<uint8_t> GetBmm1Ub() {
        uint8_t idx = bmm1Idx_;
        bmm1Idx_ ^= 1;
        return ubBase_[UB_BMM1[idx]];
    }
    __aicore__ inline uint32_t GetBmm1FwdId() { return CC_BMM1_FWD[bmm1Idx_ ^ 1]; }
    __aicore__ inline uint32_t GetBmm1BwdId() { return CC_BMM1_BWD[bmm1Idx_ ^ 1]; }

    __aicore__ inline LocalTensor<uint8_t> GetBmm2Ub() {
        uint8_t idx = bmm2Idx_;
        bmm2Idx_ ^= 1;
        return ubBase_[UB_BMM2[idx]];
    }
    __aicore__ inline uint32_t GetBmm2FwdId() { return CC_BMM2_FWD[bmm2Idx_ ^ 1]; }
    __aicore__ inline uint32_t GetBmm2BwdId() { return CC_BMM2_BWD[bmm2Idx_ ^ 1]; }

    __aicore__ inline Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> MakeL1Buffer(
        LocalTensor<uint8_t> tensor, uint32_t ccId) {
        return Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>(
            tensor, MM12R_SIZE, ccId, INVALID_CROSS_CORE_EVENT_ID, 0, 0);
    }
    __aicore__ inline Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> MakeUbBuffer(
        LocalTensor<uint8_t> tensor, uint32_t size, uint32_t fwdId, uint32_t bwdId) {
        return Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>(
            tensor, size, fwdId, bwdId, 0, 0);
    }

private:
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = nullptr;
    TPipe *tPipe = nullptr;

    static constexpr uint32_t dTemplateAlign64 = CubeBlockType::dTemplateAlign64;
    static constexpr uint32_t s1BaseSize = CubeBlockType::s1BaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr bool splitD = CubeBlockType::splitD;
    static constexpr uint32_t PRELOAD_N = 1;
    static constexpr bool POST_QUANT = VecBlockType::POST_QUANT;

    /* ============ Constexpr buffer sizes ============ */
    static constexpr uint32_t MM12R_SIZE =
        (static_cast<uint32_t>(dTemplateType) > static_cast<uint32_t>(dVTemplateType)
            ? static_cast<uint32_t>(dTemplateType) : static_cast<uint32_t>(dVTemplateType))
        * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t MM1A_SIZE = static_cast<uint32_t>(dTemplateType) * s1BaseSize * 2;
    static constexpr uint32_t MM1_RES_SIZE = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    static constexpr uint32_t MM2_RES_SIZE = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    static constexpr uint32_t L1_TOTAL = MM12R_SIZE * 3 + MM1A_SIZE;
    static constexpr uint32_t UB_TOTAL = MM2_RES_SIZE * 2 + MM1_RES_SIZE * 2;

    /* ============ Static L1 offsets ============ */
    static constexpr uint32_t L1_MM12R[3] = {0, MM12R_SIZE, MM12R_SIZE * 2};
    static constexpr uint32_t L1_MM1A_OFF = MM12R_SIZE * 3;

    /* ============ Static UB offsets ============ */
    static constexpr uint32_t UB_BMM2[2] = {0, MM2_RES_SIZE};
    static constexpr uint32_t UB_BMM1[2] = {MM2_RES_SIZE * 2, MM2_RES_SIZE * 2 + MM1_RES_SIZE};

    /* ============ Static cross-core event IDs ============ */
    static constexpr uint32_t CC_L1[3] = {0, 1, 2};
    static constexpr uint32_t CC_BMM1_FWD[2] = {3, 5};
    static constexpr uint32_t CC_BMM1_BWD[2] = {4, 6};
    static constexpr uint32_t CC_BMM2_FWD[2] = {7, 9};
    static constexpr uint32_t CC_BMM2_BWD[2] = {8, 10};

    /* ============ Static inner-core event IDs (per HardEvent type) ============ */
    static constexpr TEventID L1_EV_P2C[4] = {0, 1, 2, 3};
    static constexpr TEventID L1_EV_C2P[4] = {0, 1, 2, 3};

    /* ============ Static raw memory (no TBuf/TPipe) ============ */
    LocalTensor<uint8_t> l1Base_{TPosition::A1, 0, L1_TOTAL};
    LocalTensor<uint8_t> ubBase_{TPosition::VECIN, 0, UB_TOTAL};

    /* ============ Rotation state ============ */
    uint8_t mm12rIdx_ = 0;
    uint8_t mm12rReusedIdx_ = 0;
    uint8_t bmm1Idx_ = 0;
    uint8_t bmm2Idx_ = 0;

    /* Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;

    /* GM pointers */
    __gm__ uint8_t *currentKey;
    __gm__ uint8_t *currentValue;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> valueGm;

    static constexpr bool useDn = false;

    int32_t aicIdx;

    __gm__ int64_t *actualSeqQlenAddr;
    __gm__ int64_t *actualSeqKvlenAddr;
    uint64_t s1SizeAcc = 0;
    uint64_t s2SizeAcc = 0;
    uint64_t b1SSOffset = 0;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::Run(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipeArg)
{
    FAContext<isInfer, hasRope> ctx;

    this->tilingData = tiling;
    this->tPipe = tPipeArg;
    ctx.aivIdx = GetBlockIdx();
    ctx.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIV {
        this->aicIdx = ctx.aivIdx >> 1;
    }
    if ASCEND_IS_AIC {
        this->aicIdx = ctx.aivIdx;
    }

    CVSharedParams<isInfer, isPa> sharedParams;
    this->vecBlock.InitVecBlock(this->tPipe, this->tilingData, sharedParams, this->aicIdx, ctx.subBlockIdx,
        ctx.attenMask, ctx.pse);
    this->vecBlock.CleanOutput(softmaxLse, attentionOut, ctx);

    InitSyncFlags();

    ListTensorDesc keyListTensorDescInit((__gm__ void*)key);
    ListTensorDesc valueListTensorDescInit((__gm__ void*)value);
    currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
    currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);

    if ASCEND_IS_AIC {
        this->cubeBlock.InitCubeBlock(this->tPipe, query, currentKey, currentValue,
            blockTable, queryRope, keyRope, tiling,
            l1Base_[L1_MM1A_OFF], MM1A_SIZE,
            this->tilingData->inputParamsRegbase.isKvContinuous == 1, key, value);
    }

    ComputeConstexpr(ctx);
    ctx.isPfaGS1Merge = (ctx.isGqa && ctx.s1Size > 1);

    InitInput(ctx, query, key, value, pse, attenMask, actualSeqLengths, actualSeqLengthsKv, blockTable,
        postQuantScale, postQuantOffset, queryRope, keyRope, softmaxLse, attentionOut, workspace);

    if ASCEND_IS_AIV {
        this->vecBlock.InitLocalBuffer(this->tPipe, ctx);
    }

    DoProcess(ctx);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::InitSyncFlags()
{
    if ASCEND_IS_AIC {
        SetFlag<HardEvent::MTE1_MTE2>(L1_EV_C2P[0]);
        SetFlag<HardEvent::MTE1_MTE2>(L1_EV_C2P[1]);
        SetFlag<HardEvent::MTE1_MTE2>(L1_EV_C2P[2]);
    }
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_BWD[0]);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_BWD[1]);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_BWD[0]);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_BWD[1]);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::InitInput(
    FAContext<isInfer, hasRope> &ctx,
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
{
    if (this->tilingData->inputParamsRegbase.isKvContinuous == 1) {
        this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)currentKey);
        this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)currentValue);
    } else {
        this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
        this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    }

    if (this->tilingData->inputParamsRegbase.isActualSeqLengthsNull != 1) {
        actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
    }
    if (this->tilingData->inputParamsRegbase.isActualSeqLengthsKVNull != 1) {
        actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
    }

    this->vecBlock.InitGlobalBuffer(pse, nullptr, nullptr, nullptr, nullptr, postQuantScale, postQuantOffset,
        nullptr, attenMask, nullptr, nullptr, nullptr, nullptr, nullptr, workspace, 0, this->aicIdx, ctx);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::ComputeConstexpr(
    FAContext<isInfer, hasRope> &ctx)
{
    ctx.s1BaseSize = s1BaseSize;
    ctx.s2BaseSize = s2BaseSize;

    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;

    ctx.bSize = inputParamsRegbase.bSize;
    ctx.t1Size = inputParamsRegbase.t1Size;
    ctx.n2Size = inputParamsRegbase.n2Size;
    ctx.s1Size = inputParamsRegbase.s1Size;
    ctx.s2Size = inputParamsRegbase.s2Size;
    ctx.dSize = inputParamsRegbase.dSize;
    ctx.dSizeV = inputParamsRegbase.dSizeV;
    ctx.dBasicBlock = Align64Func((uint16_t)ctx.dSizeV);
    if constexpr (hasRope) {
        ctx.dSizeRope = inputParamsRegbase.dSizeRope;
    } else {
        ctx.dSizeRope = 0;
    }
    ctx.gSize = inputParamsRegbase.gSize;
    if (inputParamsRegbase.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_NBSD) ||
        inputParamsRegbase.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSND_NBSD) ||
        inputParamsRegbase.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSH_NBSD)) {
        ctx.t1Size = ctx.bSize * ctx.s1Size;
    }
    ctx.s1OuterSize = this->tilingData->multiCoreParamsRegbase.s1OuterSize;

    int64_t dSize = ctx.dSize;
    int64_t dSizeV = ctx.dSizeV;
    int64_t s1Size = ctx.s1Size;
    int64_t s2Size = ctx.s2Size;
    int64_t gSize = ctx.gSize;
    int64_t n2Size = ctx.n2Size;
    int64_t bSize = inputParamsRegbase.bSize;

    ctx.s1D = s1Size * dSize;
    ctx.s2D = s2Size * dSize;
    ctx.gD = gSize * dSize;
    ctx.n2D = n2Size * dSize;
    ctx.s1Dv = s1Size * dSizeV;
    ctx.s2Dv = s2Size * dSizeV;
    ctx.gDv = gSize * dSizeV;
    ctx.n2Dv = n2Size * dSizeV;
    ctx.s1S2 = s1Size * s2Size;
    ctx.gS1 = gSize * s1Size;
    ctx.n2G = n2Size * gSize;

    ctx.bN2D = bSize * ctx.n2D;
    ctx.bN2Dv = bSize * ctx.n2Dv;
    ctx.gS1D = gSize * ctx.s1D;
    ctx.n2S2D = n2Size * ctx.s2D;
    ctx.n2GD = n2Size * ctx.gD;
    ctx.bN2GD = bSize * ctx.n2GD;
    ctx.gS1Dv = gSize * ctx.s1Dv;
    ctx.n2S2Dv = n2Size * ctx.s2Dv;
    ctx.n2GDv = n2Size * ctx.gDv;
    ctx.bN2GDv = bSize * ctx.n2GDv;
    ctx.n2GS1D = n2Size * ctx.gS1D;
    ctx.n2GS1Dv = n2Size * ctx.gS1Dv;

    ctx.s2BaseN2D = s2BaseSize * ctx.n2D;
    ctx.s2BaseN2Dv = s2BaseSize * ctx.n2Dv;
    ctx.n2S2D /= inputParamsRegbase.headNumRatio;
    ctx.n2S2Dv /= inputParamsRegbase.headNumRatio;
    ctx.s2BaseN2D /= inputParamsRegbase.headNumRatio;
    ctx.s2BaseN2Dv /= inputParamsRegbase.headNumRatio;

    if constexpr (hasRope) {
        int64_t dSizeRope = ctx.dSizeRope;
        ctx.s1DR = s1Size * dSizeRope;
        ctx.s2DR = s2Size * dSizeRope;
        ctx.gDR = gSize * dSizeRope;
        ctx.n2DR = n2Size * dSizeRope;
        ctx.bN2DR = bSize * ctx.n2DR;
        ctx.gS1DR = gSize * ctx.s1DR;
        ctx.n2S2DR = n2Size * ctx.s2DR;
        ctx.n2GDR = n2Size * ctx.gDR;
        ctx.bN2GDR = bSize * ctx.n2GDR;
        ctx.n2GS1DR = n2Size * ctx.gS1DR;
        ctx.s2BaseN2DR = s2BaseSize * ctx.n2DR;
    }
    ctx.layoutType = inputParamsRegbase.layoutType;
    ctx.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        ctx.s1BaseN2GD = s1BaseSize * ctx.n2GD;
        ctx.s1BaseN2GDv = s1BaseSize * ctx.n2GDv;
        if constexpr (hasRope) {
            ctx.s1BaseDR = s1BaseSize * ctx.dSizeRope;
            ctx.mm1RopeKa = ctx.dSizeRope;
            ctx.mm1RopeKb = ctx.n2DR;
        }
        ctx.mm1Ka = dSize;
        ctx.mm1Kb = ctx.n2D;
        ctx.mm2Kb = ctx.n2Dv;
        ctx.mm1Kb /= inputParamsRegbase.headNumRatio;
        ctx.mm2Kb /= inputParamsRegbase.headNumRatio;
        ctx.attentionOutStride = 0;
    } else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            ctx.s1BaseN2GD = s1BaseSize * ctx.n2GD;
            ctx.s1BaseN2GDv = s1BaseSize * ctx.n2GDv;
            if constexpr (hasRope) {
                ctx.s1BaseDR = s1BaseSize * ctx.dSizeRope;
                ctx.mm1RopeKa = ctx.dSizeRope;
                ctx.mm1RopeKb = ctx.n2DR;
            }
            ctx.mm1Ka = dSize;
            ctx.mm1Kb = ctx.n2D;
            ctx.mm2Kb = ctx.n2Dv;
            ctx.mm1Kb /= inputParamsRegbase.headNumRatio;
            ctx.mm2Kb /= inputParamsRegbase.headNumRatio;
            ctx.attentionOutStride = 0;
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            ctx.s1BaseD = s1BaseSize * dSize;
            ctx.s2BaseD = s2BaseSize * dSize;
            ctx.s1BaseDv = s1BaseSize * dSizeV;
            ctx.s2BaseDv = s2BaseSize * dSizeV;
            if constexpr (hasRope) {
                ctx.s1BaseDR = s1BaseSize * ctx.dSizeRope;
                ctx.s2BaseDR = s2BaseSize * ctx.dSizeRope;
                ctx.mm1RopeKa = ctx.dSizeRope;
                ctx.mm1RopeKb = ctx.dSizeRope;
            }
            ctx.mm1Ka = dSize;
            ctx.mm1Kb = dSize;
            ctx.mm2Kb = dSizeV;
            ctx.attentionOutStride = 0;
        }
    }

    if constexpr (hasAtten) {
        ctx.attenMask.preTokens = inputParamsRegbase.preTokens;
        ctx.attenMask.nextTokens = inputParamsRegbase.nextTokens;
        ctx.attenMask.compressMode = inputParamsRegbase.attenMaskCompressMode;
        ctx.attenMask.attenMaskShapeType = inputParamsRegbase.attenMaskShapeType;
        ctx.attenMask.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
        ctx.attenMask.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
        ctx.attenMask.bandIndex = inputParamsRegbase.bandIndex;
    }

    ctx.isRowInvalid = inputParamsRegbase.isRowInvalid;
    ctx.headNumRatio = inputParamsRegbase.headNumRatio;
    ctx.dSizeRope = inputParamsRegbase.ropeHeadSize;
    ctx.isGqa = inputParamsRegbase.isGqa;
    ctx.n2GDR = gSize * n2Size * ctx.dSizeRope;
    ctx.n2DR = n2Size * ctx.dSizeRope;
    ctx.isKvContinuous = inputParamsRegbase.isKvContinuous;
    ctx.actualSeqLenSize = inputParamsRegbase.actualSeqLengthsSize;
    ctx.actualSeqLenKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
    ctx.isActualLenDimsNull = (inputParamsRegbase.isActualSeqLengthsNull == 1) ? true : false;
    ctx.isActualLenDimsKVNull = (inputParamsRegbase.isActualSeqLengthsKVNull == 1) ? true : false;
    ctx.isQHasLeftPadding = (inputParamsRegbase.isQHasLeftPadding == 1) ? true : false;
    ctx.isKVHasLeftPadding = (inputParamsRegbase.isKVHasLeftPadding == 1) ? true : false;
    ctx.blockTableDim2 = inputParamsRegbase.blockTableDim2;
    ctx.blockSize = inputParamsRegbase.blockSize;
    ctx.paLayoutType = inputParamsRegbase.paLayoutType;
    ctx.paBlockNumSum = inputParamsRegbase.paBlockNumSum;

    ctx.transposeLayout = inputParamsRegbase.transposeLayout;
    if (ctx.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_BSND)) {
        ctx.attentionOutStride =
            (n2Size * gSize - 1) * dSizeV * sizeof(OUTPUT_T);
    } else if (ctx.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::TND_NTD)) {
        ctx.attentionOutStride = 0;
    }

    ctx.isSoftmaxLseEnable = inputParamsRegbase.isSoftMaxLseEnable;

    if constexpr (POST_QUANT) {
        ctx.isPostQuantPerChnl = inputParamsRegbase.isPostQuantPerChnl;
        ctx.isPostQuantBF16 = inputParamsRegbase.isPostQuantBF16;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::DoProcess(
    FAContext<isInfer, hasRope> &ctx)
{
    int32_t actualCoreNums = this->tilingData->multiCoreParamsRegbase.coreNum;
    if (aicIdx >= actualCoreNums) {
        return;
    }

    int64_t gS1StartIdx = 0;
    int64_t gS1EndIdx = 1;
    uint32_t bnStartIdx = 0;
    uint32_t bnEndIdx = 1;
    int64_t s2LoopStart = 0;
    int64_t s2LoopLimit = 0;

    if constexpr (!isFd) {
        bnStartIdx = this->tilingData->multiCoreParamsRegbase.bnStartIdx[aicIdx];
        gS1StartIdx = this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx];
        if (likely((this->tilingData->multiCoreParamsRegbase.coreNum - 1) > aicIdx)) {
            bnEndIdx = this->tilingData->multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];
            if (this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1] != 0) {
                bnEndIdx++;
            }
        } else {
            bnEndIdx = this->tilingData->inputParamsRegbase.bSize * ctx.n2Size;
        }
    }
    int64_t taskId = 0;
    bool isLastBmm1 = false;
    RunInfo<isInfer> runInfo[NUM_4];
    RunParamStr<isInfer> runParam;

    int64_t multiCoreInnerIdx = 0;
    for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; bnIdx++) {
        bool lastBN = IsLastBN(ctx, bnIdx, bnEndIdx);
        if constexpr (!isFd) {
            runParam.boIdx = bnIdx / ctx.n2Size;
            runParam.n2oIdx = bnIdx % ctx.n2Size;
        }
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, ctx, ctx.attenMask,
            keyGm, actualSeqQlenAddr, actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, ctx, lastBN,
            this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1]);

        gS1EndIdx = runParam.s1LoopTimes;
        for (int64_t gS1Index = gS1StartIdx; gS1Index <runParam.s1LoopTimes; gS1Index++) {
            s2LoopLimit = 0;
            this->ComputeAxisIdxByBnAndGs1(ctx, bnIdx, gS1Index, multiCoreInnerIdx, runParam);
            bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, ctx,
                gS1Index, actualSeqQlenAddr, ctx.pse);
            bool s2NoNeedCalc = ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, ctx);
            bool lastLoopThisCore = lastBN && (gS1Index == runParam.s1LoopTimes - 1);
            bool lastBnNoNeedCalc = ComputeLastBN<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam,
                actualSeqQlenAddr);
            if (((s1NoNeedCalc || s2NoNeedCalc) && !lastLoopThisCore) || lastBnNoNeedCalc) {
                continue;
            }
            s2LoopLimit = runParam.s2LoopEndIdx - 1;
            if (lastLoopThisCore) {
                isLastBmm1 = true;
                s2LoopLimit += PRELOAD_N;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
                if (s2LoopCount < runParam.s2LoopEndIdx) {
                    RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
                    this->SetRunInfo(ctx, runInfo1, runParam, taskId, s2LoopCount, runParam.s2LoopEndIdx - 1,
                        multiCoreInnerIdx);

                    if ASCEND_IS_AIC {
                        LocalTensor<uint8_t> mm12rTensor = GetMm12RightL1();
                        uint32_t mm12rCCId = GetMm12RightCCId();
                        TEventID mm12rP2CId = GetMm12RightP2CId();
                        TEventID mm12rC2PId = GetMm12RightC2PId();

                        LocalTensor<uint8_t> bmm1Tensor = GetBmm1Ub();
                        uint32_t bmm1FwdId = GetBmm1FwdId();
                        uint32_t bmm1BwdId = GetBmm1BwdId();

                        this->cubeBlock.IterateBmm1(bmm1Tensor, bmm1FwdId, bmm1BwdId,
                            mm12rTensor, mm12rC2PId, mm12rP2CId, mm12rCCId,
                            l1Base_[L1_MM1A_OFF], L1_EV_P2C[3], L1_EV_C2P[3],
                            runInfo1, runParam, isLastBmm1 && (s2LoopCount == (runParam.s2LoopEndIdx - 1)),
                            ctx);
                    }
                    if ASCEND_IS_AIV {
                        LocalTensor<uint8_t> l1Tensor = GetMm12RightL1();
                        uint32_t l1CCId = GetMm12RightCCId();
                        auto l1Buf = MakeL1Buffer(l1Tensor, l1CCId);

                        LocalTensor<uint8_t> bmm1Tensor = GetBmm1Ub();
                        uint32_t fwd = GetBmm1FwdId();
                        uint32_t bwd = GetBmm1BwdId();
                        auto ubBuf = MakeUbBuffer(bmm1Tensor, MM1_RES_SIZE, fwd, bwd);

                        this->vecBlock.ProcessVec1(l1Buf, ubBuf, runInfo1, ctx);
                    }
                }
                if (taskId >= PRELOAD_N) {
                    RunInfo<isInfer> &runInfo2 = runInfo[(taskId - PRELOAD_N) & 3];
                    if ASCEND_IS_AIC {
                        LocalTensor<uint8_t> bmm2Tensor = GetBmm2Ub();
                        uint32_t bmm2FwdId = GetBmm2FwdId();
                        uint32_t bmm2BwdId = GetBmm2BwdId();

                        LocalTensor<uint8_t> mm12rReused = GetMm12RightL1Reused();
                        uint32_t reusedCCId = GetMm12RightReusedCCId();

                        this->cubeBlock.IterateBmm2(bmm2Tensor, bmm2FwdId, bmm2BwdId,
                            mm12rReused, reusedCCId,
                            runInfo2, ctx);
                    }
                    if ASCEND_IS_AIV {
                        LocalTensor<uint8_t> bmm2Tensor = GetBmm2Ub();
                        uint32_t fwd = GetBmm2FwdId();
                        uint32_t bwd = GetBmm2BwdId();
                        auto ubBuf = MakeUbBuffer(bmm2Tensor, MM2_RES_SIZE, fwd, bwd);
                        this->vecBlock.ProcessVec2(ubBuf, runInfo2, ctx);
                    }
                }
                taskId++;
            }
        }
        gS1StartIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    FAContext<isInfer, hasRope> &ctx, int64_t bnIndx, int64_t gS1Index, int64_t &multiCoreInnerIdx, RunParamStr<isInfer>& runParam)
{
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        if (runParam.boIdx == 0) {
            this->s1SizeAcc = 0;
            this->s2SizeAcc = 0;
        } else {
            this->s1SizeAcc = actualSeqQlenAddr[runParam.boIdx - 1];
            if constexpr (isPa) {
                this->s2SizeAcc = 0;
                for (uint32_t boIdx = 0; boIdx < runParam.boIdx; boIdx++) {
                    this->s2SizeAcc += actualSeqKvlenAddr[boIdx];
                }
            } else {
                this->s2SizeAcc = actualSeqKvlenAddr[runParam.boIdx - 1];
            }
        }
    }
    runParam.goIdx = gS1Index / ctx.s1OuterSize;
    runParam.s1oIdx = gS1Index % ctx.s1OuterSize;
    multiCoreInnerIdx++;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::SetRunInfo(
    FAContext<isInfer, hasRope> &ctx, RunInfo<isInfer> &runInfo,
    RunParamStr<isInfer>& runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.sOuterOffset = runParam.sOuterOffset;
    runInfo.s2StartIdx = runParam.s2LineStartIdx;
    runInfo.s2EndIdx = runParam.s2LineEndIdx;
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx % 2;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        runInfo.boIdx = runParam.boIdx;
        runInfo.s1SizeAcc = s1SizeAcc;
        runInfo.s2SizeAcc = s2SizeAcc;
    } else {
        runInfo.s2SizeAcc = runInfo.boIdx * ctx.s2Size;
    }
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId % 2;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        GetSeqQlenKvlenByBoidx(runParam.boIdx, ctx.s1Size, ctx.s2Size);
        runInfo.b1SSOffset = this->b1SSOffset;
    } else {
        runInfo.b1SSOffset = runInfo.boIdx * ctx.s1S2;
    }

    runInfo.actualS1Size = ctx.s1Size;
    runInfo.actualS2Size = ctx.s2Size;

    this->ComputeBmm1Tail(ctx, runInfo, runParam);
    runInfo.qRopeOffset = runParam.qRopeNBGOffset;
    InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, runInfo);
    ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, ctx, s2LoopCount + runInfo.s2StartIdx
        / s2BaseSize, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::GetSeqQlenKvlenByBoidx(
    int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvLen)
{
    if (unlikely(boIdx == 0)) {
        actualSeqQlen = actualSeqQlenAddr[0];
        actualSeqKvLen = actualSeqKvlenAddr[0];
        return;
    }
    actualSeqQlen = actualSeqQlenAddr[boIdx] - actualSeqQlenAddr[boIdx - 1];
    if constexpr (isPa) {
        actualSeqKvLen = actualSeqKvlenAddr[boIdx];
    } else {
        actualSeqKvLen = actualSeqKvlenAddr[boIdx] - actualSeqKvlenAddr[boIdx - 1];
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::ComputeBmm1Tail(
    FAContext<isInfer, hasRope> &ctx, RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam)
{
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.s1RealSizeAlign32 = runParam.s1RealSizeAlign32;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;
    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize;
    runInfo.vecCoreOffset = ctx.subBlockIdx * runInfo.firstHalfS1RealSize;

    runInfo.s2RealSize = s2BaseSize;
    if (runInfo.s2StartIdx + (runInfo.s2LoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
        runInfo.s2RealSize = runInfo.s2EndIdx - runInfo.s2LoopCount * runInfo.s2RealSize - runInfo.s2StartIdx;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FAKernelNoquantMla<CubeBlockType, VecBlockType>::IsLastBN(
    FAContext<isInfer, hasRope> &ctx, uint32_t bnStartIdx, uint32_t bnEndIdx)
{
    if constexpr(layout != LayOutTypeEnum::LAYOUT_TND) {
        return bnStartIdx == bnEndIdx - 1;
    }
    if (bnStartIdx != bnEndIdx - 1) {
        for (uint32_t bnIdx = bnStartIdx + 1; bnIdx < bnEndIdx; bnIdx++) {
            uint32_t boIdx = bnIdx / ctx.n2Size;
            uint32_t boStart = bnStartIdx / ctx.n2Size;
            if (actualSeqQlenAddr[boIdx] != actualSeqQlenAddr[boStart]) {
                return false;
            }
        }
    }
    return true;
}

#endif
