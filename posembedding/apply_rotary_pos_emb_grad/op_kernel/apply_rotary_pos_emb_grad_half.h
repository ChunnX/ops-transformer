/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file apply_rotary_pos_emb_grad_half.h
 * \brief
 */

#ifndef __APPLY_ROTARY_POS_EMB_GRAD_HALF_H__
#define __APPLY_ROTARY_POS_EMB_GRAD_HALF_H__

#include "kernel_operator.h"

namespace ApplyRotaryPosEmbGrad {

using namespace AscendC;

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
class ApplyRotaryPosEmbGradHalf {
public:
    __aicore__ inline ApplyRotaryPosEmbGradHalf(){};
    __aicore__ inline void Init(GM_ADDR grad_query_embed, GM_ADDR grad_key_embed, GM_ADDR cos, GM_ADDR sin,
                                GM_ADDR query, GM_ADDR key, GM_ADDR grad_query, GM_ADDR grad_key, GM_ADDR grad_cos,
                                GM_ADDR grad_sin, GM_ADDR workspace,
                                const ApplyRotaryPosEmbGradTilingData &tiling);
    __aicore__ inline void InitTilingData(const ApplyRotaryPosEmbGradTilingData &tiling);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void Process();
    __aicore__ inline void LoopProcess(int64_t &loopIdx);
    __aicore__ inline void Compute();
    __aicore__ inline void ProcessHeadLoop(int64_t outerLoopIdx, LocalTensor<T> &cosLocal, LocalTensor<T> &sinLocal,
                                           int64_t baseOffset, int64_t outerLoopGmStride, int64_t headInnerLoops,
                                           GlobalTensor<T> &gradEmbedGm, GlobalTensor<T> &inGm,
                                           GlobalTensor<T> &gradOutGm, uint32_t alignSrcStride,
                                           uint32_t paddedSrcStride);
    __aicore__ inline void ComputeInner(const LocalTensor<T> &cosLocal, const LocalTensor<T> &sinLocal,
                                        LocalTensor<T> &inLocal, GlobalTensor<T> &gradGm, int64_t &offset,
                                        int64_t innerPaddedSrcStride);
    __aicore__ inline void ComputeInnerOptional(LocalTensor<T> &inLocal);
    __aicore__ inline void CopyOutOptional();
    __aicore__ inline void CopyInOuter(int64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                       uint32_t dstStride);
    __aicore__ inline void CopyInInner(GlobalTensor<T> &gradEmbedGm, GlobalTensor<T> &optionalGm, int64_t &offset,
                                       uint16_t blockCount, uint32_t blockLen, uint32_t srcStride, uint32_t dstStride);
    __aicore__ inline void CopyOutInner(GlobalTensor<T> &gradGm, int64_t &offset, uint16_t blockCount,
                                        uint32_t blockLen, uint32_t srcStride, uint32_t dstStride);
    __aicore__ inline void CopyOutOuter(int64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                        uint32_t dstStride);
    __aicore__ inline void CopyInOuterAlign(int64_t &offset, uint32_t elementNum);
    __aicore__ inline void CopyInInnerAlign(GlobalTensor<T> &gradEmbedGm, GlobalTensor<T> &optionalGm, int64_t &offset,
                                            uint16_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                            uint32_t dstStride);
    __aicore__ inline void CopyOutInnerAlign(GlobalTensor<T> &gradGm, int64_t &offset, uint16_t blockCount,
                                             uint32_t blockLen, uint32_t srcStride, uint32_t dstStride);
    __aicore__ inline void CopyOutOuterAlign(int64_t &offset, uint32_t elementNum);
    __aicore__ inline void DataChunkCat(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src1Local,
                                        const LocalTensor<float> &src2Local, uint64_t offset, uint16_t blockCount,
                                        uint32_t blockLen, uint32_t srcStride, uint32_t dstStride);
    __aicore__ inline void InitCastInOuter(const LocalTensor<T> &cosLocal, const LocalTensor<T> &sinLocal);
    __aicore__ inline void CastOutOuter(const LocalTensor<T> &gradCosLocal, const LocalTensor<T> &gradSinLocal);

protected:
    // Define const value
    static constexpr int64_t BLOCK_SIZE = 32;
    static constexpr int64_t MASK_NUM_FP32 = 64;
    static constexpr int64_t BUFFER_NUM = 1;
    static constexpr int64_t LAYOUT_COMMON = 0;
    static constexpr int64_t HALF_COUNT = 2;

    // Init parameters
    int64_t outerOffset_ = 0;
    int64_t dQOffset_ = 0;
    int64_t dQOuterLoopOffset_ = 0;
    int64_t dQInnerLoopOffset_ = 0;
    int64_t dKOffset_ = 0;
    int64_t dKOuterLoopOffset_ = 0;
    int64_t dKInnerLoopOffset_ = 0;

    uint64_t dataEachBlock_ = BLOCK_SIZE / sizeof(T);
    uint64_t dataEachBlockFP32_ = BLOCK_SIZE / sizeof(float);

    int64_t fullTileLoops_ = 0;
    int64_t tailElements_ = 0;
    int64_t rowsPerTile_ = 0;
    uint32_t elementNum_ = 0;

    // Init tiling data
    int64_t hasOptionalInputs_ = 0;
    int64_t dQShapeSize_ = 0;
    int64_t dKShapeSize_ = 0;
    int64_t cosSinShapeSize_ = 0;
    int64_t dQKDimB_ = 0;
    int64_t dimS_ = 0;
    int64_t dQDimN_ = 0;
    int64_t dKDimN_ = 0;
    int64_t dimD_ = 0;
    int64_t cosSinDimB_ = 0;
    int64_t cosSinDimN_ = 0;
    int64_t halfDimDAlignNum_ = 0;

    int64_t elementsPerCore_ = 0;
    int64_t lastCoreElements_ = 0;
    int64_t fullTileLoopsPerCore_ = 0;
    int64_t tailElementsPerCore_ = 0;
    int64_t lastCoreFullTileLoops_ = 0;
    int64_t lastCoreTailElements_ = 0;
    int64_t alignUbSize_ = 0;
    int64_t calcUbSize_ = 0;
    int64_t coreUsed_ = 0;

    int64_t cosSinOuterLoops_ = 0;
    int64_t dQHeadInnerLoops_ = 0;
    int64_t dKHeadInnerLoops_ = 0;
    int64_t dQOuterLoopGmStride_ = 0;
    int64_t dKOuterLoopGmStride_ = 0;
    int64_t innerPaddedBlockLen_ = 0;
    int64_t dQInnerPaddedSrcStride_ = 0;
    int64_t dKInnerPaddedSrcStride_ = 0;

    uint32_t dQInnerAlignSrcStride_ = 0;
    uint32_t dKInnerAlignSrcStride_ = 0;

    // init gm tensors
    GlobalTensor<T> gradQueryEmbedGm_;
    GlobalTensor<T> gradKeyEmbedGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> queryGm_;
    GlobalTensor<T> keyGm_;
    GlobalTensor<T> gradQueryGm_;
    GlobalTensor<T> gradKeyGm_;
    GlobalTensor<T> gradCosGm_;
    GlobalTensor<T> gradSinGm_;

    // Init pipe
    TPipe pipe_;

    // Init MTE queues
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGradQK_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCos_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSin_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueQK_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGradQK_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGradCos_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGradSin_;

    // Init vector buffers
    TBuf<TPosition::VECCALC> tmpCalcBufA_;
    TBuf<TPosition::VECCALC> tmpCalcBufB_;
    TBuf<TPosition::VECCALC> tmpCalcBufC_;
    TBuf<TPosition::VECCALC> tmpCalcBufOutGradCos_;
    TBuf<TPosition::VECCALC> tmpCalcBufOutGradSin_;
    TBuf<TPosition::VECCALC> tmpCastGradQK_;
    TBuf<TPosition::VECCALC> tmpCalcBufCastInGradQK_;
    TBuf<TPosition::VECCALC> tmpCalcBufCastInCos_;
    TBuf<TPosition::VECCALC> tmpCalcBufCastInSin_;
    TBuf<TPosition::VECCALC> tmpCalcBufCastRes_;

    LocalTensor<float> tmpOutGradCos_;
    LocalTensor<float> tmpOutGradSin_;
    LocalTensor<float> tmpCastRes_;
    LocalTensor<float> cosCast_;
    LocalTensor<float> sinCast_;
    LocalTensor<float> gradCast_;

    LocalTensor<float> tmpLocalA_;
    LocalTensor<float> tmpLocalB_;
    LocalTensor<float> tmpLocalC_;
};

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::Init(
    GM_ADDR grad_query_embed, GM_ADDR grad_key_embed, GM_ADDR cos, GM_ADDR sin, GM_ADDR query, GM_ADDR key,
    GM_ADDR grad_query, GM_ADDR grad_key, GM_ADDR grad_cos, GM_ADDR grad_sin, GM_ADDR workspace,
    const ApplyRotaryPosEmbGradTilingData &tiling)
{
    InitTilingData(tiling);

    // Init global memory
    gradQueryEmbedGm_.SetGlobalBuffer((__gm__ T *)grad_query_embed, dQShapeSize_);
    gradKeyEmbedGm_.SetGlobalBuffer((__gm__ T *)grad_key_embed, dKShapeSize_);
    cosGm_.SetGlobalBuffer((__gm__ T *)cos, cosSinShapeSize_);
    sinGm_.SetGlobalBuffer((__gm__ T *)sin, cosSinShapeSize_);
    gradQueryGm_.SetGlobalBuffer((__gm__ T *)grad_query, dQShapeSize_);
    gradKeyGm_.SetGlobalBuffer((__gm__ T *)grad_key, dKShapeSize_);

    if constexpr (IF_NEED_BACKWARD) {
        queryGm_.SetGlobalBuffer((__gm__ T *)query, dQShapeSize_);
        keyGm_.SetGlobalBuffer((__gm__ T *)key, dKShapeSize_);
        gradCosGm_.SetGlobalBuffer((__gm__ T *)grad_cos, cosSinShapeSize_);
        gradSinGm_.SetGlobalBuffer((__gm__ T *)grad_sin, cosSinShapeSize_);
    }

    InitBuffers();
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::InitTilingData(
    const ApplyRotaryPosEmbGradTilingData &tiling)
{
    const ApplyRotaryPosEmbHalfGradParams &applyRotaryPosEmbHalfGradTiling = tiling.applyRopeHalfGradParams;
    hasOptionalInputs_ = applyRotaryPosEmbHalfGradTiling.hasOptionalInputs;
    dQShapeSize_ = applyRotaryPosEmbHalfGradTiling.dQShapeSize;
    dKShapeSize_ = applyRotaryPosEmbHalfGradTiling.dKShapeSize;
    cosSinShapeSize_ = applyRotaryPosEmbHalfGradTiling.cosSinShapeSize;
    dQKDimB_ = applyRotaryPosEmbHalfGradTiling.dQKDimB;
    dimS_ = applyRotaryPosEmbHalfGradTiling.dimS;
    dQDimN_ = applyRotaryPosEmbHalfGradTiling.dQDimN;
    dKDimN_ = applyRotaryPosEmbHalfGradTiling.dKDimN;
    dimD_ = applyRotaryPosEmbHalfGradTiling.dimD;
    cosSinDimB_ = applyRotaryPosEmbHalfGradTiling.cosSinDimB;
    cosSinDimN_ = applyRotaryPosEmbHalfGradTiling.cosSinDimN;
    halfDimDAlignNum_ = applyRotaryPosEmbHalfGradTiling.halfDimDAlignNum;

    elementsPerCore_ = applyRotaryPosEmbHalfGradTiling.elementsPerCore;
    lastCoreElements_ = applyRotaryPosEmbHalfGradTiling.lastCoreElements;
    fullTileLoopsPerCore_ = applyRotaryPosEmbHalfGradTiling.fullTileLoopsPerCore;
    tailElementsPerCore_ = applyRotaryPosEmbHalfGradTiling.tailElementsPerCore;
    lastCoreFullTileLoops_ = applyRotaryPosEmbHalfGradTiling.lastCoreFullTileLoops;
    lastCoreTailElements_ = applyRotaryPosEmbHalfGradTiling.lastCoreTailElements;
    alignUbSize_ = applyRotaryPosEmbHalfGradTiling.alignUbSize;
    calcUbSize_ = applyRotaryPosEmbHalfGradTiling.calcUbSize;
    coreUsed_ = applyRotaryPosEmbHalfGradTiling.coreUsed;

    cosSinOuterLoops_ = applyRotaryPosEmbHalfGradTiling.cosSinOuterLoops;
    dQHeadInnerLoops_ = applyRotaryPosEmbHalfGradTiling.dQHeadInnerLoops;
    dKHeadInnerLoops_ = applyRotaryPosEmbHalfGradTiling.dKHeadInnerLoops;
    dQOuterLoopGmStride_ = applyRotaryPosEmbHalfGradTiling.dQOuterLoopGmStride;
    dKOuterLoopGmStride_ = applyRotaryPosEmbHalfGradTiling.dKOuterLoopGmStride;
    innerPaddedBlockLen_ = applyRotaryPosEmbHalfGradTiling.innerPaddedBlockLen;
    dQInnerPaddedSrcStride_ = applyRotaryPosEmbHalfGradTiling.dQInnerPaddedSrcStride;
    dKInnerPaddedSrcStride_ = applyRotaryPosEmbHalfGradTiling.dKInnerPaddedSrcStride;
    dQInnerAlignSrcStride_ =
        static_cast<uint32_t>((dQInnerPaddedSrcStride_ - dimD_ / HALF_COUNT * static_cast<int64_t>(sizeof(T))));
    dKInnerAlignSrcStride_ =
        static_cast<uint32_t>((dKInnerPaddedSrcStride_ - dimD_ / HALF_COUNT * static_cast<int64_t>(sizeof(T))));
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::InitBuffers()
{
    // Set loop nums and tail element nums
    if (GetBlockIdx() != coreUsed_ - 1) {
        fullTileLoops_ = fullTileLoopsPerCore_;
        tailElements_ = tailElementsPerCore_;
    } else {
        fullTileLoops_ = lastCoreFullTileLoops_;
        tailElements_ = lastCoreTailElements_;
    }

    // Init buffers
    pipe_.InitBuffer(inQueueGradQK_, BUFFER_NUM, alignUbSize_ * sizeof(T));
    pipe_.InitBuffer(inQueueCos_, BUFFER_NUM, alignUbSize_ * sizeof(T));
    pipe_.InitBuffer(inQueueSin_, BUFFER_NUM, alignUbSize_ * sizeof(T));
    pipe_.InitBuffer(outQueueGradQK_, BUFFER_NUM, alignUbSize_ * sizeof(T));

    pipe_.InitBuffer(tmpCalcBufA_, alignUbSize_ * sizeof(float));
    pipe_.InitBuffer(tmpCalcBufB_, alignUbSize_ * sizeof(float));
    pipe_.InitBuffer(tmpCalcBufC_, alignUbSize_ * sizeof(float));

    if constexpr (IF_NEED_BACKWARD) {
        pipe_.InitBuffer(inQueueQK_, BUFFER_NUM, alignUbSize_ * sizeof(T));
        pipe_.InitBuffer(outQueueGradCos_, BUFFER_NUM, alignUbSize_ * sizeof(T));
        pipe_.InitBuffer(outQueueGradSin_, BUFFER_NUM, alignUbSize_ * sizeof(T));

        pipe_.InitBuffer(tmpCalcBufOutGradCos_, alignUbSize_ * sizeof(float));
        pipe_.InitBuffer(tmpCalcBufOutGradSin_, alignUbSize_ * sizeof(float));
    }

    if constexpr (!IsSameType<T, float>::value) {
        pipe_.InitBuffer(tmpCalcBufCastInGradQK_, alignUbSize_ * sizeof(float));
        pipe_.InitBuffer(tmpCalcBufCastInCos_, alignUbSize_ * sizeof(float));
        pipe_.InitBuffer(tmpCalcBufCastInSin_, alignUbSize_ * sizeof(float));
        pipe_.InitBuffer(tmpCalcBufCastRes_, alignUbSize_ * sizeof(float));
    }

    if constexpr (IF_NEED_BACKWARD) {
        tmpOutGradCos_ = tmpCalcBufOutGradCos_.Get<float>(alignUbSize_);
        tmpOutGradSin_ = tmpCalcBufOutGradSin_.Get<float>(alignUbSize_);
    }

    if constexpr (!IsSameType<T, float>::value) {
        gradCast_ = tmpCalcBufCastInGradQK_.Get<float>(alignUbSize_);
        cosCast_ = tmpCalcBufCastInCos_.Get<float>(alignUbSize_);
        sinCast_ = tmpCalcBufCastInSin_.Get<float>(alignUbSize_);
        tmpCastRes_ = tmpCalcBufCastRes_.Get<float>(alignUbSize_);
    }
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::Process()
{
    for (int64_t loopIdx = 0; loopIdx < fullTileLoops_; ++loopIdx) {
        elementNum_ = calcUbSize_;
        LoopProcess(loopIdx);
    }
    if (tailElements_ > 0) {
        elementNum_ = tailElements_;
        LoopProcess(fullTileLoops_);
    }
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::LoopProcess(int64_t &loopIdx)
{
    outerOffset_ = GetBlockIdx() * elementsPerCore_ + loopIdx * calcUbSize_;
    dQOffset_ = GetBlockIdx() * elementsPerCore_ * dQHeadInnerLoops_ + loopIdx * calcUbSize_ * dQHeadInnerLoops_;
    dKOffset_ = GetBlockIdx() * elementsPerCore_ * dKHeadInnerLoops_ + loopIdx * calcUbSize_ * dKHeadInnerLoops_;
    rowsPerTile_ = elementNum_ / dimD_;
    if constexpr (IF_ALIGN) {
        elementNum_ = rowsPerTile_ * dimD_;
    } else {
        elementNum_ = rowsPerTile_ * halfDimDAlignNum_ * HALF_COUNT;
    }
    Compute();
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::Compute()
{
    if constexpr (IF_ALIGN) {
        CopyInOuterAlign(outerOffset_, elementNum_);
    } else {
        CopyInOuter(outerOffset_, static_cast<uint16_t>(rowsPerTile_ * 2), static_cast<uint32_t>(innerPaddedBlockLen_),
                    0, 0);
    }

    LocalTensor<T> cosLocal = inQueueCos_.DeQue<T>();
    LocalTensor<T> sinLocal = inQueueSin_.DeQue<T>();
    InitCastInOuter(cosLocal, sinLocal);

    for (int64_t outerLoopIdx = 0; outerLoopIdx < cosSinOuterLoops_; ++outerLoopIdx) {
        // Process Query
        ProcessHeadLoop(outerLoopIdx, cosLocal, sinLocal, dQOffset_, dQOuterLoopGmStride_, dQHeadInnerLoops_,
                        gradQueryEmbedGm_, queryGm_, gradQueryGm_, dQInnerAlignSrcStride_, dQInnerPaddedSrcStride_);

        // Process Key
        ProcessHeadLoop(outerLoopIdx, cosLocal, sinLocal, dKOffset_, dKOuterLoopGmStride_, dKHeadInnerLoops_,
                        gradKeyEmbedGm_, keyGm_, gradKeyGm_, dKInnerAlignSrcStride_, dKInnerPaddedSrcStride_);
    }

    inQueueCos_.FreeTensor(cosLocal);
    inQueueSin_.FreeTensor(sinLocal);

    if constexpr (IF_NEED_BACKWARD) {
        CopyOutOptional();
    }
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::ProcessHeadLoop(
    int64_t outerLoopIdx, LocalTensor<T> &cosLocal, LocalTensor<T> &sinLocal, int64_t baseOffset,
    int64_t outerLoopGmStride, int64_t headInnerLoops, GlobalTensor<T> &gradEmbedGm, GlobalTensor<T> &inGm,
    GlobalTensor<T> &gradOutGm, uint32_t alignSrcStride, uint32_t paddedSrcStride)
{
    int64_t outerLoopOffset = baseOffset + outerLoopIdx * outerLoopGmStride;

    for (int64_t innerLoopIdx = 0; innerLoopIdx < headInnerLoops; ++innerLoopIdx) {
        int64_t innerLoopOffset = outerLoopOffset + innerLoopIdx * dimD_;

        if constexpr (IF_ALIGN) {
            CopyInInnerAlign(gradEmbedGm, inGm, innerLoopOffset, static_cast<uint16_t>(rowsPerTile_),
                             static_cast<uint32_t>(dimD_ * static_cast<int64_t>(sizeof(T))), alignSrcStride, 0);
        } else {
            CopyInInner(gradEmbedGm, inGm, innerLoopOffset, static_cast<uint16_t>(rowsPerTile_),
                        static_cast<uint32_t>(innerPaddedBlockLen_), static_cast<uint32_t>(paddedSrcStride),
                        static_cast<uint32_t>(halfDimDAlignNum_ / dataEachBlock_));
        }

        LocalTensor<T> inLocal = inQueueGradQK_.DeQue<T>();
        ComputeInner(cosLocal, sinLocal, inLocal, gradOutGm, innerLoopOffset, paddedSrcStride);

        if constexpr (IF_NEED_BACKWARD) {
            ComputeInnerOptional(inLocal);
        }
    }
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::ComputeInner(
    const LocalTensor<T> &cosLocal, const LocalTensor<T> &sinLocal, LocalTensor<T> &inLocal, GlobalTensor<T> &gradGm,
    int64_t &offset, int64_t innerPaddedSrcStride)
{
    LocalTensor<T> outLocal = outQueueGradQK_.AllocTensor<T>();
    tmpLocalA_ = tmpCalcBufA_.Get<float>(alignUbSize_);
    tmpLocalB_ = tmpCalcBufB_.Get<float>(alignUbSize_);
    tmpLocalC_ = tmpCalcBufC_.Get<float>(alignUbSize_);
    if constexpr (IsSameType<T, float>::value) {
        Mul(outLocal, cosLocal, inLocal, elementNum_);

        Mul(tmpLocalB_, sinLocal, inLocal, elementNum_);
        if (!hasOptionalInputs_) {
            inQueueGradQK_.FreeTensor(inLocal);
        }
    } else {
        Cast(gradCast_, inLocal, AscendC::RoundMode::CAST_NONE, elementNum_);
        inQueueGradQK_.FreeTensor(inLocal);
        PipeBarrier<PIPE_V>();
        Mul(tmpCastRes_, cosCast_, gradCast_, elementNum_);

        Mul(tmpLocalB_, sinCast_, gradCast_, elementNum_);
    }

    PipeBarrier<PIPE_V>();
    Muls(tmpLocalA_, tmpLocalB_, -1.0f, elementNum_);
    PipeBarrier<PIPE_V>();
    DataChunkCat(tmpLocalC_, tmpLocalB_, tmpLocalA_, halfDimDAlignNum_, static_cast<uint16_t>(rowsPerTile_),
                 static_cast<uint32_t>(halfDimDAlignNum_ / dataEachBlockFP32_),
                 static_cast<uint32_t>(halfDimDAlignNum_ / dataEachBlockFP32_),
                 static_cast<uint32_t>(halfDimDAlignNum_ / dataEachBlockFP32_));
    PipeBarrier<PIPE_V>();

    if constexpr (IsSameType<T, float>::value) {
        Add(outLocal, tmpLocalC_, outLocal, elementNum_);
    } else {
        Add(tmpCastRes_, tmpLocalC_, tmpCastRes_, elementNum_);
        PipeBarrier<PIPE_V>();
        Cast(outLocal, tmpCastRes_, AscendC::RoundMode::CAST_RINT, elementNum_);
    }

    outQueueGradQK_.EnQue<T>(outLocal);
    if constexpr (IF_ALIGN) {
        uint32_t innerAlignDstStride =
            static_cast<uint32_t>((innerPaddedSrcStride - dimD_ / HALF_COUNT * static_cast<int64_t>(sizeof(T))));
        CopyOutInnerAlign(gradGm, offset, static_cast<uint16_t>(rowsPerTile_),
                          static_cast<uint32_t>(dimD_ * static_cast<int64_t>(sizeof(T))), 0, innerAlignDstStride);
    } else {
        CopyOutInner(gradGm, offset, static_cast<uint16_t>(rowsPerTile_), static_cast<uint32_t>(innerPaddedBlockLen_),
                     static_cast<uint32_t>(halfDimDAlignNum_ / dataEachBlock_),
                     static_cast<uint32_t>(innerPaddedSrcStride));
    }
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void
ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::ComputeInnerOptional(LocalTensor<T> &inLocal)
{
    LocalTensor<T> optionalInLocal = inQueueQK_.DeQue<T>();
    if constexpr (IsSameType<T, float>::value) {
        PipeBarrier<PIPE_V>();
        Mul(tmpLocalB_, optionalInLocal, inLocal, elementNum_);
        PipeBarrier<PIPE_V>();
        Add(tmpOutGradCos_, tmpLocalB_, tmpOutGradCos_, elementNum_);
    } else {
        Cast(tmpLocalC_, optionalInLocal, AscendC::RoundMode::CAST_NONE, elementNum_);
        inQueueQK_.FreeTensor(optionalInLocal);
        PipeBarrier<PIPE_V>();
        Mul(tmpCastRes_, tmpLocalC_, gradCast_, elementNum_);
        PipeBarrier<PIPE_V>();
        Add(tmpOutGradCos_, tmpCastRes_, tmpOutGradCos_, elementNum_);
    }

    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<T, float>::value) {
        Muls(tmpLocalA_, optionalInLocal, -1.0f, elementNum_);
        PipeBarrier<PIPE_V>();
        DataChunkCat(tmpLocalB_, tmpLocalA_, optionalInLocal, halfDimDAlignNum_, static_cast<uint16_t>(rowsPerTile_),
                     static_cast<uint32_t>(halfDimDAlignNum_ / dataEachBlockFP32_),
                     static_cast<uint32_t>(halfDimDAlignNum_ / dataEachBlockFP32_),
                     static_cast<uint32_t>(halfDimDAlignNum_ / dataEachBlockFP32_));
        inQueueQK_.FreeTensor(optionalInLocal);
        PipeBarrier<PIPE_V>();
        Mul(tmpLocalA_, tmpLocalB_, inLocal, elementNum_);
        inQueueGradQK_.FreeTensor(inLocal);
        PipeBarrier<PIPE_V>();
        Add(tmpOutGradSin_, tmpLocalA_, tmpOutGradSin_, elementNum_);
    } else {
        Muls(tmpLocalA_, tmpLocalC_, -1.0f, elementNum_);
        PipeBarrier<PIPE_V>();
        DataChunkCat(tmpLocalB_, tmpLocalA_, tmpLocalC_, halfDimDAlignNum_, static_cast<uint16_t>(rowsPerTile_),
                     static_cast<uint32_t>(halfDimDAlignNum_ / dataEachBlockFP32_),
                     static_cast<uint32_t>(halfDimDAlignNum_ / dataEachBlockFP32_),
                     static_cast<uint32_t>(halfDimDAlignNum_ / dataEachBlockFP32_));
        PipeBarrier<PIPE_V>();
        Mul(tmpCastRes_, tmpLocalB_, gradCast_, elementNum_);
        PipeBarrier<PIPE_V>();
        Add(tmpOutGradSin_, tmpCastRes_, tmpOutGradSin_, elementNum_);
    }
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::CopyOutOptional()
{
    LocalTensor<T> gradCosLocal = outQueueGradCos_.AllocTensor<T>();
    LocalTensor<T> gradSinLocal = outQueueGradSin_.AllocTensor<T>();
    CastOutOuter(gradCosLocal, gradSinLocal);
    outQueueGradCos_.EnQue<T>(gradCosLocal);
    outQueueGradSin_.EnQue<T>(gradSinLocal);
    if constexpr (IF_ALIGN) {
        CopyOutOuterAlign(outerOffset_, elementNum_);
    } else {
        CopyOutOuter(outerOffset_, static_cast<uint16_t>(rowsPerTile_ * 2), static_cast<uint32_t>(innerPaddedBlockLen_),
                     0, 0);
    }
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::CopyInOuter(
    int64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride, uint32_t dstStride)
{
    DataCopyPadExtParams<T> copyPadParams{false, 0, 0, 0};

    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    LocalTensor<T> cosLocal = inQueueCos_.AllocTensor<T>();
    LocalTensor<T> sinLocal = inQueueSin_.AllocTensor<T>();
    DataCopyPad(cosLocal, cosGm_[offset], copyParams, copyPadParams);
    DataCopyPad(sinLocal, sinGm_[offset], copyParams, copyPadParams);
    inQueueCos_.EnQue(cosLocal);
    inQueueSin_.EnQue(sinLocal);
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::CopyInInner(
    GlobalTensor<T> &gradEmbedGm, GlobalTensor<T> &optionalGm, int64_t &offset, uint16_t blockCount, uint32_t blockLen,
    uint32_t srcStride, uint32_t dstStride)
{
    DataCopyPadExtParams<T> copyPadParams{false, 0, 0, 0};

    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    LocalTensor<T> inLocal = inQueueGradQK_.AllocTensor<T>();
    DataCopyPad(inLocal, gradEmbedGm[offset], copyParams, copyPadParams);
    DataCopyPad(inLocal[halfDimDAlignNum_], gradEmbedGm[offset + dimD_ / 2], copyParams, copyPadParams);
    inQueueGradQK_.EnQue(inLocal);

    if constexpr (IF_NEED_BACKWARD) {
        LocalTensor<T> optionalInLocal = inQueueQK_.AllocTensor<T>();
        DataCopyPad(optionalInLocal, optionalGm[offset], copyParams, copyPadParams);
        DataCopyPad(optionalInLocal[halfDimDAlignNum_], optionalGm[offset + dimD_ / 2], copyParams, copyPadParams);
        inQueueQK_.EnQue(optionalInLocal);
    }
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void
ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::CopyOutInner(GlobalTensor<T> &gradGm, int64_t &offset,
                                                                              uint16_t blockCount, uint32_t blockLen,
                                                                              uint32_t srcStride, uint32_t dstStride)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    LocalTensor<T> outLocal = outQueueGradQK_.DeQue<T>();
    DataCopyPad(gradGm[offset], outLocal, copyParams);
    DataCopyPad(gradGm[offset + dimD_ / 2], outLocal[halfDimDAlignNum_], copyParams);
    outQueueGradQK_.FreeTensor(outLocal);
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::CopyOutOuter(
    int64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride, uint32_t dstStride)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    LocalTensor<T> gradCosLocal = outQueueGradCos_.DeQue<T>();
    LocalTensor<T> gradSinLocal = outQueueGradSin_.DeQue<T>();
    DataCopyPad(gradCosGm_[offset], gradCosLocal, copyParams);
    DataCopyPad(gradSinGm_[offset], gradSinLocal, copyParams);
    outQueueGradCos_.FreeTensor(gradCosLocal);
    outQueueGradSin_.FreeTensor(gradSinLocal);
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void
ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::CopyInOuterAlign(int64_t &offset, uint32_t elementNum)
{
    LocalTensor<T> cosLocal = inQueueCos_.AllocTensor<T>();
    LocalTensor<T> sinLocal = inQueueSin_.AllocTensor<T>();
    DataCopy(cosLocal, cosGm_[offset], elementNum);
    DataCopy(sinLocal, sinGm_[offset], elementNum);
    inQueueCos_.EnQue(cosLocal);
    inQueueSin_.EnQue(sinLocal);
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::CopyInInnerAlign(
    GlobalTensor<T> &gradEmbedGm, GlobalTensor<T> &optionalGm, int64_t &offset, uint16_t blockCount, uint32_t blockLen,
    uint32_t srcStride, uint32_t dstStride)
{
    DataCopyPadExtParams<T> copyPadParams{false, 0, 0, 0};
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    LocalTensor<T> inLocal = inQueueGradQK_.AllocTensor<T>();
    DataCopyPad(inLocal, gradEmbedGm[offset], copyParams, copyPadParams);
    inQueueGradQK_.EnQue(inLocal);

    if constexpr (IF_NEED_BACKWARD) {
        LocalTensor<T> optionalInLocal = inQueueQK_.AllocTensor<T>();
        DataCopyPad(optionalInLocal, optionalGm[offset], copyParams, copyPadParams);
        inQueueQK_.EnQue(optionalInLocal);
    }
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::CopyOutInnerAlign(
    GlobalTensor<T> &gradGm, int64_t &offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride,
    uint32_t dstStride)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    LocalTensor<T> outLocal = outQueueGradQK_.DeQue<T>();
    DataCopyPad(gradGm[offset], outLocal, copyParams);
    outQueueGradQK_.FreeTensor(outLocal);
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void
ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::CopyOutOuterAlign(int64_t &offset, uint32_t elementNum)
{
    LocalTensor<T> gradCosLocal = outQueueGradCos_.DeQue<T>();
    LocalTensor<T> gradSinLocal = outQueueGradSin_.DeQue<T>();
    DataCopy(gradCosGm_[offset], gradCosLocal, elementNum);
    DataCopy(gradSinGm_[offset], gradSinLocal, elementNum);
    outQueueGradCos_.FreeTensor(gradCosLocal);
    outQueueGradSin_.FreeTensor(gradSinLocal);
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::DataChunkCat(
    const LocalTensor<float> &dstLocal, const LocalTensor<float> &src1Local, const LocalTensor<float> &src2Local,
    uint64_t offset, uint16_t blockCount, uint32_t blockLen, uint32_t srcStride, uint32_t dstStride)
{
    DataCopyParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = dstStride;

    DataCopy(dstLocal, src1Local[offset], copyParams);
    DataCopy(dstLocal[offset], src2Local, copyParams);
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void
ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::InitCastInOuter(const LocalTensor<T> &cosLocal,
                                                                                 const LocalTensor<T> &sinLocal)
{
    if constexpr (IF_NEED_BACKWARD) {
        Duplicate(tmpOutGradCos_, (float)0.0, elementNum_);
        Duplicate(tmpOutGradSin_, (float)0.0, elementNum_);
    }
    if constexpr (!IsSameType<T, float>::value) {
        Cast(cosCast_, cosLocal, AscendC::RoundMode::CAST_NONE, elementNum_);
        Cast(sinCast_, sinLocal, AscendC::RoundMode::CAST_NONE, elementNum_);
    }
}

template <typename T, bool IF_NEED_BACKWARD, bool IF_ALIGN>
__aicore__ inline void
ApplyRotaryPosEmbGradHalf<T, IF_NEED_BACKWARD, IF_ALIGN>::CastOutOuter(const LocalTensor<T> &gradCosLocal,
                                                                              const LocalTensor<T> &gradSinLocal)
{
    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(gradCosLocal, tmpOutGradCos_, elementNum_);
        DataCopy(gradSinLocal, tmpOutGradSin_, elementNum_);
    } else {
        Cast(gradCosLocal, tmpOutGradCos_, AscendC::RoundMode::CAST_RINT, elementNum_);
        Cast(gradSinLocal, tmpOutGradSin_, AscendC::RoundMode::CAST_RINT, elementNum_);
    }
}


} // namespace ApplyRotaryPosEmbGrad

#endif // __APPLY_ROTARY_POS_EMB_GRAD_HALF_H__
