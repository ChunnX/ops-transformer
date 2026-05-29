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
 * \file lightning_indexer_v2_service_vector.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
#define LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_v2_common.h"
#include "../arch35/vf/lightning_indexer_v2_vector1.h"
#include "../arch35/vf/lightning_indexer_v2_topk.h"

namespace LIV2Kernel {
using namespace LIV2Common;
constexpr uint32_t TRUNK_LEN_16K = 16384;
constexpr uint32_t TRUNK_LEN_8K = 8192;

template <typename LIT>
class LightningIndexerV2ServiceVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_V2_LAYOUT LAYOUT_T = LIT::layout;
    static constexpr LI_V2_LAYOUT K_LAYOUT_T = LIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr bool DT_W_FLAG = true;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using QK_T = typename LIT::queryKeyType;
    using SCORE_T = typename LIT::scoreType;
    using W_T = float;

    __aicore__ inline LightningIndexerV2ServiceVector(){};
    __aicore__ inline void ProcessVec1(const LIV2Common::RunInfo &info);
    __aicore__ inline void ProcessTopK(const LIV2Common::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct LIV2Common::ConstInfo &constInfo,
                                      const LIV2TilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<K_T> valueOutGm, GlobalTensor<int32_t> blockTableGm,
                                              GlobalTensor<int32_t> outputIdxOffsetGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<K_T> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> outputIdxOffsetGm;
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;

    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;

private:
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<float> resMm1UB_;
    // tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<W_T> weightUB_;

    // tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;

    // tmp buff for returnValue bfloat16
    TBuf<TPosition::VECCALC> uIntToBfloat16Buf_;
    LocalTensor<bfloat16_t> uIntToBfloat16Local_;
    // tmp buff for returnValue K_T
    TBuf<TPosition::VECCALC> valueOutBuf_;
    LocalTensor<K_T> valueOutLocal_;
    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

    TBuf<TPosition::VECCALC> outInvalidBuf_;
    LocalTensor<int32_t> outInvalidLocal_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    uint32_t topkCount_ = 0;
    uint32_t topkCountAlign256_ = 0; // topkCount对齐到256(直方图需要)，支持topk泛化
    uint32_t trunkLen_ = 0;
    bool returnValueFlag = false;

    struct LIV2Common::ConstInfo constInfo_;
    liV2Topk::LIV2Topk<SCORE_T> topkOp_;
};

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(float));
    resMm1UB_ = resMm1Buf_.Get<float>();
    pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightUB_ = weightBuf_.Get<W_T>();
    // 大小：2(开dB) * 2 * 128 * 4 = 2KB
    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));
    // out
    vec1OutUB_ = outBuf_.Get<SCORE_T>();
    // returnvalue
    // 大小：topK * sizeof(bfloat16_t)
    pipe->InitBuffer(uIntToBfloat16Buf_, topkCountAlign256_ * sizeof(bfloat16_t));
    // returnValue bfloat16
    uIntToBfloat16Local_ = uIntToBfloat16Buf_.Get<bfloat16_t>();
    // 大小：topK * sizeof(float)
    pipe->InitBuffer(valueOutBuf_, topkCountAlign256_ * sizeof(K_T));
    // returnValue float
    valueOutLocal_ = valueOutBuf_.Get<K_T>();
    // Topk
    // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(SCORE_T)
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    pipe->InitBuffer(indicesOutBuf_, (topkCountAlign256_ + 64) * sizeof(uint32_t));
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();
    // 大小：topkCountAlign256_ * sizeof(SCORE_T)
    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(SCORE_T));
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_);
    // 刷-1
    pipe->InitBuffer(outInvalidBuf_, topkCount_ * sizeof(int32_t));
    outInvalidLocal_ = outInvalidBuf_.Get<int32_t>();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitParams(const struct LIV2Common::ConstInfo &constInfo,
                                                   const LIV2TilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;  // 4
    s2BaseSize_ = constInfo.s2BaseSize;  // 128
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    returnValueFlag = constInfo.returnValueFlag;
    blockId_ = GetBlockIdx();
    trunkLen_ = (std::is_same<SCORE_T, uint32_t>::value) ? TRUNK_LEN_8K : TRUNK_LEN_16K;
    topkCount_ = constInfo.topk;
    topkOp_.Init(topkCount_, trunkLen_);
    topkCountAlign256_ = LIV2Common::Align(constInfo.topk, (uint64_t)256); // topkCount对齐到256
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitVecInputTensor(GlobalTensor<W_T> weightsGm,
                                                           GlobalTensor<int32_t> indiceOutGm,
                                                           GlobalTensor<K_T> valueOutGm,
                                                           GlobalTensor<int32_t> blockTableGm,
                                                           GlobalTensor<int32_t> outputIdxOffsetGm)
{
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
    this->blockTableGm = blockTableGm;
    this->outputIdxOffsetGm = outputIdxOffsetGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm)
{
    this->scoreGm = scoreGm; // resucesum*k
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    Duplicate(outInvalidLocal_, constInfo_.INVALID_IDX, constInfo_.topk);

    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    AscendC::DataCopyParams dataCopyOutParams;
    dataCopyOutParams.blockCount = 1;
    dataCopyOutParams.blockLen = constInfo_.topk * sizeof(int32_t);
    dataCopyOutParams.srcStride = 0;
    dataCopyOutParams.dstStride = 0;
    AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], outInvalidLocal_, dataCopyOutParams);
    if (returnValueFlag) {
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        uint16_t negInf = 0;
        if constexpr (std::is_same_v<K_T, float16_t>) {
            negInf = 0xFC00;
        }else {
            negInf = 0xFF80;
        }
        
        Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>(), negInf, constInfo_.topk);

        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

        AscendC::DataCopyParams copyOutValueParams;
        copyOutValueParams.blockCount = 1;
        copyOutValueParams.blockLen = constInfo_.topk * sizeof(K_T);
        copyOutValueParams.srcStride = 0;
        copyOutValueParams.dstStride = 0;
        AscendC::DataCopyPad(valueOutGm[invalidS1Offset], valueOutLocal_, copyOutValueParams);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::ProcessVec1(const LIV2Common::RunInfo &info)
{
    // CV同步
    CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(
        LIV2Common::ConstInfo::CROSS_CV_EVENT + info.loop % 2);   // V核等C核计算完mm1，mm1Res已搬运到UB
    
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    if (curAivS1ProcNum == 0) {
        // V核处理完，通知C核可以把mm1Res搬运到UB
        CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(
            LIV2Common::ConstInfo::CROSS_VC_EVENT + info.loop % 2);
        return;
    }
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + (info.loop % 2));
    // weightsGm --> weightUB_
    int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
    DataCopyPadExtParams<W_T> padWeightsParams{false, 0, 0, 0};
    DataCopyExtParams qwDataCopyExtParams;
    qwDataCopyExtParams.blockCount = 1;
    qwDataCopyExtParams.blockLen = curAivS1ProcNum * gSize_* sizeof(W_T);
    qwDataCopyExtParams.srcStride = 0;
    qwDataCopyExtParams.dstStride = 0;
    DataCopyPad(weightUB_[(info.loop % 2) * CeilDiv(s1BaseSize_, 2) * gSize_],
                weightsGm[weightGmOffset], qwDataCopyExtParams, padWeightsParams);

    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + (info.loop % 2));
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + (info.loop % 2));
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + (info.loop % 2));
    for (int64_t s1IdxTmp = 0; s1IdxTmp < curAivS1ProcNum; s1IdxTmp++) {
        liV2Vector1::MulWeightAndReduceSum(
                vec1OutUB_[(info.loop % 2) * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ + s1IdxTmp * s2BaseSize_],
                resMm1UB_[(info.loop % 2) * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ +
                            s1IdxTmp * gSize_ * s2BaseSize_],
                weightUB_[(info.loop % 2) * CeilDiv(s1BaseSize_, 2) * gSize_ + s1IdxTmp * gSize_],
                gSize_);
    }
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + (info.loop % 2));
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + (info.loop % 2));
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + (info.loop % 2));
    // outUB_ --->  scoreGm
    int64_t vec1OutGmOffset = blockId_ % 2 == 0 ? curS2Idx :
                            CeilDiv(s1BaseSize_, 2) * LIV2Common::Align(
                                (uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = (LIV2Common::Align((uint64_t)constInfo_.kSeqSize,
                                                    (uint64_t)s2BaseSize_) - s2BaseSize_) * sizeof(SCORE_T);

    DataCopyPad(scoreGm[vec1OutGmOffset],
                vec1OutUB_[(info.loop % 2) * CeilDiv(s1BaseSize_, 2) * s2BaseSize_], copyOutParams);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + (info.loop % 2));
    // V核处理完，通知C核可以把mm1Res搬运到UB
    CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(
        LIV2Common::ConstInfo::CROSS_VC_EVENT + info.loop % 2);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::ProcessTopK(const LIV2Common::RunInfo &info)
{
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    }

    int32_t validS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset = blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;
        int64_t outputIdxOffset = 0;
        if (info.isOutputIdxOffsetValid) {
            outputIdxOffset = outputIdxOffsetGm.GetValue(info.outputIdxCoreOffset + rowIdx * kHeadNum_);
        }
        
        SCORE_T zero = 0;
        K_T zeroK_T = 0;
        int32_t neg = -1;
        if (constInfo_.attenMaskFlag) {
            validS2Len = ((int32_t)i + cuRealAcSeq) / static_cast<int32_t>(constInfo_.cmpRatio);
        }
        if (validS2Len <= 0) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                 indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            if (returnValueFlag) {
                WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                uint16_t negInf = 0;
                if constexpr (std::is_same_v<K_T, float16_t>) {
                    negInf = 0xFC00;
                }else {
                    negInf = 0xFF80;
                }
                Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>(), negInf, topkCount_);

                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

                AscendC::DataCopyParams copyOutValueParams;
                copyOutValueParams.blockCount = 1;
                copyOutValueParams.blockLen = topkCount_ * sizeof(K_T);
                copyOutValueParams.srcStride = 0;
                copyOutValueParams.dstStride = 0;
                AscendC::DataCopyPad(
                    valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_],
                    valueOutLocal_,
                    copyOutValueParams);
                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            }
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
        if (validS2Len >= topkCount_) {
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            if (s2LoopNum == 1) {
                uint32_t validS2LenAlign = LIV2Common::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(mrgValueLocal_,
                                     scoreGm[vecOffset * LIV2Common::Align(
                                        (uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                                     copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign,
                        0, 1, returnValueFlag, outputIdxOffset);
            } else {
                uint32_t outputIdxOffsetTmp = 0;
                for (uint32_t loopIdx = 0; loopIdx < s2LoopNum; loopIdx++) {
                    if (loopIdx == s2LoopNum - 1) {
                        outputIdxOffsetTmp = outputIdxOffset;
                    }
                    if (loopIdx == 0) {
                        copyInParams.blockLen = trunkLen_ * sizeof(SCORE_T); // byte
                        AscendC::DataCopyPad(mrgValueLocal_,
                                             scoreGm[vecOffset * LIV2Common::Align(
                                                (uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                                            copyInParams, padParams);
                        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, trunkLen_,
                                loopIdx, s2LoopNum, returnValueFlag, outputIdxOffsetTmp);
                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t validTrunkLen = (loopIdx * trunkLen_ + trunkLen_) > validS2Len ?
                                              validS2Len % trunkLen_ : trunkLen_;
                    uint32_t offset = vecOffset * LIV2Common::Align(
                            (uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + loopIdx * trunkLen_;
                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                    // topk如果没有对齐到256，则把topkCountAlign256_ - topkCount_部分刷0
                    if (topkCountAlign256_ != topkCount_) {
                        uint64_t mask[1];
                        mask[0] = ~0;
                        mask[0] = mask[0] << (topkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        // 把topkCount_对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                        PipeBarrier<PIPE_V>();
                        // 把topk剩余对齐到256的部分刷0
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], zero,
                                  topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    copyInParams.blockLen = validTrunkLen * sizeof(SCORE_T); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + validTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256],
                                  zero, LIV2Common::Align(validTrunkLen, (uint32_t)256) - validTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset],
                                         copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, LIV2Common::Align(
                            topkCountAlign256_ + validTrunkLen, (uint32_t)256), loopIdx,
                            s2LoopNum, returnValueFlag, outputIdxOffsetTmp);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(),
                                    (int32_t)(zero + outputIdxOffset), validS2Len);
            if (returnValueFlag) {
                copyInParams.blockLen = LIV2Common::Align(validS2Len, (int32_t)32) * sizeof(uint16_t);
                AscendC::DataCopyPad(scoreOutLocal_,
                            scoreGm[vecOffset * LIV2Common::Align(
                                (uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                            copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            }
        }
        
        if (validS2Len < topkCount_) {
            uint64_t mask[1];
            mask[0] = ~0;
            mask[0] = mask[0] << (validS2Len % 8);
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, mask, 1, 1, 0);
        }
        
        if (validS2Len / 8 * 8 + 64 < topkCount_) {
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64],
                      neg, topkCount_ - (validS2Len / 8 * 8 + 64));
        }

        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                             indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        
        // 是否返回Value值
        if (returnValueFlag && (std::is_same<SCORE_T, uint16_t>::value)) {
            // uint16_t -> bfloat16
            LocalTensor<uint16_t> scoreOutLocal1_ = scoreOutLocal_.template ReinterpretCast<uint16_t>();
            liV2Vector1::UIntToFloatReturnValue(uIntToBfloat16Local_, scoreOutLocal1_, topkCountAlign256_);
            // bfloat16 -> K_T
            Cast(valueOutLocal_, uIntToBfloat16Local_, RoundMode::CAST_NONE, topkCountAlign256_);
            // 无效值刷0
            if (validS2Len < topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % 16);
                PipeBarrier<PIPE_V>();
                Duplicate(valueOutLocal_[validS2Len / 16 * 16], zeroK_T, mask, 1, 1, 0);
            }
        
            if (validS2Len / 16 * 16 + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(valueOutLocal_[validS2Len / 16 * 16 + 64], zeroK_T,
                          topkCount_ - (validS2Len / 16 * 16 + 64));
            }
            AscendC::DataCopyParams copyOutValueParams;
            copyOutValueParams.blockCount = 1;
            copyOutValueParams.blockLen = topkCount_ * sizeof(K_T); // bytes
            copyOutValueParams.srcStride = 0;
            copyOutValueParams.dstStride = 0;
            // 搬运到GM
            AscendC::DataCopyPad(valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_],
            valueOutLocal_, copyOutValueParams);
        }
    }
}
}  // namespace LIKernel
#endif