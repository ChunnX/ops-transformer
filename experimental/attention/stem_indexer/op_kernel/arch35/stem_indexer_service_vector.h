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
 * \file stem_indexer_service_vector.h
 * \brief
 */
#ifndef stem_indexer_SERVICE_VECTOR_H
#define stem_indexer_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stem_indexer_common.h"
#include "../arch35/vf/stem_indexer_vector1.h"
#include "../arch35/vf/stem_indexer_topk.h"

namespace SIKernel {
using namespace SICommon;
constexpr uint32_t TRUNK_LEN_16K = 16384;
constexpr uint32_t TRUNK_LEN_256 = 256;
template <typename SIT>
class SIVector {
public:
    // =================================类型定义区=================================
    static constexpr SI_LAYOUT Q_LAYOUT_T = SIT::layout;
    static constexpr SI_LAYOUT K_LAYOUT_T = SIT::keyLayout;

    using QK_T = float32_t;
    using SCORE_T = typename SIT::scoreType;

    __aicore__ inline SIVector(){};
    __aicore__ inline void ProcessVec1(const SICommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const SICommon::RunInfo &info, bool isFirstS2InnerLoop, bool isLastS2InnerLoop);
    __aicore__ inline void ProcessDirectOutput(const SICommon::TempLoopInfo &tempLoopInfo, uint32_t gS1Idx);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct SICommon::ConstInfo &constInfo,
                                      const StemIndexerTilingData *__restrict tilingData);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<float> vbiasGm,
                                              GlobalTensor<int32_t> numPromptTokensGm,
                                              GlobalTensor<int32_t> sparseIndicesGm,
                                              GlobalTensor<int32_t> sparseSeqLenGm);
    __aicore__ inline void initS2LenToNeg(uint32_t gS1Idx, uint32_t actMBaseSize, uint32_t actS1Size,
                                          uint64_t indiceOutOffset);
    __aicore__ inline void SetSparseSeqLenZero(uint64_t indiceLenOffset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<float> vBiasGm;
    GlobalTensor<int32_t> numPromptTokensGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> sparseSeqLenGm;
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;

    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t KSCALE_S_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;
    static constexpr uint32_t MAX_TOPK_COUNT = 256;
    static constexpr uint32_t TOPK_ALIGN_SIZE = 256;
    static constexpr uint32_t TOPK_ROW_BATCH = 4;
    static constexpr uint32_t VEC_REPEAT_SIZE = 64;
    static constexpr int32_t K_SMALL_SEQ_MAX = 56;
    static constexpr int32_t K_MEDIUM_SEQ_MAX = 160;
private:
    __aicore__ inline uint32_t CalcDynamicTopkCount(uint32_t s1Idx, int32_t kbOffset, int32_t numPromptK);
    __aicore__ inline void WriteDirectIndices(uint32_t outputLen, uint64_t baseLenOffset,
                                              uint64_t baseOutOffset, uint32_t curAivGSize,
                                              int64_t curAivGS1ProcNum,
                                              uint32_t actS1Size, uint32_t s1Idx,
                                              LocalTensor<uint32_t>& indicesOutLocal);
    __aicore__ inline void CopyOutFourRows(const SICommon::RunInfo &info,
                                           const SICommon::RowIdx4 &rowIdx4,
                                           const SICommon::TopkNum4 &topkNum4,
                                           uint32_t batchRowNum, int64_t curAivGS1Idx,
                                           bool isLastS2InnerLoop, AscendC::DataCopyParams &copyOutParams,
                                           AscendC::DataCopyParams &indiceLenCopyParams,
                                           LocalTensor<uint32_t>& indicesOutLocal);
    __aicore__ inline void RunTopKFourRows(const SICommon::RunInfo &info, bool isFirstS2InnerLoop,
                                           bool isLastS2InnerLoop,
                                           const SICommon::RowIdx4 &rowIdx4,
                                           const SICommon::TopkNum4 &topkNum4,
                                           const SICommon::S2ValidLen4 &s2ValidLen4,
                                           uint32_t batchRowNum,
                                           int64_t curAivGS1Idx,
                                           AscendC::DataCopyParams &copyOutParams,
                                           AscendC::DataCopyParams &indiceLenCopyParams);
    // ================================Local Buffer区====================================
    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1Local_;
    // tmp buff for vbias
    TBuf<TPosition::VECCALC> vBiasBuf_;
    LocalTensor<float> vBiasLocal_;

    TBuf<TPosition::VECCALC> topkIndexBuf_;
    LocalTensor<uint32_t> topkIndexLocal_;

    TBuf<TPosition::VECCALC> topkValueBuf_;
    LocalTensor<uint32_t> topkValueLocal_;

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> globalIndexBuf_;
    LocalTensor<uint32_t> globalIndexLocal_;

    TBuf<TPosition::VECCALC> indiceLenBuf_;
    LocalTensor<uint32_t> indiceLenLocal_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    uint32_t trunkLen_ = 0;
    int32_t mBaseSize_ = 0;
    float alpha_ = 1.0f;
    float kBlockNumRateMedium_ = 0.2f;
    uint32_t kBlockNumBiasMedium_ = 30U;
    float kBlockNumRateLarge_ = 0.1f;
    uint32_t kBlockNumBiasLarge_ = 30U;
    uint32_t initialBlocks_ = 4U;
    uint32_t windowSize_ = 4U;
    struct SICommon::ConstInfo constInfo_;
    SIKernel::SITopk<SCORE_T> topkOp_;
};

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(QK_T));   //大小：2(开dB) * 2 * 64 * 256 * 4 = 256KB
    resMm1Local_ = resMm1Buf_.Get<QK_T>();// qk

    pipe->InitBuffer(vBiasBuf_, 2 * s2BaseSize_ * sizeof(float));   // 大小：2(开dB) * kbSize(256) * 4 = 4KB 
    vBiasLocal_ = vBiasBuf_.Get<float>();                           // vBias

    // Topk
    pipe->InitBuffer(mrgValueBuf_, CeilDiv(constInfo_.mBaseSize, 2) * (MAX_TOPK_COUNT + trunkLen_) * sizeof(SCORE_T));     // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(SCORE_T)
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();

    pipe->InitBuffer(globalIndexBuf_, CeilDiv(constInfo_.mBaseSize, 2) * MAX_TOPK_COUNT * sizeof(uint32_t));     // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(SCORE_T)
    globalIndexLocal_ = globalIndexBuf_.Get<uint32_t>();

    // indice_len
    pipe->InitBuffer(indiceLenBuf_, CeilDiv(constInfo_.mBaseSize, 2) * sizeof(uint32_t));
    indiceLenLocal_ = indiceLenBuf_.Get<uint32_t>();
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitParams(const struct SICommon::ConstInfo &constInfo,
                                                   const StemIndexerTilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;  // 4
    s2BaseSize_ = constInfo.s2BaseSize;  // 256
    mBaseSize_ = constInfo.mBaseSize;
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    alpha_ = constInfo.alpha;
    kBlockNumRateMedium_ = constInfo.kBlockNumRateMedium;
    kBlockNumBiasMedium_ = constInfo.kBlockNumBiasMedium;
    kBlockNumRateLarge_ = constInfo.kBlockNumRateLarge;
    kBlockNumBiasLarge_ = constInfo.kBlockNumBiasLarge;
    initialBlocks_ = constInfo.initialBlocks;
    windowSize_ = constInfo.windowSize;
    blockId_ = GetBlockIdx();
    trunkLen_ = TRUNK_LEN_256;
    topkOp_.Init(MAX_TOPK_COUNT, trunkLen_);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitVecInputTensor(GlobalTensor<float> vBiasGm,
                                                        GlobalTensor<int32_t> numPromptTokensGm,
                                                        GlobalTensor<int32_t> sparseIndicesGm,
                                                        GlobalTensor<int32_t> sparseSeqLenGm)
{
    this->vBiasGm = vBiasGm;
    this->numPromptTokensGm = numPromptTokensGm;
    this->sparseIndicesGm = sparseIndicesGm;
    this->sparseSeqLenGm = sparseSeqLenGm;
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::initS2LenToNeg(uint32_t gS1Idx, uint32_t actMBaseSize, uint32_t actS1Size,
                                                     uint64_t indiceOutOffset)
{
    if (actS1Size == 0U) {
        return;
    }
    // 当前处理的GS1行起始索引（全局位置）
    int64_t curGS1Idx = gS1Idx * mBaseSize_;
    // 当前GS1行需要处理的数量（可能小于mBaseSize_，处理尾块）
    int64_t curGS1ProcNum = actMBaseSize;
    // 当前AIV核处理的GS1起始索引（双核分工）
    int64_t curAivGS1Idx = curGS1Idx + (blockId_ % 2) * CeilDiv(curGS1ProcNum, 2);
    // 当前AIV核需要处理的GS1行数量
    int64_t curAivGS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curGS1ProcNum, 2) : curGS1ProcNum / 2;
    if (curAivGS1ProcNum == 0) {
        return;
    }
    uint32_t curGlobalGIdx = curAivGS1Idx / actS1Size;
    uint32_t curGlobalS1Idx = curAivGS1Idx % actS1Size;
    int64_t lastAivGS1Idx = curAivGS1Idx + curAivGS1ProcNum - 1;
    uint32_t lastGlobalGIdx = lastAivGS1Idx / actS1Size;
    uint32_t lastGlobalS1Idx = lastAivGS1Idx % actS1Size;
    uint64_t startPhysicalRow = static_cast<uint64_t>(curGlobalGIdx) * constInfo_.qSeqSize + curGlobalS1Idx;
    uint64_t endPhysicalRow = static_cast<uint64_t>(lastGlobalGIdx) * constInfo_.qSeqSize + lastGlobalS1Idx;
    uint64_t clearRowNum = endPhysicalRow - startPhysicalRow + 1;
    int32_t neg = -1;
    uint64_t outSplit1Offset = indiceOutOffset + startPhysicalRow * constInfo_.kSeqSize;
    GlobalTensor<int32_t> indiceSplitOut = sparseIndicesGm[outSplit1Offset];
    AscendC::InitGlobalMemory(indiceSplitOut, clearRowNum * constInfo_.kSeqSize, neg);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::SetSparseSeqLenZero(uint64_t indiceLenOffset)
{
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    Duplicate(indiceLenLocal_, 0U, 1);
    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    AscendC::DataCopyParams indiceLenCp;
    indiceLenCp.blockCount = 1;
    indiceLenCp.blockLen = static_cast<uint16_t>(sizeof(uint32_t));
    indiceLenCp.srcStride = 0;
    indiceLenCp.dstStride = 0;
    DataCopyPad(sparseSeqLenGm[indiceLenOffset], indiceLenLocal_.ReinterpretCast<int32_t>(), indiceLenCp);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::ProcessVec1(const SICommon::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
    int64_t gS1BasePerVecSize_ = CeilDiv(constInfo_.mBaseSize, 2); // 当前每个vec核上分到的mbase/2行
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curMProcNum = info.actMBaseSize;
    int64_t curAivMProcNum = (blockId_ % 2 == 0) ? CeilDiv(curMProcNum, 2) : curMProcNum / 2;

    if (curAivMProcNum == 0) {
        // Vector waits for Cube to finish mm1 and move mm1 result to UB.
        CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_CV_EVENT + pingpong);
        // Vector notifies Cube after processing is complete.
        CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_VC_EVENT + pingpong);
        return;
    }
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    // vBiasGm --> vBiasLocal_ 搬运vBias
    uint64_t vBiasGmOffset = info.tensorVBiasOffset + curS2Idx;
    DataCopyPadExtParams<float> padVBiasParams{false, 0, 0, 0};
    DataCopyExtParams vBiasDataCopyExtParams;
    vBiasDataCopyExtParams.blockCount = 1;
    vBiasDataCopyExtParams.blockLen = s2BaseSize_ * sizeof(float);
    vBiasDataCopyExtParams.srcStride = 0;
    vBiasDataCopyExtParams.dstStride = 0;
    DataCopyPad(vBiasLocal_[pingpong * s2BaseSize_], vBiasGm[vBiasGmOffset], vBiasDataCopyExtParams, padVBiasParams);

    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);

    // C/V sync: Vector waits for Cube to finish mm1 and move mm1 result to UB.
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_CV_EVENT + pingpong);

    auto qkBase = resMm1Local_[pingpong * (gS1BasePerVecSize_ * s2BaseSize_)];
    auto outBase = mrgValueLocal_[MAX_TOPK_COUNT];
    auto vBiasBase = vBiasLocal_[pingpong * s2BaseSize_];

    vector1::MulRSquareAndAddVBiasVF(outBase, qkBase, vBiasBase, constInfo_.rSquare, gS1BasePerVecSize_,
                                     s2BaseSize_, (trunkLen_ + MAX_TOPK_COUNT));
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
}

template <typename SIT>
__aicore__ inline uint32_t SIVector<SIT>::CalcDynamicTopkCount(
    uint32_t s1Idx, int32_t kbOffset, int32_t numPromptK)
{
    // 计算 kStart（分段函数）
    int32_t kStart;
    if (numPromptK < K_SMALL_SEQ_MAX) {
        kStart = numPromptK;
    } else if (numPromptK < K_MEDIUM_SEQ_MAX) {
        kStart = static_cast<int32_t>(numPromptK * kBlockNumRateMedium_ + kBlockNumBiasMedium_);
    } else {
        kStart = static_cast<int32_t>(numPromptK * kBlockNumRateLarge_ + kBlockNumBiasLarge_);
    }
    
    // 计算 s1Pos 和 decayLen
    int32_t s1Pos = static_cast<int32_t>(s1Idx) + kbOffset;
    int32_t decayLen = numPromptK - kStart;
    
    // 边界条件：s1Pos < kStart 或 decayLen < 1 时，直接返回 kStart
    if (s1Pos < kStart || decayLen < 1) {
        return static_cast<uint32_t>(kStart);
    }
    
    // 计算 k_end
    int32_t k_end = static_cast<int32_t>(kStart * alpha_);
    
    // 计算插值系数 t
    float t = static_cast<float>(s1Pos - kStart) / static_cast<float>(decayLen - 1);
    
    // 计算 dynamicTopkCount
    uint32_t dynamicTopkCount = static_cast<uint32_t>(static_cast<float>(kStart) +
                                                      t * static_cast<float>(k_end - kStart));
    
    // 限制范围：下限为 1，上限为 kStart
    dynamicTopkCount = dynamicTopkCount < 1 ? 1 :
        (dynamicTopkCount > static_cast<uint32_t>(kStart) ? static_cast<uint32_t>(kStart) : dynamicTopkCount);
    
    return dynamicTopkCount;
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::WriteDirectIndices(
    uint32_t outputLen, uint64_t baseLenOffset, uint64_t baseOutOffset, uint32_t curAivGSize,
    int64_t curAivGS1ProcNum, uint32_t actS1Size, uint32_t s1Idx,
    LocalTensor<uint32_t>& indicesOutLocal)
{
    outputLen = Min(outputLen, MAX_TOPK_COUNT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

    Duplicate(indiceLenLocal_, outputLen, 1);
    CreateVecIndex(indicesOutLocal.ReinterpretCast<int32_t>(), 0, outputLen);
    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    AscendC::DataCopyParams indiceLenCp;
    indiceLenCp.blockCount = 1;
    indiceLenCp.blockLen = static_cast<uint16_t>(sizeof(uint32_t));
    indiceLenCp.srcStride = 0;
    indiceLenCp.dstStride = 0;
    AscendC::DataCopyParams copyOut;
    copyOut.blockCount = 1;
    copyOut.blockLen = static_cast<uint16_t>(outputLen * sizeof(int32_t));
    copyOut.srcStride = 0;
    copyOut.dstStride = 0;
    for (uint32_t gIdx = 0; gIdx < curAivGSize; gIdx++) {
        uint32_t mInnerIdx = gIdx * actS1Size + s1Idx;
        if (mInnerIdx >= curAivGS1ProcNum) {
            break; 
        }
        DataCopyPad(sparseSeqLenGm[baseLenOffset + static_cast<uint64_t>(gIdx) * constInfo_.qSeqSize],
                    indiceLenLocal_.ReinterpretCast<int32_t>(), indiceLenCp);
        DataCopyPad(sparseIndicesGm[baseOutOffset +
                        static_cast<uint64_t>(gIdx) * constInfo_.qSeqSize * constInfo_.kSeqSize],
                    indicesOutLocal.ReinterpretCast<int32_t>(), copyOut);
    }
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::CopyOutFourRows(const SICommon::RunInfo &info,
                                                      const SICommon::RowIdx4 &rowIdx4,
                                                      const SICommon::TopkNum4 &topkNum4,
                                                      uint32_t batchRowNum, int64_t curAivGS1Idx,
                                                      bool isLastS2InnerLoop,
                                                      AscendC::DataCopyParams &copyOutParams,
                                                      AscendC::DataCopyParams &indiceLenCopyParams,
                                                      LocalTensor<uint32_t>& indicesOutLocal)
{
    if (!isLastS2InnerLoop) {
        return;
    }

    for (uint32_t rowIdx = 0; rowIdx < batchRowNum; rowIdx++) {
        uint32_t mInnerIdx = SICommon::GetLane(rowIdx4, rowIdx);
        uint32_t nowCurAivGS1Idx = curAivGS1Idx + mInnerIdx;
        uint32_t globalGIdx = nowCurAivGS1Idx / info.actS1Size;
        uint32_t globalS1Idx = nowCurAivGS1Idx % info.actS1Size;
        uint32_t curS1RealS2Len = info.actS2Size;
        if (constInfo_.attenMaskFlag) {
            int32_t curS1RealS2LenTmp = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size) +
                                        static_cast<int32_t>(globalS1Idx) + 1;
            curS1RealS2Len = static_cast<uint32_t>(curS1RealS2LenTmp > 0 ? curS1RealS2LenTmp : 0);
        }

        uint32_t rawTopkSelectNum = SICommon::GetLane(topkNum4, rowIdx);
        uint32_t topkSelectNum = (rawTopkSelectNum > MAX_TOPK_COUNT) ? MAX_TOPK_COUNT : rawTopkSelectNum;
        uint32_t mrgValueOffset = mInnerIdx * (MAX_TOPK_COUNT + trunkLen_);
        uint32_t globalIndexOffset = mInnerIdx * MAX_TOPK_COUNT;
        uint64_t baseLenOffset = info.indiceLenOffset +
                                 static_cast<uint64_t>(globalGIdx) * constInfo_.qSeqSize + globalS1Idx;
        uint64_t baseOutOffset = info.indiceOutOffset +
                                (static_cast<uint64_t>(globalGIdx) * constInfo_.qSeqSize + globalS1Idx) *
                                    constInfo_.kSeqSize;

        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        PipeBarrier<PIPE_V>();
        if (initialBlocks_ > 0) {
            CreateVecIndex(mrgValueLocal_[mrgValueOffset].template ReinterpretCast<int32_t>(),
                           0, initialBlocks_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            copyOutParams.blockLen = static_cast<uint16_t>(initialBlocks_ * sizeof(int32_t));
            DataCopyPad(sparseIndicesGm[baseOutOffset],
                        mrgValueLocal_[mrgValueOffset].template ReinterpretCast<int32_t>(), copyOutParams);
        }

        copyOutParams.blockLen = static_cast<uint16_t>(topkSelectNum * sizeof(int32_t));
        DataCopyPad(sparseIndicesGm[baseOutOffset + initialBlocks_],
                    globalIndexLocal_[globalIndexOffset].ReinterpretCast<int32_t>(), copyOutParams);

        if (windowSize_ > 0) {
            int32_t windowBase = curS1RealS2Len - windowSize_;
            CreateVecIndex(mrgValueLocal_[mrgValueOffset + TOPK_ALIGN_SIZE].template ReinterpretCast<int32_t>(),
                           windowBase, windowSize_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            copyOutParams.blockLen = static_cast<uint16_t>(windowSize_ * sizeof(int32_t));
            DataCopyPad(sparseIndicesGm[baseOutOffset + initialBlocks_ + topkSelectNum],
                        mrgValueLocal_[mrgValueOffset + TOPK_ALIGN_SIZE].template ReinterpretCast<int32_t>(),
                        copyOutParams);
        }

        uint32_t totalOutput = initialBlocks_ + topkSelectNum + windowSize_;
        Duplicate(indiceLenLocal_, totalOutput, 1);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        DataCopyPad(sparseSeqLenGm[baseLenOffset],
                    indiceLenLocal_.ReinterpretCast<int32_t>(), indiceLenCopyParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::RunTopKFourRows(
    const SICommon::RunInfo &info, bool isFirstS2InnerLoop, bool isLastS2InnerLoop,
    const SICommon::RowIdx4 &rowIdx4,
    const SICommon::TopkNum4 &topkNum4,
    const SICommon::S2ValidLen4 &s2ValidLen4,
    uint32_t batchRowNum, int64_t curAivGS1Idx,
    AscendC::DataCopyParams &copyOutParams, AscendC::DataCopyParams &indiceLenCopyParams)
{

    // WriteDirectIndices 与 Batch4Rows 复用 resMm1Local_，先确保上一笔 direct 搬出已读完 UB。
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

    // padding（各行按自身 s2BlockValidLen 补齐到 trunkLen，非零行结果一致）
    if (isFirstS2InnerLoop) {
        vector1::PadMrgFourRowsFirstS2RangeVF<SCORE_T>(mrgValueLocal_, rowIdx4, topkNum4, s2ValidLen4,
                                              batchRowNum, MAX_TOPK_COUNT + trunkLen_,
                                              trunkLen_);
    } else {
        vector1::PadMrgFourRowsNotFirstS2RangeVF<SCORE_T>(mrgValueLocal_, rowIdx4, topkNum4, s2ValidLen4,
                                                 batchRowNum,
                                                 MAX_TOPK_COUNT + trunkLen_, trunkLen_);
    }

    uint32_t inputOffset = isFirstS2InnerLoop ? TOPK_ALIGN_SIZE : 0U;
    uint32_t validLen = isFirstS2InnerLoop ? trunkLen_ : (TOPK_ALIGN_SIZE + trunkLen_);

    LocalTensor<uint32_t> reuseMm1ResLocal =
        resMm1Local_[(info.loop % 2) * (CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_)].ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> indicesOutLocal = reuseMm1ResLocal[SIKernel::SITopk<SCORE_T>::REUSE_INDICES_OUT_OFFSET];
    LocalTensor<SCORE_T> scoreOutLocal =
        resMm1Local_[(info.loop % 2) * (CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_)]
            .template ReinterpretCast<SCORE_T>()[SIKernel::SITopk<SCORE_T>::REUSE_SCORE_OUT_OFFSET];
    PipeBarrier<PIPE_V>();

    topkOp_.Batch4Rows(mrgValueLocal_, indicesOutLocal, scoreOutLocal,
                       reuseMm1ResLocal, globalIndexLocal_,
                       rowIdx4, topkNum4,
                       batchRowNum, MAX_TOPK_COUNT + trunkLen_,
                       inputOffset, validLen,
                       info.s2Idx, info.s2LoopEnd + 1);

    CopyOutFourRows(info, rowIdx4, topkNum4,
                    batchRowNum, curAivGS1Idx,
                    isLastS2InnerLoop, copyOutParams, indiceLenCopyParams, indicesOutLocal);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::ProcessTopK(
    const SICommon::RunInfo &info, bool isFirstS2InnerLoop, bool isLastS2InnerLoop)
{
    // 当前处理的GS1行起始索引（全局位置）
    int64_t curGS1Idx = info.gS1Idx * mBaseSize_;
    // 当前GS1行需要处理的数量（可能小于mBaseSize_，处理尾块）
    int64_t curGS1ProcNum = info.actMBaseSize;
    // 当前AIV核处理的GS1起始索引（双核分工）
    int64_t curAivGS1Idx = curGS1Idx + (blockId_ % 2) * CeilDiv(curGS1ProcNum, 2);
    // 当前AIV核需要处理的GS1行数量
    int64_t curAivGS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curGS1ProcNum, 2) : curGS1ProcNum / 2;

    // S2 基本块信息
    uint32_t s2BlockStart = info.s2Idx * s2BaseSize_;

    // indice输出搬运参数
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    // indiceLen搬出参数
    AscendC::DataCopyParams indiceLenCopyParams;
    
    // 计算动态 topkCount 所需的块级常量（使用原始 actS2Size）
    int32_t kbOffset = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size);
    int32_t numPromptK = static_cast<int32_t>(info.promptLen);
    
    uint32_t curAivGSize = CeilDiv(curAivGS1ProcNum, info.actS1Size);
    // 当前核处理的 S1 轴大小
    // 当mBaseSize小于actS1Size时，curAivS1Size为curAivGS1ProcNum
    // 当mBaseSize大于等于actS1Size时， curAivS1Size为info.actS1Size
    uint32_t curAivS1Size = Min(info.actS1Size, static_cast<uint32_t>(curAivGS1ProcNum));
    LocalTensor<uint32_t> directIndicesOutLocal =
        resMm1Local_[(info.loop % 2) * (CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_)]
            .ReinterpretCast<uint32_t>()[SIKernel::SITopk<uint32_t>::REUSE_INDICES_OUT_OFFSET];

    indiceLenCopyParams.blockCount = 1;
    indiceLenCopyParams.blockLen = static_cast<uint16_t>(sizeof(uint32_t));
    indiceLenCopyParams.srcStride = 0;
    indiceLenCopyParams.dstStride = 0;

    if (isFirstS2InnerLoop && initialBlocks_ > 0) {
        vector1::PadMrgRowsInitBlockVF(mrgValueLocal_, MAX_TOPK_COUNT + trunkLen_,
                                       static_cast<uint32_t>(curAivGS1ProcNum), TOPK_ALIGN_SIZE, initialBlocks_);
    }

    SICommon::RowIdx4 rowIdx4 = {0U, 0U, 0U, 0U};
    SICommon::TopkNum4 topkNum4 = {0U, 0U, 0U, 0U};
    SICommon::S2ValidLen4 s2ValidLen4 = {0U, 0U, 0U, 0U};
    uint32_t batchRowNum = 0;
    
    for (uint32_t s1Idx = 0; s1Idx < curAivS1Size; s1Idx++) {
        uint32_t nowCurAivGS1Idx = curAivGS1Idx + s1Idx;
        uint32_t globalGIdx = nowCurAivGS1Idx / info.actS1Size;
        uint32_t globalS1Idx = nowCurAivGS1Idx % info.actS1Size;
        uint32_t curS1RealS2Len = info.actS2Size;
        if (constInfo_.attenMaskFlag) {
            int32_t curS1RealS2LenTmp = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size) +
                                        static_cast<int32_t>(globalS1Idx) + 1;
            curS1RealS2Len = static_cast<uint32_t>(curS1RealS2LenTmp > 0 ? curS1RealS2LenTmp : 0);
        }

        uint32_t dynamicTopkCount = CalcDynamicTopkCount(globalS1Idx, kbOffset, numPromptK);
        uint32_t topkSelectNum = Min(dynamicTopkCount, MAX_TOPK_COUNT);
        uint32_t totalOutput = initialBlocks_ + topkSelectNum + windowSize_;
        if ((constInfo_.attenMaskFlag && (curS1RealS2Len <= initialBlocks_ + windowSize_)) ||
            (curS1RealS2Len <= totalOutput)) {
            if (isFirstS2InnerLoop) {
                uint64_t baseLenOffset = info.indiceLenOffset +
                                        static_cast<uint64_t>(globalGIdx) * constInfo_.qSeqSize + globalS1Idx;
                uint64_t baseOutOffset = info.indiceOutOffset +
                                        (static_cast<uint64_t>(globalGIdx) * constInfo_.qSeqSize + globalS1Idx) *
                                            constInfo_.kSeqSize;
                WriteDirectIndices(curS1RealS2Len, baseLenOffset, baseOutOffset, curAivGSize,
                                curAivGS1ProcNum, info.actS1Size, s1Idx, directIndicesOutLocal);
            }
            continue;
        }

        uint32_t s2ValidEnd = Max(curS1RealS2Len - windowSize_, 0);
        uint32_t s2BlockEnd = Min(s2BlockStart + s2BaseSize_, s2ValidEnd);
        uint32_t s2BlockValidLen = (s2BlockEnd > s2BlockStart) ? (s2BlockEnd - s2BlockStart) : 0;

        for (uint32_t gIdx = 0; gIdx < curAivGSize; gIdx++) {
            uint32_t mInnerIdx = gIdx * info.actS1Size + s1Idx;
            if (mInnerIdx >= curAivGS1ProcNum) {
                break;
            }

            SICommon::SetLane(rowIdx4, batchRowNum, mInnerIdx);
            SICommon::SetLane(topkNum4, batchRowNum, topkSelectNum);
            SICommon::SetLane(s2ValidLen4, batchRowNum, s2BlockValidLen);
            batchRowNum++;
            if (batchRowNum == TOPK_ROW_BATCH) {
                RunTopKFourRows(info, isFirstS2InnerLoop, isLastS2InnerLoop,
                                rowIdx4, topkNum4, s2ValidLen4, batchRowNum,
                                curAivGS1Idx, copyOutParams, indiceLenCopyParams);
                batchRowNum = 0;
            }
        }
    }
    if (batchRowNum > 0U) {
        uint32_t tailLane = batchRowNum - 1U;
        uint32_t tailRowIdx = SICommon::GetLane(rowIdx4, tailLane);
        uint32_t tailTopkNum = SICommon::GetLane(topkNum4, tailLane);
        uint32_t tailS2ValidLen = SICommon::GetLane(s2ValidLen4, tailLane);
        for (uint32_t lane = batchRowNum; lane < TOPK_ROW_BATCH; lane++) {
            SICommon::SetLane(rowIdx4, lane, tailRowIdx);
            SICommon::SetLane(topkNum4, lane, tailTopkNum);
            SICommon::SetLane(s2ValidLen4, lane, tailS2ValidLen);
        }
        RunTopKFourRows(info, isFirstS2InnerLoop, isLastS2InnerLoop,
                        rowIdx4, topkNum4, s2ValidLen4, batchRowNum,
                        curAivGS1Idx, copyOutParams, indiceLenCopyParams);
    }
    CrossCoreSetFlag<SICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(SICommon::ConstInfo::CROSS_VC_EVENT + (info.loop % 2));   //V核处理完，通知C核可以把mm1Res搬运到UB
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::ProcessDirectOutput(const SICommon::TempLoopInfo &tempLoopInfo, uint32_t gS1Idx)
{
    if ASCEND_IS_AIV {
        uint32_t actMBaseSize = tempLoopInfo.actMBaseSize;
        uint32_t actS1Size = tempLoopInfo.actS1Size;
        uint32_t actS2Size = tempLoopInfo.actS2Size;
        uint64_t qFlatBase = static_cast<uint64_t>(tempLoopInfo.bIdx) * constInfo_.qSeqSize * constInfo_.qHeadNum
                        + static_cast<uint64_t>(tempLoopInfo.n2Idx) * constInfo_.gSize * constInfo_.qSeqSize;
        uint64_t indiceLenOffset = qFlatBase;
        uint64_t indiceOutOffset = qFlatBase * constInfo_.kSeqSize;

        initS2LenToNeg(gS1Idx, actMBaseSize, actS1Size, indiceOutOffset);

        LocalTensor<uint32_t> indicesOutLocal =
            resMm1Local_[0].ReinterpretCast<uint32_t>()[SIKernel::SITopk<SCORE_T>::REUSE_INDICES_OUT_OFFSET];

        // 当前处理的gS1行起始索引（全局位置）
        int64_t curGS1Idx = gS1Idx * mBaseSize_; 
        // 当前gS1行需要处理的数量（可能小于mBaseSize_，处理尾块）
        int64_t curGS1ProcNum = actMBaseSize;
        // 当前AIV核处理的gS1起始索引（双核分工）
        int64_t curAivGS1Idx = curGS1Idx + (blockId_ % 2) * CeilDiv(curGS1ProcNum, 2);
        // 当前AIV核需要处理的gS1行数量
        int64_t curAivGS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curGS1ProcNum, 2) : curGS1ProcNum / 2;
        
        uint32_t curS1RealS2Len = actS2Size;
        uint32_t curAivGSize = CeilDiv(curAivGS1ProcNum, actS1Size);
        // 当前核处理的 S1 轴大小
        // 当mBaseSize小于actS1Size时，curAivS1Size为curAivGS1ProcNum
        // 当mBaseSize大于等于actS1Size时， curAivS1Size为info.actS1Size
        uint32_t curAivS1Size = Min(actS1Size, static_cast<uint32_t>(curAivGS1ProcNum));
        for (uint32_t s1Idx = 0; s1Idx < curAivS1Size; s1Idx++) {
            uint32_t nowCurAivGS1Idx = curAivGS1Idx + s1Idx;
            uint32_t globalGIdx = nowCurAivGS1Idx / actS1Size;
            uint32_t globalS1Idx = nowCurAivGS1Idx % actS1Size;
            uint64_t baseLenOffset = indiceLenOffset +
                                     static_cast<uint64_t>(globalGIdx) * constInfo_.qSeqSize + globalS1Idx;
            uint64_t baseOutOffset = indiceOutOffset +
                                    (static_cast<uint64_t>(globalGIdx) * constInfo_.qSeqSize + globalS1Idx) *
                                        constInfo_.kSeqSize;
            if (constInfo_.attenMaskFlag) {
                int32_t curS1RealS2LenTmp = static_cast<int32_t>(actS2Size) - static_cast<int32_t>(actS1Size) +
                                            static_cast<int32_t>(globalS1Idx) + 1;
                curS1RealS2Len = static_cast<uint32_t>(curS1RealS2LenTmp > 0 ? curS1RealS2LenTmp : 0);
            }
            WriteDirectIndices(curS1RealS2Len, baseLenOffset, baseOutOffset, curAivGSize,
                               curAivGS1ProcNum, actS1Size, s1Idx, indicesOutLocal);
        }
    }
}
}  // namespace SIKernel
#endif
