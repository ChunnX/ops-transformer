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
#include "stem_indexer_common.h"
#include "vf/stem_indexer_vector1.h"
#include "vf/stem_indexer_topk.h"
#include "vf/stem_indexer_topk_utils.h"

namespace SIKernel {
using namespace SICommon;
template <typename SIT>
class SIVector {
public:
    // =================================类型定义区=================================
    using QK_T = float32_t;
    using SCORE_T = typename SIT::scoreType;

    __aicore__ inline SIVector(){};
    __aicore__ inline void ProcessVec1(const SICommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const SICommon::RunInfo &info, bool isFirstS2InnerLoop, bool isLastS2InnerLoop);
    __aicore__ inline void ProcessDirectOutput(const SICommon::TempLoopInfo &tempLoopInfo, uint32_t gS1Idx);
    __aicore__ inline void InitBuffers(TPipe *pipe, bool needCleanOutput);
    __aicore__ inline void InitParams(const struct SICommon::ConstInfo &constInfo, uint32_t logicalCoreIdx);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<float> vbiasGm, GlobalTensor<int32_t> sparseIndicesGm,
                                              GlobalTensor<int32_t> sparseSeqLenGm);
    __aicore__ inline void InitSparseIndicesToNegOne(uint32_t gS1Idx, uint32_t actMBaseSize, uint32_t actS1Size,
                                                     uint64_t indiceOutOffset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<float> vBiasGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> sparseSeqLenGm;
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT = EVENT_ID1;

    static constexpr uint32_t TOPK_MTE3_MTE2_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t MAX_TOPK_COUNT = 256;
    static constexpr uint32_t TOPK_ALIGN_SIZE = 256;
    static constexpr uint32_t TOPK_ROW_BATCH = 4;
    static constexpr uint32_t PACKED_ROW_STRIDE = MAX_TOPK_COUNT + 8U;
    static constexpr uint32_t OUTPUT_INIT_ALIGN_ELEMS = 512U / sizeof(int32_t);
    static constexpr int32_t K_SMALL_SEQ_MAX = 56;
    static constexpr int32_t K_MEDIUM_SEQ_MAX = 160;

    static constexpr uint32_t AIV_CORE_NUM = 64U;
    static constexpr uint32_t RES_MM1_SLOT_BYTES = 96U * 256U * sizeof(QK_T);
    static constexpr uint32_t RES_MM1_DOUBLE_BUFFER_BYTES = 2U * RES_MM1_SLOT_BYTES;
    static constexpr uint32_t VBIAS_BYTES = 2U * 256U * sizeof(float);
    static constexpr uint32_t HISTORY_SCORE_BYTES = 96U * 256U * sizeof(SCORE_T);
    static constexpr uint32_t GLOBAL_INDEX_BYTES = 96U * 256U * sizeof(uint32_t);
    static constexpr uint32_t TOPK_MERGE_OFFSET_BYTES = 0x0C000U;
    static constexpr uint32_t TOPK_WORKSPACE_OFFSET_BYTES = 0x0D000U;
    static constexpr uint32_t PACKED_OUTPUT_SLOT0_OFFSET_BYTES = 0x12C00U;
    static constexpr uint32_t PACKED_OUTPUT_SLOT1_OFFSET_BYTES = 0x13C80U;

private:
    __aicore__ inline void CleanOutput();
    __aicore__ inline void InitOutputSingleCore(GlobalTensor<int32_t> &outputGm, uint64_t totalSize, int32_t value);
    __aicore__ inline uint32_t CalcDynamicTopkCount(uint32_t s1Idx, int64_t kbOffset, int32_t numPromptK);
    __aicore__ inline void WriteDirectIndices(uint32_t outputLen, uint64_t baseLenOffset, uint64_t baseOutOffset,
                                              uint32_t curAivGSize, int64_t curAivGS1ProcNum, uint32_t actS1Size,
                                              uint32_t s1Idx, LocalTensor<uint32_t> &indicesOutLocal,
                                              LocalTensor<uint32_t> &lengthLocal);
    __aicore__ inline void CopyOutBatchRows(const SICommon::RunInfo &info, int64_t curAivGS1Idx,
                                            const SICommon::RowIdx4 &rowIdx4, const SICommon::TopkNum4 &topkNum4,
                                            uint32_t batchRowNum, LocalTensor<uint32_t> &packedRowsLocal);
    __aicore__ inline void RunTopKBatchRows(const SICommon::RunInfo &info, bool isFirstS2InnerLoop,
                                            bool isLastS2InnerLoop, const SICommon::RowIdx4 &rowIdx4,
                                            const SICommon::TopkNum4 &topkNum4,
                                            const SICommon::S2ValidLen4 &s2ValidLen4, uint32_t batchRowNum,
                                            int64_t curAivGS1Idx, uint32_t packedSlotIdx);
    // ================================Local Buffer区====================================
    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1Local_;
    // tmp buff for vbias
    TBuf<TPosition::VECCALC> vBiasBuf_;
    LocalTensor<float> vBiasLocal_;

    // Persistent TopK history. Current scores, 4-row merge, TopK scratch and packed output reuse resMm1Buf_.
    TBuf<TPosition::VECCALC> historyValueBuf_;
    LocalTensor<SCORE_T> historyValueLocal_;

    TBuf<TPosition::VECCALC> globalIndexBuf_;
    LocalTensor<uint32_t> globalIndexLocal_;

    int32_t blockId_ = -1;
    bool needCleanOutput_ = false;
    // para for vector
    int32_t s2BaseSize_ = 0;
    uint32_t mrgRowStride_ = 0;
    int32_t mBaseSize_ = 0;
    float alpha_ = 1.0f;
    float kBlockNumRateMedium_ = 0.2f;
    uint32_t kBlockNumBiasMedium_ = 30U;
    float kBlockNumRateLarge_ = 0.1f;
    uint32_t kBlockNumBiasLarge_ = 30U;
    uint32_t initialBlocks_ = 4U;
    uint32_t windowSize_ = 4U;
    struct SICommon::ConstInfo constInfo_;
    SIKernel::StemIndexerTopk<SCORE_T> topkOp_;
};

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitBuffers(TPipe *pipe, bool needCleanOutput)
{
    needCleanOutput_ = needCleanOutput;
    // 2 * 96 * 256 * 4B = 192 KiB. Each 96 KiB slot is reused after in-place score conversion.
    pipe->InitBuffer(resMm1Buf_, RES_MM1_DOUBLE_BUFFER_BYTES);
    resMm1Local_ = resMm1Buf_.Get<QK_T>(); // qk

    // 当前配置为2 * 256 * 4B = 2KB。
    pipe->InitBuffer(vBiasBuf_, VBIAS_BYTES);
    vBiasLocal_ = vBiasBuf_.Get<float>(); // vBias

    // 96 rows * 256 uint16 = 48 KiB persistent history.
    pipe->InitBuffer(historyValueBuf_, HISTORY_SCORE_BYTES);
    historyValueLocal_ = historyValueBuf_.Get<SCORE_T>();

    // 96 rows * 256 uint32 = 96 KiB persistent global indices.
    pipe->InitBuffer(globalIndexBuf_, GLOBAL_INDEX_BYTES);
    globalIndexLocal_ = globalIndexBuf_.Get<uint32_t>();

    // 两个ping-pong槽初始均归Cube使用。
    CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_VC_EVENT + 0U);
    CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_VC_EVENT + 1U);
    if (needCleanOutput_) {
        CleanOutput();
    }
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::CleanOutput()
{
    uint64_t totalIndiceSize =
        constInfo_.batchSize * constInfo_.qHeadNum * constInfo_.qSeqSize * constInfo_.sparseCount;
    InitOutputSingleCore(sparseIndicesGm, totalIndiceSize, static_cast<int32_t>(SICommon::INVALID_IDX));
    SyncAll();
    uint64_t totalLenSize = constInfo_.batchSize * constInfo_.qHeadNum * constInfo_.qSeqSize;
    InitOutputSingleCore(sparseSeqLenGm, totalLenSize, static_cast<int32_t>(0));
    SyncAll();
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitOutputSingleCore(GlobalTensor<int32_t> &outputGm, uint64_t totalSize,
                                                           int32_t value)
{
    uint64_t aivCoreNum = AIV_CORE_NUM;
    uint64_t coreIdx = static_cast<uint64_t>(blockId_);
    uint64_t singleCoreSize =
        SICommon::Align((totalSize + aivCoreNum - 1U) / aivCoreNum, static_cast<uint64_t>(OUTPUT_INIT_ALIGN_ELEMS));
    uint64_t offset = coreIdx * singleCoreSize;
    if (offset < totalSize) {
        uint64_t initSize = Min(singleCoreSize, totalSize - offset);
        InitOutput<int32_t>(outputGm[offset], initSize, value);
    }
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitParams(const struct SICommon::ConstInfo &constInfo, uint32_t logicalCoreIdx)
{
    this->constInfo_ = constInfo;
    s2BaseSize_ = constInfo.s2BaseSize; // 256
    mBaseSize_ = constInfo.mBaseSize;
    alpha_ = constInfo.alpha;
    kBlockNumRateMedium_ = constInfo.kBlockNumRateMedium;
    kBlockNumBiasMedium_ = constInfo.kBlockNumBiasMedium;
    kBlockNumRateLarge_ = constInfo.kBlockNumRateLarge;
    kBlockNumBiasLarge_ = constInfo.kBlockNumBiasLarge;
    initialBlocks_ = constInfo.initialBlocks;
    windowSize_ = constInfo.windowSize;
    blockId_ = static_cast<int32_t>(logicalCoreIdx);
    // Compact 4-row merge layout: [history 256][current 256].
    mrgRowStride_ = MAX_TOPK_COUNT + SICommon::TRUNK_LEN_256;
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitVecInputTensor(GlobalTensor<float> vBiasGm,
                                                         GlobalTensor<int32_t> sparseIndicesGm,
                                                         GlobalTensor<int32_t> sparseSeqLenGm)
{
    this->vBiasGm = vBiasGm;
    this->sparseIndicesGm = sparseIndicesGm;
    this->sparseSeqLenGm = sparseSeqLenGm;
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitSparseIndicesToNegOne(uint32_t gS1Idx, uint32_t actMBaseSize,
                                                                uint32_t actS1Size, uint64_t indiceOutOffset)
{
    if (actS1Size == 0U) {
        return;
    }
    // 当前处理的GS1行起始索引（全局位置）
    int64_t curGS1Idx = gS1Idx * mBaseSize_;
    // 当前GS1行需要处理的数量（可能小于mBaseSize_，处理尾块）
    int64_t curGS1ProcNum = actMBaseSize;
    // V121 1C:1V: paired Vector handles all rows produced by its Cube.
    int64_t curAivGS1Idx = curGS1Idx;
    int64_t curAivGS1ProcNum = curGS1ProcNum;
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
__aicore__ inline void SIVector<SIT>::ProcessVec1(const SICommon::RunInfo &info)
{
    auto pingpong = info.loop & 1U;
    int64_t gS1BasePerVecSize_ = info.actMBaseSize;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    // vBiasGm --> vBiasLocal_ 搬运vBias
    uint64_t vBiasGmOffset = info.tensorVBiasOffset + curS2Idx;
    DataCopyPadExtParams<float> padVBiasParams{false, 0, 0, 0};
    DataCopyExtParams vBiasDataCopyExtParams;
    vBiasDataCopyExtParams.blockCount = 1;
    vBiasDataCopyExtParams.blockLen = info.actualSingleProcessSInnerSize * sizeof(float);
    vBiasDataCopyExtParams.srcStride = 0;
    vBiasDataCopyExtParams.dstStride = 0;
    DataCopyPad(vBiasLocal_[pingpong * s2BaseSize_], vBiasGm[vBiasGmOffset], vBiasDataCopyExtParams, padVBiasParams);

    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);

    // CV同步：V核等待C核完成mm1并将结果搬运到UB。
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_CV_EVENT + pingpong);

    auto qkBase = resMm1Local_[pingpong * (RES_MM1_SLOT_BYTES / sizeof(QK_T))];
    // uint16 score is compacted in place into the low 48 KiB of the current 96 KiB MM1 slot.
    auto outBase = qkBase.template ReinterpretCast<SCORE_T>();
    auto vBiasBase = vBiasLocal_[pingpong * s2BaseSize_];

    vector1::MulRSquareAndAddVBiasVF(outBase, qkBase, vBiasBase, constInfo_.rSquare, gS1BasePerVecSize_, s2BaseSize_,
                                     s2BaseSize_);
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
}

template <typename SIT>
__aicore__ inline uint32_t SIVector<SIT>::CalcDynamicTopkCount(uint32_t s1Idx, int64_t kbOffset, int32_t numPromptK)
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
    int64_t s1Pos = static_cast<int64_t>(s1Idx) + kbOffset;
    int32_t decayLen = numPromptK - kStart;

    // 边界条件：s1Pos < kStart 或 decayLen <= 1 时，直接返回 kStart
    if (s1Pos < kStart || decayLen <= 1) {
        return static_cast<uint32_t>(kStart);
    }

    // 公式定义 kEnd = alpha * kStart，插值完成后再向下取整。
    float kStartFloat = static_cast<float>(kStart);
    float kEnd = kStartFloat * alpha_;

    // 计算插值系数 t
    float t = static_cast<float>(s1Pos - kStart) / static_cast<float>(decayLen - 1);
    float interpolatedTopkCount = kStartFloat + t * (kEnd - kStartFloat);

    // 正数转有符号整数等价于 floor；外推到负数时先钳制，避免无符号转换。
    int32_t dynamicTopkCount =
        interpolatedTopkCount < 1.0f ? 1 : static_cast<int32_t>(interpolatedTopkCount);
    dynamicTopkCount = Min(dynamicTopkCount, kStart);

    return static_cast<uint32_t>(dynamicTopkCount);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::WriteDirectIndices(uint32_t outputLen, uint64_t baseLenOffset,
                                                         uint64_t baseOutOffset, uint32_t curAivGSize,
                                                         int64_t curAivGS1ProcNum, uint32_t actS1Size, uint32_t s1Idx,
                                                         LocalTensor<uint32_t> &indicesOutLocal,
                                                         LocalTensor<uint32_t> &lengthLocal)
{
    outputLen = Min(outputLen, PACKED_ROW_STRIDE);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    Duplicate(lengthLocal, outputLen, 1);
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
                    lengthLocal.ReinterpretCast<int32_t>(), indiceLenCp);
        DataCopyPad(
            sparseIndicesGm[baseOutOffset + static_cast<uint64_t>(gIdx) * constInfo_.qSeqSize * constInfo_.kSeqSize],
            indicesOutLocal.ReinterpretCast<int32_t>(), copyOut);
    }
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::CopyOutBatchRows(const SICommon::RunInfo &info, int64_t curAivGS1Idx,
                                                        const SICommon::RowIdx4 &rowIdx4,
                                                        const SICommon::TopkNum4 &topkNum4, uint32_t batchRowNum,
                                                        LocalTensor<uint32_t> &packedRowsLocal)
{
    for (uint32_t lane = 0U; lane < batchRowNum; lane++) {
        uint32_t mInnerIdx = SICommon::GetLane(rowIdx4, lane);
        uint64_t globalGS1Idx = static_cast<uint64_t>(curAivGS1Idx) + mInnerIdx;
        uint32_t globalS1Idx = static_cast<uint32_t>(globalGS1Idx % info.actS1Size);
        uint32_t curS1RealS2Len = info.actS2Size;
        if (constInfo_.attenMaskFlag) {
            int32_t realLen = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size) +
                              static_cast<int32_t>(globalS1Idx) + 1;
            curS1RealS2Len = static_cast<uint32_t>(realLen > 0 ? realLen : 0);
        }

        uint32_t topkSelectNum = Min(SICommon::GetLane(topkNum4, lane), MAX_TOPK_COUNT);
        uint32_t outputLen = Min(Min(initialBlocks_ + topkSelectNum + windowSize_, PACKED_ROW_STRIDE),
                                 static_cast<uint32_t>(constInfo_.kSeqSize));
        TopkUtils::PackSparseIndicesU32VF(
            packedRowsLocal[lane * PACKED_ROW_STRIDE], globalIndexLocal_[mInnerIdx * MAX_TOPK_COUNT],
            PACKED_ROW_STRIDE, initialBlocks_, topkSelectNum, windowSize_, curS1RealS2Len - windowSize_);
        // The final indices have been consumed by the pack VF. Reuse this row's first aligned block for its length.
        Duplicate(globalIndexLocal_[mInnerIdx * MAX_TOPK_COUNT], outputLen, 1);
    }
    PipeBarrier<PIPE_V>();
    // Packing uses the alternate slot while the previous batch is still in MTE3; serialize only before issuing MTE3.
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    for (uint32_t lane = 0U; lane < batchRowNum; lane++) {
        uint32_t mInnerIdx = SICommon::GetLane(rowIdx4, lane);
        uint64_t globalGS1Idx = static_cast<uint64_t>(curAivGS1Idx) + mInnerIdx;
        uint32_t globalGIdx = static_cast<uint32_t>(globalGS1Idx / info.actS1Size);
        uint32_t globalS1Idx = static_cast<uint32_t>(globalGS1Idx % info.actS1Size);
        uint64_t physicalRow = static_cast<uint64_t>(globalGIdx) * constInfo_.qSeqSize + globalS1Idx;
        uint32_t outputLen =
            Min(Min(initialBlocks_ + Min(SICommon::GetLane(topkNum4, lane), MAX_TOPK_COUNT) + windowSize_,
                    PACKED_ROW_STRIDE),
                static_cast<uint32_t>(constInfo_.kSeqSize));

        AscendC::DataCopyParams outputParams;
        outputParams.blockCount = 1U;
        outputParams.blockLen = static_cast<uint16_t>(outputLen * sizeof(int32_t));
        outputParams.srcStride = 0U;
        outputParams.dstStride = 0U;
        AscendC::DataCopyParams lengthParams;
        lengthParams.blockCount = 1U;
        lengthParams.blockLen = static_cast<uint16_t>(sizeof(uint32_t));
        lengthParams.srcStride = 0U;
        lengthParams.dstStride = 0U;
        DataCopyPad(sparseIndicesGm[info.indiceOutOffset + physicalRow * constInfo_.kSeqSize],
                    packedRowsLocal[lane * PACKED_ROW_STRIDE].ReinterpretCast<int32_t>(), outputParams);
        DataCopyPad(sparseSeqLenGm[info.indiceLenOffset + physicalRow],
                    globalIndexLocal_[mInnerIdx * MAX_TOPK_COUNT].ReinterpretCast<int32_t>(), lengthParams);
    }
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}

template <typename SIT>
__aicore__ inline void
SIVector<SIT>::RunTopKBatchRows(const SICommon::RunInfo &info, bool isFirstS2InnerLoop, bool isLastS2InnerLoop,
                                 const SICommon::RowIdx4 &rowIdx4, const SICommon::TopkNum4 &topkNum4,
                                 const SICommon::S2ValidLen4 &s2ValidLen4, uint32_t batchRowNum, int64_t curAivGS1Idx,
                                 uint32_t packedSlotIdx)
{
    const uint32_t slotQkOffset =
        (info.loop & 1U) * (RES_MM1_SLOT_BYTES / sizeof(QK_T));
    LocalTensor<QK_T> slotLocal = resMm1Local_[slotQkOffset];
    LocalTensor<SCORE_T> currentScoreLocal = slotLocal.ReinterpretCast<SCORE_T>();
    LocalTensor<SCORE_T> mergeValueLocal =
        slotLocal[TOPK_MERGE_OFFSET_BYTES / sizeof(QK_T)].ReinterpretCast<SCORE_T>();
    LocalTensor<uint32_t> topkWorkspaceLocal =
        slotLocal[TOPK_WORKSPACE_OFFSET_BYTES / sizeof(QK_T)].ReinterpretCast<uint32_t>();
    uint32_t packedOffsetBytes =
        (packedSlotIdx & 1U) == 0U ? PACKED_OUTPUT_SLOT0_OFFSET_BYTES : PACKED_OUTPUT_SLOT1_OFFSET_BYTES;
    LocalTensor<uint32_t> packedRowsLocal = slotLocal[packedOffsetBytes / sizeof(QK_T)].ReinterpretCast<uint32_t>();

    // Build four compact [history 256][current 256] rows in the current MM1 slot.
    for (uint32_t lane = 0U; lane < TOPK_ROW_BATCH; lane++) {
        uint32_t mInnerIdx = SICommon::GetLane(rowIdx4, lane);
        if (!isFirstS2InnerLoop) {
            AscendC::DataCopy(mergeValueLocal[lane * mrgRowStride_],
                              historyValueLocal_[mInnerIdx * TOPK_ALIGN_SIZE], TOPK_ALIGN_SIZE);
        }
        AscendC::DataCopy(mergeValueLocal[lane * mrgRowStride_ + TOPK_ALIGN_SIZE],
                          currentScoreLocal[mInnerIdx * s2BaseSize_], SICommon::TRUNK_LEN_256);
    }
    PipeBarrier<PIPE_V>();

    constexpr SICommon::RowIdx4 compactRowIdx4 = {0U, 1U, 2U, 3U};
    if (isFirstS2InnerLoop && initialBlocks_ > 0U) {
        TopkUtils::PadMrgRowsInitBlockVF(mergeValueLocal, mrgRowStride_, TOPK_ROW_BATCH, TOPK_ALIGN_SIZE,
                                         initialBlocks_);
    }
    // padding（各行按自身 s2BlockValidLen 补齐到 TRUNK_LEN_256，非零行结果一致）
    if (isFirstS2InnerLoop) {
        TopkUtils::PadMrgFourRowsFirstS2RangeVF<SCORE_T>(mergeValueLocal, compactRowIdx4, topkNum4, s2ValidLen4,
                                                         mrgRowStride_);
    } else {
        TopkUtils::PadMrgFourRowsNotFirstS2RangeVF<SCORE_T>(mergeValueLocal, compactRowIdx4, topkNum4, s2ValidLen4,
                                                             mrgRowStride_);
    }

    uint32_t inputOffset = isFirstS2InnerLoop ? TOPK_ALIGN_SIZE : 0U;
    uint32_t validLen = isFirstS2InnerLoop ? SICommon::TRUNK_LEN_256 : (TOPK_ALIGN_SIZE + SICommon::TRUNK_LEN_256);

    PipeBarrier<PIPE_V>();

    topkOp_.Batch4Rows(mergeValueLocal, historyValueLocal_, topkWorkspaceLocal, globalIndexLocal_, rowIdx4, topkNum4,
                       batchRowNum, mrgRowStride_, inputOffset, validLen, info.s2Idx, info.s2LoopEnd + 1);

    if (isLastS2InnerLoop) {
        PipeBarrier<PIPE_V>();
        CopyOutBatchRows(info, curAivGS1Idx, rowIdx4, topkNum4, batchRowNum, packedRowsLocal);
    }
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::ProcessTopK(const SICommon::RunInfo &info, bool isFirstS2InnerLoop,
                                                  bool isLastS2InnerLoop)
{
    // 当前处理的GS1行起始索引（全局位置）
    int64_t curGS1Idx = info.gS1Idx * mBaseSize_;
    // 当前GS1行需要处理的数量（可能小于mBaseSize_，处理尾块）
    int64_t curGS1ProcNum = info.actMBaseSize;
    int64_t curAivGS1Idx = curGS1Idx;
    int64_t curAivGS1ProcNum = curGS1ProcNum;

    // S2 基本块信息
    uint32_t s2BlockStart = info.s2Idx * s2BaseSize_;

    // 计算动态 topkCount 所需的块级常量（使用原始 actS2Size）
    int64_t kbOffset = static_cast<int64_t>(info.actS2Size) - static_cast<int64_t>(info.actS1Size);
    int32_t numPromptK = static_cast<int32_t>(info.promptLen);

    uint32_t curAivGSize = CeilDiv(curAivGS1ProcNum, info.actS1Size);
    // 当前核处理的 S1 轴大小
    // 当mBaseSize小于actS1Size时，curAivS1Size为curAivGS1ProcNum
    // 当mBaseSize大于等于actS1Size时， curAivS1Size为info.actS1Size
    uint32_t curAivS1Size = Min(info.actS1Size, static_cast<uint32_t>(curAivGS1ProcNum));
    SICommon::RowIdx4 rowIdx4 = {0U, 0U, 0U, 0U};
    SICommon::TopkNum4 topkNum4 = {0U, 0U, 0U, 0U};
    SICommon::S2ValidLen4 s2ValidLen4 = {0U, 0U, 0U, 0U};
    uint32_t batchRowNum = 0;
    uint32_t packedSlotIdx = 0U;

    for (uint32_t s1Idx = 0; s1Idx < curAivS1Size; s1Idx++) {
        uint32_t nowCurAivGS1Idx = curAivGS1Idx + s1Idx;
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
            if (isLastS2InnerLoop) {
                uint32_t directOutputLen = Min(curS1RealS2Len, PACKED_ROW_STRIDE);
                uint64_t firstGlobalGS1Idx = static_cast<uint64_t>(curAivGS1Idx) + s1Idx;
                uint32_t firstGlobalGIdx = static_cast<uint32_t>(firstGlobalGS1Idx / info.actS1Size);
                uint32_t firstGlobalS1Idx = static_cast<uint32_t>(firstGlobalGS1Idx % info.actS1Size);
                uint64_t baseLenOffset = info.indiceLenOffset +
                                         static_cast<uint64_t>(firstGlobalGIdx) * constInfo_.qSeqSize +
                                         firstGlobalS1Idx;
                uint64_t baseOutOffset =
                    info.indiceOutOffset +
                    (static_cast<uint64_t>(firstGlobalGIdx) * constInfo_.qSeqSize + firstGlobalS1Idx) *
                        constInfo_.kSeqSize;
                LocalTensor<uint32_t> directIndicesLocal =
                    resMm1Local_[(info.loop & 1U) *
                                     (RES_MM1_SLOT_BYTES / sizeof(QK_T)) +
                                 PACKED_OUTPUT_SLOT0_OFFSET_BYTES / sizeof(QK_T)]
                        .ReinterpretCast<uint32_t>();
                LocalTensor<uint32_t> directLengthLocal = directIndicesLocal[PACKED_ROW_STRIDE];
                WriteDirectIndices(directOutputLen, baseLenOffset, baseOutOffset, curAivGSize, curAivGS1ProcNum,
                                   info.actS1Size, s1Idx, directIndicesLocal, directLengthLocal);
                // Direct output uses packed slot 0; let the next TopK batch pack into slot 1 while MTE3 drains it.
                packedSlotIdx = 1U;
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
                RunTopKBatchRows(info, isFirstS2InnerLoop, isLastS2InnerLoop, rowIdx4, topkNum4, s2ValidLen4,
                                 batchRowNum, curAivGS1Idx, packedSlotIdx);
                packedSlotIdx ^= 1U;
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
        RunTopKBatchRows(info, isFirstS2InnerLoop, isLastS2InnerLoop, rowIdx4, topkNum4, s2ValidLen4, batchRowNum,
                         curAivGS1Idx, packedSlotIdx);
    }
    if (isLastS2InnerLoop) {
        // Packed output reuses the current MM1 slot; return it to Cube only after MTE3 completes.
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
    CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_VC_EVENT + (info.loop & 1U));
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::ProcessDirectOutput(const SICommon::TempLoopInfo &tempLoopInfo, uint32_t gS1Idx)
{
    if ASCEND_IS_AIV {
        uint32_t actMBaseSize = tempLoopInfo.actMBaseSize;
        uint32_t actS1Size = tempLoopInfo.actS1Size;
        uint32_t actS2Size = tempLoopInfo.actS2Size;
        uint64_t qFlatBase = static_cast<uint64_t>(tempLoopInfo.bIdx) * constInfo_.qSeqSize * constInfo_.qHeadNum +
                             static_cast<uint64_t>(tempLoopInfo.n2Idx) * constInfo_.gSize * constInfo_.qSeqSize;
        uint64_t indiceLenOffset = qFlatBase;
        uint64_t indiceOutOffset = qFlatBase * constInfo_.kSeqSize;

        if (!needCleanOutput_) {
            InitSparseIndicesToNegOne(gS1Idx, actMBaseSize, actS1Size, indiceOutOffset);
        }

        // This branch has no Cube/Vector slot handshake. Use persistent Vector-only UB, never the Cube-owned MM1 slot.
        LocalTensor<uint32_t> indicesOutLocal = globalIndexLocal_;
        LocalTensor<uint32_t> lengthLocal = globalIndexLocal_[PACKED_ROW_STRIDE];

        // 当前处理的gS1行起始索引（全局位置）
        int64_t curGS1Idx = gS1Idx * mBaseSize_;
        // 当前gS1行需要处理的数量（可能小于mBaseSize_，处理尾块）
        int64_t curGS1ProcNum = actMBaseSize;
        int64_t curAivGS1Idx = curGS1Idx;
        int64_t curAivGS1ProcNum = curGS1ProcNum;

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
            uint64_t baseLenOffset =
                indiceLenOffset + static_cast<uint64_t>(globalGIdx) * constInfo_.qSeqSize + globalS1Idx;
            uint64_t baseOutOffset =
                indiceOutOffset +
                (static_cast<uint64_t>(globalGIdx) * constInfo_.qSeqSize + globalS1Idx) * constInfo_.kSeqSize;
            if (constInfo_.attenMaskFlag) {
                int32_t curS1RealS2LenTmp = static_cast<int32_t>(actS2Size) - static_cast<int32_t>(actS1Size) +
                                            static_cast<int32_t>(globalS1Idx) + 1;
                curS1RealS2Len = static_cast<uint32_t>(curS1RealS2LenTmp > 0 ? curS1RealS2LenTmp : 0);
            }
            WriteDirectIndices(curS1RealS2Len, baseLenOffset, baseOutOffset, curAivGSize, curAivGS1ProcNum, actS1Size,
                               s1Idx, indicesOutLocal, lengthLocal);
        }
    }
}
} // namespace SIKernel
#endif
