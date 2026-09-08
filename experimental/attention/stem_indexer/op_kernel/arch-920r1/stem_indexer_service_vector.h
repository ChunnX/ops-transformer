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

    // arch=920r1 两种 score 精度均走 score-GM 路径。
    static constexpr bool ENABLE_SCORE_GM_TOPK =
        std::is_same_v<SCORE_T, uint16_t> || std::is_same_v<SCORE_T, uint32_t>;

    __aicore__ inline SIVector(){};
    __aicore__ inline void ProcessVec1(const SICommon::RunInfo &info);
    __aicore__ inline void ProcessVec1ScoreGm(const SICommon::RunInfo &info);
    __aicore__ inline void ProcessScoreGmTopK(const SICommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const SICommon::RunInfo &info, bool isFirstS2InnerLoop, bool isLastS2InnerLoop);
    __aicore__ inline void ProcessDirectOutput(const SICommon::TempLoopInfo &tempLoopInfo, uint32_t gS1Idx);
    __aicore__ inline void InitBuffers(TPipe *pipe, bool needCleanOutput);
    __aicore__ inline void InitParams(const struct SICommon::ConstInfo &constInfo, uint32_t logicalCoreIdx);
    __aicore__ inline void InitScoreWorkspace(__gm__ uint8_t *workspace);
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
    GlobalTensor<SCORE_T> scoreWorkspaceGm;
    event_t vec1VMte2Event_[SICommon::MM1_RES_BUFFER_NUM];
    event_t vec1Mte2VEvent_[SICommon::MM1_RES_BUFFER_NUM];
    event_t topkMte3Mte2Event_;
    event_t topkVMte3Event_;
    event_t topkMte3VEvent_;

    // =================================常量区=================================
    static constexpr uint32_t MAX_TOPK_COUNT = 256;
    static constexpr uint32_t TOPK_ALIGN_SIZE = 256;
    static constexpr uint32_t TOPK_ROW_BATCH = 4;
    static constexpr uint32_t PACKED_ROW_STRIDE = MAX_TOPK_COUNT + 8U;
    static constexpr uint32_t OUTPUT_INIT_ALIGN_ELEMS = 512U / sizeof(int32_t);
    static constexpr int32_t K_SMALL_SEQ_MAX = 56;
    static constexpr int32_t K_MEDIUM_SEQ_MAX = 160;

    static constexpr uint32_t AIV_CORE_NUM = 64U;
    static constexpr uint32_t M_BASIC_BLOCK = 64U;
    static constexpr uint32_t S2_BASIC_BLOCK = 512U;
    static constexpr uint32_t RES_MM1_SLOT_BYTES = M_BASIC_BLOCK * S2_BASIC_BLOCK * sizeof(QK_T);
    static constexpr uint32_t RES_MM1_DOUBLE_BUFFER_BYTES = 2U * RES_MM1_SLOT_BYTES;
    static constexpr uint32_t VBIAS_BYTES = 2U * S2_BASIC_BLOCK * sizeof(float);
    static constexpr uint32_t FULL_HISTORY_SCORE_BYTES = M_BASIC_BLOCK * TOPK_ALIGN_SIZE * sizeof(SCORE_T);
    static constexpr uint32_t FULL_GLOBAL_INDEX_BYTES = M_BASIC_BLOCK * TOPK_ALIGN_SIZE * sizeof(uint32_t);
    static constexpr uint32_t DIRECT_INDEX_BYTES = PACKED_ROW_STRIDE * sizeof(uint32_t);
    static constexpr uint32_t DIRECT_LENGTH_BYTES = 32U;
    static constexpr uint32_t CURRENT_SCORE_BYTES = M_BASIC_BLOCK * S2_BASIC_BLOCK * sizeof(SCORE_T);
    static constexpr uint32_t TOPK_MERGE_BYTES =
        TOPK_ROW_BATCH * (TOPK_ALIGN_SIZE + SICommon::TOPK_TRUNK_LEN) * sizeof(SCORE_T);
    static constexpr uint32_t TOPK_WORKSPACE_BYTES = StemIndexerTopk<SCORE_T>::TOPK_WORKSPACE_BYTES;
    static constexpr uint32_t PACKED_OUTPUT_SLOT_BYTES = TOPK_ROW_BATCH * PACKED_ROW_STRIDE * sizeof(uint32_t);
    static constexpr uint32_t TOPK_MERGE_OFFSET_BYTES = CURRENT_SCORE_BYTES;
    static constexpr uint32_t TOPK_WORKSPACE_OFFSET_BYTES = TOPK_MERGE_OFFSET_BYTES + TOPK_MERGE_BYTES;
    static constexpr uint32_t PACKED_OUTPUT_SLOT0_OFFSET_BYTES = TOPK_WORKSPACE_OFFSET_BYTES + TOPK_WORKSPACE_BYTES;
    static constexpr uint32_t PACKED_OUTPUT_SLOT1_OFFSET_BYTES =
        PACKED_OUTPUT_SLOT0_OFFSET_BYTES + PACKED_OUTPUT_SLOT_BYTES;

    // 512 列的 score tile 写入 GM，由 1024 列的 TopK trunk 消费。
    static constexpr uint32_t SCORE_GM_TRUNK_LEN = 1024U;
    static constexpr uint32_t SCORE_GM_MERGE_ROW_STRIDE = TOPK_ALIGN_SIZE + SCORE_GM_TRUNK_LEN;
    static constexpr uint32_t SCORE_GM_MERGE_SLOT_BYTES =
        TOPK_ROW_BATCH * SCORE_GM_MERGE_ROW_STRIDE * sizeof(SCORE_T);
    static constexpr uint32_t SCORE_GM_HISTORY_BYTES = TOPK_ROW_BATCH * TOPK_ALIGN_SIZE * sizeof(SCORE_T);
    static constexpr uint32_t SCORE_GM_GLOBAL_INDEX_BYTES = TOPK_ROW_BATCH * TOPK_ALIGN_SIZE * sizeof(uint32_t);
    static constexpr uint32_t SCORE_GM_LENGTH_ROW_STRIDE = 8U;
    static constexpr uint32_t SCORE_GM_LENGTH_SLOT_BYTES =
        TOPK_ROW_BATCH * SCORE_GM_LENGTH_ROW_STRIDE * sizeof(uint32_t);
    static constexpr uint32_t SCORE_GM_TOPK_WORKSPACE_BYTES =
        StemIndexerTopk<SCORE_T, SCORE_GM_TRUNK_LEN>::TOPK_WORKSPACE_BYTES;

    // historyValueBuf_ 内的 score-GM 视图：merge0 | merge1 | history | packed0 | packed1。
    static constexpr uint32_t SCORE_GM_MERGE0_OFFSET_BYTES = 0U;
    static constexpr uint32_t SCORE_GM_MERGE1_OFFSET_BYTES = SCORE_GM_MERGE_SLOT_BYTES;
    static constexpr uint32_t SCORE_GM_HISTORY_OFFSET_BYTES = 2U * SCORE_GM_MERGE_SLOT_BYTES;
    static constexpr uint32_t SCORE_GM_PACKED0_OFFSET_BYTES = SCORE_GM_HISTORY_OFFSET_BYTES + SCORE_GM_HISTORY_BYTES;
    static constexpr uint32_t SCORE_GM_PACKED1_OFFSET_BYTES = SCORE_GM_PACKED0_OFFSET_BYTES + PACKED_OUTPUT_SLOT_BYTES;
    static constexpr uint32_t SCORE_GM_LENGTH0_OFFSET_BYTES = SCORE_GM_PACKED1_OFFSET_BYTES + PACKED_OUTPUT_SLOT_BYTES;
    static constexpr uint32_t SCORE_GM_LENGTH1_OFFSET_BYTES = SCORE_GM_LENGTH0_OFFSET_BYTES + SCORE_GM_LENGTH_SLOT_BYTES;
    static constexpr uint32_t SCORE_GM_HISTORY_BUF_USED_BYTES =
        SCORE_GM_LENGTH1_OFFSET_BYTES + SCORE_GM_LENGTH_SLOT_BYTES;

    // globalIndexBuf_ 内的 score-GM 视图：TopK workspace | 紧凑全局索引。
    static constexpr uint32_t SCORE_GM_GLOBAL_INDEX_OFFSET_BYTES = SCORE_GM_TOPK_WORKSPACE_BYTES;
    static constexpr uint32_t SCORE_GM_GLOBAL_BUF_USED_BYTES =
        SCORE_GM_GLOBAL_INDEX_OFFSET_BYTES + SCORE_GM_GLOBAL_INDEX_BYTES;

    // 保持经过验证的 uint16 分配不变；uint32 采用紧凑 4 行 Score-GM 存储，
    // 使完整 Vector 分配低于单个 384 KiB UB 上限，并为栈支持的高级 API 留出空间。
    static constexpr uint32_t SCORE_GM_HISTORY_ALLOC_BYTES =
        std::is_same_v<SCORE_T, uint32_t> ? SCORE_GM_HISTORY_BUF_USED_BYTES : FULL_HISTORY_SCORE_BYTES;
    static constexpr uint32_t SCORE_GM_GLOBAL_ALLOC_BYTES =
        std::is_same_v<SCORE_T, uint32_t> ? SCORE_GM_GLOBAL_BUF_USED_BYTES : FULL_GLOBAL_INDEX_BYTES;
    static constexpr uint32_t VECTOR_UB_CAPACITY_BYTES = 384U * 1024U;
    static constexpr uint32_t VECTOR_UB_USED_BYTES =
        RES_MM1_DOUBLE_BUFFER_BYTES + VBIAS_BYTES + SCORE_GM_HISTORY_ALLOC_BYTES + SCORE_GM_GLOBAL_ALLOC_BYTES +
        DIRECT_INDEX_BYTES + DIRECT_LENGTH_BYTES;

    static_assert(SCORE_GM_HISTORY_BUF_USED_BYTES <= SCORE_GM_HISTORY_ALLOC_BYTES,
                  "score-GM fast path exceeds the history UB allocation");
    static_assert(SCORE_GM_GLOBAL_BUF_USED_BYTES <= SCORE_GM_GLOBAL_ALLOC_BYTES,
                  "score-GM fast path exceeds the global-index UB allocation");
    static_assert(SCORE_GM_HISTORY_ALLOC_BYTES % 32U == 0U && SCORE_GM_GLOBAL_ALLOC_BYTES % 32U == 0U,
                  "score-GM UB allocations must remain 32-byte aligned");
    static_assert(VECTOR_UB_USED_BYTES <= VECTOR_UB_CAPACITY_BYTES,
                  "stem-indexer Vector buffers exceed the V121 384 KiB UB capacity");

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
    __aicore__ inline void StoreScoreToWorkspace(const SICommon::RunInfo &info);
    __aicore__ inline void IssueLoadScorePair(const SICommon::RowIdx4 &rowIdx4, uint32_t pairIdx,
                                              uint32_t activeTileNum, uint32_t mergeSlotIdx);
    __aicore__ inline void CopyOutScoreGmBatchRows(const SICommon::RunInfo &info,
                                                   const SICommon::RowIdx4 &rowIdx4,
                                                   const SICommon::TopkNum4 &topkNum4,
                                                   uint32_t batchRowNum,
                                                   LocalTensor<uint32_t> &packedRowsLocal,
                                                   LocalTensor<uint32_t> &lengthLocal,
                                                   LocalTensor<uint32_t> &batchGlobalIndexLocal);
    __aicore__ inline void CopyOutScoreGmDirectRow(const SICommon::RunInfo &info, uint32_t mInnerIdx,
                                                   uint32_t outputLen, LocalTensor<uint32_t> &indicesLocal,
                                                   LocalTensor<uint32_t> &lengthLocal);
    __aicore__ inline void RunScoreGmTopKBatch(const SICommon::RunInfo &info,
                                               const SICommon::RowIdx4 &rowIdx4,
                                               const SICommon::TopkNum4 &topkNum4,
                                               const SICommon::S2ValidLen4 &validEnd4, uint32_t batchRowNum,
                                               uint32_t packedSlotIdx);
    // ================================Local Buffer区====================================
    // vector 临时 buffer
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1Local_;
    // vbias 临时 buffer
    TBuf<TPosition::VECCALC> vBiasBuf_;
    LocalTensor<float> vBiasLocal_;

    // 常驻 TopK history。当前 score、4 行 merge、TopK scratch 与打包输出均复用 resMm1Buf_。
    TBuf<TPosition::VECCALC> historyValueBuf_;
    LocalTensor<SCORE_T> historyValueLocal_;

    TBuf<TPosition::VECCALC> globalIndexBuf_;
    LocalTensor<uint32_t> globalIndexLocal_;

    // 直写输出路径的 Vector 专用 scratch。与 TopK history、Cube 持有的 MM1 均保持独立。
    TBuf<TPosition::VECCALC> directIndexBuf_;
    LocalTensor<uint32_t> directIndexLocal_;
    TBuf<TPosition::VECCALC> directLengthBuf_;
    LocalTensor<uint32_t> directLengthLocal_;

    int32_t blockId_ = -1;
    bool needCleanOutput_ = false;
    // vector 参数
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
    uint64_t scoreGmCoreElems_ = 0ULL;
    struct SICommon::ConstInfo constInfo_;
    SIKernel::StemIndexerTopk<SCORE_T> topkOp_;
    SIKernel::StemIndexerTopk<SCORE_T, SCORE_GM_TRUNK_LEN> scoreGmTopkOp_;
};

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitBuffers(TPipe *pipe, bool needCleanOutput)
{
    needCleanOutput_ = needCleanOutput;
    // 2 * 64 * 512 * 4B = 256 KB。每个 128 KB slot 在原地 score 转换后复用。
    pipe->InitBuffer(resMm1Buf_, RES_MM1_DOUBLE_BUFFER_BYTES);
    resMm1Local_ = resMm1Buf_.Get<QK_T>(); // qk

    // 当前配置为2 * 512 * 4B = 4 KB。
    pipe->InitBuffer(vBiasBuf_, VBIAS_BYTES);
    vBiasLocal_ = vBiasBuf_.Get<float>(); // vBias

    // uint16 保持既有 32 KiB 分配；uint32 采用紧凑 4 行 Score-GM 布局。
    pipe->InitBuffer(historyValueBuf_, SCORE_GM_HISTORY_ALLOC_BYTES);
    historyValueLocal_ = historyValueBuf_.Get<SCORE_T>();

    // uint16 保持既有 64 KiB 分配；uint32 仅存放 TopK scratch 和 4 行紧凑索引。
    pipe->InitBuffer(globalIndexBuf_, SCORE_GM_GLOBAL_ALLOC_BYTES);
    globalIndexLocal_ = globalIndexBuf_.Get<uint32_t>();

    // 直写输出持有独占 UB，其短索引/长度写入不会与常驻 TopK 状态混叠。
    pipe->InitBuffer(directIndexBuf_, DIRECT_INDEX_BYTES);
    directIndexLocal_ = directIndexBuf_.Get<uint32_t>();
    pipe->InitBuffer(directLengthBuf_, DIRECT_LENGTH_BYTES);
    directLengthLocal_ = directLengthBuf_.Get<uint32_t>();

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
    uint64_t aivCoreNum = constInfo_.usedCoreNum;
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
    s2BaseSize_ = constInfo.s2BaseSize; // 512
    mBaseSize_ = constInfo.mBaseSize;
    alpha_ = constInfo.alpha;
    kBlockNumRateMedium_ = constInfo.kBlockNumRateMedium;
    kBlockNumBiasMedium_ = constInfo.kBlockNumBiasMedium;
    kBlockNumRateLarge_ = constInfo.kBlockNumRateLarge;
    kBlockNumBiasLarge_ = constInfo.kBlockNumBiasLarge;
    initialBlocks_ = constInfo.initialBlocks;
    windowSize_ = constInfo.windowSize;
    blockId_ = static_cast<int32_t>(logicalCoreIdx);
    uint32_t scoreGmTileNum = static_cast<uint32_t>(
        (constInfo.kSeqSize + static_cast<uint64_t>(S2_BASIC_BLOCK) - 1ULL) / S2_BASIC_BLOCK);
    scoreGmCoreElems_ = static_cast<uint64_t>(M_BASIC_BLOCK) * scoreGmTileNum * S2_BASIC_BLOCK;
    // 紧凑的 4 行 merge 布局：[history 256][current 512]。
    mrgRowStride_ = MAX_TOPK_COUNT + SICommon::TOPK_TRUNK_LEN;
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitScoreWorkspace(__gm__ uint8_t *workspace)
{
    uint64_t coreOffset = static_cast<uint64_t>(blockId_) * scoreGmCoreElems_;
    scoreWorkspaceGm.SetGlobalBuffer((__gm__ SCORE_T *)workspace + coreOffset, scoreGmCoreElems_);
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
    for (uint32_t slot = 0U; slot < SICommon::MM1_RES_BUFFER_NUM; slot++) {
        vec1VMte2Event_[slot] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        vec1Mte2VEvent_[slot] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::V_MTE2>(vec1VMte2Event_[slot]);
    }
    topkMte3Mte2Event_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    topkVMte3Event_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    topkMte3VEvent_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    SetFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::FreeEventID()
{
    for (uint32_t slot = 0U; slot < SICommon::MM1_RES_BUFFER_NUM; slot++) {
        WaitFlag<HardEvent::V_MTE2>(vec1VMte2Event_[slot]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vec1VMte2Event_[slot]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(vec1Mte2VEvent_[slot]);
    }
    WaitFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(topkMte3Mte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(topkVMte3Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(topkMte3VEvent_);
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
    // arch-920r1 1C:1V：配对的 Vector 处理其 Cube 产出的全部行。
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
    WaitFlag<HardEvent::V_MTE2>(vec1VMte2Event_[pingpong]);
    // vBiasGm --> vBiasLocal_ 搬运vBias
    uint64_t vBiasGmOffset = info.tensorVBiasOffset + curS2Idx;
    DataCopyPadExtParams<float> padVBiasParams{false, 0, 0, 0};
    DataCopyExtParams vBiasDataCopyExtParams;
    vBiasDataCopyExtParams.blockCount = 1;
    vBiasDataCopyExtParams.blockLen = info.actualSingleProcessSInnerSize * sizeof(float);
    vBiasDataCopyExtParams.srcStride = 0;
    vBiasDataCopyExtParams.dstStride = 0;
    DataCopyPad(vBiasLocal_[pingpong * s2BaseSize_], vBiasGm[vBiasGmOffset], vBiasDataCopyExtParams, padVBiasParams);

    SetFlag<HardEvent::MTE2_V>(vec1Mte2VEvent_[pingpong]);
    WaitFlag<HardEvent::MTE2_V>(vec1Mte2VEvent_[pingpong]);

    // CV同步：V核等待C核完成mm1并将结果搬运到UB。
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_CV_EVENT + pingpong);

    auto qkBase = resMm1Local_[pingpong * (RES_MM1_SLOT_BYTES / sizeof(QK_T))];
    // SCORE_T 原地生成于当前 128 KiB MM1 slot。
    auto outBase = qkBase.template ReinterpretCast<SCORE_T>();
    auto vBiasBase = vBiasLocal_[pingpong * s2BaseSize_];

    vector1::MulRSquareAndAddVBiasVF(outBase, qkBase, vBiasBase, constInfo_.rSquare, gS1BasePerVecSize_, s2BaseSize_,
                                     s2BaseSize_);
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE2>(vec1VMte2Event_[pingpong]);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::StoreScoreToWorkspace(const SICommon::RunInfo &info)
{
    uint32_t pingpong = info.loop & 1U;
    uint32_t slotOffset = pingpong * (RES_MM1_SLOT_BYTES / sizeof(QK_T));
    LocalTensor<SCORE_T> scoreLocal = resMm1Local_[slotOffset].ReinterpretCast<SCORE_T>();
    uint64_t scoreGmOffset = static_cast<uint64_t>(info.s2Idx) * M_BASIC_BLOCK * S2_BASIC_BLOCK;

    // MTE3 消费完原地 score 之前，VC slot 不能归还给 Cube。
    WaitFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
    // initial blocks 作为固定前缀输出，不得再次参与 TopK。在源 score tile 中将其mask，
    // 使后续每次 GM reload/merge 都遵循相同的排除不变量。
    if (info.s2Idx == 0U && initialBlocks_ > 0U) {
        TopkUtils::PadMrgRowsInitBlockVF(scoreLocal, S2_BASIC_BLOCK, info.actMBaseSize, 0U, initialBlocks_);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(topkVMte3Event_);
    WaitFlag<HardEvent::V_MTE3>(topkVMte3Event_);
    DataCopy(scoreWorkspaceGm[scoreGmOffset], scoreLocal, info.actMBaseSize * S2_BASIC_BLOCK);
    // 最后一个 score tile 在本侧 store 后立即经 MTE2 消费。MTE3->V 只保护源 UB，
    // 不约束独立的 MTE2 流水，因此需直接建立 GM 写/读依赖。
    // 中间态 store 直到最后一个 tile 才会被 reload，不得占用该 event slot。
    if (info.isLastS2InnerLoop) {
        SetFlag<HardEvent::MTE3_MTE2>(topkMte3Mte2Event_);
    }
    SetFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
    WaitFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
    SetFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::ProcessVec1ScoreGm(const SICommon::RunInfo &info)
{
    ProcessVec1(info);
    StoreScoreToWorkspace(info);
    CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_VC_EVENT + (info.loop & 1U));
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::IssueLoadScorePair(const SICommon::RowIdx4 &rowIdx4, uint32_t pairIdx,
                                                         uint32_t activeTileNum, uint32_t mergeSlotIdx)
{
    LocalTensor<SCORE_T> fastBuf = historyValueBuf_.Get<SCORE_T>();
    uint32_t mergeOffset = mergeSlotIdx == 0U ? SCORE_GM_MERGE0_OFFSET_BYTES : SCORE_GM_MERGE1_OFFSET_BYTES;
    LocalTensor<SCORE_T> mergeLocal = fastBuf[mergeOffset / sizeof(SCORE_T)];

    WaitFlag<HardEvent::V_MTE2>(vec1VMte2Event_[mergeSlotIdx]);
    uint32_t tile0Idx = pairIdx * 2U;
    for (uint32_t lane = 0U; lane < TOPK_ROW_BATCH; lane++) {
        uint32_t row = SICommon::GetLane(rowIdx4, lane);
        uint64_t tile0Offset = static_cast<uint64_t>(tile0Idx) * M_BASIC_BLOCK * S2_BASIC_BLOCK +
                               static_cast<uint64_t>(row) * S2_BASIC_BLOCK;
        uint32_t dstOffset = lane * SCORE_GM_MERGE_ROW_STRIDE + TOPK_ALIGN_SIZE;
        DataCopy(mergeLocal[dstOffset], scoreWorkspaceGm[tile0Offset], S2_BASIC_BLOCK);
        if (tile0Idx + 1U < activeTileNum) {
            uint64_t tile1Offset = tile0Offset + static_cast<uint64_t>(M_BASIC_BLOCK) * S2_BASIC_BLOCK;
            DataCopy(mergeLocal[dstOffset + S2_BASIC_BLOCK], scoreWorkspaceGm[tile1Offset], S2_BASIC_BLOCK);
        }
    }
    SetFlag<HardEvent::MTE2_V>(vec1Mte2VEvent_[mergeSlotIdx]);
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
    WaitFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
    Duplicate(lengthLocal, outputLen, 1);
    if (outputLen > 0U) {
        CreateVecIndex(indicesOutLocal.ReinterpretCast<int32_t>(), 0, outputLen);
    }
    SetFlag<HardEvent::V_MTE3>(topkVMte3Event_);
    WaitFlag<HardEvent::V_MTE3>(topkVMte3Event_);

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
        if (outputLen > 0U) {
            DataCopyPad(sparseIndicesGm[baseOutOffset +
                                        static_cast<uint64_t>(gIdx) * constInfo_.qSeqSize * constInfo_.kSeqSize],
                        indicesOutLocal.ReinterpretCast<int32_t>(), copyOut);
        }
    }
    SetFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
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
        // 最终索引已被 VF 消费。复用该行首个对齐 block 写入其长度。
        Duplicate(globalIndexLocal_[mInnerIdx * MAX_TOPK_COUNT], outputLen, 1);
    }
    PipeBarrier<PIPE_V>();
    // 上一个 batch 仍在 MTE3 中时，打包使用另一个 slot；仅在下发 MTE3 前做串行化。
    WaitFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
    SetFlag<HardEvent::V_MTE3>(topkVMte3Event_);
    WaitFlag<HardEvent::V_MTE3>(topkVMte3Event_);

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
    SetFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
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

    // 在当前 MM1 slot 中构造 4 行紧凑的 [history 256][current 512]。
    for (uint32_t lane = 0U; lane < TOPK_ROW_BATCH; lane++) {
        uint32_t mInnerIdx = SICommon::GetLane(rowIdx4, lane);
        if (!isFirstS2InnerLoop) {
            AscendC::DataCopy(mergeValueLocal[lane * mrgRowStride_],
                              historyValueLocal_[mInnerIdx * TOPK_ALIGN_SIZE], TOPK_ALIGN_SIZE);
        }
        AscendC::DataCopy(mergeValueLocal[lane * mrgRowStride_ + TOPK_ALIGN_SIZE],
                          currentScoreLocal[mInnerIdx * s2BaseSize_], SICommon::TOPK_TRUNK_LEN);
    }
    PipeBarrier<PIPE_V>();

    constexpr SICommon::RowIdx4 compactRowIdx4 = {0U, 1U, 2U, 3U};
    if (isFirstS2InnerLoop && initialBlocks_ > 0U) {
        TopkUtils::PadMrgRowsInitBlockVF(mergeValueLocal, mrgRowStride_, TOPK_ROW_BATCH, TOPK_ALIGN_SIZE,
                                         initialBlocks_);
    }
    // padding（各行按自身s2BlockValidLen补齐到TOPK_TRUNK_LEN，非零行结果一致）
    if (isFirstS2InnerLoop) {
        TopkUtils::PadMrgFourRowsFirstS2RangeVF<SCORE_T>(mergeValueLocal, compactRowIdx4, topkNum4, s2ValidLen4,
                                                         mrgRowStride_);
    } else {
        TopkUtils::PadMrgFourRowsNotFirstS2RangeVF<SCORE_T>(mergeValueLocal, compactRowIdx4, topkNum4, s2ValidLen4,
                                                             mrgRowStride_);
    }

    uint32_t inputOffset = isFirstS2InnerLoop ? TOPK_ALIGN_SIZE : 0U;
    uint32_t validLen =
        isFirstS2InnerLoop ? SICommon::TOPK_TRUNK_LEN : (TOPK_ALIGN_SIZE + SICommon::TOPK_TRUNK_LEN);

    PipeBarrier<PIPE_V>();

    topkOp_.Batch4Rows(mergeValueLocal, historyValueLocal_, topkWorkspaceLocal, globalIndexLocal_, rowIdx4, topkNum4,
                       batchRowNum, mrgRowStride_, inputOffset, validLen, info.s2Idx, info.s2LoopEnd + 1);

    if (isLastS2InnerLoop) {
        PipeBarrier<PIPE_V>();
        CopyOutBatchRows(info, curAivGS1Idx, rowIdx4, topkNum4, batchRowNum, packedRowsLocal);
    }
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::CopyOutScoreGmBatchRows(
    const SICommon::RunInfo &info, const SICommon::RowIdx4 &rowIdx4, const SICommon::TopkNum4 &topkNum4,
    uint32_t batchRowNum, LocalTensor<uint32_t> &packedRowsLocal, LocalTensor<uint32_t> &lengthLocal,
    LocalTensor<uint32_t> &batchGlobalIndexLocal)
{
    int64_t curGS1Idx = static_cast<int64_t>(info.gS1Idx) * mBaseSize_;
    for (uint32_t lane = 0U; lane < batchRowNum; lane++) {
        uint32_t mInnerIdx = SICommon::GetLane(rowIdx4, lane);
        uint64_t globalGS1Idx = static_cast<uint64_t>(curGS1Idx) + mInnerIdx;
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
        uint32_t windowStart = curS1RealS2Len > windowSize_ ? curS1RealS2Len - windowSize_ : 0U;
        TopkUtils::PackSparseIndicesU32VF(
            packedRowsLocal[lane * PACKED_ROW_STRIDE], batchGlobalIndexLocal[lane * MAX_TOPK_COUNT],
            PACKED_ROW_STRIDE, initialBlocks_, topkSelectNum, windowSize_, windowStart);
        Duplicate(lengthLocal[lane * SCORE_GM_LENGTH_ROW_STRIDE], outputLen, 1U);
    }
    PipeBarrier<PIPE_V>();

    WaitFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
    SetFlag<HardEvent::V_MTE3>(topkVMte3Event_);
    WaitFlag<HardEvent::V_MTE3>(topkVMte3Event_);
    for (uint32_t lane = 0U; lane < batchRowNum; lane++) {
        uint32_t mInnerIdx = SICommon::GetLane(rowIdx4, lane);
        uint64_t globalGS1Idx = static_cast<uint64_t>(curGS1Idx) + mInnerIdx;
        uint32_t globalGIdx = static_cast<uint32_t>(globalGS1Idx / info.actS1Size);
        uint32_t globalS1Idx = static_cast<uint32_t>(globalGS1Idx % info.actS1Size);
        uint64_t physicalRow = static_cast<uint64_t>(globalGIdx) * constInfo_.qSeqSize + globalS1Idx;
        uint32_t outputLen =
            Min(Min(initialBlocks_ + Min(SICommon::GetLane(topkNum4, lane), MAX_TOPK_COUNT) + windowSize_,
                    PACKED_ROW_STRIDE),
                static_cast<uint32_t>(constInfo_.kSeqSize));

        AscendC::DataCopyParams outputParams{1U, static_cast<uint16_t>(outputLen * sizeof(int32_t)), 0U, 0U};
        AscendC::DataCopyParams lengthParams{1U, static_cast<uint16_t>(sizeof(uint32_t)), 0U, 0U};
        DataCopyPad(sparseIndicesGm[info.indiceOutOffset + physicalRow * constInfo_.kSeqSize],
                    packedRowsLocal[lane * PACKED_ROW_STRIDE].ReinterpretCast<int32_t>(), outputParams);
        DataCopyPad(sparseSeqLenGm[info.indiceLenOffset + physicalRow],
                    lengthLocal[lane * SCORE_GM_LENGTH_ROW_STRIDE].ReinterpretCast<int32_t>(), lengthParams);
    }
    SetFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::CopyOutScoreGmDirectRow(const SICommon::RunInfo &info, uint32_t mInnerIdx,
                                                              uint32_t outputLen,
                                                              LocalTensor<uint32_t> &indicesLocal,
                                                              LocalTensor<uint32_t> &lengthLocal)
{
    outputLen = Min(outputLen, PACKED_ROW_STRIDE);
    if (outputLen > 0U) {
        CreateVecIndex(indicesLocal.ReinterpretCast<int32_t>(), 0, outputLen);
    }
    Duplicate(lengthLocal, outputLen, 1U);
    PipeBarrier<PIPE_V>();

    int64_t curGS1Idx = static_cast<int64_t>(info.gS1Idx) * mBaseSize_;
    uint64_t globalGS1Idx = static_cast<uint64_t>(curGS1Idx) + mInnerIdx;
    uint32_t globalGIdx = static_cast<uint32_t>(globalGS1Idx / info.actS1Size);
    uint32_t globalS1Idx = static_cast<uint32_t>(globalGS1Idx % info.actS1Size);
    uint64_t physicalRow = static_cast<uint64_t>(globalGIdx) * constInfo_.qSeqSize + globalS1Idx;

    WaitFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
    SetFlag<HardEvent::V_MTE3>(topkVMte3Event_);
    WaitFlag<HardEvent::V_MTE3>(topkVMte3Event_);
    if (outputLen > 0U) {
        AscendC::DataCopyParams outputParams{1U, static_cast<uint16_t>(outputLen * sizeof(int32_t)), 0U, 0U};
        DataCopyPad(sparseIndicesGm[info.indiceOutOffset + physicalRow * constInfo_.kSeqSize],
                    indicesLocal.ReinterpretCast<int32_t>(), outputParams);
    }
    AscendC::DataCopyParams lengthParams{1U, static_cast<uint16_t>(sizeof(uint32_t)), 0U, 0U};
    DataCopyPad(sparseSeqLenGm[info.indiceLenOffset + physicalRow], lengthLocal.ReinterpretCast<int32_t>(),
                lengthParams);
    SetFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::RunScoreGmTopKBatch(
    const SICommon::RunInfo &info, const SICommon::RowIdx4 &rowIdx4, const SICommon::TopkNum4 &topkNum4,
    const SICommon::S2ValidLen4 &validEnd4, uint32_t batchRowNum, uint32_t packedSlotIdx)
{
    LocalTensor<SCORE_T> fastBuf = historyValueBuf_.Get<SCORE_T>();
    LocalTensor<SCORE_T> batchHistoryLocal = fastBuf[SCORE_GM_HISTORY_OFFSET_BYTES / sizeof(SCORE_T)];
    LocalTensor<uint32_t> fastGlobalBuf = globalIndexBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> topkWorkspaceLocal = fastGlobalBuf;
    LocalTensor<uint32_t> batchGlobalIndexLocal =
        fastGlobalBuf[SCORE_GM_GLOBAL_INDEX_OFFSET_BYTES / sizeof(uint32_t)];
    constexpr SICommon::RowIdx4 compactRowIdx4 = {0U, 1U, 2U, 3U};

    uint32_t maxValidEnd = 0U;
    for (uint32_t lane = 0U; lane < batchRowNum; lane++) {
        maxValidEnd = Max(maxValidEnd, SICommon::GetLane(validEnd4, lane));
    }
    uint32_t s2LoopNum = CeilDiv(maxValidEnd, SCORE_GM_TRUNK_LEN);
    s2LoopNum = s2LoopNum == 0U ? 1U : s2LoopNum;
    uint32_t activeTileNum = CeilDiv(info.actS2Size, S2_BASIC_BLOCK);

    IssueLoadScorePair(rowIdx4, 0U, activeTileNum, 0U);
    for (uint32_t s2LoopIdx = 0U; s2LoopIdx < s2LoopNum; s2LoopIdx++) {
        uint32_t mergeSlotIdx = s2LoopIdx & 1U;
        WaitFlag<HardEvent::MTE2_V>(vec1Mte2VEvent_[mergeSlotIdx]);
        if (s2LoopIdx + 1U < s2LoopNum) {
            IssueLoadScorePair(rowIdx4, s2LoopIdx + 1U, activeTileNum, mergeSlotIdx ^ 1U);
        }

        uint32_t mergeOffset =
            mergeSlotIdx == 0U ? SCORE_GM_MERGE0_OFFSET_BYTES : SCORE_GM_MERGE1_OFFSET_BYTES;
        LocalTensor<SCORE_T> mergeLocal = fastBuf[mergeOffset / sizeof(SCORE_T)];
        if (s2LoopIdx > 0U) {
            for (uint32_t lane = 0U; lane < TOPK_ROW_BATCH; lane++) {
                AscendC::DataCopy(mergeLocal[lane * SCORE_GM_MERGE_ROW_STRIDE],
                                  batchHistoryLocal[lane * TOPK_ALIGN_SIZE], TOPK_ALIGN_SIZE);
            }
            PipeBarrier<PIPE_V>();
        } else if (initialBlocks_ > 0U) {
            TopkUtils::PadMrgRowsInitBlockVF(mergeLocal, SCORE_GM_MERGE_ROW_STRIDE, TOPK_ROW_BATCH,
                                             TOPK_ALIGN_SIZE, initialBlocks_);
        }

        SICommon::S2ValidLen4 s2ValidLen4 = {0U, 0U, 0U, 0U};
        uint32_t trunkStart = s2LoopIdx * SCORE_GM_TRUNK_LEN;
        for (uint32_t lane = 0U; lane < TOPK_ROW_BATCH; lane++) {
            uint32_t validEnd = SICommon::GetLane(validEnd4, lane);
            uint32_t validLen =
                validEnd > trunkStart ? Min(validEnd - trunkStart, SCORE_GM_TRUNK_LEN) : 0U;
            SICommon::SetLane(s2ValidLen4, lane, validLen);
        }
        if (s2LoopIdx == 0U) {
            TopkUtils::PadMrgFourRowsFirstS2RangeVF<SCORE_T, SCORE_GM_TRUNK_LEN>(
                mergeLocal, compactRowIdx4, topkNum4, s2ValidLen4, SCORE_GM_MERGE_ROW_STRIDE);
        } else {
            TopkUtils::PadMrgFourRowsNotFirstS2RangeVF<SCORE_T, SCORE_GM_TRUNK_LEN>(
                mergeLocal, compactRowIdx4, topkNum4, s2ValidLen4, SCORE_GM_MERGE_ROW_STRIDE);
        }
        PipeBarrier<PIPE_V>();

        uint32_t inputOffset = s2LoopIdx == 0U ? TOPK_ALIGN_SIZE : 0U;
        uint32_t validLen =
            s2LoopIdx == 0U ? SCORE_GM_TRUNK_LEN : TOPK_ALIGN_SIZE + SCORE_GM_TRUNK_LEN;
        scoreGmTopkOp_.Batch4Rows(mergeLocal, batchHistoryLocal, topkWorkspaceLocal, batchGlobalIndexLocal,
                                  compactRowIdx4, topkNum4, batchRowNum, SCORE_GM_MERGE_ROW_STRIDE,
                                  inputOffset, validLen, s2LoopIdx, s2LoopNum);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(vec1VMte2Event_[mergeSlotIdx]);
    }

    uint32_t packedOffset = packedSlotIdx == 0U ? SCORE_GM_PACKED0_OFFSET_BYTES : SCORE_GM_PACKED1_OFFSET_BYTES;
    uint32_t lengthOffset = packedSlotIdx == 0U ? SCORE_GM_LENGTH0_OFFSET_BYTES : SCORE_GM_LENGTH1_OFFSET_BYTES;
    LocalTensor<uint32_t> packedRowsLocal =
        fastBuf[packedOffset / sizeof(SCORE_T)].template ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> lengthLocal =
        fastBuf[lengthOffset / sizeof(SCORE_T)].template ReinterpretCast<uint32_t>();
    CopyOutScoreGmBatchRows(info, rowIdx4, topkNum4, batchRowNum, packedRowsLocal, lengthLocal,
                            batchGlobalIndexLocal);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::ProcessScoreGmTopK(const SICommon::RunInfo &info)
{
    // 每次最终 score 存储都要消费一次该依赖，包括仅直写输出的 M 块。当存储
    // 行被 reload 时，此 wait 与随后的 DataCopy 都在 MTE2 上，因此不会被重排。
    WaitFlag<HardEvent::MTE3_MTE2>(topkMte3Mte2Event_);

    LocalTensor<SCORE_T> fastBuf = historyValueBuf_.Get<SCORE_T>();
    int64_t kbOffset = static_cast<int64_t>(info.actS2Size) - static_cast<int64_t>(info.actS1Size);
    int32_t numPromptK = static_cast<int32_t>(info.promptLen);
    int64_t curGS1Idx = static_cast<int64_t>(info.gS1Idx) * mBaseSize_;
    SICommon::RowIdx4 rowIdx4 = {0U, 0U, 0U, 0U};
    SICommon::TopkNum4 topkNum4 = {0U, 0U, 0U, 0U};
    SICommon::S2ValidLen4 validEnd4 = {0U, 0U, 0U, 0U};
    uint32_t batchRowNum = 0U;
    uint32_t packedSlotIdx = 0U;

    for (uint32_t mInnerIdx = 0U; mInnerIdx < info.actMBaseSize; mInnerIdx++) {
        uint64_t globalGS1Idx = static_cast<uint64_t>(curGS1Idx) + mInnerIdx;
        uint32_t globalS1Idx = static_cast<uint32_t>(globalGS1Idx % info.actS1Size);
        uint32_t realS2Len = info.actS2Size;
        if (constInfo_.attenMaskFlag) {
            int32_t realLen = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size) +
                              static_cast<int32_t>(globalS1Idx) + 1;
            realS2Len = static_cast<uint32_t>(realLen > 0 ? realLen : 0);
        }

        uint32_t topkSelectNum = Min(CalcDynamicTopkCount(globalS1Idx, kbOffset, numPromptK), MAX_TOPK_COUNT);
        uint32_t totalOutput = initialBlocks_ + topkSelectNum + windowSize_;
        bool directOutput =
            (constInfo_.attenMaskFlag && realS2Len <= initialBlocks_ + windowSize_) || realS2Len <= totalOutput;
        if (directOutput) {
            uint32_t packedOffset =
                packedSlotIdx == 0U ? SCORE_GM_PACKED0_OFFSET_BYTES : SCORE_GM_PACKED1_OFFSET_BYTES;
            uint32_t lengthOffset =
                packedSlotIdx == 0U ? SCORE_GM_LENGTH0_OFFSET_BYTES : SCORE_GM_LENGTH1_OFFSET_BYTES;
            LocalTensor<uint32_t> indicesLocal =
                fastBuf[packedOffset / sizeof(SCORE_T)].template ReinterpretCast<uint32_t>();
            LocalTensor<uint32_t> lengthLocal =
                fastBuf[lengthOffset / sizeof(SCORE_T)].template ReinterpretCast<uint32_t>();
            CopyOutScoreGmDirectRow(info, mInnerIdx, realS2Len, indicesLocal, lengthLocal);
            packedSlotIdx ^= 1U;
            continue;
        }

        SICommon::SetLane(rowIdx4, batchRowNum, mInnerIdx);
        SICommon::SetLane(topkNum4, batchRowNum, topkSelectNum);
        SICommon::SetLane(validEnd4, batchRowNum, realS2Len > windowSize_ ? realS2Len - windowSize_ : 0U);
        batchRowNum++;
        if (batchRowNum == TOPK_ROW_BATCH) {
            RunScoreGmTopKBatch(info, rowIdx4, topkNum4, validEnd4, batchRowNum, packedSlotIdx);
            packedSlotIdx ^= 1U;
            batchRowNum = 0U;
        }
    }

    if (batchRowNum > 0U) {
        uint32_t tailLane = batchRowNum - 1U;
        uint32_t tailRowIdx = SICommon::GetLane(rowIdx4, tailLane);
        uint32_t tailTopkNum = SICommon::GetLane(topkNum4, tailLane);
        uint32_t tailValidEnd = SICommon::GetLane(validEnd4, tailLane);
        for (uint32_t lane = batchRowNum; lane < TOPK_ROW_BATCH; lane++) {
            SICommon::SetLane(rowIdx4, lane, tailRowIdx);
            SICommon::SetLane(topkNum4, lane, tailTopkNum);
            SICommon::SetLane(validEnd4, lane, tailValidEnd);
        }
        RunScoreGmTopKBatch(info, rowIdx4, topkNum4, validEnd4, batchRowNum, packedSlotIdx);
    }

    WaitFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
    SetFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
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
                // 直写输出使用 packed slot 0；让下一个 TopK 批次在 MTE3 排空期间打包到 slot 1。
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
        // 打包输出复用当前 MM1 slot；MTE3 完成后才归还给 Cube。
        WaitFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
        SetFlag<HardEvent::MTE3_V>(topkMte3VEvent_);
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

        // 该分支没有 Cube/Vector slot 握手。使用 Vector 专用 UB，绝不用 Cube 持有的 MM1 slot。
        LocalTensor<uint32_t> indicesOutLocal = directIndexLocal_;
        LocalTensor<uint32_t> lengthLocal = directLengthLocal_;

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
