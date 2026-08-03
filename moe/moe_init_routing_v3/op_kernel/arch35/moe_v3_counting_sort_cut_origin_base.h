/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_counting_sort_cut_origin_base.h
 * \brief A5 CountingSort CutOrigin 模板基类（模板2）。
 *        多核切 T，核内按 4096 chunk 切 T（persistent + max(phase) UB 模型）。
 *        Phase A: 4096 chunk 过滤+计数（核间并行）
 *        Phase B: 核间计数归约 + 全局偏移计算
 *        Phase C: 按 perLoopCols 分列搬运 expandedX/expandedScale/expandedRowIdx
 *
 *        PERF TODO: 计数段后续可替换为 AscendC::Reg::Histograms。
 */
#ifndef MOE_V3_COUNTING_SORT_CUT_ORIGIN_BASE_H
#define MOE_V3_COUNTING_SORT_CUT_ORIGIN_BASE_H

#include "moe_v3_common.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3CountingSortCutOriginBase {
public:
    __aicore__ inline MoeV3CountingSortCutOriginBase(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX,
                                GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitGlobalBuffers(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX,
                                             GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                             GM_ADDR expandedScale, GM_ADDR workspace);
    __aicore__ inline void ComputeUbLayout();
    __aicore__ inline void LoadExpertIdxAndScale();
    __aicore__ inline void FilterAndCount();
    __aicore__ inline void WriteExpertCountToWorkspace();
    __aicore__ inline void ComputeGlobalOffset();
    __aicore__ inline void WriteExpertTokens();
    __aicore__ inline void GatherAndWrite();
    __aicore__ inline void GatherOneRow(int64_t newPos, int64_t tokenRow, int64_t origFlatIdx);

protected:
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;

    TPipe *pipe_;
    TBuf<TPosition::VECCALC> buf_;

    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int64_t> expertTokensGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<int32_t> workspaceGm_;

    int64_t blockIdx_;
    int64_t n_;
    int64_t k_;
    int64_t cols_;
    int64_t totalLength_;
    int64_t expertStart_;
    int64_t expertEnd_;
    int64_t actualExpertNum_;
    int64_t expertNum_;
    int64_t rowIdxType_;
    int64_t isInputScale_;
    int64_t filterNeedCoreNum_;
    int64_t coreNum_;
    int64_t expertTokensNumFlag_;
    int64_t expertTokensNumType_;
    int64_t activeNum_;
    int64_t dropPadMode_;
    int64_t expertCapacity_;
    int64_t outputRows_;

    int64_t coreTokenStart_;
    int64_t coreTokenEnd_;
    int64_t coreTokenNum_;
    int64_t coreFlatStart_;
    int64_t coreEntries_;
    int64_t expertCountStride_;
    int64_t colsAligned_;
    int64_t scaleSlotSize_;
    int64_t maxFilteredCount_;
    int64_t filteredCount_;
    int64_t filterChunkSize_;
    int64_t entriesAligned_;
    int64_t maskBytes_;

    // UB region offsets
    int64_t expertIdxLocalOffset_;
    int64_t scaleLocalOffset_;
    int64_t expertCountLocalOffset_;
    int64_t allCoreExpertCountLocalOffset_;
    int64_t expertTokensLocalOffset_;
    int64_t prefixSumLocalOffset_;
    int64_t filteredPairsLocalOffset_;
    int64_t xLocalOffset_;
    int64_t expertIdxFp32LocalOffset_;
    int64_t compareMask0Offset_;
    int64_t compareMask1Offset_;
    int64_t gatherMaskOffset_;
    int64_t gatheredExpertLocalOffset_;
    int64_t flatIdxBufferLocalOffset_;
    int64_t gatheredIdxLocalOffset_;
    int64_t totalBufSize_;
};

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::Init(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
    const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();

    n_ = tiling->n;
    k_ = tiling->k;
    cols_ = tiling->cols;
    totalLength_ = n_ * k_;
    expertStart_ = tiling->expertStart;
    expertEnd_ = tiling->expertEnd;
    actualExpertNum_ = tiling->actualExpertNum;
    expertNum_ = tiling->expertNum;
    rowIdxType_ = tiling->rowIdxType;
    isInputScale_ = tiling->isInputScale;
    filterNeedCoreNum_ = tiling->countingSortParamsOp.filterNeedCoreNum;
    coreNum_ = tiling->coreNum;
    expertTokensNumFlag_ = tiling->expertTokensNumFlag;
    expertTokensNumType_ = tiling->expertTokensNumType;
    activeNum_ = tiling->activeNum;
    dropPadMode_ = tiling->dropPadMode;
    expertCapacity_ = tiling->expertCapacity;
    outputRows_ = dropPadMode_ == DROP_PAD_MODE ? expertNum_ * expertCapacity_ : activeNum_;

    int64_t filterPerCoreTokens = tiling->countingSortParamsOp.filterPerCoreTokens;
    coreTokenStart_ = blockIdx_ * filterPerCoreTokens;
    coreTokenEnd_ = Min(coreTokenStart_ + filterPerCoreTokens, n_);
    if (blockIdx_ == filterNeedCoreNum_ - 1) {
        coreTokenEnd_ = n_;
    }
    coreTokenNum_ = coreTokenEnd_ - coreTokenStart_;
    coreFlatStart_ = coreTokenStart_ * k_;
    coreEntries_ = coreTokenNum_ * k_;

    expertCountStride_ = AlignElem(actualExpertNum_, COUNTING_SORT_ONE_BLOCK_ELEMENT);
    colsAligned_ = Align(cols_, static_cast<int64_t>(sizeof(T)));
    scaleSlotSize_ = BLOCK_BYTES / static_cast<int64_t>(sizeof(float));
    maxFilteredCount_ = coreEntries_;
    filteredCount_ = 0;
    filterChunkSize_ = tiling->countingSortParamsOp.filterChunkSize > 0
                           ? tiling->countingSortParamsOp.filterChunkSize
                           : COUNTING_SORT_FILTER_CHUNK_SIZE;

    entriesAligned_ = Ceil(filterChunkSize_, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;
    maskBytes_ = AlignBytes(Ceil(entriesAligned_, static_cast<int64_t>(8)), static_cast<int64_t>(1));

    InitGlobalBuffers(x, expertIdx, scale, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expandedScale,
                      workspace);

    if (rowIdxType_ == GATHER) {
        if (blockIdx_ < filterNeedCoreNum_) {
            GlobalTensor<int32_t> expandedRowIdxGmTmp = expandedRowIdxGm_[filterPerCoreTokens * k_ * blockIdx_];
            InitGlobalMemory(expandedRowIdxGmTmp, coreEntries_, static_cast<int32_t>(-1));
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
        SyncAll();
    }

    ComputeUbLayout();
    pipe_->InitBuffer(buf_, totalBufSize_);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::InitGlobalBuffers(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace)
{
    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx);
    expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx);
    workspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace);
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        expertTokensGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokensCountOrCumsum);
    }
    if (isInputScale_) {
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale);
        expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::ComputeUbLayout()
{
    // Phase A/B/C persistent + max(phase) 模型。这里按最宽数据区预算 UB。
    int64_t offset = 0;

    // expertIdxLocal: filterChunkSize int32（Phase A 输入）
    expertIdxLocalOffset_ = offset;
    offset += AlignBytes(filterChunkSize_, static_cast<int64_t>(sizeof(int32_t)));

    // expertCountLocal: expertCountStride int32（persistent 常驻）
    expertCountLocalOffset_ = offset;
    offset += AlignBytes(expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    // allCoreExpertCountLocal: filterNeedCoreNum * expertCountStride int32（Phase B）
    allCoreExpertCountLocalOffset_ = offset;
    offset += AlignBytes(filterNeedCoreNum_ * expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    // expertTokensLocal: actualExpertNum int64（Phase B）
    expertTokensLocalOffset_ = offset;
    offset += AlignBytes(actualExpertNum_, static_cast<int64_t>(sizeof(int64_t)));

    // prefixSumLocal: expertCountStride int32
    prefixSumLocalOffset_ = offset;
    offset += AlignBytes(expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    // filteredPairsLocal: maxFilteredCount * 2 int32（Phase A 输出 / Phase C 输入）
    filteredPairsLocalOffset_ = offset;
    offset += AlignBytes(maxFilteredCount_ * 2, static_cast<int64_t>(sizeof(int32_t)));

    // xLocal: 单 token 行，Phase C 用（按 csPerLoopCols 对齐）
    int64_t perLoopCols = cols_; // 简化：单列搬运
    xLocalOffset_ = offset;
    offset += AlignBytes(perLoopCols, static_cast<int64_t>(sizeof(T)));

    // scaleLocal: 单 scale slot（Phase C 用）
    scaleLocalOffset_ = offset;
    if (isInputScale_) {
        offset += AlignBytes(scaleSlotSize_, static_cast<int64_t>(sizeof(float)));
    }

    // Phase A 向量过滤辅助区
    expertIdxFp32LocalOffset_ = offset;
    offset += AlignBytes(entriesAligned_, static_cast<int64_t>(sizeof(float)));

    compareMask0Offset_ = offset;
    offset += maskBytes_;
    compareMask1Offset_ = offset;
    offset += maskBytes_;
    gatherMaskOffset_ = offset;
    offset += maskBytes_;

    gatheredExpertLocalOffset_ = offset;
    offset += AlignBytes(entriesAligned_, static_cast<int64_t>(sizeof(int32_t)));

    flatIdxBufferLocalOffset_ = offset;
    offset += AlignBytes(entriesAligned_, static_cast<int64_t>(sizeof(int32_t)));

    gatheredIdxLocalOffset_ = offset;
    offset += AlignBytes(entriesAligned_, static_cast<int64_t>(sizeof(int32_t)));

    totalBufSize_ = offset;
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::Process()
{
    if (blockIdx_ >= filterNeedCoreNum_) {
        SyncAll();
        return;
    }

    // Phase A: chunk 过滤+计数（核间并行）
    LoadExpertIdxAndScale();
    FilterAndCount();
    WriteExpertCountToWorkspace();

    SyncAll();

    // Phase B: 核间归约 + 偏移
    ComputeGlobalOffset();
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        WriteExpertTokens();
    }

    // Phase C: 搬运
    GatherAndWrite();
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::LoadExpertIdxAndScale()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(coreEntries_ * sizeof(int32_t)), 0, 0,
                                 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(expertIdxLocal, expertIdxGm_[coreFlatStart_], copyParams, padParams);
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::FilterAndCount()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // PERF TODO: 计数段可替换为 AscendC::Reg::Histograms。当前简化为一次性扫描整个 coreEntries_。
    // 标量扫描：构建 pairs + 计数。coreEntries_ 已经被 tiling 限制在 UB 范围内。
    filteredCount_ = 0;
    for (int64_t j = 0; j < coreEntries_; j++) {
        int32_t expertId = expertIdxLocal.GetValue(j);
        if (expertId < static_cast<int32_t>(expertStart_) || expertId >= static_cast<int32_t>(expertEnd_)) {
            continue;
        }
        int32_t expertOffset = expertId - static_cast<int32_t>(expertStart_);
        int32_t curCount = expertCountLocal.GetValue(expertOffset);
        expertCountLocal.SetValue(expertOffset, curCount + 1);
        filteredPairsLocal.SetValue(filteredCount_ * 2, static_cast<int32_t>(j));
        filteredPairsLocal.SetValue(filteredCount_ * 2 + 1, expertOffset);
        filteredCount_++;
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::WriteExpertCountToWorkspace()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(expertCountStride_ * sizeof(int32_t)),
                                 0, 0, 0};
    DataCopyPad(workspaceGm_[blockIdx_ * expertCountStride_], expertCountLocal, copyParams);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::ComputeGlobalOffset()
{
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workspaceGm_);

    LocalTensor<int32_t> allCoreExpertCountLocal =
        buf_.Get<int32_t>()[allCoreExpertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> prefixSumLocal = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    int64_t totalWsElements = filterNeedCoreNum_ * expertCountStride_;
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(totalWsElements * sizeof(int32_t)), 0,
                                 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(allCoreExpertCountLocal, workspaceGm_[0], copyParams, padParams);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    LocalTensor<int32_t> totalCountLocal = prefixSumLocal;
    Duplicate(totalCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    for (int64_t c = 0; c < filterNeedCoreNum_; c++) {
        Add(totalCountLocal, totalCountLocal, allCoreExpertCountLocal[c * expertCountStride_], expertCountStride_);
        PipeBarrier<PIPE_V>();
    }
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    int32_t totalForExpertArr[128];
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        totalForExpertArr[e] = totalCountLocal.GetValue(e);
    }

    Duplicate(prefixSumLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    for (int64_t c = 0; c < blockIdx_; c++) {
        Add(prefixSumLocal, prefixSumLocal, allCoreExpertCountLocal[c * expertCountStride_], expertCountStride_);
        PipeBarrier<PIPE_V>();
    }
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    LocalTensor<int64_t> expertTokensLocal = buf_.Get<int64_t>()[expertTokensLocalOffset_ / sizeof(int64_t)];
    int64_t cumulativeSum = 0;
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        int32_t totalForExpert = totalForExpertArr[e];
        int32_t prefixForExpert = prefixSumLocal.GetValue(e);

        if (blockIdx_ == 0 && expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
            expertTokensLocal.SetValue(e, static_cast<int64_t>(totalForExpert));
        }
        expertCountLocal.SetValue(e, static_cast<int32_t>(cumulativeSum) + prefixForExpert);
        cumulativeSum += totalForExpert;
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::WriteExpertTokens()
{
    if (blockIdx_ != 0) {
        return;
    }
    LocalTensor<int64_t> expertTokensLocal = buf_.Get<int64_t>()[expertTokensLocalOffset_ / sizeof(int64_t)];
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(actualExpertNum_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::GatherOneRow(int64_t newPos, int64_t tokenRow,
                                                                       int64_t origFlatIdx)
{
    LocalTensor<int32_t> idxBuf = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    if (rowIdxType_ == GATHER) {
        idxBuf.SetValue(0, static_cast<int32_t>(newPos));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expandedRowIdxGm_[origFlatIdx], idxBuf, idxCopyParams);
    } else {
        idxBuf.SetValue(0, static_cast<int32_t>(origFlatIdx));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expandedRowIdxGm_[newPos], idxBuf, idxCopyParams);
    }

    if (newPos < outputRows_) {
        // 单行 x：GM -> GM（先读到 UB 再写出）
        LocalTensor<T> xLocal = buf_.Get<T>()[xLocalOffset_ / sizeof(T)];
        DataCopyExtParams xCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        DataCopyPad(xLocal, xGm_[(coreTokenStart_ + tokenRow) * cols_], xCopyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        DataCopyPad(expandedXGm_[newPos * cols_], xLocal, xCopyParams);
        if (isInputScale_) {
            LocalTensor<float> scaleLocal = buf_.Get<float>()[scaleLocalOffset_ / sizeof(float)];
            DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0};
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            DataCopyPad(scaleLocal, scaleGm_[coreTokenStart_ + tokenRow], scaleCopyParams, scalePadParams);
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            DataCopyPad(expandedScaleGm_[newPos], scaleLocal, scaleCopyParams);
        }
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::GatherAndWrite()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    for (int64_t fIdx = 0; fIdx < filteredCount_; fIdx++) {
        int32_t localFlatIdx = filteredPairsLocal.GetValue(fIdx * 2);
        int32_t expertOffset = filteredPairsLocal.GetValue(fIdx * 2 + 1);

        int32_t newPos = expertCountLocal.GetValue(expertOffset);
        expertCountLocal.SetValue(expertOffset, newPos + 1);

        int64_t tokenRow = static_cast<int64_t>(localFlatIdx) / k_;
        int64_t origFlatIdx = coreFlatStart_ + static_cast<int64_t>(localFlatIdx);
        GatherOneRow(static_cast<int64_t>(newPos), tokenRow, origFlatIdx);
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_CUT_ORIGIN_BASE_H
