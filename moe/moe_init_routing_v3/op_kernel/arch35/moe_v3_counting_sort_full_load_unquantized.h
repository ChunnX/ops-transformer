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
 * \file moe_v3_counting_sort_full_load_unquantized.h
 * \brief A5 CountingSort FullLoad 非量化子类。
 *        expandedX 直接搬运 x（无量化），expandedScale 直接搬运 per-token scale（isInputScale_）。
 */
#ifndef MOE_V3_COUNTING_SORT_FULL_LOAD_UNQUANTIZED_H
#define MOE_V3_COUNTING_SORT_FULL_LOAD_UNQUANTIZED_H

#include "moe_v3_counting_sort_full_load_base.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3CountingSortFullLoadUnquantized : public MoeV3CountingSortFullLoadBase<T> {
public:
    __aicore__ inline MoeV3CountingSortFullLoadUnquantized(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX,
                                GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadScale();
    __aicore__ inline void GatherAndWrite();
    __aicore__ inline void GatherOneRow(int64_t newPos, int64_t tokenRow, int64_t origFlatIdx);

    GlobalTensor<T> xGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> expandedScaleGm_;
};

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::Init(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
    const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe)
{
    this->InitCommon(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, workspace, tiling, pipe);

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX);
    if (this->isInputScale_) {
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale);
        expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);
    }

    // GATHER 预填 rowIdx=-1；dropPad 还需整块清零 expandedX（空槽零填充）
    this->PrefillAndSync();
    if (this->rowIdxType_ == GATHER && this->dropPadMode_ == DROP_PAD_MODE) {
        this->ZeroOutExpandedDropPad(expandedXGm_);
        if (this->isInputScale_) {
            // scale 透传空槽清零（golden 语义：空槽为 0）
            int64_t perCoreRows = Ceil(this->outputRows_, this->filterNeedCoreNum_);
            int64_t rowStart = this->blockIdx_ * perCoreRows;
            int64_t rowEnd = Min(rowStart + perCoreRows, this->outputRows_);
            if (rowStart < rowEnd) {
                GlobalTensor<float> scaleZeroSeg = expandedScaleGm_[rowStart];
                InitGlobalMemory(scaleZeroSeg, rowEnd - rowStart, static_cast<float>(0));
                SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            }
        }
    }
    if (this->rowIdxType_ == GATHER) {
        SyncAll();
    }

    // 非量化无量化临时区，UB 尺寸即公共区尺寸
    this->pipe_->InitBuffer(this->buf_, this->commonBufSize_);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::LoadScale()
{
    if (!this->isInputScale_) {
        return;
    }
    LocalTensor<float> scaleLocal = this->buf_.template Get<float>()[this->scaleLocalOffset_ / sizeof(float)];
    DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(this->coreTokenNum_), static_cast<uint32_t>(sizeof(float)),
                                      0, 0, 0};
    DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0};
    DataCopyPad(scaleLocal, scaleGm_[this->coreTokenStart_], scaleCopyParams, scalePadParams);
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::GatherOneRow(int64_t newPos, int64_t tokenRow,
                                                                             int64_t origFlatIdx)
{
    LocalTensor<T> xLocal = this->buf_.template Get<T>()[this->xLocalOffset_ / sizeof(T)];
    LocalTensor<float> scaleLocal = this->isInputScale_
                                        ? this->buf_.template Get<float>()[this->scaleLocalOffset_ / sizeof(float)]
                                        : LocalTensor<float>();

    DataCopyExtParams xCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    LocalTensor<int32_t> idxBuf = this->buf_.template Get<int32_t>()[this->prefixSumLocalOffset_ / sizeof(int32_t)];
    if (this->rowIdxType_ == GATHER) {
        idxBuf.SetValue(0, static_cast<int32_t>(newPos));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(this->expandedRowIdxGm_[origFlatIdx], idxBuf, idxCopyParams);
    } else {
        idxBuf.SetValue(0, static_cast<int32_t>(origFlatIdx));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(this->expandedRowIdxGm_[newPos], idxBuf, idxCopyParams);
    }

    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    if (newPos < this->outputRows_) {
        DataCopyPad(expandedXGm_[static_cast<int64_t>(newPos) * this->cols_],
                    xLocal[tokenRow * this->colsAligned_], xCopyParams);
        if (this->isInputScale_) {
            DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(expandedScaleGm_[newPos], scaleLocal[tokenRow * this->scaleSlotSize_], scaleCopyParams);
        }
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::GatherAndWrite()
{
    LocalTensor<int32_t> expertCountLocal =
        this->buf_.template Get<int32_t>()[this->expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal =
        this->buf_.template Get<int32_t>()[this->filteredPairsLocalOffset_ / sizeof(int32_t)];

    for (int64_t fIdx = 0; fIdx < this->filteredCount_; fIdx++) {
        int32_t localFlatIdx = filteredPairsLocal.GetValue(fIdx * 2);
        int32_t expertOffset = filteredPairsLocal.GetValue(fIdx * 2 + 1);

        int32_t slot = expertCountLocal.GetValue(expertOffset);
        expertCountLocal.SetValue(expertOffset, slot + 1);

        int64_t tokenRow = static_cast<int64_t>(localFlatIdx) / this->k_;
        int64_t origFlatIdx = this->coreFlatStart_ + static_cast<int64_t>(localFlatIdx);

        int64_t newPos;
        if (this->dropPadMode_ == DROP_PAD_MODE) {
            if (slot >= this->expertCapacity_) {
                if (this->rowIdxType_ == GATHER) {
                    LocalTensor<int32_t> dropIdxBuf =
                        this->buf_.template Get<int32_t>()[this->prefixSumLocalOffset_ / sizeof(int32_t)];
                    DataCopyExtParams dropIdxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)),
                                                        0, 0, 0};
                    dropIdxBuf.SetValue(0, static_cast<int32_t>(-1));
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    DataCopyPad(this->expandedRowIdxGm_[origFlatIdx], dropIdxBuf, dropIdxCopyParams);
                    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                }
                continue;
            }
            newPos = (static_cast<int64_t>(expertOffset) + this->expertStart_) * this->expertCapacity_ + slot;
        } else {
            newPos = static_cast<int64_t>(slot);
        }

        GatherOneRow(newPos, tokenRow, origFlatIdx);
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::Process()
{
    if (this->blockIdx_ >= this->filterNeedCoreNum_) {
        SyncAll();
        return;
    }

    // Phase A: Load + Filter + Count（核间并行）
    this->LoadExpertIdx();
    LoadScale();
    this->LoadXBackground(xGm_);
    this->FilterAndCount();
    this->WriteExpertCountToWorkspace();

    SyncAll();

    // Phase B: 核间归约 + 偏移计算
    this->ComputeGlobalOffset();
    if (this->expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        this->WriteExpertTokens();
    }

    // Phase C: 等待 x 加载 + 数据搬运
    this->WaitXLoadCommon();
    GatherAndWrite();
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_FULL_LOAD_UNQUANTIZED_H
