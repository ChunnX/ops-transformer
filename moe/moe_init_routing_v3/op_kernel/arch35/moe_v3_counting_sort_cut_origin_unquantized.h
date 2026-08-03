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
 * \file moe_v3_counting_sort_cut_origin_unquantized.h
 * \brief A5 CountingSort CutOrigin 非量化子类。
 *        Phase C 单行流式：xGm -> UB -> expandedX（直接搬运 x）；expandedScale 透传 per-token scale（isInputScale_）。
 */
#ifndef MOE_V3_COUNTING_SORT_CUT_ORIGIN_UNQUANTIZED_H
#define MOE_V3_COUNTING_SORT_CUT_ORIGIN_UNQUANTIZED_H

#include "moe_v3_counting_sort_cut_origin_base.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3CountingSortCutOriginUnquantized : public MoeV3CountingSortCutOriginBase<T> {
public:
    __aicore__ inline MoeV3CountingSortCutOriginUnquantized(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX,
                                GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GatherAndWrite();
    __aicore__ inline void GatherOneRow(int64_t newPos, int64_t tokenRow, int64_t origFlatIdx);
    __aicore__ inline void GatherAndWriteDoubleBuffer();

    GlobalTensor<T> xGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> expandedScaleGm_;
};

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginUnquantized<T>::Init(
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

    // GATHER 预填 rowIdx=-1；dropPad 还需整块清零 expandedX（空槽零填充）+ scale 透传空槽清零
    this->PrefillAndSync();
    if (this->rowIdxType_ == GATHER && this->dropPadMode_ == DROP_PAD_MODE) {
        this->ZeroOutExpandedDropPad(expandedXGm_);
        if (this->isInputScale_) {
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
__aicore__ inline void MoeV3CountingSortCutOriginUnquantized<T>::GatherOneRow(int64_t newPos, int64_t tokenRow,
                                                                              int64_t origFlatIdx)
{
    LocalTensor<int32_t> idxBuf = this->buf_.template Get<int32_t>()[this->prefixSumLocalOffset_ / sizeof(int32_t)];
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    if (this->rowIdxType_ == GATHER) {
        idxBuf.SetValue(0, static_cast<int32_t>(newPos));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(this->expandedRowIdxGm_[origFlatIdx], idxBuf, idxCopyParams);
    } else {
        idxBuf.SetValue(0, static_cast<int32_t>(origFlatIdx));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(this->expandedRowIdxGm_[newPos], idxBuf, idxCopyParams);
    }

    if (newPos >= this->outputRows_) {
        return;
    }

    // 单行 x：GM -> UB
    LocalTensor<T> xLocal = this->buf_.template Get<T>()[this->xLocalOffset_ / sizeof(T)];
    DataCopyExtParams xCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    DataCopyPad(xLocal, xGm_[(this->coreTokenStart_ + tokenRow) * this->cols_], xCopyParams, padParams);

    SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
    DataCopyPad(expandedXGm_[newPos * this->cols_], xLocal, xCopyParams);
    if (this->isInputScale_) {
        LocalTensor<float> scaleLocal = this->buf_.template Get<float>()[this->scaleLocalOffset_ / sizeof(float)];
        DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0};
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        DataCopyPad(scaleLocal, scaleGm_[this->coreTokenStart_ + tokenRow], scaleCopyParams, scalePadParams);
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        DataCopyPad(expandedScaleGm_[newPos], scaleLocal, scaleCopyParams);
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginUnquantized<T>::GatherAndWrite()
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

// Phase C 双缓冲 ping-pong（非量化，参照 A3 GatherAndWriteChunked colsLoops==1 路径）：
// 先用 A5 原有 slot/newPos 语义逐 pair 预计算 newPos/tokenRow/origFlatIdx（dropPad 丢弃在此处理），
// 再对存活行做 x 载入(MTE2)/写出(MTE3) ping-pong 重叠，隐藏 cols 大列整行搬运延迟。
template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginUnquantized<T>::GatherAndWriteDoubleBuffer()
{
    LocalTensor<int32_t> expertCountLocal =
        this->buf_.template Get<int32_t>()[this->expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal =
        this->buf_.template Get<int32_t>()[this->filteredPairsLocalOffset_ / sizeof(int32_t)];
    LocalTensor<T> xBuf0 = this->buf_.template Get<T>()[this->xBufOff0_ / sizeof(T)];
    LocalTensor<T> xBuf1 = this->buf_.template Get<T>()[this->xBufOff1_ / sizeof(T)];
    LocalTensor<int32_t> idxBuf0 = this->buf_.template Get<int32_t>()[this->idxBufDbOff0_ / sizeof(int32_t)];
    LocalTensor<int32_t> idxBuf1 = this->buf_.template Get<int32_t>()[this->idxBufDbOff1_ / sizeof(int32_t)];

    DataCopyExtParams xCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> xPadParams{false, 0, 0, 0};
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0};

    // 持久 event：两 slot 的 MTE2->MTE3（载入完成可写出）/ MTE3->MTE2（写出完成 slot 可复用）
    event_t evMte2[2] = {static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_MTE3)),
                         static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_MTE3))};
    event_t evMte3[2] = {static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_MTE2)),
                         static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_MTE2))};

    int64_t pairCursor = 0;
    while (pairCursor < this->filteredCount_) {
        int64_t batchSize = Min(static_cast<int64_t>(MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP),
                                this->filteredCount_ - pairCursor);

        // 预计算：应用 A5 原有 slot/newPos 语义。dropPad 丢弃行写 -1 rowIdx 并标记 skip。
        int32_t newPosArr[MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP];
        int32_t srcRowArr[MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP];
        int32_t origFlatIdxArr[MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP];
        bool skipArr[MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP];
        for (int64_t i = 0; i < batchSize; i++) {
            int64_t fIdx = pairCursor + i;
            int32_t localFlatIdx = filteredPairsLocal.GetValue(fIdx * 2);
            int32_t expertOffset = filteredPairsLocal.GetValue(fIdx * 2 + 1);
            int32_t slot = expertCountLocal.GetValue(expertOffset);
            expertCountLocal.SetValue(expertOffset, slot + 1);
            int64_t origFlatIdx = this->coreFlatStart_ + static_cast<int64_t>(localFlatIdx);
            srcRowArr[i] = static_cast<int32_t>(localFlatIdx) / static_cast<int32_t>(this->k_);
            origFlatIdxArr[i] = static_cast<int32_t>(origFlatIdx);
            if (this->dropPadMode_ == DROP_PAD_MODE) {
                if (slot >= this->expertCapacity_) {
                    skipArr[i] = true;
                    newPosArr[i] = -1;
                    continue;
                }
                newPosArr[i] = static_cast<int32_t>(
                    (static_cast<int64_t>(expertOffset) + this->expertStart_) * this->expertCapacity_ + slot);
            } else {
                newPosArr[i] = slot;
            }
            skipArr[i] = false;
        }

        // 先把所有行的 rowIdx 写出（含 dropPad 丢弃行写 -1），标量与 x 搬运解耦。
        // GATHER 且 origFlatIdx 整批连续（全通过 range 常见态）：批量 staging 后一条 DMA 写出，
        // 消除逐行 4B 小 DMA + S↔MTE3 双向同步往返（逐行写是 GATHER 场景的最长串行链）。
        bool contiguous = (this->rowIdxType_ == GATHER);
        for (int64_t i = 1; contiguous && i < batchSize; i++) {
            if (origFlatIdxArr[i] != origFlatIdxArr[0] + static_cast<int32_t>(i)) {
                contiguous = false;
            }
        }
        if (contiguous) {
            LocalTensor<int32_t> idxStage =
                this->buf_.template Get<int32_t>()[this->idxStageBufOff_ / sizeof(int32_t)];
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S); // 上一批 DMA 读完才可覆写 staging
            for (int64_t i = 0; i < batchSize; i++) {
                idxStage.SetValue(i, skipArr[i] ? static_cast<int32_t>(-1) : newPosArr[i]);
            }
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyExtParams stageCopyParams{static_cast<uint16_t>(1),
                                              static_cast<uint32_t>(batchSize * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(this->expandedRowIdxGm_[origFlatIdxArr[0]], idxStage, stageCopyParams);
        } else {
            // 注意 MTE3_S：S_MTE3 只保证本次 SetValue 先于本次 DMA；复用 idxBuf 前必须等上一次
            // MTE3 读完，否则背靠背小 DMA 下 S 覆写未读数据（反向冒险），rowIdx 大面积错值。
            for (int64_t i = 0; i < batchSize; i++) {
                LocalTensor<int32_t> &curIdxBuf = (i & 1) == 0 ? idxBuf0 : idxBuf1;
                if (this->rowIdxType_ == GATHER) {
                    int32_t val = skipArr[i] ? static_cast<int32_t>(-1) : newPosArr[i];
                    curIdxBuf.SetValue(0, val);
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    DataCopyPad(this->expandedRowIdxGm_[origFlatIdxArr[i]], curIdxBuf, idxCopyParams);
                    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                } else if (!skipArr[i]) {
                    curIdxBuf.SetValue(0, origFlatIdxArr[i]);
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    DataCopyPad(this->expandedRowIdxGm_[newPosArr[i]], curIdxBuf, idxCopyParams);
                    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                }
            }
        }
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

        // 收集存活行（newPos < outputRows_ 且未丢弃）用于 x ping-pong 搬运。
        // 复用 srcRowArr/newPosArr 原地压缩为存活序列。
        int64_t validCnt = 0;
        for (int64_t i = 0; i < batchSize; i++) {
            if (!skipArr[i] && newPosArr[i] < static_cast<int32_t>(this->outputRows_)) {
                srcRowArr[validCnt] = srcRowArr[i];
                newPosArr[validCnt] = newPosArr[i];
                validCnt++;
            }
        }

        if (validCnt > 0) {
            // isInputScale 透传：scale 双 slot 与 xBuf 同事件搭车（同 MTE2 载入/同 MTE3 写出，无需独立 event）
            LocalTensor<float> scBuf0;
            LocalTensor<float> scBuf1;
            if (this->isInputScale_) {
                scBuf0 = this->buf_.template Get<float>()[this->scaleBufDbOff0_ / sizeof(float)];
                scBuf1 = this->buf_.template Get<float>()[this->scaleBufDbOff1_ / sizeof(float)];
            }
            // Prologue: 预取存活行 0 -> xBuf0
            DataCopyPad(xBuf0, xGm_[static_cast<int64_t>(this->coreTokenStart_ + srcRowArr[0]) * this->cols_],
                        xCopyParams, xPadParams);
            if (this->isInputScale_) {
                DataCopyPad(scBuf0, scaleGm_[this->coreTokenStart_ + srcRowArr[0]], scaleCopyParams, scalePadParams);
            }
            SetFlag<HardEvent::MTE2_MTE3>(evMte2[0]);
            if (validCnt >= 2) {
                DataCopyPad(xBuf1, xGm_[static_cast<int64_t>(this->coreTokenStart_ + srcRowArr[1]) * this->cols_],
                            xCopyParams, xPadParams);
                if (this->isInputScale_) {
                    DataCopyPad(scBuf1, scaleGm_[this->coreTokenStart_ + srcRowArr[1]], scaleCopyParams,
                                scalePadParams);
                }
                SetFlag<HardEvent::MTE2_MTE3>(evMte2[1]);
            }
            for (int64_t i = 0; i < validCnt; i++) {
                int32_t bc = static_cast<int32_t>(i & 1);
                LocalTensor<T> &curXBuf = (bc == 0) ? xBuf0 : xBuf1;
                WaitFlag<HardEvent::MTE2_MTE3>(evMte2[bc]);
                DataCopyPad(expandedXGm_[static_cast<int64_t>(newPosArr[i]) * this->cols_], curXBuf, xCopyParams);
                if (this->isInputScale_) {
                    LocalTensor<float> &curScBuf = (bc == 0) ? scBuf0 : scBuf1;
                    DataCopyPad(expandedScaleGm_[static_cast<int64_t>(newPosArr[i])], curScBuf, scaleCopyParams);
                }
                SetFlag<HardEvent::MTE3_MTE2>(evMte3[bc]);
                if (i + 2 < validCnt) {
                    WaitFlag<HardEvent::MTE3_MTE2>(evMte3[bc]);
                    DataCopyPad(curXBuf,
                                xGm_[static_cast<int64_t>(this->coreTokenStart_ + srcRowArr[i + 2]) * this->cols_],
                                xCopyParams, xPadParams);
                    if (this->isInputScale_) {
                        LocalTensor<float> &curScBuf = (bc == 0) ? scBuf0 : scBuf1;
                        DataCopyPad(curScBuf, scaleGm_[this->coreTokenStart_ + srcRowArr[i + 2]], scaleCopyParams,
                                    scalePadParams);
                    }
                    SetFlag<HardEvent::MTE2_MTE3>(evMte2[bc]);
                }
            }
            // Epilogue: drain 剩余 MTE3 flag
            WaitFlag<HardEvent::MTE3_MTE2>(evMte3[(validCnt - 1) & 1]);
            if (validCnt >= 2) {
                WaitFlag<HardEvent::MTE3_MTE2>(evMte3[1 - ((validCnt - 1) & 1)]);
            }
        }

        pairCursor += batchSize;
    }
}


template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginUnquantized<T>::Process()
{
    if (this->blockIdx_ >= this->filterNeedCoreNum_) {
        SyncAll();
        return;
    }

    // Phase A: chunk 过滤+计数（核间并行）
    this->LoadExpertIdx();
    this->FilterAndCount();
    this->WriteExpertCountToWorkspace();

    SyncAll();

    // Phase B: 核间归约 + 偏移
    this->ComputeGlobalOffset();
    if (this->expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        this->WriteExpertTokens();
    }

    // Phase C: 搬运
    // csDbEnable_==0（tiling 判定双缓冲超 UB，大列场景）时走串行 fallback；
    // isInputScale 透传由 ping-pong 内 scale 双 slot 与 xBuf 同事件搭车覆盖。
    if (this->csDbEnable_ != 0) {
        GatherAndWriteDoubleBuffer();
    } else {
        GatherAndWrite();
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_CUT_ORIGIN_UNQUANTIZED_H
