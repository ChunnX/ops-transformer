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
 * \file moe_v3_counting_sort_cut_origin_static_quant.h
 * \brief A5 CountingSort CutOrigin 静态量化子类。
 *        per-tensor 标量 scale/offset：q = round((x*scale+offset)) -> int8，全程 fp32 域运算对齐 golden。
 *        Phase C 单行流式：xGm -> UB -> 量化 -> expandedX(int8)。
 */
#ifndef MOE_V3_COUNTING_SORT_CUT_ORIGIN_STATIC_QUANT_H
#define MOE_V3_COUNTING_SORT_CUT_ORIGIN_STATIC_QUANT_H

#include "moe_v3_counting_sort_cut_origin_base.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3CountingSortCutOriginStaticQuant : public MoeV3CountingSortCutOriginBase<T> {
public:
    __aicore__ inline MoeV3CountingSortCutOriginStaticQuant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX,
                                GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void QuantStaticOneRow();
    __aicore__ inline void GatherAndWrite();
    __aicore__ inline void GatherOneRow(int64_t newPos, int64_t tokenRow, int64_t origFlatIdx);
    __aicore__ inline void GatherAndWriteDoubleBuffer();
    __aicore__ inline void QuantStaticSlot(LocalTensor<T> &xSlot, LocalTensor<int8_t> &i8Slot);

    GlobalTensor<T> xGm_;
    GlobalTensor<int8_t> expandedXInt8Gm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;
    float scale_ = 1.0f;
    float offset_ = 0.0f;

    int64_t quantI8LocalOffset_;
    int64_t quantI8LocalOffset1_; // 第二 int8 slot（双缓冲）

    constexpr static MicroAPI::CastTrait castTraitF32ToF16Static = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static MicroAPI::CastTrait castTraitF16ToI8Static = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
};

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginStaticQuant<T>::Init(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
    const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe)
{
    this->InitCommon(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, workspace, tiling, pipe);

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expandedXInt8Gm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);
    scaleGm_.SetGlobalBuffer((__gm__ float *)scale, 1);
    offsetGm_.SetGlobalBuffer((__gm__ float *)offset, 1);
    scale_ = scaleGm_.GetValue(0);
    offset_ = offsetGm_.GetValue(0);

    // 量化临时区：单行 int8 输出，追加在公共区之后
    quantI8LocalOffset_ = this->commonBufSize_;
    int64_t totalBufSize = quantI8LocalOffset_ + AlignBytes(this->colsAligned_, static_cast<int64_t>(sizeof(int8_t)));
    // 双缓冲：追加第二 int8 slot（x 双 slot 已在 base 公共区 xBufOff0_/xBufOff1_）；csDbEnable_==0 时不追加
    quantI8LocalOffset1_ = quantI8LocalOffset_;
    if (this->csDbEnable_ != 0) {
        quantI8LocalOffset1_ = totalBufSize;
        totalBufSize = quantI8LocalOffset1_ + AlignBytes(this->colsAligned_, static_cast<int64_t>(sizeof(int8_t)));
    }

    // 静态量化无 dropPad；GATHER 预填 rowIdx=-1
    this->PrefillAndSync();
    if (this->rowIdxType_ == GATHER) {
        SyncAll();
    }

    this->pipe_->InitBuffer(this->buf_, totalBufSize);
}

// 静态量化：读 xLocal 单行（已载入 offset 0），Muls(scale_)、Adds(offset_)、Cast->f16->int8，写 quantI8Local。
template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginStaticQuant<T>::QuantStaticOneRow()
{
    LocalTensor<T> xLocal = this->buf_.template Get<T>()[this->xLocalOffset_ / sizeof(T)];
    LocalTensor<int8_t> quantI8Local = this->buf_.template Get<int8_t>()[quantI8LocalOffset_];

    // quantI8Local 单行复用：等待上一行 DataCopyPad(MTE3) 读完再让 V 覆写，避免跨行 race
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

    __local_mem__ T *inUbAddr = (__local_mem__ T *)xLocal.GetPhyAddr();
    __local_mem__ int8_t *outUbAddr = (__local_mem__ int8_t *)quantI8Local.GetPhyAddr();

    uint16_t repeatTimes = Ceil(this->cols_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg = static_cast<uint32_t>(this->cols_);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg;
        MicroAPI::RegTensor<half> outRegF16;
        MicroAPI::RegTensor<int8_t> outRegI8;
        MicroAPI::MaskReg maskRegInLoop;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddr, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH);
            // scale/offset 运算在 fp32 域完成（对齐 golden：round(x*scale+offset) 全程 fp32），
            // 避免先窄化到 fp16 造成 ±0.5 边界舍入偏差；最终 f32->f16->int8 均用 RINT(round-half-even)。
            MicroAPI::Muls(inReg, inReg, scale_, maskRegInLoop);
            MicroAPI::Adds(inReg, inReg, offset_, maskRegInLoop);
            MicroAPI::Cast<half, float, castTraitF32ToF16Static>(outRegF16, inReg, maskRegInLoop);
            MicroAPI::Cast<int8_t, half, castTraitF16ToI8Static>(outRegI8, outRegF16, maskRegInLoop);
            MicroAPI::DataCopy<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                           outRegI8, maskRegInLoop);
        }
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginStaticQuant<T>::GatherOneRow(int64_t newPos, int64_t tokenRow,
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

    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    LocalTensor<int8_t> quantI8Local = this->buf_.template Get<int8_t>()[quantI8LocalOffset_];
    DataCopyExtParams xI8CopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0,
                                    0};
    QuantStaticOneRow();
    DataCopyPad(expandedXInt8Gm_[newPos * this->cols_], quantI8Local, xI8CopyParams);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginStaticQuant<T>::GatherAndWrite()
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

        // 静态量化非 dropPad：newPos 即全局 slot
        int64_t newPos = static_cast<int64_t>(slot);
        GatherOneRow(newPos, tokenRow, origFlatIdx);
    }
}

// 参数化静态量化：对给定 x slot 量化到给定 int8 slot（不含内部同步 flag，由调用方 ping-pong 管理）。
template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginStaticQuant<T>::QuantStaticSlot(LocalTensor<T> &xSlot,
                                                                                 LocalTensor<int8_t> &i8Slot)
{
    __local_mem__ T *inUbAddr = (__local_mem__ T *)xSlot.GetPhyAddr();
    __local_mem__ int8_t *outUbAddr = (__local_mem__ int8_t *)i8Slot.GetPhyAddr();
    uint16_t repeatTimes = Ceil(this->cols_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg = static_cast<uint32_t>(this->cols_);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg;
        MicroAPI::RegTensor<half> outRegF16;
        MicroAPI::RegTensor<int8_t> outRegI8;
        MicroAPI::MaskReg maskRegInLoop;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddr, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::Muls(inReg, inReg, scale_, maskRegInLoop);
            MicroAPI::Adds(inReg, inReg, offset_, maskRegInLoop);
            MicroAPI::Cast<half, float, castTraitF32ToF16Static>(outRegF16, inReg, maskRegInLoop);
            MicroAPI::Cast<int8_t, half, castTraitF16ToI8Static>(outRegI8, outRegF16, maskRegInLoop);
            MicroAPI::DataCopy<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                           outRegI8, maskRegInLoop);
        }
    }
}

// Phase C 双缓冲（静态量化，参照 A3 GatherAndWriteChunked static 路径）：
// x 双 slot 载入 + int8 双 slot 输出，V/MTE 交替重叠。先预计算存活行序列（静态量化无 dropPad，
// 存活条件 newPos<outputRows_），再对存活行 ping-pong。rowIdx 单独写出（含跳过行）。
// 每个 slot 三类独立 event（角色单一，避免同 event 多义导致死锁）：
//   evXReady[bc] (MTE2_V): x 载入完成 -> V 可量化
//   evXFree[bc]  (V_MTE2): x 被 V 读完 -> MTE2 可重载
//   evI8Free[bc] (MTE3_V): int8 写出完成 -> V 可覆写该 slot
template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginStaticQuant<T>::GatherAndWriteDoubleBuffer()
{
    LocalTensor<int32_t> expertCountLocal =
        this->buf_.template Get<int32_t>()[this->expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal =
        this->buf_.template Get<int32_t>()[this->filteredPairsLocalOffset_ / sizeof(int32_t)];
    LocalTensor<T> xBuf[2] = {this->buf_.template Get<T>()[this->xBufOff0_ / sizeof(T)],
                              this->buf_.template Get<T>()[this->xBufOff1_ / sizeof(T)]};
    LocalTensor<int8_t> i8Buf[2] = {this->buf_.template Get<int8_t>()[quantI8LocalOffset_],
                                    this->buf_.template Get<int8_t>()[quantI8LocalOffset1_]};
    LocalTensor<int32_t> idxBuf0 = this->buf_.template Get<int32_t>()[this->idxBufDbOff0_ / sizeof(int32_t)];
    LocalTensor<int32_t> idxBuf1 = this->buf_.template Get<int32_t>()[this->idxBufDbOff1_ / sizeof(int32_t)];

    DataCopyExtParams xCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> xPadParams{false, 0, 0, 0};
    DataCopyExtParams i8CopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0,
                                   0};
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    event_t evXReady[2] = {static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_V)),
                           static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_V))};
    event_t evXFree[2] = {static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE2)),
                          static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE2))};
    event_t evI8Free[2] = {static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_V)),
                           static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_V))};

    int64_t pairCursor = 0;
    while (pairCursor < this->filteredCount_) {
        int64_t batchSize = Min(static_cast<int64_t>(MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP),
                                this->filteredCount_ - pairCursor);
        int32_t newPosArr[MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP];
        int32_t srcRowArr[MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP];
        int32_t origFlatIdxArr[MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP];
        for (int64_t i = 0; i < batchSize; i++) {
            int64_t fIdx = pairCursor + i;
            int32_t localFlatIdx = filteredPairsLocal.GetValue(fIdx * 2);
            int32_t expertOffset = filteredPairsLocal.GetValue(fIdx * 2 + 1);
            int32_t slot = expertCountLocal.GetValue(expertOffset);
            expertCountLocal.SetValue(expertOffset, slot + 1);
            newPosArr[i] = slot; // 静态量化非 dropPad
            srcRowArr[i] = static_cast<int32_t>(localFlatIdx) / static_cast<int32_t>(this->k_);
            origFlatIdxArr[i] = static_cast<int32_t>(this->coreFlatStart_ + static_cast<int64_t>(localFlatIdx));
        }

        // rowIdx 全部写出。GATHER 且 origFlatIdx 整批连续（全通过 range 常见态）：批量 staging 一条 DMA 写出，
        // 消除逐行 4B 小 DMA + S↔MTE3 双向同步往返。
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
                idxStage.SetValue(i, newPosArr[i]);
            }
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyExtParams stageCopyParams{static_cast<uint16_t>(1),
                                              static_cast<uint32_t>(batchSize * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(this->expandedRowIdxGm_[origFlatIdxArr[0]], idxStage, stageCopyParams);
        } else {
            // 注意 MTE3_S：S_MTE3 只保证本次 SetValue 先于本次 DMA；
            // 复用 idxBuf 前必须等上一次 MTE3 读完，否则背靠背小 DMA 下 S 覆写未读数据（反向冒险），rowIdx 大面积错值。
            for (int64_t i = 0; i < batchSize; i++) {
                LocalTensor<int32_t> &curIdxBuf = (i & 1) == 0 ? idxBuf0 : idxBuf1;
                if (this->rowIdxType_ == GATHER) {
                    curIdxBuf.SetValue(0, newPosArr[i]);
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    DataCopyPad(this->expandedRowIdxGm_[origFlatIdxArr[i]], curIdxBuf, idxCopyParams);
                    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                } else {
                    curIdxBuf.SetValue(0, origFlatIdxArr[i]);
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    DataCopyPad(this->expandedRowIdxGm_[newPosArr[i]], curIdxBuf, idxCopyParams);
                    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                }
            }
        }
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

        // 压缩存活行
        int64_t validCnt = 0;
        for (int64_t i = 0; i < batchSize; i++) {
            if (newPosArr[i] < static_cast<int32_t>(this->outputRows_)) {
                srcRowArr[validCnt] = srcRowArr[i];
                newPosArr[validCnt] = newPosArr[i];
                validCnt++;
            }
        }

        if (validCnt > 0) {
            // 初始信号：两 int8 slot 空闲（V 可写）、slot1 x 空闲（MTE2 可载）
            SetFlag<HardEvent::MTE3_V>(evI8Free[0]);
            SetFlag<HardEvent::MTE3_V>(evI8Free[1]);
            SetFlag<HardEvent::V_MTE2>(evXFree[1]);

            // Prologue: 预取存活行 0 -> xBuf[0]
            DataCopyPad(xBuf[0], xGm_[static_cast<int64_t>(this->coreTokenStart_ + srcRowArr[0]) * this->cols_],
                        xCopyParams, xPadParams);
            SetFlag<HardEvent::MTE2_V>(evXReady[0]);

            for (int64_t i = 0; i < validCnt; i++) {
                int32_t bc = static_cast<int32_t>(i & 1);
                int32_t nc = bc ^ 1;

                // 预取下一 pair 到 nextXBuf（在量化当前 slot 之前发射，与 V 重叠）
                if (i + 1 < validCnt) {
                    WaitFlag<HardEvent::V_MTE2>(evXFree[nc]);
                    DataCopyPad(xBuf[nc],
                                xGm_[static_cast<int64_t>(this->coreTokenStart_ + srcRowArr[i + 1]) * this->cols_],
                                xCopyParams, xPadParams);
                    SetFlag<HardEvent::MTE2_V>(evXReady[nc]);
                }

                // 量化当前 slot：等 x 就绪 + int8 slot 空闲
                WaitFlag<HardEvent::MTE2_V>(evXReady[bc]);
                WaitFlag<HardEvent::MTE3_V>(evI8Free[bc]);
                QuantStaticSlot(xBuf[bc], i8Buf[bc]);
                SetFlag<HardEvent::V_MTE2>(evXFree[bc]); // x 读完，可重载

                // 写出 int8
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyPad(expandedXInt8Gm_[static_cast<int64_t>(newPosArr[i]) * this->cols_], i8Buf[bc],
                            i8CopyParams);
                SetFlag<HardEvent::MTE3_V>(evI8Free[bc]); // int8 slot 写出，可复用
            }

            // Epilogue: drain 所有未消费信号。配对核算：每 slot evXFree 恰余 1 个 Set
            // （slot1 初始 Set + 循环各迭代 Set，减去 validCnt-1 次预取 Wait），evI8Free 同理各余 1。
            WaitFlag<HardEvent::V_MTE2>(evXFree[0]);
            WaitFlag<HardEvent::V_MTE2>(evXFree[1]);
            WaitFlag<HardEvent::MTE3_V>(evI8Free[0]);
            WaitFlag<HardEvent::MTE3_V>(evI8Free[1]);
        }

        pairCursor += batchSize;
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginStaticQuant<T>::Process()
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
    if (this->csDbEnable_ != 0) {
        GatherAndWriteDoubleBuffer();
    } else {
        GatherAndWrite();
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_CUT_ORIGIN_STATIC_QUANT_H
