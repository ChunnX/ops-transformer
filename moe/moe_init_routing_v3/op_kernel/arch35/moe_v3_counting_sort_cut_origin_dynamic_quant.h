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
 * \file moe_v3_counting_sort_cut_origin_dynamic_quant.h
 * \brief A5 CountingSort CutOrigin 动态量化(int8)子类。
 *        per-token absmax scale：可选 per-expert smooth((LE,H))；q = round(x/scale) -> int8，
 *        per-token scale 透传到 expandedScale。pass2 与内置 ComputeMultiPass 对齐（smooth 预乘写回、孤立 Div）。
 *        Phase C 单行流式：xGm -> UB -> 量化 -> expandedX(int8) + expandedScale。
 */
#ifndef MOE_V3_COUNTING_SORT_CUT_ORIGIN_DYNAMIC_QUANT_H
#define MOE_V3_COUNTING_SORT_CUT_ORIGIN_DYNAMIC_QUANT_H

#include "moe_v3_counting_sort_cut_origin_base.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3CountingSortCutOriginDynamicQuant : public MoeV3CountingSortCutOriginBase<T> {
public:
    __aicore__ inline MoeV3CountingSortCutOriginDynamicQuant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX,
                                GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void QuantDynamicOneRow(int64_t expertOffset, float &outScale);
    __aicore__ inline void GatherAndWrite();
    __aicore__ inline void GatherOneRow(int64_t newPos, int64_t tokenRow, int64_t origFlatIdx, int64_t expertOffset);
    __aicore__ inline void GatherAndWriteDoubleBuffer();
    __aicore__ inline void QuantDynamicSlot(LocalTensor<T> &xSlot, LocalTensor<int8_t> &i8Slot,
                                            LocalTensor<float> &scaleSlot, __local_mem__ float *smoothUbAddr);

    GlobalTensor<T> xGm_;
    GlobalTensor<int8_t> expandedXInt8Gm_;
    GlobalTensor<float> scaleGm_;         // per-expert smooth (LE,H)，仅 isInputScale_
    GlobalTensor<float> expandedScaleGm_; // per-token scale 输出

    int64_t quantI8LocalOffset_;   // 单行 int8 输出
    int64_t quantScaleSlotOffset_; // 动态量化 scale slot
    int64_t smoothLocalOffset_;    // per-expert smooth 向量单行缓冲（colsAligned floats，仅 isInputScale_）
    int64_t quantI8LocalOffset1_;   // 第二 int8 slot（双缓冲）
    int64_t quantScaleSlotOffset1_; // 第二 scale slot
    int64_t smoothLocalOffset1_;    // 第二 smooth slot（smooth 兼作 V 预乘写回 scratch，须随 slot 双份）

    constexpr static MicroAPI::CastTrait castTraitF32ToF16Dyn = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    constexpr static MicroAPI::CastTrait castTraitF16ToI8Dyn = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
};

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginDynamicQuant<T>::Init(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
    const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe)
{
    this->InitCommon(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, workspace, tiling, pipe);

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expandedXInt8Gm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);
    expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);
    if (this->isInputScale_) {
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale);
    }

    // 量化临时区：单行 int8 输出 + scale slot + 可选 per-expert smooth 单行，追加在公共区之后
    int64_t offsetBytes = this->commonBufSize_;
    quantI8LocalOffset_ = offsetBytes;
    offsetBytes += AlignBytes(this->colsAligned_, static_cast<int64_t>(sizeof(int8_t)));
    quantScaleSlotOffset_ = offsetBytes;
    offsetBytes += AlignBytes(this->scaleSlotSize_, static_cast<int64_t>(sizeof(float)));
    smoothLocalOffset_ = offsetBytes;
    if (this->isInputScale_) {
        offsetBytes += AlignBytes(this->colsAligned_, static_cast<int64_t>(sizeof(float)));
    }
    // 双缓冲：第二 int8/scale/smooth slot（x 双 slot 已在 base 公共区 xBufOff0_/xBufOff1_）。
    // smooth slot 兼作 pass1 预乘写回 scratch，必须随 slot 双份，避免跨行覆写。csDbEnable_==0 时不追加。
    quantI8LocalOffset1_ = quantI8LocalOffset_;
    quantScaleSlotOffset1_ = quantScaleSlotOffset_;
    smoothLocalOffset1_ = smoothLocalOffset_;
    if (this->csDbEnable_ != 0) {
        quantI8LocalOffset1_ = offsetBytes;
        offsetBytes += AlignBytes(this->colsAligned_, static_cast<int64_t>(sizeof(int8_t)));
        quantScaleSlotOffset1_ = offsetBytes;
        offsetBytes += AlignBytes(this->scaleSlotSize_, static_cast<int64_t>(sizeof(float)));
        smoothLocalOffset1_ = offsetBytes;
        if (this->isInputScale_) {
            offsetBytes += AlignBytes(this->colsAligned_, static_cast<int64_t>(sizeof(float)));
        }
    }
    int64_t totalBufSize = offsetBytes;

    // 动态量化无 dropPad；GATHER 预填 rowIdx=-1
    this->PrefillAndSync();
    if (this->rowIdxType_ == GATHER) {
        SyncAll();
    }

    this->pipe_->InitBuffer(this->buf_, totalBufSize);
}

// 动态量化(int8)：可选 per-expert smooth（x*=smooth[expert]）-> per-token absmax -> scale=absmax/127；
// q=round(x/scale)->int8。读 xLocal 单行（已载入 offset 0），写 quantI8Local + outScale。
template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginDynamicQuant<T>::QuantDynamicOneRow(int64_t expertOffset,
                                                                                     float &outScale)
{
    LocalTensor<T> xLocal = this->buf_.template Get<T>()[this->xLocalOffset_ / sizeof(T)];
    LocalTensor<int8_t> quantI8Local = this->buf_.template Get<int8_t>()[quantI8LocalOffset_];
    LocalTensor<float> scaleSlot = this->buf_.template Get<float>()[quantScaleSlotOffset_ / sizeof(float)];

    // 动态量化透传 smooth：按专家取 (LE,H) 的一行 H 列到单行缓冲
    LocalTensor<float> smoothLocal;
    if (this->isInputScale_) {
        smoothLocal = this->buf_.template Get<float>()[smoothLocalOffset_ / sizeof(float)];
        DataCopyExtParams smoothCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(float)),
                                           0, 0, 0};
        DataCopyPadExtParams<float> smoothPadParams{false, 0, 0, 0};
        // smoothLocal 单行复用：等待上一行 V 读完再让本行 MTE2 覆写
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        DataCopyPad(smoothLocal, scaleGm_[expertOffset * this->cols_], smoothCopyParams, smoothPadParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }

    // quantI8Local/scaleSlot 单行复用：等待上一行 DataCopyPad(MTE3) 读完再让 V 覆写，避免跨行 race
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

    __local_mem__ T *inUbAddr = (__local_mem__ T *)xLocal.GetPhyAddr();
    __local_mem__ int8_t *outUbAddr = (__local_mem__ int8_t *)quantI8Local.GetPhyAddr();
    __local_mem__ float *scaleUbAddr = (__local_mem__ float *)scaleSlot.GetPhyAddr();
    __local_mem__ float *smoothUbAddr = this->isInputScale_ ? (__local_mem__ float *)smoothLocal.GetPhyAddr() : nullptr;
    bool applySmooth = (this->isInputScale_ != 0);

    uint16_t repeatTimes = Ceil(this->cols_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t maskLength = static_cast<uint32_t>(this->cols_);
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, absReg, smoothReg, scaleValueReg;
        MicroAPI::Duplicate(scaleValueReg, 0.0f);
        MicroAPI::RegTensor<half> outRegF16;
        MicroAPI::RegTensor<int8_t> outRegI8;
        MicroAPI::MaskReg maskRegInLoop;
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegVL1 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

        // pass1: (可选 smooth) -> 写回 smoothUbAddr -> absmax
        // 与内置 ComputeMultiPass 对齐：smooth 在 pass1 预乘并写回 UB，pass2 只做孤立 Div，
        // 避免 pass2 连续 Mul+Div 融合(FMA)带来的中间舍入差异导致边界元素 ±1。
        sreg = maskLength;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddr, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH);
            if (applySmooth) {
                MicroAPI::DataCopy(smoothReg, smoothUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
                MicroAPI::Mul(inReg, inReg, smoothReg, maskRegInLoop);
                MicroAPI::DataCopy(smoothUbAddr + i * FLOAT_REG_TENSOR_LENGTH, inReg, maskRegInLoop);
            }
            MicroAPI::Abs(absReg, inReg, maskRegInLoop);
            MicroAPI::Max(scaleValueReg, scaleValueReg, absReg, maskRegAll);
        }
        MicroAPI::ReduceMax(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / CS_DYNAMIC_QUANT_INT8_SCALE, maskRegVL1);
        MicroAPI::Duplicate(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::DataCopy(scaleUbAddr, scaleValueReg, maskRegVL1);
        if (applySmooth) {
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }

        // pass2: q = round(x_pre / scale) -> int8（x_pre 为 pass1 预乘 smooth 后的值，只做孤立 Div）
        sreg = maskLength;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            if (applySmooth) {
                MicroAPI::DataCopy(inReg, smoothUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            } else {
                ops::LoadOneTensorForDtypeT<T>(inUbAddr, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH);
            }
            MicroAPI::Div(inReg, inReg, scaleValueReg, maskRegInLoop);
            MicroAPI::Cast<half, float, castTraitF32ToF16Dyn>(outRegF16, inReg, maskRegInLoop);
            MicroAPI::Cast<int8_t, half, castTraitF16ToI8Dyn>(outRegI8, outRegF16, maskRegInLoop);
            MicroAPI::DataCopy<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                           outRegI8, maskRegInLoop);
        }
    }
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    outScale = scaleSlot.GetValue(0);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginDynamicQuant<T>::GatherOneRow(int64_t newPos, int64_t tokenRow,
                                                                               int64_t origFlatIdx, int64_t expertOffset)
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
    LocalTensor<float> scaleSlot = this->buf_.template Get<float>()[quantScaleSlotOffset_ / sizeof(float)];
    DataCopyExtParams xI8CopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0,
                                    0};
    DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    float tokenScale = 0.0f;
    QuantDynamicOneRow(expertOffset, tokenScale);
    DataCopyPad(expandedXInt8Gm_[newPos * this->cols_], quantI8Local, xI8CopyParams);
    DataCopyPad(expandedScaleGm_[newPos], scaleSlot, scaleCopyParams);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginDynamicQuant<T>::GatherAndWrite()
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

        // 动态量化非 dropPad：newPos 即全局 slot
        int64_t newPos = static_cast<int64_t>(slot);
        GatherOneRow(newPos, tokenRow, origFlatIdx, static_cast<int64_t>(expertOffset));
    }
}

// 参数化动态量化：对给定 x slot 量化到给定 int8/scale slot（不含内部同步 flag，由调用方 ping-pong 管理）。
// smoothUbAddr 非空时执行 per-expert smooth 预乘（pass1 写回该 scratch，pass2 孤立 Div，对齐 golden）。
// 与 QuantDynamicOneRow 唯一差异：per-token scale 不再 V_S 标量回读（双缓冲路径直接从 scaleSlot DMA 写出）。
template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginDynamicQuant<T>::QuantDynamicSlot(LocalTensor<T> &xSlot,
                                                                                   LocalTensor<int8_t> &i8Slot,
                                                                                   LocalTensor<float> &scaleSlot,
                                                                                   __local_mem__ float *smoothUbAddr)
{
    __local_mem__ T *inUbAddr = (__local_mem__ T *)xSlot.GetPhyAddr();
    __local_mem__ int8_t *outUbAddr = (__local_mem__ int8_t *)i8Slot.GetPhyAddr();
    __local_mem__ float *scaleUbAddr = (__local_mem__ float *)scaleSlot.GetPhyAddr();
    bool applySmooth = (smoothUbAddr != nullptr);

    uint16_t repeatTimes = Ceil(this->cols_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t maskLength = static_cast<uint32_t>(this->cols_);
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, absReg, smoothReg, scaleValueReg;
        MicroAPI::Duplicate(scaleValueReg, 0.0f);
        MicroAPI::RegTensor<half> outRegF16;
        MicroAPI::RegTensor<int8_t> outRegI8;
        MicroAPI::MaskReg maskRegInLoop;
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegVL1 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

        // pass1: (可选 smooth 预乘写回) -> absmax（与 QuantDynamicOneRow 完全一致）
        sreg = maskLength;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddr, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH);
            if (applySmooth) {
                MicroAPI::DataCopy(smoothReg, smoothUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
                MicroAPI::Mul(inReg, inReg, smoothReg, maskRegInLoop);
                MicroAPI::DataCopy(smoothUbAddr + i * FLOAT_REG_TENSOR_LENGTH, inReg, maskRegInLoop);
            }
            MicroAPI::Abs(absReg, inReg, maskRegInLoop);
            MicroAPI::Max(scaleValueReg, scaleValueReg, absReg, maskRegAll);
        }
        MicroAPI::ReduceMax(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / CS_DYNAMIC_QUANT_INT8_SCALE, maskRegVL1);
        MicroAPI::Duplicate(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::DataCopy(scaleUbAddr, scaleValueReg, maskRegVL1);
        if (applySmooth) {
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }

        // pass2: q = round(x_pre / scale) -> int8
        sreg = maskLength;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            if (applySmooth) {
                MicroAPI::DataCopy(inReg, smoothUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            } else {
                ops::LoadOneTensorForDtypeT<T>(inUbAddr, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH);
            }
            MicroAPI::Div(inReg, inReg, scaleValueReg, maskRegInLoop);
            MicroAPI::Cast<half, float, castTraitF32ToF16Dyn>(outRegF16, inReg, maskRegInLoop);
            MicroAPI::Cast<int8_t, half, castTraitF16ToI8Dyn>(outRegI8, outRegF16, maskRegInLoop);
            MicroAPI::DataCopy<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                            outRegI8, maskRegInLoop);
        }
    }
}

// Phase C 双缓冲（动态量化，参照 A3 GatherAndWriteDroppad dynamic 路径）：
// x 双 slot + int8/scale 双 slot 输出（smooth scratch 亦双 slot），V 量化（两趟 VF，最长串行链）
// 与相邻行 MTE2 载入 / MTE3 写出重叠。先预计算存活行序列（动态量化无 dropPad，存活条件
// newPos<outputRows_），再对存活行 ping-pong。rowIdx 单独写出。
// 每个 slot 三类独立 event（角色单一，避免同 event 多义导致死锁）：
//   evXReady[bc] (MTE2_V): x（含 smooth）载入完成 -> V 可量化
//   evXFree[bc]  (V_MTE2): x/smooth 被 V 读完 -> MTE2 可重载
//   evOutFree[bc](MTE3_V): int8+scale 写出完成 -> V 可覆写该 slot
template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginDynamicQuant<T>::GatherAndWriteDoubleBuffer()
{
    LocalTensor<int32_t> expertCountLocal =
        this->buf_.template Get<int32_t>()[this->expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal =
        this->buf_.template Get<int32_t>()[this->filteredPairsLocalOffset_ / sizeof(int32_t)];
    LocalTensor<T> xBuf[2] = {this->buf_.template Get<T>()[this->xBufOff0_ / sizeof(T)],
                              this->buf_.template Get<T>()[this->xBufOff1_ / sizeof(T)]};
    LocalTensor<int8_t> i8Buf[2] = {this->buf_.template Get<int8_t>()[quantI8LocalOffset_],
                                    this->buf_.template Get<int8_t>()[quantI8LocalOffset1_]};
    LocalTensor<float> scaleBuf[2] = {this->buf_.template Get<float>()[quantScaleSlotOffset_ / sizeof(float)],
                                      this->buf_.template Get<float>()[quantScaleSlotOffset1_ / sizeof(float)]};
    LocalTensor<float> smoothBuf[2];
    __local_mem__ float *smoothUbAddr[2] = {nullptr, nullptr};
    if (this->isInputScale_) {
        smoothBuf[0] = this->buf_.template Get<float>()[smoothLocalOffset_ / sizeof(float)];
        smoothBuf[1] = this->buf_.template Get<float>()[smoothLocalOffset1_ / sizeof(float)];
        smoothUbAddr[0] = (__local_mem__ float *)smoothBuf[0].GetPhyAddr();
        smoothUbAddr[1] = (__local_mem__ float *)smoothBuf[1].GetPhyAddr();
    }
    LocalTensor<int32_t> idxBuf0 = this->buf_.template Get<int32_t>()[this->idxBufDbOff0_ / sizeof(int32_t)];
    LocalTensor<int32_t> idxBuf1 = this->buf_.template Get<int32_t>()[this->idxBufDbOff1_ / sizeof(int32_t)];

    DataCopyExtParams xCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> xPadParams{false, 0, 0, 0};
    DataCopyExtParams smoothCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(float)), 0,
                                       0, 0};
    DataCopyPadExtParams<float> smoothPadParams{false, 0, 0, 0};
    DataCopyExtParams i8CopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0,
                                   0};
    DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    event_t evXReady[2] = {static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_V)),
                           static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_V))};
    event_t evXFree[2] = {static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE2)),
                          static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE2))};
    event_t evOutFree[2] = {static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_V)),
                            static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_V))};

    int64_t pairCursor = 0;
    while (pairCursor < this->filteredCount_) {
        int64_t batchSize = Min(static_cast<int64_t>(MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP),
                                this->filteredCount_ - pairCursor);
        int32_t newPosArr[MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP];
        int32_t srcRowArr[MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP];
        int32_t origFlatIdxArr[MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP];
        int32_t expertOffArr[MoeV3CountingSortCutOriginBase<T>::CS_DB_BATCH_CAP];
        for (int64_t i = 0; i < batchSize; i++) {
            int64_t fIdx = pairCursor + i;
            int32_t localFlatIdx = filteredPairsLocal.GetValue(fIdx * 2);
            int32_t expertOffset = filteredPairsLocal.GetValue(fIdx * 2 + 1);
            int32_t slot = expertCountLocal.GetValue(expertOffset);
            expertCountLocal.SetValue(expertOffset, slot + 1);
            newPosArr[i] = slot; // 动态量化非 dropPad
            srcRowArr[i] = static_cast<int32_t>(localFlatIdx) / static_cast<int32_t>(this->k_);
            origFlatIdxArr[i] = static_cast<int32_t>(this->coreFlatStart_ + static_cast<int64_t>(localFlatIdx));
            expertOffArr[i] = expertOffset;
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
                expertOffArr[validCnt] = expertOffArr[i];
                validCnt++;
            }
        }

        if (validCnt > 0) {
            // 初始信号：两输出 slot 空闲（V 可写）、slot1 x 空闲（MTE2 可载）
            SetFlag<HardEvent::MTE3_V>(evOutFree[0]);
            SetFlag<HardEvent::MTE3_V>(evOutFree[1]);
            SetFlag<HardEvent::V_MTE2>(evXFree[1]);

            // Prologue: 预取存活行 0 -> xBuf[0]（含 smooth）
            DataCopyPad(xBuf[0], xGm_[static_cast<int64_t>(this->coreTokenStart_ + srcRowArr[0]) * this->cols_],
                        xCopyParams, xPadParams);
            if (this->isInputScale_) {
                DataCopyPad(smoothBuf[0], scaleGm_[static_cast<int64_t>(expertOffArr[0]) * this->cols_],
                            smoothCopyParams, smoothPadParams);
            }
            SetFlag<HardEvent::MTE2_V>(evXReady[0]);

            for (int64_t i = 0; i < validCnt; i++) {
                int32_t bc = static_cast<int32_t>(i & 1);
                int32_t nc = bc ^ 1;

                // 预取下一行到 nextXBuf（在量化当前 slot 之前发射，与两趟 VF 重叠）
                if (i + 1 < validCnt) {
                    WaitFlag<HardEvent::V_MTE2>(evXFree[nc]);
                    DataCopyPad(xBuf[nc],
                                xGm_[static_cast<int64_t>(this->coreTokenStart_ + srcRowArr[i + 1]) * this->cols_],
                                xCopyParams, xPadParams);
                    if (this->isInputScale_) {
                        DataCopyPad(smoothBuf[nc], scaleGm_[static_cast<int64_t>(expertOffArr[i + 1]) * this->cols_],
                                    smoothCopyParams, smoothPadParams);
                    }
                    SetFlag<HardEvent::MTE2_V>(evXReady[nc]);
                }

                // 量化当前 slot：等 x/smooth 就绪 + 输出 slot 空闲
                WaitFlag<HardEvent::MTE2_V>(evXReady[bc]);
                WaitFlag<HardEvent::MTE3_V>(evOutFree[bc]);
                QuantDynamicSlot(xBuf[bc], i8Buf[bc], scaleBuf[bc], smoothUbAddr[bc]);
                SetFlag<HardEvent::V_MTE2>(evXFree[bc]); // x/smooth 读完，可重载

                // 写出 int8 + per-token scale
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyPad(expandedXInt8Gm_[static_cast<int64_t>(newPosArr[i]) * this->cols_], i8Buf[bc],
                            i8CopyParams);
                DataCopyPad(expandedScaleGm_[newPosArr[i]], scaleBuf[bc], scaleCopyParams);
                SetFlag<HardEvent::MTE3_V>(evOutFree[bc]); // 输出 slot 写出，可复用
            }

            // Epilogue: drain 所有未消费信号。配对核算：每 slot evXFree/evOutFree 各恰余 1 个 Set。
            WaitFlag<HardEvent::V_MTE2>(evXFree[0]);
            WaitFlag<HardEvent::V_MTE2>(evXFree[1]);
            WaitFlag<HardEvent::MTE3_V>(evOutFree[0]);
            WaitFlag<HardEvent::MTE3_V>(evOutFree[1]);
        }

        pairCursor += batchSize;
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginDynamicQuant<T>::Process()
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
#endif // MOE_V3_COUNTING_SORT_CUT_ORIGIN_DYNAMIC_QUANT_H
