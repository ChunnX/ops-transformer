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
 * \file moe_v3_counting_sort_full_load_dynamic_quant.h
 * \brief A5 CountingSort FullLoad 动态量化(int8)子类。
 *        per-token absmax scale：可选 per-expert smooth((LE,H))；q = round(x/scale) -> int8，
 *        per-token scale 透传到 expandedScale。pass2 与内置 ComputeMultiPass 对齐（smooth 预乘写回、孤立 Div）。
 */
#ifndef MOE_V3_COUNTING_SORT_FULL_LOAD_DYNAMIC_QUANT_H
#define MOE_V3_COUNTING_SORT_FULL_LOAD_DYNAMIC_QUANT_H

#include "moe_v3_counting_sort_full_load_base.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3CountingSortFullLoadDynamicQuant : public MoeV3CountingSortFullLoadBase<T> {
public:
    __aicore__ inline MoeV3CountingSortFullLoadDynamicQuant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX,
                                GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void QuantDynamicOneRow(int64_t tokenRow, int64_t expertOffset, float &outScale);
    __aicore__ inline void GatherAndWrite();
    __aicore__ inline void GatherOneRow(int64_t newPos, int64_t tokenRow, int64_t origFlatIdx, int64_t expertOffset);

    GlobalTensor<T> xGm_;
    GlobalTensor<int8_t> expandedXInt8Gm_;
    GlobalTensor<float> scaleGm_;         // per-expert smooth (LE,H)，仅 isInputScale_
    GlobalTensor<float> expandedScaleGm_; // per-token scale 输出

    int64_t quantI8LocalOffset_;   // 单行 int8 输出
    int64_t quantScaleSlotOffset_; // 动态量化 scale slot
    int64_t smoothLocalOffset_;    // per-expert smooth 向量单行缓冲（colsAligned floats，仅 isInputScale_）

    constexpr static MicroAPI::CastTrait castTraitF32ToF16Dyn = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    constexpr static MicroAPI::CastTrait castTraitF16ToI8Dyn = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
};

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadDynamicQuant<T>::Init(
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
    int64_t totalBufSize = offsetBytes;

    // 动态量化无 dropPad；GATHER 预填 rowIdx=-1
    this->PrefillAndSync();
    if (this->rowIdxType_ == GATHER) {
        SyncAll();
    }

    this->pipe_->InitBuffer(this->buf_, totalBufSize);
}

// 动态量化(int8)：可选 per-expert smooth（x*=smooth[expert]）-> per-token absmax -> scale=absmax/127；
// q=round(x/scale)->int8。写 quantI8Local + outScale。
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadDynamicQuant<T>::QuantDynamicOneRow(int64_t tokenRow,
                                                                                    int64_t expertOffset,
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

    __local_mem__ T *inUbAddr = (__local_mem__ T *)xLocal[tokenRow * this->colsAligned_].GetPhyAddr();
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
__aicore__ inline void MoeV3CountingSortFullLoadDynamicQuant<T>::GatherOneRow(int64_t newPos, int64_t tokenRow,
                                                                              int64_t origFlatIdx, int64_t expertOffset)
{
    DataCopyExtParams xI8CopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->cols_ * sizeof(int8_t)), 0, 0,
                                    0};
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
        LocalTensor<int8_t> quantI8Local = this->buf_.template Get<int8_t>()[quantI8LocalOffset_];
        float tokenScale = 0.0f;
        QuantDynamicOneRow(tokenRow, expertOffset, tokenScale);
        DataCopyPad(expandedXInt8Gm_[static_cast<int64_t>(newPos) * this->cols_], quantI8Local, xI8CopyParams);
        // 写 per-token scale（与上面 int8 写出同为 MTE3，顺序发射，无需额外同步）
        LocalTensor<float> scaleSlot = this->buf_.template Get<float>()[quantScaleSlotOffset_ / sizeof(float)];
        DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(expandedScaleGm_[newPos], scaleSlot, scaleCopyParams);
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadDynamicQuant<T>::GatherAndWrite()
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

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadDynamicQuant<T>::Process()
{
    if (this->blockIdx_ >= this->filterNeedCoreNum_) {
        SyncAll();
        return;
    }

    this->LoadExpertIdx();
    this->LoadXBackground(xGm_);
    this->FilterAndCount();
    this->WriteExpertCountToWorkspace();

    SyncAll();

    this->ComputeGlobalOffset();
    if (this->expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        this->WriteExpertTokens();
    }

    this->WaitXLoadCommon();
    // 量化模式下 xLocal 经向量流水读取，需 MTE2->V 同步
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    GatherAndWrite();
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_FULL_LOAD_DYNAMIC_QUANT_H
