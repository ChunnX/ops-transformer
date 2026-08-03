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

    GlobalTensor<T> xGm_;
    GlobalTensor<int8_t> expandedXInt8Gm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;
    float scale_ = 1.0f;
    float offset_ = 0.0f;

    int64_t quantI8LocalOffset_;

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
    GatherAndWrite();
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_CUT_ORIGIN_STATIC_QUANT_H
