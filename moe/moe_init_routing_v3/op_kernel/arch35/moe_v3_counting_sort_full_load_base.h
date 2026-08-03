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
 * \file moe_v3_counting_sort_full_load_base.h
 * \brief A5 CountingSort FullLoad 模板基类（模板1）。
 *        借鉴 A3 的 MoeV3FullLoadCutOriginT，多核切 T 后核内 UB 全载。
 *        UB 使用模型：叠加（filter + 计数 + 搬运同时驻留）。
 *
 *        PERF TODO: 后续可将 FilterAndCountVector 的计数段（CompareScalar + And + 标量累加）
 *        替换为 SIMD VF Histograms API（AscendC::Reg::Histograms<uint8_t, uint16_t, BIN0, FREQUENCY>），
 *        需要寄存器级 API（RegTensor/LoadAlign/MaskReg），与 LocalTensor 风格代码混合需谨慎。
 */
#ifndef MOE_V3_COUNTING_SORT_FULL_LOAD_BASE_H
#define MOE_V3_COUNTING_SORT_FULL_LOAD_BASE_H

#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3CountingSortFullLoadBase {
public:
    __aicore__ inline MoeV3CountingSortFullLoadBase(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX,
                                GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitGlobalBuffers(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
                                             GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                             GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                             GM_ADDR workspace);
    __aicore__ inline void ZeroOutExpandedDropPad();
    __aicore__ inline void ComputeUbLayout();
    __aicore__ inline void LoadExpertIdxAndScale();
    __aicore__ inline void LoadXBackground();
    __aicore__ inline void FilterAndCount();
    __aicore__ inline void FilterAndCountVector();
    __aicore__ inline void WriteExpertCountToWorkspace();
    __aicore__ inline void ComputeGlobalOffset();
    __aicore__ inline void WriteExpertTokens();
    __aicore__ inline void WaitXLoad();
    __aicore__ inline void GatherAndWrite();
    __aicore__ inline void GatherOneRow(int64_t newPos, int64_t tokenRow, int64_t origFlatIdx, int64_t expertOffset);
    // 量化：将 xLocal 的 tokenRow 行量化为 int8 写入 quantI8Local（buf_ 内 quantI8LocalOffset_）
    __aicore__ inline void QuantStaticOneRow(int64_t tokenRow);
    __aicore__ inline void QuantDynamicOneRow(int64_t tokenRow, float &outScale);

protected:
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;

    TPipe *pipe_;
    TBuf<TPosition::VECCALC> buf_;

    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<int8_t> expandedXInt8Gm_; // 静态/动态量化时 expandedX 重解释为 int8
    GlobalTensor<float> offsetGm_;         // 静态量化 per-tensor offset
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int64_t> expertTokensGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<int32_t> workspaceGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    int64_t quantMode_;
    float scale_ = 1.0f;  // 静态量化标量
    float offset_ = 0.0f; // 静态量化标量

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
    int64_t entriesAligned_;
    int64_t maskBytes_;
    int64_t expertTokensBufElems_;  // expertTokensLocal 缓冲元素数（KEY_VALUE 需 2 倍）
    int64_t expertCountElements_;   // 实际写出 expertTokens 的元素数

    // UB region offsets (byte offsets from buf_ start)
    int64_t xLocalOffset_;
    int64_t expertIdxLocalOffset_;
    int64_t scaleLocalOffset_;
    int64_t expertCountLocalOffset_;
    int64_t allCoreExpertCountLocalOffset_;
    int64_t expertTokensLocalOffset_;
    int64_t prefixSumLocalOffset_;
    int64_t filteredPairsLocalOffset_;
    int64_t expertIdxFp32LocalOffset_;
    int64_t compareMask0Offset_;
    int64_t compareMask1Offset_;
    int64_t gatherMaskOffset_;
    int64_t gatheredExpertLocalOffset_;
    int64_t flatIdxBufferLocalOffset_;
    int64_t gatheredIdxLocalOffset_;
    // 量化临时区（quantMode_ != -1 时启用）
    int64_t quantI8LocalOffset_;    // 单行 int8 输出
    int64_t quantScaleSlotOffset_;  // 动态量化 scale slot
    int64_t totalBufSize_;

    constexpr static MicroAPI::CastTrait castTraitF32ToF16Static = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static MicroAPI::CastTrait castTraitF16ToI8Static = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16Dyn = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    constexpr static MicroAPI::CastTrait castTraitF16ToI8Dyn = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
};

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::Init(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
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
    quantMode_ = tiling->quantMode;
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
    scaleSlotSize_ = BLOCK_BYTES / static_cast<int64_t>(sizeof(float)); // 8
    maxFilteredCount_ = coreEntries_;
    filteredCount_ = 0;

    // expertTokens 缓冲：KEY_VALUE 输出 [expertId,count] 键值对，需 actualExpertNum*2 元素；COUNT/CUMSUM 为 actualExpertNum。
    expertTokensBufElems_ = (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) ? actualExpertNum_ * EXPERT_ID_VALUE_NUM
                                                                              : actualExpertNum_;
    expertCountElements_ = expertTokensBufElems_;

    entriesAligned_ = Ceil(coreEntries_, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;
    maskBytes_ = AlignBytes(Ceil(entriesAligned_, static_cast<int64_t>(8)), static_cast<int64_t>(1));

    InitGlobalBuffers(x, expertIdx, scale, offset, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expandedScale,
                      workspace);

    // GATHER 模式预填 -1；dropPad 模式还需整块清零 expandedX（空槽零填充）
    if (rowIdxType_ == GATHER) {
        if (blockIdx_ < filterNeedCoreNum_) {
            GlobalTensor<int32_t> expandedRowIdxGmTmp = expandedRowIdxGm_[filterPerCoreTokens * k_ * blockIdx_];
            InitGlobalMemory(expandedRowIdxGmTmp, coreEntries_, static_cast<int32_t>(-1));
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
        if (dropPadMode_ == DROP_PAD_MODE) {
            ZeroOutExpandedDropPad();
        }
        SyncAll();
    }

    ComputeUbLayout();
    pipe_->InitBuffer(buf_, totalBufSize_);
}

// dropPad 空槽零填充：跨核均分 outputRows_，每核清零自己负责的 expandedX 行段。
// dropPad 为非量化专属（算子契约），走 T 型 expandedXGm_。
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::ZeroOutExpandedDropPad()
{
    int64_t perCoreRows = Ceil(outputRows_, filterNeedCoreNum_);
    int64_t rowStart = blockIdx_ * perCoreRows;
    int64_t rowEnd = Min(rowStart + perCoreRows, outputRows_);
    if (rowStart >= rowEnd) {
        return;
    }
    int64_t elemCnt = (rowEnd - rowStart) * cols_;
    GlobalTensor<T> zeroSeg = expandedXGm_[rowStart * cols_];
    InitGlobalMemory(zeroSeg, elemCnt, static_cast<T>(0));
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

    // scale 透传（isInputScale_）时，expandedScale 的空槽也需清零（golden 语义：空槽为 0）
    if (isInputScale_) {
        GlobalTensor<float> scaleZeroSeg = expandedScaleGm_[rowStart];
        InitGlobalMemory(scaleZeroSeg, rowEnd - rowStart, static_cast<float>(0));
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::InitGlobalBuffers(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace)
{
    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx);
    // 非量化：expandedX 为 T；静态/动态量化：expandedX 重解释为 int8
    if (quantMode_ == CS_QUANT_MODE_UNQUANT) {
        expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX);
    } else {
        expandedXInt8Gm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);
    }
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx);
    workspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace);
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        expertTokensGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokensCountOrCumsum);
    }

    if (quantMode_ == CS_QUANT_MODE_STATIC) {
        // 静态量化：per-tensor 标量 scale/offset
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale, 1);
        offsetGm_.SetGlobalBuffer((__gm__ float *)offset, 1);
        scale_ = scaleGm_.GetValue(0);
        offset_ = offsetGm_.GetValue(0);
    } else if (quantMode_ == CS_QUANT_MODE_DYNAMIC) {
        // 动态量化：写 per-token expandedScale；smooth 可选（isInputScale_）
        expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);
        if (isInputScale_) {
            scaleGm_.SetGlobalBuffer((__gm__ float *)scale);
        }
    } else if (isInputScale_) {
        // 非量化：scale 透传路径
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale);
        expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::ComputeUbLayout()
{
    int64_t offset = 0;
    xLocalOffset_ = offset;
    offset += AlignBytes(coreTokenNum_ * colsAligned_, static_cast<int64_t>(sizeof(T)));

    expertIdxLocalOffset_ = offset;
    offset += AlignBytes(coreEntries_, static_cast<int64_t>(sizeof(int32_t)));

    scaleLocalOffset_ = offset;
    if (isInputScale_) {
        offset += AlignBytes(coreTokenNum_ * scaleSlotSize_, static_cast<int64_t>(sizeof(float)));
    }

    expertCountLocalOffset_ = offset;
    offset += AlignBytes(expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    allCoreExpertCountLocalOffset_ = offset;
    offset += AlignBytes(filterNeedCoreNum_ * expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    expertTokensLocalOffset_ = offset;
    offset += AlignBytes(expertTokensBufElems_, static_cast<int64_t>(sizeof(int64_t)));

    prefixSumLocalOffset_ = offset;
    offset += AlignBytes(expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    filteredPairsLocalOffset_ = offset;
    offset += AlignBytes(maxFilteredCount_ * 2, static_cast<int64_t>(sizeof(int32_t)));

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

    // 量化临时区：单行 int8 输出 + 动态量化 scale slot
    quantI8LocalOffset_ = offset;
    quantScaleSlotOffset_ = offset;
    if (quantMode_ != CS_QUANT_MODE_UNQUANT) {
        offset += AlignBytes(colsAligned_, static_cast<int64_t>(sizeof(int8_t)));
        quantScaleSlotOffset_ = offset;
        offset += AlignBytes(scaleSlotSize_, static_cast<int64_t>(sizeof(float)));
    }

    totalBufSize_ = offset;
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::Process()
{
    if (blockIdx_ >= filterNeedCoreNum_) {
        SyncAll();
        return;
    }

    // Phase A: Load + Filter + Count (核间并行)
    LoadExpertIdxAndScale();
    LoadXBackground();
    FilterAndCount();
    WriteExpertCountToWorkspace();

    SyncAll();

    // Phase B: 核间归约 + 偏移计算
    ComputeGlobalOffset();
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        WriteExpertTokens();
    }

    // Phase C: 等待 x 加载 + 数据搬运
    WaitXLoad();
    GatherAndWrite();
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::LoadExpertIdxAndScale()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(coreEntries_ * sizeof(int32_t)), 0, 0,
                                 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(expertIdxLocal, expertIdxGm_[coreFlatStart_], copyParams, padParams);

    if (isInputScale_) {
        LocalTensor<float> scaleLocal = buf_.Get<float>()[scaleLocalOffset_ / sizeof(float)];
        DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(coreTokenNum_), static_cast<uint32_t>(sizeof(float)), 0,
                                          0, 0};
        DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0};
        DataCopyPad(scaleLocal, scaleGm_[coreTokenStart_], scaleCopyParams, scalePadParams);
    }

    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::LoadXBackground()
{
    LocalTensor<T> xLocal = buf_.Get<T>()[xLocalOffset_ / sizeof(T)];

    SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
    // dstStride 单位是 32B datablock（非字节）。xLocal 每行按 colsAligned_ 元素排布，
    // 而 colsAligned_*sizeof(T) == CeilAlign(cols_*sizeof(T),32)，DataCopyPad 每 block 自动 pad 到 32B，
    // 各行本就紧邻，故 dstStride 恒为 0。此前误用 (colsAligned_-cols_)*sizeof(T)（字节值），
    // 在 cols 未 32B 对齐（如 cols=1）时会被当作 block 数导致 DDR 地址越界。
    DataCopyExtParams copyParams{static_cast<uint16_t>(coreTokenNum_), static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0,
                                 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[coreTokenStart_ * cols_], copyParams, padParams);
    // 不在这里 wait，x 后台加载，在 WaitXLoad() 中同步
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::FilterAndCount()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    FilterAndCountVector();
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::FilterAndCountVector()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    // PERF TODO: 计数段可替换为 AscendC::Reg::Histograms<uint8_t, uint16_t, BIN0, FREQUENCY>，
    // 需 Cast int32->uint8（按 expertStart 偏置）+ LoadAlign 循环 + StoreAlign。
    // 当前保留 A3 SIMT 风格（CompareScalar + And + GatherMask）作为稳定基线。

    // Cast int32 -> fp32 用于 CompareScalar
    LocalTensor<float> expertIdxFp32Local = buf_.Get<float>()[expertIdxFp32LocalOffset_ / sizeof(float)];
    Cast(expertIdxFp32Local, expertIdxLocal, RoundMode::CAST_ROUND, entriesAligned_);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> compareMask0 = buf_.Get<uint8_t>()[compareMask0Offset_];
    LocalTensor<uint8_t> compareMask1 = buf_.Get<uint8_t>()[compareMask1Offset_];
    LocalTensor<uint8_t> gatherMaskLocal = buf_.Get<uint8_t>()[gatherMaskOffset_];

    CompareScalar(compareMask0, expertIdxFp32Local, static_cast<float>(expertStart_), CMPMODE::GE, entriesAligned_);
    PipeBarrier<PIPE_V>();
    CompareScalar(compareMask1, expertIdxFp32Local, static_cast<float>(expertEnd_), CMPMODE::LT, entriesAligned_);
    PipeBarrier<PIPE_V>();
    And(gatherMaskLocal.ReinterpretCast<uint16_t>(), compareMask0.ReinterpretCast<uint16_t>(),
        compareMask1.ReinterpretCast<uint16_t>(),
        Ceil(entriesAligned_, MASK_STRIDE) * MASK_STRIDE / DST_REP_STRIDE / 2);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> gatheredExpertLocal = buf_.Get<int32_t>()[gatheredExpertLocalOffset_ / sizeof(int32_t)];
    uint64_t rsvdCnt = 0;
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 1;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = DST_REP_STRIDE;
    gatherMaskParams.src1RepeatStride = DST_REP_STRIDE;
    GatherMask(gatheredExpertLocal, expertIdxLocal, gatherMaskLocal.ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(coreEntries_), gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();
    int64_t filteredInBatch = static_cast<int64_t>(rsvdCnt);

    LocalTensor<int32_t> flatIdxBufferLocal = buf_.Get<int32_t>()[flatIdxBufferLocalOffset_ / sizeof(int32_t)];
    ArithProgression<int32_t>(flatIdxBufferLocal, static_cast<int32_t>(0), 1, static_cast<int32_t>(coreEntries_));
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> gatheredIdxLocal = buf_.Get<int32_t>()[gatheredIdxLocalOffset_ / sizeof(int32_t)];
    uint64_t idxRsvdCnt = 0;
    GatherMask(gatheredIdxLocal, flatIdxBufferLocal, gatherMaskLocal.ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(coreEntries_), gatherMaskParams, idxRsvdCnt);
    PipeBarrier<PIPE_V>();

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // 标量：构建 pairs + 累加 expert count
    filteredCount_ = 0;
    for (int64_t j = 0; j < filteredInBatch; j++) {
        int32_t expertId = gatheredExpertLocal.GetValue(j);
        int32_t localFlatIdx = gatheredIdxLocal.GetValue(j);
        int32_t expertOffset = expertId - static_cast<int32_t>(expertStart_);

        int32_t curCount = expertCountLocal.GetValue(expertOffset);
        expertCountLocal.SetValue(expertOffset, curCount + 1);

        filteredPairsLocal.SetValue(filteredCount_ * 2, localFlatIdx);
        filteredPairsLocal.SetValue(filteredCount_ * 2 + 1, expertOffset);
        filteredCount_++;
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::WriteExpertCountToWorkspace()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(expertCountStride_ * sizeof(int32_t)),
                                 0, 0, 0};
    DataCopyPad(workspaceGm_[blockIdx_ * expertCountStride_], expertCountLocal, copyParams);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::ComputeGlobalOffset()
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

    // 复用 prefixSumLocal 作为 totalCountLocal：向量化核间求和
    LocalTensor<int32_t> totalCountLocal = prefixSumLocal;
    Duplicate(totalCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    for (int64_t c = 0; c < filterNeedCoreNum_; c++) {
        Add(totalCountLocal, totalCountLocal, allCoreExpertCountLocal[c * expertCountStride_], expertCountStride_);
        PipeBarrier<PIPE_V>();
    }

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    int32_t totalForExpertArr[COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM]; // actualExpertNum_ <= COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        totalForExpertArr[e] = totalCountLocal.GetValue(e);
    }

    // 本核 prefix sum：核 [0, blockIdx_) 的计数累加
    Duplicate(prefixSumLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    for (int64_t c = 0; c < blockIdx_; c++) {
        Add(prefixSumLocal, prefixSumLocal, allCoreExpertCountLocal[c * expertCountStride_], expertCountStride_);
        PipeBarrier<PIPE_V>();
    }

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // 标量：本核在专家 e 全局起始位置
    LocalTensor<int64_t> expertTokensLocal = buf_.Get<int64_t>()[expertTokensLocalOffset_ / sizeof(int64_t)];
    int64_t cumulativeSum = 0;
    int64_t keyValueOffset = 0; // KEY_VALUE 模式下的紧凑写出下标（跳过 count==0 的专家）
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        int32_t totalForExpert = totalForExpertArr[e];
        int32_t prefixForExpert = prefixSumLocal.GetValue(e);

        // expertTokens 由 core 0 统一写出，支持 COUNT / CUMSUM / KEY_VALUE 三种模式
        if (blockIdx_ == 0 && expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
            if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
                // 仅对 count!=0 的专家写 [expertId(全局), count] 键值对
                if (totalForExpert != 0) {
                    expertTokensLocal.SetValue(keyValueOffset * EXPERT_ID_VALUE_NUM,
                                               static_cast<int64_t>(e + expertStart_));
                    expertTokensLocal.SetValue(keyValueOffset * EXPERT_ID_VALUE_NUM + 1,
                                               static_cast<int64_t>(totalForExpert));
                    keyValueOffset++;
                }
            } else if (expertTokensNumType_ == EXERPT_TOKENS_COUNT) {
                expertTokensLocal.SetValue(e, static_cast<int64_t>(totalForExpert));
            } else {
                // CUMSUM：写累计和
                expertTokensLocal.SetValue(e, cumulativeSum + static_cast<int64_t>(totalForExpert));
            }
        }

        // dropPad：seed 为该专家跨核 per-expert 前缀（本核在该专家的起始 slot）；
        // dropless：seed 为全局跨专家 cumulative + per-expert 前缀。
        if (dropPadMode_ == DROP_PAD_MODE) {
            expertCountLocal.SetValue(e, prefixForExpert);
        } else {
            expertCountLocal.SetValue(e, static_cast<int32_t>(cumulativeSum) + prefixForExpert);
        }
        cumulativeSum += totalForExpert;
    }

    // KEY_VALUE 模式实际写出元素数 = 非零专家数 * 2；上限为 actualExpertNum*2。
    if (blockIdx_ == 0 && expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
        expertCountElements_ = keyValueOffset * EXPERT_ID_VALUE_NUM;
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::WriteExpertTokens()
{
    if (blockIdx_ != 0) {
        return;
    }
    LocalTensor<int64_t> expertTokensLocal = buf_.Get<int64_t>()[expertTokensLocalOffset_ / sizeof(int64_t)];
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    // COUNT/CUMSUM: expertCountElements_==actualExpertNum_；KEY_VALUE: 非零专家数*2
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(expertCountElements_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::WaitXLoad()
{
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
    // 量化模式下 xLocal 经向量流水读取，需 MTE2->V 同步
    if (quantMode_ != CS_QUANT_MODE_UNQUANT) {
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }
}

// 静态量化：读 xLocal 的 tokenRow 行，Cast->f16、Muls(scale_)、Adds(offset_)、Cast->int8，写 quantI8Local。
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::QuantStaticOneRow(int64_t tokenRow)
{
    LocalTensor<T> xLocal = buf_.Get<T>()[xLocalOffset_ / sizeof(T)];
    LocalTensor<int8_t> quantI8Local = buf_.Get<int8_t>()[quantI8LocalOffset_];

    // quantI8Local 是单行复用缓冲：等待上一行的 DataCopyPad(MTE3) 读完，再让 V 覆写，避免跨行 race
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

    __local_mem__ T *inUbAddr = (__local_mem__ T *)xLocal[tokenRow * colsAligned_].GetPhyAddr();
    __local_mem__ int8_t *outUbAddr = (__local_mem__ int8_t *)quantI8Local.GetPhyAddr();

    uint16_t repeatTimes = Ceil(cols_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg = static_cast<uint32_t>(cols_);
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

// 动态量化(int8)：per-token absmax -> scale=absmax/127；q=round(x/scale)->int8。写 quantI8Local + outScale。
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::QuantDynamicOneRow(int64_t tokenRow, float &outScale)
{
    LocalTensor<T> xLocal = buf_.Get<T>()[xLocalOffset_ / sizeof(T)];
    LocalTensor<int8_t> quantI8Local = buf_.Get<int8_t>()[quantI8LocalOffset_];
    LocalTensor<float> scaleSlot = buf_.Get<float>()[quantScaleSlotOffset_ / sizeof(float)];

    // quantI8Local/scaleSlot 单行复用：等待上一行 DataCopyPad(MTE3) 读完再让 V 覆写，避免跨行 race
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

    __local_mem__ T *inUbAddr = (__local_mem__ T *)xLocal[tokenRow * colsAligned_].GetPhyAddr();
    __local_mem__ int8_t *outUbAddr = (__local_mem__ int8_t *)quantI8Local.GetPhyAddr();
    __local_mem__ float *scaleUbAddr = (__local_mem__ float *)scaleSlot.GetPhyAddr();

    uint16_t repeatTimes = Ceil(cols_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t maskLength = static_cast<uint32_t>(cols_);
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, absReg, scaleValueReg, invScaleReg;
        MicroAPI::Duplicate(scaleValueReg, 0.0f);
        MicroAPI::RegTensor<half> outRegF16;
        MicroAPI::RegTensor<int8_t> outRegI8;
        MicroAPI::MaskReg maskRegInLoop;
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegVL1 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

        // pass1: absmax
        sreg = maskLength;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddr, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::Abs(absReg, inReg, maskRegInLoop);
            MicroAPI::Max(scaleValueReg, scaleValueReg, absReg, maskRegAll);
        }
        MicroAPI::ReduceMax(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / CS_DYNAMIC_QUANT_INT8_SCALE, maskRegVL1);
        MicroAPI::Duplicate(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::DataCopy(scaleUbAddr, scaleValueReg, maskRegVL1);

        // pass2: q = round(x / scale) -> int8
        sreg = maskLength;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddr, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH);
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
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::GatherOneRow(int64_t newPos, int64_t tokenRow,
                                                                      int64_t origFlatIdx, int64_t expertOffset)
{
    LocalTensor<T> xLocal = buf_.Get<T>()[xLocalOffset_ / sizeof(T)];
    LocalTensor<float> scaleLocal = isInputScale_ ? buf_.Get<float>()[scaleLocalOffset_ / sizeof(float)]
                                                  : LocalTensor<float>();

    DataCopyExtParams xCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams xI8CopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(cols_ * sizeof(int8_t)), 0, 0, 0};
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    LocalTensor<int32_t> idxBuf = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    if (rowIdxType_ == GATHER) {
        idxBuf.SetValue(0, static_cast<int32_t>(newPos));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expandedRowIdxGm_[origFlatIdx], idxBuf, idxCopyParams);
    } else {
        idxBuf.SetValue(0, static_cast<int32_t>(origFlatIdx));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expandedRowIdxGm_[newPos], idxBuf, idxCopyParams);
    }

    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    if (newPos < outputRows_) {
        if (quantMode_ == CS_QUANT_MODE_STATIC) {
            LocalTensor<int8_t> quantI8Local = buf_.Get<int8_t>()[quantI8LocalOffset_];
            QuantStaticOneRow(tokenRow);
            DataCopyPad(expandedXInt8Gm_[static_cast<int64_t>(newPos) * cols_], quantI8Local, xI8CopyParams);
        } else if (quantMode_ == CS_QUANT_MODE_DYNAMIC) {
            LocalTensor<int8_t> quantI8Local = buf_.Get<int8_t>()[quantI8LocalOffset_];
            float tokenScale = 0.0f;
            QuantDynamicOneRow(tokenRow, tokenScale);
            DataCopyPad(expandedXInt8Gm_[static_cast<int64_t>(newPos) * cols_], quantI8Local, xI8CopyParams);
            // 写 per-token scale（与上面 int8 写出同为 MTE3，顺序发射，无需额外同步）
            LocalTensor<float> scaleSlot = buf_.Get<float>()[quantScaleSlotOffset_ / sizeof(float)];
            DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(expandedScaleGm_[newPos], scaleSlot, scaleCopyParams);
        } else {
            DataCopyPad(expandedXGm_[static_cast<int64_t>(newPos) * cols_], xLocal[tokenRow * colsAligned_],
                        xCopyParams);
            if (isInputScale_) {
                DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0,
                                                  0};
                DataCopyPad(expandedScaleGm_[newPos], scaleLocal[tokenRow * scaleSlotSize_], scaleCopyParams);
            }
        }
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::GatherAndWrite()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    for (int64_t fIdx = 0; fIdx < filteredCount_; fIdx++) {
        int32_t localFlatIdx = filteredPairsLocal.GetValue(fIdx * 2);
        int32_t expertOffset = filteredPairsLocal.GetValue(fIdx * 2 + 1);

        int32_t slot = expertCountLocal.GetValue(expertOffset);
        expertCountLocal.SetValue(expertOffset, slot + 1);

        int64_t tokenRow = static_cast<int64_t>(localFlatIdx) / k_;
        int64_t origFlatIdx = coreFlatStart_ + static_cast<int64_t>(localFlatIdx);

        int64_t newPos;
        if (dropPadMode_ == DROP_PAD_MODE) {
            // slot 为该专家的 per-expert 计数；溢出 capacity 则丢弃
            if (slot >= expertCapacity_) {
                // 丢弃 token：GATHER 模式显式写 rowIdx=-1（预填不可靠，此处强制写出）
                if (rowIdxType_ == GATHER) {
                    LocalTensor<int32_t> dropIdxBuf = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
                    DataCopyExtParams dropIdxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)),
                                                        0, 0, 0};
                    dropIdxBuf.SetValue(0, static_cast<int32_t>(-1));
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    DataCopyPad(expandedRowIdxGm_[origFlatIdx], dropIdxBuf, dropIdxCopyParams);
                    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                }
                continue;
            }
            newPos = (static_cast<int64_t>(expertOffset) + expertStart_) * expertCapacity_ + slot;
        } else {
            newPos = static_cast<int64_t>(slot);
        }

        GatherOneRow(newPos, tokenRow, origFlatIdx, static_cast<int64_t>(expertOffset));
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_FULL_LOAD_BASE_H
