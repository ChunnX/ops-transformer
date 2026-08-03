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
 * \brief A5 CountingSort FullLoad 模板基类（模板1）。仅承载公共逻辑：
 *        Phase A 索引全载 + 向量过滤计数、Phase B 核间计数归约 + 全局偏移，
 *        以及 x 后台加载/等待、dropPad 空槽清零、公共 UB 布局。
 *        量化/搬运（GatherAndWrite/GatherOneRow/Quant*OneRow）由子类实现，避免运行时量化分支。
 *
 *        PERF TODO: FilterAndCountVector 的计数段（CompareScalar + And + 标量累加）
 *        后续可替换为 SIMD VF Histograms API。
 */
#ifndef MOE_V3_COUNTING_SORT_FULL_LOAD_BASE_H
#define MOE_V3_COUNTING_SORT_FULL_LOAD_BASE_H

// FullLoad Histograms 计数 A/B（2026-08-08，device 6）：Phase A FilterAndCountVector 的标量 count 循环
// 替换为 SIMD VF Histograms API。与 CutOrigin 的 MOE_CS_USE_HISTOGRAM 独立（两个模板按 TilingKey 分流，互不污染）。
// 实测 10 条（cs_coverage.csv ti=0-9）：平均 Δ%≈+2.3%（一致劣化），✅0/⚪1/❌9，详见 cs_vs_builtin_perf.md。
// 根因：FullLoad cols=128 小列、baseline 耗时短（~65-76µs），Histograms 固定开销 ~1-3µs 在小基数上占比高。
// 宏默认关闭，代码保留供未来窄 active_expert_range / 极端倾斜场景复验。方案见 docs/fullload_histogram_plan.md。
// #define MOE_CS_FULLLOAD_USE_HISTOGRAM

#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

#ifdef MOE_CS_FULLLOAD_USE_HISTOGRAM
// FullLoad Histograms 计数 SIMD VF（复制自 cut_origin_base.h MoeV3CutOriginHistogramVF，独立维护避免交叉影响）。
// 源 uint8（值域 [0,256)），目的 uint16（256 bins：BIN0=[0,127] / BIN1=[128,255]）。
__simd_vf__ inline void MoeV3FullLoadHistogramVF(__ubuf__ uint8_t *srcAddr, __ubuf__ uint16_t *dstAddr,
                                                 uint32_t validCount)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t U8_REPEAT = 256; // 一个 VL(256B) 处理 256 个 uint8
    RegTensor<uint8_t> srcReg;
    RegTensor<uint16_t> dstReg0, dstReg1;
    Duplicate(dstReg0, static_cast<uint16_t>(0));
    Duplicate(dstReg1, static_cast<uint16_t>(0));
    uint16_t repeatTimes = static_cast<uint16_t>((validCount + U8_REPEAT - 1) / U8_REPEAT);
    uint32_t remain = validCount;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        MaskReg mask = UpdateMask<uint8_t>(remain);
        LoadAlign(srcReg, srcAddr + U8_REPEAT * i);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::FREQUENCY>(dstReg0, srcReg, mask);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::FREQUENCY>(dstReg1, srcReg, mask);
    }
    MaskReg maskAll = CreateMask<uint16_t, MaskPattern::ALL>();
    StoreAlign(dstAddr, dstReg0, maskAll);
    StoreAlign(dstAddr + 128, dstReg1, maskAll);
}
#endif // MOE_CS_FULLLOAD_USE_HISTOGRAM

template <typename T>
class MoeV3CountingSortFullLoadBase {
public:
    __aicore__ inline MoeV3CountingSortFullLoadBase(){};

protected:
    // 公共 Init 骨架：解析 tiling、切分核间 token、设置公共 GM、GATHER 预填 + SyncAll。
    // x/scale/offset/expandedX/expandedScale GM、量化 UB 区、InitBuffer 由子类补齐。
    __aicore__ inline void InitCommon(GM_ADDR expertIdx, GM_ADDR expandedRowIdx,
                                      GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                      const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void PrefillAndSync();
    __aicore__ inline void ZeroOutExpandedDropPad(GlobalTensor<T> &expandedXGm);
    __aicore__ inline void ComputeCommonUbLayout();
    __aicore__ inline void LoadExpertIdx();
    __aicore__ inline void LoadXBackground(GlobalTensor<T> &xGm);
    __aicore__ inline void FilterAndCount();
    __aicore__ inline void FilterAndCountVector();
#ifdef MOE_CS_FULLLOAD_USE_HISTOGRAM
    __aicore__ inline void FilterAndCountHistogram();
#endif
    __aicore__ inline void WriteExpertCountToWorkspace();
    __aicore__ inline void ComputeGlobalOffset();
    __aicore__ inline void WriteExpertTokens();
    __aicore__ inline void WaitXLoadCommon();

protected:
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;

    TPipe *pipe_;
    TBuf<TPosition::VECCALC> buf_;

    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> workspaceGm_;
    GlobalTensor<int64_t> expertTokensGm_;

    int64_t quantMode_;

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
    int64_t filterPerCoreTokens_;

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

    // 公共 UB region offsets (byte offsets from buf_ start)
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
#ifdef MOE_CS_FULLLOAD_USE_HISTOGRAM
    int64_t expertU8LocalOffset_;   // uint8 源（filteredInBatch 个 expertOffset，值域 [0,256)）
    int64_t histBinOffset_;         // uint16 bins（256 个 = BIN0 128 + BIN1 128）
#endif
    int64_t commonBufSize_;         // 公共区结尾 offset；子类在其后追加量化区
};

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::InitCommon(
    GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
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

    filterPerCoreTokens_ = tiling->countingSortParamsOp.filterPerCoreTokens;
    coreTokenStart_ = blockIdx_ * filterPerCoreTokens_;
    coreTokenEnd_ = Min(coreTokenStart_ + filterPerCoreTokens_, n_);
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

    // 公共 GM
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx);
    workspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace);
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        expertTokensGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokensCountOrCumsum);
    }

    ComputeCommonUbLayout();
}

// GATHER 模式预填 rowIdx=-1 + dropPad 清零后 SyncAll。子类在设置好 expandedX/expandedScale GM 后调用。
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::PrefillAndSync()
{
    if (rowIdxType_ == GATHER) {
        if (blockIdx_ < filterNeedCoreNum_) {
            GlobalTensor<int32_t> expandedRowIdxGmTmp = expandedRowIdxGm_[filterPerCoreTokens_ * k_ * blockIdx_];
            InitGlobalMemory(expandedRowIdxGmTmp, coreEntries_, static_cast<int32_t>(-1));
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
    }
}

// dropPad 空槽零填充：跨核均分 outputRows_，每核清零自己负责的 expandedX 行段。
// dropPad 为非量化专属（算子契约），走 T 型 expandedXGm。scale 透传空槽清零由子类传入。
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::ZeroOutExpandedDropPad(GlobalTensor<T> &expandedXGm)
{
    int64_t perCoreRows = Ceil(outputRows_, filterNeedCoreNum_);
    int64_t rowStart = blockIdx_ * perCoreRows;
    int64_t rowEnd = Min(rowStart + perCoreRows, outputRows_);
    if (rowStart >= rowEnd) {
        return;
    }
    int64_t elemCnt = (rowEnd - rowStart) * cols_;
    GlobalTensor<T> zeroSeg = expandedXGm[rowStart * cols_];
    InitGlobalMemory(zeroSeg, elemCnt, static_cast<T>(0));
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::ComputeCommonUbLayout()
{
    int64_t offset = 0;
    xLocalOffset_ = offset;
    offset += AlignBytes(coreTokenNum_ * colsAligned_, static_cast<int64_t>(sizeof(T)));

    expertIdxLocalOffset_ = offset;
    offset += AlignBytes(coreEntries_, static_cast<int64_t>(sizeof(int32_t)));

    scaleLocalOffset_ = offset;
    // per-token scale 全载缓冲仅非量化透传路径需要；动态量化的 (LE,H) smooth 走单行缓冲（子类）
    if (isInputScale_ && quantMode_ != CS_QUANT_MODE_DYNAMIC) {
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

#ifdef MOE_CS_FULLLOAD_USE_HISTOGRAM
    // Histograms 专属：uint8 源（entriesAligned 对齐 256）+ uint16 bins（256 个）。
    expertU8LocalOffset_ = offset;
    offset += AlignBytes(Ceil(entriesAligned_, static_cast<int64_t>(256)) * 256,
                         static_cast<int64_t>(sizeof(uint8_t)));
    histBinOffset_ = offset;
    offset += AlignBytes(256, static_cast<int64_t>(sizeof(uint16_t)));
#endif

    // 公共区结尾；子类量化区在其后追加
    commonBufSize_ = offset;
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::LoadExpertIdx()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(coreEntries_ * sizeof(int32_t)), 0, 0,
                                 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(expertIdxLocal, expertIdxGm_[coreFlatStart_], copyParams, padParams);
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::LoadXBackground(GlobalTensor<T> &xGm)
{
    LocalTensor<T> xLocal = buf_.Get<T>()[xLocalOffset_ / sizeof(T)];

    SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
    // dstStride 单位是 32B datablock（非字节）。xLocal 每行按 colsAligned_ 元素排布，
    // 而 colsAligned_*sizeof(T) == CeilAlign(cols_*sizeof(T),32)，DataCopyPad 每 block 自动 pad 到 32B，
    // 各行本就紧邻，故 dstStride 恒为 0。
    DataCopyExtParams copyParams{static_cast<uint16_t>(coreTokenNum_), static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0,
                                 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[coreTokenStart_ * cols_], copyParams, padParams);
    // 不在这里 wait，x 后台加载，在 WaitXLoadCommon() 中同步
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::FilterAndCount()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

#ifdef MOE_CS_FULLLOAD_USE_HISTOGRAM
    FilterAndCountHistogram();
#else
    FilterAndCountVector();
#endif
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::FilterAndCountVector()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

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

#ifdef MOE_CS_FULLLOAD_USE_HISTOGRAM
// FullLoad Histograms 计数路径：向量过滤(复用 FilterAndCountVector 前段) -> Adds(-expertStart) 窄化 uint8
// -> Histograms 计数 -> 回写 expertCountLocal(int32)；pairs 在压缩结果上标量填充（不再做 count 累加）。
// 注意：expertCountLocal 清零由 FilterAndCount() 入口完成，此处不重复。计数结果须与旧标量路径逐 bin 一致。
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::FilterAndCountHistogram()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    // 向量过滤：Cast int32->fp32 -> CompareScalar(GE/LT) -> And -> GatherMask 压缩合法专家 + 原始下标 j
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

    // 窄化：gatheredExpert - expertStart_ -> [0, actualExpertNum_) -> uint8。源已剔除越界专家，无污染。
    Adds(gatheredExpertLocal, gatheredExpertLocal, static_cast<int32_t>(-expertStart_),
         static_cast<int32_t>(filteredInBatch));
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> expertU8Local = buf_.Get<uint8_t>()[expertU8LocalOffset_];
    // int32 -> int16 -> uint8 两步窄化（值域已保证 [0,256)）
    Cast(expertU8Local.ReinterpretCast<int16_t>(), gatheredExpertLocal, RoundMode::CAST_NONE,
         static_cast<int32_t>(filteredInBatch));
    PipeBarrier<PIPE_V>();
    Cast(expertU8Local, expertU8Local.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE,
         static_cast<int32_t>(filteredInBatch));
    PipeBarrier<PIPE_V>();

    // Histograms 计数 -> uint16 bins
    LocalTensor<uint16_t> histBinLocal = buf_.Get<uint16_t>()[histBinOffset_ / sizeof(uint16_t)];
    MoeV3FullLoadHistogramVF((__ubuf__ uint8_t *)expertU8Local.GetPhyAddr(),
                             (__ubuf__ uint16_t *)histBinLocal.GetPhyAddr(),
                             static_cast<uint32_t>(filteredInBatch));
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // uint16 bins -> int32 expertCountLocal（actualExpertNum_ <= 256）
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        expertCountLocal.SetValue(e, static_cast<int32_t>(histBinLocal.GetValue(e)));
    }

    // pairs 构建：在压缩结果上标量填充（不再做 count 累加），expertOffset 已减 expertStart_
    filteredCount_ = 0;
    for (int64_t f = 0; f < filteredInBatch; f++) {
        int32_t j = gatheredIdxLocal.GetValue(f);
        int32_t expertOffset = gatheredExpertLocal.GetValue(f);
        filteredPairsLocal.SetValue(filteredCount_ * 2, j);
        filteredPairsLocal.SetValue(filteredCount_ * 2 + 1, expertOffset);
        filteredCount_++;
    }
}
#endif // MOE_CS_FULLLOAD_USE_HISTOGRAM

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
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::WaitXLoadCommon()
{
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_FULL_LOAD_BASE_H
