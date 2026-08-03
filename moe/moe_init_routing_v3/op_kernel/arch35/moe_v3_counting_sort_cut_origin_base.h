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
 * \brief A5 CountingSort CutOrigin 模板基类（模板2）。仅承载公共逻辑：
 *        Phase A 索引全载 + 标量过滤计数、Phase B 核间计数归约 + 全局偏移，
 *        以及 dropPad 空槽清零、公共 UB 布局。
 *        量化/搬运（GatherAndWrite/GatherOneRow/Quant*OneRow）由子类实现，避免运行时量化分支。
 *        Phase C 单行流式搬运（xGm -> UB -> 量化/写出）也由子类完成。
 *
 *        PERF TODO: 计数段后续可替换为 AscendC::Reg::Histograms。
 */
#ifndef MOE_V3_COUNTING_SORT_CUT_ORIGIN_BASE_H
#define MOE_V3_COUNTING_SORT_CUT_ORIGIN_BASE_H

// Histograms 计数 A/B：v1(2026-08-05) + v2(2026-08-08) 两轮均为轻微劣化/持平（AER=[0,256] 全通过用例）。
// v2 在 Phase C 双缓冲落地后复测：CutOrigin 8 条平均 Δ%≈+0.3%，✅1/⚪7/❌0，详见 cs_vs_builtin_perf.md。
// 宏默认关闭，代码保留供未来窄 active_expert_range / 极端倾斜场景复验。
// #define MOE_CS_USE_HISTOGRAM

// A5 CutOrigin 追平 A3 性能优化：
//   Phase C 双缓冲 ping-pong 搬运（主因，隐藏大列整行搬运延迟）已作为默认实现固化（原 MOE_CS_CUTORIGIN_DOUBLE_BUFFER
//     宏移除）。大列超 UB 场景由运行时 csDbEnable_（tiling 判定）降级到串行 GatherAndWrite fallback，串行路径保留。
//   MOE_CS_CUTORIGIN_VECTOR_FILTER: Phase A 向量过滤替换纯标量单扫描（次因）。本测试集 active_range 全通过
//     （coreEntries_ 全合法），向量过滤压缩无淘汰 → 收益有限；窄 active_expert_range 场景收益更大，默认关保留可用。
// #define MOE_CS_CUTORIGIN_VECTOR_FILTER

#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3CountingSortCutOriginBase {
public:
    __aicore__ inline MoeV3CountingSortCutOriginBase(){};

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
    __aicore__ inline void FilterAndCount();
#ifdef MOE_CS_CUTORIGIN_VECTOR_FILTER
    __aicore__ inline void FilterAndCountVector();
#endif
#ifdef MOE_CS_USE_HISTOGRAM
    __aicore__ inline void FilterAndCountHistogram();
#endif
    __aicore__ inline void WriteExpertCountToWorkspace();
    __aicore__ inline void ComputeGlobalOffset();
    __aicore__ inline void WriteExpertTokens();

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
    int64_t filterChunkSize_;
    int64_t entriesAligned_;
    int64_t maskBytes_;
    int64_t expertTokensBufElems_;  // expertTokensLocal 缓冲元素数（KEY_VALUE 需 2 倍）
    int64_t expertCountElements_;   // 实际写出 expertTokens 的元素数

    // 公共 UB region offsets
    int64_t expertIdxLocalOffset_;
    int64_t scaleLocalOffset_;
    int64_t expertCountLocalOffset_;
    int64_t allCoreExpertCountLocalOffset_;
    int64_t expertTokensLocalOffset_;
    int64_t prefixSumLocalOffset_;
    int64_t filteredPairsLocalOffset_;
    int64_t xLocalOffset_;
    int64_t commonBufSize_;         // 公共区结尾 offset；子类在其后追加量化区
    // Phase C 双缓冲 ping-pong 专用 UB（Phase C 主因修复，默认实现）。
    // 两个 x 载入 slot（各 cols 对齐），ping-pong 交替；idxBuf 双 slot 供 rowIdx 写出。
    // csDbEnable_ 由 tiling 的 csColsLoops 信号驱动（1=双缓冲可容纳；0=大列超 UB，走串行 fallback 单 slot 布局）。
    int64_t csDbEnable_;
    int64_t xBufOff0_;
    int64_t xBufOff1_;
    int64_t idxBufDbOff0_;
    int64_t idxBufDbOff1_;
    int64_t idxStageBufOff_; // GATHER 连续批 rowIdx 批量写出 staging（CS_DB_BATCH_CAP int32）
    int64_t scaleBufDbOff0_; // 非量化 isInputScale 透传 scale 双 slot（各 BLOCK_BYTES）
    int64_t scaleBufDbOff1_;
    static constexpr int64_t CS_DB_BATCH_CAP = 512; // 批量预算数组栈占 ~8KB（2048 会超 32KB 默认栈上限致 RTS 注册失败） // 每批预算的 pair 数上限（批量预算 newPos/srcRow/origFlatIdx）
#if defined(MOE_CS_USE_HISTOGRAM) || defined(MOE_CS_CUTORIGIN_VECTOR_FILTER)
    // 向量过滤计数路径专用缓冲：fp32(CompareScalar) + 3 mask + gather 压缩结果 + flatIdx 序列。
    // 向量过滤按整个 coreEntries_ 对齐（与 FullLoad 一致），非 filterChunkSize_。
    // Histograms 路径额外追加 uint8 源 + uint16 直方图。
    int64_t expertIdxFp32LocalOffset_;
    int64_t compareMask0Offset_;
    int64_t compareMask1Offset_;
    int64_t gatherMaskOffset_;
    int64_t gatheredExpertLocalOffset_;
    int64_t gatheredIdxLocalOffset_;
    int64_t flatIdxBufferLocalOffset_;
#ifdef MOE_CS_USE_HISTOGRAM
    int64_t expertU8LocalOffset_;
    int64_t histBinOffset_;
#endif
#endif
};

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::InitCommon(
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
    scaleSlotSize_ = BLOCK_BYTES / static_cast<int64_t>(sizeof(float));
    maxFilteredCount_ = coreEntries_;
    filteredCount_ = 0;

    // expertTokens 缓冲：KEY_VALUE 输出 [expertId,count] 键值对需 actualExpertNum*2 元素；COUNT/CUMSUM 为 actualExpertNum。
    expertTokensBufElems_ = (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) ? actualExpertNum_ * EXPERT_ID_VALUE_NUM
                                                                              : actualExpertNum_;
    expertCountElements_ = expertTokensBufElems_;
    filterChunkSize_ = tiling->countingSortParamsOp.filterChunkSize > 0
                           ? tiling->countingSortParamsOp.filterChunkSize
                           : COUNTING_SORT_FILTER_CHUNK_SIZE;

    // 向量过滤/Histograms 均按整个 coreEntries_ 对齐（与 FullLoad 一致），非 filterChunkSize_。
    entriesAligned_ = Ceil(coreEntries_, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;
    maskBytes_ = AlignBytes(Ceil(entriesAligned_, static_cast<int64_t>(8)), static_cast<int64_t>(1));

    // tiling 复用 csColsLoops 传双缓冲使能：1=双缓冲 UB 可容纳；0=大列超 UB，降级串行单 slot
    csDbEnable_ = tiling->countingSortParamsOp.csColsLoops;

    // 公共 GM
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx);
    workspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace);
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        expertTokensGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokensCountOrCumsum);
    }

    ComputeCommonUbLayout();
}

// GATHER 模式预填 rowIdx=-1。子类在设置好 expandedX/expandedScale GM 后调用（dropPad 清零由子类补齐）。
template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::PrefillAndSync()
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
// dropPad 为非量化专属（算子契约），走 T 型 expandedXGm。scale 透传空槽清零由子类处理。
template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::ZeroOutExpandedDropPad(GlobalTensor<T> &expandedXGm)
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
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::ComputeCommonUbLayout()
{
    // Phase A/B/C persistent + max(phase) 模型。这里按最宽公共数据区预算 UB。
    int64_t offset = 0;

    // expertIdxLocal: coreEntries int32（Phase A 输入，索引一次性全载；x 行在 Phase C 单行流式搬运）
    // 注意：不能用 filterChunkSize_ 尺寸，LoadExpertIdx/FilterAndCount 扫描的是整个 coreEntries_，
    // 否则 coreEntries_ > filterChunkSize_ 时会越界写入 expertCountLocal 区域。
    expertIdxLocalOffset_ = offset;
    offset += AlignBytes(coreEntries_, static_cast<int64_t>(sizeof(int32_t)));

    // expertCountLocal: expertCountStride int32（persistent 常驻）
    expertCountLocalOffset_ = offset;
    offset += AlignBytes(expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    // allCoreExpertCountLocal: filterNeedCoreNum * expertCountStride int32（Phase B）
    allCoreExpertCountLocalOffset_ = offset;
    offset += AlignBytes(filterNeedCoreNum_ * expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    // expertTokensLocal: expertTokensBufElems int64（Phase B，KEY_VALUE 需 2 倍）
    expertTokensLocalOffset_ = offset;
    offset += AlignBytes(expertTokensBufElems_, static_cast<int64_t>(sizeof(int64_t)));

    // prefixSumLocal: expertCountStride int32
    prefixSumLocalOffset_ = offset;
    offset += AlignBytes(expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    // filteredPairsLocal: maxFilteredCount * 2 int32（Phase A 输出 / Phase C 输入）
    filteredPairsLocalOffset_ = offset;
    offset += AlignBytes(maxFilteredCount_ * 2, static_cast<int64_t>(sizeof(int32_t)));

    // xLocal: 单 token 行，Phase C 用
    xLocalOffset_ = offset;
    offset += AlignBytes(cols_, static_cast<int64_t>(sizeof(T)));

    // scaleLocal: 单 scale slot（Phase C 用，非量化透传/动态量化无此区，仅 isInputScale 非量化路径由子类使用）
    scaleLocalOffset_ = offset;
    if (isInputScale_ && quantMode_ != CS_QUANT_MODE_DYNAMIC) {
        offset += AlignBytes(scaleSlotSize_, static_cast<int64_t>(sizeof(float)));
    }

    // Phase C 双缓冲：slot0 复用上方 xLocal 区（串行 fallback 与双缓冲同一 Process 内互斥，不会同时使用），
    // 仅追加 slot1 + 2 个 rowIdx 写出 slot，节省一个整行 cols slot（cols=32768 时省 64KB）。
    // 非量化直接用这两个 x slot 做 ping-pong；量化子类另在 commonBufSize_ 之后追加 int8/scale 双 slot。
    // csDbEnable_==0（tiling 判定双缓冲超 UB）时不追加任何 DB 区，布局与串行 fallback 一致。
    xBufOff0_ = xLocalOffset_;
    xBufOff1_ = xLocalOffset_;
    idxBufDbOff0_ = prefixSumLocalOffset_;
    idxBufDbOff1_ = prefixSumLocalOffset_;
    idxStageBufOff_ = prefixSumLocalOffset_;
    scaleBufDbOff0_ = scaleLocalOffset_; // slot0 复用串行 scaleLocal 区
    scaleBufDbOff1_ = scaleLocalOffset_;
    if (csDbEnable_ != 0) {
        xBufOff1_ = offset;
        offset += AlignBytes(cols_, static_cast<int64_t>(sizeof(T)));
        idxBufDbOff0_ = offset;
        offset += BLOCK_BYTES;
        idxBufDbOff1_ = offset;
        offset += BLOCK_BYTES;
        // GATHER 连续批 rowIdx 批量写出 staging：把整批 newPos SetValue 进 UB 后一条 DMA 写出，
        // 消除每行 4B 小 DMA + S↔MTE3 双向同步往返（原逐行写是 GATHER 场景的最长串行链）。
        idxStageBufOff_ = offset;
        offset += AlignBytes(CS_DB_BATCH_CAP, static_cast<int64_t>(sizeof(int32_t)));
        // 非量化 isInputScale 透传 scale 第二 slot（slot0 复用 scaleLocal）
        if (isInputScale_ && quantMode_ != CS_QUANT_MODE_DYNAMIC) {
            scaleBufDbOff1_ = offset;
            offset += BLOCK_BYTES;
        }
    }

#if defined(MOE_CS_USE_HISTOGRAM) || defined(MOE_CS_CUTORIGIN_VECTOR_FILTER)
    // 向量过滤计数路径缓冲：fp32 + 3 mask + gather 压缩（整个 coreEntries_ 对齐，与 FullLoad 一致）。
    // Histograms 路径额外追加 uint8 源 + uint16 直方图。
    int64_t histEntriesAligned = Ceil(coreEntries_, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;
    int64_t histMaskBytes = AlignBytes(Ceil(histEntriesAligned, static_cast<int64_t>(8)), BLOCK_BYTES);

    expertIdxFp32LocalOffset_ = offset;
    offset += AlignBytes(histEntriesAligned, static_cast<int64_t>(sizeof(float)));
    compareMask0Offset_ = offset;
    offset += histMaskBytes;
    compareMask1Offset_ = offset;
    offset += histMaskBytes;
    gatherMaskOffset_ = offset;
    offset += histMaskBytes;
    gatheredExpertLocalOffset_ = offset;
    offset += AlignBytes(coreEntries_, static_cast<int64_t>(sizeof(int32_t)));
    gatheredIdxLocalOffset_ = offset;
    offset += AlignBytes(coreEntries_, static_cast<int64_t>(sizeof(int32_t)));
    flatIdxBufferLocalOffset_ = offset;
    offset += AlignBytes(coreEntries_, static_cast<int64_t>(sizeof(int32_t)));
#ifdef MOE_CS_USE_HISTOGRAM
    // uint8 源：coreEntries 对齐到 256（一个 Histograms repeat 处理 256 元素）
    expertU8LocalOffset_ = offset;
    offset += AlignBytes(Ceil(coreEntries_, static_cast<int64_t>(256)) * 256, static_cast<int64_t>(sizeof(uint8_t)));
    // uint16 直方图：256 bins（BIN0 128 + BIN1 128）
    histBinOffset_ = offset;
    offset += AlignBytes(256, static_cast<int64_t>(sizeof(uint16_t)));
#endif
#endif

    // 公共区结尾；子类量化区在其后追加
    commonBufSize_ = offset;
}

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::LoadExpertIdx()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(coreEntries_ * sizeof(int32_t)), 0, 0,
                                 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(expertIdxLocal, expertIdxGm_[coreFlatStart_], copyParams, padParams);
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
}

#ifdef MOE_CS_USE_HISTOGRAM
// 直方图计数 VF：uint8 源（已窄化的 expertOffset，仅前 validCount 有效）-> Histograms -> uint16 bins。
// mask 限定每 repeat 的有效元素数，padding 不计入 bin，避免污染。
__simd_vf__ inline void MoeV3CutOriginHistogramVF(__ubuf__ uint8_t *srcAddr, __ubuf__ uint16_t *dstAddr,
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

// 新计数路径：向量过滤(复用 FullLoad 思路) -> 窄化 uint8 -> Histograms 计数 -> 回写 expertCountLocal(int32)；
// pairs 在压缩结果上标量填充（不再做 count 累加）。计数结果须与旧标量路径逐 bin 一致。
template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::FilterAndCountHistogram()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();

    // 向量过滤：Cast int32->fp32 -> CompareScalar(GE/LT) -> And -> GatherMask 压缩合法专家 + 原始下标 j。
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
    MoeV3CutOriginHistogramVF((__ubuf__ uint8_t *)expertU8Local.GetPhyAddr(),
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
#endif // MOE_CS_USE_HISTOGRAM

template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::FilterAndCount()
{
#ifdef MOE_CS_USE_HISTOGRAM
    FilterAndCountHistogram();
#elif defined(MOE_CS_CUTORIGIN_VECTOR_FILTER)
    FilterAndCountVector();
#else
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

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
#endif
}

#ifdef MOE_CS_CUTORIGIN_VECTOR_FILTER
// 向量过滤（阶段二，宏 MOE_CS_CUTORIGIN_VECTOR_FILTER）：Cast int32->fp32 -> CompareScalar(GE/LT) -> And
// -> GatherMask 压缩合法专家 + 原始下标，仅对压缩后的 filteredInBatch 个做标量 count + pairs 填充。
// count/pairs 结果与旧标量路径逐核 bit 一致（同专家、同 flatIdx、同顺序）。复用 FullLoad 已验证实现。
template <typename T>
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::FilterAndCountVector()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

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

    // 标量：构建 pairs + 累加 expert count（仅压缩后 filteredInBatch 个）
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
#endif // MOE_CS_CUTORIGIN_VECTOR_FILTER
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

    int32_t totalForExpertArr[COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM];
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
    int64_t keyValueOffset = 0; // KEY_VALUE 模式下的紧凑写出下标（跳过 count==0 的专家）
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        int32_t totalForExpert = totalForExpertArr[e];
        int32_t prefixForExpert = prefixSumLocal.GetValue(e);

        // expertTokens 由 core 0 统一写出，支持 COUNT / CUMSUM / KEY_VALUE 三种模式
        if (blockIdx_ == 0 && expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
            if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
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
                expertTokensLocal.SetValue(e, cumulativeSum + static_cast<int64_t>(totalForExpert));
            }
        }
        // dropPad：seed 为该专家跨核 per-expert 前缀；dropless：全局 cumulative + per-expert 前缀。
        if (dropPadMode_ == DROP_PAD_MODE) {
            expertCountLocal.SetValue(e, prefixForExpert);
        } else {
            expertCountLocal.SetValue(e, static_cast<int32_t>(cumulativeSum) + prefixForExpert);
        }
        cumulativeSum += totalForExpert;
    }

    if (blockIdx_ == 0 && expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
        expertCountElements_ = keyValueOffset * EXPERT_ID_VALUE_NUM;
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
    // COUNT/CUMSUM: expertCountElements_==actualExpertNum_；KEY_VALUE: 非零专家数*2
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(expertCountElements_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_CUT_ORIGIN_BASE_H
