/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_counting_sort_unfull_load.h
 * \brief 计数排序非全载（CutOrigin）模板
 */
#ifndef MOE_V3_COUNTING_SORT_UNFULL_LOAD_H
#define MOE_V3_COUNTING_SORT_UNFULL_LOAD_H

#include "moe_v3_common.h"
#include "simt_api/asc_simt.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

// Histograms 的 BIN0 覆盖 value ∈ [0, 128)，共 128 个 bin；CutOrigin 准入保证 actualExpertNum <= 128，
// 专家编号经 -expertStart_ 归一后只落在 BIN0 内，故只统计 BIN0。
constexpr int64_t CUT_HIST_BIN0_NUM = 128;
// 一次 LoadAlign 覆盖的 uint8 元素数（一个 VL = 256B）
constexpr uint32_t CUT_HIST_REPEAT_U8 = 256;
// u16->u32 加宽 vcvt 是 de-interleave（ZERO 取偶数 lane、ONE 取奇数 lane），故需 EVEN/ODD 双 trait
// 分两次取；trait 是引用型非类型模板参数，须定义在命名空间作用域（无局部链接性）。
static constexpr Reg::CastTrait CUT_CAST_TRAIT_U16_U32_LOW = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
static constexpr Reg::CastTrait CUT_CAST_TRAIT_U16_U32_HIGH = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

// 各专家命中计数的频次统计，自包含：一次 VF_CALL 完成 u8 直方图（BIN0，128 bin）+ u16->u32 加宽，
// 输出 128 个 int32 频次（交错还原连续 bin 序）。调用侧只需先做 classic int32->uint8 窄化并随后
// classic Add：窄化若也放进 VF，则每轮只能吃 64 个元素（b32 寄存器仅 64 lane），比 classic Cast 慢 4 倍。
__simd_vf__ inline void MoeV3CutOriginHistogramVF(__ubuf__ uint8_t *srcAddr, __ubuf__ uint32_t *dstAddr,
                                                  uint32_t validCount)
{
    using namespace AscendC::Reg;
    RegTensor<uint8_t> srcReg;
    RegTensor<uint16_t> freqReg;
    RegTensor<uint32_t> freqLow, freqHigh;
    MaskReg maskB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg maskB32 = CreateMask<uint32_t, MaskPattern::ALL>();

    Duplicate(freqReg, static_cast<uint16_t>(0));
    uint16_t repeatTimes = static_cast<uint16_t>((validCount + CUT_HIST_REPEAT_U8 - 1) / CUT_HIST_REPEAT_U8);
    uint32_t remain = validCount;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        MaskReg mask = UpdateMask<uint8_t>(remain);
        LoadAlign<uint8_t>(srcReg, srcAddr + CUT_HIST_REPEAT_U8 * i);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::FREQUENCY>(freqReg, srcReg, mask);
    }
    // 掩码按源类型（u16，128 lane）计；加宽后按 EVEN/ODD 交错写回，还原连续 bin 序
    Cast<uint32_t, uint16_t, CUT_CAST_TRAIT_U16_U32_LOW>(freqLow, freqReg, maskB16);
    Cast<uint32_t, uint16_t, CUT_CAST_TRAIT_U16_U32_HIGH>(freqHigh, freqReg, maskB16);
    StoreAlign<uint32_t, StoreDist::DIST_INTLV_B32>(dstAddr, freqLow, freqHigh, maskB32);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ScatterPairsStableSimt(
    int32_t batchSize, __ubuf__ int32_t *flatIdxLocalAddr, __ubuf__ int32_t *rankLocalAddr, __gm__ int32_t *dstGmAddr)
{
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < batchSize; i += static_cast<int32_t>(blockDim.x)) {
        dstGmAddr[rankLocalAddr[i]] = flatIdxLocalAddr[i];
    }
}

// dropPad 桥接：求「每段最后一个有效 token 落在哪个专家、该专家在本段的计数」。
// 原实现每段做一次 O(E) 标量线性查找，再把 8B 结果单独 DataCopyPad 出去，且段间被 S_MTE3/MTE3_S
// 串死（必须等上一笔 8B DMA 回来才能写下一次）。这里改为「一线程一段」：含式游标数组单增，
// 二分求首个 > pos 的下标即目标专家（与线性首命中严格等价），结果在 UB 上按段序连续落位，
// 调用侧一次 DataCopyPad 搬完全部段。
// LAUNCH_BOUND 取远小于 SIMT_THREAD_NUM：段数上限只有 aivCoreNum_，线程少则每线程寄存器多，
// 二分循环里的 int64 局部量才不会被挤到栈上（与 SIMT 侧其它小核数 kernel 同理）；
// 线程内再按 blockDim.x 步进，故段数超过线程数也正确。
constexpr int64_t CUT_SEG_SIMT_THREAD_NUM = 256;

__simt_vf__ __aicore__ LAUNCH_BOUND(CUT_SEG_SIMT_THREAD_NUM) inline void DropPadSegLastExpertSimt(
    int32_t segNum, int64_t perCoreRows, int64_t lastCoreRows, int64_t expertTotalCount, int32_t expertStart,
    int32_t expertNum, __ubuf__ int32_t *cursorAddr, __ubuf__ int32_t *outAddr)
{
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < segNum; i += static_cast<int32_t>(blockDim.x)) {
        int64_t segStart = static_cast<int64_t>(i) * perCoreRows;
        int64_t segLen = (i == segNum - 1) ? lastCoreRows : perCoreRows;
        int64_t lastExpertId = -1;
        int64_t lastTokenCount = 0;
        if (segStart < expertTotalCount) {
            // 段内最后一个有效 token 的全局位置
            int64_t segEnd = segStart + segLen;
            int64_t cap = (segEnd < expertTotalCount) ? segEnd : expertTotalCount;
            int64_t lastValidPos = cap - 1;
            // 首个 cursor[e] > lastValidPos 的下标；cursor[e] = 专家 e 的全局含式前缀
            int32_t lo = 0;
            int32_t hi = expertNum;
            while (lo < hi) {
                int32_t mid = lo + ((hi - lo) >> 1);
                if (static_cast<int64_t>(cursorAddr[mid]) <= lastValidPos) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            // prefix = 专家的独占前缀 = 上一个专家的含式前缀；边界专家在段内计数 = pos - max(segStart, prefix) + 1
            int64_t prefix = (lo == 0) ? 0 : static_cast<int64_t>(cursorAddr[lo - 1]);
            int64_t segBase = (segStart > prefix) ? segStart : prefix;
            lastExpertId = static_cast<int64_t>(lo) + expertStart;
            lastTokenCount = lastValidPos - segBase + 1;
        }
        outAddr[i * 2] = static_cast<int32_t>(lastExpertId);
        outAddr[i * 2 + 1] = static_cast<int32_t>(lastTokenCount);
    }
}

template <typename T>
class MoeV3CutOriginPhaseAB {
public:
    __aicore__ inline MoeV3CutOriginPhaseAB(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR expertTokens, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void FilterAndCountChunked();
    __aicore__ inline void CountChunkWithFilter(int64_t chunkLength, int64_t gmFlatOffset, int64_t &pairCursor);
    // 过滤核心：接收 CountChunkWithFilter 载入的 expertIdxLocal，调用方不再重复 DataCopyPad
    __aicore__ inline void CountChunkFilteredWithLoaded(int64_t chunkLength, int64_t gmFlatOffset,
                                                        LocalTensor<int32_t> &expertIdxLocal, int64_t &pairCursor);
    // 过滤两段式：构造 [expertStart_, expertEnd_) 命中掩码 / 按掩码收拢命中并计数写回（细分以缩小
    // CountChunkWithFilter）
    __aicore__ inline void BuildChunkFilterMask(int64_t chunkLength, int64_t alignedLen,
                                                LocalTensor<int32_t> &expertIdxLocal,
                                                LocalTensor<float> &expertIdxFp32Local,
                                                LocalTensor<uint8_t> &compareMask0, LocalTensor<uint8_t> &compareMask1,
                                                LocalTensor<uint8_t> &gatherMaskLocal);
    __aicore__ inline void GatherAndCountFilteredHits(int64_t chunkLength, int64_t gmFlatOffset,
                                                      LocalTensor<int32_t> &expertIdxLocal,
                                                      LocalTensor<uint8_t> &gatherMaskLocal,
                                                      LocalTensor<int32_t> &gatheredExpertLocal,
                                                      LocalTensor<int32_t> &flatIdxBufferLocal,
                                                      LocalTensor<int32_t> &gatheredIdxLocal, int64_t &pairCursor);
    __aicore__ inline void WritePairBatchToWs(int64_t pairCursor, int64_t count, LocalTensor<int32_t> &idxSrc,
                                              LocalTensor<int32_t> &expertSrc);
    __aicore__ inline void WriteExpertCountToWorkspace();
    // Phase B
    __aicore__ inline void ComputeGlobalOffset();
    __aicore__ inline void ScatterToSortedRowIdx();
    __aicore__ inline void WriteExpandedExpertIdx();
    __aicore__ inline void WriteExpertIdxValue();
    __aicore__ inline void WriteExpertTotalCount();
    __aicore__ inline void WriteExpertTokens();

private:
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;

    TPipe *pipe_;
    TBuf<TPosition::VECCALC> buf_;

    // GM tensors
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> pairsWorkspaceGm_;
    GlobalTensor<int32_t> expertCountWorkspaceGm_;
    GlobalTensor<int32_t> sortedRowIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    GlobalTensor<int32_t> expandedExpertIdxGm_;
    GlobalTensor<int32_t> expertIdxValueGm_;
    GlobalTensor<int64_t> expertTokensGm_;

    // Tiling params
    int64_t blockIdx_;
    int64_t n_;
    int64_t k_;
    int64_t expertStart_;
    int64_t expertEnd_;
    int64_t actualExpertNum_;
    int64_t expertNum_;
    int64_t rowIdxType_;
    int64_t filterNeedCoreNum_;
    int64_t filterChunkSize_;
    int64_t expertTokensNumFlag_;
    int64_t expertTokensNumType_;
    int64_t dropPadMode_;

    // Per-core distribution（Phase A）
    int64_t coreFlatStart_;
    int64_t coreEntries_;

    // Derived constants
    int64_t expertCountStride_;
    int64_t pairsPerCore_;
    int64_t maxChunks_;
    int64_t chunkAligned_;
    int64_t maskBytes_;

    int64_t dropPadNeedCoreNum_;
    int64_t dropPadPerCoreRows_;
    int64_t dropPadLastCoreRows_;

    // UB offsets
    int64_t expertCountLocalOffset_;
    int64_t totalCountLocalOffset_;
    int64_t prefixSumLocalOffset_;
    int64_t oneCoreExpertCountLocalOffset_; // batchBuf
    int64_t expertTokensLocalOffset_;
    int64_t expertCountAlign_;
    int64_t persistentSize_;
    int64_t totalBufSize_;
    int64_t batchBufSize_;
    int64_t pairsBatchElements_;
    int64_t scatterFlatIdxOffset_;
    int64_t scatterExpertIdxOffset_;
    int64_t scatterIdxBufOffset_;
    int64_t globalInclPrefixOffset_;
    int64_t dropPadSegOutOffset_;

    // Results
    int64_t coreTotalPairs_;
    int64_t expertTotalCount_;
};

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR expertTokens,
                                                      GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tiling,
                                                      TPipe *pipe)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();

    n_ = tiling->n;
    k_ = tiling->k;
    expertStart_ = tiling->expertStart;
    expertEnd_ = tiling->expertEnd;
    actualExpertNum_ = tiling->actualExpertNum;
    expertNum_ = tiling->expertNum;
    rowIdxType_ = tiling->rowIdxType;
    filterNeedCoreNum_ = tiling->countingSortParamsOp.filterNeedCoreNum;
    expertTokensNumFlag_ = tiling->expertTokensNumFlag;
    expertTokensNumType_ = tiling->expertTokensNumType;
    dropPadMode_ = tiling->dropPadMode;
    filterChunkSize_ = COUNTING_SORT_FILTER_CHUNK_SIZE;

    int64_t filterPerCoreTokens = tiling->countingSortParamsOp.filterPerCoreTokens;
    int64_t coreTokenStart = blockIdx_ * filterPerCoreTokens;
    int64_t coreTokenEnd = Min(coreTokenStart + filterPerCoreTokens, n_);
    if (blockIdx_ == filterNeedCoreNum_ - 1) {
        coreTokenEnd = n_;
    }
    coreEntries_ = (coreTokenEnd - coreTokenStart) * k_;
    coreFlatStart_ = coreTokenStart * k_;

    expertCountStride_ = AlignElem(actualExpertNum_, static_cast<int64_t>(8));
    chunkAligned_ = AlignElem(filterChunkSize_, static_cast<int64_t>(8));
    maxChunks_ = Ceil(coreEntries_, filterChunkSize_);
    maskBytes_ = AlignElem(Ceil(chunkAligned_, static_cast<int64_t>(8)), static_cast<int64_t>(sizeof(int8_t)));

    // Workspace stride: consistent across all cores (use max possible per-core entries)
    int64_t lastCoreTokens = n_ - (filterNeedCoreNum_ - 1) * filterPerCoreTokens;
    int64_t maxCoreTokens = Max(filterPerCoreTokens, lastCoreTokens);
    int64_t maxCoreEntries = maxCoreTokens * k_;
    pairsPerCore_ = AlignElem(maxCoreEntries, static_cast<int64_t>(8)) * 2;

    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx);

    if (rowIdxType_ == GATHER) {
        if (blockIdx_ < filterNeedCoreNum_) {
            GlobalTensor<int32_t> expandedRowIdxGmTmp = expandedRowIdxGm_[filterPerCoreTokens * k_ * blockIdx_];
            if (blockIdx_ == filterNeedCoreNum_ - 1) {
                InitGlobalMemory(expandedRowIdxGmTmp, lastCoreTokens * k_, -1);
            } else {
                InitGlobalMemory(expandedRowIdxGmTmp, maxCoreEntries, -1);
            }
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
        SyncAll();
    }
    int64_t wsOffset = tiling->countingSortParamsOp.pairsWsOffset;
    pairsWorkspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + wsOffset);
    wsOffset += filterNeedCoreNum_ * pairsPerCore_;
    expertCountWorkspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + wsOffset);

    sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                    Align(n_ * k_, static_cast<int64_t>(sizeof(int32_t))));
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                        Align(n_ * k_, static_cast<int64_t>(sizeof(int32_t))) * 2 +
                                        Align(actualExpertNum_, static_cast<int64_t>(sizeof(int32_t))));
    expandedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace);
    expertIdxValueGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                      Align(n_ * k_, static_cast<int64_t>(sizeof(int32_t))) * 2 +
                                      Align(actualExpertNum_, static_cast<int64_t>(sizeof(int32_t))));

    // SCATTER + useGatherCopy（逆表协议）：本模板只产"位置->源"正表(expandedRowIdx)，
    // 不产 stage3 RowIdxGather 消费/产出的 "源->位置" 逆表(workspace+Align(n*k))。
    // GatherOut 在 useGatherCopy 下按全源域 [0,n*k) 遍历逆表，仅 outIndex<0 视为空位；
    // 专家子区间(ep_==1)时正表只覆盖 [0,expertTotalCount)，大量无效源槽若残留旧值会被误当合法 dest，
    // 因此在阶段切换前把逆表槽整段预填 -1（与 GATHER 模式预填 expandedRowIdx=-1 同理）。
    // ep_==0 为全域正置换，RowIdxGather 会写满每个有效源，无需预填。
    if (rowIdxType_ == SCATTER && tiling->useGatherCopy == 1) {
        if (blockIdx_ < filterNeedCoreNum_) {
            GlobalTensor<int32_t> sortedRowIdxGmTmp = sortedRowIdxGm_[filterPerCoreTokens * k_ * blockIdx_];
            if (blockIdx_ == filterNeedCoreNum_ - 1) {
                InitGlobalMemory(sortedRowIdxGmTmp, lastCoreTokens * k_, -1);
            } else {
                InitGlobalMemory(sortedRowIdxGmTmp, maxCoreEntries, -1);
            }
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
        SyncAll();
    }
    dropPadNeedCoreNum_ = tiling->srcToDstDropPadParamsOp.needCoreNum;
    dropPadPerCoreRows_ = tiling->srcToDstDropPadParamsOp.perCoreRows;
    dropPadLastCoreRows_ = tiling->srcToDstDropPadParamsOp.lastCoreRows;
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        expertTokensGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokens);
    }

    // UB 布局及各段大小由 tiling 下发（host 统一计算），kernel 仅按偏移使用
    expertCountLocalOffset_ = 0;
    expertCountAlign_ = tiling->countingSortParamsOp.coutSortExpertCountAlign;
    persistentSize_ = tiling->countingSortParamsOp.coutSortPersistentSize;
    batchBufSize_ = tiling->countingSortParamsOp.coutSortBatchBufSize;
    pairsBatchElements_ = tiling->countingSortParamsOp.coutSortPairsBatchElements;

    totalCountLocalOffset_ = persistentSize_;
    prefixSumLocalOffset_ = persistentSize_ + expertCountAlign_;
    oneCoreExpertCountLocalOffset_ = persistentSize_ + 2 * expertCountAlign_;
    // expertTokensLocal 复用 prefixSum + batchBuf 区（scalar loop 后 prefixSum 不再需要）
    expertTokensLocalOffset_ = prefixSumLocalOffset_;

    // 离散搬出临时区：pairs batch（flatIdx + expertIdx）+ idxBuf，放在 batchBuf 之后
    int64_t pairsBatchSlot = AlignElem(pairsBatchElements_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    int64_t scatterBase = persistentSize_ + 2 * expertCountAlign_ + batchBufSize_;
    scatterFlatIdxOffset_ = scatterBase;
    scatterExpertIdxOffset_ = scatterBase + pairsBatchSlot;
    scatterIdxBufOffset_ = scatterExpertIdxOffset_ + pairsBatchSlot;

    // 全局含式前缀（int32）：host 把持久区扩成 2*expertCountAlign_，本槽位是其后半段。
    // ComputeGlobalOffset 走本核游标时顺手落盘，Phase B 收尾段两处复用（SIMT 二分 / ExpertTokens CUMSUM）
    globalInclPrefixOffset_ = expertCountAlign_;
    // dropPad 段结果暂存：借 Phase B 离散搬出用过的 expertIdxLocal 槽位（scatter 之后即空闲）
    dropPadSegOutOffset_ = scatterExpertIdxOffset_;

    totalBufSize_ = tiling->countingSortParamsOp.coutSortTotalBufSize;
    pipe_->InitBuffer(buf_, totalBufSize_);
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::Process()
{
    if (blockIdx_ >= filterNeedCoreNum_) {
        SyncAll(); // 等 Phase A：各核计数写回 GM 完成
        SyncAll(); // 等 Phase B：离散搬出 sortedRowIdx 完成
        SyncAll(); // 等收尾：dropPad 桥接产物 / ExpertTokens 写 GM 完成
        return;
    }
    // Phase A: filter + count
    FilterAndCountChunked();
    WriteExpertCountToWorkspace();
    SyncAll();

    // Phase B: global offset + discrete scatter + bridge products
    ComputeGlobalOffset();
    ScatterToSortedRowIdx();
    SyncAll();
    if (dropPadMode_ == DROP_PAD_MODE) {
        // dropPad：离散搬出后补两个桥接产物（expandedExpertIdx + expertIdxValue），
        // 供 Stage3 RowIdxGatherDropPad 消费；expertTotalCount 在 dropPad 下不写（同一槽位是 expertIdxValue）。
        WriteExpandedExpertIdx();
        WriteExpertIdxValue();
    } else {
        WriteExpertTotalCount();
    }
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        WriteExpertTokens();
    }
    SyncAll();
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::FilterAndCountChunked()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();

    // 本核 pair 总数直接复用 Phase A 的写入游标：它统计的正是「本核已写入 pairs 工作区的 pair 数」，
    // 与本核计数表行和恒等，故 Phase B 无需再从计数表做一次行归约
    coreTotalPairs_ = 0;

    for (int64_t chunkIdx = 0; chunkIdx < maxChunks_; chunkIdx++) {
        int64_t chunkStart = chunkIdx * filterChunkSize_;
        int64_t chunkLength = Min(filterChunkSize_, coreEntries_ - chunkStart);
        int64_t gmFlatOffset = coreFlatStart_ + chunkStart;

        CountChunkWithFilter(chunkLength, gmFlatOffset, coreTotalPairs_);
    }
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::CountChunkWithFilter(int64_t chunkLength, int64_t gmFlatOffset,
                                                                      int64_t &pairCursor)
{
    int64_t expertIdxOff = persistentSize_;
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxOff / sizeof(int32_t)];

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(chunkLength * sizeof(int32_t)), 0, 0,
                                 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(expertIdxLocal, expertIdxGm_[gmFlatOffset], copyParams, padParams);

    // 载入同步：MTE2 载入完成 + 上一 chunk 的 MTE3 搬出完成
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

    CountChunkFilteredWithLoaded(chunkLength, gmFlatOffset, expertIdxLocal, pairCursor);
}

// 过滤核心：expertIdxLocal 由 CountChunkWithFilter 载入后传入。
template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::CountChunkFilteredWithLoaded(int64_t chunkLength, int64_t gmFlatOffset,
                                                                              LocalTensor<int32_t> &expertIdxLocal,
                                                                              int64_t &pairCursor)
{
    int64_t off = persistentSize_ + AlignElem(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

    int64_t expertIdxFp32Off = off;
    off += AlignElem(chunkAligned_ * static_cast<int64_t>(sizeof(float)), BLOCK_BYTES);
    int64_t compareMask0Off = off;
    off += maskBytes_;
    int64_t compareMask1Off = off;
    off += maskBytes_;
    int64_t gatherMaskOff = off;
    off += maskBytes_;
    int64_t flatIdxBufferOff = off;
    off += AlignElem(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

    int64_t gatheredExpertOff = off;
    off += AlignElem(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    int64_t gatheredIdxOff = off;

    LocalTensor<float> expertIdxFp32Local = buf_.Get<float>()[expertIdxFp32Off / sizeof(float)];
    LocalTensor<uint8_t> compareMask0 = buf_.Get<uint8_t>()[compareMask0Off];
    LocalTensor<uint8_t> compareMask1 = buf_.Get<uint8_t>()[compareMask1Off];
    LocalTensor<uint8_t> gatherMaskLocal = buf_.Get<uint8_t>()[gatherMaskOff];
    LocalTensor<int32_t> flatIdxBufferLocal = buf_.Get<int32_t>()[flatIdxBufferOff / sizeof(int32_t)];
    LocalTensor<int32_t> gatheredExpertLocal = buf_.Get<int32_t>()[gatheredExpertOff / sizeof(int32_t)];
    LocalTensor<int32_t> gatheredIdxLocal = buf_.Get<int32_t>()[gatheredIdxOff / sizeof(int32_t)];

    int64_t alignedLen = Ceil(chunkLength, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;

    BuildChunkFilterMask(chunkLength, alignedLen, expertIdxLocal, expertIdxFp32Local, compareMask0, compareMask1,
                         gatherMaskLocal);
    GatherAndCountFilteredHits(chunkLength, gmFlatOffset, expertIdxLocal, gatherMaskLocal, gatheredExpertLocal,
                               flatIdxBufferLocal, gatheredIdxLocal, pairCursor);
}

// 段1：构造 [expertStart_, expertEnd_) 命中掩码
template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::BuildChunkFilterMask(int64_t chunkLength, int64_t alignedLen,
                                                                      LocalTensor<int32_t> &expertIdxLocal,
                                                                      LocalTensor<float> &expertIdxFp32Local,
                                                                      LocalTensor<uint8_t> &compareMask0,
                                                                      LocalTensor<uint8_t> &compareMask1,
                                                                      LocalTensor<uint8_t> &gatherMaskLocal)
{
    Cast(expertIdxFp32Local, expertIdxLocal, RoundMode::CAST_ROUND, alignedLen);
    PipeBarrier<PIPE_V>();

    if (chunkLength < alignedLen) {
        // 尾块全部落在最后一个 repeat 内，故以该 repeat 起点（256B 对齐）为基址、按位掩码只填 [chunkLength, alignedLen)
        int64_t tailBase = alignedLen - ONE_REPEAT_COMPARE_NUM;
        uint64_t mask0 = UINT64_MAX << static_cast<uint32_t>(chunkLength - tailBase);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdxFp32Local[tailBase], static_cast<float>(-1), mask, 1, 1, 1);
        PipeBarrier<PIPE_V>();
    }

    CompareScalar(compareMask0, expertIdxFp32Local, static_cast<float>(expertStart_), CMPMODE::GE, alignedLen);
    CompareScalar(compareMask1, expertIdxFp32Local, static_cast<float>(expertEnd_), CMPMODE::LT, alignedLen);
    PipeBarrier<PIPE_V>();
    And(gatherMaskLocal.ReinterpretCast<uint16_t>(), compareMask0.ReinterpretCast<uint16_t>(),
        compareMask1.ReinterpretCast<uint16_t>(), Ceil(alignedLen, MASK_STRIDE) * MASK_STRIDE / DST_REP_STRIDE / 2);
    PipeBarrier<PIPE_V>();
}

// 段2：按掩码收拢命中并计数写回 pairs
template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::GatherAndCountFilteredHits(
    int64_t chunkLength, int64_t gmFlatOffset, LocalTensor<int32_t> &expertIdxLocal,
    LocalTensor<uint8_t> &gatherMaskLocal, LocalTensor<int32_t> &gatheredExpertLocal,
    LocalTensor<int32_t> &flatIdxBufferLocal, LocalTensor<int32_t> &gatheredIdxLocal, int64_t &pairCursor)
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    uint64_t rsvdCnt = 0;
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 1;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = DST_REP_STRIDE;
    gatherMaskParams.src1RepeatStride = DST_REP_STRIDE;
    GatherMask(gatheredExpertLocal, expertIdxLocal, gatherMaskLocal.ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(chunkLength), gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();

    int64_t filteredInChunk = static_cast<int64_t>(rsvdCnt);
    if (filteredInChunk > 0) {
        ArithProgression<int32_t>(flatIdxBufferLocal, static_cast<int32_t>(gmFlatOffset), 1,
                                  static_cast<int32_t>(chunkLength));
        PipeBarrier<PIPE_V>();

        uint64_t idxRsvdCnt = 0;
        GatherMask(gatheredIdxLocal, flatIdxBufferLocal, gatherMaskLocal.ReinterpretCast<uint32_t>(), true,
                   static_cast<uint32_t>(chunkLength), gatherMaskParams, idxRsvdCnt);
        PipeBarrier<PIPE_V>();

        Adds(gatheredExpertLocal, gatheredExpertLocal, static_cast<int32_t>(-expertStart_),
             static_cast<int32_t>(filteredInChunk));
        PipeBarrier<PIPE_V>();

        // 各专家计数改为直方图统计，替代原先逐元素 GetValue/SetValue。
        // expert 已归一到 [0, actualExpertNum_)、且 actualExpertNum_ <= 128，故只需 BIN0（128 个 bin）。
        // 两段 scratch（u8 源 / u32 频次）复用已死的 flatIdxBuffer 区：容量为 chunkAligned_ 个 int32，
        // 而 u8 源最多只用 chunkAligned_ 字节，余量足够放下 512B 的 u32 频次区，故不新增 UB 预留、不改 host 布局。
        int64_t histAddLen = Min(expertCountStride_, CUT_HIST_BIN0_NUM);
        LocalTensor<uint8_t> expertU8Local = flatIdxBufferLocal.ReinterpretCast<uint8_t>();
        LocalTensor<uint32_t> histBinU32Local =
            flatIdxBufferLocal[chunkAligned_ / sizeof(int32_t)].ReinterpretCast<uint32_t>();
        // int32 -> uint8 一步精确转换（expert 已非负且 < 128）
        Cast(expertU8Local, gatheredExpertLocal, RoundMode::CAST_NONE, static_cast<int32_t>(filteredInChunk));
        PipeBarrier<PIPE_V>();
        VF_CALL<MoeV3CutOriginHistogramVF>((__ubuf__ uint8_t *)expertU8Local.GetPhyAddr(),
                                           (__ubuf__ uint32_t *)histBinU32Local.GetPhyAddr(),
                                           static_cast<uint32_t>(filteredInChunk));
        PipeBarrier<PIPE_V>();
        Add(expertCountLocal, expertCountLocal, histBinU32Local.ReinterpretCast<int32_t>(),
            static_cast<int32_t>(histAddLen));
        PipeBarrier<PIPE_V>();

        WritePairBatchToWs(pairCursor, filteredInChunk, gatheredIdxLocal, gatheredExpertLocal);
        pairCursor += filteredInChunk;
    }
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::WritePairBatchToWs(int64_t pairCursor, int64_t count,
                                                                    LocalTensor<int32_t> &idxSrc,
                                                                    LocalTensor<int32_t> &expertSrc)
{
    if (count <= 0) {
        return;
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyExtParams wpCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairCursor], idxSrc, wpCopyParams);
    DataCopyPad(pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairsPerCore_ / 2 + pairCursor], expertSrc, wpCopyParams);
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::WriteExpertCountToWorkspace()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    // expertCountLocal 由 V(Add) 写出（原为标量 S 逐元素累加），MTE3 搬出需等 V 完成
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(expertCountStride_ * sizeof(int32_t)),
                                 0, 0, 0};
    DataCopyPad(expertCountWorkspaceGm_[blockIdx_ * expertCountStride_], expertCountLocal, copyParams);
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::ComputeGlobalOffset()
{
    int64_t batchCores = batchBufSize_ / expertCountAlign_;
    if (batchCores > filterNeedCoreNum_) {
        batchCores = filterNeedCoreNum_;
    }
    if (batchCores < 1) {
        batchCores = 1;
    }
    LocalTensor<int32_t> batchBufLocal = buf_.Get<int32_t>()[oneCoreExpertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> totalCountLocal = buf_.Get<int32_t>()[totalCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> prefixSumLocal = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    Duplicate(totalCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    Duplicate(prefixSumLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();

    // 列归约参数：dst 与 src1 同址（原地叠加）、二者迭代间步长为 0，src0 每次迭代沿行推进一个
    // expertCountStride_；步长单位为 datablock(32B)，故除以 COUNTING_SORT_ONE_BLOCK_ELEMENT
    BinaryRepeatParams colReduceParams{
        1, 1, 1, 0, static_cast<uint8_t>(expertCountStride_ / COUNTING_SORT_ONE_BLOCK_ELEMENT), 0};

    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    for (int64_t batchStart = 0; batchStart < filterNeedCoreNum_; batchStart += batchCores) {
        int64_t batchEnd = Min(batchStart + batchCores, filterNeedCoreNum_);
        int64_t curBatchSize = batchEnd - batchStart;

        int64_t batchElements = curBatchSize * expertCountStride_;
        DataCopyExtParams loadParams{static_cast<uint16_t>(1), static_cast<uint32_t>(batchElements * sizeof(int32_t)),
                                     0, 0, 0};
        DataCopyPad(batchBufLocal, expertCountWorkspaceGm_[batchStart * expertCountStride_], loadParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        // 整批行一次列归约，替代原先「逐核 Add + PipeBarrier」的两重循环。
        // 连续模式 mask 上限为一个 repeat 的元素数（int32 为 64），而 expertCountStride_ 可达 128，故按 64 分列下发
        int64_t prefixRows = Min(batchEnd, blockIdx_) - batchStart; // 每批中用到的「本核之前的行」数，可能 <= 0
        for (int64_t col = 0; col < expertCountStride_; col += ONE_REPEAT_COMPARE_NUM) {
            uint64_t laneNum = static_cast<uint64_t>(Min(ONE_REPEAT_COMPARE_NUM, expertCountStride_ - col));
            // 全核总数：本批全部行
            Add(totalCountLocal[col], batchBufLocal[col], totalCountLocal[col], laneNum,
                static_cast<uint8_t>(curBatchSize), colReduceParams);
            // 本核前缀和：只累加块号小于本核的行
            if (prefixRows > 0) {
                Add(prefixSumLocal[col], batchBufLocal[col], prefixSumLocal[col], laneNum,
                    static_cast<uint8_t>(prefixRows), colReduceParams);
            }
        }

        if (batchStart + batchCores < filterNeedCoreNum_) {
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        }
    }

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    LocalTensor<int32_t> inclPrefixLocal = buf_.Get<int32_t>()[globalInclPrefixOffset_ / sizeof(int32_t)];
    int64_t cumulativeSum = 0;
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        int32_t totalForExpert = totalCountLocal.GetValue(e);
        int32_t prefixForExpert = prefixSumLocal.GetValue(e);
        expertCountLocal.SetValue(e, static_cast<int32_t>(cumulativeSum) + prefixForExpert);
        cumulativeSum += totalForExpert;
        // 含式全局前缀（与核无关）：这里本就在累加同一个 totalCountLocal，顺手落盘供 Phase B 收尾段复用，
        // 免得 WriteExpertIdxValue / WriteExpertTokens 各再走一遍标量前缀
        inclPrefixLocal.SetValue(e, static_cast<int32_t>(cumulativeSum));
    }
    expertTotalCount_ = cumulativeSum;
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::ScatterToSortedRowIdx()
{
    if (coreTotalPairs_ <= 0) {
        return;
    }

    // Invalidate cache to see pairs written in Phase A
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(pairsWorkspaceGm_);
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> flatIdxLocal = buf_.Get<int32_t>()[scatterFlatIdxOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[scatterExpertIdxOffset_ / sizeof(int32_t)];

    // 目标 GM 基址：GATHER -> sortedRowIdx（workspace+Align(n*k)）；SCATTER -> expandedRowIdx（输出）
    __gm__ int32_t *dstGmAddr = (rowIdxType_ == GATHER) ? (__gm__ int32_t *)sortedRowIdxGm_.GetPhyAddr() :
                                                          (__gm__ int32_t *)expandedRowIdxGm_.GetPhyAddr();
    __ubuf__ int32_t *flatIdxAddr = (__ubuf__ int32_t *)flatIdxLocal.GetPhyAddr();
    __ubuf__ int32_t *rankAddr = (__ubuf__ int32_t *)expertIdxLocal.GetPhyAddr();

    DataCopyPadExtParams<int32_t> intPadParams{false, 0, 0, 0};

    int64_t pairCursor = 0;
    while (pairCursor < coreTotalPairs_) {
        int64_t batchSize = Min(pairsBatchElements_, coreTotalPairs_ - pairCursor);

        DataCopyExtParams pairLoadParams{static_cast<uint16_t>(1), static_cast<uint32_t>(batchSize * sizeof(int32_t)),
                                         0, 0, 0};
        DataCopyPad(flatIdxLocal, pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairCursor], pairLoadParams,
                    intPadParams);
        DataCopyPad(expertIdxLocal, pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairsPerCore_ / 2 + pairCursor],
                    pairLoadParams, intPadParams);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S); // MTE2 → scalar：UB 数据就绪
        for (int64_t i = 0; i < batchSize; i++) {
            int32_t e = expertIdxLocal.GetValue(i);
            int32_t pos = expertCountLocal.GetValue(e);
            expertIdxLocal.SetValue(i, pos);
            expertCountLocal.SetValue(e, pos + 1);
        }

        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        asc_vf_call<ScatterPairsStableSimt>(dim3{SIMT_THREAD_NUM, 1, 1}, static_cast<int32_t>(batchSize), flatIdxAddr,
                                            rankAddr, dstGmAddr);
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);

        pairCursor += batchSize;
    }
}

// dropPad 专用
template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::WriteExpandedExpertIdx()
{
    LocalTensor<int32_t> totalCountLocal = buf_.Get<int32_t>()[totalCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> prefixSumLocal = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> fillBuf = buf_.Get<int32_t>()[scatterFlatIdxOffset_ / sizeof(int32_t)];

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // 本核全局输出区间 [globalStart, globalEnd)：globalStart = sum_e prefixSumLocal[e]（之前所有核的 token 数）
    int64_t globalStart = 0;
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        globalStart += static_cast<int64_t>(prefixSumLocal.GetValue(e));
    }
    int64_t globalEnd = globalStart + coreTotalPairs_;

    // 每专家连续 run：expert e 占全局位置 [globalPrefix, globalPrefix + totalCountLocal[e])
    int64_t globalPrefix = 0;
    DataCopyExtParams runCp{1, 0, 0, 0, 0};
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        int32_t count = totalCountLocal.GetValue(e);
        int64_t runStart = Max(globalStart, globalPrefix);
        int64_t runEnd = Min(globalEnd, globalPrefix + static_cast<int64_t>(count));
        globalPrefix += count;
        if (runStart >= runEnd) {
            continue;
        }
        int32_t globalExpertId = static_cast<int32_t>(e + expertStart_);
        for (int64_t off = runStart; off < runEnd; off += pairsBatchElements_) {
            int64_t n = Min(pairsBatchElements_, runEnd - off);
            Duplicate(fillBuf, globalExpertId, static_cast<int32_t>(n));
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            runCp.blockLen = static_cast<uint32_t>(n * sizeof(int32_t));
            DataCopyPad(expandedExpertIdxGm_[off], fillBuf, runCp);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }

    // 无效尾区 [expertTotalCount, n*k) 填 -1：blockIdx 0 负责（与所有核的有效写 [0, expertTotalCount) 无重叠）
    if (blockIdx_ == 0 && expertTotalCount_ < n_ * k_) {
        for (int64_t off = expertTotalCount_; off < n_ * k_; off += pairsBatchElements_) {
            int64_t n = Min(pairsBatchElements_, n_ * k_ - off);
            Duplicate(fillBuf, static_cast<int32_t>(-1), static_cast<int32_t>(n));
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            runCp.blockLen = static_cast<uint32_t>(n * sizeof(int32_t));
            DataCopyPad(expandedExpertIdxGm_[off], fillBuf, runCp);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
}

// dropPad 专用
template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::WriteExpertIdxValue()
{
    if (blockIdx_ != 0 || dropPadNeedCoreNum_ <= 0) {
        return;
    }
    LocalTensor<int32_t> cursorLocal = buf_.Get<int32_t>()[globalInclPrefixOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> segOutLocal = buf_.Get<int32_t>()[dropPadSegOutOffset_ / sizeof(int32_t)];

    // 含式前缀游标由 ComputeGlobalOffset 的标量写落盘（那边本就在走同一个前缀，不必再走一遍），
    // 此处只需把该标量写发布给 SIMT
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

    asc_vf_call<DropPadSegLastExpertSimt>(
        dim3{CUT_SEG_SIMT_THREAD_NUM, 1, 1}, static_cast<int32_t>(dropPadNeedCoreNum_), dropPadPerCoreRows_,
        dropPadLastCoreRows_, expertTotalCount_, static_cast<int32_t>(expertStart_),
        static_cast<int32_t>(actualExpertNum_), (__ubuf__ int32_t *)cursorLocal.GetPhyAddr(),
        (__ubuf__ int32_t *)segOutLocal.GetPhyAddr());

    // SIMT 写 UB -> 一次搬出全部段（每段 2 个 int32，段序连续）
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyExtParams cp{static_cast<uint16_t>(1), static_cast<uint32_t>(dropPadNeedCoreNum_ * 2 * sizeof(int32_t)), 0,
                         0, 0};
    DataCopyPad(expertIdxValueGm_[0], segOutLocal, cp);
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::WriteExpertTotalCount()
{
    // 非 dropPad 才写：dropPad 场景同一 GM 槽位是 expertIdxValueGm_，且 Stage3 dropPad 不读此值
    if (blockIdx_ == 0 && dropPadMode_ != DROP_PAD_MODE) {
        LocalTensor<int32_t> totalCountBuf = buf_.Get<int32_t>()[scatterIdxBufOffset_ / sizeof(int32_t)];
        totalCountBuf.SetValue(0, static_cast<int32_t>(expertTotalCount_));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(expertTotalCountGm_[0], totalCountBuf, copyParams);
    }
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::WriteExpertTokens()
{
    if (blockIdx_ == 0 && expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        LocalTensor<int32_t> totalCountLocal = buf_.Get<int32_t>()[totalCountLocalOffset_ / sizeof(int32_t)];
        LocalTensor<int64_t> expertTokensLocal = buf_.Get<int64_t>()[expertTokensLocalOffset_ / sizeof(int64_t)];

        if (expertTokensNumType_ == EXERPT_TOKENS_COUNT) {
            Cast(expertTokensLocal, totalCountLocal, RoundMode::CAST_NONE, static_cast<int32_t>(actualExpertNum_));
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                         static_cast<uint32_t>(actualExpertNum_ * sizeof(int64_t)), 0, 0, 0};
            DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);

        } else if (expertTokensNumType_ == EXERPT_TOKENS_CUMSUM) {
            // 含式前缀由 ComputeGlobalOffset 一次性落盘（那里本就在走同一个前缀），此处退化为一次加宽，
            // 与 COUNT 分支同形，不再逐专家 GetValue/SetValue 走标量前缀
            LocalTensor<int32_t> inclPrefixLocal = buf_.Get<int32_t>()[globalInclPrefixOffset_ / sizeof(int32_t)];
            SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
            Cast(expertTokensLocal, inclPrefixLocal, RoundMode::CAST_NONE, static_cast<int32_t>(actualExpertNum_));
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                         static_cast<uint32_t>(actualExpertNum_ * sizeof(int64_t)), 0, 0, 0};
            DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);

        } else if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
            int64_t kvTotalElements =
                (actualExpertNum_ == expertNum_) ? actualExpertNum_ * 2 : (actualExpertNum_ + 1) * 2;
            Duplicate(expertTokensLocal, static_cast<int64_t>(0), static_cast<int32_t>(kvTotalElements));
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            int64_t kvOffset = 0;
            for (int64_t e = 0; e < actualExpertNum_; e++) {
                int32_t count = totalCountLocal.GetValue(e);
                if (count != 0) {
                    expertTokensLocal.SetValue(kvOffset * 2, static_cast<int64_t>(e + expertStart_));
                    expertTokensLocal.SetValue(kvOffset * 2 + 1, static_cast<int64_t>(count));
                    kvOffset++;
                }
            }
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                         static_cast<uint32_t>(kvTotalElements * sizeof(int64_t)), 0, 0, 0};
            DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);
        }
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_UNFULL_LOAD_H
