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
 * \brief A5 CountingSort FullLoad
 */
#ifndef MOE_V3_COUNTING_SORT_FULL_LOAD_UNQUANTIZED_H
#define MOE_V3_COUNTING_SORT_FULL_LOAD_UNQUANTIZED_H

#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"
#include "simt_api/asc_simt.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;
using namespace AscendC::Reg;

constexpr int64_t COUNT_SOURT_SIMT_THREAD_NUM = 1024;
constexpr int64_t COUNT_SORT_HIST_U8_REPEAT = 256;
constexpr int64_t COUNT_SORT_HIST_BIN_NUM = 256;
constexpr int64_t COUNT_SORT_HIST_CHUNK = 32768;
constexpr int64_t COUNT_SORT_PAIRS_I32_REPEAT = 64;
constexpr uint32_t CS_HIST_REPEAT_U8 = static_cast<uint32_t>(COUNT_SORT_HIST_U8_REPEAT);
constexpr uint32_t CS_HIST_REPEAT_CHUNK = static_cast<uint32_t>(COUNT_SORT_HIST_CHUNK);
constexpr uint32_t CS_PAIRS_REPEAT_I32 = static_cast<uint32_t>(COUNT_SORT_PAIRS_I32_REPEAT);
constexpr uint32_t CS_SCAN_ROUNDS = 5;
constexpr int32_t CS_I32_PER_I64 = static_cast<int32_t>(sizeof(int64_t) / sizeof(int32_t));
constexpr int32_t CS_PAIR_I32_STRIDE = static_cast<int32_t>(EXPERT_ID_VALUE_NUM) * CS_I32_PER_I64;

static constexpr Reg::CastTrait CAST_TRAIT_U16_U32_LOW = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
static constexpr Reg::CastTrait CAST_TRAIT_U16_U32_HIGH = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

__simt_vf__ __aicore__ LAUNCH_BOUND(COUNT_SOURT_SIMT_THREAD_NUM) inline void CoutSortFullLoadTopkWeightScatterSimt(
    int64_t elements, int64_t dstBase, int64_t totalLength, __gm__ int32_t *dstToSrcRow, __gm__ float *topkWeight,
    __gm__ volatile float *expandedTopkWeight)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < elements; i += static_cast<int64_t>(blockDim.x)) {
        int64_t dstIndex = dstBase + i;
        int32_t srcIndex = dstToSrcRow[dstIndex];
        if (srcIndex >= 0 && srcIndex < totalLength) {
            expandedTopkWeight[dstIndex] = topkWeight[srcIndex];
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(COUNT_SOURT_SIMT_THREAD_NUM) inline void CoutSortFullLoadTopkWeightGatherSimt(
    int64_t elements, int64_t srcBase, int64_t outputRows, __gm__ int32_t *srcToDstRow, __gm__ float *topkWeight,
    __gm__ volatile float *expandedTopkWeight)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < elements; i += static_cast<int64_t>(blockDim.x)) {
        int64_t srcIndex = srcBase + i;
        int32_t dstIndex = srcToDstRow[srcIndex];
        if (dstIndex >= 0 && dstIndex < outputRows) {
            expandedTopkWeight[dstIndex] = topkWeight[srcIndex];
        }
    }
}

// dstAddr[0,128) 为频次、[128,256) 为包含式前缀和；块间累加由调用方补一条 Add
__simd_vf__ inline void MoeV3FullLoadHistogramVF(__ubuf__ uint8_t *srcAddr, __ubuf__ uint16_t *dstAddr,
                                                 uint32_t validCount)
{
    RegTensor<uint8_t> srcReg;
    RegTensor<uint16_t> freqReg, cumReg;
    Duplicate(freqReg, static_cast<uint16_t>(0));
    Duplicate(cumReg, static_cast<uint16_t>(0));
    uint16_t repeatTimes = static_cast<uint16_t>((validCount + CS_HIST_REPEAT_U8 - 1) / CS_HIST_REPEAT_U8);
    uint32_t remain = validCount;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        MaskReg mask = UpdateMask<uint8_t>(remain);
        LoadAlign<uint8_t>(srcReg, srcAddr + CS_HIST_REPEAT_U8 * i);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::FREQUENCY>(freqReg, srcReg, mask);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(cumReg, srcReg, mask);
    }
    MaskReg maskAll = CreateMask<uint16_t, MaskPattern::ALL>();
    StoreAlign<uint16_t>(dstAddr, freqReg, maskAll);
    StoreAlign<uint16_t>(dstAddr + 128, cumReg, maskAll);
}

// BucketByExpert 第一遍：一次出各专家频次与包含式前缀和，32bit 累加全在寄存器内，输出固定
// COUNT_SORT_HIST_BIN_NUM / 2 个 int32。u16→u32 加宽 vcvt 是 de-interleave，故需 EVEN/ODD 双 trait
// （须定义在命名空间作用域）分两次取，再用 DIST_INTLV_B32 还原连续 bin 序。

__simd_vf__ inline void MoeV3FullLoadHistCountCumVF(__ubuf__ uint8_t *srcAddr, uint32_t validCount,
                                                    __ubuf__ uint32_t *countAddr, __ubuf__ uint32_t *cumAddr)
{
    RegTensor<uint8_t> srcReg;
    RegTensor<uint16_t> freqReg, cumReg;   // 单 chunk 内的 u16 累加器
    RegTensor<uint32_t> freqLow, freqHigh; // 跨 chunk 的 u32 累加器，EVEN/ODD 分离
    RegTensor<uint32_t> cumLow, cumHigh;
    RegTensor<uint32_t> tmpLow, tmpHigh; // Cast 落点，避免覆盖累加器（Add 非原地）
    // 掩码按源类型（u16，128 lane）计，与 vf_topk_gather 的 pregB16 一致
    MaskReg maskB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg maskB32 = CreateMask<uint32_t, MaskPattern::ALL>();
    Duplicate(freqLow, static_cast<uint32_t>(0));
    Duplicate(freqHigh, static_cast<uint32_t>(0));
    Duplicate(cumLow, static_cast<uint32_t>(0));
    Duplicate(cumHigh, static_cast<uint32_t>(0));

    for (uint32_t chunkStart = 0; chunkStart < validCount; chunkStart += CS_HIST_REPEAT_CHUNK) {
        uint32_t chunkElems =
            (validCount - chunkStart < CS_HIST_REPEAT_CHUNK) ? (validCount - chunkStart) : CS_HIST_REPEAT_CHUNK;
        Duplicate(freqReg, static_cast<uint16_t>(0));
        Duplicate(cumReg, static_cast<uint16_t>(0));
        uint16_t repeatTimes = static_cast<uint16_t>((chunkElems + CS_HIST_REPEAT_U8 - 1) / CS_HIST_REPEAT_U8);
        uint32_t remain = chunkElems;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            MaskReg mask = UpdateMask<uint8_t>(remain);
            LoadAlign<uint8_t>(srcReg, srcAddr + chunkStart + CS_HIST_REPEAT_U8 * i);
            Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::FREQUENCY>(freqReg, srcReg, mask);
            Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(cumReg, srcReg, mask);
        }
        Cast<uint32_t, uint16_t, CAST_TRAIT_U16_U32_LOW>(tmpLow, freqReg, maskB16);
        Cast<uint32_t, uint16_t, CAST_TRAIT_U16_U32_HIGH>(tmpHigh, freqReg, maskB16);
        Add(freqLow, freqLow, tmpLow, maskB32);
        Add(freqHigh, freqHigh, tmpHigh, maskB32);
        Cast<uint32_t, uint16_t, CAST_TRAIT_U16_U32_LOW>(tmpLow, cumReg, maskB16);
        Cast<uint32_t, uint16_t, CAST_TRAIT_U16_U32_HIGH>(tmpHigh, cumReg, maskB16);
        Add(cumLow, cumLow, tmpLow, maskB32);
        Add(cumHigh, cumHigh, tmpHigh, maskB32);
    }
    // 交错写回：交替取 EVEN/ODD 寄存器还原连续 bin 序
    StoreAlign<uint32_t, StoreDist::DIST_INTLV_B32>(countAddr, freqLow, freqHigh, maskB32);
    StoreAlign<uint32_t, StoreDist::DIST_INTLV_B32>(cumAddr, cumLow, cumHigh, maskB32);
}

// COUNT：expertTokens[e] = 全核总数 total[e]
__simd_vf__ inline void MoeV3FullLoadSeedsAndTokensCountVF(__ubuf__ int32_t *totalAddr, __ubuf__ int32_t *prefixAddr,
                                                           __ubuf__ int32_t *seedAddr, __ubuf__ int32_t *tokenAddr,
                                                           __ubuf__ int32_t *sumAddr, uint32_t expertNum)
{
    RegTensor<int32_t> idxReg;
    RegTensor<int32_t> shiftReg;
    RegTensor<int32_t> totalReg;
    RegTensor<int32_t> prefixReg;
    RegTensor<int32_t> scanReg;
    RegTensor<int32_t> gatherReg;
    RegTensor<int32_t> seedReg;
    RegTensor<int32_t> zeroReg;
    RegTensor<int32_t> sumReg;
    MaskReg maskAll = CreateMask<int32_t, MaskPattern::ALL>();
    MaskReg maskExpert = UpdateMask<int32_t>(expertNum);

    Duplicate(zeroReg, static_cast<int32_t>(0));
    Arange(idxReg, static_cast<int32_t>(0));
    // 全 VL 载入：预留不足一个 VL，但同 TBuf 之后仍有其它缓冲，越读不越界
    LoadAlign<int32_t>(totalReg, totalAddr);
    LoadAlign<int32_t>(prefixReg, prefixAddr);
    Select(totalReg, totalReg, zeroReg, maskExpert);
    Add(scanReg, totalReg, zeroReg, maskAll);
    for (uint32_t k = 0; k < CS_SCAN_ROUNDS; ++k) {
        uint32_t step = 1u << k;
        Adds(shiftReg, idxReg, -static_cast<int32_t>(step), maskAll);
        Gather(gatherReg, scanReg, (RegTensor<uint32_t> &)shiftReg);
        Add(scanReg, scanReg, gatherReg, maskAll);
    }
    Sub(seedReg, prefixReg, totalReg, maskAll);
    Add(seedReg, scanReg, seedReg, maskAll);
    StoreAlign<int32_t, StoreDist::DIST_NORM>(seedAddr, seedReg, maskExpert);
    StoreAlign<int32_t, StoreDist::DIST_INTLV_B32>(tokenAddr, totalReg, zeroReg, maskAll);
    // ReduceSum 结果广播到各 lane，取首个元素落盘供调用侧标量读回
    ReduceSum<int32_t, int32_t>(sumReg, totalReg, maskExpert);
    StoreAlign<int32_t, StoreDist::DIST_FIRST_ELEMENT_B32>(sumAddr, sumReg, maskAll);
}

__simd_vf__ inline void MoeV3FullLoadSeedsAndTokensCumSumVF(__ubuf__ int32_t *totalAddr, __ubuf__ int32_t *prefixAddr,
                                                            __ubuf__ int32_t *seedAddr, __ubuf__ int32_t *tokenAddr,
                                                            __ubuf__ int32_t *sumAddr, uint32_t expertNum)
{
    RegTensor<int32_t> idxReg;
    RegTensor<int32_t> shiftReg;
    RegTensor<int32_t> totalReg;
    RegTensor<int32_t> prefixReg;
    RegTensor<int32_t> scanReg;
    RegTensor<int32_t> gatherReg;
    RegTensor<int32_t> seedReg;
    RegTensor<int32_t> zeroReg;
    RegTensor<int32_t> sumReg;
    MaskReg maskAll = CreateMask<int32_t, MaskPattern::ALL>();
    MaskReg maskExpert = UpdateMask<int32_t>(expertNum);

    Duplicate(zeroReg, static_cast<int32_t>(0));
    Arange(idxReg, static_cast<int32_t>(0));
    LoadAlign<int32_t>(totalReg, totalAddr);
    LoadAlign<int32_t>(prefixReg, prefixAddr);
    Select(totalReg, totalReg, zeroReg, maskExpert);
    Add(scanReg, totalReg, zeroReg, maskAll);
    for (uint32_t k = 0; k < CS_SCAN_ROUNDS; ++k) {
        uint32_t step = 1u << k;
        Adds(shiftReg, idxReg, -static_cast<int32_t>(step), maskAll);
        Gather(gatherReg, scanReg, (RegTensor<uint32_t> &)shiftReg);
        Add(scanReg, scanReg, gatherReg, maskAll);
    }
    Sub(seedReg, prefixReg, totalReg, maskAll);
    Add(seedReg, scanReg, seedReg, maskAll);
    StoreAlign<int32_t, StoreDist::DIST_NORM>(seedAddr, seedReg, maskExpert);
    StoreAlign<int32_t, StoreDist::DIST_INTLV_B32>(tokenAddr, scanReg, zeroReg, maskAll);
    ReduceSum<int32_t, int32_t>(sumReg, totalReg, maskExpert);
    StoreAlign<int32_t, StoreDist::DIST_FIRST_ELEMENT_B32>(sumAddr, sumReg, maskAll);
}

// KEY_VALUE：只取非零专家、按 e 升序压紧的 (全局 id, total) 流式压缩。键值对为 int64 无 DIST_INTLV_B64，
// 故用 Scatter 只写掩码为 1 的 lane；尾部哨兵 (0,0) 依赖调用方事前的整块清零。
__simd_vf__ inline void MoeV3FullLoadSeedsAndTokensKeyValueVF(__ubuf__ int32_t *totalAddr, __ubuf__ int32_t *prefixAddr,
                                                              __ubuf__ int32_t *seedAddr, __ubuf__ int32_t *tokenAddr,
                                                              __ubuf__ int32_t *sumAddr, uint32_t expertNum,
                                                              int32_t expertStart)
{
    RegTensor<int32_t> idxReg;
    RegTensor<int32_t> shiftReg;
    RegTensor<int32_t> totalReg;
    RegTensor<int32_t> prefixReg;
    RegTensor<int32_t> scanReg;
    RegTensor<int32_t> gatherReg;
    RegTensor<int32_t> flagReg;
    RegTensor<int32_t> kvIdxReg;
    RegTensor<int32_t> payloadReg;
    RegTensor<int32_t> zeroReg;
    RegTensor<int32_t> sumReg;
    MaskReg maskAll = CreateMask<int32_t, MaskPattern::ALL>();
    MaskReg maskExpert = UpdateMask<int32_t>(expertNum);
    MaskReg maskNonZero;

    Duplicate(zeroReg, static_cast<int32_t>(0));
    Arange(idxReg, static_cast<int32_t>(0));
    LoadAlign<int32_t>(totalReg, totalAddr);
    LoadAlign<int32_t>(prefixReg, prefixAddr);
    Select(totalReg, totalReg, zeroReg, maskExpert);
    // 非零指示（Mins 夹到 1）的独占式前缀和 = 该专家的压紧落位 k
    Compares<int32_t, CMPMODE::NE>(maskNonZero, totalReg, static_cast<int32_t>(0), maskExpert);
    Mins(flagReg, totalReg, static_cast<int32_t>(1), maskAll);
    Add(kvIdxReg, flagReg, zeroReg, maskAll);
    for (uint32_t k = 0; k < CS_SCAN_ROUNDS; ++k) {
        uint32_t step = 1u << k;
        Adds(shiftReg, idxReg, -static_cast<int32_t>(step), maskAll);
        Gather(gatherReg, kvIdxReg, (RegTensor<uint32_t> &)shiftReg);
        Add(kvIdxReg, kvIdxReg, gatherReg, maskAll);
    }
    Sub(kvIdxReg, kvIdxReg, flagReg, maskAll);
    Muls(kvIdxReg, kvIdxReg, CS_PAIR_I32_STRIDE, maskAll);
    Adds(payloadReg, idxReg, expertStart, maskAll);
    Select(payloadReg, payloadReg, zeroReg, maskExpert);
    Scatter<int32_t, uint32_t>(tokenAddr, payloadReg, (RegTensor<uint32_t> &)kvIdxReg, maskNonZero);
    Adds(kvIdxReg, kvIdxReg, CS_I32_PER_I64, maskAll);
    Scatter<int32_t, uint32_t>(tokenAddr, totalReg, (RegTensor<uint32_t> &)kvIdxReg, maskNonZero);

    Add(scanReg, totalReg, zeroReg, maskAll);
    for (uint32_t k = 0; k < CS_SCAN_ROUNDS; ++k) {
        uint32_t step = 1u << k;
        Adds(shiftReg, idxReg, -static_cast<int32_t>(step), maskAll);
        Gather(gatherReg, scanReg, (RegTensor<uint32_t> &)shiftReg);
        Add(scanReg, scanReg, gatherReg, maskAll);
    }
    Sub(payloadReg, prefixReg, totalReg, maskAll);
    Add(payloadReg, scanReg, payloadReg, maskAll);
    StoreAlign<int32_t, StoreDist::DIST_NORM>(seedAddr, payloadReg, maskExpert);
    ReduceSum<int32_t, int32_t>(sumReg, totalReg, maskExpert);
    StoreAlign<int32_t, StoreDist::DIST_FIRST_ELEMENT_B32>(sumAddr, sumReg, maskAll);
}

// pairs 交错写：[localFlatIdx] / [expertOffset] 按 [j0,e0,j1,e1,...] 一次写回 UB；尾部 padding 垃圾由调用方
// 按 filteredCount_ 消费，写入长度不超 pairs 区容量（同 64 元素粒度预留）。
__simd_vf__ inline void MoeV3FullLoadPairsInterleaveVF(__ubuf__ uint32_t *idxAddr, __ubuf__ uint32_t *expertAddr,
                                                       __ubuf__ uint32_t *pairsAddr, uint32_t validCount)
{
    // 须与 ComputeCommonUbLayout 中 pairs 区预留粒度一致
    RegTensor<uint32_t> idxReg;
    RegTensor<uint32_t> expertReg;
    MaskReg maskAll = CreateMask<uint32_t, MaskPattern::ALL>();
    uint16_t repeatTimes = static_cast<uint16_t>((validCount + CS_PAIRS_REPEAT_I32 - 1) / CS_PAIRS_REPEAT_I32);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        LoadAlign<uint32_t>(idxReg, idxAddr + CS_PAIRS_REPEAT_I32 * i);
        LoadAlign<uint32_t>(expertReg, expertAddr + CS_PAIRS_REPEAT_I32 * i);
        StoreAlign<uint32_t, StoreDist::DIST_INTLV_B32>(pairsAddr + 2 * CS_PAIRS_REPEAT_I32 * i, idxReg, expertReg,
                                                        maskAll);
    }
}

// 分桶散填（VF，全专家）：以 bucketOffsetTbl[k]（排他前缀和）为桶起点，把 gatheredExpert[i] == k 的
// gatheredIdx[i] 按原 i 序稳定压紧写入。桶起点非 32B 对齐故走 Squeeze<STORE_REG> + StoreUnAlign；空桶
// 掩码全 0 不写字节；尾部须用 Arange + Compares(LT) 造有效掩码，否则 padding 垃圾会写进下一个桶。
__simd_vf__ inline void MoeV3FullLoadBucketCompactStoreVF(__ubuf__ int32_t *idxAddr, __ubuf__ int32_t *expertAddr,
                                                          __ubuf__ int32_t *bucketAddr, __ubuf__ int32_t *offsetTblAddr,
                                                          uint32_t expertNum, uint32_t validCount)
{
    // 与 pairs 区同一粒度：一个 VL(256B) 处理 64 个 int32
    RegTensor<int32_t> posReg;
    RegTensor<int32_t> expertReg;
    RegTensor<int32_t> idxReg;
    RegTensor<int32_t> packedReg;
    MaskReg maskAll = CreateMask<int32_t, MaskPattern::ALL>();
    MaskReg posMask;
    MaskReg keepMask;
    for (uint32_t k = 0; k < expertNum; k++) {
        __ubuf__ int32_t *dstAddr = bucketAddr + offsetTblAddr[k];
        UnalignRegForStore ureg;
        // AR 是全局特殊寄存器，每桶都须清零，否则上桶的有效元素数会带进本桶
        ClearSpr<AscendC::SpecialPurposeReg::AR>();
        for (uint32_t base = 0; base < validCount; base += CS_PAIRS_REPEAT_I32) {
            Arange(posReg, static_cast<int32_t>(base));
            Compares<int32_t, CMPMODE::LT>(posMask, posReg, static_cast<int32_t>(validCount), maskAll);
            LoadAlign<int32_t>(expertReg, expertAddr + base);
            Compares<int32_t, CMPMODE::EQ>(keepMask, expertReg, static_cast<int32_t>(k), posMask);
            LoadAlign<int32_t>(idxReg, idxAddr + base);
            // GatherMaskMode 在 AscendC 与 AscendC::Reg 下同名，须显式限定
            Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(packedReg, idxReg, keepMask);
            StoreUnAlign<int32_t, PostLiteral::POST_MODE_UPDATE>(dstAddr, packedReg, ureg);
        }
        StoreUnAlignPost(dstAddr, ureg); // 每桶必须冲刷尾块，否则残留会串进下一个桶
    }
}

template <typename T>
class MoeV3CountingSortFullLoadUnquantized {
public:
    __aicore__ inline MoeV3CountingSortFullLoadUnquantized(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR topkWeight,
                                GM_ADDR expandedTopkWeight, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

private:
    // --- 初始化：tiling 解析 / UB 布局 / 预填 ---
    __aicore__ inline void InitCommon(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                      GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void ComputeCommonUbLayout();
    __aicore__ inline void PrefillAndSync();
    // --- Phase A：载入 + 过滤计数 （Process 中同步点之前） ---
    __aicore__ inline void LoadExpertIdx();
    __aicore__ inline void LoadScale();
    __aicore__ inline void LoadXBackground(GlobalTensor<T> &xGm);
    __aicore__ inline void FilterAndCount();
    __aicore__ inline void FilterAndCountHistogram();
    __aicore__ inline void FilterAndCountVector();
    __aicore__ inline void WriteExpertCountToWorkspace();
    // --- Phase B：跨核归约 + 偏移（SyncAll 之后） ---
    __aicore__ inline void ComputeGlobalOffset();
    __aicore__ inline void LoadAndReduceAllCoreExpertCount(LocalTensor<int32_t> &allCoreExpertCountLocal,
                                                           LocalTensor<int32_t> &prefixSumLocal,
                                                           LocalTensor<int32_t> &expertCountLocal);
    __aicore__ inline int64_t ComputeSeedsAndExpertTokens(LocalTensor<int32_t> &prefixSumLocal,
                                                          LocalTensor<int32_t> &expertCountLocal, int64_t expertNum);
    __aicore__ inline void WriteExpertTokens();
    // --- Phase C：等待 x 加载 + 按专家搬出 ---
    __aicore__ inline void WaitXLoadCommon();
    __aicore__ inline void GatherAndWrite();
    __aicore__ inline void BucketByExpert();
    __aicore__ inline void GatherAndWriteByExpert(LocalTensor<T> &xLocal, LocalTensor<float> &scaleLocal,
                                                  GlobalTensor<T> &expandedXGmRef,
                                                  GlobalTensor<float> &expandedScaleGmRef);
    __aicore__ inline void CopyGatherRowToOutBuf(LocalTensor<T> &gatherOutBuf, LocalTensor<T> &xLocal, int64_t dstRow,
                                                 int64_t tokenRow);
    __aicore__ inline void WriteRowIdxAndScale(LocalTensor<int32_t> &idxBuf, LocalTensor<float> &scaleLocal,
                                               GlobalTensor<float> &expandedScaleGmRef, int64_t newPos,
                                               int64_t origFlatIdx, int64_t tokenRow);
    __aicore__ inline void FlushBatchToGm(GlobalTensor<T> &expandedXGmRef, LocalTensor<T> &gatherOutBuf,
                                          int64_t batchStartNewPos, int64_t batchRows);
    __aicore__ inline void GatherOneRow(int64_t newPos, int64_t tokenRow, int64_t origFlatIdx);
    // --- Phase D：topkWeight 重排输出 ---
    __aicore__ inline void TopkWeightPhase();
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;

    TPipe *pipe_;
    TBuf<TPosition::VECCALC> buf_;

    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> workspaceGm_;
    GlobalTensor<int64_t> expertTokensGm_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<float> topkWeightGm_;
    GlobalTensor<float> expandedTopkWeightGm_;

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
    int64_t isInputTopkWeight_{0};
    int64_t filterNeedCoreNum_;
    int64_t expertTokensNumFlag_;
    int64_t expertTokensNumType_;
    int64_t activeNum_;
    int64_t outputRows_;
    int64_t expertTotalCount_; // 实际保留（写满）的行数：dropless 下由 ComputeGlobalOffset 求得的 cumulativeSum，<=
                               // outputRows_
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
    int64_t expertTokensBufElems_; // expertTokensLocal 缓冲元素数（KEY_VALUE 需 2 倍）
    int64_t expertCountElements_;  // 实际写出 expertTokens 的元素数

    // 公共 UB region offsets (byte offsets from buf_ start)
    int64_t xLocalOffset_;
    int64_t expertIdxLocalOffset_;
    int64_t scaleLocalOffset_;
    int64_t expertCountLocalOffset_;
    int64_t allCoreExpertCountLocalOffset_;
    int64_t expertTokensLocalOffset_;
    int64_t prefixSumLocalOffset_;
    int64_t seedSumLocalOffset_; // 向量路径求 cumulativeSum 的归约落点（32B，供标量读回）
    int64_t filteredPairsLocalOffset_;
    int64_t expertIdxFp32LocalOffset_;
    int64_t compareMask0Offset_;
    int64_t compareMask1Offset_;
    int64_t gatherMaskOffset_;
    int64_t gatheredExpertLocalOffset_;
    int64_t flatIdxBufferLocalOffset_;
    int64_t gatheredIdxLocalOffset_;
    int64_t expertU8LocalOffset_; // Histogram 路径：专家偏移窄化到 uint8 的暂存区
    int64_t histBinOffset_;       // Histogram 路径：uint16 桶区（COUNT_SORT_HIST_BIN_NUM 个）
    int64_t histBinU32Offset_;    // Histogram 路径：频次桶的 uint32 扩展区（分块累加用）
    int64_t commonBufSize_;       // 公共区结尾 offset

    // 聚合搬出参数（从 tiling 读取）
    int64_t coutSortAggrEnable_{0};
    int64_t aggrOutRows_{0};        // k：搬出聚合 UB 容纳行数
    int64_t aggrOutBufBytes_{0};    // 搬出聚合区字节数
    int64_t gatherOutBufOffset_{0}; // 搬出聚合区在 buf_ 中的字节偏移
    // 分桶区（按专家外循环聚合用）
    int64_t bucketBaseOffset_{0};      // 桶数据基址（存 localFlatIdx，按 expert 分段）
    int64_t bucketOffsetTblOffset_{0}; // 偏移表基址（actualExpertNum_ 个 int32，每桶起始偏移）
    int64_t bucketCountTblOffset_{0};  // 每桶元素数（actualExpertNum_ 个 int32）
};

// 成员函数定义：与类内声明同序
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::Init(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR topkWeight, GM_ADDR expandedTopkWeight,
    GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe)
{
    InitCommon(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, workspace, tiling, pipe);

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX);
    if (isInputScale_) {
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale);
        expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);
    }
    // topkWeight 重排输出仅在输入携带 topk_weight 时启用（dropPad 已被 host gating 排除）
    if (isInputTopkWeight_) {
        topkWeightGm_.SetGlobalBuffer((__gm__ float *)topkWeight, totalLength_);
        expandedTopkWeightGm_.SetGlobalBuffer((__gm__ float *)expandedTopkWeight, totalLength_);
    }

    // GATHER 预填 rowIdx=-1
    PrefillAndSync();

    pipe_->InitBuffer(buf_, commonBufSize_);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::Process()
{
    if (blockIdx_ >= filterNeedCoreNum_) {
        SyncAll(); // 等 Phase A：各核计数写回 GM 完成
        if (isInputTopkWeight_) {
            SyncAll(); // 等 Phase C：各核 expandedRowIdx 写 GM 完成（tail 核与会者补全，保证 barrier 对称）
        }
        return;
    }

    // Phase A: Load + Filter + Count（核间并行）
    LoadExpertIdx();
    LoadScale();
    LoadXBackground(xGm_);
    FilterAndCount();
    WriteExpertCountToWorkspace();

    SyncAll();

    // Phase B: 核间归约 + 偏移计算
    ComputeGlobalOffset();
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        WriteExpertTokens();
    }

    // Phase C: 等待 x 加载 + 数据搬运
    WaitXLoadCommon();
    GatherAndWrite();

    // Phase D: topk 重排输出（scatter/gather dropless，见 TopkWeightPhase）
    if (isInputTopkWeight_) {
        SyncAll(); // 等全部核的 expandedRowIdx 写 GM 完成，随后 SIMT 读回 map
        TopkWeightPhase();
    }
}

// --- 初始化：tiling 解析 / UB 布局 / 预填 ---
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::InitCommon(
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
    isInputTopkWeight_ = tiling->isInputTopkWeight;
    filterNeedCoreNum_ = tiling->countingSortParamsOp.filterNeedCoreNum;
    expertTokensNumFlag_ = tiling->expertTokensNumFlag;
    expertTokensNumType_ = tiling->expertTokensNumType;
    activeNum_ = tiling->activeNum;
    outputRows_ = activeNum_;
    expertTotalCount_ = activeNum_;
    coutSortAggrEnable_ = tiling->countingSortParamsOp.coutSortAggrEnable;
    aggrOutRows_ = tiling->countingSortParamsOp.coutSortAggrOutRows;
    aggrOutBufBytes_ = tiling->countingSortParamsOp.coutSortAggrOutBufBytes;

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

    if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
        expertTokensBufElems_ = Min(actualExpertNum_ + 1, expertNum_) * EXPERT_ID_VALUE_NUM;
    } else {
        expertTokensBufElems_ = actualExpertNum_;
    }
    expertCountElements_ = expertTokensBufElems_;

    entriesAligned_ = Ceil(coreEntries_, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;
    maskBytes_ = AlignBytes(Ceil(entriesAligned_, DST_REP_STRIDE), static_cast<int64_t>(1));

    // 公共 GM
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx);
    workspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace);
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        expertTokensGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokensCountOrCumsum);
    }

    ComputeCommonUbLayout();
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::ComputeCommonUbLayout()
{
    int64_t offset = 0;
    // 聚合搬出：UB 最前独立预留 aggrOutBufBytes_ 作为 gatherOutBuf（不挤占 xLocal）
    if (coutSortAggrEnable_ == 1 && aggrOutBufBytes_ > 0) {
        gatherOutBufOffset_ = offset;
        offset += AlignBytes(aggrOutBufBytes_, static_cast<int64_t>(sizeof(T)));
    }
    xLocalOffset_ = offset;
    int64_t xLocalBytes = AlignBytes(coreTokenNum_ * colsAligned_, static_cast<int64_t>(sizeof(T)));
    offset += xLocalBytes;

    expertIdxLocalOffset_ = offset;
    offset += AlignBytes(coreEntries_, static_cast<int64_t>(sizeof(int32_t)));

    scaleLocalOffset_ = offset;
    // per-token scale 全载缓冲仅非量化透传路径需要；动态量化的 (LE,H) smooth 走单行缓冲（子类）
    if (isInputScale_) {
        offset += AlignBytes(coreTokenNum_ * scaleSlotSize_, static_cast<int64_t>(sizeof(float)));
    }

    expertCountLocalOffset_ = offset;
    offset += AlignBytes(expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    allCoreExpertCountLocalOffset_ = offset;
    offset += AlignBytes(filterNeedCoreNum_ * expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    expertTokensLocalOffset_ = offset;
    // 非 KEY_VALUE 由 SeedsAndTokens VF 以 DIST_INTLV_B32 一次落盘（忽略 mask、写满 VL*2），预留须覆盖该宽度
    int64_t expertTokensUbBytes = AlignBytes(expertTokensBufElems_, static_cast<int64_t>(sizeof(int64_t)));
    if (expertTokensNumType_ != EXERPT_TOKENS_KEY_VALUE) {
        int64_t tokenIntlvBytes = AlignBytes(COUNT_SORT_HIST_BIN_NUM / 2, static_cast<int64_t>(sizeof(int32_t)));
        if (tokenIntlvBytes > expertTokensUbBytes) {
            expertTokensUbBytes = tokenIntlvBytes;
        }
    }
    offset += expertTokensUbBytes;

    prefixSumLocalOffset_ = offset;
    offset += AlignBytes(expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    seedSumLocalOffset_ = offset;
    offset += AlignBytes(1, static_cast<int64_t>(sizeof(int32_t)));

    filteredPairsLocalOffset_ = offset;
    // 按 64 元素粒度整批落盘、尾部会多写不足一批，容量须按 Ceil(maxFilteredCount_,64)*2*64 预留
    offset += AlignBytes(Ceil(maxFilteredCount_, COUNT_SORT_PAIRS_I32_REPEAT) * 2 * COUNT_SORT_PAIRS_I32_REPEAT,
                         static_cast<int64_t>(sizeof(int32_t)));

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

    // Histograms 按整 VL 读入，末尾不满的 VL 会多读至多 255B，故须额外留 1 个 VL 余量
    expertU8LocalOffset_ = offset;
    offset += AlignBytes(coreEntries_, static_cast<int64_t>(sizeof(uint8_t))) + COUNT_SORT_HIST_U8_REPEAT;

    histBinOffset_ = offset;
    offset += AlignBytes(COUNT_SORT_HIST_BIN_NUM, static_cast<int64_t>(sizeof(uint16_t)));

    histBinU32Offset_ = offset;
    offset += AlignBytes(Min(expertCountStride_, COUNT_SORT_HIST_BIN_NUM), static_cast<int64_t>(sizeof(uint32_t)));

    // 分桶区：两张桶表由 HistCountCumVF 以 DIST_INTLV_B32 一次刷 COUNT_SORT_HIST_BIN_NUM / 2 个 int32，
    // 故须同宽预留；消费侧只用前 actualExpertNum_ 项
    if (coutSortAggrEnable_ == 1) {
        bucketBaseOffset_ = offset;
        offset += AlignBytes(maxFilteredCount_, static_cast<int64_t>(sizeof(int32_t)));
        bucketOffsetTblOffset_ = offset;
        offset += AlignBytes(COUNT_SORT_HIST_BIN_NUM / 2, static_cast<int64_t>(sizeof(int32_t)));
        bucketCountTblOffset_ = offset;
        offset += AlignBytes(COUNT_SORT_HIST_BIN_NUM / 2, static_cast<int64_t>(sizeof(int32_t)));
    }
    commonBufSize_ = offset; // 公共区结尾
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::PrefillAndSync()
{
    if (rowIdxType_ == GATHER) {
        if (blockIdx_ < filterNeedCoreNum_) {
            GlobalTensor<int32_t> expandedRowIdxGmTmp = expandedRowIdxGm_[filterPerCoreTokens_ * k_ * blockIdx_];
            InitGlobalMemory(expandedRowIdxGmTmp, coreEntries_, static_cast<int32_t>(-1));
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
    }
}

// --- Phase A：载入 + 过滤计数 （Process 中同步点之前） ---
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::LoadExpertIdx()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(coreEntries_ * sizeof(int32_t)), 0, 0,
                                 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(expertIdxLocal, expertIdxGm_[coreFlatStart_], copyParams, padParams);
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::LoadScale()
{
    if (!isInputScale_) {
        return;
    }
    LocalTensor<float> scaleLocal = buf_.template Get<float>()[scaleLocalOffset_ / sizeof(float)];
    DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(coreTokenNum_), static_cast<uint32_t>(sizeof(float)), 0, 0,
                                      0};
    DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0};
    DataCopyPad(scaleLocal, scaleGm_[coreTokenStart_], scaleCopyParams, scalePadParams);
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::LoadXBackground(GlobalTensor<T> &xGm)
{
    LocalTensor<T> xLocal = buf_.Get<T>()[xLocalOffset_ / sizeof(T)];
    DataCopyExtParams copyParams{static_cast<uint16_t>(coreTokenNum_), static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0,
                                 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[coreTokenStart_ * cols_], copyParams, padParams);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::FilterAndCount()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    FilterAndCountHistogram();
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::FilterAndCountHistogram()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    // Cast->CompareScalar(GE/LT)->And->GatherMask 压缩合法专家，同时取原始下标 j
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

    // 窄化：- expertStart_ 归一到 [0, actualExpertNum_)，<= 32 无溢出，int32->uint8 一步精确转换
    Adds(gatheredExpertLocal, gatheredExpertLocal, static_cast<int32_t>(-expertStart_),
         static_cast<int32_t>(filteredInBatch));
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> expertU8Local = buf_.Get<uint8_t>()[expertU8LocalOffset_];
    Cast(expertU8Local, gatheredExpertLocal, RoundMode::CAST_NONE, static_cast<int32_t>(filteredInBatch));
    PipeBarrier<PIPE_V>();

    // 单桶计数上限受 uint16 限制，故按 COUNT_SORT_HIST_CHUNK 分块统计并扩成 32bit 累加
    LocalTensor<uint16_t> histBinLocal = buf_.Get<uint16_t>()[histBinOffset_ / sizeof(uint16_t)];
    LocalTensor<uint32_t> histBinU32Local = buf_.Get<uint32_t>()[histBinU32Offset_ / sizeof(uint32_t)];
    __ubuf__ uint8_t *expertU8Addr = (__ubuf__ uint8_t *)expertU8Local.GetPhyAddr();
    __ubuf__ uint16_t *histBinAddr = (__ubuf__ uint16_t *)histBinLocal.GetPhyAddr();
    int64_t histAddLen = Min(expertCountStride_, COUNT_SORT_HIST_BIN_NUM);
    for (int64_t chunkStart = 0; chunkStart < filteredInBatch; chunkStart += COUNT_SORT_HIST_CHUNK) {
        int64_t chunkElems = Min(filteredInBatch - chunkStart, COUNT_SORT_HIST_CHUNK);
        VF_CALL<MoeV3FullLoadHistogramVF>(expertU8Addr + chunkStart, histBinAddr, static_cast<uint32_t>(chunkElems));
        PipeBarrier<PIPE_V>();
        // Cast 只支持 uint16->uint32，故先扩成 uint32 再按位视图转 int32（频次恒非负，无歧义）
        Cast(histBinU32Local, histBinLocal, RoundMode::CAST_NONE, static_cast<int32_t>(histAddLen));
        PipeBarrier<PIPE_V>();
        Add(expertCountLocal, expertCountLocal, histBinU32Local.ReinterpretCast<int32_t>(),
            static_cast<int32_t>(histAddLen));
        PipeBarrier<PIPE_V>();
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

    // pairs 构建：两条向量交错写回，均非负 int32，按位交错故直接以 uint32_t 视图传入
    VF_CALL<MoeV3FullLoadPairsInterleaveVF>(
        (__ubuf__ uint32_t *)gatheredIdxLocal.GetPhyAddr(), (__ubuf__ uint32_t *)gatheredExpertLocal.GetPhyAddr(),
        (__ubuf__ uint32_t *)filteredPairsLocal.GetPhyAddr(), static_cast<uint32_t>(filteredInBatch));
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    filteredCount_ = filteredInBatch;
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::FilterAndCountVector()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

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

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

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
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::WriteExpertCountToWorkspace()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(expertCountStride_ * sizeof(int32_t)),
                                 0, 0, 0};
    DataCopyPad(workspaceGm_[blockIdx_ * expertCountStride_], expertCountLocal, copyParams);
}

// --- Phase B：跨核归约 + 偏移（SyncAll 之后） ---
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::ComputeGlobalOffset()
{
    LocalTensor<int32_t> allCoreExpertCountLocal =
        buf_.Get<int32_t>()[allCoreExpertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> prefixSumLocal = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    LoadAndReduceAllCoreExpertCount(allCoreExpertCountLocal, prefixSumLocal, expertCountLocal);

    int64_t cumulativeSum = ComputeSeedsAndExpertTokens(prefixSumLocal, expertCountLocal, actualExpertNum_);

    // 仅 SCATTER 需要 expertTotalCount_ 上界（Phase D 只展开已写满的 dst 前缀）；GATHER 保持 Init 初值
    if (rowIdxType_ == SCATTER) {
        expertTotalCount_ = cumulativeSum;
    }
}

// DCCI 基址读回各 filter 核写出的 expert count：先算核 [0, blockIdx_) 的前缀和，再把剩余行累加其上得到全核
// 总数（前缀和是总数的一部分，故只读剩余行，避免两次完整遍历）。累加器复用 expertCountLocal（其 Phase A
// 计数已被 WriteExpertCountToWorkspace 消费）。
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::LoadAndReduceAllCoreExpertCount(
    LocalTensor<int32_t> &allCoreExpertCountLocal, LocalTensor<int32_t> &prefixSumLocal,
    LocalTensor<int32_t> &expertCountLocal)
{
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workspaceGm_);

    int64_t totalWsElements = filterNeedCoreNum_ * expertCountStride_;
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(totalWsElements * sizeof(int32_t)), 0,
                                 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(allCoreExpertCountLocal, workspaceGm_[0], copyParams, padParams);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    // 高维切分 Add 按列累加：dst 与 src1 同址且 repStride = 0，src0 沿行推进，等价于逐行 Add 循环
    BinaryRepeatParams colReduceParams{
        1, 1, 1, 0, static_cast<uint8_t>(expertCountStride_ / COUNTING_SORT_ONE_BLOCK_ELEMENT), 0};

    // 1) 本核前缀和：核 [0, blockIdx_)。blockIdx_ == 0 时 repeatTime 为 0 属未定义行为，故跳过 Add
    Duplicate(prefixSumLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    if (blockIdx_ > 0) {
        Add(prefixSumLocal, allCoreExpertCountLocal, prefixSumLocal, static_cast<uint64_t>(expertCountStride_),
            static_cast<uint8_t>(blockIdx_), colReduceParams);
    }

    // 2) 全核总数：把剩余行 [blockIdx_, filterNeedCoreNum_) 累加到前缀和副本上（此处剩余行数 ≥ 1）
    Adds(expertCountLocal, prefixSumLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    int64_t remainCoreNum = filterNeedCoreNum_ - blockIdx_;
    if (remainCoreNum > 0) {
        Add(expertCountLocal, allCoreExpertCountLocal[static_cast<int32_t>(blockIdx_ * expertCountStride_)],
            expertCountLocal, static_cast<uint64_t>(expertCountStride_), static_cast<uint8_t>(remainCoreNum),
            colReduceParams);
    }
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
}

// core 0 写 expertTokens + 各核写 seed；expertCountLocal 就地覆盖为 seed，返回跨专家保留行总数
template <typename T>
__aicore__ inline int64_t MoeV3CountingSortFullLoadUnquantized<T>::ComputeSeedsAndExpertTokens(
    LocalTensor<int32_t> &prefixSumLocal, LocalTensor<int32_t> &expertCountLocal, int64_t expertNum)
{
    LocalTensor<int64_t> expertTokensLocal = buf_.Get<int64_t>()[expertTokensLocalOffset_ / sizeof(int64_t)];
    LocalTensor<int32_t> seedSumLocal = buf_.Get<int32_t>()[seedSumLocalOffset_ / sizeof(int32_t)];
    __ubuf__ int32_t *totalAddr = (__ubuf__ int32_t *)expertCountLocal.GetPhyAddr();
    __ubuf__ int32_t *prefixAddr = (__ubuf__ int32_t *)prefixSumLocal.GetPhyAddr();
    __ubuf__ int32_t *tokenAddr = (__ubuf__ int32_t *)expertTokensLocal.GetPhyAddr();
    __ubuf__ int32_t *sumAddr = (__ubuf__ int32_t *)seedSumLocal.GetPhyAddr();
    uint32_t vfExpertNum = static_cast<uint32_t>(expertNum);

    if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
        // 整块清零（含尾部哨兵 0,0）；与 VF 内的 Scatter 同在 PIPE_V 上按程序序执行，无需事件同步
        Duplicate(expertTokensLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
                  static_cast<int32_t>(expertTokensBufElems_ * EXPERT_ID_VALUE_NUM));
        PipeBarrier<PIPE_V>();
        VF_CALL<MoeV3FullLoadSeedsAndTokensKeyValueVF>(totalAddr, prefixAddr, totalAddr, tokenAddr, sumAddr,
                                                       vfExpertNum, static_cast<int32_t>(expertStart_));
    } else if (blockIdx_ == 0 && expertTokensNumFlag_ != EXERPT_TOKENS_NONE &&
               expertTokensNumType_ != EXERPT_TOKENS_COUNT) {
        VF_CALL<MoeV3FullLoadSeedsAndTokensCumSumVF>(totalAddr, prefixAddr, totalAddr, tokenAddr, sumAddr, vfExpertNum);
    } else {
        VF_CALL<MoeV3FullLoadSeedsAndTokensCountVF>(totalAddr, prefixAddr, totalAddr, tokenAddr, sumAddr, vfExpertNum);
    }
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    return static_cast<int64_t>(seedSumLocal.GetValue(0));
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::WriteExpertTokens()
{
    if (blockIdx_ != 0) {
        return;
    }
    LocalTensor<int64_t> expertTokensLocal = buf_.Get<int64_t>()[expertTokensLocalOffset_ / sizeof(int64_t)];
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    // COUNT/CUMSUM: expertCountElements_==actualExpertNum_；KEY_VALUE: 非零对*2+哨兵
    uint32_t expertCount = static_cast<uint32_t>(expertCountElements_);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(expertCount * sizeof(int64_t)), 0, 0,
                                 0};
    DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);
}

// --- Phase C：等待 x 加载 + 按专家搬出 ---
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::WaitXLoadCommon()
{
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::GatherAndWrite()
{
    if (coutSortAggrEnable_) {
        LocalTensor<T> xLocal = buf_.template Get<T>()[xLocalOffset_ / sizeof(T)];
        LocalTensor<float> scaleLocal =
            isInputScale_ ? buf_.template Get<float>()[scaleLocalOffset_ / sizeof(float)] : LocalTensor<float>();
        BucketByExpert();
        GatherAndWriteByExpert(xLocal, scaleLocal, expandedXGm_, expandedScaleGm_);
        return;
    }

    LocalTensor<int32_t> expertCountLocal = buf_.template Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.template Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    for (int64_t fIdx = 0; fIdx < filteredCount_; fIdx++) {
        int32_t localFlatIdx = filteredPairsLocal.GetValue(fIdx * 2);
        int32_t expertOffset = filteredPairsLocal.GetValue(fIdx * 2 + 1);

        int32_t slot = expertCountLocal.GetValue(expertOffset);
        expertCountLocal.SetValue(expertOffset, slot + 1);

        int64_t tokenRow = static_cast<int64_t>(localFlatIdx) / k_;
        int64_t origFlatIdx = coreFlatStart_ + static_cast<int64_t>(localFlatIdx);

        int64_t newPos = static_cast<int64_t>(slot);
        GatherOneRow(newPos, tokenRow, origFlatIdx);
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::BucketByExpert()
{
    LocalTensor<int32_t> bucketBase = buf_.Get<int32_t>()[bucketBaseOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> bucketOffsetTbl = buf_.Get<int32_t>()[bucketOffsetTblOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> bucketCountTbl = buf_.Get<int32_t>()[bucketCountTblOffset_ / sizeof(int32_t)];

    // 计数对象是 Phase A 已归一的紧凑 expertOffset，故直接复用「窄化 uint8 + Histograms」路径
    LocalTensor<int32_t> gatheredExpertLocal = buf_.Get<int32_t>()[gatheredExpertLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> gatheredIdxLocal = buf_.Get<int32_t>()[gatheredIdxLocalOffset_ / sizeof(int32_t)];
    LocalTensor<uint8_t> expertU8Local = buf_.Get<uint8_t>()[expertU8LocalOffset_];

    Cast(expertU8Local, gatheredExpertLocal, RoundMode::CAST_NONE, static_cast<int32_t>(filteredCount_));
    PipeBarrier<PIPE_V>();

    VF_CALL<MoeV3FullLoadHistCountCumVF>(
        (__ubuf__ uint8_t *)expertU8Local.GetPhyAddr(), static_cast<uint32_t>(filteredCount_),
        (__ubuf__ uint32_t *)bucketCountTbl.GetPhyAddr(), (__ubuf__ uint32_t *)bucketOffsetTbl.GetPhyAddr());
    PipeBarrier<PIPE_V>();

    // 独占式前缀和 = 包含式前缀和 - 自身频次 = 各桶起始偏移（和 = filteredCount_，不越界）
    Sub(bucketOffsetTbl, bucketOffsetTbl, bucketCountTbl, static_cast<int32_t>(actualExpertNum_));

    // 第三遍（VF）：k 循环整体收进 VF，调用侧只发一次 VF_CALL，直接把 expert == k 的 localFlatIdx 按原 i 序
    // 稳定写入桶；不再走 filteredPairsLocal 交错布局，省掉 2N 次标量读、N 次 SetValue 与 N 个 PipeBarrier。
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S); // 前缀和/频次表（V 结果）先对 VF 内的标量读可见
    VF_CALL<MoeV3FullLoadBucketCompactStoreVF>(
        (__ubuf__ int32_t *)gatheredIdxLocal.GetPhyAddr(), (__ubuf__ int32_t *)gatheredExpertLocal.GetPhyAddr(),
        (__ubuf__ int32_t *)bucketBase.GetPhyAddr(), (__ubuf__ int32_t *)bucketOffsetTbl.GetPhyAddr(),
        static_cast<uint32_t>(actualExpertNum_), static_cast<uint32_t>(filteredCount_));
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S); // VF 回写的桶数据供 GatherAndWriteByExpert 标量读
}

// 聚合搬出主循环：外层 expertOffset，内层按 k 切批，每批一次 DataCopyPad；rowIdx/scale 逐行穿插
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::GatherAndWriteByExpert(
    LocalTensor<T> &xLocal, LocalTensor<float> &scaleLocal, GlobalTensor<T> &expandedXGmRef,
    GlobalTensor<float> &expandedScaleGmRef)
{
    LocalTensor<int32_t> bucketBase = buf_.Get<int32_t>()[bucketBaseOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> bucketOffsetTbl = buf_.Get<int32_t>()[bucketOffsetTblOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> bucketCountTbl = buf_.Get<int32_t>()[bucketCountTblOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> idxBuf = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    LocalTensor<T> gatherOutBuf = buf_.Get<T>()[gatherOutBufOffset_ / sizeof(T)];

    for (int64_t eo = 0; eo < actualExpertNum_; eo++) {
        int64_t count_e = static_cast<int64_t>(bucketCountTbl.GetValue(eo));
        if (count_e == 0) {
            continue;
        }
        int64_t expertSeed = static_cast<int64_t>(expertCountLocal.GetValue(eo));
        int64_t baseOff = static_cast<int64_t>(bucketOffsetTbl.GetValue(eo));
        int64_t slotStart = 0;
        int64_t remaining = count_e;

        while (remaining > 0) {
            int64_t batchRows = Min(aggrOutRows_, remaining);
            int64_t batchStartNewPos = expertSeed + slotStart;
            int64_t origBaseOff = baseOff + slotStart;

            for (int64_t r = 0; r < batchRows; r++) {
                int32_t localFlatIdx = bucketBase.GetValue(origBaseOff + r);
                int64_t tokenRow = static_cast<int64_t>(localFlatIdx) / k_;
                int64_t origFlatIdx = coreFlatStart_ + static_cast<int64_t>(localFlatIdx);
                int64_t newPos = batchStartNewPos + r;

                CopyGatherRowToOutBuf(gatherOutBuf, xLocal, r, tokenRow);
                if (newPos < outputRows_) {
                    WriteRowIdxAndScale(idxBuf, scaleLocal, expandedScaleGmRef, newPos, origFlatIdx, tokenRow);
                }
            }

            FlushBatchToGm(expandedXGmRef, gatherOutBuf, batchStartNewPos, batchRows);

            slotStart += batchRows;
            remaining -= batchRows;
        }
    }
}

// 单行 UB→UB 拷贝：非 32B 对齐时 DataCopy 会丢失尾部元素，故改用 VF Copy（内部 mask 处理尾部）
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::CopyGatherRowToOutBuf(LocalTensor<T> &gatherOutBuf,
                                                                                      LocalTensor<T> &xLocal,
                                                                                      int64_t dstRow, int64_t tokenRow)
{
    if ((cols_ * static_cast<int64_t>(sizeof(T))) % BLOCK_BYTES == 0) {
        DataCopy(gatherOutBuf[dstRow * colsAligned_], xLocal[tokenRow * colsAligned_], static_cast<int32_t>(cols_));
    } else {
        Copy(gatherOutBuf[dstRow * colsAligned_], xLocal[tokenRow * colsAligned_], static_cast<uint32_t>(cols_));
    }
}

// rowIdx 写（GATHER: [origFlatIdx]=newPos；SCATTER: [newPos]=origFlatIdx）+ scale 逐行写（isInputScale_ 时）
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::WriteRowIdxAndScale(
    LocalTensor<int32_t> &idxBuf, LocalTensor<float> &scaleLocal, GlobalTensor<float> &expandedScaleGmRef,
    int64_t newPos, int64_t origFlatIdx, int64_t tokenRow)
{
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
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

    if (isInputScale_) {
        DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(expandedScaleGmRef[newPos], scaleLocal[tokenRow * scaleSlotSize_], scaleCopyParams);
    }
}

// 一次 DataCopyPad 写连续 GM 段（3 参版：GM, Local, extParams）
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::FlushBatchToGm(GlobalTensor<T> &expandedXGmRef,
                                                                               LocalTensor<T> &gatherOutBuf,
                                                                               int64_t batchStartNewPos,
                                                                               int64_t batchRows)
{
    if (batchStartNewPos >= outputRows_) {
        return;
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyExtParams xCopyParams{static_cast<uint16_t>(batchRows), static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(expandedXGmRef[batchStartNewPos * cols_], gatherOutBuf, xCopyParams);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::GatherOneRow(int64_t newPos, int64_t tokenRow,
                                                                             int64_t origFlatIdx)
{
    LocalTensor<T> xLocal = buf_.template Get<T>()[xLocalOffset_ / sizeof(T)];
    LocalTensor<float> scaleLocal =
        isInputScale_ ? buf_.template Get<float>()[scaleLocalOffset_ / sizeof(float)] : LocalTensor<float>();

    DataCopyExtParams xCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    LocalTensor<int32_t> idxBuf = buf_.template Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
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
        DataCopyPad(expandedXGm_[static_cast<int64_t>(newPos) * cols_], xLocal[tokenRow * colsAligned_], xCopyParams);
        if (isInputScale_) {
            DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(expandedScaleGm_[newPos], scaleLocal[tokenRow * scaleSlotSize_], scaleCopyParams);
        }
    }
}

// --- Phase D：topkWeight 重排输出 ---
// SIMT 直读 GM 的 map：scatter 按 dst 连续段展开写，各核互斥覆盖前缀；gather 按 src 遍历 + dst<0 跳过
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::TopkWeightPhase()
{
    // DCCI 使 SyncAll 后各核新写的 map 对 SIMT __gm__ 直读可见（同 ComputeGlobalOffset）
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expandedRowIdxGm_);

    if (rowIdxType_ == SCATTER) {
        int64_t dstRows = Min(outputRows_, expertTotalCount_);
        int64_t perCoreRows = Ceil(dstRows, filterNeedCoreNum_);
        int64_t dstStart = blockIdx_ * perCoreRows;
        int64_t dstEnd = Min(dstStart + perCoreRows, dstRows);
        if (dstStart >= dstEnd) {
            return;
        }
        int64_t segmentRows = dstEnd - dstStart;
        uint32_t threadNum = static_cast<uint32_t>(Min(segmentRows, static_cast<int64_t>(COUNT_SOURT_SIMT_THREAD_NUM)));
        asc_vf_call<CoutSortFullLoadTopkWeightScatterSimt>(dim3{threadNum, 1, 1}, segmentRows, dstStart, totalLength_,
                                                           (__gm__ int32_t *)expandedRowIdxGm_.GetPhyAddr(),
                                                           (__gm__ float *)topkWeightGm_.GetPhyAddr(),
                                                           (__gm__ volatile float *)expandedTopkWeightGm_.GetPhyAddr());
        return;
    }

    // gather：expandedRowIdx 为 src->dst map（无效 id 预填 -1），各核只扫 Phase C 的范围
    uint32_t threadNum = static_cast<uint32_t>(Min(coreEntries_, static_cast<int64_t>(COUNT_SOURT_SIMT_THREAD_NUM)));
    asc_vf_call<CoutSortFullLoadTopkWeightGatherSimt>(dim3{threadNum, 1, 1}, coreEntries_, coreFlatStart_, outputRows_,
                                                      (__gm__ int32_t *)expandedRowIdxGm_.GetPhyAddr(),
                                                      (__gm__ float *)topkWeightGm_.GetPhyAddr(),
                                                      (__gm__ volatile float *)expandedTopkWeightGm_.GetPhyAddr());
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_FULL_LOAD_UNQUANTIZED_H
