/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_pa_kv_cache_nd.h
 * \brief
 */

#ifndef GATHER_PA_KV_CACHE_ND_H_
#define GATHER_PA_KV_CACHE_ND_H_

#include "kernel_operator.h"

namespace GatherPaKvCacheV35 {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t DOUBLE_BUFFER = 2;

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
class GatherPaKvCacheNd {
public:
    __aicore__ inline GatherPaKvCacheNd(TPipe *pipe, const GatherPaKvCacheTilingDataV35 *__restrict tiling)
        : pipe_(pipe), tl_(tiling){};
    __aicore__ inline void Init(GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens,
                                GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset, GM_ADDR key_out,
                                GM_ADDR value_out);
    __aicore__ inline void Process();
    __aicore__ inline void InitParams();
    __aicore__ inline void GatherKvCache(GlobalTensor<T> dstCacheGm, GlobalTensor<T> srcCacheGm, uint64_t curLen,
                                         bool isFilledWithZero);
    __aicore__ inline void GatherKvCacheNonContiguous(
        GlobalTensor<T> dstGm, GlobalTensor<T> srcGm,
        uint64_t curBlockSize, uint64_t hiddenSize,
        int64_t stride0, int64_t stride1, int64_t stride2,
        bool isSlotNonContig, bool isHeadNonContig,
        bool isFilledWithZero, bool isKey);
    // 把一段连续字节经UB搬到目标, 按maxUbHiddenSize_自动切块, 保证UB不越界。
    // 所有偏移/长度均为字节粒度。isFilledWithZero时不读src, 直接填零写出。
    __aicore__ inline void GatherNcContigRun(GlobalTensor<T> dstGm, GlobalTensor<T> srcGm,
                                             uint64_t dstByteOff, uint64_t srcByteOff,
                                             uint64_t totalBytes, bool isFilledWithZero);
    __aicore__ inline T_INDEX CalcKvCoreOffset(int64_t reduceLen);

private:
private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> cacheQueue_;
    TQue<QuePosition::VECIN, 1> seqLensQueue_;
    TBuf<TPosition::VECCALC> prefixSumBuffer_;

    DataCopyPadExtParams<T> padExtParams_;

    const GatherPaKvCacheTilingDataV35 *tl_;

    GlobalTensor<T> keyCacheGm_;
    GlobalTensor<T> valueCacheGm_;
    GlobalTensor<T_INDEX> blockTablesGm_;
    GlobalTensor<T_INDEX> seqLensGm_;
    GlobalTensor<T_INDEX> seqOffsetGm_;
    GlobalTensor<T> outKeyGm_;
    GlobalTensor<T> outValueGm_;

    uint32_t batchPerCore_;
    uint32_t needCoreNum_;
    int64_t batchCount_;
    uint32_t seqLenAccSize_;
    int64_t cacheBlockSize_;
    int64_t blockTableWidth_;
    uint64_t maxUbHiddenSizeK_;
    uint64_t maxUbHiddenSizeV_;
    uint64_t maxUbHiddenSize_;
    int64_t numBlocks_;
    uint64_t hiddenSizeK_;
    uint64_t hiddenSizeV_;
    int64_t numTokens_;

    // 非连续支持
    uint32_t nonContiguousFlag_;
    int64_t kCacheStride0_;
    int64_t kCacheStride1_;
    int64_t kCacheStride2_;
    int64_t vCacheStride0_;
    int64_t vCacheStride1_;
    int64_t vCacheStride2_;
    int64_t keyOutStride0_;
    int64_t keyOutStride1_;
    int64_t valueOutStride0_;
    int64_t valueOutStride1_;
    uint64_t numHeadsK_;
    uint64_t numHeadsV_;
};

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::Init(
    GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens, GM_ADDR key_in, GM_ADDR value_in,
    GM_ADDR seq_offset, GM_ADDR key_out, GM_ADDR value_out)
{
    InitParams();

    keyCacheGm_.SetGlobalBuffer((__gm__ T *)(key_cache));
    valueCacheGm_.SetGlobalBuffer((__gm__ T *)(value_cache));
    blockTablesGm_.SetGlobalBuffer((__gm__ T_INDEX *)block_tables);
    seqLensGm_.SetGlobalBuffer((__gm__ T_INDEX *)(seq_lens));
    seqOffsetGm_.SetGlobalBuffer((__gm__ T_INDEX *)(seq_offset));

    outKeyGm_.SetGlobalBuffer((__gm__ T *)(key_out));
    outValueGm_.SetGlobalBuffer((__gm__ T *)(value_out));

    pipe_->InitBuffer(cacheQueue_, DOUBLE_BUFFER, (maxUbHiddenSize_) * sizeof(T));
    pipe_->InitBuffer(seqLensQueue_, DOUBLE_BUFFER, (seqLenAccSize_) * sizeof(T_INDEX));
    pipe_->InitBuffer(prefixSumBuffer_, BLOCK_SIZE);
}

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::InitParams()
{
    cacheBlockSize_ = tl_->kvCacheBlockSize;
    batchPerCore_ = tl_->batchPerCore;
    needCoreNum_ = tl_->needCoreNum;
    batchCount_ = tl_->batchCount;
    blockTableWidth_ = tl_->blockTableWidth;
    // UB放得下的kv Cache Block大小
    maxUbHiddenSizeK_ = tl_->maxUbHiddenSizeK * sizeof(DTYPE_KEY);
    maxUbHiddenSizeV_ = tl_->maxUbHiddenSizeV * sizeof(DTYPE_VALUE);
    maxUbHiddenSize_ = tl_->maxUbHiddenSize * sizeof(DTYPE_KEY);
    seqLenAccSize_ = tl_->seqLenAccumSize;
    numBlocks_ = tl_->numBlocks;
    hiddenSizeK_ = tl_->hiddenSizeK * sizeof(DTYPE_KEY);
    hiddenSizeV_ = tl_->hiddenSizeV * sizeof(DTYPE_VALUE);
    numTokens_ = tl_->numTokens;

    // 非连续支持
    nonContiguousFlag_ = tl_->nonContiguousFlag;
    kCacheStride0_ = tl_->kCacheStride0;
    kCacheStride1_ = tl_->kCacheStride1;
    kCacheStride2_ = tl_->kCacheStride2;
    vCacheStride0_ = tl_->vCacheStride0;
    vCacheStride1_ = tl_->vCacheStride1;
    vCacheStride2_ = tl_->vCacheStride2;
    keyOutStride0_ = tl_->keyOutStride0;
    keyOutStride1_ = tl_->keyOutStride1;
    valueOutStride0_ = tl_->valueOutStride0;
    valueOutStride1_ = tl_->valueOutStride1;
    numHeadsK_ = tl_->numHeadsK;
    numHeadsV_ = tl_->numHeadsV;

    padExtParams_.isPad = false;
    padExtParams_.leftPadding = 0;
    padExtParams_.rightPadding = 0;
    padExtParams_.paddingValue = 0;
}

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::Process()
{
    int64_t batchStart = GetBlockIdx() * batchPerCore_;
    int64_t batchEnd = batchStart + batchPerCore_;
    if (GetBlockIdx() == needCoreNum_ - 1) {
        batchEnd = batchCount_;
    }

    // 如果isSeqLensCumsum为false，在此处计算前缀和，即每个核的偏移
    int64_t coreOffset;
    if constexpr (isSeqLensCumsum) {
        coreOffset = seqLensGm_.GetValue(batchStart);
    } else {
        coreOffset = CalcKvCoreOffset(batchStart);
    }

    for (uint32_t i = batchStart; i < batchEnd; i++) {
        // 读取cache的数量
        // 累加模式
        T_INDEX seqLen;
        int64_t batchOffset, accumSeqLen;
        if constexpr (isSeqLensCumsum) {
            // 当前batch对应的seqLen
            seqLen = seqLensGm_.GetValue(i + 1) - seqLensGm_.GetValue(i);
            batchOffset = seqLensGm_.GetValue(i);
            accumSeqLen = seqLensGm_.GetValue(i + 1);
        } else {
            seqLen = seqLensGm_.GetValue(i);
            batchOffset = coreOffset;
            accumSeqLen = coreOffset + seqLensGm_.GetValue(i);
            coreOffset = accumSeqLen;
        }

        if (batchOffset >= numTokens_) {
            break;
        }

        // 如果numTokens小于seqLen总和，尾块需要截断
        if (numTokens_ > batchOffset && numTokens_ <= accumSeqLen) {
            seqLen = numTokens_ - batchOffset;
        }

        // block起点 + 偏移
        uint64_t seqOffset = 0;
        if constexpr (hasSeqOffset) {
            seqOffset = seqOffsetGm_.GetValue(i) / cacheBlockSize_; // 除以blocksize表示在block_tables中的偏移
        }

        // 当前batch关联到几个block
        uint64_t blockCount = CeilDivision(seqLen, cacheBlockSize_);
        uint64_t keyOffset = batchOffset * hiddenSizeK_;
        uint64_t valueOffset = batchOffset * hiddenSizeV_;
        // 对blockIdx（用来从block_tables中取blockId）循环
        for (uint32_t blockIdx = 0; blockIdx < blockCount; blockIdx++) {
            uint64_t blockTableOffset = blockIdx + seqOffset;
            bool isFilledWithZero;
            int64_t blockId;
            if ((blockTableOffset >= blockTableWidth_) || (blockTableOffset < 0)) {
                isFilledWithZero = true;
                blockId = 0;
            } else {
                isFilledWithZero = false;
                blockId = blockTablesGm_.GetValue(blockTableWidth_ * i + blockTableOffset);
                // blockId数值越界时填零, 与golden对齐(非连续view场景numBlocks_为view dim0,
                // 而block_tables中的blockId按storage范围取值, 可能 >= numBlocks_导致GM访存越界)。
                if (blockId >= numBlocks_ || blockId < 0) {
                    isFilledWithZero = true;
                    blockId = 0;
                }
            }

            uint64_t curBlockSize = cacheBlockSize_;
            if (blockIdx == blockCount - 1) {
                curBlockSize = seqLen - (blockCount - 1) * cacheBlockSize_; // 尾块处理
            }

            if (nonContiguousFlag_ == 0) {
                // ===== 原有连续路径 =====
                uint64_t keyCacheStart = blockId * cacheBlockSize_ * hiddenSizeK_;
                uint64_t valueCacheStart = blockId * cacheBlockSize_ * hiddenSizeV_;
                // key
                uint32_t fracBlockCount = CeilDivision(curBlockSize * hiddenSizeK_, maxUbHiddenSizeK_);
                for (uint32_t fracBlockId = 0; fracBlockId < fracBlockCount; fracBlockId++) {
                    uint64_t curFracBlockLen = maxUbHiddenSizeK_;
                    if (fracBlockId == fracBlockCount - 1) {
                        curFracBlockLen =
                            curBlockSize * hiddenSizeK_ - (fracBlockCount - 1) * maxUbHiddenSizeK_;
                    }
                    uint64_t keyCacheOffset = keyCacheStart + fracBlockId * maxUbHiddenSizeK_;
                    GatherKvCache(outKeyGm_[keyOffset], keyCacheGm_[keyCacheOffset], curFracBlockLen, isFilledWithZero);
                    keyOffset += curFracBlockLen;
                }
                // value
                fracBlockCount = CeilDivision(curBlockSize * hiddenSizeV_, maxUbHiddenSizeV_);
                for (uint32_t fracBlockId = 0; fracBlockId < fracBlockCount; fracBlockId++) {
                    uint64_t curFracBlockLen = maxUbHiddenSizeV_;
                    if (fracBlockId == fracBlockCount - 1) {
                        curFracBlockLen =
                            curBlockSize * hiddenSizeV_ - (fracBlockCount - 1) * maxUbHiddenSizeV_;
                    }
                    uint64_t valueCacheOffset = valueCacheStart + fracBlockId * maxUbHiddenSizeV_;
                    GatherKvCache(outValueGm_[valueOffset], valueCacheGm_[valueCacheOffset], curFracBlockLen,
                                  isFilledWithZero);
                    valueOffset += curFracBlockLen;
                }
            } else {
                // ===== 非连续路径 =====
                // 统一用stride计算block起始地址（连续时stride已填充连续值）
                // 注意: kernel模板T恒为uint8_t(规避B8编译问题), sizeof(T)==1,
                // 而stride为元素粒度, GlobalTensor<uint8_t>按字节索引,
                // 故必须乘以真实元素字节宽(sizeof(DTYPE_KEY)/DTYPE_VALUE)将元素偏移转为字节偏移。
                uint64_t keyCacheStart = blockId * kCacheStride0_ * sizeof(DTYPE_KEY);
                uint64_t valueCacheStart = blockId * vCacheStride0_ * sizeof(DTYPE_VALUE);
                bool isKSlotNC = (nonContiguousFlag_ >> 4) & 1;
                bool isKHeadNC = (nonContiguousFlag_ >> 5) & 1;
                bool isVSlotNC = (nonContiguousFlag_ >> 6) & 1;
                bool isVHeadNC = (nonContiguousFlag_ >> 7) & 1;

                if (!isKSlotNC) {
                    // Case A: 仅blockNum轴非连续, block内连续
                    // 复用原有GatherKvCache, 只是起始地址用stride计算
                    uint32_t fracBlockCount = CeilDivision(curBlockSize * hiddenSizeK_, maxUbHiddenSizeK_);
                    for (uint32_t fracBlockId = 0; fracBlockId < fracBlockCount; fracBlockId++) {
                        uint64_t curFracBlockLen = maxUbHiddenSizeK_;
                        if (fracBlockId == fracBlockCount - 1) {
                            curFracBlockLen =
                                curBlockSize * hiddenSizeK_ - (fracBlockCount - 1) * maxUbHiddenSizeK_;
                        }
                        uint64_t keyCacheOffset = keyCacheStart + fracBlockId * maxUbHiddenSizeK_;
                        GatherKvCache(outKeyGm_[keyOffset], keyCacheGm_[keyCacheOffset],
                                      curFracBlockLen, isFilledWithZero);
                        keyOffset += curFracBlockLen;
                    }
                } else {
                    // Case B/C: block内非连续, 用loop模式搬运
                    GatherKvCacheNonContiguous(
                        outKeyGm_[keyOffset], keyCacheGm_[keyCacheStart],
                        curBlockSize, hiddenSizeK_,
                        kCacheStride0_, kCacheStride1_, kCacheStride2_,
                        isKSlotNC, isKHeadNC, isFilledWithZero, true);
                    keyOffset += curBlockSize * hiddenSizeK_;
                }

                if (!isVSlotNC) {
                    uint32_t fracBlockCount = CeilDivision(curBlockSize * hiddenSizeV_, maxUbHiddenSizeV_);
                    for (uint32_t fracBlockId = 0; fracBlockId < fracBlockCount; fracBlockId++) {
                        uint64_t curFracBlockLen = maxUbHiddenSizeV_;
                        if (fracBlockId == fracBlockCount - 1) {
                            curFracBlockLen =
                                curBlockSize * hiddenSizeV_ - (fracBlockCount - 1) * maxUbHiddenSizeV_;
                        }
                        uint64_t valueCacheOffset = valueCacheStart + fracBlockId * maxUbHiddenSizeV_;
                        GatherKvCache(outValueGm_[valueOffset], valueCacheGm_[valueCacheOffset], curFracBlockLen,
                                      isFilledWithZero);
                        valueOffset += curFracBlockLen;
                    }
                } else {
                    GatherKvCacheNonContiguous(
                        outValueGm_[valueOffset], valueCacheGm_[valueCacheStart],
                        curBlockSize, hiddenSizeV_,
                        vCacheStride0_, vCacheStride1_, vCacheStride2_,
                        isVSlotNC, isVHeadNC, isFilledWithZero, false);
                    valueOffset += curBlockSize * hiddenSizeV_;
                }
            }
        }
    }
}

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline T_INDEX
GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::CalcKvCoreOffset(int64_t reduceLen)
{
    LocalTensor<T_INDEX> prefixSumLocal = prefixSumBuffer_.Get<T_INDEX>();
    uint64_t loopTimes = CeilDivision(reduceLen, seqLenAccSize_);
    DataCopyPadExtParams<T_INDEX> padParams = {false, 0, 0, 0};
    DataCopyExtParams seqLensCopyParams;
    seqLensCopyParams.blockCount = 1;
    seqLensCopyParams.srcStride = 0;
    seqLensCopyParams.dstStride = 0; // 通用设置
    uint32_t seqOffset, seqLength;

    T_INDEX coreOffset = 0;
    for (uint64_t i = 0; i < loopTimes; i++) {
        seqOffset = i * seqLenAccSize_;
        seqLength = seqLenAccSize_;
        if (i == loopTimes - 1) {
            seqLength = reduceLen - (loopTimes - 1) * seqLenAccSize_;
        }
        seqLensCopyParams.blockLen = static_cast<uint32_t>(seqLength * sizeof(T_INDEX));
        LocalTensor<T_INDEX> seqLenLocal = seqLensQueue_.AllocTensor<T_INDEX>();
        DataCopyPad(seqLenLocal, seqLensGm_[seqOffset], seqLensCopyParams, padParams); // 把数据放入UB
        seqLensQueue_.EnQue<T_INDEX>(seqLenLocal);

        seqLenLocal = seqLensQueue_.DeQue<T_INDEX>();
        uint32_t srcShape[2] = {uint32_t(1), seqLength};
        AscendC::ReduceSum<T_INDEX, Pattern::Reduce::AR, true>(prefixSumLocal, seqLenLocal, srcShape, false);
        AscendC::TEventID eventIdVecToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventIdVecToS);
        WaitFlag<HardEvent::V_S>(eventIdVecToS);
        seqLensQueue_.FreeTensor<T_INDEX>(seqLenLocal);
        coreOffset += prefixSumLocal.GetValue(0);
    }

    return coreOffset;
}

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::GatherKvCache(
    GlobalTensor<T> dstCacheGm, GlobalTensor<T> srcCacheGm, uint64_t curLen, bool isFilledWithZero)
{
    LocalTensor<T> cacheLocal = cacheQueue_.AllocTensor<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = curLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    if (isFilledWithZero) {
        AscendC::TEventID eventIdMTE3ToVec = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
        Duplicate<T>(cacheLocal, 0, curLen);
        AscendC::TEventID eventIdVecToMTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);
    } else {
        DataCopyPad<T, PaddingMode::Normal>(cacheLocal, srcCacheGm, dataCopyParams, padExtParams_);
        cacheQueue_.EnQue<T>(cacheLocal);
        cacheLocal = cacheQueue_.DeQue<T>();
    }

    DataCopyPad<T, PaddingMode::Normal>(dstCacheGm, cacheLocal, dataCopyParams);
    event_t eventIdMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
    cacheQueue_.FreeTensor(cacheLocal);
}

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::GatherKvCacheNonContiguous(
    GlobalTensor<T> dstGm, GlobalTensor<T> srcGm,
    uint64_t curBlockSize, uint64_t hiddenSize,
    int64_t stride0, int64_t stride1, int64_t stride2,
    bool isSlotNonContig, bool isHeadNonContig,
    bool isFilledWithZero, bool isKey)
{
    // T恒为uint8_t, sizeof(T)==1; stride为元素粒度, 须乘真实元素字节宽转字节偏移。
    // hiddenSize入参已是字节(InitParams已乘sizeof(DTYPE_*))。
    (void)stride0;          // block起始地址已在调用侧用stride0计算
    (void)isSlotNonContig;  // 进入本函数即slot非连续, 标志无需再用
    const uint64_t elemSize = isKey ? sizeof(DTYPE_KEY) : sizeof(DTYPE_VALUE);
    const uint64_t slotStrideBytes = static_cast<uint64_t>(stride1) * elemSize;  // slot间源步长(字节)
    const uint64_t headStrideBytes = static_cast<uint64_t>(stride2) * elemSize;  // head间源步长(字节)

    // 输出侧: 判断是否非连续写回, 计算token(slot)间目标步长(字节)
    bool isOutNonContig = isKey ? ((nonContiguousFlag_ >> 2) & 1) : ((nonContiguousFlag_ >> 3) & 1);
    int64_t outStride1 = isKey ? keyOutStride1_ : valueOutStride1_;
    const uint64_t dstSlotStrideBytes = isOutNonContig
                                            ? static_cast<uint64_t>(outStride1) * elemSize  // 输出非连续: token间有间隙
                                            : hiddenSize;                                    // 输出连续: 紧密排列

    uint64_t numHeads = isKey ? numHeadsK_ : numHeadsV_;
    uint64_t headBytes = (numHeads > 0) ? (hiddenSize / numHeads) : hiddenSize;  // 单head字节数

    for (uint64_t slot = 0; slot < curBlockSize; slot++) {
        uint64_t dstByteOff = slot * dstSlotStrideBytes;

        if (isFilledWithZero) {
            // 越界block: 整slot填零写出
            GatherNcContigRun(dstGm, srcGm, dstByteOff, 0, hiddenSize, true);
        } else if (!isHeadNonContig) {
            // Case B: slot内连续(N*D字节连续), 整slot一次性按UB切块搬运
            uint64_t srcByteOff = slot * slotStrideBytes;
            GatherNcContigRun(dstGm, srcGm, dstByteOff, srcByteOff, hiddenSize, false);
        } else {
            // Case C: head轴也非连续, 逐head搬运(每个head内连续)
            for (uint64_t h = 0; h < numHeads; h++) {
                uint64_t srcByteOff = slot * slotStrideBytes + h * headStrideBytes;
                uint64_t dstHeadOff = dstByteOff + h * headBytes;
                GatherNcContigRun(dstGm, srcGm, dstHeadOff, srcByteOff, headBytes, false);
            }
        }
    }
}

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::GatherNcContigRun(
    GlobalTensor<T> dstGm, GlobalTensor<T> srcGm, uint64_t dstByteOff, uint64_t srcByteOff,
    uint64_t totalBytes, bool isFilledWithZero)
{
    // 按UB容量(maxUbHiddenSize_字节)切块, 逐块搬运, 保证UB不越界。
    uint64_t handled = 0;
    while (handled < totalBytes) {
        uint64_t curBytes = totalBytes - handled;
        if (curBytes > maxUbHiddenSize_) {
            curBytes = maxUbHiddenSize_;
        }

        LocalTensor<T> cacheLocal = cacheQueue_.AllocTensor<T>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(curBytes);
        params.srcStride = 0;
        params.dstStride = 0;

        if (isFilledWithZero) {
            AscendC::TEventID eventIdMTE3ToVec = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
            WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
            Duplicate<T>(cacheLocal, 0, curBytes);
            AscendC::TEventID eventIdVecToMTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);
        } else {
            DataCopyPad<T, PaddingMode::Normal>(cacheLocal, srcGm[srcByteOff + handled], params, padExtParams_);
            cacheQueue_.EnQue<T>(cacheLocal);
            cacheLocal = cacheQueue_.DeQue<T>();
        }

        DataCopyPad<T, PaddingMode::Normal>(dstGm[dstByteOff + handled], cacheLocal, params);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        cacheQueue_.FreeTensor(cacheLocal);

        handled += curBytes;
    }
}

} // namespace GatherPaKvCacheV35
#endif