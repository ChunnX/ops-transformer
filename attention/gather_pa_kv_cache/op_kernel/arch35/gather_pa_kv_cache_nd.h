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
 * \brief SIMT kernel for gather_pa_kv_cache ND mode.
 *        Replaces SIMD DataCopyPad/Duplicate/ReduceSum with __simt_vf__ grid-stride
 *        copy kernels launched via asc_vf_call. No TQue/TBuf/UB needed — VF threads
 *        read/write GM directly.
 */

#ifndef GATHER_PA_KV_CACHE_ND_H_
#define GATHER_PA_KV_CACHE_ND_H_

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace GatherPaKvCacheV35 {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint64_t MAX_BLOCK_CNT = 4095;  // DataCopyPad blockCount 上限 (保留供参考)
constexpr uint32_t THREAD_NUM = 1024;     // VF 线程数，编译期常量

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
class GatherPaKvCacheNd {
public:
    __aicore__ inline GatherPaKvCacheNd(TPipe *pipe, const GatherPaKvCacheTilingDataV35 *__restrict tiling)
        : tl_(tiling) { (void)pipe; };
    __aicore__ inline void Init(GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens,
                                GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset, GM_ADDR key_out,
                                GM_ADDR value_out);
    __aicore__ inline void Process();
    __aicore__ inline void InitParams();
    __aicore__ inline T_INDEX CalcKvCoreOffset(int64_t reduceLen);

private:
    // ===== SIMT VF kernels =====
    // 连续字节拷贝(或填零): 每个线程拷贝一个字节, grid-stride 遍历 [0, count)。
    static __simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void GatherCopyContigVF(
        uint64_t dstOff, uint64_t srcOff, uint64_t count,
        __gm__ T *dstGm, __gm__ T *srcGm, uint32_t isFilledWithZero)
    {
        for (uint64_t idx = static_cast<uint64_t>(threadIdx.x);
             idx < count;
             idx += static_cast<uint64_t>(blockDim.x)) {
            if (isFilledWithZero) {
                dstGm[dstOff + idx] = static_cast<T>(0);
            } else {
                dstGm[dstOff + idx] = srcGm[srcOff + idx];
            }
        }
    }

    // 跨步 slot 拷贝(或填零): totalElems = numSlots * unitBytes, 每个线程处理一个字节,
    // 通过 Simt::UintDiv 快除定位 slot 和 slot 内偏移, 按 src/dst slot stride 跳搬。
    static __simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void GatherCopyStridedVF(
        uint64_t totalElems, uint64_t unitBytes,
        uint64_t dstBaseOff, uint64_t srcBaseOff,
        uint64_t dstSlotStride, uint64_t srcSlotStride,
        __gm__ T *dstGm, __gm__ T *srcGm,
        uint32_t isFilledWithZero, uint64_t magic, uint64_t shift)
    {
        for (uint64_t idx = static_cast<uint64_t>(threadIdx.x);
             idx < totalElems;
             idx += static_cast<uint64_t>(blockDim.x)) {
            uint64_t slot = Simt::UintDiv<uint64_t>(idx, magic, shift);
            uint64_t byteInSlot = idx - slot * unitBytes;
            uint64_t dstOff = dstBaseOff + slot * dstSlotStride + byteInSlot;
            if (isFilledWithZero) {
                dstGm[dstOff] = static_cast<T>(0);
            } else {
                uint64_t srcOff = srcBaseOff + slot * srcSlotStride + byteInSlot;
                dstGm[dstOff] = srcGm[srcOff];
            }
        }
    }

    // ===== Process 子函数 =====
    __aicore__ inline void ProcessNonContig();
    // 连续路径: 处理一个 batch 的所有 block
    __aicore__ inline void ProcessBatchBlocks(
        uint32_t batchIdx, T_INDEX seqLen, uint64_t seqOffset,
        uint64_t &keyOffset, uint64_t &valueOffset);
    // 非连续路径: 处理一个 block
    __aicore__ inline void ProcessNcBlock(
        uint32_t batchIdx, uint64_t blockIdx, uint64_t blockCount,
        T_INDEX seqLen, uint64_t seqOffset, int64_t batchOffset);
    // 连续字节拷贝封装
    __aicore__ inline void GatherKvCache(
        __gm__ T *dstGm, __gm__ T *srcGm,
        uint64_t dstOff, uint64_t srcOff, uint64_t curLen, bool isFilledWithZero);
    // 非连续拷贝封装: 按 slot/head stride 跳搬
    __aicore__ inline void GatherKvCacheNonContiguous(
        __gm__ T *dstGm, __gm__ T *srcGm,
        uint64_t curBlockSize, uint64_t hiddenSize,
        int64_t stride0, int64_t stride1, int64_t stride2,
        bool isSlotNonContig, bool isHeadNonContig,
        bool isFilledWithZero, bool isKey);

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

// ====================================================================
// Init / InitParams
// ====================================================================

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
}

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::InitParams()
{
    cacheBlockSize_ = tl_->kvCacheBlockSize;
    batchPerCore_ = tl_->batchPerCore;
    needCoreNum_ = tl_->needCoreNum;
    batchCount_ = tl_->batchCount;
    blockTableWidth_ = tl_->blockTableWidth;
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
}

// ====================================================================
// CalcKvCoreOffset — 简化为标量循环 (原 SIMD 用 DataCopyPad+ReduceSum)
// ====================================================================

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline T_INDEX
GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::CalcKvCoreOffset(int64_t reduceLen)
{
    T_INDEX coreOffset = 0;
    for (int64_t i = 0; i < reduceLen; i++) {
        coreOffset += seqLensGm_.GetValue(i);
    }
    return coreOffset;
}

// ====================================================================
// GatherKvCache — 连续字节拷贝: asc_vf_call 发射 GatherCopyContigVF
// ====================================================================

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::GatherKvCache(
    __gm__ T *dstGm, __gm__ T *srcGm, uint64_t dstOff, uint64_t srcOff, uint64_t curLen, bool isFilledWithZero)
{
    asc_vf_call<GatherCopyContigVF>(
        dim3(THREAD_NUM), dstOff, srcOff, curLen, dstGm, srcGm, isFilledWithZero ? 1u : 0u);
}

// ====================================================================
// GatherKvCacheNonContiguous — 跨步 slot 拷贝: asc_vf_call 发射 GatherCopyStridedVF
// ====================================================================

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::GatherKvCacheNonContiguous(
    __gm__ T *dstGm, __gm__ T *srcGm,
    uint64_t curBlockSize, uint64_t hiddenSize,
    int64_t stride0, int64_t stride1, int64_t stride2,
    bool isSlotNonContig, bool isHeadNonContig,
    bool isFilledWithZero, bool isKey)
{
    (void)stride0;          // block起始地址已在调用侧用stride0计算
    (void)isSlotNonContig;  // 进入本函数即slot非连续 或 输出非连续, 标志无需再用
    const uint64_t elemSize = isKey ? sizeof(DTYPE_KEY) : sizeof(DTYPE_VALUE);
    const uint64_t srcSlotStrideBytes = static_cast<uint64_t>(stride1) * elemSize;
    const uint64_t srcHeadStrideBytes = static_cast<uint64_t>(stride2) * elemSize;

    bool isOutNonContig = isKey ? ((nonContiguousFlag_ >> 2) & 1) : ((nonContiguousFlag_ >> 3) & 1);
    bool isOutHeadNonContig = isKey ? ((nonContiguousFlag_ >> 8) & 1) : ((nonContiguousFlag_ >> 9) & 1);
    int64_t outStride0 = isKey ? keyOutStride0_ : valueOutStride0_;
    int64_t outStride1 = isKey ? keyOutStride1_ : valueOutStride1_;
    const uint64_t dstSlotStrideBytes = isOutNonContig
                                            ? static_cast<uint64_t>(outStride0) * elemSize
                                            : hiddenSize;
    const uint64_t dstHeadStrideBytes = static_cast<uint64_t>(outStride1) * elemSize;

    uint64_t numHeads = isKey ? numHeadsK_ : numHeadsV_;
    uint64_t headBytes = (numHeads > 0) ? (hiddenSize / numHeads) : hiddenSize;
    bool needHeadLoop = isHeadNonContig || isOutHeadNonContig;

    uint64_t unitBytes = needHeadLoop ? headBytes : hiddenSize;
    uint64_t totalElems = curBlockSize * unitBytes;
    uint64_t magic = 0;
    uint64_t shift = 0;
    GetUintDivMagicAndShift(magic, shift, unitBytes);

    if (!needHeadLoop) {
        asc_vf_call<GatherCopyStridedVF>(
            dim3(THREAD_NUM), totalElems, unitBytes, 0UL, 0UL,
            dstSlotStrideBytes, srcSlotStrideBytes, dstGm, srcGm,
            isFilledWithZero ? 1u : 0u, magic, shift);
    } else {
        for (uint64_t h = 0; h < numHeads; h++) {
            asc_vf_call<GatherCopyStridedVF>(
                dim3(THREAD_NUM), totalElems, unitBytes,
                h * dstHeadStrideBytes, h * srcHeadStrideBytes,
                dstSlotStrideBytes, srcSlotStrideBytes, dstGm, srcGm,
                isFilledWithZero ? 1u : 0u, magic, shift);
        }
    }
}

// ====================================================================
// ProcessBatchBlocks — 连续路径: 处理一个 batch 的所有 block
// ====================================================================

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::ProcessBatchBlocks(
    uint32_t batchIdx, T_INDEX seqLen, uint64_t seqOffset,
    uint64_t &keyOffset, uint64_t &valueOffset)
{
    uint64_t blockCount = CeilDivision(seqLen, cacheBlockSize_);
    for (uint32_t blockIdx = 0; blockIdx < blockCount; blockIdx++) {
        uint64_t blockTableOffset = blockIdx + seqOffset;
        bool isFilledWithZero;
        int64_t blockId;
        if ((blockTableOffset >= blockTableWidth_) || (blockTableOffset < 0)) {
            isFilledWithZero = true;
            blockId = 0;
        } else {
            isFilledWithZero = false;
            blockId = blockTablesGm_.GetValue(blockTableWidth_ * batchIdx + blockTableOffset);
            if (blockId >= numBlocks_ || blockId < 0) {
                isFilledWithZero = true;
                blockId = 0;
            }
        }

        uint64_t curBlockSize = cacheBlockSize_;
        if (blockIdx == blockCount - 1) {
            curBlockSize = seqLen - (blockCount - 1) * cacheBlockSize_;
        }

        uint64_t keyCacheStart = blockId * cacheBlockSize_ * hiddenSizeK_;
        uint64_t valueCacheStart = blockId * cacheBlockSize_ * hiddenSizeV_;
        uint64_t keyLen = curBlockSize * hiddenSizeK_;
        uint64_t valueLen = curBlockSize * hiddenSizeV_;
        GatherKvCache((__gm__ T *)(outKeyGm_.GetPhyAddr()), (__gm__ T *)(keyCacheGm_.GetPhyAddr()),
                      keyOffset, keyCacheStart, keyLen, isFilledWithZero);
        GatherKvCache((__gm__ T *)(outValueGm_.GetPhyAddr()), (__gm__ T *)(valueCacheGm_.GetPhyAddr()),
                      valueOffset, valueCacheStart, valueLen, isFilledWithZero);
        keyOffset += keyLen;
        valueOffset += valueLen;
    }
}

// ====================================================================
// Process — 连续路径入口
// ====================================================================

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::Process()
{
    // 非连续: 走全局block轮询分核(用满所有核)。连续路径维持原batch分核。
    if (nonContiguousFlag_ != 0) {
        ProcessNonContig();
        return;
    }

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
        T_INDEX seqLen;
        int64_t batchOffset, accumSeqLen;
        if constexpr (isSeqLensCumsum) {
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
        if (numTokens_ > batchOffset && numTokens_ <= accumSeqLen) {
            seqLen = numTokens_ - batchOffset;
        }

        uint64_t seqOffset = 0;
        if constexpr (hasSeqOffset) {
            seqOffset = seqOffsetGm_.GetValue(i) / cacheBlockSize_;
        }

        uint64_t keyOffset = batchOffset * hiddenSizeK_;
        uint64_t valueOffset = batchOffset * hiddenSizeV_;
        ProcessBatchBlocks(i, seqLen, seqOffset, keyOffset, valueOffset);
    }
}

// ====================================================================
// ProcessNcBlock — 非连续路径: 处理一个 block 的 key+value
// ====================================================================

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::ProcessNcBlock(
    uint32_t batchIdx, uint64_t blockIdx, uint64_t blockCount,
    T_INDEX seqLen, uint64_t seqOffset, int64_t batchOffset)
{
    uint64_t blockTableOffset = blockIdx + seqOffset;
    bool isFilledWithZero;
    int64_t blockId;
    if ((blockTableOffset >= static_cast<uint64_t>(blockTableWidth_))) {
        isFilledWithZero = true;
        blockId = 0;
    } else {
        isFilledWithZero = false;
        blockId = blockTablesGm_.GetValue(blockTableWidth_ * batchIdx + blockTableOffset);
        if (blockId >= numBlocks_ || blockId < 0) {
            isFilledWithZero = true;
            blockId = 0;
        }
    }

    uint64_t curBlockSize = cacheBlockSize_;
    if (blockIdx == blockCount - 1) {
        curBlockSize = seqLen - (blockCount - 1) * cacheBlockSize_;
    }

    bool isKSlotNC = (nonContiguousFlag_ >> 4) & 1;
    bool isKHeadNC = (nonContiguousFlag_ >> 5) & 1;
    bool isVSlotNC = (nonContiguousFlag_ >> 6) & 1;
    bool isVHeadNC = (nonContiguousFlag_ >> 7) & 1;
    bool isKOutNC  = (nonContiguousFlag_ >> 2) & 1;
    bool isVOutNC  = (nonContiguousFlag_ >> 3) & 1;

    uint64_t baseToken = batchOffset + blockIdx * cacheBlockSize_;
    uint64_t keyOffset = isKOutNC ? (baseToken * keyOutStride0_ * sizeof(DTYPE_KEY))
                                  : (baseToken * hiddenSizeK_);
    uint64_t valueOffset = isVOutNC ? (baseToken * valueOutStride0_ * sizeof(DTYPE_VALUE))
                                    : (baseToken * hiddenSizeV_);

    uint64_t keyCacheStart = blockId * kCacheStride0_ * sizeof(DTYPE_KEY);
    uint64_t valueCacheStart = blockId * vCacheStride0_ * sizeof(DTYPE_VALUE);

    if (!isKSlotNC && !isKOutNC) {
        GatherKvCache((__gm__ T *)(outKeyGm_.GetPhyAddr()), (__gm__ T *)(keyCacheGm_.GetPhyAddr()),
                      keyOffset, keyCacheStart, curBlockSize * hiddenSizeK_, isFilledWithZero);
    } else {
        GatherKvCacheNonContiguous(
            (__gm__ T *)(outKeyGm_.GetPhyAddr()), (__gm__ T *)(keyCacheGm_.GetPhyAddr()),
            curBlockSize, hiddenSizeK_,
            kCacheStride0_, kCacheStride1_, kCacheStride2_,
            isKSlotNC, isKHeadNC, isFilledWithZero, true);
    }

    if (!isVSlotNC && !isVOutNC) {
        GatherKvCache((__gm__ T *)(outValueGm_.GetPhyAddr()), (__gm__ T *)(valueCacheGm_.GetPhyAddr()),
                      valueOffset, valueCacheStart, curBlockSize * hiddenSizeV_, isFilledWithZero);
    } else {
        GatherKvCacheNonContiguous(
            (__gm__ T *)(outValueGm_.GetPhyAddr()), (__gm__ T *)(valueCacheGm_.GetPhyAddr()),
            curBlockSize, hiddenSizeV_,
            vCacheStride0_, vCacheStride1_, vCacheStride2_,
            isVSlotNC, isVHeadNC, isFilledWithZero, false);
    }
}

// ====================================================================
// ProcessNonContig — 非连续路径入口: 全局block轮询分核
// ====================================================================

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::ProcessNonContig()
{
    const uint64_t coreIdx = GetBlockIdx();
    const uint64_t coreNum = needCoreNum_;
    uint64_t globalBlk = 0;
    int64_t runningOffset = 0;

    for (uint32_t i = 0; i < batchCount_; i++) {
        T_INDEX seqLen;
        int64_t batchOffset, accumSeqLen;
        if constexpr (isSeqLensCumsum) {
            seqLen = seqLensGm_.GetValue(i + 1) - seqLensGm_.GetValue(i);
            batchOffset = seqLensGm_.GetValue(i);
            accumSeqLen = seqLensGm_.GetValue(i + 1);
        } else {
            seqLen = seqLensGm_.GetValue(i);
            batchOffset = runningOffset;
            accumSeqLen = runningOffset + seqLensGm_.GetValue(i);
            runningOffset = accumSeqLen;
        }

        if (batchOffset >= numTokens_) {
            break;
        }
        if (numTokens_ > batchOffset && numTokens_ <= accumSeqLen) {
            seqLen = numTokens_ - batchOffset;
        }

        uint64_t seqOffset = 0;
        if constexpr (hasSeqOffset) {
            seqOffset = seqOffsetGm_.GetValue(i) / cacheBlockSize_;
        }

        uint64_t blockCount = CeilDivision(seqLen, cacheBlockSize_);
        for (uint64_t blockIdx = 0; blockIdx < blockCount; blockIdx++) {
            if ((globalBlk + blockIdx) % coreNum != coreIdx) {
                continue;
            }
            ProcessNcBlock(i, blockIdx, blockCount, seqLen, seqOffset, batchOffset);
        }
        globalBlk += blockCount;
    }
}

} // namespace GatherPaKvCacheV35
#endif
