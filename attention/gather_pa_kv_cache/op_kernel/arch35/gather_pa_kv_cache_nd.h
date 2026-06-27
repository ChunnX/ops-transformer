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
 * \brief SIMT implementation of gather_pa_kv_cache ND kernel (arch35)
 */

#ifndef GATHER_PA_KV_CACHE_ND_H_
#define GATHER_PA_KV_CACHE_ND_H_

#include "kernel_operator.h"

namespace NsGatherPaKvCacheNd {
using namespace AscendC;

constexpr uint32_t THREADS_32 = 2048;
constexpr uint32_t THREADS_64 = 1024;
constexpr uint32_t DEFAULT_THREADS = THREADS_32;

constexpr uint32_t NC_K_CACHE = 1 << 0;
constexpr uint32_t NC_V_CACHE = 1 << 1;
constexpr uint32_t NC_K_OUT   = 1 << 2;
constexpr uint32_t NC_V_OUT   = 1 << 3;
constexpr uint32_t NC_K_SLOT  = 1 << 4;
constexpr uint32_t NC_K_HEAD  = 1 << 5;
constexpr uint32_t NC_V_SLOT  = 1 << 6;
constexpr uint32_t NC_V_HEAD  = 1 << 7;
constexpr uint32_t NC_K_OUT_HEAD = 1 << 8;
constexpr uint32_t NC_V_OUT_HEAD = 1 << 9;

// =================================================================================
// VF kernels: Grid-Stride 线程并行字节级 GM 搬运
// =================================================================================

template <typename IDX_T, uint32_t THREADS>
__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline
void GatherCopyVf(IDX_T byteCount, __gm__ uint8_t* __restrict dst, __gm__ const uint8_t* __restrict src)
{
    IDX_T tid = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x);
    IDX_T stride = static_cast<IDX_T>(blockDim.x * gridDim.x);
    for (IDX_T i = tid; i < byteCount; i += stride) {
        dst[i] = src[i];
    }
}

template <typename IDX_T, uint32_t THREADS>
__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline
void GatherZeroVf(IDX_T byteCount, __gm__ uint8_t* __restrict dst)
{
    IDX_T tid = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x);
    IDX_T stride = static_cast<IDX_T>(blockDim.x * gridDim.x);
    for (IDX_T i = tid; i < byteCount; i += stride) {
        dst[i] = 0;
    }
}

template <typename IDX_T, uint32_t THREADS>
__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline
void GatherCopyStridedVf(IDX_T outerCount, IDX_T innerBytes,
    __gm__ uint8_t* __restrict dst, __gm__ const uint8_t* __restrict src,
    IDX_T dstStrideBytes, IDX_T srcStrideBytes)
{
    IDX_T tid = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x);
    IDX_T stride = static_cast<IDX_T>(blockDim.x * gridDim.x);
    IDX_T total = outerCount * innerBytes;
    for (IDX_T i = tid; i < total; i += stride) {
        IDX_T outer = i / innerBytes;
        IDX_T inner = i - outer * innerBytes;
        dst[outer * dstStrideBytes + inner] = src[outer * srcStrideBytes + inner];
    }
}

template <typename IDX_T, uint32_t THREADS>
__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline
void GatherZeroStridedVf(IDX_T outerCount, IDX_T innerBytes,
    __gm__ uint8_t* __restrict dst, IDX_T dstStrideBytes)
{
    IDX_T tid = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x);
    IDX_T stride = static_cast<IDX_T>(blockDim.x * gridDim.x);
    IDX_T total = outerCount * innerBytes;
    for (IDX_T i = tid; i < total; i += stride) {
        IDX_T outer = i / innerBytes;
        IDX_T inner = i - outer * innerBytes;
        dst[outer * dstStrideBytes + inner] = 0;
    }
}

// =================================================================================
// Scalar helpers
// =================================================================================

inline uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    return (x + y - 1) / y;
}

__aicore__ inline void LaunchCopyOrZero(uint64_t byteCount, __gm__ uint8_t* dst,
    __gm__ const uint8_t* src, bool isZeroFill)
{
    if (byteCount == 0) return;
    bool use32 = (byteCount <= static_cast<uint64_t>(INT32_MAX));
    if (isZeroFill) {
        if (use32) {
            asc_vf_call<GatherZeroVf<uint32_t, THREADS_32>>(
                dim3(THREADS_32), static_cast<uint32_t>(byteCount), dst);
        } else {
            asc_vf_call<GatherZeroVf<uint64_t, THREADS_64>>(
                dim3(THREADS_64), byteCount, dst);
        }
    } else {
        if (use32) {
            asc_vf_call<GatherCopyVf<uint32_t, THREADS_32>>(
                dim3(THREADS_32), static_cast<uint32_t>(byteCount), dst, src);
        } else {
            asc_vf_call<GatherCopyVf<uint64_t, THREADS_64>>(
                dim3(THREADS_64), byteCount, dst, src);
        }
    }
}

__aicore__ inline void LaunchCopyStridedOrZero(
    uint64_t outerCount, uint64_t innerBytes,
    __gm__ uint8_t* dst, __gm__ const uint8_t* src,
    uint64_t dstStrideBytes, uint64_t srcStrideBytes, bool isZeroFill)
{
    if (outerCount == 0 || innerBytes == 0) return;
    uint64_t total = outerCount * innerBytes;
    bool use32 = (total <= static_cast<uint64_t>(INT32_MAX));
    if (isZeroFill) {
        if (use32) {
            asc_vf_call<GatherZeroStridedVf<uint32_t, THREADS_32>>(
                dim3(THREADS_32), static_cast<uint32_t>(outerCount), static_cast<uint32_t>(innerBytes),
                dst, static_cast<uint32_t>(dstStrideBytes));
        } else {
            asc_vf_call<GatherZeroStridedVf<uint64_t, THREADS_64>>(
                dim3(THREADS_64), outerCount, innerBytes, dst, dstStrideBytes);
        }
    } else {
        if (use32) {
            asc_vf_call<GatherCopyStridedVf<uint32_t, THREADS_32>>(
                dim3(THREADS_32), static_cast<uint32_t>(outerCount), static_cast<uint32_t>(innerBytes),
                dst, src, static_cast<uint32_t>(dstStrideBytes), static_cast<uint32_t>(srcStrideBytes));
        } else {
            asc_vf_call<GatherCopyStridedVf<uint64_t, THREADS_64>>(
                dim3(THREADS_64), outerCount, innerBytes, dst, src, dstStrideBytes, srcStrideBytes);
        }
    }
}

// =================================================================================
// 连续路径: 按UB分块搬运整段连续数据
// =================================================================================
__aicore__ inline void GatherKvCacheContiguous(
    __gm__ uint8_t* dst, __gm__ const uint8_t* src,
    uint64_t totalBytes, uint64_t maxUbBytes, bool isFilledWithZero)
{
    uint64_t handled = 0;
    while (handled < totalBytes) {
        uint64_t curBytes = totalBytes - handled;
        if (curBytes > maxUbBytes) {
            curBytes = maxUbBytes;
        }
        LaunchCopyOrZero(curBytes, dst + handled, src + handled, isFilledWithZero);
        handled += curBytes;
    }
}

// =================================================================================
// 非连续路径: slot维批量化拷贝 (受UB/blockCount限制按slot分批)
// =================================================================================
__aicore__ inline void GatherKvCacheSlotBatch(
    __gm__ uint8_t* dst, __gm__ const uint8_t* src,
    uint64_t dstBaseOff, uint64_t srcBaseOff,
    uint64_t numSlots, uint64_t unitBytes,
    uint64_t srcSlotStrideBytes, uint64_t dstSlotStrideBytes,
    uint64_t maxUbBytes, bool isFilledWithZero)
{
    constexpr uint64_t MAX_BLOCK_CNT = 4095;
    if (unitBytes > maxUbBytes) {
        for (uint64_t s = 0; s < numSlots; s++) {
            GatherKvCacheContiguous(
                dst + dstBaseOff + s * dstSlotStrideBytes,
                src + srcBaseOff + s * srcSlotStrideBytes,
                unitBytes, maxUbBytes, isFilledWithZero);
        }
        return;
    }

    uint64_t slotsPerBatch = maxUbBytes / unitBytes;
    if (slotsPerBatch == 0) slotsPerBatch = 1;
    if (slotsPerBatch > MAX_BLOCK_CNT) slotsPerBatch = MAX_BLOCK_CNT;
    if (slotsPerBatch > numSlots) slotsPerBatch = numSlots;

    uint64_t slotDone = 0;
    while (slotDone < numSlots) {
        uint64_t curSlots = numSlots - slotDone;
        if (curSlots > slotsPerBatch) curSlots = slotsPerBatch;

        LaunchCopyStridedOrZero(
            curSlots, unitBytes,
            dst + dstBaseOff + slotDone * dstSlotStrideBytes,
            src + srcBaseOff + slotDone * srcSlotStrideBytes,
            dstSlotStrideBytes, srcSlotStrideBytes, isFilledWithZero);

        slotDone += curSlots;
    }
}

// =================================================================================
// 非连续路径: 逐head再逐slot
// =================================================================================
__aicore__ inline void GatherKvCacheNonContiguousDispatch(
    __gm__ uint8_t* dst, __gm__ const uint8_t* src,
    uint64_t dstBaseOff, uint64_t srcBaseOff,
    uint64_t curBlockSize, uint64_t hiddenSizeBytes,
    int64_t srcStride1, int64_t srcStride2,
    int64_t outStride1, uint64_t numHeads,
    bool isSlotNC, bool isHeadNC, bool isOutNC, bool isOutHeadNC,
    uint64_t maxUbBytes, bool isFilledWithZero, uint64_t elemByteSize)
{
    (void)isSlotNC;
    uint64_t srcSlotStrideBytes = static_cast<uint64_t>(srcStride1) * elemByteSize;
    uint64_t dstSlotStrideBytes = isOutNC
        ? static_cast<uint64_t>(outStride1) * elemByteSize
        : hiddenSizeBytes;
    uint64_t srcHeadStrideBytes = static_cast<uint64_t>(srcStride2) * elemByteSize;
    uint64_t dstHeadStrideBytes = static_cast<uint64_t>(outStride1) * elemByteSize;

    bool needHeadLoop = isHeadNC || isOutHeadNC;
    uint64_t headBytes = (numHeads > 0) ? (hiddenSizeBytes / numHeads) : hiddenSizeBytes;

    if (!needHeadLoop) {
        GatherKvCacheSlotBatch(dst, src, dstBaseOff, srcBaseOff,
            curBlockSize, hiddenSizeBytes,
            srcSlotStrideBytes, dstSlotStrideBytes,
            maxUbBytes, isFilledWithZero);
    } else {
        for (uint64_t h = 0; h < numHeads; h++) {
            GatherKvCacheSlotBatch(dst, src,
                dstBaseOff + h * dstHeadStrideBytes,
                srcBaseOff + h * srcHeadStrideBytes,
                curBlockSize, headBytes,
                srcSlotStrideBytes, dstSlotStrideBytes,
                maxUbBytes, isFilledWithZero);
        }
    }
}

// =================================================================================
// 前缀和计算: 标量循环累加 (替代原SIMD ReduceSum)
// =================================================================================
template <typename T_INDEX>
__aicore__ inline T_INDEX CalcCoreOffset(
    __gm__ const T_INDEX* seqLens, int64_t reduceLen)
{
    T_INDEX coreOffset = 0;
    for (int64_t i = 0; i < reduceLen; i++) {
        coreOffset += seqLens[i];
    }
    return coreOffset;
}

// =================================================================================
// Process: 连续路径主流程
// =================================================================================
template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void ProcessContiguous(
    const GatherPaKvCacheTilingDataV35& td,
    __gm__ uint8_t* keyCache, __gm__ uint8_t* valueCache,
    __gm__ const T_INDEX* blockTables, __gm__ const T_INDEX* seqLens,
    __gm__ const T_INDEX* seqOffset, __gm__ uint8_t* keyOut, __gm__ uint8_t* valueOut)
{
    uint64_t needCoreNum = td.needCoreNum;
    int64_t batchPerCore = td.batchPerCore;
    int64_t batchCount = td.batchCount;
    int64_t cacheBlockSize = td.kvCacheBlockSize;
    int64_t blockTableWidth = td.blockTableWidth;
    int64_t numBlocks = td.numBlocks;
    int64_t numTokens = td.numTokens;
    uint64_t hiddenSizeK = td.hiddenSizeK;
    uint64_t hiddenSizeV = td.hiddenSizeV;
    uint64_t maxUbHiddenSize = td.maxUbHiddenSize;

    uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
    int64_t batchStart = static_cast<int64_t>(coreIdx) * batchPerCore;
    int64_t batchEnd = batchStart + batchPerCore;
    if (coreIdx == needCoreNum - 1) {
        batchEnd = batchCount;
    }

    int64_t coreOffset;
    if constexpr (isSeqLensCumsum) {
        coreOffset = static_cast<int64_t>(seqLens[batchStart]);
    } else {
        coreOffset = static_cast<int64_t>(CalcCoreOffset<T_INDEX>(seqLens, batchStart));
    }

    for (int64_t b = batchStart; b < batchEnd; b++) {
        T_INDEX seqLen;
        int64_t batchOffset, accumSeqLen;
        if constexpr (isSeqLensCumsum) {
            seqLen = static_cast<T_INDEX>(seqLens[b + 1] - seqLens[b]);
            batchOffset = static_cast<int64_t>(seqLens[b]);
            accumSeqLen = static_cast<int64_t>(seqLens[b + 1]);
        } else {
            seqLen = seqLens[b];
            batchOffset = coreOffset;
            accumSeqLen = coreOffset + static_cast<int64_t>(seqLens[b]);
            coreOffset = accumSeqLen;
        }

        if (batchOffset >= numTokens) break;

        if (numTokens > batchOffset && numTokens <= accumSeqLen) {
            seqLen = static_cast<T_INDEX>(numTokens - batchOffset);
        }

        uint64_t seqOff = 0;
        if constexpr (hasSeqOffset) {
            seqOff = static_cast<uint64_t>(seqOffset[b]) / static_cast<uint64_t>(cacheBlockSize);
        }

        uint64_t blockCount = CeilDiv(static_cast<uint64_t>(seqLen), static_cast<uint64_t>(cacheBlockSize));
        uint64_t keyOffset = static_cast<uint64_t>(batchOffset) * hiddenSizeK;
        uint64_t valueOffset = static_cast<uint64_t>(batchOffset) * hiddenSizeV;

        for (uint64_t blkIdx = 0; blkIdx < blockCount; blkIdx++) {
            uint64_t btOff = blkIdx + seqOff;
            bool isZeroFill;
            int64_t blockId;
            if (btOff >= static_cast<uint64_t>(blockTableWidth)) {
                isZeroFill = true;
                blockId = 0;
            } else {
                isZeroFill = false;
                blockId = static_cast<int64_t>(blockTables[b * blockTableWidth + btOff]);
                if (blockId >= numBlocks || blockId < 0) {
                    isZeroFill = true;
                    blockId = 0;
                }
            }

            uint64_t curBlockSize = static_cast<uint64_t>(cacheBlockSize);
            if (blkIdx == blockCount - 1) {
                curBlockSize = static_cast<uint64_t>(seqLen) - (blockCount - 1) * static_cast<uint64_t>(cacheBlockSize);
            }

            uint64_t keyBlockBytes = curBlockSize * hiddenSizeK;
            uint64_t valBlockBytes = curBlockSize * hiddenSizeV;
            uint64_t keyCacheStart = static_cast<uint64_t>(blockId) * static_cast<uint64_t>(cacheBlockSize) * hiddenSizeK;
            uint64_t valueCacheStart = static_cast<uint64_t>(blockId) * static_cast<uint64_t>(cacheBlockSize) * hiddenSizeV;

            GatherKvCacheContiguous(keyOut + keyOffset, keyCache + keyCacheStart,
                keyBlockBytes, maxUbHiddenSize, isZeroFill);
            keyOffset += keyBlockBytes;

            GatherKvCacheContiguous(valueOut + valueOffset, valueCache + valueCacheStart,
                valBlockBytes, maxUbHiddenSize, isZeroFill);
            valueOffset += valBlockBytes;
        }
    }
}

// =================================================================================
// Process: 非连续路径主流程 (全局block轮询分核)
// =================================================================================
template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void ProcessNonContiguous(
    const GatherPaKvCacheTilingDataV35& td,
    __gm__ uint8_t* keyCache, __gm__ uint8_t* valueCache,
    __gm__ const T_INDEX* blockTables, __gm__ const T_INDEX* seqLens,
    __gm__ const T_INDEX* seqOffset, __gm__ uint8_t* keyOut, __gm__ uint8_t* valueOut)
{
    uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
    uint64_t coreNum = td.needCoreNum;
    int64_t batchCount = td.batchCount;
    int64_t cacheBlockSize = td.kvCacheBlockSize;
    int64_t blockTableWidth = td.blockTableWidth;
    int64_t numBlocks = td.numBlocks;
    int64_t numTokens = td.numTokens;
    uint64_t hiddenSizeK = td.hiddenSizeK;
    uint64_t hiddenSizeV = td.hiddenSizeV;
    uint64_t maxUbHiddenSize = td.maxUbHiddenSize;

    uint64_t nonContiguousFlag = td.nonContiguousFlag;
    uint64_t numHeadsK = td.numHeadsK;
    uint64_t numHeadsV = td.numHeadsV;

    bool isKOutNC = (nonContiguousFlag & NC_K_OUT) != 0;
    bool isVOutNC = (nonContiguousFlag & NC_V_OUT) != 0;
    bool isKSlotNC = (nonContiguousFlag & NC_K_SLOT) != 0;
    bool isKHeadNC = (nonContiguousFlag & NC_K_HEAD) != 0;
    bool isVSlotNC = (nonContiguousFlag & NC_V_SLOT) != 0;
    bool isVHeadNC = (nonContiguousFlag & NC_V_HEAD) != 0;
    bool isKOutHeadNC = (nonContiguousFlag & NC_K_OUT_HEAD) != 0;
    bool isVOutHeadNC = (nonContiguousFlag & NC_V_OUT_HEAD) != 0;

    uint64_t kElemByte = hiddenSizeK > 0 ? hiddenSizeK : 1;  // byte for key element
    uint64_t vElemByte = hiddenSizeV > 0 ? hiddenSizeV : 1;  // byte for value element
    (void)kElemByte;
    (void)vElemByte;

    uint64_t globalBlk = 0;
    int64_t runningOffset = 0;

    for (int64_t b = 0; b < batchCount; b++) {
        T_INDEX seqLen;
        int64_t batchOffset, accumSeqLen;
        if constexpr (isSeqLensCumsum) {
            seqLen = static_cast<T_INDEX>(seqLens[b + 1] - seqLens[b]);
            batchOffset = static_cast<int64_t>(seqLens[b]);
            accumSeqLen = static_cast<int64_t>(seqLens[b + 1]);
        } else {
            seqLen = seqLens[b];
            batchOffset = runningOffset;
            accumSeqLen = runningOffset + static_cast<int64_t>(seqLens[b]);
            runningOffset = accumSeqLen;
        }

        if (batchOffset >= numTokens) break;
        if (numTokens > batchOffset && numTokens <= accumSeqLen) {
            seqLen = static_cast<T_INDEX>(numTokens - batchOffset);
        }

        uint64_t seqOff = 0;
        if constexpr (hasSeqOffset) {
            seqOff = static_cast<uint64_t>(seqOffset[b]) / static_cast<uint64_t>(cacheBlockSize);
        }

        uint64_t blockCount = CeilDiv(static_cast<uint64_t>(seqLen), static_cast<uint64_t>(cacheBlockSize));
        for (uint64_t blkIdx = 0; blkIdx < blockCount; blkIdx++) {
            if ((globalBlk + blkIdx) % coreNum != coreIdx) continue;

            uint64_t btOff = blkIdx + seqOff;
            bool isZeroFill;
            int64_t blockId;
            if (btOff >= static_cast<uint64_t>(blockTableWidth)) {
                isZeroFill = true;
                blockId = 0;
            } else {
                isZeroFill = false;
                blockId = static_cast<int64_t>(blockTables[b * blockTableWidth + btOff]);
                if (blockId >= numBlocks || blockId < 0) {
                    isZeroFill = true;
                    blockId = 0;
                }
            }

            uint64_t curBlockSize = static_cast<uint64_t>(cacheBlockSize);
            if (blkIdx == blockCount - 1) {
                curBlockSize = static_cast<uint64_t>(seqLen) - (blockCount - 1) * static_cast<uint64_t>(cacheBlockSize);
            }

            uint64_t baseToken = static_cast<uint64_t>(batchOffset) + blkIdx * static_cast<uint64_t>(cacheBlockSize);
            uint64_t keyOutOff = isKOutNC
                ? (baseToken * static_cast<uint64_t>(td.keyOutStride0))
                : (baseToken * hiddenSizeK);
            uint64_t valueOutOff = isVOutNC
                ? (baseToken * static_cast<uint64_t>(td.valueOutStride0))
                : (baseToken * hiddenSizeV);

            uint64_t keyCacheStart = static_cast<uint64_t>(blockId) * static_cast<uint64_t>(td.kCacheStride0);
            uint64_t valueCacheStart = static_cast<uint64_t>(blockId) * static_cast<uint64_t>(td.vCacheStride0);

            // Key: 不连续或输出不连续 → 走非连续dispatch
            bool kNeedNC = isKSlotNC || isKHeadNC || isKOutNC || isKOutHeadNC;
            if (!kNeedNC) {
                GatherKvCacheContiguous(keyOut + keyOutOff, keyCache + keyCacheStart,
                    curBlockSize * hiddenSizeK, maxUbHiddenSize, isZeroFill);
            } else {
                GatherKvCacheNonContiguousDispatch(
                    keyOut + keyOutOff, keyCache + keyCacheStart,
                    0, 0,  // dst/src 指针已含偏移, baseOff 补 0
                    curBlockSize, hiddenSizeK,
                    td.kCacheStride1, td.kCacheStride2,
                    td.keyOutStride1, numHeadsK,
                    isKSlotNC, isKHeadNC, isKOutNC, isKOutHeadNC,
                    maxUbHiddenSize, isZeroFill,
                    (hiddenSizeK > 0 && numHeadsK > 0) ? (hiddenSizeK / numHeadsK) : 1);
            }

            // Value
            bool vNeedNC = isVSlotNC || isVHeadNC || isVOutNC || isVOutHeadNC;
            if (!vNeedNC) {
                GatherKvCacheContiguous(valueOut + valueOutOff, valueCache + valueCacheStart,
                    curBlockSize * hiddenSizeV, maxUbHiddenSize, isZeroFill);
            } else {
                GatherKvCacheNonContiguousDispatch(
                    valueOut + valueOutOff, valueCache + valueCacheStart,
                    0, 0,
                    curBlockSize, hiddenSizeV,
                    td.vCacheStride1, td.vCacheStride2,
                    td.valueOutStride1, numHeadsV,
                    isVSlotNC, isVHeadNC, isVOutNC, isVOutHeadNC,
                    maxUbHiddenSize, isZeroFill,
                    (hiddenSizeV > 0 && numHeadsV > 0) ? (hiddenSizeV / numHeadsV) : 1);
            }
        }
        globalBlk += blockCount;
    }
}

// =================================================================================
// Process: 顶层入口, 根据nonContiguousFlag分派连续/非连续路径
// =================================================================================
template <typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void Process(
    const GatherPaKvCacheTilingDataV35& td,
    __gm__ uint8_t* keyCache, __gm__ uint8_t* valueCache,
    __gm__ const T_INDEX* blockTables, __gm__ const T_INDEX* seqLens,
    __gm__ const T_INDEX* seqOffset, __gm__ uint8_t* keyOut, __gm__ uint8_t* valueOut)
{
    if (td.nonContiguousFlag != 0) {
        ProcessNonContiguous<T_INDEX, isSeqLensCumsum, hasSeqOffset>(
            td, keyCache, valueCache, blockTables, seqLens, seqOffset, keyOut, valueOut);
    } else {
        ProcessContiguous<T_INDEX, isSeqLensCumsum, hasSeqOffset>(
            td, keyCache, valueCache, blockTables, seqLens, seqOffset, keyOut, valueOut);
    }

    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
}

} // namespace NsGatherPaKvCacheNd
#endif // GATHER_PA_KV_CACHE_ND_H_
