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
 * \file gather_pa_kv_cache_simt.h
 * \brief SIMT implementation of GatherPaKvCache ND format.
 *
 *  Replaces gather_pa_kv_cache_nd.h VEC_SCOPE path with SIMT kernels.
 *  NZ format continues to use gather_pa_kv_cache_nz.h unchanged.
 *
 *  VEC_SCOPE → SIMT mapping:
 *    TPipe/TBuf/DataCopyPad → __simt_vf__ + asc_vf_call + __gm__ pointer
 *    class-based → namespace + free functions
 */

#ifndef GATHER_PA_KV_CACHE_SIMT_H
#define GATHER_PA_KV_CACHE_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"

namespace NsGatherPaKvCache {
using namespace AscendC;

constexpr uint32_t THREADS_32 = 2048;
constexpr uint32_t THREADS_64 = 1024;

constexpr uint32_t NC_K_CACHE = 1 << 0;
constexpr uint32_t NC_V_CACHE = 1 << 1;
constexpr uint32_t NC_K_OUT   = 1 << 2;
constexpr uint32_t NC_V_OUT   = 1 << 3;
constexpr uint32_t NC_K_SLOT  = 1 << 4;
constexpr uint32_t NC_K_HEAD  = 1 << 5;
constexpr uint32_t NC_V_SLOT  = 1 << 6;
constexpr uint32_t NC_V_HEAD  = 1 << 7;

// ==================== VF: 连续块拷贝 ====================
template <typename IDX_T, uint32_t THREADS>
__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline void GatherCopyBlock(
    IDX_T byteCount, __gm__ uint8_t* dst, __gm__ uint8_t* src)
{
    for (IDX_T i = static_cast<IDX_T>(blockIdx.x) * THREADS + static_cast<IDX_T>(threadIdx.x);
         i < byteCount; i += THREADS * static_cast<IDX_T>(gridDim.x)) {
        dst[i] = src[i];
    }
}

// ==================== VF: 连续块填零 ====================
template <typename IDX_T, uint32_t THREADS>
__simt_vf__ __aicore__ __launch_bounds__(THREADS) inline void GatherZeroBlock(
    IDX_T byteCount, __gm__ uint8_t* dst)
{
    for (IDX_T i = static_cast<IDX_T>(blockIdx.x) * THREADS + static_cast<IDX_T>(threadIdx.x);
         i < byteCount; i += THREADS * static_cast<IDX_T>(gridDim.x)) {
        dst[i] = 0;
    }
}

// ==================== Launch Helpers ====================
template <typename IDX_T, uint32_t THREADS>
__aicore__ inline void LaunchCopy(IDX_T byteCount, __gm__ uint8_t* dst, __gm__ uint8_t* src)
{
    asc_vf_call<GatherCopyBlock<IDX_T, THREADS>>(dim3(THREADS), byteCount, dst, src);
}

template <typename IDX_T, uint32_t THREADS>
__aicore__ inline void LaunchZero(IDX_T byteCount, __gm__ uint8_t* dst)
{
    asc_vf_call<GatherZeroBlock<IDX_T, THREADS>>(dim3(THREADS), byteCount, dst);
}

// ==================== CeilDiv ====================
template <typename A, typename B>
__aicore__ __attribute__((always_inline)) inline A CeilDiv(A a, B b) { return (a + b - 1) / b; }

// ==================== 连续块处理 ====================
template <typename IDX_T, uint32_t THREADS>
__aicore__ inline void CopyBlockContig(
    __gm__ uint8_t* cacheGm, __gm__ uint8_t* outGm,
    uint64_t cacheStartBytes, uint64_t totalBytes, bool isZero, uint64_t& outOffBytes)
{
    if (isZero) {
        LaunchZero<IDX_T, THREADS>(static_cast<IDX_T>(totalBytes), outGm + outOffBytes);
    } else {
        LaunchCopy<IDX_T, THREADS>(static_cast<IDX_T>(totalBytes), outGm + outOffBytes, cacheGm + cacheStartBytes);
    }
    outOffBytes += totalBytes;
}

// ==================== 非连续slot拷贝 ====================
template <typename IDX_T, uint32_t THREADS>
__aicore__ inline void CopyBlockNcSlot(
    __gm__ uint8_t* cacheGm, __gm__ uint8_t* outGm,
    int64_t curBlockSize, uint64_t hiddenSizeB,
    uint64_t blockStartBytes, uint64_t slotSrcStride,
    uint64_t dstSlotStride, bool isZero, uint64_t& outOffBytes)
{
    uint64_t baseOff = outOffBytes;
    for (int64_t s = 0; s < curBlockSize; s++) {
        uint64_t dstOff = baseOff + static_cast<uint64_t>(s) * dstSlotStride;
        if (isZero) {
            LaunchZero<IDX_T, THREADS>(static_cast<IDX_T>(hiddenSizeB), outGm + dstOff);
        } else {
            uint64_t srcOff = blockStartBytes + static_cast<uint64_t>(s) * slotSrcStride;
            LaunchCopy<IDX_T, THREADS>(static_cast<IDX_T>(hiddenSizeB), outGm + dstOff, cacheGm + srcOff);
        }
    }
    outOffBytes = baseOff + static_cast<uint64_t>(curBlockSize) * dstSlotStride;
}

// ==================== 非连续slot+head拷贝 ====================
template <typename IDX_T, uint32_t THREADS>
__aicore__ inline void CopyBlockNcHead(
    __gm__ uint8_t* cacheGm, __gm__ uint8_t* outGm,
    int64_t curBlockSize, uint64_t numHeads, uint64_t headBytes,
    uint64_t blockStartBytes, uint64_t slotSrcStride, uint64_t headSrcStride,
    uint64_t dstSlotStride, bool isZero, uint64_t& outOffBytes)
{
    uint64_t baseOff = outOffBytes;
    for (int64_t s = 0; s < curBlockSize; s++) {
        uint64_t dstSlotOff = baseOff + static_cast<uint64_t>(s) * dstSlotStride;
        for (uint64_t h = 0; h < numHeads; h++) {
            uint64_t dstOff = dstSlotOff + h * headBytes;
            if (isZero) {
                LaunchZero<IDX_T, THREADS>(static_cast<IDX_T>(headBytes), outGm + dstOff);
            } else {
                uint64_t srcOff = blockStartBytes + static_cast<uint64_t>(s) * slotSrcStride + h * headSrcStride;
                LaunchCopy<IDX_T, THREADS>(static_cast<IDX_T>(headBytes), outGm + dstOff, cacheGm + srcOff);
            }
        }
    }
    outOffBytes = baseOff + static_cast<uint64_t>(curBlockSize) * dstSlotStride;
}

// ==================== 非连续整体调度 ====================
template <typename IDX_T, uint32_t THREADS>
__aicore__ inline void CopyBlockNonContig(
    __gm__ uint8_t* cacheGm, __gm__ uint8_t* outGm,
    int64_t blockId, int64_t curBlockSize,
    uint32_t ncFlag, uint64_t hiddenSizeB, uint64_t elemByteSize,
    int64_t kvCacheBlockSize, uint64_t numHeads,
    int64_t cacheStride0, int64_t cacheStride1, int64_t cacheStride2,
    int64_t outStride1, bool isSlotNC, bool isHeadNC, bool isOutNC,
    bool isZero, uint64_t& outOffBytes)
{
    uint64_t slotSrcStride = static_cast<uint64_t>(cacheStride1) * elemByteSize;
    uint64_t headSrcStride = static_cast<uint64_t>(cacheStride2) * elemByteSize;
    uint64_t blockStartBytes = static_cast<uint64_t>(blockId) * static_cast<uint64_t>(cacheStride0) * elemByteSize;
    uint64_t dstSlotStride = isOutNC ? static_cast<uint64_t>(outStride1) * elemByteSize : hiddenSizeB;

    if (!isSlotNC) {
        uint64_t totalBytes = static_cast<uint64_t>(curBlockSize) * hiddenSizeB;
        CopyBlockContig<IDX_T, THREADS>(cacheGm, outGm, blockStartBytes, totalBytes, isZero, outOffBytes);
    } else if (!isHeadNC) {
        CopyBlockNcSlot<IDX_T, THREADS>(cacheGm, outGm, curBlockSize, hiddenSizeB,
            blockStartBytes, slotSrcStride, dstSlotStride, isZero, outOffBytes);
    } else {
        uint64_t headB = (numHeads > 0) ? (hiddenSizeB / numHeads) : hiddenSizeB;
        CopyBlockNcHead<IDX_T, THREADS>(cacheGm, outGm, curBlockSize, numHeads, headB,
            blockStartBytes, slotSrcStride, headSrcStride, dstSlotStride, isZero, outOffBytes);
    }
}

// ==================== Process 主函数 ====================
template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void Process(
    GM_ADDR key_cache, GM_ADDR value_cache,
    GM_ADDR block_tables, GM_ADDR seq_lens,
    GM_ADDR key_out, GM_ADDR value_out,
    GM_ADDR seq_offset,
    const GatherPaKvCacheTilingDataV35* td)
{
    __gm__ uint8_t* kc = (__gm__ uint8_t*)key_cache;
    __gm__ uint8_t* vc = (__gm__ uint8_t*)value_cache;
    __gm__ T_INDEX* bt = (__gm__ T_INDEX*)block_tables;
    __gm__ T_INDEX* sl = (__gm__ T_INDEX*)seq_lens;
    __gm__ uint8_t* ko = (__gm__ uint8_t*)key_out;
    __gm__ uint8_t* vo = (__gm__ uint8_t*)value_out;
    __gm__ T_INDEX* so = (__gm__ T_INDEX*)seq_offset;

    constexpr uint64_t kByte = sizeof(DTYPE_KEY);
    constexpr uint64_t vByte = sizeof(DTYPE_VALUE);

    PRINTF("[SIMT] gather_pa_kv_cache ND enter, core=%lld, isCumsum=%d, hasSeqOff=%d\n",
           static_cast<long long>(GetBlockIdx()),
           static_cast<int>(isSeqLensCumsum),
           static_cast<int>(hasSeqOffset));

    int64_t bc = td->batchCount;
    int64_t bpc = td->batchPerCore;
    int64_t ncn = static_cast<int64_t>(td->needCoreNum);
    int64_t cbs = td->kvCacheBlockSize;
    int64_t btw = td->blockTableWidth;
    int64_t nb = td->numBlocks;
    int64_t nt = td->numTokens;
    uint64_t hkb = td->hiddenSizeK * kByte;
    uint64_t hvb = td->hiddenSizeV * vByte;
    uint64_t cbbK = static_cast<uint64_t>(cbs) * hkb;
    uint64_t cbbV = static_cast<uint64_t>(cbs) * hvb;
    uint32_t ncF = td->nonContiguousFlag;

    int64_t coreIdx = GetBlockIdx();
    int64_t batchStart = coreIdx * bpc;
    int64_t batchEnd = batchStart + bpc;
    if (coreIdx == ncn - 1) { batchEnd = bc; }

    int64_t coreOff = 0;
    if constexpr (!isSeqLensCumsum) {
        for (int64_t i = 0; i < batchStart; i++) { coreOff += static_cast<int64_t>(sl[i]); }
    }

    for (int64_t b = batchStart; b < batchEnd; b++) {
        T_INDEX seqLen;
        int64_t batchOffset;
        int64_t accum;
        if constexpr (isSeqLensCumsum) {
            seqLen = sl[b + 1] - sl[b];
            batchOffset = static_cast<int64_t>(sl[b]);
            accum = static_cast<int64_t>(sl[b + 1]);
        } else {
            seqLen = sl[b];
            batchOffset = coreOff;
            accum = coreOff + static_cast<int64_t>(sl[b]);
            coreOff = accum;
        }
        if (batchOffset >= nt) break;
        if (nt > batchOffset && nt <= accum) { seqLen = static_cast<T_INDEX>(nt - batchOffset); }

        int64_t seqOff = 0;
        if constexpr (hasSeqOffset) { seqOff = static_cast<int64_t>(so[b]) / cbs; }

        int64_t blkCnt = static_cast<int64_t>(CeilDiv(seqLen, static_cast<T_INDEX>(cbs)));
        uint64_t keyOff = static_cast<uint64_t>(batchOffset) * hkb;
        uint64_t valOff = static_cast<uint64_t>(batchOffset) * hvb;

        for (int64_t blk = 0; blk < blkCnt; blk++) {
            int64_t btIdx = blk + seqOff;
            bool isZero;
            int64_t blockId;
            if (btIdx >= btw || btIdx < 0) {
                isZero = true; blockId = 0;
            } else {
                isZero = false;
                blockId = static_cast<int64_t>(bt[b * btw + btIdx]);
                if (blockId >= nb || blockId < 0) { isZero = true; blockId = 0; }
            }
            int64_t curSz = cbs;
            if (blk == blkCnt - 1) { curSz = static_cast<int64_t>(seqLen) - (blkCnt - 1) * cbs; }

            bool kNC = (ncF & (NC_K_CACHE | NC_K_SLOT | NC_K_HEAD | NC_K_OUT)) != 0;
            bool vNC = (ncF & (NC_V_CACHE | NC_V_SLOT | NC_V_HEAD | NC_V_OUT)) != 0;
            uint64_t kBlkBytes = static_cast<uint64_t>(curSz) * hkb;
            uint64_t vBlkBytes = static_cast<uint64_t>(curSz) * hvb;
            bool use32 = (kBlkBytes <= static_cast<uint64_t>(INT32_MAX) &&
                          vBlkBytes <= static_cast<uint64_t>(INT32_MAX));

            if (use32) {
                if (!kNC) {
                    CopyBlockContig<uint32_t, THREADS_32>(kc, ko, static_cast<uint64_t>(blockId) * cbbK, kBlkBytes, isZero, keyOff);
                } else {
                    CopyBlockNonContig<uint32_t, THREADS_32>(kc, ko, blockId, curSz, ncF, hkb, kByte, cbs, td->numHeadsK,
                        td->kCacheStride0, td->kCacheStride1, td->kCacheStride2, td->keyOutStride1,
                        (ncF & NC_K_SLOT) != 0, (ncF & NC_K_HEAD) != 0, (ncF & NC_K_OUT) != 0, isZero, keyOff);
                }
                if (!vNC) {
                    CopyBlockContig<uint32_t, THREADS_32>(vc, vo, static_cast<uint64_t>(blockId) * cbbV, vBlkBytes, isZero, valOff);
                } else {
                    CopyBlockNonContig<uint32_t, THREADS_32>(vc, vo, blockId, curSz, ncF, hvb, vByte, cbs, td->numHeadsV,
                        td->vCacheStride0, td->vCacheStride1, td->vCacheStride2, td->valueOutStride1,
                        (ncF & NC_V_SLOT) != 0, (ncF & NC_V_HEAD) != 0, (ncF & NC_V_OUT) != 0, isZero, valOff);
                }
            } else {
                if (!kNC) {
                    CopyBlockContig<uint64_t, THREADS_64>(kc, ko, static_cast<uint64_t>(blockId) * cbbK, kBlkBytes, isZero, keyOff);
                } else {
                    CopyBlockNonContig<uint64_t, THREADS_64>(kc, ko, blockId, curSz, ncF, hkb, kByte, cbs, td->numHeadsK,
                        td->kCacheStride0, td->kCacheStride1, td->kCacheStride2, td->keyOutStride1,
                        (ncF & NC_K_SLOT) != 0, (ncF & NC_K_HEAD) != 0, (ncF & NC_K_OUT) != 0, isZero, keyOff);
                }
                if (!vNC) {
                    CopyBlockContig<uint64_t, THREADS_64>(vc, vo, static_cast<uint64_t>(blockId) * cbbV, vBlkBytes, isZero, valOff);
                } else {
                    CopyBlockNonContig<uint64_t, THREADS_64>(vc, vo, blockId, curSz, ncF, hvb, vByte, cbs, td->numHeadsV,
                        td->vCacheStride0, td->vCacheStride1, td->vCacheStride2, td->valueOutStride1,
                        (ncF & NC_V_SLOT) != 0, (ncF & NC_V_HEAD) != 0, (ncF & NC_V_OUT) != 0, isZero, valOff);
                }
            }
        }
    }

    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);

    PRINTF("[SIMT] gather_pa_kv_cache ND exit, core=%lld\n",
           static_cast<long long>(GetBlockIdx()));
}

} // namespace NsGatherPaKvCache
#endif // GATHER_PA_KV_CACHE_SIMT_H
