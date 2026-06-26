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
 * \brief SIMT kernel for GatherPaKvCache ND format.
 *
 *  VEC_SCOPE (TPipe/TBuf/DataCopyPad) → SIMT (__simt_vf__ + asc_vf_call) conversion:
 *  - VF functions handle byte-level block copy via direct __gm__ pointer access
 *  - Launch helpers compute magic/shift and call asc_vf_call
 *  - Process (__aicore__ scalar) handles batch dispatch, block_tables indirection,
 *    stride decomposition for non-contiguous memory
 */

#ifndef GATHER_PA_KV_CACHE_SIMT_H
#define GATHER_PA_KV_CACHE_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_sync_functions.h"
#include "gather_pa_kv_cache_tiling_data.h"
#include "gather_pa_kv_cache_tiling_key.h"

namespace NsGatherPaKvCache {
using namespace AscendC;

constexpr size_t SIZEOF_32BIT = 4;
constexpr uint32_t THREADS_32 = 2048;
constexpr uint32_t THREADS_64 = 1024;

template <typename IDX_T>
static constexpr uint32_t THREADS = (sizeof(IDX_T) == SIZEOF_32BIT) ? THREADS_32 : THREADS_64;

// ==================== 非连续标志位 ====================
constexpr uint32_t NC_K_CACHE = 1 << 0;
constexpr uint32_t NC_V_CACHE = 1 << 1;
constexpr uint32_t NC_K_OUT   = 1 << 2;
constexpr uint32_t NC_V_OUT   = 1 << 3;
constexpr uint32_t NC_K_SLOT  = 1 << 4;
constexpr uint32_t NC_K_HEAD  = 1 << 5;
constexpr uint32_t NC_V_SLOT  = 1 << 6;
constexpr uint32_t NC_V_HEAD  = 1 << 7;

// ==================== VF: 连续块拷贝 ====================
template <typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void GatherCopyBlock(
    IDX_T byteCount, __gm__ uint8_t* dst, __gm__ uint8_t* src)
{
    for (IDX_T i = static_cast<IDX_T>(blockIdx.x) * THREADS<IDX_T> + static_cast<IDX_T>(threadIdx.x);
         i < byteCount; i += THREADS<IDX_T> * static_cast<IDX_T>(gridDim.x)) {
        dst[i] = src[i];
    }
}

// ==================== VF: 连续块填零 ====================
template <typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void GatherZeroBlock(
    IDX_T byteCount, __gm__ uint8_t* dst)
{
    for (IDX_T i = static_cast<IDX_T>(blockIdx.x) * THREADS<IDX_T> + static_cast<IDX_T>(threadIdx.x);
         i < byteCount; i += THREADS<IDX_T> * static_cast<IDX_T>(gridDim.x)) {
        dst[i] = 0;
    }
}

// ==================== VF: 跨slot连续段拷贝(非连续cache) ====================
template <typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void GatherCopyStrided(
    IDX_T byteCount, __gm__ uint8_t* dst, __gm__ uint8_t* src,
    IDX_T dstStride, IDX_T srcStride, IDX_T innerCount)
{
    for (IDX_T i = static_cast<IDX_T>(blockIdx.x) * THREADS<IDX_T> + static_cast<IDX_T>(threadIdx.x);
         i < byteCount; i += THREADS<IDX_T> * static_cast<IDX_T>(gridDim.x)) {
        IDX_T outer = i / innerCount;
        IDX_T inner = i % innerCount;
        dst[outer * dstStride + inner] = src[outer * srcStride + inner];
    }
}

// ==================== Launch Helpers ====================
template <typename IDX_T>
__aicore__ inline void LaunchCopyBlock(IDX_T byteCount, __gm__ uint8_t* dst, __gm__ uint8_t* src)
{
    asc_vf_call<GatherCopyBlock<IDX_T>>(dim3(THREADS<IDX_T>), byteCount, dst, src);
}

template <typename IDX_T>
__aicore__ inline void LaunchZeroBlock(IDX_T byteCount, __gm__ uint8_t* dst)
{
    asc_vf_call<GatherZeroBlock<IDX_T>>(dim3(THREADS<IDX_T>), byteCount, dst);
}

template <typename IDX_T>
__aicore__ inline void LaunchCopyStrided(IDX_T byteCount, __gm__ uint8_t* dst, __gm__ uint8_t* src,
                                          IDX_T dstStride, IDX_T srcStride, IDX_T innerCount)
{
    asc_vf_call<GatherCopyStrided<IDX_T>>(dim3(THREADS<IDX_T>), byteCount, dst, src,
                                           dstStride, srcStride, innerCount);
}

// ==================== CeilDiv ====================
template <typename T>
__aicore__ __attribute__((always_inline)) inline T CeilDiv(T a, T b) { return (a + b - 1) / b; }

// ==================== 连续块处理 ====================
template <typename T_INDEX, typename IDX_T>
__aicore__ inline void ProcessBlockContig(
    __gm__ uint8_t* cacheGm, __gm__ uint8_t* outGm,
    uint64_t cacheStartBytes, int64_t curBlockSize,
    uint64_t hiddenSizeBytes, bool isZero, uint64_t& outOffBytes)
{
    uint64_t totalBytes = static_cast<uint64_t>(curBlockSize) * hiddenSizeBytes;
    if (isZero) {
        LaunchZeroBlock<IDX_T>(static_cast<IDX_T>(totalBytes), outGm + outOffBytes);
    } else {
        LaunchCopyBlock<IDX_T>(static_cast<IDX_T>(totalBytes), outGm + outOffBytes, cacheGm + cacheStartBytes);
    }
    outOffBytes += totalBytes;
}

// ==================== 非连续slot拷贝 ====================
template <typename T_INDEX, typename IDX_T>
__aicore__ inline void ProcessBlockNcSlot(
    __gm__ uint8_t* cacheGm, __gm__ uint8_t* outGm,
    int64_t curBlockSize, uint64_t hiddenSizeBytes,
    uint64_t blockStartBytes, uint64_t slotSrcStride,
    uint64_t dstSlotStride, bool isZero, uint64_t& outOffBytes)
{
    uint64_t baseOff = outOffBytes;
    for (int64_t s = 0; s < curBlockSize; s++) {
        uint64_t dstOff = baseOff + static_cast<uint64_t>(s) * dstSlotStride;
        if (isZero) {
            LaunchZeroBlock<IDX_T>(static_cast<IDX_T>(hiddenSizeBytes), outGm + dstOff);
        } else {
            uint64_t srcOff = blockStartBytes + static_cast<uint64_t>(s) * slotSrcStride;
            LaunchCopyBlock<IDX_T>(static_cast<IDX_T>(hiddenSizeBytes), outGm + dstOff, cacheGm + srcOff);
        }
    }
    outOffBytes = baseOff + static_cast<uint64_t>(curBlockSize) * dstSlotStride;
}

// ==================== 非连续slot+head拷贝 ====================
template <typename T_INDEX, typename IDX_T>
__aicore__ inline void ProcessBlockNcHead(
    __gm__ uint8_t* cacheGm, __gm__ uint8_t* outGm,
    int64_t curBlockSize, uint64_t hiddenSizeBytes,
    uint64_t numHeads, uint64_t headBytes,
    uint64_t blockStartBytes, uint64_t slotSrcStride, uint64_t headSrcStride,
    uint64_t dstSlotStride, bool isZero, uint64_t& outOffBytes)
{
    uint64_t baseOff = outOffBytes;
    for (int64_t s = 0; s < curBlockSize; s++) {
        uint64_t dstSlotOff = baseOff + static_cast<uint64_t>(s) * dstSlotStride;
        for (uint64_t h = 0; h < numHeads; h++) {
            uint64_t dstOff = dstSlotOff + h * headBytes;
            if (isZero) {
                LaunchZeroBlock<IDX_T>(static_cast<IDX_T>(headBytes), outGm + dstOff);
            } else {
                uint64_t srcOff = blockStartBytes + static_cast<uint64_t>(s) * slotSrcStride + h * headSrcStride;
                LaunchCopyBlock<IDX_T>(static_cast<IDX_T>(headBytes), outGm + dstOff, cacheGm + srcOff);
            }
        }
    }
    outOffBytes = baseOff + static_cast<uint64_t>(curBlockSize) * dstSlotStride;
}

// ==================== 非连续整体调度 ====================
template <typename T_INDEX, typename IDX_T>
__aicore__ inline void ProcessBlockNonContig(
    __gm__ uint8_t* cacheGm, __gm__ uint8_t* outGm,
    int64_t blockId, int64_t curBlockSize,
    const GatherPaKvCacheSimtTilingData* t, bool isKey,
    bool isZero, uint64_t& outOffBytes)
{
    uint32_t nc = t->nonContiguousFlag;
    uint64_t hb = isKey ? t->hiddenSizeK : t->hiddenSizeV;
    uint64_t eb = isKey ? t->keyByteSize : t->valueByteSize;
    uint64_t nhs = isKey ? t->numHeadsK : t->numHeadsV;
    bool isSlotNC, isHeadNC;
    uint64_t slotSrcStride, headSrcStride, blockStartBytes;
    if (isKey) {
        isSlotNC = (nc & NC_K_SLOT) != 0;
        isHeadNC = (nc & NC_K_HEAD) != 0;
        slotSrcStride = static_cast<uint64_t>(t->kCacheStride1) * eb;
        headSrcStride = static_cast<uint64_t>(t->kCacheStride2) * eb;
        blockStartBytes = static_cast<uint64_t>(blockId) * static_cast<uint64_t>(t->kCacheStride0) * eb;
    } else {
        isSlotNC = (nc & NC_V_SLOT) != 0;
        isHeadNC = (nc & NC_V_HEAD) != 0;
        slotSrcStride = static_cast<uint64_t>(t->vCacheStride1) * eb;
        headSrcStride = static_cast<uint64_t>(t->vCacheStride2) * eb;
        blockStartBytes = static_cast<uint64_t>(blockId) * static_cast<uint64_t>(t->vCacheStride0) * eb;
    }

    uint64_t dstSlotStride = hb;
    if (isKey && (nc & NC_K_OUT)) {
        dstSlotStride = static_cast<uint64_t>(t->keyOutStride1) * eb;
    } else if (!isKey && (nc & NC_V_OUT)) {
        dstSlotStride = static_cast<uint64_t>(t->valueOutStride1) * eb;
    }

    if (!isSlotNC) {
        ProcessBlockContig<T_INDEX, IDX_T>(cacheGm, outGm, blockStartBytes, curBlockSize,
            hb, isZero, outOffBytes);
    } else if (!isHeadNC) {
        ProcessBlockNcSlot<T_INDEX, IDX_T>(cacheGm, outGm, curBlockSize, hb,
            blockStartBytes, slotSrcStride, dstSlotStride, isZero, outOffBytes);
    } else {
        uint64_t headBytes = (nhs > 0) ? (hb / nhs) : hb;
        ProcessBlockNcHead<T_INDEX, IDX_T>(cacheGm, outGm, curBlockSize, hb,
            nhs, headBytes, blockStartBytes, slotSrcStride, headSrcStride,
            dstSlotStride, isZero, outOffBytes);
    }
}

// ==================== Process 主函数 ====================
template <typename T_INDEX, uint32_t SCH_MODE>
__aicore__ inline void Process(
    GM_ADDR key_cache, GM_ADDR value_cache,
    GM_ADDR block_tables, GM_ADDR seq_lens,
    GM_ADDR key_out, GM_ADDR value_out,
    GM_ADDR seq_offset,
    const GatherPaKvCacheSimtTilingData* td)
{
    __gm__ uint8_t* kc = (__gm__ uint8_t*)key_cache;
    __gm__ uint8_t* vc = (__gm__ uint8_t*)value_cache;
    __gm__ T_INDEX* bt = (__gm__ T_INDEX*)block_tables;
    __gm__ T_INDEX* sl = (__gm__ T_INDEX*)seq_lens;
    __gm__ uint8_t* ko = (__gm__ uint8_t*)key_out;
    __gm__ uint8_t* vo = (__gm__ uint8_t*)value_out;
    __gm__ T_INDEX* so = (__gm__ T_INDEX*)seq_offset;

    int64_t bc = td->batchCount;
    int64_t bpc = td->batchPerCore;
    int64_t ncn = td->needCoreNum;
    int64_t cbs = td->kvCacheBlockSize;
    int64_t btw = td->blockTableWidth;
    int64_t nb = td->numBlocks;
    int64_t nt = td->numTokens;
    uint64_t hkb = td->hiddenSizeK;
    uint64_t hvb = td->hiddenSizeV;
    uint64_t cbbK = static_cast<uint64_t>(cbs) * hkb;  // cache block bytes (key)
    uint64_t cbbV = static_cast<uint64_t>(cbs) * hvb;  // cache block bytes (value)
    uint32_t ncF = td->nonContiguousFlag;

    int64_t coreIdx = GetBlockIdx();
    int64_t batchStart = coreIdx * bpc;
    int64_t batchEnd = batchStart + bpc;
    if (coreIdx == ncn - 1) { batchEnd = bc; }

    int64_t coreOffset = 0;
    if constexpr (SCH_MODE != GATHER_PA_KV_CACHE_SCENE_1110 &&
                  SCH_MODE != GATHER_PA_KV_CACHE_SCENE_1111) {
        for (int64_t i = 0; i < batchStart; i++) {
            coreOffset += static_cast<int64_t>(sl[i]);
        }
    }

    for (int64_t b = batchStart; b < batchEnd; b++) {
        T_INDEX seqLen;
        int64_t batchOffset;
        int64_t accum;
        if constexpr (SCH_MODE == GATHER_PA_KV_CACHE_SCENE_1110 ||
                      SCH_MODE == GATHER_PA_KV_CACHE_SCENE_1111) {
            seqLen = sl[b + 1] - sl[b];
            batchOffset = static_cast<int64_t>(sl[b]);
            accum = static_cast<int64_t>(sl[b + 1]);
        } else {
            seqLen = sl[b];
            batchOffset = coreOffset;
            accum = coreOffset + static_cast<int64_t>(sl[b]);
            coreOffset = accum;
        }
        if (batchOffset >= nt) break;
        if (nt > batchOffset && nt <= accum) {
            seqLen = static_cast<T_INDEX>(nt - batchOffset);
        }

        int64_t seqOff = 0;
        if constexpr (SCH_MODE == GATHER_PA_KV_CACHE_SCENE_1101 ||
                      SCH_MODE == GATHER_PA_KV_CACHE_SCENE_1111) {
            seqOff = static_cast<int64_t>(so[b]) / cbs;
        }

        int64_t blkCnt = static_cast<int64_t>(
            CeilDiv<T_INDEX>(seqLen, static_cast<T_INDEX>(cbs)));
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
                if (blockId >= nb || blockId < 0) {
                    isZero = true; blockId = 0;
                }
            }
            int64_t curSz = cbs;
            if (blk == blkCnt - 1) {
                curSz = static_cast<int64_t>(seqLen) - (blkCnt - 1) * cbs;
            }

            bool kCacheNC = (ncF & NC_K_CACHE) || (ncF & NC_K_SLOT);
            bool vCacheNC = (ncF & NC_V_CACHE) || (ncF & NC_V_SLOT);
            uint64_t kBlkBytes = static_cast<uint64_t>(curSz) * hkb;
            uint64_t vBlkBytes = static_cast<uint64_t>(curSz) * hvb;
            bool use32 = (kBlkBytes <= static_cast<uint64_t>(INT32_MAX) &&
                          vBlkBytes <= static_cast<uint64_t>(INT32_MAX));

            uint64_t kBlkStart = static_cast<uint64_t>(blockId) * cbbK;
            uint64_t vBlkStart = static_cast<uint64_t>(blockId) * cbbV;

            if (use32) {
                if (!kCacheNC) {
                    ProcessBlockContig<T_INDEX, uint32_t>(kc, ko, kBlkStart, curSz, hkb, isZero, keyOff);
                } else {
                    ProcessBlockNonContig<T_INDEX, uint32_t>(kc, ko, blockId, curSz, td, true, isZero, keyOff);
                }
                if (!vCacheNC) {
                    ProcessBlockContig<T_INDEX, uint32_t>(vc, vo, vBlkStart, curSz, hvb, isZero, valOff);
                } else {
                    ProcessBlockNonContig<T_INDEX, uint32_t>(vc, vo, blockId, curSz, td, false, isZero, valOff);
                }
            } else {
                if (!kCacheNC) {
                    ProcessBlockContig<T_INDEX, uint64_t>(kc, ko, kBlkStart, curSz, hkb, isZero, keyOff);
                } else {
                    ProcessBlockNonContig<T_INDEX, uint64_t>(kc, ko, blockId, curSz, td, true, isZero, keyOff);
                }
                if (!vCacheNC) {
                    ProcessBlockContig<T_INDEX, uint64_t>(vc, vo, vBlkStart, curSz, hvb, isZero, valOff);
                } else {
                    ProcessBlockNonContig<T_INDEX, uint64_t>(vc, vo, blockId, curSz, td, false, isZero, valOff);
                }
            }
        }
    }

    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
}

} // namespace NsGatherPaKvCache
#endif // GATHER_PA_KV_CACHE_SIMT_H
