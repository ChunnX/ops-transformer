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
 * \brief SIMT kernel for gather_pa_kv_cache ND mode (V35/arch35).
 * Replaces SIMD(DataCopyPad/UB) with SIMT direct-GM grid-stride.
 */

#ifndef GATHER_PA_KV_CACHE_ND_H_
#define GATHER_PA_KV_CACHE_ND_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"

namespace GatherPaKvCacheV35 {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint64_t MAX_BLOCK_CNT = 4095;

// Thread count depends on index type (matching scatter SIMT pattern):
// 64-bit indices use more registers, need fewer threads to avoid spill.
constexpr size_t  SIZEOF_32BIT_IDX = 4;
constexpr size_t  SIZEOF_64BIT_IDX = 8;
constexpr uint32_t THREADS_FOR_32BIT_IDX = 2048;
constexpr uint32_t THREADS_FOR_64BIT_IDX = 1024;
template <typename IDX_T>
static constexpr uint32_t THREADS = (sizeof(IDX_T) == SIZEOF_32BIT_IDX) ? THREADS_FOR_32BIT_IDX : THREADS_FOR_64BIT_IDX;

// ===== Standalone VF kernel =====
// Mirrors original SIMD V35 Process logic exactly:
// - numTokens_ truncation (accumLen guard)
// - batchOffset >= numTokens_ early break
// - isFilledWithZero for out-of-bounds blockId

template <typename T_INDEX, bool IS_CUMSUM, bool HAS_SEQOFFSET>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<T_INDEX>) inline void GatherKvNdContigVf(
    int32_t batchStart, int32_t batchEnd, int64_t coreOffset,
    int32_t blockSize, int32_t numblkTabCol, int32_t tokenSizeK, int32_t tokenSizeV,
    int32_t numBlocks, int64_t numTokens,
    __gm__ uint8_t *keyCache, __gm__ uint8_t *valueCache,
    __gm__ T_INDEX *blockTables, __gm__ T_INDEX *seqLens, __gm__ T_INDEX *seqOffset,
    __gm__ uint8_t *keyOut, __gm__ uint8_t *valueOut)
{
    uint64_t tid = static_cast<uint64_t>(threadIdx.x);
    uint64_t tNum = static_cast<uint64_t>(blockDim.x);
    int64_t kvGmCoreOffset = coreOffset;

    for (int32_t b = batchStart; b < batchEnd; b++) {
        // 1. Compute seqLen and batchOffset (matches original CalcKvCoreOffset pattern)
        int64_t seqLen;
        int64_t batchOffset;
        if (IS_CUMSUM) {
            seqLen      = static_cast<int64_t>(seqLens[b + 1]) - static_cast<int64_t>(seqLens[b]);
            batchOffset = static_cast<int64_t>(seqLens[b]);
        } else {
            seqLen      = static_cast<int64_t>(seqLens[b]);
            batchOffset = kvGmCoreOffset;
        }

        // 2. numTokens_ early-break truncation (matches original V35 Process)
        if (batchOffset >= numTokens) {
            break;
        }
        int64_t accumSeqLen = batchOffset + seqLen;
        if (numTokens > batchOffset && numTokens <= accumSeqLen) {
            seqLen = numTokens - batchOffset;
        }

        // 3. seqOffset
        int64_t seqOff = 0;
        if (HAS_SEQOFFSET) {
            seqOff = static_cast<int64_t>(seqOffset[b]) / blockSize;
        }

        int64_t blockCount = (seqLen + blockSize - 1) / blockSize;

        // 4. Per-block copy
        int64_t blkKOffset = batchOffset * tokenSizeK;
        int64_t blkVOffset = batchOffset * tokenSizeV;

        for (int64_t blkIdx = 0; blkIdx < blockCount; blkIdx++) {
            int64_t tableOffset = blkIdx + seqOff;

            // 4a. BlockId boundary check (isFilledWithZero)
            bool isFilledWithZero = false;
            int64_t blockId = 0;
            if (tableOffset < 0 || tableOffset >= numblkTabCol) {
                isFilledWithZero = true;
            } else {
                blockId = static_cast<int64_t>(blockTables[b * static_cast<int64_t>(numblkTabCol) + tableOffset]);
                if (blockId < 0 || blockId >= numBlocks) {
                    isFilledWithZero = true;
                    blockId = 0;
                }
            }

            // 4b. Tail block truncation
            int64_t curBlockSize = blockSize;
            if (blkIdx == blockCount - 1) {
                curBlockSize = seqLen - (blockCount - 1) * blockSize;
            }

            // 4c. Current block offsets (accumulated blkKOffset/blkVOffset)
            int64_t keyOffset = blkKOffset + blkIdx * static_cast<int64_t>(blockSize) * tokenSizeK;
            int64_t valOffset = blkVOffset + blkIdx * static_cast<int64_t>(blockSize) * tokenSizeV;
            int64_t keyCacheOff = blockId * static_cast<int64_t>(blockSize) * tokenSizeK;
            int64_t valCacheOff = blockId * static_cast<int64_t>(blockSize) * tokenSizeV;

            uint64_t keyBytes = static_cast<uint64_t>(curBlockSize) * static_cast<uint64_t>(tokenSizeK);
            uint64_t valBytes = static_cast<uint64_t>(curBlockSize) * static_cast<uint64_t>(tokenSizeV);

            // Key grid-stride copy
            for (uint64_t i = tid; i < keyBytes; i += tNum) {
                if (isFilledWithZero) {
                    keyOut[keyOffset + static_cast<int64_t>(i)] = 0;
                } else {
                    keyOut[keyOffset + static_cast<int64_t>(i)] = keyCache[keyCacheOff + static_cast<int64_t>(i)];
                }
            }

            // Value grid-stride copy
            for (uint64_t j = tid; j < valBytes; j += tNum) {
                if (isFilledWithZero) {
                    valueOut[valOffset + static_cast<int64_t>(j)] = 0;
                } else {
                    valueOut[valOffset + static_cast<int64_t>(j)] = valueCache[valCacheOff + static_cast<int64_t>(j)];
                }
            }
        }

        // 5. Advance kvGmCoreOffset for next batch (non-cumsum mode)
        if (!IS_CUMSUM) {
            kvGmCoreOffset += seqLen;
        }
    }
}

// ===== Class wrapper (backward compatible with GATHER_PA_KV_CACHE_APT.CPP) =====

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
class GatherPaKvCacheNd {
public:
    __aicore__ inline GatherPaKvCacheNd(TPipe *pipe, const GatherPaKvCacheTilingDataV35 *__restrict tiling)
        : pipe_(pipe), tl_(tiling) {}
    __aicore__ inline void Init(GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens,
                                GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset, GM_ADDR key_out,
                                GM_ADDR value_out);
    __aicore__ inline void Process();

    // Legacy stubs required by external test/linker code
    __aicore__ inline void InitParams() {}
    __aicore__ inline void GatherKvCache(GlobalTensor<T> dstCacheGm, GlobalTensor<T> srcCacheGm, uint64_t curLen,
                                         bool isFilledWithZero) {}
    __aicore__ inline void GatherKvCacheNonContiguous(
        GlobalTensor<T> dstGm, GlobalTensor<T> srcGm,
        uint64_t curBlockSize, uint64_t hiddenSize,
        int64_t stride0, int64_t stride1, int64_t stride2,
        bool isSlotNonContig, bool isHeadNonContig,
        bool isFilledWithZero, bool isKey) {}
    __aicore__ inline void GatherNcContigRun(GlobalTensor<T> dstGm, GlobalTensor<T> srcGm,
                                             uint64_t dstByteOff, uint64_t srcByteOff,
                                             uint64_t totalBytes, bool isFilledWithZero) {}
    __aicore__ inline void GatherNcSlotBatchRun(GlobalTensor<T> dstGm, GlobalTensor<T> srcGm,
                                                uint64_t dstBaseOff, uint64_t srcBaseOff,
                                                uint64_t numSlots, uint64_t unitBytes,
                                                uint64_t srcSlotStrideBytes, uint64_t dstSlotStrideBytes,
                                                bool isFilledWithZero) {}
    __aicore__ inline T_INDEX CalcKvCoreOffset(int64_t reduceLen) { return 0; }

private:
    TPipe *pipe_;
    const GatherPaKvCacheTilingDataV35 *tl_;

    __gm__ uint8_t *keyCacheGm_;
    __gm__ uint8_t *valueCacheGm_;
    __gm__ T_INDEX  *blockTablesGm_;
    __gm__ T_INDEX  *seqLensGm_;
    __gm__ T_INDEX  *seqOffsetGm_;
    __gm__ uint8_t *keyOutGm_;
    __gm__ uint8_t *valueOutGm_;

    uint32_t batchPerCore_;
    uint32_t needCoreNum_;
    int64_t  batchCount_;
    int64_t  cacheBlockSize_;
    int64_t  blockTableWidth_;
    int64_t  numBlocks_;
    uint64_t hiddenSizeK_;
    uint64_t hiddenSizeV_;
    int64_t  numTokens_;

    // Non-contiguous compat (unused in SIMT)
    uint32_t nonContiguousFlag_;
    uint64_t maxUbHiddenSizeK_, maxUbHiddenSizeV_, maxUbHiddenSize_;
    uint32_t seqLenAccSize_;
    int64_t  kCacheStride0_, kCacheStride1_, kCacheStride2_;
    int64_t  vCacheStride0_, vCacheStride1_, vCacheStride2_;
    int64_t  keyOutStride0_, keyOutStride1_;
    int64_t  valueOutStride0_, valueOutStride1_;
    uint64_t numHeadsK_, numHeadsV_;

    // SIMD compat stubs
    TPipe *_pipe_stub;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> cacheQueue_;
    TQue<QuePosition::VECIN, 1> seqLensQueue_;
    TBuf<TPosition::VECCALC> prefixSumBuffer_;
    DataCopyPadExtParams<T> padExtParams_;
};

template <typename T, typename T_INDEX, bool isC, bool isS>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isC, isS>::Init(
    GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens,
    GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset, GM_ADDR key_out, GM_ADDR value_out)
{
    batchPerCore_   = static_cast<uint32_t>(tl_->batchPerCore);
    needCoreNum_    = tl_->needCoreNum;
    batchCount_     = tl_->batchCount;
    cacheBlockSize_ = tl_->kvCacheBlockSize;
    blockTableWidth_ = tl_->blockTableWidth;
    numBlocks_       = tl_->numBlocks;
    hiddenSizeK_     = tl_->hiddenSizeK;
    hiddenSizeV_     = tl_->hiddenSizeV;
    numTokens_       = tl_->numTokens;
    nonContiguousFlag_ = tl_->nonContiguousFlag;

    keyCacheGm_    = (__gm__ uint8_t *)key_cache;
    valueCacheGm_  = (__gm__ uint8_t *)value_cache;
    blockTablesGm_ = (__gm__ T_INDEX *)block_tables;
    seqLensGm_     = (__gm__ T_INDEX *)seq_lens;
    seqOffsetGm_   = (__gm__ T_INDEX *)seq_offset;
    keyOutGm_      = (__gm__ uint8_t *)key_out;
    valueOutGm_    = (__gm__ uint8_t *)value_out;
}

template <typename T, typename T_INDEX, bool isC, bool isS>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isC, isS>::Process()
{
    if (needCoreNum_ == 0) return;

    int64_t batchStart = static_cast<int64_t>(GetBlockIdx()) * batchPerCore_;
    int64_t batchEnd   = batchStart + batchPerCore_;
    if (GetBlockIdx() == needCoreNum_ - 1) {
        batchEnd = batchCount_;
    }
    if (batchStart >= batchCount_) return;

    // coreOffset: cumsum→seqLens[batchStart]; non-cumsum→sum_{0}^{batchStart-1}
    T_INDEX coreOffset = 0;
    if constexpr (isC) {
        coreOffset = seqLensGm_[batchStart];
    } else {
        for (int64_t i = 0; i < batchStart; i++) {
            coreOffset += seqLensGm_[i];
        }
    }

    asc_vf_call<GatherKvNdContigVf<T_INDEX, isC, isS>>(dim3(THREADS<T_INDEX>),
        static_cast<int32_t>(batchStart), static_cast<int32_t>(batchEnd),
        static_cast<int64_t>(coreOffset),
        static_cast<int32_t>(cacheBlockSize_),
        static_cast<int32_t>(blockTableWidth_),
        static_cast<int32_t>(hiddenSizeK_),
        static_cast<int32_t>(hiddenSizeV_),
        static_cast<int32_t>(numBlocks_),
        numTokens_,
        keyCacheGm_, valueCacheGm_, blockTablesGm_, seqLensGm_, seqOffsetGm_,
        keyOutGm_, valueOutGm_);

    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
}

} // namespace GatherPaKvCacheV35
#endif
