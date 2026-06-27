/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under Apache 2.0.
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

namespace GatherPaKvCacheV35 {
using namespace AscendC;

// Thread count per index type (scatter SIMT pattern: 64bit indices need fewer threads)
constexpr uint32_t THREADS_FOR_32 = 2048;
constexpr uint32_t THREADS_FOR_64 = 1024;
template <typename IDX_T>
static constexpr uint32_t THREADS = (sizeof(IDX_T) == 4) ? THREADS_FOR_32 : THREADS_FOR_64;

// ===== VF kernel =====
template <typename T_INDEX, bool IS_CUMSUM, bool HAS_SEQOFFSET>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<T_INDEX>) inline void GatherKvNdContigVf(
    int64_t batchStart, int64_t batchEnd, int64_t coreOffset,
    int64_t numTokens, int32_t blockSize, int32_t numblkTabCol,
    int32_t tokenSizeK, int32_t tokenSizeV,
    __gm__ uint8_t *keyCache, __gm__ uint8_t *valueCache,
    __gm__ T_INDEX *blockTables, __gm__ T_INDEX *seqLens, __gm__ T_INDEX *seqOffset,
    __gm__ uint8_t *keyOut, __gm__ uint8_t *valueOut)
{
    uint64_t tid = static_cast<uint64_t>(threadIdx.x);
    uint64_t tNum = static_cast<uint64_t>(blockDim.x);
    int64_t kvGmCoreOffset = coreOffset;

    for (int64_t i = batchStart; i < batchEnd; i++) {
        int64_t seqLen;
        int64_t batchOffset;
        if (IS_CUMSUM) {
            seqLen      = static_cast<int64_t>(seqLens[i + 1]) - static_cast<int64_t>(seqLens[i]);
            batchOffset = static_cast<int64_t>(seqLens[i]);
        } else {
            seqLen      = static_cast<int64_t>(seqLens[i]);
            batchOffset = kvGmCoreOffset;
        }

        if (batchOffset >= numTokens) break;

        int64_t accumLen = batchOffset + seqLen;
        if (numTokens > batchOffset && numTokens <= accumLen) {
            seqLen = numTokens - batchOffset;
        }

        if (seqLen <= 0) continue;

        int64_t seqOff = 0;
        if (HAS_SEQOFFSET) {
            seqOff = static_cast<int64_t>(seqOffset[i]) / blockSize;
        }

        int64_t blockCount = (seqLen + blockSize - 1) / blockSize;

        for (int64_t blkIdx = 0; blkIdx < blockCount; blkIdx++) {
            int64_t tblOffset = blkIdx + seqOff;

            // blockId directly from blockTables (matches original SIMD continuous path,
            // which trusts the block table; no isFilledWithZero in the hot loop)
            int64_t blockId = (tblOffset < 0 || tblOffset >= numblkTabCol)
                ? 0
                : static_cast<int64_t>(blockTables[i * numblkTabCol + tblOffset]);

            int64_t curBlockSize = (blkIdx == blockCount - 1)
                ? seqLen - (blockCount - 1) * blockSize
                : blockSize;

            int64_t keyOutBase  = (batchOffset + blkIdx * blockSize) * tokenSizeK;
            int64_t valOutBase  = (batchOffset + blkIdx * blockSize) * tokenSizeV;
            int64_t keyCacheOff = blockId * blockSize * tokenSizeK;
            int64_t valCacheOff = blockId * blockSize * tokenSizeV;

            uint64_t keyBytes = static_cast<uint64_t>(curBlockSize) * static_cast<uint64_t>(tokenSizeK);
            uint64_t valBytes = static_cast<uint64_t>(curBlockSize) * static_cast<uint64_t>(tokenSizeV);

            for (uint64_t k = tid; k < keyBytes; k += tNum) {
                keyOut[keyOutBase + static_cast<int64_t>(k)] = keyCache[keyCacheOff + static_cast<int64_t>(k)];
            }
            for (uint64_t v = tid; v < valBytes; v += tNum) {
                valueOut[valOutBase + static_cast<int64_t>(v)] = valueCache[valCacheOff + static_cast<int64_t>(v)];
            }
        }

        if (!IS_CUMSUM) {
            kvGmCoreOffset += seqLen;
        }
    }
}

// ===== Class wrapper =====
template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
class GatherPaKvCacheNd {
public:
    __aicore__ inline GatherPaKvCacheNd(TPipe *pipe, const GatherPaKvCacheTilingDataV35 *__restrict tiling)
        : pipe_(pipe), tl_(tiling) {}
    __aicore__ inline void Init(GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens,
                                GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset, GM_ADDR key_out,
                                GM_ADDR value_out);
    __aicore__ inline void Process();

private:
    TPipe *pipe_;
    const GatherPaKvCacheTilingDataV35 *tl_;
    __gm__ uint8_t *keyCacheGm_, *valueCacheGm_, *keyOutGm_, *valueOutGm_;
    __gm__ T_INDEX *blockTablesGm_, *seqLensGm_, *seqOffsetGm_;

    int64_t batchCount_, cacheBlockSize_, blockTableWidth_, numTokens_;
    uint64_t hiddenSizeK_, hiddenSizeV_;
    int64_t batchPerCore_;
    uint32_t needCoreNum_;
};

template <typename T, typename T_INDEX, bool isC, bool isS>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isC, isS>::Init(
    GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens,
    GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset, GM_ADDR key_out, GM_ADDR value_out)
{
    batchCount_      = tl_->batchCount;
    cacheBlockSize_  = tl_->kvCacheBlockSize;
    blockTableWidth_ = tl_->blockTableWidth;
    numTokens_       = tl_->numTokens;
    batchPerCore_    = tl_->batchPerCore;
    needCoreNum_     = tl_->needCoreNum;
    hiddenSizeK_     = tl_->hiddenSizeK * sizeof(DTYPE_KEY_CACHE);   // elements -> bytes
    hiddenSizeV_     = tl_->hiddenSizeV * sizeof(DTYPE_KEY_CACHE);

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
    int64_t batchStart = static_cast<int64_t>(GetBlockIdx()) * batchPerCore_;
    int64_t batchEnd   = batchStart + batchPerCore_;
    if (GetBlockIdx() == needCoreNum_ - 1) {
        batchEnd = batchCount_;
    }
    if (batchStart >= batchCount_) return;

    // Compute cumulative token offset for the first batch of this core
    T_INDEX coreOffset = 0;
    if constexpr (isC) {
        coreOffset = seqLensGm_[batchStart];
    } else {
        for (int64_t i = 0; i < batchStart; i++) {
            coreOffset += seqLensGm_[i];
        }
    }

    asc_vf_call<GatherKvNdContigVf<T_INDEX, isC, isS>>(dim3(THREADS<T_INDEX>),
        batchStart, batchEnd, static_cast<int64_t>(coreOffset),
        numTokens_,
        static_cast<int32_t>(cacheBlockSize_),
        static_cast<int32_t>(blockTableWidth_),
        static_cast<int32_t>(hiddenSizeK_),
        static_cast<int32_t>(hiddenSizeV_),
        keyCacheGm_, valueCacheGm_, blockTablesGm_, seqLensGm_, seqOffsetGm_,
        keyOutGm_, valueOutGm_);
}

} // namespace GatherPaKvCacheV35
#endif
