/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under Apache 2.0.
 */

/*!
 * \file gather_pa_kv_cache_nd.h
 * \brief SIMT kernel for gather_pa_kv_cache ND mode (V35/arch35).
 */

#ifndef GATHER_PA_KV_CACHE_ND_H_
#define GATHER_PA_KV_CACHE_ND_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"

namespace GatherPaKvCacheV35 {
using namespace AscendC;

constexpr uint32_t THREADS_FOR_32 = 2048;
constexpr uint32_t THREADS_FOR_64 = 1024;
template <typename IDX_T>
static constexpr uint32_t THREADS = (sizeof(IDX_T) == 4) ? THREADS_FOR_32 : THREADS_FOR_64;

// ===== VF kernel: flat per-element grid-stride =====
template <typename T_INDEX, bool IS_CUMSUM, bool HAS_SEQOFFSET>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<T_INDEX>) inline void GatherKvNdContigVf(
    int64_t batchStart, int64_t batchEnd, int64_t coreOffset,
    int64_t numTokens, int32_t blockSize, int32_t numblkTabCol,
    int32_t tokenSizeK, int32_t tokenSizeV, int32_t numBlocks,
    __gm__ uint8_t *keyCache, __gm__ uint8_t *valueCache,
    __gm__ T_INDEX *blockTables, __gm__ T_INDEX *seqLens, __gm__ T_INDEX *seqOffset,
    __gm__ uint8_t *keyOut, __gm__ uint8_t *valueOut)
{
    uint64_t tidL = static_cast<uint64_t>(threadIdx.x);
    uint64_t tNum  = static_cast<uint64_t>(blockDim.x);
    uint64_t tk    = static_cast<uint64_t>(tokenSizeK);
    uint64_t tv    = static_cast<uint64_t>(tokenSizeV);
    uint64_t bSz   = static_cast<uint64_t>(blockSize);
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
        if (seqLen <= 0) {
            if (!IS_CUMSUM) kvGmCoreOffset += seqLen;
            continue;
        }

        int64_t seqOff = 0;
        if (HAS_SEQOFFSET) {
            seqOff = static_cast<int64_t>(seqOffset[i]) / bSz;
        }

        int64_t outKeyBase = batchOffset * static_cast<int64_t>(tk);
        int64_t outValBase = batchOffset * static_cast<int64_t>(tv);
        uint64_t totalKey = static_cast<uint64_t>(seqLen) * tk;
        uint64_t totalVal = static_cast<uint64_t>(seqLen) * tv;
        int64_t blkKeyStr = static_cast<int64_t>(bSz) * static_cast<int64_t>(tk);
        int64_t blkValStr = static_cast<int64_t>(bSz) * static_cast<int64_t>(tv);

        // Key: for idx = tid; idx < totalKey; idx += tNum:
        for (uint64_t idx = tidL; idx < totalKey; idx += tNum) {
            uint64_t j   = idx / tk;
            uint64_t sub = idx - j * tk;
            uint64_t blk = j / bSz;
            uint64_t off = j - blk * bSz;

            int64_t tableIdx = seqOff + static_cast<int64_t>(blk);
            int64_t blockIdVal = 0;
            bool zeroFill = false;
            if (tableIdx >= 0 && tableIdx < numblkTabCol) {
                blockIdVal = static_cast<int64_t>(blockTables[i * static_cast<int64_t>(numblkTabCol) + tableIdx]);
                zeroFill = (blockIdVal < 0 || blockIdVal >= numBlocks);
            } else {
                zeroFill = true;
            }

            if (zeroFill) {
                keyOut[outKeyBase + static_cast<int64_t>(idx)] = 0;
            } else {
                int64_t cacheOff = blockIdVal * blkKeyStr +
                                   static_cast<int64_t>(off) * static_cast<int64_t>(tk) +
                                   static_cast<int64_t>(sub);
                keyOut[outKeyBase + static_cast<int64_t>(idx)] = keyCache[cacheOff];
            }
        }

        // Value: for idx = tid; idx < totalVal; idx += tNum:
        for (uint64_t idx = tidL; idx < totalVal; idx += tNum) {
            uint64_t j   = idx / tv;
            uint64_t sub = idx - j * tv;
            uint64_t blk = j / bSz;
            uint64_t off = j - blk * bSz;

            int64_t tableIdx = seqOff + static_cast<int64_t>(blk);
            int64_t blockIdVal = 0;
            bool zeroFill = false;
            if (tableIdx >= 0 && tableIdx < numblkTabCol) {
                blockIdVal = static_cast<int64_t>(blockTables[i * static_cast<int64_t>(numblkTabCol) + tableIdx]);
                zeroFill = (blockIdVal < 0 || blockIdVal >= numBlocks);
            } else {
                zeroFill = true;
            }

            if (zeroFill) {
                valueOut[outValBase + static_cast<int64_t>(idx)] = 0;
            } else {
                int64_t cacheOff = blockIdVal * blkValStr +
                                   static_cast<int64_t>(off) * static_cast<int64_t>(tv) +
                                   static_cast<int64_t>(sub);
                valueOut[outValBase + static_cast<int64_t>(idx)] = valueCache[cacheOff];
            }
        }

        if (!IS_CUMSUM) kvGmCoreOffset += seqLen;
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

    int64_t batchCount_, cacheBlockSize_, blockTableWidth_, numBlocks_, numTokens_;
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
    numBlocks_       = tl_->numBlocks;
    numTokens_       = tl_->numTokens;
    batchPerCore_    = tl_->batchPerCore;
    needCoreNum_     = tl_->needCoreNum;
    hiddenSizeK_     = tl_->hiddenSizeK * sizeof(DTYPE_KEY_CACHE);
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
    if (GetBlockIdx() == needCoreNum_ - 1) batchEnd = batchCount_;
    if (batchStart >= batchCount_) return;

    T_INDEX coreOffset = 0;
    if constexpr (isC) {
        coreOffset = seqLensGm_[batchStart];
    } else {
        for (int64_t i = 0; i < batchStart; i++) coreOffset += seqLensGm_[i];
    }

    asc_vf_call<GatherKvNdContigVf<T_INDEX, isC, isS>>(dim3(THREADS<T_INDEX>),
        batchStart, batchEnd, static_cast<int64_t>(coreOffset),
        numTokens_,
        static_cast<int32_t>(cacheBlockSize_),
        static_cast<int32_t>(blockTableWidth_),
        static_cast<int32_t>(hiddenSizeK_),
        static_cast<int32_t>(hiddenSizeV_),
        static_cast<int32_t>(numBlocks_),
        keyCacheGm_, valueCacheGm_, blockTablesGm_, seqLensGm_, seqOffsetGm_,
        keyOutGm_, valueOutGm_);
}

} // namespace GatherPaKvCacheV35
#endif
