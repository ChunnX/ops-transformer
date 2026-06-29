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

// ===== VF kernel: token-level grid-stride, key+value in one loop =====
template <typename T_INDEX, bool IS_CUMSUM, bool HAS_SEQOFFSET>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<T_INDEX>) inline void GatherKvNdContigVf(
    int64_t batchCount, int64_t numTokens,
    int32_t blockSize, int32_t numblkTabCol, int32_t tokenSizeK, int32_t tokenSizeV,
    int32_t numBlocks,
    int64_t magicBlk, int64_t shiftBlk,
    __gm__ uint8_t *keyCache, __gm__ uint8_t *valueCache,
    __gm__ T_INDEX *blockTables, __gm__ T_INDEX *seqLens, __gm__ T_INDEX *seqOffset,
    __gm__ uint8_t *keyOut, __gm__ uint8_t *valueOut)
{
    int64_t kvGmCoreOffset = 0;

    for (int64_t i = 0; i < batchCount; i++) {
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
        if (seqLen <= 0) { if (!IS_CUMSUM) kvGmCoreOffset += seqLen; continue; }

        int64_t seqOff = 0;
        if (HAS_SEQOFFSET) seqOff = static_cast<int64_t>(seqOffset[i]) / blockSize;

        uint64_t tk   = static_cast<uint64_t>(tokenSizeK);
        uint64_t tv   = static_cast<uint64_t>(tokenSizeV);
        uint64_t bSz  = static_cast<uint64_t>(blockSize);
        uint64_t totalK = static_cast<uint64_t>(seqLen) * tk;
        uint64_t totalV = static_cast<uint64_t>(seqLen) * tv;
        uint64_t totalMax = (totalK > totalV) ? totalK : totalV;
        int64_t outK = batchOffset * static_cast<int64_t>(tk);
        int64_t outV = batchOffset * static_cast<int64_t>(tv);
        int64_t blkKS = static_cast<int32_t>(blockSize) * tokenSizeK;
        int64_t blkVS = static_cast<int32_t>(blockSize) * tokenSizeV;
        int64_t btBase = i * static_cast<int64_t>(numblkTabCol);

        uint64_t stride = static_cast<uint64_t>(gridDim.x * blockDim.x);
        for (uint64_t idx = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
             idx < totalMax; idx += stride) {
            if (idx < totalK) {
                uint64_t j   = idx / tk;
                uint64_t sub = idx - j * tk;
                uint64_t blk = Simt::UintDiv(j, static_cast<uint64_t>(magicBlk), static_cast<uint64_t>(shiftBlk));
                uint64_t off = j - blk * bSz;

                int64_t tidx = seqOff + static_cast<int64_t>(blk);
                int64_t bid = 0;
                bool zf = (tidx < 0 || tidx >= numblkTabCol);
                if (!zf) {
                    bid = static_cast<int64_t>(blockTables[btBase + tidx]);
                    zf = (bid < 0 || bid >= numBlocks);
                }
                if (zf) {
                    keyOut[outK + static_cast<int64_t>(idx)] = 0;
                } else {
                    keyOut[outK + static_cast<int64_t>(idx)] =
                        keyCache[bid * blkKS + static_cast<int64_t>(off) * tokenSizeK + static_cast<int64_t>(sub)];
                }
            }
            if (idx < totalV) {
                uint64_t j   = idx / tv;
                uint64_t sub = idx - j * tv;
                uint64_t blk = Simt::UintDiv(j, static_cast<uint64_t>(magicBlk), static_cast<uint64_t>(shiftBlk));
                uint64_t off = j - blk * bSz;

                int64_t tidx = seqOff + static_cast<int64_t>(blk);
                int64_t bid = 0;
                bool zf = (tidx < 0 || tidx >= numblkTabCol);
                if (!zf) {
                    bid = static_cast<int64_t>(blockTables[btBase + tidx]);
                    zf = (bid < 0 || bid >= numBlocks);
                }
                if (zf) {
                    valueOut[outV + static_cast<int64_t>(idx)] = 0;
                } else {
                    valueOut[outV + static_cast<int64_t>(idx)] =
                        valueCache[bid * blkVS + static_cast<int64_t>(off) * tokenSizeV + static_cast<int64_t>(sub)];
                }
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
    uint64_t magicBlk = 0, shiftBlk = 0;
    GetUintDivMagicAndShift(magicBlk, shiftBlk, static_cast<uint64_t>(cacheBlockSize_));

    asc_vf_call<GatherKvNdContigVf<T_INDEX, isC, isS>>(dim3(THREADS<T_INDEX>),
        batchCount_, numTokens_,
        static_cast<int32_t>(cacheBlockSize_),
        static_cast<int32_t>(blockTableWidth_),
        static_cast<int32_t>(hiddenSizeK_),
        static_cast<int32_t>(hiddenSizeV_),
        static_cast<int32_t>(numBlocks_),
        static_cast<int64_t>(magicBlk), static_cast<int64_t>(shiftBlk),
        keyCacheGm_, valueCacheGm_, blockTablesGm_, seqLensGm_, seqOffsetGm_,
        keyOutGm_, valueOutGm_);
}

} // namespace GatherPaKvCacheV35
#endif
