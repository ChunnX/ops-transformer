/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under Apache 2.0.
 */

/*!
 * \file gather_pa_kv_cache_nd.h
 * \brief SIMT kernel for gather_pa_kv_cache ND mode (V35/arch35).
 * Replaces SIMD(DataCopyPad/UB) with SIMT direct-GM grid-stride.
 * VF kernel iterates ALL batches internally (no per-core batch splitting).
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

// Thread count per index type (scatter SIMT pattern)
constexpr size_t  SZ32 = 4;
constexpr size_t  SZ64 = 8;
constexpr uint32_t THREADS_FOR_32 = 2048;
constexpr uint32_t THREADS_FOR_64 = 1024;
template <typename IDX_T>
static constexpr uint32_t THREADS = (sizeof(IDX_T) == SZ32) ? THREADS_FOR_32 : THREADS_FOR_64;

// ===== VF kernel: iterates ALL batches, grid-stride copy =====
template <typename T_INDEX, bool IS_CUMSUM, bool HAS_SEQOFFSET>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<T_INDEX>) inline void GatherKvNdContigVf(
    int64_t batchCount, int64_t numTokens,
    int32_t blockSize, int32_t numblkTabCol, int32_t tokenSizeK, int32_t tokenSizeV,
    int32_t numBlocks,
    __gm__ uint8_t *keyCache, __gm__ uint8_t *valueCache,
    __gm__ T_INDEX *blockTables, __gm__ T_INDEX *seqLens, __gm__ T_INDEX *seqOffset,
    __gm__ uint8_t *keyOut, __gm__ uint8_t *valueOut)
{
    uint64_t tid = static_cast<uint64_t>(threadIdx.x);
    uint64_t tNum = static_cast<uint64_t>(blockDim.x);
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

        if (batchOffset >= numTokens) {
            break;
        }
        int64_t accumLen = batchOffset + seqLen;
        if (numTokens > batchOffset && numTokens <= accumLen) {
            seqLen = numTokens - batchOffset;
        }

        if (seqLen <= 0) {
            if (!IS_CUMSUM) {
                kvGmCoreOffset += seqLen; // seqLen already 0 here
            }
            continue;
        }

        int64_t seqOff = 0;
        if (HAS_SEQOFFSET) {
            seqOff = static_cast<int64_t>(seqOffset[i]) / blockSize;
        }

        int64_t blockCount = (seqLen + blockSize - 1) / blockSize;

        for (int64_t blkIdx = 0; blkIdx < blockCount; blkIdx++) {
            int64_t tblOffset = blkIdx + seqOff;

            bool isFilledWithZero = false;
            int64_t blockId = 0;
            if (tblOffset < 0 || tblOffset >= numblkTabCol) {
                isFilledWithZero = true;
            } else {
                blockId = static_cast<int64_t>(blockTables[i * numblkTabCol + tblOffset]);
                if (blockId < 0 || blockId >= numBlocks) {
                    isFilledWithZero = true;
                    blockId = 0;
                }
            }

            int64_t curBlockSize = blockSize;
            if (blkIdx == blockCount - 1) {
                curBlockSize = seqLen - (blockCount - 1) * blockSize;
            }

            int64_t keyOutBase  = (batchOffset + blkIdx * blockSize) * tokenSizeK;
            int64_t valOutBase  = (batchOffset + blkIdx * blockSize) * tokenSizeV;
            int64_t keyCacheOff = blockId * static_cast<int64_t>(blockSize) * tokenSizeK;
            int64_t valCacheOff = blockId * static_cast<int64_t>(blockSize) * tokenSizeV;

            uint64_t keyBytes = static_cast<uint64_t>(curBlockSize) * static_cast<uint64_t>(tokenSizeK);
            uint64_t valBytes = static_cast<uint64_t>(curBlockSize) * static_cast<uint64_t>(tokenSizeV);

            for (uint64_t k = tid; k < keyBytes; k += tNum) {
                keyOut[keyOutBase + static_cast<int64_t>(k)] =
                    isFilledWithZero ? static_cast<uint8_t>(0) : keyCache[keyCacheOff + static_cast<int64_t>(k)];
            }
            for (uint64_t v = tid; v < valBytes; v += tNum) {
                valueOut[valOutBase + static_cast<int64_t>(v)] =
                    isFilledWithZero ? static_cast<uint8_t>(0) : valueCache[valCacheOff + static_cast<int64_t>(v)];
            }
        }

        if (!IS_CUMSUM) {
            kvGmCoreOffset += seqLen;
        }
    }
}

// ===== Class wrapper (same public interface as original SIMD) =====

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
class GatherPaKvCacheNd {
public:
    __aicore__ inline GatherPaKvCacheNd(TPipe *pipe, const GatherPaKvCacheTilingDataV35 *__restrict tiling)
        : pipe_(pipe), tl_(tiling) {}
    __aicore__ inline void Init(GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens,
                                GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset, GM_ADDR key_out,
                                GM_ADDR value_out);
    __aicore__ inline void Process();

    // legacy stubs
    __aicore__ inline void InitParams() {}
    __aicore__ inline void GatherKvCache(GlobalTensor<T> dstCacheGm, GlobalTensor<T> srcCacheGm, uint64_t curLen, bool isFilledWithZero) {}
    __aicore__ inline void GatherKvCacheNonContiguous(GlobalTensor<T> dstGm, GlobalTensor<T> srcGm, uint64_t curBlockSize, uint64_t hiddenSize, int64_t stride0, int64_t stride1, int64_t stride2, bool isSlotNonContig, bool isHeadNonContig, bool isFilledWithZero, bool isKey) {}
    __aicore__ inline void GatherNcContigRun(GlobalTensor<T> dstGm, GlobalTensor<T> srcGm, uint64_t dstByteOff, uint64_t srcByteOff, uint64_t totalBytes, bool isFilledWithZero) {}
    __aicore__ inline void GatherNcSlotBatchRun(GlobalTensor<T> dstGm, GlobalTensor<T> srcGm, uint64_t dstBaseOff, uint64_t srcBaseOff, uint64_t numSlots, uint64_t unitBytes, uint64_t srcSlotStrideBytes, uint64_t dstSlotStrideBytes, bool isFilledWithZero) {}
    __aicore__ inline T_INDEX CalcKvCoreOffset(int64_t reduceLen) { return 0; }

private:
    TPipe *pipe_;
    const GatherPaKvCacheTilingDataV35 *tl_;
    __gm__ uint8_t *keyCacheGm_, *valueCacheGm_, *keyOutGm_, *valueOutGm_;
    __gm__ T_INDEX *blockTablesGm_, *seqLensGm_, *seqOffsetGm_;

    int64_t batchCount_, cacheBlockSize_, blockTableWidth_, numBlocks_, numTokens_;
    uint64_t hiddenSizeK_, hiddenSizeV_;

    // compat stubs
    uint32_t batchPerCore_, needCoreNum_, nonContiguousFlag_, seqLenAccSize_;
    uint64_t maxUbHiddenSizeK_, maxUbHiddenSizeV_, maxUbHiddenSize_;
    int64_t kCacheStride0_, kCacheStride1_, kCacheStride2_, vCacheStride0_, vCacheStride1_, vCacheStride2_, keyOutStride0_, keyOutStride1_, valueOutStride0_, valueOutStride1_;
    uint64_t numHeadsK_, numHeadsV_;
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
    batchCount_     = tl_->batchCount;
    cacheBlockSize_ = tl_->kvCacheBlockSize;
    blockTableWidth_ = tl_->blockTableWidth;
    numBlocks_       = tl_->numBlocks;
    hiddenSizeK_     = tl_->hiddenSizeK;
    hiddenSizeV_     = tl_->hiddenSizeV;
    numTokens_       = tl_->numTokens;

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
    asc_vf_call<GatherKvNdContigVf<T_INDEX, isC, isS>>(dim3(THREADS<T_INDEX>),
        batchCount_, numTokens_,
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
