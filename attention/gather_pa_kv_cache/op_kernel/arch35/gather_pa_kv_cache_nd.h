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
 * Pattern matches scatter_pa_kv_cache_with_k_scale_simt.h: standalone VF
 * kernels + __launch_bounds__ + asc_vf_call.
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
constexpr uint32_t THREAD_NUM = 2048;
constexpr uint64_t MAX_BLOCK_CNT = 4095;

// ===== Standalone VF kernel: continuous path =====
// All threads redundantly iterate batch/block loops;
// data copy uses grid-stride (threadIdx + blockDim).
// blockIdx/gridDim default to 0/1 from dim3(THREAD_NUM).

template <typename T_INDEX, bool IS_CUMSUM, bool HAS_SEQOFFSET>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void GatherKvNdContigVf(
    int32_t batchStart, int32_t batchEnd, int32_t tokenStart,
    int32_t blockSize, int32_t numblkTabCol, int32_t tokenSizeK, int32_t tokenSizeV,
    __gm__ uint8_t *keyCache, __gm__ uint8_t *valueCache,
    __gm__ T_INDEX *blockTables, __gm__ T_INDEX *seqLens, __gm__ T_INDEX *seqOffset,
    __gm__ uint8_t *keyOut, __gm__ uint8_t *valueOut)
{
    uint64_t tid = static_cast<uint64_t>(threadIdx.x);
    uint64_t tNum = static_cast<uint64_t>(blockDim.x);
    int32_t curTok = tokenStart;

    if (tid == 0) {
        PRINTF("[SIMT GatherKvNdContigVf] batch[%d,%d) tokStart=%d bs=%d hK=%d hV=%d cumsum=%d so=%d\n",
               batchStart, batchEnd, tokenStart, blockSize, tokenSizeK, tokenSizeV,
               (int)IS_CUMSUM, (int)HAS_SEQOFFSET);
    }

    for (int32_t b = batchStart; b < batchEnd; b++) {
        int32_t numToken;
        if (IS_CUMSUM) {
            numToken = static_cast<int32_t>(seqLens[b + 1]) - curTok;
        } else {
            numToken = static_cast<int32_t>(seqLens[b]);
        }

        if (numToken <= 0) {
            if (IS_CUMSUM) {
                curTok = static_cast<int32_t>(seqLens[b + 1]);
            }
            continue;
        }

        int32_t nBlk = (numToken + blockSize - 1) / blockSize;
        int32_t lastTok = numToken % blockSize;
        lastTok = (lastTok == 0) ? blockSize : lastTok;

        int32_t tblOff = 0;
        if (HAS_SEQOFFSET) {
            tblOff = static_cast<int32_t>(seqOffset[b]) / blockSize;
        }

        for (int32_t blk = 0; blk < nBlk; blk++) {
            int32_t blockID = static_cast<int32_t>(blockTables[b * numblkTabCol + tblOff + blk]);
            int32_t curSize = (blk == nBlk - 1) ? lastTok : blockSize;

            // Key: byte-level grid-stride
            int64_t ko = (static_cast<int64_t>(curTok) + static_cast<int64_t>(blk) * blockSize) * tokenSizeK;
            int64_t ki = static_cast<int64_t>(blockID) * blockSize * tokenSizeK;
            uint64_t kb = static_cast<uint64_t>(curSize) * static_cast<uint64_t>(tokenSizeK);
            for (uint64_t i = tid; i < kb; i += tNum) {
                keyOut[ko + static_cast<int64_t>(i)] = keyCache[ki + static_cast<int64_t>(i)];
            }

            // Value: byte-level grid-stride
            int64_t vo = (static_cast<int64_t>(curTok) + static_cast<int64_t>(blk) * blockSize) * tokenSizeV;
            int64_t vi = static_cast<int64_t>(blockID) * blockSize * tokenSizeV;
            uint64_t vb = static_cast<uint64_t>(curSize) * static_cast<uint64_t>(tokenSizeV);
            for (uint64_t j = tid; j < vb; j += tNum) {
                valueOut[vo + static_cast<int64_t>(j)] = valueCache[vi + static_cast<int64_t>(j)];
            }
        }

        if (IS_CUMSUM) {
            curTok = static_cast<int32_t>(seqLens[b + 1]);
        } else {
            curTok += numToken;
        }
    }
}

// ===== Class wrapper (backward compatible with GATHER_PA_KV_CACHE_APT.CPP) =====
// Maintains the exact same public interface as the original SIMD class.
// Internally converts to a flat Process-style call + asc_vf_call.

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
class GatherPaKvCacheNd {
public:
    __aicore__ inline GatherPaKvCacheNd(TPipe *pipe, const GatherPaKvCacheTilingDataV35 *__restrict tiling)
        : pipe_(pipe), tl_(tiling) {}
    __aicore__ inline void Init(GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens,
                                GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset, GM_ADDR key_out,
                                GM_ADDR value_out);
    __aicore__ inline void Process();

    // Legacy stubs required by previous external code
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

    // Non-contiguous fields (retained for tiling data compatibility)
    uint32_t nonContiguousFlag_;
    uint64_t maxUbHiddenSizeK_;
    uint64_t maxUbHiddenSizeV_;
    uint64_t maxUbHiddenSize_;
    uint32_t seqLenAccSize_;
    int64_t  kCacheStride0_, kCacheStride1_, kCacheStride2_;
    int64_t  vCacheStride0_, vCacheStride1_, vCacheStride2_;
    int64_t  keyOutStride0_, keyOutStride1_;
    int64_t  valueOutStride0_, valueOutStride1_;
    uint64_t numHeadsK_, numHeadsV_;

    // SIMD compat stubs (no-op, VF handles all work)
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
    int64_t batchStart = static_cast<int64_t>(GetBlockIdx()) * batchPerCore_;
    int64_t batchEnd   = batchStart + batchPerCore_;
    if (GetBlockIdx() == needCoreNum_ - 1) {
        batchEnd = batchCount_;
    }
    if (batchStart >= batchCount_) {
        return;
    }

    // coreOffset for non-cumsum mode
    T_INDEX coreOffset = 0;
    if constexpr (isC) {
        coreOffset = seqLensGm_[batchStart];
    } else {
        for (int64_t i = 0; i < batchStart; i++) {
            coreOffset += seqLensGm_[i];
        }
    }

    int32_t bSz  = static_cast<int32_t>(cacheBlockSize_);
    int32_t btW  = static_cast<int32_t>(blockTableWidth_);
    int32_t tszK = static_cast<int32_t>(hiddenSizeK_);
    int32_t tszV = static_cast<int32_t>(hiddenSizeV_);

    asc_vf_call<GatherKvNdContigVf<T_INDEX, isC, isS>>(dim3(THREAD_NUM),
        static_cast<int32_t>(batchStart), static_cast<int32_t>(batchEnd),
        static_cast<int32_t>(coreOffset),
        bSz, btW, tszK, tszV,
        keyCacheGm_, valueCacheGm_, blockTablesGm_, seqLensGm_, seqOffsetGm_,
        keyOutGm_, valueOutGm_);

    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
}

} // namespace GatherPaKvCacheV35
#endif
