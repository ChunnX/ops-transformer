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
 * \brief SIMT kernel implementation for gather_pa_kv_cache ND mode.
 * Replaces the original SIMD (DataCopy+UB) implementation with SIMT direct-GM-access.
 */

#ifndef GATHER_PA_KV_CACHE_ND_H_
#define GATHER_PA_KV_CACHE_ND_H_

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
namespace GatherPaKvCache {

constexpr uint32_t THREAD_NUM = 2048;

template <typename T>
class GatherPaKvCacheNd {
public:
    __aicore__ inline GatherPaKvCacheNd(AscendC::TPipe *p) : pipe(p){};
    __aicore__ inline void Init(GM_ADDR keyCacheInGm, GM_ADDR valueCacheInGm, GM_ADDR blockTablesInGm,
                                GM_ADDR seqLensInGm, GM_ADDR seqOffsetInGm, GM_ADDR keyOutGm, GM_ADDR valueOutGm,
                                const GatherPaKvCacheTilingData *__restrict tilingData);
    __aicore__ inline void Process();

private:
    template <const bool IS_CUMSUM, const bool HAS_SEQSTARTS>
    static __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
    inline void GatherKvSimt(int32_t batchStart, int32_t batchEnd, int32_t tokenStart,
                             int32_t blockSize, int32_t numblkTabCol,
                             int32_t tokenSizeK, int32_t tokenSizeV,
                             __gm__ T *keyCacheGm, __gm__ T *valueCacheGm,
                             __gm__ int32_t *blockTablesGm, __gm__ int32_t *seqLensGm,
                             __gm__ int32_t *seqOffsetGm,
                             __gm__ T *keyOutGm, __gm__ T *valueOutGm);

    AscendC::TPipe *pipe;
    __gm__ T *keyCacheGm_;
    __gm__ T *valueCacheGm_;
    __gm__ int32_t *blockTablesGm_;
    __gm__ int32_t *seqLensGm_;
    __gm__ int32_t *seqOffsetGm_;
    __gm__ T *keyOutGm_;
    __gm__ T *valueOutGm_;
    int32_t blockSize_ = 0;
    int32_t numBatchs_ = 0;
    int32_t numblkTabCol_ = 0;
    int32_t tokenSizeK_ = 0;
    int32_t tokenSizeV_ = 0;
    int32_t hasSeqStarts_ = 0;
    int32_t isSeqLensCumsum_ = 0;
};

template <typename T>
__aicore__ inline void GatherPaKvCacheNd<T>::Init(GM_ADDR keyCacheInGm, GM_ADDR valueCacheInGm,
                                                  GM_ADDR blockTablesInGm, GM_ADDR seqLensInGm, GM_ADDR seqOffsetInGm,
                                                  GM_ADDR keyOutGm, GM_ADDR valueOutGm,
                                                  const GatherPaKvCacheTilingData *__restrict tilingData)
{
    blockSize_ = tilingData->blockSize;
    numBatchs_ = tilingData->numTokens;
    numblkTabCol_ = tilingData->numblkTabCol;
    tokenSizeK_ = tilingData->tokenSizeK;
    tokenSizeV_ = tilingData->tokenSizeV;
    hasSeqStarts_ = tilingData->hasSeqStarts;
    isSeqLensCumsum_ = tilingData->isSeqLensCumsum;

    keyCacheGm_ = (__gm__ T *)keyCacheInGm;
    valueCacheGm_ = (__gm__ T *)valueCacheInGm;
    blockTablesGm_ = (__gm__ int32_t *)blockTablesInGm;
    seqLensGm_ = (__gm__ int32_t *)seqLensInGm;
    seqOffsetGm_ = (__gm__ int32_t *)seqOffsetInGm;
    keyOutGm_ = (__gm__ T *)keyOutGm;
    valueOutGm_ = (__gm__ T *)valueOutGm;
}

template <typename T>
__aicore__ inline void GatherPaKvCacheNd<T>::Process()
{
    if (numBatchs_ == 0) {
        return;
    }

    int32_t coreNum = static_cast<int32_t>(AscendC::GetBlockNum());
    int32_t coreIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
    int32_t batchPerCore = (numBatchs_ + coreNum - 1) / coreNum;
    int32_t batchStart = coreIdx * batchPerCore;
    int32_t batchEnd = batchStart + batchPerCore;
    if (batchStart >= numBatchs_) {
        return;
    }
    if (batchEnd > numBatchs_) {
        batchEnd = numBatchs_;
    }

    // Compute the output token offset (tokenStart) for this core's first batch.
    // For cumsum mode: tokenStart = seqLensGm_[batchStart]
    // For non-cumsum mode: tokenStart = sum_{0}^{batchStart-1} seqLensGm_[i]
    int32_t tokenStart = 0;
    if (isSeqLensCumsum_) {
        tokenStart = seqLensGm_[batchStart];
    } else {
        for (int32_t i = 0; i < batchStart; i++) {
            tokenStart += seqLensGm_[i];
        }
    }

    // Dispatch to the correct template instantiation based on runtime flags.
    // RT_IS_CUMSUM / RT_HAS_SEQSTARTS pattern: branch on runtime flags, each branch
    // calls the corresponding template specialization.
#define GE_RT_CALL(cumsum, hasOffset)                                                          \
    asc_vf_call<GatherKvSimt<cumsum, hasOffset>>(dim3(THREAD_NUM),                              \
        batchStart, batchEnd, tokenStart, blockSize_, numblkTabCol_, tokenSizeK_, tokenSizeV_, \
        keyCacheGm_, valueCacheGm_, blockTablesGm_, seqLensGm_, seqOffsetGm_, keyOutGm_, valueOutGm_)

    if (isSeqLensCumsum_) {
        if (hasSeqStarts_) {
            GE_RT_CALL(true, true);
        } else {
            GE_RT_CALL(true, false);
        }
    } else {
        if (hasSeqStarts_) {
            GE_RT_CALL(false, true);
        } else {
            GE_RT_CALL(false, false);
        }
    }
#undef GE_RT_CALL
}

// ===== SIMT VF Kernel =====
// All threads redundantly iterate the same batch/block loops (standard SIMT pattern).
// Each thread processes elements at grid-stride (threadIdx.x + blockDim.x) within each block's data region.
//
// Token position mapping (same as SIMD golden):
//   cumsum:   token_{i+1} = token_{i} + seqLens[batch]. tokenStart = seqLens[batchStart]
//   non-cumsum: token_{i+1} = token_{i} + seqLens[batch]. tokenStart = sum of previous seqLens
//   Output offset: (tokenStart + blkIdx * blockSize) * tokenSizeK/V
//   Cache offset:  blockId * blockSize * tokenSizeK/V
template <typename T>
template <const bool IS_CUMSUM, const bool HAS_SEQSTARTS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void GatherPaKvCacheNd<T>::GatherKvSimt(
    int32_t batchStart, int32_t batchEnd, int32_t tokenStart,
    int32_t blockSize, int32_t numblkTabCol,
    int32_t tokenSizeK, int32_t tokenSizeV,
    __gm__ T *keyCacheGm, __gm__ T *valueCacheGm,
    __gm__ int32_t *blockTablesGm, __gm__ int32_t *seqLensGm,
    __gm__ int32_t *seqOffsetGm,
    __gm__ T *keyOutGm, __gm__ T *valueOutGm)
{
    uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    uint32_t tNum = static_cast<uint32_t>(blockDim.x);

    int32_t curTokenStart = tokenStart;

    for (int32_t batchId = batchStart; batchId < batchEnd; batchId++) {
        // 1. Compute numToken (seqLen) for this batch
        int32_t numToken;
        if (IS_CUMSUM) {
            numToken = seqLensGm[batchId + 1] - curTokenStart;
        } else {
            numToken = seqLensGm[batchId];
        }

        if (numToken <= 0) {
            if (IS_CUMSUM) {
                curTokenStart = seqLensGm[batchId + 1];
            }
            continue;
        }

        // 2. Compute block count and tail block size
        int32_t numBlk = (numToken + blockSize - 1) / blockSize;
        int32_t lastBlkToken = numToken % blockSize;
        lastBlkToken = (lastBlkToken == 0) ? blockSize : lastBlkToken;

        // 3. Compute seqOffset (blockTable offset)
        int32_t tableOffset = 0;
        if (HAS_SEQSTARTS) {
            tableOffset = seqOffsetGm[batchId] / blockSize;
        }

        // 4. Per-block copy
        for (int32_t tableID = 0; tableID < numBlk; tableID++) {
            int32_t blockID = blockTablesGm[batchId * numblkTabCol + tableOffset + tableID];
            int32_t curBlockSize = (tableID == numBlk - 1) ? lastBlkToken : blockSize;

            // Key copy: outputBase = (curTokenStart + tableID * blockSize) * tokenSizeK
            //         cacheBase  = blockID * blockSize * tokenSizeK
            int32_t keyOutBase = (curTokenStart + tableID * blockSize) * tokenSizeK;
            int32_t keyCacheBase = blockID * blockSize * tokenSizeK;
            int32_t keyElements = curBlockSize * tokenSizeK;
            for (int32_t i = static_cast<int32_t>(tid); i < keyElements; i += static_cast<int32_t>(tNum)) {
                keyOutGm[keyOutBase + i] = keyCacheGm[keyCacheBase + i];
            }

            // Value copy: same pattern as key
            int32_t valOutBase = (curTokenStart + tableID * blockSize) * tokenSizeV;
            int32_t valCacheBase = blockID * blockSize * tokenSizeV;
            int32_t valElements = curBlockSize * tokenSizeV;
            for (int32_t i = static_cast<int32_t>(tid); i < valElements; i += static_cast<int32_t>(tNum)) {
                valueOutGm[valOutBase + i] = valueCacheGm[valCacheBase + i];
            }
        }

        // 5. Advance tokenStart for next batch
        if (IS_CUMSUM) {
            curTokenStart = seqLensGm[batchId + 1];
        } else {
            curTokenStart += numToken;
        }
    }
}

} // namespace GatherPaKvCache

#endif // GATHER_PA_KV_CACHE_ND_H_
