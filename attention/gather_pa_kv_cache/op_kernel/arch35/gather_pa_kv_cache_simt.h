/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GATHER_PA_KV_CACHE_SIMT_H_
#define GATHER_PA_KV_CACHE_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_sync_functions.h"

namespace NsGatherPaKvCacheNd {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

// ==================== VF-A: Contiguous block copy ====================

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM)
inline void GatherContigCopyVf(uint64_t count, __gm__ T *dst, __gm__ T *src, uint32_t fillZero)
{
    if (static_cast<uint32_t>(blockIdx.x) == 0 && static_cast<uint32_t>(threadIdx.x) == 0) {
        AscendC::Simt::printf("[SIMT-VF-Contig] count=%llu fillZero=%u\n", count, fillZero);
    }
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * THREAD_NUM + static_cast<uint64_t>(threadIdx.x);
         idx < count; idx += THREAD_NUM * static_cast<uint64_t>(gridDim.x)) {
        if (fillZero) {
            dst[idx] = static_cast<T>(0);
        } else {
            dst[idx] = src[idx];
        }
    }
}

// ==================== Process ====================

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void Process(
    GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR blockTables, GM_ADDR seqLens,
    GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR seqOffset,
    GM_ADDR keyOut, GM_ADDR valueOut,
    const GatherPaKvCacheTilingDataV35 *tl)
{
    __gm__ T *keyCacheGm = (__gm__ T *)keyCache;
    __gm__ T *valueCacheGm = (__gm__ T *)valueCache;
    __gm__ T_INDEX *blockTablesGm = (__gm__ T_INDEX *)blockTables;
    __gm__ T_INDEX *seqLensGm = (__gm__ T_INDEX *)seqLens;
    __gm__ T_INDEX *seqOffGm = (__gm__ T_INDEX *)seqOffset;
    __gm__ T *outKeyGm = (__gm__ T *)keyOut;
    __gm__ T *outValueGm = (__gm__ T *)valueOut;

    int64_t cacheBlockSize = tl->kvCacheBlockSize;
    uint32_t batchPerCore = tl->batchPerCore;
    uint32_t needCoreNum = tl->needCoreNum;
    int64_t batchCount = tl->batchCount;
    int64_t blockTableWidth = tl->blockTableWidth;
    uint64_t hiddenSizeK = tl->hiddenSizeK * sizeof(DTYPE_KEY);
    uint64_t hiddenSizeV = tl->hiddenSizeV * sizeof(DTYPE_VALUE);
    int64_t numBlocks = tl->numBlocks;
    int64_t numTokens = tl->numTokens;
    uint32_t nonContiguousFlag = tl->nonContiguousFlag;

    uint64_t hiddenSize = (hiddenSizeK > hiddenSizeV) ? hiddenSizeK : hiddenSizeV;
    // Non-contiguous path uses maxUbHiddenSize from tiling; contiguous path doesn't need it for SIMT
    uint64_t maxUb = tl->maxUbHiddenSize;
    int64_t kCacheStride0 = tl->kCacheStride0;
    int64_t kCacheStride1 = tl->kCacheStride1;
    int64_t kCacheStride2 = tl->kCacheStride2;
    int64_t vCacheStride0 = tl->vCacheStride0;
    int64_t vCacheStride1 = tl->vCacheStride1;
    int64_t vCacheStride2 = tl->vCacheStride2;
    int64_t keyOutStride0 = tl->keyOutStride0;
    int64_t keyOutStride1 = tl->keyOutStride1;
    int64_t valueOutStride0 = tl->valueOutStride0;
    int64_t valueOutStride1 = tl->valueOutStride1;
    uint64_t numHeadsK = tl->numHeadsK;
    uint64_t numHeadsV = tl->numHeadsV;
    uint64_t elemSizeK = sizeof(DTYPE_KEY);
    uint64_t elemSizeV = sizeof(DTYPE_VALUE);

    PRINTF("[SIMT-ND] core=%d batchCount=%ld blockTableWidth=%ld numTokens=%ld nonContiguousFlag=%u hiddenSizeK=%lu hiddenSizeV=%lu cacheBlockSize=%ld\n",
           GetBlockIdx(), batchCount, blockTableWidth, numTokens, nonContiguousFlag, hiddenSizeK, hiddenSizeV, cacheBlockSize);

    // Non-contiguous: global block round-robin distribution (uses full core count)
    if (nonContiguousFlag != 0) {
        const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t coreNum = static_cast<uint64_t>(needCoreNum);
        uint64_t globalBlk = 0;
        int64_t runningOffset = 0;

        for (int64_t i = 0; i < batchCount; i++) {
            int64_t seqLen;
            int64_t batchOffset, accumSeqLen;
            if constexpr (isSeqLensCumsum) {
                seqLen = static_cast<int64_t>(seqLensGm[i + 1]) - static_cast<int64_t>(seqLensGm[i]);
                batchOffset = static_cast<int64_t>(seqLensGm[i]);
                accumSeqLen = static_cast<int64_t>(seqLensGm[i + 1]);
            } else {
                seqLen = static_cast<int64_t>(seqLensGm[i]);
                batchOffset = runningOffset;
                accumSeqLen = runningOffset + seqLen;
                runningOffset = accumSeqLen;
            }
            if (batchOffset >= numTokens) {
                break;
            }
            if (numTokens > batchOffset && numTokens <= accumSeqLen) {
                seqLen = numTokens - batchOffset;
            }

            uint64_t seqOff = 0;
            if constexpr (hasSeqOffset) {
                seqOff = static_cast<uint64_t>(seqOffGm[i]) / static_cast<uint64_t>(cacheBlockSize);
            }

            uint64_t blkCnt = static_cast<uint64_t>((seqLen + cacheBlockSize - 1) / cacheBlockSize);
            uint64_t _seqLen = static_cast<uint64_t>(seqLen);

            for (uint64_t blkIdx = 0; blkIdx < blkCnt; blkIdx++) {
                if ((globalBlk + blkIdx) % coreNum != coreIdx) {
                    continue;
                }

                uint64_t tblOff = blkIdx + seqOff;
                uint32_t fillZero = 0;
                int64_t blockId = 0;
                if (tblOff >= static_cast<uint64_t>(blockTableWidth)) {
                    fillZero = 1;
                } else {
                    blockId = static_cast<int64_t>(blockTablesGm[static_cast<uint64_t>(blockTableWidth) *
                                                                  static_cast<uint64_t>(i) +
                                                              tblOff]);
                    if (blockId >= numBlocks || blockId < 0) {
                        fillZero = 1;
                        blockId = 0;
                    }
                }

                uint64_t curSize = static_cast<uint64_t>(cacheBlockSize);
                if (blkIdx == blkCnt - 1) {
                    curSize = _seqLen - (blkCnt - 1) * static_cast<uint64_t>(cacheBlockSize);
                }

                bool isKSlotNC = (nonContiguousFlag >> 4) & 1;
                bool isKHeadNC = (nonContiguousFlag >> 5) & 1;
                bool isVSlotNC = (nonContiguousFlag >> 6) & 1;
                bool isVHeadNC = (nonContiguousFlag >> 7) & 1;
                bool isKOutNC  = (nonContiguousFlag >> 2) & 1;
                bool isVOutNC  = (nonContiguousFlag >> 3) & 1;

                uint64_t baseToken = static_cast<uint64_t>(batchOffset) + blkIdx * static_cast<uint64_t>(cacheBlockSize);
                uint64_t keyBaseOff = isKOutNC ? (baseToken * static_cast<uint64_t>(keyOutStride0) * elemSizeK)
                                               : (baseToken * hiddenSizeK);
                uint64_t valueBaseOff = isVOutNC ? (baseToken * static_cast<uint64_t>(valueOutStride0) * elemSizeV)
                                                 : (baseToken * hiddenSizeV);

                uint64_t keySrcBase = static_cast<uint64_t>(blockId) * static_cast<uint64_t>(kCacheStride0) * elemSizeK;
                uint64_t valueSrcBase = static_cast<uint64_t>(blockId) * static_cast<uint64_t>(vCacheStride0) * elemSizeV;

                if (!isKSlotNC && !isKHeadNC && !isKOutNC) {
                    uint64_t kCnt = curSize * hiddenSizeK;
                    asc_vf_call<GatherContigCopyVf<T>>(dim3(THREAD_NUM), kCnt,
                                                       outKeyGm + keyBaseOff,
                                                       keyCacheGm + keySrcBase, fillZero);
                } else {
                    uint64_t kDstSlotStride = isKOutNC ? static_cast<uint64_t>(keyOutStride0) * elemSizeK : hiddenSizeK;
                    uint64_t kSrcSlotStride = static_cast<uint64_t>(kCacheStride1) * elemSizeK;
                    uint64_t kSrcHeadStride = static_cast<uint64_t>(kCacheStride2) * elemSizeK;
                    uint64_t kDstHeadStride = static_cast<uint64_t>(keyOutStride1) * elemSizeK;
                    bool isKOutHeadNC = (nonContiguousFlag >> 8) & 1;
                    bool needHeadLoop = isKHeadNC || isKOutHeadNC;
                    uint64_t kHBytes = (numHeadsK > 0) ? (hiddenSizeK / numHeadsK) : hiddenSizeK;

                    for (uint64_t t = 0; t < curSize; t++) {
                        uint64_t tokenDstOff = keyBaseOff + t * kDstSlotStride;
                        uint64_t tokenSrcOff = keySrcBase + t * kSrcSlotStride;
                        if (!needHeadLoop) {
                            asc_vf_call<GatherContigCopyVf<T>>(dim3(THREAD_NUM), hiddenSizeK,
                                                               outKeyGm + tokenDstOff,
                                                               keyCacheGm + tokenSrcOff, fillZero);
                        } else {
                            for (uint64_t h = 0; h < numHeadsK; h++) {
                                asc_vf_call<GatherContigCopyVf<T>>(dim3(THREAD_NUM), kHBytes,
                                                                   outKeyGm + tokenDstOff + h * kDstHeadStride,
                                                                   keyCacheGm + tokenSrcOff + h * kSrcHeadStride,
                                                                   fillZero);
                                SetFlag<HardEvent::V_S>(0);
                                WaitFlag<HardEvent::V_S>(0);
                            }
                        }
                    }
                }
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);

                if (!isVSlotNC && !isVHeadNC && !isVOutNC) {
                    uint64_t vCnt = curSize * hiddenSizeV;
                    asc_vf_call<GatherContigCopyVf<T>>(dim3(THREAD_NUM), vCnt,
                                                       outValueGm + valueBaseOff,
                                                       valueCacheGm + valueSrcBase, fillZero);
                } else {
                    uint64_t vDstSlotStride = isVOutNC ? static_cast<uint64_t>(valueOutStride0) * elemSizeV : hiddenSizeV;
                    uint64_t vSrcSlotStride = static_cast<uint64_t>(vCacheStride1) * elemSizeV;
                    uint64_t vSrcHeadStride = static_cast<uint64_t>(vCacheStride2) * elemSizeV;
                    uint64_t vDstHeadStride = static_cast<uint64_t>(valueOutStride1) * elemSizeV;
                    bool isVOutHeadNC = (nonContiguousFlag >> 9) & 1;
                    bool needHeadLoop = isVHeadNC || isVOutHeadNC;
                    uint64_t vHBytes = (numHeadsV > 0) ? (hiddenSizeV / numHeadsV) : hiddenSizeV;

                    for (uint64_t t = 0; t < curSize; t++) {
                        uint64_t tokenDstOff = valueBaseOff + t * vDstSlotStride;
                        uint64_t tokenSrcOff = valueSrcBase + t * vSrcSlotStride;
                        if (!needHeadLoop) {
                            asc_vf_call<GatherContigCopyVf<T>>(dim3(THREAD_NUM), hiddenSizeV,
                                                               outValueGm + tokenDstOff,
                                                               valueCacheGm + tokenSrcOff, fillZero);
                        } else {
                            for (uint64_t h = 0; h < numHeadsV; h++) {
                                asc_vf_call<GatherContigCopyVf<T>>(dim3(THREAD_NUM), vHBytes,
                                                                   outValueGm + tokenDstOff + h * vDstHeadStride,
                                                                   valueCacheGm + tokenSrcOff + h * vSrcHeadStride,
                                                                   fillZero);
                                SetFlag<HardEvent::V_S>(0);
                                WaitFlag<HardEvent::V_S>(0);
                            }
                        }
                    }
                }
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);
            }
            globalBlk += blkCnt;
        }
        return;
    }

    // ===== Contiguous path =====
    int64_t batchStart = static_cast<int64_t>(GetBlockIdx()) * static_cast<int64_t>(batchPerCore);
    int64_t batchEnd = batchStart + static_cast<int64_t>(batchPerCore);
    if (static_cast<uint32_t>(GetBlockIdx()) == needCoreNum - 1) {
        batchEnd = batchCount;
    }

    int64_t coreOff = 0;
    if constexpr (isSeqLensCumsum) {
        coreOff = static_cast<int64_t>(seqLensGm[batchStart]);
    } else {
        for (int64_t j = 0; j < batchStart; j++) {
            coreOff += static_cast<int64_t>(seqLensGm[j]);
        }
    }

    PRINTF("[SIMT-CONTIG] core=%d batchStart=%ld batchEnd=%ld coreOff=%ld needCoreNum=%u batchPerCore=%u\n",
           GetBlockIdx(), batchStart, batchEnd, coreOff, needCoreNum, batchPerCore);

    for (int64_t i = batchStart; i < batchEnd; i++) {
        int64_t seqLen;
        int64_t batchOffset, accumSeqLen;
        if constexpr (isSeqLensCumsum) {
            seqLen = static_cast<int64_t>(seqLensGm[i + 1]) - static_cast<int64_t>(seqLensGm[i]);
            batchOffset = static_cast<int64_t>(seqLensGm[i]);
            accumSeqLen = static_cast<int64_t>(seqLensGm[i + 1]);
        } else {
            seqLen = static_cast<int64_t>(seqLensGm[i]);
            batchOffset = coreOff;
            accumSeqLen = coreOff + seqLen;
            coreOff = accumSeqLen;
        }

        if (batchOffset >= numTokens) {
            break;
        }

        uint64_t _seqLen = static_cast<uint64_t>(seqLen);
        if (numTokens > batchOffset && numTokens <= accumSeqLen) {
            _seqLen = static_cast<uint64_t>(numTokens - batchOffset);
        }

        uint64_t seqOff = 0;
        if constexpr (hasSeqOffset) {
            seqOff = static_cast<uint64_t>(seqOffGm[i]) / static_cast<uint64_t>(cacheBlockSize);
        }

        uint64_t blkCnt = (_seqLen + static_cast<uint64_t>(cacheBlockSize) - 1) /
                          static_cast<uint64_t>(cacheBlockSize);
        uint64_t keyOff = static_cast<uint64_t>(batchOffset) * hiddenSizeK;
        uint64_t valueOff = static_cast<uint64_t>(batchOffset) * hiddenSizeV;

        if (i - batchStart < 3) {
            PRINTF("[SIMT-BATCH] i=%ld seqLen=%ld batchOffset=%ld _seqLen=%lu blkCnt=%lu seqOff=%lu hasSeqOff=%d\n",
                   i, seqLen, batchOffset, _seqLen, blkCnt, seqOff, hasSeqOffset ? 1 : 0);
            if (i == batchStart) {
                PRINTF("[SIMT-SEQLENS]");
                for (int64_t k = 0; k < batchCount && k < 10; k++) {
                    PRINTF(" %ld", static_cast<int64_t>(seqLensGm[k]));
                }
                PRINTF("\n");
                if (hasSeqOffset) {
                    PRINTF("[SIMT-SEQOFF]");
                    for (int64_t k = 0; k < batchCount && k < 10; k++) {
                        PRINTF(" %ld", static_cast<int64_t>(seqOffGm[k]));
                    }
                    PRINTF("\n");
                }
                PRINTF("[SIMT-BLKTBL]");
                for (int64_t k = 0; k < 5 && k < blockTableWidth; k++) {
                    PRINTF(" %ld", static_cast<int64_t>(blockTablesGm[static_cast<uint64_t>(blockTableWidth)*static_cast<uint64_t>(i) + static_cast<uint64_t>(k)]));
                }
                PRINTF("\n");
            }
        }

        for (uint64_t blkIdx = 0; blkIdx < blkCnt; blkIdx++) {
            uint64_t tblOff = blkIdx + seqOff;
            uint32_t fillZero = 0;
            int64_t blockId = 0;
            if (tblOff >= static_cast<uint64_t>(blockTableWidth)) {
                fillZero = 1;
            } else {
                blockId = static_cast<int64_t>(blockTablesGm[static_cast<uint64_t>(blockTableWidth) *
                                                              static_cast<uint64_t>(i) +
                                                          tblOff]);
                if (blockId >= numBlocks || blockId < 0) {
                    fillZero = 1;
                    blockId = 0;
                }
            }

            uint64_t curSize = static_cast<uint64_t>(cacheBlockSize);
            if (blkIdx == blkCnt - 1) {
                curSize = _seqLen - (blkCnt - 1) * static_cast<uint64_t>(cacheBlockSize);
            }

            uint64_t keySrcBase = static_cast<uint64_t>(blockId) * static_cast<uint64_t>(cacheBlockSize) * hiddenSizeK;
            uint64_t valueSrcBase = static_cast<uint64_t>(blockId) * static_cast<uint64_t>(cacheBlockSize) * hiddenSizeV;

            uint64_t kCnt = curSize * hiddenSizeK;
            uint64_t vCnt = curSize * hiddenSizeV;
            if (i == batchStart && blkIdx < 2) {
                PRINTF("[SIMT-BLK] i=%ld blk=%lu tblOff=%lu blockId=%ld fillZero=%u curSize=%lu keySrcBase=%lu keyOff=%lu kCnt=%lu hiddenSizeK=%lu cacheBlockSize=%ld numHeadsK=%lu\n",
                       i, blkIdx, tblOff, blockId, fillZero, curSize, keySrcBase, keyOff, kCnt, hiddenSizeK, cacheBlockSize, numHeadsK);
            }
            if (fillZero) {
                for (uint64_t b = 0; b < kCnt; b++) { outKeyGm[keyOff + b] = static_cast<T>(0); }
            } else {
                for (uint64_t b = 0; b < kCnt; b++) { outKeyGm[keyOff + b] = keyCacheGm[keySrcBase + b]; }
            }
            keyOff += kCnt;

            if (fillZero) {
                for (uint64_t b = 0; b < vCnt; b++) { outValueGm[valueOff + b] = static_cast<T>(0); }
            } else {
                for (uint64_t b = 0; b < vCnt; b++) { outValueGm[valueOff + b] = valueCacheGm[valueSrcBase + b]; }
            }
            valueOff += vCnt;
        }
    }
}

} // namespace NsGatherPaKvCacheNd
#endif
