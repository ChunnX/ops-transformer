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
 * \brief SIMT kernel implementation for gather_pa_kv_cache ND mode (V35/arch35).
 * Replaces the original SIMD (DataCopyPad/UB) ND kernel with SIMT direct-GM-access.
 */

#ifndef GATHER_PA_KV_CACHE_ND_H_
#define GATHER_PA_KV_CACHE_ND_H_

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace GatherPaKvCacheV35 {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t THREAD_NUM = 2048;
constexpr uint64_t MAX_BLOCK_CNT = 4095;  // DataCopyPad blockCount (still referenced by extern code)

// ===== UB tiling parameters passed to VF =====
// Pack into UB struct to stay under asc_vf_call 112-byte parameter limit.
struct GatherPaKvCacheSimtUb {
    int64_t  batchStart;
    int64_t  batchEnd;
    int64_t  coreOffset;
    int64_t  cacheBlockSize;
    int64_t  blockTableWidth;
    int64_t  numBlocks;
    int64_t  numTokens;
    uint64_t hiddenSizeK;
    uint64_t hiddenSizeV;
    int32_t  nonContiguousFlag;
    int64_t  kCacheStride0;
    int64_t  kCacheStride1;
    int64_t  kCacheStride2;
    int64_t  vCacheStride0;
    int64_t  vCacheStride1;
    int64_t  vCacheStride2;
    int64_t  keyOutStride0;
    int64_t  keyOutStride1;
    int64_t  valueOutStride0;
    int64_t  valueOutStride1;
    uint64_t numHeadsK;
    uint64_t numHeadsV;
};

// ===== SIMT VF Kernel: continuous path =====
// T = uint8_t, hiddenSize* are in bytes. All threads redundantly iterate batch/block loops
// and use grid-stride (threadIdx + blockDim) for data copy within each block.
// Uses compile-time template bools (IS_CUMSUM, HAS_SEQOFFSET) for efficient code paths.
template <typename T_INDEX, const bool IS_CUMSUM, const bool HAS_SEQOFFSET>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GatherKvSimtContig(
    __gm__ uint8_t *keyCache, __gm__ uint8_t *valueCache,
    __gm__ T_INDEX *blockTables, __gm__ T_INDEX *seqLens, __gm__ T_INDEX *seqOffset,
    __gm__ uint8_t *keyOut, __gm__ uint8_t *valueOut,
    __ubuf__ GatherPaKvCacheSimtUb *ub)
{
    const int64_t  batchStart         = ub->batchStart;
    const int64_t  batchEnd           = ub->batchEnd;
    int64_t        coreOffset          = ub->coreOffset;
    const int64_t  cacheBlockSize     = ub->cacheBlockSize;
    const int64_t  blockTableWidth    = ub->blockTableWidth;
    const int64_t  numBlocks          = ub->numBlocks;
    const int64_t  numTokens          = ub->numTokens;
    const uint64_t hiddenSizeK        = ub->hiddenSizeK;
    const uint64_t hiddenSizeV        = ub->hiddenSizeV;

    uint64_t tid = static_cast<uint64_t>(threadIdx.x);
    uint64_t tNum = static_cast<uint64_t>(blockDim.x);

    for (int64_t i = batchStart; i < batchEnd; i++) {
        int64_t seqLen;
        int64_t batchOffset;
        if (IS_CUMSUM) {
            seqLen = static_cast<int64_t>(seqLens[i + 1]) - static_cast<int64_t>(seqLens[i]);
            batchOffset = static_cast<int64_t>(seqLens[i]);
        } else {
            seqLen = static_cast<int64_t>(seqLens[i]);
            batchOffset = coreOffset;
        }

        if (batchOffset >= numTokens) {
            break;
        }

        int64_t accumSeqLen = batchOffset + seqLen;
        if (numTokens > batchOffset && numTokens <= accumSeqLen) {
            seqLen = numTokens - batchOffset;
        }

        int64_t seqOff = 0;
        if (HAS_SEQOFFSET) {
            seqOff = static_cast<int64_t>(seqOffset[i]) / cacheBlockSize;
        }

        int64_t blockCount = (seqLen + cacheBlockSize - 1) / cacheBlockSize;

        for (int64_t blkIdx = 0; blkIdx < blockCount; blkIdx++) {
            int64_t tableOffset = blkIdx + seqOff;

            bool isFilledWithZero = false;
            int64_t blockId = 0;
            if (tableOffset < 0 || tableOffset >= blockTableWidth) {
                isFilledWithZero = true;
            } else {
                blockId = static_cast<int64_t>(blockTables[i * blockTableWidth + tableOffset]);
                if (blockId < 0 || blockId >= numBlocks) {
                    isFilledWithZero = true;
                    blockId = 0;
                }
            }

            int64_t curBlockSize = cacheBlockSize;
            if (blkIdx == blockCount - 1) {
                curBlockSize = seqLen - (blockCount - 1) * cacheBlockSize;
            }

            // Key copy: continuous regions, bytes as uint8_t
            int64_t keyOutBase   = batchOffset * static_cast<int64_t>(hiddenSizeK) +
                                   blkIdx * cacheBlockSize * static_cast<int64_t>(hiddenSizeK);
            int64_t keyCacheBase = blockId * cacheBlockSize * static_cast<int64_t>(hiddenSizeK);
            uint64_t keyBytes = static_cast<uint64_t>(curBlockSize) * hiddenSizeK;

            for (uint64_t k = tid; k < keyBytes; k += tNum) {
                if (isFilledWithZero) {
                    keyOut[keyOutBase + static_cast<int64_t>(k)] = 0;
                } else {
                    keyOut[keyOutBase + static_cast<int64_t>(k)] = keyCache[keyCacheBase + static_cast<int64_t>(k)];
                }
            }

            // Value copy
            int64_t valOutBase   = batchOffset * static_cast<int64_t>(hiddenSizeV) +
                                   blkIdx * cacheBlockSize * static_cast<int64_t>(hiddenSizeV);
            int64_t valCacheBase = blockId * cacheBlockSize * static_cast<int64_t>(hiddenSizeV);
            uint64_t valBytes = static_cast<uint64_t>(curBlockSize) * hiddenSizeV;

            for (uint64_t v = tid; v < valBytes; v += tNum) {
                if (isFilledWithZero) {
                    valueOut[valOutBase + static_cast<int64_t>(v)] = 0;
                } else {
                    valueOut[valOutBase + static_cast<int64_t>(v)] = valueCache[valCacheBase + static_cast<int64_t>(v)];
                }
            }
        }

        if (!IS_CUMSUM) {
            coreOffset += seqLen;
        }
    }
}

// ===== SIMT VF Kernel: non-contiguous path =====
// Handles non-contiguous cache/ref addressing using stride information.
// Per-token outer loop, with optional per-head inner loop for head-non-contiguous cases.
// Fill-zero blockId check follows same logic as SIMD golden.
template <typename T_INDEX, const bool IS_CUMSUM, const bool HAS_SEQOFFSET>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GatherKvSimtNonContig(
    __gm__ uint8_t *keyCache, __gm__ uint8_t *valueCache,
    __gm__ T_INDEX *blockTables, __gm__ T_INDEX *seqLens, __gm__ T_INDEX *seqOffset,
    __gm__ uint8_t *keyOut, __gm__ uint8_t *valueOut,
    __ubuf__ GatherPaKvCacheSimtUb *ub)
{
    const int64_t  batchStart         = ub->batchStart;
    const int64_t  batchEnd           = ub->batchEnd;
    int64_t        coreOffset          = ub->coreOffset;
    const int64_t  cacheBlockSize     = ub->cacheBlockSize;
    const int64_t  blockTableWidth    = ub->blockTableWidth;
    const int64_t  numBlocks          = ub->numBlocks;
    const int64_t  numTokens          = ub->numTokens;
    const uint64_t hiddenSizeK        = ub->hiddenSizeK;
    const uint64_t hiddenSizeV        = ub->hiddenSizeV;
    const int32_t  nonContiguousFlag  = ub->nonContiguousFlag;
    const int64_t  kCacheStride0      = ub->kCacheStride0;
    const int64_t  kCacheStride1      = ub->kCacheStride1;
    const int64_t  kCacheStride2      = ub->kCacheStride2;
    const int64_t  vCacheStride0      = ub->vCacheStride0;
    const int64_t  vCacheStride1      = ub->vCacheStride1;
    const int64_t  vCacheStride2      = ub->vCacheStride2;
    const int64_t  keyOutStride0      = ub->keyOutStride0;
    const int64_t  keyOutStride1      = ub->keyOutStride1;
    const int64_t  valueOutStride0    = ub->valueOutStride0;
    const int64_t  valueOutStride1    = ub->valueOutStride1;
    const uint64_t numHeadsK         = ub->numHeadsK;
    const uint64_t numHeadsV         = ub->numHeadsV;

    uint64_t tid = static_cast<uint64_t>(threadIdx.x);
    uint64_t tNum = static_cast<uint64_t>(blockDim.x);

    // Non-contiguous flag bits (matching SIMD tiling convention):
    // bit2: keyOut non-contiguous, bit3: valueOut non-contiguous
    // bit4: kCache slot non-contiguous, bit5: kCache head non-contiguous
    // bit6: vCache slot non-contiguous, bit7: vCache head non-contiguous
    // bit8: keyOut head non-contiguous, bit9: valueOut head non-contiguous
    bool isKSlotNC = (nonContiguousFlag >> 4) & 1;
    bool isKHeadNC = (nonContiguousFlag >> 5) & 1;
    bool isVSlotNC = (nonContiguousFlag >> 6) & 1;
    bool isVHeadNC = (nonContiguousFlag >> 7) & 1;
    bool isKOutNC  = (nonContiguousFlag >> 2) & 1;
    bool isVOutNC  = (nonContiguousFlag >> 3) & 1;
    bool isKOutHeadNC = (nonContiguousFlag >> 8) & 1;
    bool isVOutHeadNC = (nonContiguousFlag >> 9) & 1;

    uint64_t globalBlk = 0;
    for (int64_t i = batchStart; i < batchEnd; i++) {
        int64_t seqLen;
        int64_t batchOffset;
        if (IS_CUMSUM) {
            seqLen = static_cast<int64_t>(seqLens[i + 1]) - static_cast<int64_t>(seqLens[i]);
            batchOffset = static_cast<int64_t>(seqLens[i]);
        } else {
            seqLen = static_cast<int64_t>(seqLens[i]);
            batchOffset = coreOffset;
        }

        if (batchOffset >= numTokens) {
            break;
        }

        int64_t accumSeqLen = batchOffset + seqLen;
        if (numTokens > batchOffset && numTokens <= accumSeqLen) {
            seqLen = numTokens - batchOffset;
        }

        int64_t seqOff = 0;
        if (HAS_SEQOFFSET) {
            seqOff = static_cast<int64_t>(seqOffset[i]) / cacheBlockSize;
        }

        int64_t blockCount = (seqLen + cacheBlockSize - 1) / cacheBlockSize;

        for (int64_t blkIdx = 0; blkIdx < blockCount; blkIdx++) {
            // Same boundary/blockId logic as continuous path
            int64_t tableOffset = blkIdx + seqOff;
            bool isFilledWithZero = false;
            int64_t blockId = 0;
            if (tableOffset < 0 || tableOffset >= blockTableWidth) {
                isFilledWithZero = true;
            } else {
                blockId = static_cast<int64_t>(blockTables[i * blockTableWidth + tableOffset]);
                if (blockId < 0 || blockId >= numBlocks) {
                    isFilledWithZero = true;
                    blockId = 0;
                }
            }

            int64_t curBlockSize = cacheBlockSize;
            if (blkIdx == blockCount - 1) {
                curBlockSize = seqLen - (blockCount - 1) * cacheBlockSize;
            }

            // Output base position (absolute token index)
            int64_t baseToken = batchOffset + blkIdx * cacheBlockSize;

            // ===== Key copy (non-contiguous) =====
            int64_t keyOutBaseByte = isKOutNC
                ? (baseToken * keyOutStride0)
                : (baseToken * static_cast<int64_t>(hiddenSizeK));
            int64_t keyCacheBaseByte = blockId * kCacheStride0;

            for (int64_t tok = 0; tok < curBlockSize; tok++) {
                int64_t srcTokOff = tok * kCacheStride1;
                int64_t dstTokOff = isKOutNC
                    ? (tok * keyOutStride0)
                    : (tok * static_cast<int64_t>(hiddenSizeK));

                uint64_t headBytes = (numHeadsK > 0) ? (hiddenSizeK / numHeadsK) : hiddenSizeK;
                bool needHeadLoopK = isKHeadNC || isKOutHeadNC;

                if (!needHeadLoopK) {
                    // All heads contiguous within token: one grid-stride block
                    uint64_t kBytes = hiddenSizeK;
                    for (uint64_t b = tid; b < kBytes; b += tNum) {
                        if (isFilledWithZero) {
                            keyOut[keyOutBaseByte + dstTokOff + static_cast<int64_t>(b)] = 0;
                        } else {
                            keyOut[keyOutBaseByte + dstTokOff + static_cast<int64_t>(b)] =
                                keyCache[keyCacheBaseByte + srcTokOff + static_cast<int64_t>(b)];
                        }
                    }
                } else {
                    // Head-non-contiguous: per-head grid-stride
                    for (uint64_t h = 0; h < numHeadsK; h++) {
                        int64_t srcHeadOff = h * kCacheStride2;
                        int64_t dstHeadOff = h * keyOutStride1;
                        for (uint64_t b = tid; b < headBytes; b += tNum) {
                            if (isFilledWithZero) {
                                keyOut[keyOutBaseByte + dstTokOff + dstHeadOff + static_cast<int64_t>(b)] = 0;
                            } else {
                                keyOut[keyOutBaseByte + dstTokOff + dstHeadOff + static_cast<int64_t>(b)] =
                                    keyCache[keyCacheBaseByte + srcTokOff + srcHeadOff + static_cast<int64_t>(b)];
                            }
                        }
                    }
                }
            }

            // ===== Value copy (non-contiguous) =====
            int64_t valOutBaseByte = isVOutNC
                ? (baseToken * valueOutStride0)
                : (baseToken * static_cast<int64_t>(hiddenSizeV));
            int64_t valCacheBaseByte = blockId * vCacheStride0;

            for (int64_t tok = 0; tok < curBlockSize; tok++) {
                int64_t srcTokOff = tok * vCacheStride1;
                int64_t dstTokOff = isVOutNC
                    ? (tok * valueOutStride0)
                    : (tok * static_cast<int64_t>(hiddenSizeV));

                uint64_t headBytesV = (numHeadsV > 0) ? (hiddenSizeV / numHeadsV) : hiddenSizeV;
                bool needHeadLoopV = isVHeadNC || isVOutHeadNC;

                if (!needHeadLoopV) {
                    uint64_t vBytes = hiddenSizeV;
                    for (uint64_t b = tid; b < vBytes; b += tNum) {
                        if (isFilledWithZero) {
                            valueOut[valOutBaseByte + dstTokOff + static_cast<int64_t>(b)] = 0;
                        } else {
                            valueOut[valOutBaseByte + dstTokOff + static_cast<int64_t>(b)] =
                                valueCache[valCacheBaseByte + srcTokOff + static_cast<int64_t>(b)];
                        }
                    }
                } else {
                    for (uint64_t h = 0; h < numHeadsV; h++) {
                        int64_t srcHeadOff = h * vCacheStride2;
                        int64_t dstHeadOff = h * valueOutStride1;
                        for (uint64_t b = tid; b < headBytesV; b += tNum) {
                            if (isFilledWithZero) {
                                valueOut[valOutBaseByte + dstTokOff + dstHeadOff + static_cast<int64_t>(b)] = 0;
                            } else {
                                valueOut[valOutBaseByte + dstTokOff + dstHeadOff + static_cast<int64_t>(b)] =
                                    valueCache[valCacheBaseByte + srcTokOff + srcHeadOff + static_cast<int64_t>(b)];
                            }
                        }
                    }
                }
            }
        }
        if (!IS_CUMSUM) {
            coreOffset += seqLen;
        }
    }
}

// ===== Class interface (backward compatible with GATHER_PA_KV_CACHE_APT.CPP) =====
template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
class GatherPaKvCacheNd {
public:
    __aicore__ inline GatherPaKvCacheNd(TPipe *pipe, const GatherPaKvCacheTilingDataV35 *__restrict tiling)
        : pipe_(pipe), tl_(tiling) {}
    __aicore__ inline void Init(GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens,
                                GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset, GM_ADDR key_out,
                                GM_ADDR value_out);
    __aicore__ inline void Process();
    __aicore__ inline void InitParams();

private:
    TPipe *pipe_;
    const GatherPaKvCacheTilingDataV35 *tl_;

    // GM pointers (raw, for VF call)
    __gm__ uint8_t *keyCacheGm_;
    __gm__ uint8_t *valueCacheGm_;
    __gm__ T_INDEX  *blockTablesGm_;
    __gm__ T_INDEX  *seqLensGm_;
    __gm__ T_INDEX  *seqOffsetGm_;
    __gm__ uint8_t *keyOutGm_;
    __gm__ uint8_t *valueOutGm_;

    // Tiling parameters (from InitParams)
    int64_t  cacheBlockSize_;
    int64_t  blockTableWidth_;
    int64_t  numBlocks_;
    int64_t  numTokens_;
    uint64_t hiddenSizeK_;
    uint64_t hiddenSizeV_;

    // Core scheduling
    uint32_t batchPerCore_;
    uint32_t needCoreNum_;
    int64_t  batchCount_;

    // Non-contiguous support
    uint32_t nonContiguousFlag_;
    int64_t  kCacheStride0_;
    int64_t  kCacheStride1_;
    int64_t  kCacheStride2_;
    int64_t  vCacheStride0_;
    int64_t  vCacheStride1_;
    int64_t  vCacheStride2_;
    int64_t  keyOutStride0_;
    int64_t  keyOutStride1_;
    int64_t  valueOutStride0_;
    int64_t  valueOutStride1_;
    uint64_t numHeadsK_;
    uint64_t numHeadsV_;
    uint64_t maxUbHiddenSizeK_;
    uint64_t maxUbHiddenSizeV_;
    uint64_t maxUbHiddenSize_;
    uint32_t seqLenAccSize_;
};

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::Init(
    GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens,
    GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset, GM_ADDR key_out, GM_ADDR value_out)
{
    InitParams();

    keyCacheGm_    = static_cast<__gm__ uint8_t *>(key_cache);
    valueCacheGm_  = static_cast<__gm__ uint8_t *>(value_cache);
    blockTablesGm_ = static_cast<__gm__ T_INDEX *>(block_tables);
    seqLensGm_     = static_cast<__gm__ T_INDEX *>(seq_lens);
    seqOffsetGm_   = static_cast<__gm__ T_INDEX *>(seq_offset);
    keyOutGm_      = static_cast<__gm__ uint8_t *>(key_out);
    valueOutGm_    = static_cast<__gm__ uint8_t *>(value_out);
}

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::InitParams()
{
    cacheBlockSize_   = tl_->kvCacheBlockSize;
    batchPerCore_     = tl_->batchPerCore;
    needCoreNum_      = tl_->needCoreNum;
    batchCount_       = tl_->batchCount;
    blockTableWidth_  = tl_->blockTableWidth;
    numBlocks_        = tl_->numBlocks;
    hiddenSizeK_      = tl_->hiddenSizeK;
    hiddenSizeV_      = tl_->hiddenSizeV;
    numTokens_        = tl_->numTokens;
    maxUbHiddenSizeK_ = tl_->maxUbHiddenSizeK;
    maxUbHiddenSizeV_ = tl_->maxUbHiddenSizeV;
    maxUbHiddenSize_  = tl_->maxUbHiddenSize;
    seqLenAccSize_    = tl_->seqLenAccumSize;

    nonContiguousFlag_  = tl_->nonContiguousFlag;
    kCacheStride0_      = tl_->kCacheStride0;
    kCacheStride1_      = tl_->kCacheStride1;
    kCacheStride2_      = tl_->kCacheStride2;
    vCacheStride0_      = tl_->vCacheStride0;
    vCacheStride1_      = tl_->vCacheStride1;
    vCacheStride2_      = tl_->vCacheStride2;
    keyOutStride0_      = tl_->keyOutStride0;
    keyOutStride1_      = tl_->keyOutStride1;
    valueOutStride0_    = tl_->valueOutStride0;
    valueOutStride1_    = tl_->valueOutStride1;
    numHeadsK_          = tl_->numHeadsK;
    numHeadsV_          = tl_->numHeadsV;
}

// ===== Process dispatcher =====
// Computes core scheduling (batch distribution) and launches the appropriate SIMT VF kernel.
// Uses compile-time template bools (isSeqLensCumsum, hasSeqOffset) to dispatch to the correct
// VF kernel specialization without runtime branching.

// Macro to reduce code duplication for the 4 VF launch branches.
#define GE35_RT_CALL_C(kernel, cumsum, hasOff)                                                      \
    do {                                                                                             \
        LocalMemAllocator<Hardware::UB> ubAlloc;                                                     \
        LocalTensor<GatherPaKvCacheSimtUb> tilingUbLocal = ubAlloc.Alloc<GatherPaKvCacheSimtUb>(1); \
        __ubuf__ GatherPaKvCacheSimtUb *ub =                                                         \
            reinterpret_cast<__ubuf__ GatherPaKvCacheSimtUb *>(tilingUbLocal.GetPhyAddr());          \
        /* Fill UB with all tiling parameters */                                                     \
        ub->batchStart        = batchStart;                                                          \
        ub->batchEnd          = batchEnd;                                                            \
        ub->coreOffset        = static_cast<int64_t>(coreOffset);                                    \
        ub->cacheBlockSize    = cacheBlockSize_;                                                     \
        ub->blockTableWidth   = blockTableWidth_;                                                    \
        ub->numBlocks         = numBlocks_;                                                          \
        ub->numTokens         = numTokens_;                                                          \
        ub->hiddenSizeK       = hiddenSizeK_;                                                        \
        ub->hiddenSizeV       = hiddenSizeV_;                                                        \
        ub->nonContiguousFlag = static_cast<int32_t>(nonContiguousFlag_);                            \
        ub->kCacheStride0     = kCacheStride0_;                                                      \
        ub->kCacheStride1     = kCacheStride1_;                                                      \
        ub->kCacheStride2     = kCacheStride2_;                                                      \
        ub->vCacheStride0     = vCacheStride0_;                                                      \
        ub->vCacheStride1     = vCacheStride1_;                                                      \
        ub->vCacheStride2     = vCacheStride2_;                                                      \
        ub->keyOutStride0     = keyOutStride0_;                                                      \
        ub->keyOutStride1     = keyOutStride1_;                                                      \
        ub->valueOutStride0   = valueOutStride0_;                                                    \
        ub->valueOutStride1   = valueOutStride1_;                                                    \
        ub->numHeadsK         = numHeadsK_;                                                          \
        ub->numHeadsV         = numHeadsV_;                                                          \
        DataSyncBarrier<MemDsbT::UB>();                                                               \
        asc_vf_call<kernel<T_INDEX, cumsum, hasOff>>(dim3(THREAD_NUM),                                \
            keyCacheGm_, valueCacheGm_, blockTablesGm_, seqLensGm_, seqOffsetGm_,                     \
            keyOutGm_, valueOutGm_, ub);                                                              \
    } while (0)

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::Process()
{
    int64_t batchStart = static_cast<int64_t>(GetBlockIdx()) * batchPerCore_;
    int64_t batchEnd   = batchStart + batchPerCore_;
    if (GetBlockIdx() == needCoreNum_ - 1) {
        batchEnd = batchCount_;
    }
    if (batchStart >= batchCount_) {
        return;
    }

    // Compute coreOffset for non-cumsum mode.
    // For cumsum, coreOffset = seqLens[batchStart].
    // For non-cumsum, coreOffset = sum_{0}^{batchStart-1} seqLens[i].
    T_INDEX coreOffset = 0;
    if constexpr (isSeqLensCumsum) {
        coreOffset = seqLensGm_[batchStart];
    } else {
        for (int64_t i = 0; i < batchStart; i++) {
            coreOffset += seqLensGm_[i];
        }
    }

    if (nonContiguousFlag_ != 0) {
        // Non-contiguous path: each core handles its batch range with
        // stride-based per-token addressing in the VF kernel.
        GE35_RT_CALL_C(GatherKvSimtNonContig, isSeqLensCumsum, hasSeqOffset);
    } else {
        // Continuous path: all data regions are contiguous in memory.
        GE35_RT_CALL_C(GatherKvSimtContig, isSeqLensCumsum, hasSeqOffset);
    }
}

#undef GE35_RT_CALL_C

} // namespace GatherPaKvCacheV35
#endif
