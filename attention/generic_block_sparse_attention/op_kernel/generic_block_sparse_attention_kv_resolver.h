/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_KV_RESOLVER_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_KV_RESOLVER_H

#include "kernel_operator.h"

namespace GbsaKernel {

// Sparse selection, storage pages and compute tiles use separate coordinates.
struct KvTokenRange {
    uint32_t start;
    uint32_t count;
};

struct KvTile {
    int32_t pageId;       // Physical PA page; unused for TND.
    uint32_t tokenOffset; // Within the PA page, or relative to the TND batch.
    uint32_t tokenCount;  // Clipped by sparse range, compute tile and PA page end.
};

template <bool PAGED_CACHE_FLAG>
class KvBlockResolver {
public:
    __aicore__ inline void Init(GM_ADDR cuKv, GM_ADDR usedKv, GM_ADDR blockTable)
    {
        hasUsedKv_ = usedKv != nullptr;
        if (hasUsedKv_) {
            usedKv_.SetGlobalBuffer((__gm__ int32_t *)usedKv);
        }
        if constexpr (PAGED_CACHE_FLAG) {
            blockTable_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        } else {
            cuKv_.SetGlobalBuffer((__gm__ int64_t *)cuKv);
        }
    }

    __aicore__ inline bool ReadSequence(uint32_t batch, int64_t &storageStart, uint32_t &actualLen)
    {
        storageStart = 0;
        actualLen = 0;
        int64_t length = 0;
        if constexpr (PAGED_CACHE_FLAG) {
            length = usedKv_.GetValue(batch);
        } else {
            storageStart = cuKv_.GetValue(batch);
            const int64_t storageEnd = cuKv_.GetValue(static_cast<int64_t>(batch) + 1);
            // Check signed values before subtracting or converting to unsigned.
            if (storageStart < 0 || storageEnd < storageStart || cuKv_.GetValue(0) != 0) {
                return false;
            }
            const int64_t storageLen = storageEnd - storageStart;
            length = hasUsedKv_ ? static_cast<int64_t>(usedKv_.GetValue(batch)) : storageLen;
            if (length > storageLen) {
                return false;
            }
        }
        if (length < 0 || length > 0xFFFFFFFFLL) {
            return false;
        }
        actualLen = static_cast<uint32_t>(length);
        return true;
    }

    __aicore__ static inline bool ResolveSparseRange(int32_t sparseBlockId, uint32_t sparseBlockSize,
                                                     uint32_t causalEnd, KvTokenRange &range)
    {
        if (sparseBlockId < 0 || sparseBlockSize == 0U) {
            return false;
        }
        const uint64_t start = static_cast<uint64_t>(sparseBlockId) * sparseBlockSize;
        if (start >= causalEnd) {
            return false;
        }
        range.start = static_cast<uint32_t>(start);
        const uint32_t remaining = causalEnd - range.start;
        range.count = remaining < sparseBlockSize ? remaining : sparseBlockSize;
        return true;
    }

    // Resolve the first contiguous compute tile in a logical token range.
    // To split a range, advance start and subtract count by tile.tokenCount,
    // including when a PA page is missing (false with a nonzero tokenCount).
    __aicore__ inline bool ResolveTile(const KvTokenRange &range, uint32_t batch, uint32_t maxPages,
                                       uint32_t pageSize, uint32_t kvTileSize, KvTile &tile)
    {
        tile.pageId = 0;
        tile.tokenOffset = range.start;
        tile.tokenCount = range.count < kvTileSize ? range.count : kvTileSize;
        if (tile.tokenCount == 0U) {
            return false;
        }
        if constexpr (PAGED_CACHE_FLAG) {
            if (pageSize == 0U) {
                tile.tokenCount = 0;
                return false;
            }
            const uint32_t logicalPage = range.start / pageSize;
            tile.tokenOffset = range.start % pageSize;
            const uint32_t pageRemaining = pageSize - tile.tokenOffset;
            tile.tokenCount = tile.tokenCount < pageRemaining ? tile.tokenCount : pageRemaining;
            if (logicalPage >= maxPages) {
                return false;
            }
            tile.pageId = blockTable_.GetValue(static_cast<int64_t>(batch) * maxPages + logicalPage);
            return tile.pageId >= 0;
        }
        return true;
    }

    // Current host constraints guarantee one tile per selected sparse block.
    // Keep that restriction explicit; future multi-tile scheduling must consume
    // the entire range with ResolveTile instead of silently truncating it here.
    __aicore__ inline bool ResolveSingleTileBlock(int32_t sparseBlockId, uint32_t sparseBlockSize,
                                                  uint32_t causalEnd, uint32_t batch, uint32_t maxPages,
                                                  uint32_t pageSize, uint32_t kvTileSize, KvTile &tile)
    {
        KvTokenRange range;
        return ResolveSparseRange(sparseBlockId, sparseBlockSize, causalEnd, range) &&
               ResolveTile(range, batch, maxPages, pageSize, kvTileSize, tile) && tile.tokenCount == range.count;
    }

    // Like FIA, compute the batch/head base once outside the QK/PV tile loop.
    __aicore__ static inline int64_t BatchHeadOffset(int64_t storageStart, uint64_t rowStride,
                                                    uint32_t head, uint32_t embed)
    {
        int64_t offset = static_cast<int64_t>(head) * embed;
        if constexpr (!PAGED_CACHE_FLAG) {
            offset += storageStart * static_cast<int64_t>(rowStride);
        }
        return offset;
    }

    __aicore__ static inline int64_t Offset(const KvTile &tile, int64_t batchHeadOffset,
                                           uint64_t pageStride, uint64_t rowStride)
    {
        int64_t offset = batchHeadOffset + static_cast<int64_t>(tile.tokenOffset) * rowStride;
        if constexpr (PAGED_CACHE_FLAG) {
            offset += static_cast<int64_t>(tile.pageId) * static_cast<int64_t>(pageStride);
        }
        return offset;
    }

private:
    AscendC::GlobalTensor<int64_t> cuKv_;
    AscendC::GlobalTensor<int32_t> usedKv_;
    AscendC::GlobalTensor<int32_t> blockTable_;
    bool hasUsedKv_ = false;
};

} // namespace GbsaKernel
#endif // GENERIC_BLOCK_SPARSE_ATTENTION_KV_RESOLVER_H
