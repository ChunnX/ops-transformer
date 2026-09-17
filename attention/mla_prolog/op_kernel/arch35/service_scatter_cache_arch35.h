/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file service_scatter_cache_arch35.h
 * \brief
 */

#ifndef SERVICE_SCATTER_CACHE_ARCH35_H
#define SERVICE_SCATTER_CACHE_ARCH35_H

#include "mla_prolog_comm_arch35.h"

namespace MlaProlog {

__aicore__ inline int64_t CeilDiv64(int64_t x, int64_t y)
{
    if (y == 0) {
        return static_cast<int64_t>(0);
    }
    return (x + y - 1) / y;
};

/**
 * @brief PA场景，将inputLocal中的数据scatter到cacheGm，支持ND和Nz cache
 * @param cacheGm 输出tensor
 *     ND [blockNum, blockSize, col]
 *     Nz [blockNum, ceil(col/col0), blockSize, col0]
 * @param inputLocal 输入tensor，[row, col]，一行对应一个token，只支持单行数据处理，row为1
 * @param scatterCacheParams 所需相关参数，包括
          blockSize KV blocks的大小
          paTokenIndex 待处理token在cache中的全局index，取值[0, blockNum*blockSize)
          row 待处理的行数
          col 待处理的列数，需满足32 bytes对齐
 */

struct ScatterCacheParams {
    int64_t blockSize;
    int64_t paTokenIndex;
    int64_t row;
    int64_t col;
    int64_t stride;
    int64_t cacheStride0;
    int64_t seqLength;
    int64_t tokenIndex;
};

__aicore__ inline int64_t GetCacheOffset(int64_t paTokenIndex, int64_t blockSize, int64_t tokenStride,
                                         int64_t cacheStride0)
{
    if (blockSize <= 0) {
        return paTokenIndex * cacheStride0;
    }
    int64_t sc35BlockIndex = paTokenIndex / blockSize;
    int64_t sc35TokenIndexInBlock = paTokenIndex % blockSize;
    return sc35BlockIndex * cacheStride0 + sc35TokenIndexInBlock * tokenStride;
}

template <typename T, bool IS_NZ>
__aicore__ inline void ScatterCache(const GlobalTensor<T> &cacheGm, const LocalTensor<T> &inputLocal,
                                    const ScatterCacheParams &scatterCacheParams)
{
    if (scatterCacheParams.paTokenIndex < 0) {
        return;
    }
    if constexpr (!IS_NZ) {
        int64_t sc35CacheOffset = GetCacheOffset(scatterCacheParams.paTokenIndex, scatterCacheParams.blockSize,
                                                 scatterCacheParams.stride, scatterCacheParams.cacheStride0);
        DataCopy(cacheGm[sc35CacheOffset], inputLocal, scatterCacheParams.col);
    } else {
        constexpr uint8_t sc35Col0 = ALIGN_BLOCK_SIZE / sizeof(T);
        int64_t sc35CacheOffsetNz = GetCacheOffset(scatterCacheParams.paTokenIndex, scatterCacheParams.blockSize,
                                                   sc35Col0, scatterCacheParams.cacheStride0);
        DataCopyParams sc35CopyParams{static_cast<uint16_t>(scatterCacheParams.col / sc35Col0), 1, 0,
                                      static_cast<uint16_t>(scatterCacheParams.blockSize - 1)};
        DataCopy(cacheGm[sc35CacheOffsetNz], inputLocal, sc35CopyParams);
    }
}

template <typename T, bool IS_NZ>
__aicore__ inline void ScatterCacheUnAligned(const GlobalTensor<T> &cacheGm, const LocalTensor<T> &inputLocal,
                                             const ScatterCacheParams &scatterCacheParams)
{
    if (scatterCacheParams.paTokenIndex < 0) {
        return;
    }
    if constexpr (!IS_NZ) {
        int64_t sc35CacheOffset = GetCacheOffset(scatterCacheParams.paTokenIndex, scatterCacheParams.blockSize,
                                                 scatterCacheParams.stride, scatterCacheParams.cacheStride0);
        // blockCount, blockLen, srcStride, dstStride
        DataCopyParams sc35DataCopyParams{1, static_cast<uint16_t>(scatterCacheParams.col * sizeof(T)), 0, 0};
        DataCopyPad(cacheGm[sc35CacheOffset], inputLocal, sc35DataCopyParams);
    }
}

template <typename T, bool IS_NZ>
__aicore__ inline void ScatterCacheMultiRows(GlobalTensor<T> &cacheGm, const LocalTensor<T> &inputLocal,
                                             const ScatterCacheParams &scatterCacheParams, int64_t rowsInCurBatch,
                                             int64_t cacheOffset, int64_t nextBatchOffset)
{
    if (cacheOffset < 0) {
        return;
    }
    int64_t sc35CopyCnt = scatterCacheParams.col * rowsInCurBatch;

    if constexpr (!IS_NZ) {
        DataCopy(cacheGm[cacheOffset], inputLocal, sc35CopyCnt);
        if (rowsInCurBatch != scatterCacheParams.row) {
            DataCopy(cacheGm[nextBatchOffset], inputLocal[sc35CopyCnt],
                     (scatterCacheParams.row - rowsInCurBatch) * scatterCacheParams.col);
        }
    } else {
        constexpr uint8_t sc35Col0 = ALIGN_BLOCK_SIZE / sizeof(T);
        DataCopyParams sc35CopyParams{static_cast<uint16_t>(scatterCacheParams.col / sc35Col0), 1, 0,
                                      static_cast<uint16_t>(scatterCacheParams.blockSize - 1)};
        DataCopy(cacheGm[cacheOffset], inputLocal, sc35CopyParams);
        if (rowsInCurBatch != scatterCacheParams.row) {
            for (int64_t sc35Row = 0; sc35Row < scatterCacheParams.row - rowsInCurBatch; ++sc35Row) {
                DataCopy(cacheGm[nextBatchOffset + sc35Row * sc35Col0], inputLocal[sc35CopyCnt + sc35Row * sc35Col0],
                         sc35CopyParams);
            }
        }
    }
}

template <typename T, bool IS_NZ>
__aicore__ inline void MaterializeOffsetsWithHeadSize(int64_t pageId, int64_t tokenOffsetInPage, int64_t rowsThisStep,
                                                      bool spill, int64_t nextPageId, int64_t headSize,
                                                      int64_t cacheStride0, CkvkrParams &ckvkrParams)
{
    ckvkrParams.rowsInCurBatch = rowsThisStep;
    if constexpr (IS_NZ) {
        constexpr uint8_t sc35Col0 = ALIGN_BLOCK_SIZE / sizeof(T);
        ckvkrParams.cacheOffset = pageId * cacheStride0 + tokenOffsetInPage * sc35Col0;
    } else {
        ckvkrParams.cacheOffset = pageId * cacheStride0 + tokenOffsetInPage * headSize;
    }
    ckvkrParams.nextBatchOffset = (spill && nextPageId >= 0) ? nextPageId * cacheStride0 : 0;
}

} // namespace MlaProlog

#endif
