/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_pa_kv_cache_tiling_data.h
 * \brief Tiling data struct for GatherPaKvCache SIMT kernel (ND format)
 */

#ifndef GATHER_PA_KV_CACHE_SIMT_TILING_DATA_H
#define GATHER_PA_KV_CACHE_SIMT_TILING_DATA_H

#include <cstdint>

struct GatherPaKvCacheSimtTilingData {
    int64_t batchCount;
    int64_t batchPerCore;
    int64_t needCoreNum;
    int64_t kvCacheBlockSize;
    int64_t blockTableWidth;
    int64_t numBlocks;
    int64_t numTokens;
    uint64_t hiddenSizeK;
    uint64_t hiddenSizeV;
    uint64_t keyByteSize;
    uint64_t valueByteSize;
    uint64_t indexByteSize;
    int64_t seqLenAccumSize;

    // non-contiguous support
    uint32_t nonContiguousFlag;
    int64_t kCacheStride0;
    int64_t kCacheStride1;
    int64_t kCacheStride2;
    int64_t vCacheStride0;
    int64_t vCacheStride1;
    int64_t vCacheStride2;
    int64_t keyOutStride0;
    int64_t keyOutStride1;
    int64_t valueOutStride0;
    int64_t valueOutStride1;
    uint64_t numHeadsK;
    uint64_t numHeadsV;
};

#endif // GATHER_PA_KV_CACHE_SIMT_TILING_DATA_H
