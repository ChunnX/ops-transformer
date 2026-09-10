/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_COMMON_HPP
#define KERNEL_COMMON_HPP

#include <cstdint>

namespace GenericBlockSparseAttn {

// Kernel-side mirror of GenericBlockSparseAttentionTilingData (host tiling).
// Field order must match BEGIN_TILING_DATA_DEF in generic_block_sparse_attention_tiling.h.
struct GenericBlockSparseAttentionTilingData {
    uint32_t batch;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t embeddingSize;
    uint32_t blockShapeX; // Q sparse block size (default 1)
    uint32_t blockShapeY; // KV sparse block size (default 128)
    uint32_t blockSize;   // PA page size in tokens; zero for TND (no pages)
    uint32_t topK;        // sparseBlockIdx dim-3 (KVb): selected KV blocks per Q block
    uint32_t qBlockNum;   // sparseBlockIdx dim-2 (Qb): total Q blocks
    uint32_t maxBlocksPerBatch;
    float scaleValue;
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint64_t updateSize;
    uint64_t workSpaceSize;
    uint64_t tilingKey;
    uint32_t groupSize;
    // BaseTileInfo
    uint32_t qBaseTile;
    uint32_t kvBaseTile; // Compute tile width, independent of sparse block and PA page sizes.
    // MmPhaseL1TileInfo
    uint32_t mm1L1TileM;
    uint32_t mm1L1TileN;
    uint32_t mm1L1TileKLeft;
    uint32_t mm1L1TileKRight;
    uint32_t mm2L1TileM;
    uint32_t mm2L1TileN;
    uint32_t mm2L1TileKLeft;
    uint32_t mm2L1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;
    uint32_t vL1BufNum;
    uint32_t pL1BufNum;
    // Tensor dim0 stride in elements: PA physical page stride or TND token stride Nkv*D.
    uint64_t kStride0;
    uint64_t vStride0;
    uint32_t fdStaticEnabled;
    uint32_t fdLseSubStride;
    uint32_t fdPartialCapacity;
    uint64_t fdPartialLseOffset;
    uint64_t fdPartialOOffset;
};

} // namespace GenericBlockSparseAttn

#endif // KERNEL_COMMON_HPP
