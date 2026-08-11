/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_TILING_DATA_H
#define BLOCK_ATTN_RES_PREPARE_TILING_DATA_H

#include <cstdint>

#include "tiling/tiling_api.h"

namespace optiling {

constexpr uint64_t BLOCK_ATTN_RES_PREPARE_VECTOR_KEY = 1000UL;
constexpr uint64_t BLOCK_ATTN_RES_PREPARE_MIX_CUBE_KEY = 2000UL;

enum class BlockAttnResPrepareVCacheMode : uint32_t {
    RELOAD = 0,
    CACHE_IF_RUNTIME_FITS = 1,
};

// Tiling data is deliberately declared as a plain struct.  The host writes
// the scalar fields directly into the raw tiling buffer and the kernel reads
// the same layout, so no generated setter/serialization methods are required
// for the operator-owned fields below.
// Keep the bounded dimensions/work count compact; D remains uint64_t for the
// aligned FP32 dimension range.  totalD is placed after the uint32 fields to
// keep the layout stable across host and device compilation.
struct BlockAttnResPrepareTilingData {
    uint32_t totalT;
    uint32_t totalN;
    uint32_t totalS;
    uint32_t totalWorkUnits;
    uint64_t totalD;

    uint32_t usedCoreNum;
    uint32_t bigCoreNum;
    uint32_t blockFactor;
    uint32_t tailBlockFactor;

    uint32_t baseD;

    // Vector cache policy is selected at runtime from validN and vCacheRows;
    // no static cache-mode field is serialized for the Vector template.
    // N is limited to one FP32 vector register in the Vector template, and all
    // queue depths are 1 or 2, so these values fit in uint8_t.
    uint8_t vCacheRows;
    uint8_t qBufferNum;
    uint8_t vBufferNum;
    uint8_t oBufferNum;

    // Vector kernel derives dAlign/baseDAlign (32B UB alignment), dTileNum and
    // the Q/V/O single-buffer element count from totalD/baseD.
    uint32_t statUbElems;
    uint32_t workUbElems;
    float epsilon;
};

// Cube has an independent data layout because its AIC/AIV work mapping and its L1/UB
// capacity plan are different from the Vector template.  In particular, workspace and
// E/dot rows use nAlign rather than the physical N pitch.
struct BlockAttnResPrepareCubeTilingData {
    // Keep the bounded dimensions/work count compact; D remains uint64_t for
    // the aligned FP32 dimension range.
    uint32_t totalT;
    uint32_t totalN;
    uint32_t totalS;
    uint32_t totalWorkUnits;
    uint64_t totalD;

    uint32_t usedCoreNum;
    uint32_t aicCoreNum;
    uint32_t aivCoreNum;
    uint32_t bigCoreNum;
    uint32_t blockFactor;
    uint32_t tailBlockFactor;

    uint32_t baseT;
    uint32_t baseS;
    uint32_t baseD;
    uint32_t baseDAlign;
    uint32_t sTileNum;
    uint32_t dTileNum;
    uint32_t sAlign;
    uint32_t nAlign;
    uint32_t dAlign;

    uint8_t qL1BufferNum;
    uint8_t vL1BufferNum;
    uint8_t vUbBufferNum;
    uint32_t l1CacheMode;
    uint8_t vL1CacheRows;
    uint64_t qL1Elems;
    uint64_t vL1Elems;
    uint64_t eL1Elems;
    uint64_t vUbElems;
    uint64_t dotUbElems;
    uint64_t reduceUbElems;
    uint64_t softmaxUbElems;
    uint64_t workspaceOffset;
    uint64_t workspacePerCoreElems;
    float epsilon;

    TCubeTiling mm1Tiling{};
    TCubeTiling mm2Tiling{};
};

} // namespace optiling

#endif // BLOCK_ATTN_RES_PREPARE_TILING_DATA_H
