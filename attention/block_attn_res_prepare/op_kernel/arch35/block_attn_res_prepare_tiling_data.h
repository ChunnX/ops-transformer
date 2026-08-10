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

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr uint64_t BLOCK_ATTN_RES_PREPARE_VECTOR_KEY = 1000UL;
constexpr uint64_t BLOCK_ATTN_RES_PREPARE_MIX_CUBE_KEY = 2000UL;

enum class BlockAttnResPrepareVCacheMode : uint32_t {
    RELOAD = 0,
    CACHE_IF_RUNTIME_FITS = 1,
};

BEGIN_TILING_DATA_DEF(BlockAttnResPrepareTilingData)
// Keep the bounded dimensions/work count compact; D remains uint64_t for the
// aligned FP32 dimension range.  totalD is placed after the uint32 fields to
// avoid padding between the first five values.
TILING_DATA_FIELD_DEF(uint32_t, totalT);
TILING_DATA_FIELD_DEF(uint32_t, totalN);
TILING_DATA_FIELD_DEF(uint32_t, totalS);
TILING_DATA_FIELD_DEF(uint32_t, totalWorkUnits);
TILING_DATA_FIELD_DEF(uint64_t, totalD);

TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, blockFactor);
TILING_DATA_FIELD_DEF(uint32_t, tailBlockFactor);

TILING_DATA_FIELD_DEF(uint32_t, baseD);

// Vector cache policy is selected at runtime from validN and vCacheRows;
// no static cache-mode field is serialized for the Vector template.
// N is limited to one FP32 vector register in the Vector template, and all queue
// depths are 1 or 2, so these values fit in uint8_t.
TILING_DATA_FIELD_DEF(uint8_t, vCacheRows);
TILING_DATA_FIELD_DEF(uint8_t, qBufferNum);
TILING_DATA_FIELD_DEF(uint8_t, vBufferNum);
TILING_DATA_FIELD_DEF(uint8_t, oBufferNum);

// Vector kernel derives dAlign/baseDAlign (32B UB alignment), dTileNum and the
// Q/V/O single-buffer element count from totalD/baseD.  VF register width is an
// implementation detail and must not inflate the UB pitch.
TILING_DATA_FIELD_DEF(uint32_t, statUbElems);
TILING_DATA_FIELD_DEF(uint32_t, workUbElems);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BlockAttnResPrepare_1000, BlockAttnResPrepareTilingData)

// Cube has an independent data layout because its AIC/AIV work mapping and its L1/UB
// capacity plan are different from the Vector template.  In particular, workspace and
// E/dot rows use nAlign rather than the physical N pitch.
BEGIN_TILING_DATA_DEF(BlockAttnResPrepareCubeTilingData)
// Keep the bounded dimensions/work count compact; D remains uint64_t for the
// aligned FP32 dimension range.
TILING_DATA_FIELD_DEF(uint32_t, totalT);
TILING_DATA_FIELD_DEF(uint32_t, totalN);
TILING_DATA_FIELD_DEF(uint32_t, totalS);
TILING_DATA_FIELD_DEF(uint32_t, totalWorkUnits);
TILING_DATA_FIELD_DEF(uint64_t, totalD);

TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, aicCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, aivCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, blockFactor);
TILING_DATA_FIELD_DEF(uint32_t, tailBlockFactor);

TILING_DATA_FIELD_DEF(uint32_t, baseT);
TILING_DATA_FIELD_DEF(uint32_t, baseS);
TILING_DATA_FIELD_DEF(uint32_t, baseD);
TILING_DATA_FIELD_DEF(uint32_t, baseDAlign);
TILING_DATA_FIELD_DEF(uint32_t, sTileNum);
TILING_DATA_FIELD_DEF(uint32_t, dTileNum);
TILING_DATA_FIELD_DEF(uint32_t, sAlign);
TILING_DATA_FIELD_DEF(uint32_t, nAlign);
TILING_DATA_FIELD_DEF(uint32_t, dAlign);

TILING_DATA_FIELD_DEF(uint8_t, qL1BufferNum);
TILING_DATA_FIELD_DEF(uint8_t, vL1BufferNum);
TILING_DATA_FIELD_DEF(uint8_t, vUbBufferNum);
TILING_DATA_FIELD_DEF(uint32_t, l1CacheMode);
TILING_DATA_FIELD_DEF(uint8_t, vL1CacheRows);
TILING_DATA_FIELD_DEF(uint64_t, qL1Elems);
TILING_DATA_FIELD_DEF(uint64_t, vL1Elems);
TILING_DATA_FIELD_DEF(uint64_t, eL1Elems);
TILING_DATA_FIELD_DEF(uint64_t, vUbElems);
TILING_DATA_FIELD_DEF(uint64_t, dotUbElems);
TILING_DATA_FIELD_DEF(uint64_t, reduceUbElems);
TILING_DATA_FIELD_DEF(uint64_t, softmaxUbElems);
TILING_DATA_FIELD_DEF(uint64_t, workspaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, workspacePerCoreElems);
TILING_DATA_FIELD_DEF(float, epsilon);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1Tiling);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2Tiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BlockAttnResPrepare_2000, BlockAttnResPrepareCubeTilingData)

} // namespace optiling

#endif // BLOCK_ATTN_RES_PREPARE_TILING_DATA_H
