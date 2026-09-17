/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_SANDWICH_NORM_POST_PREONLY_TILING_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MhcSandwichNormPostPreonlyTilingData)
// Shape dimensions (flatten view: totalTokens = B*S)
TILING_DATA_FIELD_DEF(uint32_t, totalTokens); // M = B*S
TILING_DATA_FIELD_DEF(uint32_t, headCount);   // N (=4)
TILING_DATA_FIELD_DEF(uint32_t, embedDim);    // D (=2560)

// Core mapping (split on totalTokens)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCoreSp);
TILING_DATA_FIELD_DEF(uint32_t, blockPivot);

// D-axis tiling
TILING_DATA_FIELD_DEF(uint32_t, dimTile);     // D tile size (elements)
TILING_DATA_FIELD_DEF(uint32_t, dimLoop);     // number of D tiles
TILING_DATA_FIELD_DEF(uint32_t, lastDimTile); // last tile len (0=exact)

// mhc_pre parameters (pre-branch only: N rows instead of N^2+2N)
TILING_DATA_FIELD_DEF(uint32_t, phiRows); // N = 4 (pre-branch only)
TILING_DATA_FIELD_DEF(uint32_t, phiCols); // N*D

// Epsilon values
TILING_DATA_FIELD_DEF(float, normEps);
TILING_DATA_FIELD_DEF(float, mhcPreNormEps);
TILING_DATA_FIELD_DEF(float, hcEps);

// Pre-computed float reciprocals (avoid int->float cast in kernel)
TILING_DATA_FIELD_DEF(float, invD); // 1.0f / D
TILING_DATA_FIELD_DEF(float, invK); // 1.0f / (N*D)

// Whether to apply RMSNorm_mid between mhc_post and mhc_pre
TILING_DATA_FIELD_DEF(uint32_t, useRmsMid); // 0 = disabled, 1 = enabled
// [R6] Dual-core N-axis split
TILING_DATA_FIELD_DEF(uint32_t, coresPerToken);    // 1 or 2
TILING_DATA_FIELD_DEF(uint32_t, myHeadCount);      // N / coresPerToken (heads per core)
TILING_DATA_FIELD_DEF(uint32_t, maxTokensPerPair); // max tokens per pair (for SyncAll alignment)

// Sync region offset: start of host-reserved cross-core sync bytes, expressed
// in float elements from workspace base. Host computes this as
// `coreNumAiv * wsPerUnit_MAX`, so kernel's sync writes land in exactly the
// region host tiling reserved (not in the data region middle).
TILING_DATA_FIELD_DEF(uint32_t, syncGmOffsetFloats);

// Whether return h_in_f32
TILING_DATA_FIELD_DEF(int8_t, returnHInF32); // TILING_DATA_FIELD_DEF接口不支持bool
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcSandwichNormPostPreonly, MhcSandwichNormPostPreonlyTilingData)

struct MhcSandwichNormPostPreonlyCompileInfo {};

} // namespace optiling

#endif
