/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tutel_dispatch_tiling_data.h
 * \brief Shared Ascend 950 tiling-data POD.
 */

#ifndef MOE_TUTEL_DISPATCH_TILING_DATA_H_
#define MOE_TUTEL_DISPATCH_TILING_DATA_H_

#include <cstdint>

struct MoeTutelDispatchTilingData {
    int64_t samples;     // S = x.dim(0)
    int64_t hidden;      // H = x.dim(1)
    int64_t expertE;     // E = caller-provided y.dim(0), independent of K
    int64_t capacity;    // C = attr capacity = y.dim(1)
    int64_t routeNum;    // K * S
    int64_t yElemCnt;    // E * C * H
    int64_t usedCoreNum; // exactly equal to blockDim
    int64_t routeBatch;  // FULL_ROW batch; SPLIT_H uses 1
    int64_t hChunkElems; // FULL_ROW uses H; SPLIT_H uses a UB-bounded chunk
};

static_assert(sizeof(MoeTutelDispatchTilingData) % sizeof(uint64_t) == 0,
              "MoeTutelDispatch tiling data must be 8-byte aligned");

#endif // MOE_TUTEL_DISPATCH_TILING_DATA_H_
