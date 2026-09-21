/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * \file sort_lib_params.h
 * \brief SortLib kernel-side parameter struct.
 */

#ifndef SORT_LIB_PARAMS_H
#define SORT_LIB_PARAMS_H

#include <cstdint>

namespace SortLib {

struct SortParams {
    uint32_t numTileData = 0;  // max elements per tile (UB-bound, computed by host tiling)
    uint32_t tileCount = 0;    // ceil(totalElements / numTileData)
    uint32_t activeCores = 0;  // min(coreCount, tileCount)
    uint32_t tmpUbSize = 0;    // AscendC::Sort temp UB bytes
    int64_t totalElements = 0; // element count to sort
    uint32_t isSingleCore = 0; // 1 = single-core fast path, 0 = multi-core LSD radix sort
};

} // namespace SortLib

#endif
