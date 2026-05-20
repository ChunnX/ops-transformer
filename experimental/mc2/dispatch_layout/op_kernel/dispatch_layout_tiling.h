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
 * \file dispatch_layout_tiling.h
 * \brief dispatch layout tiling data structure
 */

#ifndef DISPATCH_LAYOUT_TILING_H
#define DISPATCH_LAYOUT_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct DispatchLayoutInfo {
    uint32_t numTokens;
    uint32_t numRanks;
    uint32_t numExperts;
    uint32_t numTopk;
    uint32_t localRankSize;
    uint64_t totalUbSize;
};

struct DispatchLayoutTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    DispatchLayoutInfo dispatchLayoutInfo;
};

#endif // DISPATCH_LAYOUT_TILING_H