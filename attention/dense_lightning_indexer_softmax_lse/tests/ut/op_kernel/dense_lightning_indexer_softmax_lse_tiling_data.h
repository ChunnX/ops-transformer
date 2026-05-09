/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_TILING_DATA_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_TILING_DATA_H

#include <cstdint>

struct DenseLISoftmaxLseTilingData {
    uint32_t bSize;
    uint32_t n2Size;
    uint32_t gSize;
    uint64_t s1Size;
    uint64_t s2Size;
    uint32_t usedCoreNum;
    uint32_t sparseMode;
    int64_t preTokens;
    int64_t nextTokens;
};

#endif
