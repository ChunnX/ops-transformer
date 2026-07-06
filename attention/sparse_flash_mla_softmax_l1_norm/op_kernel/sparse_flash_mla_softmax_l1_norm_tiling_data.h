/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_mla_softmax_l1_norm_tiling_data.h
 * \brief kernel 侧 TilingData，字段必须与 Host 侧 tiling.h 的 BEGIN_TILING_DATA_DEF 完全一致
 */
#ifndef SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_TILING_DATA_H
#define SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {
    struct SmlaSoftmaxL1NormTilingData {
        int64_t b;
        int64_t n2;
        int64_t g;
        int64_t s1;
        int64_t s2;
        int64_t d;
        int64_t selectedBlockCount;
        int64_t usedCoreNum;
        int64_t layout;
        int32_t maskMode;
        float scaleValue;
        int32_t cmpRatio;
        uint8_t hasUsedSeqQ;
        uint8_t hasUsedSeqK;
        uint8_t hasTopK;
    };
} // namespace optiling

#endif // SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_TILING_DATA_H
