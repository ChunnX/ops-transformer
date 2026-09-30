/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Stage3：沿 query head 规约。

#ifndef GLI_STAGE_REDUCE_H
#define GLI_STAGE_REDUCE_H

#include <cstdint>

namespace gli {

struct Sum {
    static constexpr uint32_t id = 1;
    static constexpr bool usesWeights = false;
    static constexpr bool keepHeads = false;
};

struct WeightedSum {
    static constexpr uint32_t id = 0;
    static constexpr bool usesWeights = true;
    static constexpr bool keepHeads = false;
};

struct KeepHeads {
    static constexpr uint32_t id = 2;
    static constexpr bool usesWeights = false;
    static constexpr bool keepHeads = true;
};

} // namespace gli

#endif
