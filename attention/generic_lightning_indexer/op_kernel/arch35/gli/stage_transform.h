/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Stage2：相关性变换。逐元素，不改变 [G, S2]。
// 节点按 EVG 的 Load / Compute / Store 声明，调度仍由 Indexer kernel 驱动。

#ifndef GLI_STAGE_TRANSFORM_H
#define GLI_STAGE_TRANSFORM_H

#include <cstdint>

namespace gli {

struct Relu {
    static constexpr uint32_t id = 0;
    static constexpr bool changesShape = false;
};

struct Identity {
    static constexpr uint32_t id = 1;
    static constexpr bool changesShape = false;
};

} // namespace gli

#endif
