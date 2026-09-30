/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Qwen Compressed Lightweight Indexer 剖面。
// 换模型时新增 kernel_*.h，只改这三行 using，不复制调度骨架。

#ifndef GLI_KERNEL_QWEN_CLI_H
#define GLI_KERNEL_QWEN_CLI_H

#include "stage_reduce.h"
#include "stage_select.h"
#include "stage_transform.h"

namespace gli {

template <typename Transform, typename Reduce, typename Select>
struct IndexerKernel {
    using transform = Transform;
    using reduce = Reduce;
    using select = Select;
};

using QwenCliKernel = IndexerKernel<Relu, Sum, TopKExpandPartial>;

struct QwenCliProfile {
    using Kernel = QwenCliKernel;
    static constexpr int32_t sparseMode = 4;
    static constexpr int32_t cmpRatio = 4;
    static constexpr bool hasWeightedSum = false;
    static constexpr bool expandIndices = true;
    static constexpr bool appendPartialBlock = true;
};

static_assert(QwenCliProfile::Kernel::transform::id == 0, "Qwen transform is ReLU");
static_assert(!QwenCliProfile::Kernel::reduce::usesWeights, "Qwen reduce is unweighted sum");
static_assert(QwenCliProfile::Kernel::select::appendPartialBlock, "Qwen appends the partial micro-block");

} // namespace gli

#endif
