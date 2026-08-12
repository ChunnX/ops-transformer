/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATTENTION_WORKER_COMBINE_ARCH35_COMMON_H_
#define ATTENTION_WORKER_COMBINE_ARCH35_COMMON_H_

#include "kernel_operator.h"
#include "attention_worker_combine_tiling_struct.h"

namespace AttentionWorkerCombineRegbase {
using namespace AscendC;
using optiling::AttentionWorkerCombineRegbaseTilingData;

constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int64_t BLOCK_SIZE = 32;

template <typename T, typename U>
__aicore__ inline auto AlignUp(T a, U b)
{
    return (a + b - 1) / b * b;
}

} // namespace AttentionWorkerCombineRegbase

#endif // ATTENTION_WORKER_COMBINE_ARCH35_COMMON_H_
