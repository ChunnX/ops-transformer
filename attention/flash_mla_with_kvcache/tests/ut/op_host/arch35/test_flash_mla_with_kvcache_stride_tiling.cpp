/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/stride.h"
#include "graph/types.h"
#include "../../../../op_host/flash_mla_with_kvcache_tiling_utils.h"

namespace FlashMlaWithKvcacheUT {
namespace {
struct StrideCase {
    gert::Shape shape;
    gert::Stride stride;
    int32_t expectedAxis;
};
}

TEST(FlashMlaWithKvcacheStrideTilingTest, ReportsInnermostNoncontiguousAxis)
{
    // Axis must be returned to the parser, otherwise all failures appear as axis 0.
    const StrideCase cases[] = {
        {{4, 128, 1, 576}, {73728, 576, 576, 1}, -1},
        {{4, 128, 1, 576}, {147456, 576, 576, 1}, 0},
        {{4, 128, 1, 576}, {147456, 1152, 576, 1}, 1},
        {{4, 1, 36, 128, 16}, {73728, 73728, 2048, 16, 1}, -1},
        {{4, 1, 36, 128, 16}, {147456, 73728, 2048, 16, 1}, 0},
        {{4, 1, 36, 128, 16}, {73728, 147456, 2048, 16, 1}, 1},
        {{4, 1, 36, 128, 16}, {147456, 147456, 2048, 16, 1}, 1},
        {{4, 1, 36, 128, 16}, {147456, 73728, 4096, 16, 1}, 2},
        {{4, 1, 36, 128, 16}, {147456, 73728, 2048, 32, 1}, 3},
        {{4, 1, 36, 128, 16}, {147456, 73728, 2048, 16, 2}, 4},
    };
    for (const auto &test : cases) {
        int32_t axis = 0;
        const auto status = optiling::flash_mla_with_kvcache::CheckTensorContiguous(
            test.shape.GetDimNum(), test.shape, &test.stride, axis);
        EXPECT_EQ(status, test.expectedAxis == -1 ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED);
        EXPECT_EQ(axis, test.expectedAxis);
    }
}

} // namespace FlashMlaWithKvcacheUT
