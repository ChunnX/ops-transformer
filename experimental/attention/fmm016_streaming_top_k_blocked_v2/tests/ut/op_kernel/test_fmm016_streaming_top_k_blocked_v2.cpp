/**
 * Copyright (c) 2026 fk2323
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "../../../op_kernel/fmm016_router_tiling_v2.h"

extern "C" __global__ __aicore__ void fmm016_streaming_top_k_blocked_v2(GM_ADDR qIndex, GM_ADDR kIndexTranspose,
                                                                        GM_ADDR indexWeight, GM_ADDR selectedIndex,
                                                                        GM_ADDR selectedValid, GM_ADDR workspace,
                                                                        GM_ADDR tiling);

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>

#include "tikicpulib.h"

TEST(Fmm016StreamingTopKBlockedV2KernelTest, SmokeAndValidSelection)
{
    constexpr uint32_t batch = 1;
    constexpr uint32_t heads = 1;
    constexpr uint32_t leaves = 9;
    constexpr uint32_t groups = 1;
    constexpr uint32_t indexDim = 8;
    constexpr uint32_t paddedLeaves = 1024;
    constexpr uint32_t topK = 4;
    constexpr uint32_t totalRows = batch * heads * leaves;
    const size_t qBytes = static_cast<size_t>(totalRows) * groups * indexDim * sizeof(float);
    const size_t kBytes = static_cast<size_t>(batch) * heads * indexDim * paddedLeaves * sizeof(float);
    const size_t weightBytes = static_cast<size_t>(totalRows) * groups * sizeof(float);
    const size_t outElements = static_cast<size_t>(totalRows) * topK;
    const size_t tilingBytes = sizeof(Fmm016StreamingTopKBlockedTilingData);

    auto *q = reinterpret_cast<float *>(AscendC::GmAlloc(qBytes));
    auto *k = reinterpret_cast<float *>(AscendC::GmAlloc(kBytes));
    auto *weight = reinterpret_cast<float *>(AscendC::GmAlloc(weightBytes));
    auto *selectedIndex = reinterpret_cast<int32_t *>(AscendC::GmAlloc(outElements * sizeof(int32_t)));
    auto *selectedValid = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(outElements));
    auto *workspace = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(1U << 20));
    auto *tiling = reinterpret_cast<Fmm016StreamingTopKBlockedTilingData *>(AscendC::GmAlloc(tilingBytes));
    ASSERT_NE(q, nullptr);
    ASSERT_NE(k, nullptr);
    ASSERT_NE(weight, nullptr);
    ASSERT_NE(selectedIndex, nullptr);
    ASSERT_NE(selectedValid, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    std::fill(q, q + qBytes / sizeof(float), 0.0F);
    std::fill(k, k + kBytes / sizeof(float), 0.0F);
    std::fill(weight, weight + weightBytes / sizeof(float), 1.0F);
    std::fill(selectedIndex, selectedIndex + outElements, -1);
    std::fill(selectedValid, selectedValid + outElements, 0);
    std::memset(workspace, 0, 1U << 20);
    tiling->bhCount = batch * heads;
    tiling->leaves = leaves;
    tiling->paddedLeaves = paddedLeaves;
    tiling->indexHeads = groups;
    tiling->indexDim = indexDim;
    tiling->topK = topK;
    tiling->totalRows = totalRows;
    tiling->invSqrtIndexDim = 1.0F / std::sqrt(static_cast<float>(indexDim));

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(fmm016_streaming_top_k_blocked_v2, 1, reinterpret_cast<GM_ADDR>(q), reinterpret_cast<GM_ADDR>(k),
                reinterpret_cast<GM_ADDR>(weight), reinterpret_cast<GM_ADDR>(selectedIndex),
                reinterpret_cast<GM_ADDR>(selectedValid), reinterpret_cast<GM_ADDR>(workspace),
                reinterpret_cast<GM_ADDR>(tiling));

    for (uint32_t row = 0; row < totalRows; ++row) {
        std::set<int32_t> unique;
        for (uint32_t slot = 0; slot < topK; ++slot) {
            const size_t offset = static_cast<size_t>(row) * topK + slot;
            ASSERT_EQ(selectedValid[offset], 1) << "row=" << row << " slot=" << slot;
            ASSERT_GE(selectedIndex[offset], 0);
            ASSERT_LT(selectedIndex[offset], static_cast<int32_t>(leaves));
            ASSERT_GT(std::abs(selectedIndex[offset] - static_cast<int32_t>(row)), 2);
            ASSERT_TRUE(unique.insert(selectedIndex[offset]).second);
        }
    }

    AscendC::GmFree(q);
    AscendC::GmFree(k);
    AscendC::GmFree(weight);
    AscendC::GmFree(selectedIndex);
    AscendC::GmFree(selectedValid);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
