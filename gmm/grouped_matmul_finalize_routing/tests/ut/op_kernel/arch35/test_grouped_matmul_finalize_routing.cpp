/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../op_kernel/grouped_matmul_finalize_routing_apt.cpp"

namespace {

using GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingTilingData;

class GroupedMatmulFinalizeRoutingArch35Test : public testing::Test {};

TEST_F(GroupedMatmulFinalizeRoutingArch35Test, NoSharedInputDoesNotAccessResidualGm)
{
    constexpr uint32_t blockDim = 24;
    constexpr size_t outputSize = 16 * sizeof(bfloat16_t);
    std::array<uint8_t, outputSize> expectedOutput{};
    expectedOutput.fill(0x5A);

    auto *yGm = static_cast<uint8_t *>(AscendC::GmAlloc(outputSize));
    auto *tilingGm = static_cast<uint8_t *>(AscendC::GmAlloc(sizeof(GMMFinalizeRoutingTilingData)));
    ASSERT_NE(yGm, nullptr);
    ASSERT_NE(tilingGm, nullptr);

    std::memcpy(yGm, expectedOutput.data(), outputSize);
    std::memset(tilingGm, 0, sizeof(GMMFinalizeRoutingTilingData));

    auto *tiling = reinterpret_cast<GMMFinalizeRoutingTilingData *>(tilingGm);
    tiling->gmmFinalizeRoutingDataParams.groupNum = 0;
    tiling->gmmFinalizeRoutingDataParams.batch = 1;
    tiling->gmmFinalizeRoutingDataParams.sharedInputOffset = 0;
    tiling->gmmFinalizeRoutingDataParams.sharedInputLen = 0;
    tiling->gmmFinalizeRoutingDataParams.residualScale = 1.0F;
    tiling->matmulTiling.M = 1;
    tiling->matmulTiling.N = 16;
    tiling->matmulTiling.Ka = 16;
    tiling->matmulTiling.Kb = 16;
    tiling->matmulTiling.baseM = 16;
    tiling->matmulTiling.baseN = 16;
    tiling->matmulTiling.baseK = 16;

    // The null share_input pointer is intentional. With sharedInputLen == 0,
    // the prologue must return before CopyInShareInput dereferences residualGm_.
    ICPU_SET_TILING_KEY(0UL);
    ICPU_RUN_KF((grouped_matmul_finalize_routing<0, 0, 0, 0>), blockDim,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, yGm, nullptr, tilingGm);

    EXPECT_EQ(std::memcmp(yGm, expectedOutput.data(), outputSize), 0);

    AscendC::GmFree(yGm);
    AscendC::GmFree(tilingGm);
}

} // namespace
