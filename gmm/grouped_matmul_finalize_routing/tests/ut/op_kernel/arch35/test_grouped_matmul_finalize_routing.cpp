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
#include "../../../../op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h"

template <int ATRANS, int BTRANS, int SCALETYPE, int ROWINDEXTYPE>
__global__ __aicore__ void grouped_matmul_finalize_routing(
    GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR groupList, GM_ADDR shareInput,
    GM_ADDR logit, GM_ADDR rowIndex, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

namespace {

using GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingTilingData;

class GroupedMatmulFinalizeRoutingArch35Test : public testing::Test {};

TEST_F(GroupedMatmulFinalizeRoutingArch35Test, NoSharedInputDoesNotAccessResidualGm)
{
    constexpr uint32_t blockDim = 24;
    constexpr size_t outputSize = 16 * sizeof(bfloat16_t);
    constexpr size_t dummyGmSize = 4096;
    std::array<uint8_t, outputSize> expectedOutput{};
    expectedOutput.fill(0x5A);

    auto *yGm = static_cast<uint8_t *>(AscendC::GmAlloc(outputSize));
    auto *tilingGm = static_cast<uint8_t *>(AscendC::GmAlloc(sizeof(GMMFinalizeRoutingTilingData)));
    auto *dummyGm = static_cast<uint8_t *>(AscendC::GmAlloc(dummyGmSize));
    ASSERT_NE(yGm, nullptr);
    ASSERT_NE(tilingGm, nullptr);
    ASSERT_NE(dummyGm, nullptr);

    std::memcpy(yGm, expectedOutput.data(), outputSize);
    std::memset(tilingGm, 0, sizeof(GMMFinalizeRoutingTilingData));
    std::memset(dummyGm, 0, dummyGmSize);

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
    // Other inputs use valid dummy GM addresses so a failure can be attributed
    // specifically to an unexpected residual access.
    ICPU_SET_TILING_KEY(0UL);
    ICPU_RUN_KF((grouped_matmul_finalize_routing<0, 0, 0, 0>), blockDim,
                dummyGm, dummyGm, dummyGm, dummyGm, dummyGm, dummyGm,
                nullptr, dummyGm, dummyGm, dummyGm, yGm, dummyGm, tilingGm);

    EXPECT_EQ(std::memcmp(yGm, expectedOutput.data(), outputSize), 0);

    AscendC::GmFree(yGm);
    AscendC::GmFree(tilingGm);
    AscendC::GmFree(dummyGm);
}

} // namespace
