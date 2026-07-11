/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../op_kernel/arch35/moe_token_permute_regbase.h"

extern "C" __global__ __aicore__ void moe_token_permute(
    GM_ADDR tokens, GM_ADDR indices, GM_ADDR permuteTokens, GM_ADDR sortedIndices, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr int64_t TOKEN_NUM = 4;
constexpr int64_t TOP_K = 2;
constexpr int64_t COLS = 3;
constexpr int64_t TOTAL_LENGTH = TOKEN_NUM * TOP_K;
constexpr int64_t ACTIVE_ROWS = 5;
constexpr size_t WORKSPACE_SIZE = 16 * 1024 * 1024 + 4096;

MoeTokenPermuteTilingData MakeTilingData()
{
    MoeTokenPermuteTilingData tilingData{};
    tilingData.coreNum = 1;
    tilingData.n = TOKEN_NUM;
    tilingData.cols = COLS;
    tilingData.colsAlign = 16;
    tilingData.topK = TOP_K;

    auto& sort = tilingData.vbsComputeParamsOp;
    sort.needCoreNum = 1;
    sort.perCoreElements = TOTAL_LENGTH;
    sort.perCoreLoops = 1;
    sort.perCorePerLoopElements = TOTAL_LENGTH;
    sort.perCoreLastLoopElements = TOTAL_LENGTH;
    sort.lastCoreElements = TOTAL_LENGTH;
    sort.lastCoreLoops = 1;
    sort.lastCorePerLoopElements = TOTAL_LENGTH;
    sort.lastCoreLastLoopElements = TOTAL_LENGTH;
    sort.oneLoopMaxElements = 5440;
    tilingData.sortOutComputeParamsOp.oneLoopMaxElements = 1024;

    auto& gather = tilingData.indexCopyComputeParamsOp;
    gather.needCoreNum = 1;
    gather.coreCalcNum = TOTAL_LENGTH;
    gather.coreCalcTail = TOTAL_LENGTH;
    gather.onceUbTokenNums = TOTAL_LENGTH;
    gather.onceIndicesTokenNums = TOTAL_LENGTH;
    gather.oneTokenlastMove = COLS;
    gather.oneTokenOnceMove = COLS;
    gather.oneTokenMoveTimes = 1;
    gather.frontCoreLoop = 1;
    gather.frontCoreLastTokenNums = TOTAL_LENGTH;
    gather.tailCoreLoop = 1;
    gather.tailLastIndicesLastTokenNums = TOTAL_LENGTH;
    gather.numOutTokens = ACTIVE_ROWS;
    return tilingData;
}
} // namespace

TEST(MoeTokenPermuteArch35Kernel, Int64IndicesSliceResult)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    const size_t tokenBytes = TOKEN_NUM * COLS * sizeof(uint16_t);
    const size_t indicesBytes = TOTAL_LENGTH * sizeof(int64_t);
    const size_t outputBytes = ACTIVE_ROWS * COLS * sizeof(uint16_t);
    const size_t sortedIndicesBytes = TOTAL_LENGTH * sizeof(int32_t);

    auto* tokens = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tokenBytes));
    auto* indices = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(indicesBytes));
    auto* permuteTokens = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputBytes));
    auto* sortedIndices = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sortedIndicesBytes));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(WORKSPACE_SIZE));
    auto* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(MoeTokenPermuteTilingData)));

    ASSERT_NE(tokens, nullptr);
    ASSERT_NE(indices, nullptr);
    ASSERT_NE(permuteTokens, nullptr);
    ASSERT_NE(sortedIndices, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    auto* tokenData = reinterpret_cast<uint16_t*>(tokens);
    for (int64_t i = 0; i < TOKEN_NUM * COLS; ++i) {
        tokenData[i] = static_cast<uint16_t>(0x1000 + i);
    }
    const std::array<int64_t, TOTAL_LENGTH> indicesData = {2, 0, 1, 0, 2, 1, 0, 1};
    std::memcpy(indices, indicesData.data(), indicesBytes);
    std::memset(permuteTokens, 0xff, outputBytes);
    std::memset(sortedIndices, 0xff, sortedIndicesBytes);
    std::memset(workspace, 0, WORKSPACE_SIZE);
    const auto tilingData = MakeTilingData();
    std::memcpy(tiling, &tilingData, sizeof(tilingData));

    ICPU_SET_TILING_KEY(5);
    ICPU_RUN_KF(moe_token_permute, 1, tokens, indices, permuteTokens, sortedIndices, workspace, tiling);

    const std::array<int64_t, ACTIVE_ROWS> expectedTokenRows = {0, 1, 3, 1, 2};
    const auto* outputData = reinterpret_cast<const uint16_t*>(permuteTokens);
    for (int64_t outputRow = 0; outputRow < ACTIVE_ROWS; ++outputRow) {
        for (int64_t col = 0; col < COLS; ++col) {
            EXPECT_EQ(outputData[outputRow * COLS + col], tokenData[expectedTokenRows[outputRow] * COLS + col]);
        }
    }

    const std::array<int32_t, TOTAL_LENGTH> expectedSortedIndices = {6, 0, 3, 1, 7, 4, 2, 5};
    const auto* sortedIndicesData = reinterpret_cast<const int32_t*>(sortedIndices);
    for (int64_t i = 0; i < TOTAL_LENGTH; ++i) {
        EXPECT_EQ(sortedIndicesData[i], expectedSortedIndices[i]);
    }

    AscendC::GmFree(tokens);
    AscendC::GmFree(indices);
    AscendC::GmFree(permuteTokens);
    AscendC::GmFree(sortedIndices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST(MoeTokenPermuteArch35Kernel, CompatGatherLoopMapping)
{
    MoeTokenPermuteTilingData source{};
    source.coreNum = 2;
    source.n = 5;
    source.cols = 3;
    source.topK = 2;
    source.indexCopyComputeParamsOp.needCoreNum = 2;
    source.indexCopyComputeParamsOp.coreCalcNum = 5;
    source.indexCopyComputeParamsOp.coreCalcTail = 4;
    source.indexCopyComputeParamsOp.onceIndicesTokenNums = 3;
    source.indexCopyComputeParamsOp.frontCoreLastTokenNums = 2;
    source.indexCopyComputeParamsOp.onceUbTokenNums = 4;
    source.indexCopyComputeParamsOp.tailLastIndicesLastTokenNums = 4;
    source.indexCopyComputeParamsOp.frontCoreLoop = 2;
    source.indexCopyComputeParamsOp.tailCoreLoop = 1;
    source.indexCopyComputeParamsOp.oneTokenOnceMove = 3;
    source.indexCopyComputeParamsOp.oneTokenlastMove = 3;
    source.indexCopyComputeParamsOp.oneTokenMoveTimes = 1;
    source.indexCopyComputeParamsOp.numOutTokens = 9;

    MoeInitRoutingV2TilingData compat{};
    MoeTokenPermute::InitRegBaseCompatTiling(compat, &source);

    EXPECT_EQ(compat.gatherOutComputeParamsOp.perCoreRows, 5);
    EXPECT_EQ(compat.gatherOutComputeParamsOp.lastCoreRows, 4);
    EXPECT_EQ(compat.gatherOutComputeParamsOp.perCoreLoops, 1);
    EXPECT_EQ(compat.gatherOutComputeParamsOp.lastCoreLoops, 2);
    EXPECT_EQ(compat.srcToDstComputeParamsOp.needCoreNum, 2);
    EXPECT_EQ(compat.gatherOutComputeParamsOp.activateRows, 9);
}
