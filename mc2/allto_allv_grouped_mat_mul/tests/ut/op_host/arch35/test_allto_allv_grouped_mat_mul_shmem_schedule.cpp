/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <limits>

#include "../../../../op_kernel/allto_allv_grouped_mat_mul_shmem_schedule.h"

namespace {
using namespace AlltoAllvGmmShmem;

TEST(AlltoAllvGroupedMatMulShmemSchedule, CeilDivDoesNotOverflowAtUint64Limit)
{
    EXPECT_EQ(CeilDivU64(std::numeric_limits<uint64_t>::max(), 2), 1ULL << 63);
    EXPECT_EQ(CeilDivU64(17, 0), 0U);
}

TEST(AlltoAllvGroupedMatMulShmemSchedule, SplitsExpertAcrossRounds)
{
    const auto first = GetRoundExpertSlice(100, 80, 0, 128);
    EXPECT_EQ(first.stageRow, 100U);
    EXPECT_EQ(first.expertRow, 0U);
    EXPECT_EQ(first.rows, 28U);

    const auto second = GetRoundExpertSlice(100, 80, 128, 128);
    EXPECT_EQ(second.stageRow, 0U);
    EXPECT_EQ(second.expertRow, 28U);
    EXPECT_EQ(second.rows, 52U);
}

TEST(AlltoAllvGroupedMatMulShmemSchedule, HandlesZeroAndDisjointSlices)
{
    EXPECT_EQ(GetRoundExpertSlice(0, 0, 0, 128).rows, 0U);
    EXPECT_EQ(GetRoundExpertSlice(256, 32, 0, 128).rows, 0U);
    EXPECT_EQ(GetRoundExpertSlice(0, 32, 128, 128).rows, 0U);
}

TEST(AlltoAllvGroupedMatMulShmemSchedule, UsesSourceAlignedDistributedWorkspaceLayout)
{
    // BF16 rows use a 256-element (512-byte) alignment, hence K=257 becomes 512.
    EXPECT_EQ(GetStagePeerOffset(0, 0, 4, 1280, 257, 2), 0U);
    EXPECT_EQ(GetStagePeerOffset(0, 1, 4, 1280, 257, 2), 1280ULL * 512ULL);
    EXPECT_EQ(GetStagePeerOffset(1, 0, 4, 1280, 257, 2), 4ULL * 1280ULL * 512ULL);
}

TEST(AlltoAllvGroupedMatMulShmemSchedule, CountsSignalsAtCommunicationBlockGranularity)
{
    EXPECT_EQ(GetCompletedCommBlocks(0, 64), 0U);
    EXPECT_EQ(GetCompletedCommBlocks(1, 64), 1U);
    EXPECT_EQ(GetCompletedCommBlocks(64, 64), 1U);
    EXPECT_EQ(GetCompletedCommBlocks(65, 64), 2U);
}

TEST(AlltoAllvGroupedMatMulShmemSchedule, PreservesSourceExpertMajorOutputOrder)
{
    // Stored rank-major: rank0=[3,5], rank1=[7,11]. Output order is
    // expert0/rank0, expert0/rank1, expert1/rank0, expert1/rank1.
    const uint32_t recvCounts[] = {3, 5, 7, 11};
    EXPECT_EQ(GetExpertOutputOffset(recvCounts, 0, 0, 2, 2), 0U);
    EXPECT_EQ(GetExpertOutputOffset(recvCounts, 0, 1, 2, 2), 3U);
    EXPECT_EQ(GetExpertOutputOffset(recvCounts, 1, 0, 2, 2), 10U);
    EXPECT_EQ(GetExpertOutputOffset(recvCounts, 1, 1, 2, 2), 15U);
    EXPECT_EQ(GetExpertBeginInPeer(recvCounts, 1, 1, 2), 7U);
}
} // namespace
