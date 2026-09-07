/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include "../../../op_kernel/arch22/grouped_mat_mul_allto_allv_mte_tiling.h"

namespace {

namespace Mte = MC2KernelTemplate::Gmma2avMteTiling;

TEST(GroupedMatMulAlltoAllvMteKernelTest, SupportsMaximumGlobalExpertTable)
{
    constexpr uint32_t maxRankSize = 128U;
    constexpr uint32_t maxExpertPerRank = 8U;
    MC2KernelTemplate::GroupedMatMulAlltoAllvMteTaskTilingInfo task = {};

    EXPECT_EQ(maxRankSize * maxExpertPerRank,
              MC2KernelTemplate::GMMA2AV_MTE_MAX_COUNT_NUM);
    EXPECT_EQ(sizeof(task.sendCnt) / sizeof(task.sendCnt[0]),
              MC2KernelTemplate::GMMA2AV_MTE_MAX_COUNT_NUM);
    EXPECT_EQ(sizeof(task.recvCnt) / sizeof(task.recvCnt[0]),
              MC2KernelTemplate::GMMA2AV_MTE_MAX_COUNT_NUM);
}

TEST(GroupedMatMulAlltoAllvMteKernelTest, SerializesCompleteMteTilingAbi)
{
    constexpr size_t countTableBytes =
        MC2KernelTemplate::GMMA2AV_MTE_MAX_COUNT_NUM * sizeof(int32_t);

    EXPECT_GT(sizeof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData),
              2U * countTableBytes);
    EXPECT_LT(offsetof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData,
                       taskTilingInfo),
              offsetof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData,
                       commBufferSize));
    EXPECT_LT(offsetof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData,
                       commBufferSize),
              offsetof(MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData,
                       cocTiling));
}

TEST(GroupedMatMulAlltoAllvMteKernelTest, KeepsFixedCatlassTileContract)
{
    EXPECT_EQ(Mte::L1_TILE_M, 128);
    EXPECT_EQ(Mte::L1_TILE_K, 256);
    EXPECT_EQ(Mte::L1_TILE_N, 256);
    EXPECT_EQ(Mte::L0_TILE_M, 128);
    EXPECT_EQ(Mte::L0_TILE_K, 64);
    EXPECT_EQ(Mte::L0_TILE_N, 256);
    EXPECT_EQ(Mte::WORKSPACE_ALIGNMENT, 512UL);
    EXPECT_EQ(Mte::UB_MOVE_ELEMENTS, 4U * 1024U);
}

TEST(GroupedMatMulAlltoAllvMteKernelTest, SeparatesCountAndSynchronizationRegions)
{
    constexpr uint64_t countTableBytes =
        MC2KernelTemplate::GMMA2AV_MTE_MAX_COUNT_NUM * sizeof(int32_t);
    constexpr uint64_t regionGap =
        Mte::SEND_COUNT_STAGING_FROM_TAIL - Mte::SYNC_REGION_FROM_TAIL;
    constexpr uint64_t syncStorageBytes =
        Mte::SYNC_STORAGE_SLOT_COUNT * Mte::SYNC_SLOT_BYTES;

    EXPECT_EQ(Mte::MIN_CCL_BUFFER_BYTES, 200UL * 1024UL * 1024UL);
    EXPECT_EQ(Mte::SEND_COUNT_STAGING_FROM_TAIL, 21UL * 1024UL * 1024UL);
    EXPECT_EQ(Mte::SYNC_REGION_FROM_TAIL, 20UL * 1024UL * 1024UL);
    EXPECT_GT(Mte::SEND_COUNT_STAGING_FROM_TAIL, Mte::SYNC_REGION_FROM_TAIL);
    EXPECT_GE(regionGap, countTableBytes);
    EXPECT_LE(syncStorageBytes, Mte::SYNC_REGION_FROM_TAIL);
    EXPECT_LT(Mte::SEND_COUNT_STAGING_FROM_TAIL, Mte::MIN_CCL_BUFFER_BYTES);
}

} // namespace
