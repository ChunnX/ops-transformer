/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "tiling_case_executor.h"

namespace AlltoAllvGroupedMatMulShmemUT {
namespace {
constexpr size_t SHMEM_WORKSPACE_BYTES = 400ULL * 1024ULL * 1024ULL + 4096ULL;

void RunCase(int64_t epWorldSize, int64_t m, int64_t k, int64_t n, int64_t experts,
             const std::vector<int64_t> &sendCounts, const std::vector<int64_t> &recvCounts,
             ge::graphStatus expected)
{
    struct CompileInfo {} compileInfo;
    gert::TilingContextPara para(
        "AlltoAllvGroupedMatMulShmem",
        {
            {{{m, k}, {m, k}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{experts, k, n}, {experts, k, n}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {{{static_cast<int64_t>(std::accumulate(recvCounts.begin(), recvCounts.end(), int64_t{0})), n},
          {static_cast<int64_t>(std::accumulate(recvCounts.begin(), recvCounts.end(), int64_t{0})), n}},
         ge::DT_BF16,
         ge::FORMAT_ND}},
        {
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(recvCounts)},
        },
        &compileInfo, "Ascend950", 20, 196608, 8192);
    ExecuteTestCase(para, expected, expected == ge::GRAPH_SUCCESS ? 1UL : 0UL, "",
                    expected == ge::GRAPH_SUCCESS ? std::vector<size_t>{SHMEM_WORKSPACE_BYTES} : std::vector<size_t>{});
}
} // namespace

TEST(AlltoAllvGroupedMatMulShmemTiling, AcceptsNonPowerOfTwoEpFromSource)
{
    std::vector<int64_t> counts(3 * 2, 16);
    RunCase(3, 96, 256, 128, 2, counts, counts, ge::GRAPH_SUCCESS);
}

TEST(AlltoAllvGroupedMatMulShmemTiling, RejectsInvalidCountContracts)
{
    std::vector<int64_t> valid(4, 16);
    std::vector<int64_t> shortCounts(3, 16);
    std::vector<int64_t> negative = valid;
    negative[0] = -1;
    RunCase(2, 64, 256, 128, 2, shortCounts, valid, ge::GRAPH_FAILED);
    RunCase(2, 64, 256, 128, 2, negative, valid, ge::GRAPH_FAILED);
}

TEST(AlltoAllvGroupedMatMulShmemTiling, RejectsWorkspaceCapacityOverflow)
{
    // Two stages * EP * commInterval * M0 * K no longer fit the fixed 400 MiB source window.
    std::vector<int64_t> counts(8, 1);
    RunCase(8, 8, 131072, 128, 1, counts, counts, ge::GRAPH_FAILED);
}
} // namespace AlltoAllvGroupedMatMulShmemUT
