/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLTO_ALLV_GROUPED_MAT_MUL_SHMEM_TILING_H
#define ALLTO_ALLV_GROUPED_MAT_MUL_SHMEM_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace AlltoAllvGmmShmem {
constexpr uint32_t MAX_EP_WORLD_SIZE = 8;
constexpr uint32_t MAX_LOCAL_EXPERTS = 64;
constexpr uint32_t MAX_COUNT_ENTRIES = MAX_EP_WORLD_SIZE * MAX_LOCAL_EXPERTS;
constexpr uint32_t WORKSPACE_STAGES = 2;
constexpr uint64_t WORKSPACE_DATA_BYTES = 400ULL * 1024ULL * 1024ULL;
constexpr uint64_t WORKSPACE_BYTES = WORKSPACE_DATA_BYTES + 4096ULL;
constexpr uint64_t SYNC_MMAD_OFFSET = WORKSPACE_DATA_BYTES;
constexpr uint64_t SYNC_COMM_OFFSET = SYNC_MMAD_OFFSET + 4ULL * sizeof(int64_t);

#pragma pack(push, 8)
struct AlltoAllvGroupedMatMulShmemTilingData {
    TCubeTiling matmulTiling;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t epWorldSize;
    uint32_t localExpertNum;
    uint32_t aicCoreNum;
    uint32_t aivCoreNum;
    uint32_t countSize;
    uint32_t m0;
    uint32_t n0;
    uint32_t k0;
    uint32_t l0K;
    uint32_t commInterval;
    uint32_t commTileM;
    uint32_t commBlockM;
    uint32_t commLoops;
    uint64_t recvTotal;
    uint64_t workspaceDataBytes;
    uint64_t syncMmadOffset;
    uint64_t syncCommOffset;
    uint32_t sendCounts[MAX_COUNT_ENTRIES];
    uint32_t recvCounts[MAX_COUNT_ENTRIES];
};
#pragma pack(pop)
static_assert(sizeof(AlltoAllvGroupedMatMulShmemTilingData) <= 8192,
              "SHMEM tiling data must fit the framework-normal tiling buffer.");
} // namespace AlltoAllvGmmShmem

#endif // ALLTO_ALLV_GROUPED_MAT_MUL_SHMEM_TILING_H
