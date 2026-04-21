/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_dispatch_ffn_combine_workspace_info.h
 * \brief
 */

#ifndef GMMSQ_ARGS_H
#define GMMSQ_ARGS_H
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include "kernel_operator.h"
#define HOST_DEVICE __forceinline__ [aicore]
#else
#define GM_ADDR uint8_t*
#define HOST_DEVICE 
#endif

namespace {
constexpr int MNK_M = 0;
constexpr int MNK_N = 1;
constexpr int MNK_K = 2;
constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_A_SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_B_SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_BIAS_OFFSET = 4UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_C_SCALE_OFFSET = 6UL;
constexpr uint64_t IDX_FLAG_OFFSET = 7UL;
constexpr uint64_t IDX_B2_OFFSET = 8UL;
constexpr uint64_t IDX_B2_SCALE_OFFSET = 9UL;
constexpr uint64_t IDX_Y2_OFFSET = 10UL;
constexpr uint64_t IDX_M_OFFSET = 11UL;
constexpr uint64_t IDX_M_TILEIDX = 0UL;
constexpr uint64_t IDX_N_TILEIDX = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDX = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDX = 3UL;
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t FLAG_ID_MAX = 16;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint32_t SWIGLU_N_HALF = 2;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int32_t PEERMEM_DATA_OFFSET = 1024 * 60;
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr uint32_t ALIGN_512 = 512;
constexpr uint32_t ALIGN_128 = 128;
constexpr int32_t INT_CACHELINE = 16;
constexpr int32_t MAX_AICORE_NUM = 36;
constexpr uint32_t NUM_128 = 128;
constexpr uint32_t NUM_256 = 256;
constexpr uint32_t NUM_384 = 384;
constexpr uint32_t NUM_512 = 512;
constexpr uint32_t BUFFERALIGN = 96U * 1024U * 2U;                        
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t ALIGN32 = 32U;
constexpr uint32_t DATARES_MUL = 3U;
} // namespace

HOST_DEVICE uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

HOST_DEVICE int64_t CeilAlign(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

struct WorkspaceInfo{
    GM_ADDR gmCtx;
    GM_ADDR ptrA;
    GM_ADDR ptrAScale;
    GM_ADDR ptrA2;
    GM_ADDR ptrA2Scale;
    GM_ADDR ptrcumsumMM;
    GM_ADDR expandedRowIdx;
    GM_ADDR expandedRowIdxGather;
    GM_ADDR ptrSumBeforeRank;
    GM_ADDR ptrFlagList;
    GM_ADDR ptrFlagList2;
    int64_t workspaceSize;
    HOST_DEVICE WorkspaceInfo() = default;
    HOST_DEVICE WorkspaceInfo(GM_ADDR base, const MoeDispatchFFNCombineTilingData *tilingData) {
        workspaceSize = 0;
        ptrA = base;
        workspaceSize += CeilAlign(static_cast<int64_t>(tilingData->maxOutputSize) * tilingData->k * sizeof(int8_t), ALIGN_512);
        ptrAScale = base + workspaceSize;
        workspaceSize += CeilAlign(static_cast<int64_t>(tilingData->maxOutputSize) * tilingData->k / MXFP_SCALE_GROUP_NUM * sizeof(int8_t), ALIGN_512);
        ptrA2 = base + workspaceSize;
        workspaceSize += CeilAlign(static_cast<int64_t>(tilingData->maxOutputSize) * tilingData->n / SWIGLU_N_HALF * sizeof(int8_t), ALIGN_512);
        ptrA2Scale = base + workspaceSize;
        workspaceSize += CeilAlign(static_cast<int64_t>(tilingData->maxOutputSize) * tilingData->n / SWIGLU_N_HALF / MXFP_SCALE_GROUP_NUM * sizeof(int8_t), ALIGN_512);
        ptrcumsumMM = base + workspaceSize;
        workspaceSize += CeilAlign(static_cast<int64_t>(tilingData->EP) * tilingData->expertPerRank * sizeof(int32_t), ALIGN_512);
        expandedRowIdx = base + workspaceSize;
        workspaceSize += CeilAlign(tilingData->m, 256) * tilingData->topK * sizeof(int32_t);
        ptrSumBeforeRank = base + workspaceSize;
        workspaceSize += CeilAlign(static_cast<int64_t>(tilingData->EP) * tilingData->expertPerRank, ALIGN_128) * sizeof(int32_t);
        ptrFlagList = base + workspaceSize;
        workspaceSize += static_cast<int64_t>(tilingData->expertPerRank) * INT_CACHELINE * sizeof(int32_t);
        ptrFlagList2 = base + workspaceSize;
        workspaceSize += static_cast<int64_t>(tilingData->expertPerRank) * INT_CACHELINE * sizeof(int32_t);
        gmCtx = base + workspaceSize;
        // workspaceSize += CeilAlign(sizeof(SchedulerContext), ALIGN_512);
        expandedRowIdxGather = ptrA;    // Reuse
    }
};
#endif // GMMSQ_ARGS_H