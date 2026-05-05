/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* V2 workspace layout: ring buffers + per-stage GM arrival counters/masks */

#ifndef MEGA_MOE_V2_WORKSPACE_INFO_H
#define MEGA_MOE_V2_WORKSPACE_INFO_H

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include "kernel_operator.h"
#define HOST_DEVICE __forceinline__ [aicore]
#else
#define GM_ADDR uint8_t*
#define HOST_DEVICE
#endif

namespace {
constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_A_SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_B_SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_C_SCALE_OFFSET = 6UL;
constexpr uint64_t IDX_FLAG_OFFSET = 7UL;
constexpr uint64_t IDX_B2_OFFSET = 8UL;
constexpr uint64_t IDX_B2_SCALE_OFFSET = 9UL;
constexpr uint64_t IDX_Y2_OFFSET = 10UL;
constexpr uint64_t IDX_M_OFFSET = 11UL;
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint32_t SWIGLU_N_HALF = 2U;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int32_t PEERMEM_DATA_OFFSET = 1024 * 60;
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int32_t INT_CACHELINE = 16;
constexpr int32_t MAX_AICORE_NUM = 36;
constexpr uint32_t BUFFER_ALIGN = 96U * 1024U * 2U;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t UNPERMUTE_LIST_NUM = 3U;
constexpr int64_t SIZE_INT_8 = 1U;
constexpr int64_t SIZE_INT_32 = 4U;
constexpr int64_t SIZE_UINT_64 = 8U;
} // namespace

struct WorkspaceInfo {
    GM_ADDR gmCtx;
    // ring buffers (V2)
    GM_ADDR ptrA;             // chunkDepth*chunkM*hidden*1B (dispatch pull dest)
    GM_ADDR ptrAScale;        // chunkDepth*chunkM*hidden/32*1B
    GM_ADDR ptrA2;            // chunkDepth*chunkM*intermediate/2*1B (GMM1 quant out)
    GM_ADDR ptrA2Scale;       // chunkDepth*chunkM*intermediate/2/32*1B

    // routing/cumsum (sized as v1)
    GM_ADDR ptrcumsumMM;
    GM_ADDR expandedRowIdx;
    GM_ADDR expandedRowIdxGather;
    GM_ADDR ptrSumBeforeRank;

    // V2 stage arrival counters
    GM_ADDR ptrDispatchArrival;  // uint32 GM atomic counter, len=dispatchArrivalLen
    GM_ADDR ptrL2ArrivalMask;    // uint64 GM bitmask, len=l2ArrivalMaskLen
    GM_ADDR ptrGmm2ToCombine;    // uint32 GM atomic counter, len=gmm2ToCombineLen

    // legacy v1-aliased pointers for combine-send/epilogue compatibility
    GM_ADDR ptrFlagSwiGluToGmm2; // = ptrL2ArrivalMask (compat alias)
    GM_ADDR ptrFlagDispatchToGmm1; // = ptrDispatchArrival (compat alias)

    int64_t workspaceSize;

    HOST_DEVICE WorkspaceInfo() = default;
    HOST_DEVICE WorkspaceInfo(GM_ADDR base, const MegaMoeV2TilingData *tilingData)
    {
        workspaceSize = 0;
        // ring A : chunkDepth * chunkM * hidden(k)
        ptrA = base;
        int64_t ringRows = static_cast<int64_t>(tilingData->chunkDepth) * tilingData->chunkM;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * ringRows * tilingData->k, ALIGN_512);

        // ring AScale : chunkDepth * chunkM * hidden/32
        ptrAScale = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_8 * ringRows * tilingData->k / MXFP_SCALE_GROUP_NUM, ALIGN_512);

        // ring A2 : chunkDepth * chunkM * (n / SWIGLU_N_HALF)
        ptrA2 = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_8 * ringRows * tilingData->n / SWIGLU_N_HALF, ALIGN_512);

        // ring A2Scale : chunkDepth * chunkM * (n / SWIGLU_N_HALF / 32)
        ptrA2Scale = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_8 * ringRows * tilingData->n / SWIGLU_N_HALF / MXFP_SCALE_GROUP_NUM, ALIGN_512);

        // routing/cumsum (same layout as v1)
        ptrcumsumMM = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_32 * tilingData->epWorldSize * tilingData->expertPerRank, ALIGN_512);

        expandedRowIdx = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * Ops::Base::CeilAlign(
            static_cast<int64_t>(tilingData->m), ALIGN_256) * tilingData->topK;

        ptrSumBeforeRank = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * Ops::Base::CeilAlign(
            static_cast<int64_t>(tilingData->epWorldSize) * tilingData->expertPerRank, ALIGN_128);

        // dispatchArrival : uint32 counters [dispatchArrivalLen]
        ptrDispatchArrival = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_32 * static_cast<int64_t>(tilingData->dispatchArrivalLen), ALIGN_512);

        // l2ArrivalMask : uint64 [l2ArrivalMaskLen]
        ptrL2ArrivalMask = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_UINT_64 * static_cast<int64_t>(tilingData->l2ArrivalMaskLen), ALIGN_512);

        // gmm2ToCombine : uint32 counters [gmm2ToCombineLen]
        ptrGmm2ToCombine = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_32 * static_cast<int64_t>(tilingData->gmm2ToCombineLen), ALIGN_512);

        // legacy aliases for combine_send.h / epilogue
        ptrFlagSwiGluToGmm2 = ptrL2ArrivalMask;
        ptrFlagDispatchToGmm1 = ptrDispatchArrival;

        // expandedRowIdxGather aliases ring A (consistent with v1: scratch reused after dispatch)
        expandedRowIdxGather = ptrA;
    }
};
#endif // MEGA_MOE_V2_WORKSPACE_INFO_H
