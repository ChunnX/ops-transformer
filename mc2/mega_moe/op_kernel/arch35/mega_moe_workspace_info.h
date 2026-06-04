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
 * \file mega_moe_workspace_info.h
 * \brief Workspace 全局内存布局定义
 */

#ifndef MEGA_MOE_WORKSPACE_INFO_H
#define MEGA_MOE_WORKSPACE_INFO_H

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include "kernel_operator.h"
#define HOST_DEVICE __forceinline__ [aicore]
#else
#define GM_ADDR uint8_t*
#define HOST_DEVICE
#endif

namespace {
// problem shape 与 block-offset 的 tuple 索引
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

// AIC/AIV 跨核同步常量
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;

// 量化 / tiling 常量
constexpr uint32_t SWIGLU_N_HALF = 2U;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t PEERMEM_DATA_OFFSET = 1024 * 60LL;
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int32_t INT_CACHELINE = 16;
constexpr int32_t MAX_AICORE_NUM = 36;

// wave-grain dispatch：tile 高度，必须与 MegaMoeImpl::L1_TILE_M 一致
constexpr int64_t DISPATCH_WAVE_TILE_M = 256;

constexpr uint32_t BUFFER_ALIGN = 96U * 1024U * 2U;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t UNPERMUTE_LIST_NUM = 3U;
constexpr int64_t SIZE_INT_8 = 1U;
constexpr int64_t SIZE_INT_32 = 4U;
constexpr int64_t E5M2_QUANT = 3U;
constexpr int64_t E2M1_QUANT = 5U;
constexpr int64_t FP4_NUM = 2U;
} // namespace

// Workspace 全局内存布局：按顺序分配以下连续 GM 区域 —
//   GMM1 输入 data + scale（dispatch 目标）
//   SwiGLU 量化输出 data + scale（GMM2 的 A 矩阵输入）
//   Cumsum 工作区（epWorldSize × expertPerRank 的 int32 矩阵）
//   展开后的 row 索引（routing 输出，bs×topK 个 int32）
//   各 rank 的前缀和（dispatch 时 per-expert 偏移）
//   SwiGluToGmm2 flag 区（每 expert INT_CACHELINE 个 int32 槽位）
//   DispatchToGmm1 flag 区（wave-grain：每 expert maxWavesPerExpert 个槽位，INT_CACHELINE 对齐）
// 注：expandedRowIdxGather 复用 ptrGmm1InputData 内存（dispatch 完成后不再需要）。
struct WorkSpaceInfo {
    GM_ADDR ptrGmm1InputData;
    GM_ADDR ptrGmm1InputScale;
    GM_ADDR ptrSwiGluQuantData;
    GM_ADDR ptrSwiGluQuantScale;
    GM_ADDR ptrCumsumWorkSpace;
    GM_ADDR expandedRowIdx;
    GM_ADDR expandedRowIdxGather;
    GM_ADDR ptrSumBeforeRank;
    GM_ADDR ptrFlagSwiGluToGmm2;
    GM_ADDR ptrFlagDispatchToGmm1;
    int64_t workSpaceSize;
    HOST_DEVICE WorkSpaceInfo() = default;
    HOST_DEVICE WorkSpaceInfo(GM_ADDR base, const MegaMoeTilingData *tilingData)
    {
        workSpaceSize = 0;
        ptrGmm1InputData = base;
        workSpaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->h, ALIGN_512);
        ptrGmm1InputScale = base + workSpaceSize;
        workSpaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->h /
            MXFP_SCALE_GROUP_NUM, ALIGN_512);
        ptrSwiGluQuantData = base + workSpaceSize;
        workSpaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize *
            tilingData->hiddenDim / SWIGLU_N_HALF, ALIGN_512);
        ptrSwiGluQuantScale = base + workSpaceSize;
        workSpaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->hiddenDim /
            SWIGLU_N_HALF / MXFP_SCALE_GROUP_NUM, ALIGN_512);
        ptrCumsumWorkSpace = base + workSpaceSize;
        workSpaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->epWorldSize *
            tilingData->expertPerRank, ALIGN_512);
        expandedRowIdx = base + workSpaceSize;
        workSpaceSize += SIZE_INT_32 * Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->bs), ALIGN_256) *
            tilingData->topK;
        ptrSumBeforeRank = base + workSpaceSize;
        workSpaceSize += SIZE_INT_32 * Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->epWorldSize) *
            tilingData->expertPerRank, ALIGN_128);
        ptrFlagSwiGluToGmm2 = base + workSpaceSize;
        workSpaceSize += SIZE_INT_32 * tilingData->expertPerRank * INT_CACHELINE;
        ptrFlagDispatchToGmm1 = base + workSpaceSize;
        // wave-grain dispatch：每 expert 每 wave (L1_TILE_M=256行) 分配一个槽位，INT_CACHELINE 对齐
        int64_t maxWavesPerExpert = Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize),
            DISPATCH_WAVE_TILE_M);
        int64_t dispatchFlagSlotsPerExpert = Ops::Base::CeilAlign(maxWavesPerExpert,
            static_cast<int64_t>(INT_CACHELINE));
        workSpaceSize += SIZE_INT_32 * tilingData->expertPerRank * dispatchFlagSlotsPerExpert;
        expandedRowIdxGather = ptrGmm1InputData;
    }
};
#endif // MEGA_MOE_WORKSPACE_INFO_H
