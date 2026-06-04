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
 * \file mega_moe_base.h
 * \brief MegaMoE 基础类型与工具函数：上下文、地址信息结构体、AIC/AIV 同步原语
 */

#ifndef MEGA_MOE_BASE_H
#define MEGA_MOE_BASE_H

#include "lib/std/tuple.h"
#include "mega_moe_tiling.h"
#include "moe_init_routing_v3/moe_init_routing_v3_apt.h"
#include "mega_moe_workspace_info.h"

using namespace AscendC;

// EP 上下文：rank 信息 + HCCL peer-window 地址表
struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t epRankSize;
    uint64_t winSize;
    uint64_t epHcclBuffer[1024];
};

// 单次 GMM 调用的 GM 指针集合：A/B 数据、scale、两组 flag
struct GMMAddrInfo {
    GM_ADDR aGlobal;
    GM_ADDR bGlobal;
    GM_ADDR aScaleGlobal;
    GM_ADDR bScaleGlobal;
    __gm__ int32_t* groupFlagList;   // SwiGluToGmm2 完成标记基地址
    __gm__ int32_t* groupFlagList2;  // DispatchToGmm1 完成标记基地址（wave-grain）
};

// Peer memory 布局：per-expert token 数、init-routing 结果、combine 目标
struct PeermemInfo {
    GM_ADDR ptrBase;
    GM_ADDR ptrTokenPerExpert;
    GM_ADDR ptrInitRoutingResult;  // moe_init_routing 输出，含 data + scale
    GM_ADDR ptrCombineOutput;      // combine send 目标 buffer
    __aicore__ inline PeermemInfo() = default;

    // 从 base + PEERMEM_DATA_OFFSET 开始依次排布 peer-memory 各区域
    __aicore__ inline PeermemInfo(GM_ADDR base, const MegaMoeTilingData *tilingData)
    {
        ptrBase = base;
        int64_t offset = PEERMEM_DATA_OFFSET;
        ptrTokenPerExpert = base + offset;
        offset += tilingData->epWorldSize * Ops::Base::CeilAlign(
            (int64_t)(tilingData->epWorldSize * tilingData->expertPerRank), (int64_t)ALIGN_128) * sizeof(int32_t);
        ptrInitRoutingResult = base + offset;
        offset += tilingData->bs * tilingData->topK *
            (tilingData->h + tilingData->h / MXFP_SCALE_GROUP_NUM) * sizeof(int8_t);
        ptrCombineOutput = base + offset;
    }
};

// 聚合 kernel 参数：所有输入/输出 GM 地址 + workspace + peer-memory 布局
struct Params {
    GM_ADDR aGmAddr;
    GM_ADDR expertIdxGmAddr;
    GM_ADDR bGmAddr;
    GM_ADDR bScaleGmAddr;
    GM_ADDR b2GmAddr;
    GM_ADDR b2ScaleGmAddr;
    GM_ADDR probsGmAddr;
    GM_ADDR y2GmAddr;
    GM_ADDR expertTokenNumsOutGmAddr;
    WorkSpaceInfo workSpaceInfo;
    PeermemInfo peermemInfo;
    MegaMoeTilingData *tilingData;
};

// ---------------------------------------------------------------------------
// AIC / AIV 跨核同步原语
// ---------------------------------------------------------------------------

// AIV → AIC 通知
__aicore__ inline void NotifyCube(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

// AIC 等待 AIV 通知（可选 ping-pong index）
__aicore__ inline void WaitForVector(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}

// AIC → AIV 通知
__aicore__ inline void NotifyVector(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

// AIV 等待 AIC 通知（可选 ping-pong index）
__aicore__ inline void WaitForCube(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

// 预注册的 peer-window 基地址（由 Mc2MoeContext::epHcclBuffer 填充）
inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];

// 返回指定 rank 在给定偏移处的 peer-memory 地址
__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(winRankAddr_[rankId] + offset);
}

// 解引用多级 GM 指针（用于 tiling 结构体中存储的 weight tensor 地址）
__aicore__ inline GM_ADDR GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<GM_ADDR>(*(retPtr + index));
}

// tuple 元素递归访问 — 单索引
template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(t));
}

// tuple 元素递归访问 — 多索引
template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(t)));
}

// SIMT kernel：逆映射 sorted expert indices → expanded row indices
// 对每个元素 i: expandedRowIdxGmAddr[sorted[i]] = indexBase + i
__simt_vf__ __aicore__ LAUNCH_BOUND(MoeInitRoutingV3::SIMT_THREAD_NUM)
inline void ArgsortSimt(int64_t elements, int64_t indexBase, __gm__ int32_t *sortedExpertIndicesGmAddr,
            __gm__ int32_t *expandedRowIdxGmAddr)
{
    for (int32_t index = static_cast<int32_t>(Simt::GetThreadIdx()); index < static_cast<int32_t>(elements);
        index += static_cast<int32_t>(Simt::GetThreadNum())) {
        int64_t outIndices = sortedExpertIndicesGmAddr[indexBase + index];
        expandedRowIdxGmAddr[outIndices] = indexBase + index;
    }
}

// Set-then-wait 静态事件同步：eventId 范围 [0, 5]
template<AscendC::HardEvent event, int32_t eventId>
__aicore__ inline void SyncFuncStatic()
{
    AscendC::SetFlag<event>(static_cast<event_t>(eventId));
    AscendC::WaitFlag<event>(static_cast<event_t>(eventId));
}

#endif