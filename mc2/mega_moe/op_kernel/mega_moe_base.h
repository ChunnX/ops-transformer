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
 * \brief
 */

#ifndef MEGA_MOE_BASE_H
#define MEGA_MOE_BASE_H

#include "lib/std/tuple.h"
#include "mega_moe_tiling.h"
#include "moe_init_routing_v3/arch35/moe_init_routing_v3_apt.h"
#include "mega_moe_workspace_info.h"

using namespace AscendC;

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t epRankSize;
    uint64_t winSize;
    uint64_t epHcclBuffer[1024];
};

struct GMMAddrInfo {
    GM_ADDR aGlobal;
    GM_ADDR bGlobal;
    GM_ADDR aScaleGlobal;
    GM_ADDR bScaleGlobal;
    __gm__ int32_t* groupFlagList;
    __gm__ int32_t* groupFlagList2;
    GM_ADDR gmm2OutGlobal;
};

struct PeermemInfoOptimize {
    GM_ADDR rankSyncInWorldPtr;
    GM_ADDR topKInfoInWorldSizePtr;
    GM_ADDR quantTokenScalePtr;      // initrouting结果，包括data+scale
    GM_ADDR combineSendPtr;
    __aicore__ inline PeermemInfoOptimize() = default;
    __aicore__ inline PeermemInfoOptimize(GM_ADDR base, const MegaMoeTilingData *tilingData)
    {
        rankSyncInWorldPtr = base;
        int64_t offset = PEERMEM_DATA_OFFSET;
        topKInfoInWorldSizePtr = base + offset;
        offset += tilingData->epWorldSize * 
            Ops::Base::CeilAlign((int64_t)(tilingData->bs * tilingData->topK * sizeof(int32_t)), (int64_t)ALIGN_512);
 
        quantTokenScalePtr = base + offset;
        uint32_t mxScaleNum = Ops::Base::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32));
        uint32_t dataBytes = Ops::Base::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_256)) * sizeof(int8_t);
        uint32_t scaleBytes = mxScaleNum * sizeof(int8_t);
        uint32_t tokenBytes = Ops::Base::CeilAlign(dataBytes + scaleBytes, static_cast<uint32_t>(ALIGN_32));
        offset += Ops::Base::CeilAlign((int64_t)(tilingData->bs * tokenBytes * sizeof(int8_t)), (int64_t)ALIGN_512);
        combineSendPtr = base + offset;
    }
};

struct ParamsOptimize {
    GM_ADDR aGmAddr;
    GM_ADDR expertIdxGmAddr;
    GM_ADDR bGmAddr;
    GM_ADDR bScaleGmAddr;
    GM_ADDR b2GmAddr;
    GM_ADDR b2ScaleGmAddr;
    GM_ADDR probsGmAddr;
    GM_ADDR y2GmAddr;
    GM_ADDR expertTokenNumsOutGmAddr;
    WorkspaceInfoOptimize workspaceInfo;
    PeermemInfoOptimize peermemInfo;
    MegaMoeTilingData *tilingData;
};

__aicore__ inline void NotifyCube(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void WaitForVector(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void NotifyVector(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void WaitForCube(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];
__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(winRankAddr_[rankId] + offset);
}

__aicore__ inline GM_ADDR GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<GM_ADDR>(*(retPtr + index));
}

// Base template: handles single-index case
template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(t));
}

// Recursive template: handles multiple index cases
template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(t)));
}

// MegaMoeTrace: 占位空实现，所有打点编译成 0 开销 inline 空函数
namespace MegaMoeTrace {
constexpr int P_KERNEL_ENTRY = 0;
constexpr int P_ROUTING_DONE = 1;
constexpr int P_CROSSRANK_DONE = 2;
constexpr int P_EXPERT_LOOP_END = 3;
constexpr int P_PROCESS_END = 4;
constexpr int SP_WAVE_ENTRY = 0;
constexpr int SP_GMM1_BEGIN = 1;
constexpr int SP_GMM1_FIRST_WAIT_DONE = 2;
constexpr int SP_GMM1_FIRST_TILE_DONE = 3;
constexpr int SP_GMM1_LAST_TILE_DONE = 4;
constexpr int SP_GMM1_END = 5;
constexpr int SP_GMM2_BEGIN = 6;
constexpr int SP_GMM2_FIRST_WAIT_DONE = 7;
constexpr int SP_GMM2_FIRST_TILE_DONE = 8;
constexpr int SP_GMM2_LAST_TILE_DONE = 9;
constexpr int SP_GMM2_END = 10;
constexpr int SP_WAVE_END = 11;
__aicore__ inline void RecordTimestampAt(GM_ADDR, int) {}
__aicore__ inline void RecordDispatchPoint(GM_ADDR, int, bool) {}
__aicore__ inline void RecordWavePoint(GM_ADDR, int, int) {}
} // namespace MegaMoeTrace

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

// 静态 eventId 版本的 Set/Wait pair：不依赖 TPipe（AllocEventID/FetchEventID）。
// 用法：SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, 0>();
// 约束：eventId ∈ [0, 5]（6/7 保留给硬件）。同一 HardEvent 类型下不同调用点用不同 id，
// 防止 Set 还没被 Wait 消费就被下一次 Set 覆盖。Set 紧跟 Wait 的场景任何 id 都安全。
template<AscendC::HardEvent event, int32_t eventId>
__aicore__ inline void SyncFuncStatic()
{
    // static_assert(eventId >= 0 && eventId <= 5, "SyncFuncStatic eventId must be in [0, 5]");
    AscendC::SetFlag<event>(static_cast<event_t>(eventId));
    AscendC::WaitFlag<event>(static_cast<event_t>(eventId));
}

#endif