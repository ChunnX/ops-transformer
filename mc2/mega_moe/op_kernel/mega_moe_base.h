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

// ============================================================================
// MegaMoe trace 编译期开关
// ============================================================================
// 取消注释下一行（或者 -DMEGA_MOE_TRACE_ENABLE）启用 us 时间戳打点。启用后：
//   1) 所有打点用 AscendC::GetSystemCycle() / 1000 写到 y2GmAddr（即 op 的 y 输出）；
//   2) 解析脚本 scripts/parse_trace.py 读 csv，做 stage / dispatch / wave 表；
//   3) 精度对比会 FAIL（y 是 trace，不是 unpermute 结果）—— 这是预期行为；
//   4) host 侧需把 y2GmAddr 内容 dump 到 csv（保持与 parse_trace.py 表头一致）。
//
#define MEGA_MOE_TRACE_ENABLE

// ============================================================================
// MegaMoeTrace: 时间戳打点（写到 y2GmAddr，bs>=64 时 y 足够大）
// ============================================================================
// y 视为 uint64[POINTS_PER_CORE * (MAX_AIC + MAX_AIV)]，行 = core，列 = point。
//   globalCoreIdx 分配：
//     AIC blockIdx          ∈ [0,           MAX_AIC_CORES)
//     AIV blockIdx + 36     ∈ [MAX_AIC_CORES, MAX_AIC_CORES + MAX_AIV_CORES)
//
// 容量（bs=512, h=4096 → y = 512*4096*2 = 4MB = 524288 个 uint64）：
//   108 core × 260 point × 8B = 219 KB ≪ 4MB ✓
//
// Layout 分三段：
//   [P_STAGE_BASE,    +5)    stage 级（Process 全局）
//   [P_DISPATCH_BASE, +16)   per-expert dispatch 起止（AIV1 only，最多 8 expert）
//   [P_WAVE_BASE,     +240)  per-wave 12 sub-point × 最多 19 wave / core
namespace MegaMoeTrace {
constexpr int POINTS_PER_CORE = 260;
constexpr int MAX_AIC_CORES = 36;
constexpr int MAX_AIV_CORES = 72;
// 写入单位：cycle / CYCLES_PER_US = us（按 ~1GHz 计），后续 csv 直接做加减得 us 差值
constexpr uint64_t CYCLES_PER_US = 1000UL;

// ----- Stage 级（每核固定 1 个） -----
constexpr int P_KERNEL_ENTRY = 0;
constexpr int P_ROUTING_DONE = 1;
constexpr int P_CROSSRANK_DONE = 2;
constexpr int P_EXPERT_LOOP_END = 3;
constexpr int P_PROCESS_END = 4;
constexpr int P_STAGE_NUM = 5;

// ----- Per-expert dispatch（仅 AIV1=subBlock1 上有效） -----
constexpr int P_DISPATCH_BASE = P_STAGE_NUM;     // 5
constexpr int MAX_EXPERTS_TRACED = 8;
constexpr int DISPATCH_PTS_PER_EXPERT = 2;       // 0=begin, 1=end
constexpr int P_DISPATCH_SLOTS = MAX_EXPERTS_TRACED * DISPATCH_PTS_PER_EXPERT;  // 16

// ----- Per-wave sub-points -----
constexpr int P_WAVE_BASE = P_DISPATCH_BASE + P_DISPATCH_SLOTS;                 // 21
constexpr int WAVE_POINTS_PER_ITER = 12;
constexpr int MAX_WAVES_TRACED = (POINTS_PER_CORE - P_WAVE_BASE) / WAVE_POINTS_PER_ITER;  // (260-21)/12 = 19

// 单 wave 内 sub-point 含义（AIC / AIV 视角）：
//
//  sp  | 位置                | AIC 含义                                     | AIV0 (sub0)                        | AIV1 (sub1)
//  ----+---------------------+----------------------------------------------+------------------------------------+-----------------------------
//   0  | wave loop body      | wave loop iteration 入口                     | 同左                               | 同左
//   1  |                     | GMM1 函数调用前                              | 同左 (不进 GMM1 时 idle)           | 不进 GMM1，标记位置参考
//   2  | impl.h GMM1         | 第一个 tile：等 dispatch flag 完成           | 第一个 tile WaitForCube 完成       | -
//   3  | impl.h GMM1         | 第一个 tile：mmad 发出 + NotifyVector 后     | 第一个 tile epi + NotifyCube 后    | -
//   4  | impl.h GMM1         | 最后一个 tile：mmad 发出 + NotifyVector 后   | 最后一个 tile epi + NotifyCube 后  | -
//   5  | wave loop body      | GMM1 + EndSync 之后                          | 同左                               | 同左
//   6  |                     | GMM2 函数调用前                              | (sub0 不进 GMM2 combine)           | GMM2 combine 进入前
//   7  | impl.h GMM2         | 第一个 tile：等 GMM1 swiglu flag 完成        | -                                  | 第一个 tile 等 rowReady 完成
//   8  | impl.h GMM2         | 第一个 tile：mmad+atomic_add 后              | -                                  | 第一个 tile combine MTE3 后
//   9  | impl.h GMM2         | 最后一个 tile                                | -                                  | 最后一个 tile combine MTE3 后
//  10  | wave loop body      | GMM2 函数返回后                              | 同左                               | 同左
//  11  | wave loop body      | wave loop iteration 出口                     | 同左                               | 同左
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

__aicore__ inline int GetGlobalCoreIdx()
{
    int coreIdx = GetBlockIdx();
    if constexpr (g_coreType == AIC) {
        return (coreIdx < MAX_AIC_CORES) ? coreIdx : -1;
    } else {
        return (coreIdx < MAX_AIV_CORES) ? (MAX_AIC_CORES + coreIdx) : -1;
    }
}

// 核心写入函数：把当前时刻（单位 us）写到 y[globalCoreIdx * POINTS_PER_CORE + pointIndex]。
//   关闭 MEGA_MOE_TRACE_ENABLE 时函数体为空，inline 后调用被编译器消除，零运行开销。
__aicore__ inline void RecordTimestampAt(GM_ADDR baseAddr, int pointIndex)
{
#ifdef MEGA_MOE_TRACE_ENABLE
    int globalCoreIdx = GetGlobalCoreIdx();
    if (globalCoreIdx < 0 || pointIndex < 0 || pointIndex >= POINTS_PER_CORE) {
        return;
    }
    GlobalTensor<uint64_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(baseAddr));
    PipeBarrier<PIPE_ALL>();
    uint64_t us = static_cast<uint64_t>(AscendC::GetSystemCycle()) / CYCLES_PER_US;
    output.SetValue(static_cast<uint64_t>(globalCoreIdx) *
                    static_cast<uint64_t>(POINTS_PER_CORE) +
                    static_cast<uint64_t>(pointIndex), us);
#else
    (void)baseAddr; (void)pointIndex;
#endif
}

// 便捷封装：dispatch begin / end，AIV1 (subBlock1) 上调用。
__aicore__ inline void RecordDispatchPoint(GM_ADDR baseAddr, int expertIdx, bool isEnd)
{
#ifdef MEGA_MOE_TRACE_ENABLE
    if (expertIdx < 0 || expertIdx >= MAX_EXPERTS_TRACED) return;
    int idx = P_DISPATCH_BASE + expertIdx * DISPATCH_PTS_PER_EXPERT + (isEnd ? 1 : 0);
    RecordTimestampAt(baseAddr, idx);
#else
    (void)baseAddr; (void)expertIdx; (void)isEnd;
#endif
}

// 便捷封装：wave-level / tile-level sub-point。
//   waveTraceIdx 是本核连续累计的 wave iteration 序号（跨 expert 累加），从 0 开始。
//   超过 MAX_WAVES_TRACED 的 wave 不打点（数据丢失，Python 端通过 0 值识别）。
__aicore__ inline void RecordWavePoint(GM_ADDR baseAddr, int waveTraceIdx, int subPoint)
{
#ifdef MEGA_MOE_TRACE_ENABLE
    if (waveTraceIdx < 0 || waveTraceIdx >= MAX_WAVES_TRACED) return;
    if (subPoint < 0 || subPoint >= WAVE_POINTS_PER_ITER) return;
    int idx = P_WAVE_BASE + waveTraceIdx * WAVE_POINTS_PER_ITER + subPoint;
    RecordTimestampAt(baseAddr, idx);
#else
    (void)baseAddr; (void)waveTraceIdx; (void)subPoint;
#endif
}

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