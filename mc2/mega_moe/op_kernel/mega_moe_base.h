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

struct PeermemInfo {
    GM_ADDR ptrBase;
    GM_ADDR ptrTokenPerExpert;
    GM_ADDR ptrA0;      // initrouting结果，包括data+scale
    GM_ADDR ptrD;
    __aicore__ inline PeermemInfo() = default;
    __aicore__ inline PeermemInfo(GM_ADDR base, const MegaMoeTilingData *tilingData)
    {
        ptrBase = base;
        int64_t offset = PEERMEM_DATA_OFFSET;
        ptrTokenPerExpert = base + offset;
        offset += tilingData->epWorldSize * Ops::Base::CeilAlign(
            (int64_t)(tilingData->epWorldSize * tilingData->expertPerRank), (int64_t)ALIGN_128) * sizeof(int32_t);
        ptrA0 = base + offset;
        offset += tilingData->m * tilingData->topK *
            (tilingData->k + tilingData->k / MXFP_SCALE_GROUP_NUM) * sizeof(int8_t);
        ptrD = base + offset;
    }
};

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
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;
    MegaMoeTilingData *tilingData;
};

// AscendC cube/vec 信号在 AIV0 / AIV1 上 flag id 映射不对称：
//   AIV0 SetFlag(N)        ↔ AIC  WaitFlag(N)
//   AIV1 SetFlag(N)        ↔ AIC  WaitFlag(N + 16)
//   AIC  SetFlag(N)        ↔ AIV0 WaitFlag(N)
//   AIC  SetFlag(N + 16)   ↔ AIV1 WaitFlag(N)
// 下面的 NotifyCube/WaitForVector/NotifyVector/WaitForCube 默认按 AIV0 ↔ AIC 这一组。
// 涉及 AIV1 的同步必须用带 AIV1 前/后缀的版本，否则 AIC 等不到 AIV1 的信号。
constexpr uint16_t AIV1_TO_AIC_FLAG_OFFSET = 16U;
constexpr uint16_t AIC_TO_AIV1_FLAG_OFFSET = 16U;

__aicore__ inline void NotifyCube(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void WaitForVector(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}

// AIV1 SetFlag(AIV_SYNC_AIC_FLAG + value) 在 AIC 端要 wait (AIV_SYNC_AIC_FLAG + value + 16)
__aicore__ inline void WaitForVectorFromAIV1(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value + AIV1_TO_AIC_FLAG_OFFSET);
}

__aicore__ inline void NotifyVector(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void WaitForCube(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

// AIC 想通知 AIV1 时要把 flag id +16，AIV1 自己 wait 时还是用原 value
__aicore__ inline void NotifyVectorAIV1(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value + AIC_TO_AIV1_FLAG_OFFSET);
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
//   [P_WAVE_BASE,     +240)  per-wave 12 sub-point × 最多 20 wave / core
namespace MegaMoeTrace {
constexpr int POINTS_PER_CORE = 260;
constexpr int MAX_AIC_CORES = 36;
constexpr int MAX_AIV_CORES = 72;

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
//  sp  | mega_moe.h / impl.h | AIC 含义                                     | AIV0 (sub0)                        | AIV1 (sub1)
//  ----+---------------------+----------------------------------------------+------------------------------------+-----------------------------
//   0  | mega_moe.h          | wave loop iteration 入口                     | 同左                               | 同左
//   1  | mega_moe.h          | GMM1 函数调用前                              | 同左 (不进 GMM1 时 idle)           | 不进 GMM1，标记位置参考
//   2  | impl.h GMM1         | 第一个 tile：等 dispatch flag 完成           | 第一个 tile WaitForCube 完成       | -
//   3  | impl.h GMM1         | 第一个 tile：mmad 发出 + NotifyVector 后     | 第一个 tile epi + NotifyCube 后    | -
//   4  | impl.h GMM1         | 最后一个 tile：mmad 发出 + NotifyVector 后   | 最后一个 tile epi + NotifyCube 后  | -
//   5  | mega_moe.h          | GMM1 + EndSync 之后                          | 同左                               | 同左
//   6  | mega_moe.h          | GMM2 函数调用前                              | (sub0 不进 GMM2 combine)           | GMM2 combine 进入前
//   7  | impl.h GMM2         | 第一个 tile：等 GMM1 swiglu flag 完成        | -                                  | 第一个 tile 等 rowReady 完成
//   8  | impl.h GMM2         | 第一个 tile：mmad+atomic_add 后              | -                                  | 第一个 tile combine MTE3 后
//   9  | impl.h GMM2         | 最后一个 tile                                | -                                  | 最后一个 tile combine MTE3 后
//  10  | mega_moe.h          | GMM2 函数返回后                              | 同左                               | 同左
//  11  | mega_moe.h          | wave loop iteration 出口                     | 同左                               | 同左
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

// 核心写入函数：把当前 cycle 写到 y[globalCoreIdx * POINTS_PER_CORE + pointIndex]
//   PipeBarrier<PIPE_ALL> 强制 drain 所有 pipe（cube mmad / vec / mte），让 cycle 反映
//   "前面工作真的完成"的时刻。打点本身因此会引入一定 stall，但这是 trace 的必要代价。
__aicore__ inline void RecordTimestampAt(GM_ADDR baseAddr, int pointIndex)
{
    int globalCoreIdx = GetGlobalCoreIdx();
    if (globalCoreIdx < 0 || pointIndex < 0 || pointIndex >= POINTS_PER_CORE) {
        return;
    }
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<uint64_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(baseAddr));
    uint64_t cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
    output.SetValue(static_cast<uint64_t>(globalCoreIdx) *
                    static_cast<uint64_t>(POINTS_PER_CORE) +
                    static_cast<uint64_t>(pointIndex), cycle);
}

// 便捷封装：dispatch begin / end，AIV1 (subBlock1) 上调用。
__aicore__ inline void RecordDispatchPoint(GM_ADDR baseAddr, int expertIdx, bool isEnd)
{
    if (expertIdx < 0 || expertIdx >= MAX_EXPERTS_TRACED) return;
    int idx = P_DISPATCH_BASE + expertIdx * DISPATCH_PTS_PER_EXPERT + (isEnd ? 1 : 0);
    RecordTimestampAt(baseAddr, idx);
}

// 便捷封装：wave-level / tile-level sub-point。
//   waveTraceIdx 是本核连续累计的 wave iteration 序号（跨 expert 累加），从 0 开始。
//   超过 MAX_WAVES_TRACED 的 wave 不打点（数据丢失，Python 端通过 0 值识别）。
__aicore__ inline void RecordWavePoint(GM_ADDR baseAddr, int waveTraceIdx, int subPoint)
{
    if (waveTraceIdx < 0 || waveTraceIdx >= MAX_WAVES_TRACED) return;
    if (subPoint < 0 || subPoint >= WAVE_POINTS_PER_ITER) return;
    int idx = P_WAVE_BASE + waveTraceIdx * WAVE_POINTS_PER_ITER + subPoint;
    RecordTimestampAt(baseAddr, idx);
}

} // namespace MegaMoeTrace

#endif