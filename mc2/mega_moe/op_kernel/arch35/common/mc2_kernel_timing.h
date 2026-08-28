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
 * \file mc2_kernel_timing.h
 * \brief MegaMoe wave 路径 kernel 内部打点计时（AIC + AIV 混合模式）。
 *
 * 全部打点受 MC2_KERNEL_TIMING_ENABLE 宏门控：未开启时所有宏退化为空语句、
 * 全局状态数组不参与编译，零运行时开销，因此打点可常驻主线。
 * 开启后每核把 [accCycles|firstStart|lastStop|hitCount]×MAX_SLOTS 写到
 * timingBase（挂 yOut 头部），该次运行的 y 不可用于精度比对。
 * 状态用 POD 全局数组（同仓先例 g_winRankAddr_）：aicore 不允许带构造的类类型全局。
 */

#ifndef MC2_KERNEL_TIMING_H
#define MC2_KERNEL_TIMING_H

// MC2_TIMING_SWITCH_ANCHOR （timing 构建由脚本在下一行插入 #define MC2_KERNEL_TIMING_ENABLE 1）

#include "kernel_operator.h"

namespace Mc2Kernel {

// 每 slot 在 GM 中占 4 个 uint64：[0]=accCycles [1]=firstStart [2]=lastStop [3]=hitCount
constexpr int32_t TIMING_WORDS_PER_SLOT = 4;
constexpr int32_t MAX_TIMING_SLOTS = 32; // 每核 GM 占用 32*4*8B = 1KB

// wave(A8W8) 路径 slot 分配。编号一经发布不再改动，Python 解析侧一一对应。
// 「累加」slot 位于专家/wave 循环体内，多次 Start/Stop 自动累加并记命中次数。
// 同一 slot 在 AIC / AIV0 / AIV1 上执行不同分支，解析须按核类型分组看。
enum MegaMoeWaveTs : int32_t {
    TS_TOTAL = 0,               // Process 全程（各核）
    TS_INPUT_PREP = 1,          // quant + send_mask + reset (+prep) 段（AIV）
    TS_CROSS_SYNC_IN = 2,       // 输入侧跨卡屏障（AIV）
    TS_DISPATCH_ALL = 3,        // DispatchAllExpertTokens 整段（AIV1）
    TS_DISPATCH_EXPERT = 4,     // 累加：单专家 RunMoeExpertDispatchStage（AIV1）
    TS_GMM1_WAIT_COUNT = 5,     // 累加：WaitForMoeExpertTokenCountReady（AIC/AIV0）
    TS_GMM1_WAVE = 6,           // 累加：ProcessGmm1Wave 每次调用（AIC/AIV0）
    TS_GMM1_WAIT_DATA = 7,      // 累加：WaitForGmm1InputReady 自旋（AIC）
    TS_GMM2_WAVE = 8,           // 累加：ProcessGmm2Wave 每次调用（AIC）
    TS_GMM2_WAIT_ACT = 9,       // 累加：WaitForGmm2InputReady 自旋（AIC）
    TS_COMBINE_TOTAL = 10,      // 累加：ProcessCombineExperts 每次调用（AIV1）
    TS_COMBINE_WAIT_READY = 11, // 累加：WaitWaveGmm2Ready 自旋（AIV1）
    TS_SYNC_AFTER_MOE = 12,     // MoE 段后 SyncAll（AIV）
    TS_CROSS_SYNC_OUT = 13,     // 输出侧跨卡屏障（AIV）
    TS_UNPERMUTE = 14,          // Unpermute（AIV）
    TS_AIC_WAIT_VECTOR = 15,    // 累加：WaitForVector 硬等待（AIC 等 AIV 释放 pingpong 缓冲）
    TS_AIV_WAIT_CUBE = 16,      // 累加：WaitForCube 硬等待（AIV 等 AIC 的 mmad 结果）
    TS_EPILOGUE_TILE = 17,      // 累加：AIV0 单 tile epilogue 本体（swiglu+MX 量化+GM 写）
    TS_MMAD_TILE = 18,          // 累加：AIC 单 tile mmad 本体（含 B 加载与 FIXP UB 写）
    TS_DISPATCH_THROTTLE = 19,  // 累加：AIV1 节流空转（dispatch 领先额度用尽且无就绪 combine）
    TS_WAVE_SLOT_END = 20,      // 哨兵
};

} // namespace Mc2Kernel

#ifdef MC2_KERNEL_TIMING_ENABLE

// aicore 只允许 GM_ADDR（GM 指针）类型的全局（同仓先例 g_winRankAddr_，连 POD u64
// 数组都会报 "global variable is not allowed in aicore function"）。借指针槽按值存
// u64：单份布局 [0,32)=acc [32,64)=start [64,96)=first [96,128)=last [128,160)=hit
// [160]=base [161]=coreId。
// 实测同 block 的 AIV0/AIV1 子核共享这份全局存储（AIC 独立、AIV 两子核互踩出垃圾；
// g_winRankAddr_ 不炸是因为全核写同值幂等），故开双份、按 GetSubBlockIdx() 分半寻址。
constexpr int32_t MC2_TS_REGION_WORDS = Mc2Kernel::MAX_TIMING_SLOTS * 5 + 2;
inline GM_ADDR g_mc2TsWords_[MC2_TS_REGION_WORDS * 2];

__aicore__ inline int32_t Mc2TsRegionBase()
{
    if constexpr (g_coreType == AscendC::AIV) {
        return static_cast<int32_t>(AscendC::GetSubBlockIdx()) * MC2_TS_REGION_WORDS;
    }
    return 0;
}

__aicore__ inline uint64_t Mc2TsGet(int32_t idx)
{
    return reinterpret_cast<uint64_t>(g_mc2TsWords_[Mc2TsRegionBase() + idx]);
}

__aicore__ inline void Mc2TsSet(int32_t idx, uint64_t value)
{
    g_mc2TsWords_[Mc2TsRegionBase() + idx] = reinterpret_cast<GM_ADDR>(value);
}

__aicore__ inline void Mc2TsInit(GM_ADDR timingBase, int32_t coreId)
{
    for (int32_t i = 0; i < Mc2Kernel::MAX_TIMING_SLOTS * 5; i++) {
        Mc2TsSet(i, 0UL);
    }
    g_mc2TsWords_[Mc2TsRegionBase() + Mc2Kernel::MAX_TIMING_SLOTS * 5] = timingBase;
    Mc2TsSet(Mc2Kernel::MAX_TIMING_SLOTS * 5 + 1, static_cast<uint64_t>(coreId));
}

__aicore__ inline void Mc2TsStart(int32_t slot)
{
    if (slot < 0 || slot >= Mc2Kernel::MAX_TIMING_SLOTS) {
        return;
    }
    uint64_t now = static_cast<uint64_t>(AscendC::GetSystemCycle());
    Mc2TsSet(Mc2Kernel::MAX_TIMING_SLOTS + slot, now);             // start
    if (Mc2TsGet(Mc2Kernel::MAX_TIMING_SLOTS * 4 + slot) == 0UL) { // hit==0
        Mc2TsSet(Mc2Kernel::MAX_TIMING_SLOTS * 2 + slot, now);     // first
    }
}

__aicore__ inline void Mc2TsStop(int32_t slot)
{
    if (slot < 0 || slot >= Mc2Kernel::MAX_TIMING_SLOTS) {
        return;
    }
    uint64_t now = static_cast<uint64_t>(AscendC::GetSystemCycle());
    uint64_t start = Mc2TsGet(Mc2Kernel::MAX_TIMING_SLOTS + slot);
    if (start == 0UL) {
        return; // 未配对的 Stop 丢弃
    }
    Mc2TsSet(slot, Mc2TsGet(slot) + (now - start));        // acc
    Mc2TsSet(Mc2Kernel::MAX_TIMING_SLOTS * 3 + slot, now); // last
    Mc2TsSet(Mc2Kernel::MAX_TIMING_SLOTS * 4 + slot,
             Mc2TsGet(Mc2Kernel::MAX_TIMING_SLOTS * 4 + slot) + 1UL); // hit
    Mc2TsSet(Mc2Kernel::MAX_TIMING_SLOTS + slot, 0UL);                // start 清零
}

// 核退出前调用一次；每核只写自己 coreId 的区域，无跨核同步。
// timing 区若复用业务缓冲（yOut 头），调用方须先做覆盖全部核的 SyncAll。
__aicore__ inline void Mc2TsFlush()
{
    GM_ADDR timingBase = g_mc2TsWords_[Mc2TsRegionBase() + Mc2Kernel::MAX_TIMING_SLOTS * 5];
    if (timingBase == nullptr) {
        return;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<uint64_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(timingBase));
    int64_t coreId = static_cast<int64_t>(Mc2TsGet(Mc2Kernel::MAX_TIMING_SLOTS * 5 + 1));
    int64_t base = coreId * Mc2Kernel::MAX_TIMING_SLOTS * Mc2Kernel::TIMING_WORDS_PER_SLOT;
    for (int32_t i = 0; i < Mc2Kernel::MAX_TIMING_SLOTS; i++) {
        int64_t off = base + static_cast<int64_t>(i) * Mc2Kernel::TIMING_WORDS_PER_SLOT;
        output.SetValue(off + 0, Mc2TsGet(i));
        output.SetValue(off + 1, Mc2TsGet(Mc2Kernel::MAX_TIMING_SLOTS * 2 + i));
        output.SetValue(off + 2, Mc2TsGet(Mc2Kernel::MAX_TIMING_SLOTS * 3 + i));
        output.SetValue(off + 3, Mc2TsGet(Mc2Kernel::MAX_TIMING_SLOTS * 4 + i));
    }
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(output);
}

#define MC2_TS_INIT(base, coreId) Mc2TsInit((base), (coreId))
#define MC2_TS_START(slot) Mc2TsStart(Mc2Kernel::slot)
#define MC2_TS_STOP(slot) Mc2TsStop(Mc2Kernel::slot)
#define MC2_TS_FLUSH() Mc2TsFlush()
#else
#define MC2_TS_INIT(base, coreId) ((void)0)
#define MC2_TS_START(slot) ((void)0)
#define MC2_TS_STOP(slot) ((void)0)
#define MC2_TS_FLUSH() ((void)0)
#endif

#endif // MC2_KERNEL_TIMING_H
