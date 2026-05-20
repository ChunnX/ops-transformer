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
 * \file mega_moe_impl.h
 * \brief
 */

#ifndef MEGA_MOE_IMPL_H
#define MEGA_MOE_IMPL_H

#include "kernel_operator.h"

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_base.h"

#include "include/tensor_api/tensor.h"
#include "blaze/block/block_mmad_mx.h"
#include "blaze/block/block_scheduler_swizzle.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"

#include "mega_moe_combine_send.h"

using namespace AscendC;

namespace MegaMoeImpl {
using BlockScheduler = typename Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 1>;  // 3: SwizzleOffset
constexpr uint32_t ALIGN32 = 32U;
constexpr uint32_t L1_TILE_M = 256;
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t L1_TILE_K = 256;
constexpr uint32_t L0_TILE_K = 128;
constexpr uint32_t SCALE_K_L1_RATE = 2;
constexpr uint32_t SWIGLU_N_HALF = 2;
constexpr uint32_t MAX_SINGLE_MN = L1_TILE_M * L1_TILE_N;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM = (MAX_SINGLE_MN + 31U) / ALIGN32 * ALIGN32;
// UB 容量（Da Vinci AIV 每核 256KB），用于判断 GMM1 是否还能 pingpong（需要 2 块 UbFirst+UbSecond）。
constexpr uint32_t UB_CAPACITY_BYTES = 256U * 1024U;

// ============================================================================
// 动态调度调试打印 —— 定位卡死 / UB 崩溃位置
// ----------------------------------------------------------------------------
// 把下面的 1 改成 0 即可彻底关闭所有 PRINTF（每条 PRINTF 都用 #if 隔离, 关闭后零开销）。
// 开启时每个关键步骤打印一条 "[核类型 bk=核号] ..." 日志：
//   * 卡死：最后打印的那条 = 卡住的那一步的"前一步"（下一步没打出来）。
//   * 崩溃：最后打印的那条 = 崩溃的那一步的"前一步"。
//   * 读到的 task 值（op/wave/nTile/waveM）也打出来——若是垃圾值会立刻看出来。
// 用 `grep "bk=5"` 可单独看某个核的时间线。
// 注意 PRINTF 是同步的, 开启会让 kernel 慢几个数量级 —— 仅用于 debug。
// ============================================================================
#define MEGA_MOE_DYN_DEBUG 1
// 只看两组 (AIC + AIV0 + AIV1) 的 PRINTF, 32 核全打日志噪音太大且会爆缓冲。
// 调试别的核改这两个即可。
constexpr int32_t DBG_BK_A = 0;
constexpr int32_t DBG_BK_B = 25;  // 之前是 30, 改成 25 因为日志显示 bk=25 出现 -2 over-clear, 重点观察

// spin 超时阈值 (单位: cycle); 超过这个时间没等到就打日志 + break, 不再死等。
// 1GHz 时钟下 5_000_000 cycle ≈ 5ms — 远超正常 spin (~几 us), 比 50ms 触发快 10x, 更快暴露问题。
// 真正卡死的话 5ms 也足够远超正常进度差。
constexpr int64_t SPIN_TIMEOUT_CYCLES = 5000000LL;

// ============================================================================
// DYN_SCHED_DESIGN — 单核 V 调度器 (Centralized Scheduler on a Single AIV)
// ============================================================================
//
// 问题回顾：上一版用 per-(expert, wave) HBM 原子计数器 + AIC AtomicAdd 自取号 的
// 去中心化认领方案。看似清爽，实测多次卡死。根因：本硬件上 AscendC::AtomicAdd
// 只保证"加法对内存的更新原子"，并不保证返回值是 fetch-and-add 的旧值。两个核
// 并发对同一 counter 各加 1，最终值 +2 没错，但两核拿到的返回值可能都是 0。
// 这就让"用返回值当 nTileIdx" 的认领协议失效：两个核认领了同一个 tile，导致 epi
// 多 atomic_add 一次 → groupFlag 永远超过 tileNumGmm1 但某些 tile 实际没跑，其它
// 核卡在 ClaimNextTask 自旋等不到那一个 tile 的完成。
//
// --- 新设计：单写者调度 ---
//
// 用 bk=0 的 AIV0 当 scheduler（单核单线程），它独占写 taskSlot 的权利。其它
// 31 个 AIC + AIV0 + AIV1 当 worker。这样：
//   - 任务计数器只有 scheduler 一个写者 → AtomicAdd 的弱语义不再是问题
//   - worker 端逻辑大幅简化：spin slot → 处理 → clear slot
//   - 跨 expert 之间也是 scheduler 一手管理状态
//
// 代价：丢 bk=0 的 AIC（动态分核能让剩 31 核更均衡，弥补 ~3% 算力）。
//
// --- 任务空间 ---
//
// 每 expert 仍是 GMM1[w][n] + GMM2[w][n]：
//   GMM1[w][n]：n ∈ [0, tileNumGmm1)
//   GMM2[w][n]：n ∈ [0, tileNumGmm2)，需 groupFlagListGm[w] ≥ tileNumGmm1
//
// --- 通信 slot 布局 ---
//
// 每 worker (bk ∈ [1, blockNum)) 有 4 个 slot：
//   slot[bk][channel=0=GMM1][owner=0=AIC] : scheduler → AIC bk
//   slot[bk][channel=0=GMM1][owner=1=AIV] : scheduler → AIV0 bk
//   slot[bk][channel=1=GMM2][owner=0=AIC] : scheduler → AIC bk
//   slot[bk][channel=1=GMM2][owner=1=AIV] : scheduler → AIV1 bk
//
// slotIdx = bk * 4 + channel * 2 + owner，与 workspace 已有布局一致。
//
// 每个 task 调度器写 2 个 slot（AIC + 对应 AIV 各一份），保证 AIC 和 AIV 都能独立
// 读到完整 task 描述。
//
// --- 跨核同步 ---
//
// GMM1 (AIC bk ↔ AIV0 bk 同 pair)：UB 需要 cube↔vec 屏障。沿用静态
// GroupMatmulSwigluQuant 内部的 WaitForCube(0) / NotifyVector(0) / NotifyCube(0)。
// 每 task 调一次该静态函数，vecSetSyncCom=0 让函数内的 WaitForVector 不触发，由
// 外层在两个 task 之间手动 WaitForVector(0) 保证 UB 复用前 AIV0 已写完。
//
// GMM2 (AIC bk ↔ AIV1 bk 同 pair)：cube 直接写 GM，rowReadyFlags HBM 计数器同步，
// 不依赖 cube↔vec flag。沿用静态 GroupMatmul2 内部的实现。
//
// 跨 worker 间没有同步——所有同步都 worker 本核内 + worker 与 scheduler 间。
// scheduler 与 worker 间用 slot 的 HBM 读写做无锁通信（写者唯一 → 无 race）。
//
// --- AIC worker 主循环 ---
//
//   g1_first = true; g1_end = false; g2_end = false;
//   while (!(g1_end && g2_end)):
//     if (!g1_end && ReadSlot(slot_aic_g1) != 0):
//       packed = ReadSlot(slot_aic_g1)
//       ClearSlot(slot_aic_g1, packed)
//       if (!g1_first) WaitForVector(0)   // 等上一个 G1 task 的 AIV0 NotifyCube
//       if (task.op == END):
//         g1_end = true; continue
//       call static GMM1 (single tile, vecSetSyncCom=0)
//         // 内含: mmad + NotifyVector(0)
//       g1_first = false; continue
//     if (!g2_end && ReadSlot(slot_aic_g2) != 0):
//       packed = ReadSlot(slot_aic_g2)
//       ClearSlot(slot_aic_g2, packed)
//       if (task.op == END):
//         g2_end = true; continue
//       call static GMM2 (single tile)
//         // 内含: mmad + atomic_add rowReadyFlags
//       continue
//
// --- AIV worker 主循环（AIV0 / AIV1 镜像） ---
//
//   while (true):
//     spin slot_aiv != 0
//     packed = Read; ClearSlot
//     if (END): break
//     call static GMM1 (AIV0) / GMM2 (AIV1) (single tile)
//
// --- Scheduler 主循环 ---
//
//   wait dispatch flag
//   for (bk = 1..blockNum-1):
//     while (still有未发完的 G1 或 G2 task / END):
//       // GMM1 channel
//       if (!end_g1[bk] && slot_aic_g1[bk]==0 && slot_aiv_g1[bk]==0):
//         task = next_G1_task() (轮 wave，挑还有空 tile 的)
//         if (task): write 2 slots, dispatched_g1[w]++
//         else if (allG1Dispatched()): write END to 2 slots, end_g1[bk] = true
//       // GMM2 channel
//       if (!end_g2[bk] && slot_aic_g2[bk]==0 && slot_aiv_g2[bk]==0):
//         task = next_G2_task() (只要 swigluFlag[w] ≥ tnG1 的 wave)
//         if (task): write 2 slots, dispatched_g2[w]++
//         else if (allG2Dispatched()): write END to 2 slots, end_g2[bk] = true
//   wait all slots == 0 (worker ack END)
//
// 调度器是 AIV core，操作的是 HBM 上的 int32 slot；每 task 几次 ReadGmByPassDCache
// + 2 次 AtomicAdd（slot 初始 0 → 加上 packed 等价于赋值）。开销很小。
//
// --- 跨 expert ---
//
// 每个 expert 结束时所有 slot 都被 worker clear 回 0，scheduler 也 drain 完 ACK。
// 下一 expert scheduler 重置局部状态（dispatched_*、end_*），进入新一轮。
//
// ============================================================================

// slot owner enum：用 pp 字段表达"AIC slot vs AIV slot"
//   pp=0：scheduler → AIC bk 的 slot
//   pp=1：scheduler → AIV bk 的 slot
constexpr int32_t DYN_SLOT_AIC = 0;
constexpr int32_t DYN_SLOT_AIV = 1;

// ----------------------------------------------------------------------------
// taskSlot 读写: 每 slot 1 个 int32, AtomicAdd 写 / ReadGmByPassDCache 读 / AtomicAdd 清 0
// ----------------------------------------------------------------------------
// slot 索引 = blockIdx * 4 + channel * 2 + owner。
// 协议（slot 初始 0，startup reset）:
//   scheduler 写: 此前 slot=0（worker 已 clear），AtomicAdd(slot, packed) → slot=packed
//   worker 读 : ReadGmByPassDCache → packed → unpack
//   worker 清 : AtomicAdd(slot, -packed) → slot=0  （让 scheduler 知道 task 已被消费）
// 单写者保证：scheduler 是 slot 的唯一写者（worker 的 clear 是减去自己读到的 packed，
// 等价于回到 0），所以没有 "fetch-and-add 弱语义" 相关 race。
__aicore__ inline __gm__ int32_t* GetTaskSlotPtr(
    const Params& params, int32_t blockIdx, int32_t channel, int32_t owner)
{
    int32_t slotIdx = blockIdx * 4 + channel * 2 + owner;
    return (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot + slotIdx;
}

__aicore__ inline void WriteTaskSlot(
    const Params& params, int32_t blockIdx, int32_t channel,
    int32_t owner, const DynamicTask& task)
{
    __gm__ int32_t* p = GetTaskSlotPtr(params, blockIdx, channel, owner);
    AscendC::AtomicAdd(p, PackDynamicTask(task));
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline int32_t ReadTaskSlot(
    const Params& params, int32_t blockIdx, int32_t channel,
    int32_t owner, DynamicTask& outTask)
{
    __gm__ int32_t* p = GetTaskSlotPtr(params, blockIdx, channel, owner);
    int32_t packed = AscendC::ReadGmByPassDCache(p);
    UnpackDynamicTask(packed, outTask);
    return packed;
}

__aicore__ inline void ClearTaskSlot(
    const Params& params, int32_t blockIdx, int32_t channel,
    int32_t owner, int32_t packed)
{
    __gm__ int32_t* p = GetTaskSlotPtr(params, blockIdx, channel, owner);
    AscendC::AtomicAdd(p, -packed);
    PipeBarrier<PIPE_ALL>();
}


// GroupMatmulSwigluQuant - GMM1 + SwiGLU + Quant
// coreInGroup / coresPerWave 控制本核在当前 wave 内的分片任务；多组核之间并行处理不同的 wave。
// cube↔vec 信号是 per-pair 的（同 blockIdx 的 AIC 与 AIV 共享一个 flag 通道），不同 pair
// 之间天然隔离，所以这里无需再用 coreGroupId 区分 flag value。
//
// IsInterleaved 模板分支：
//   false（默认）：B 的列布局是 [gate || up]，scheduler 在 outputN=n/2 上 tile；每 tile 做 2 次
//                  mmad 分别写 UbFirst/UbSecond，swiglu 跨 2 个 UB 读。
//   true        ：B 已被预先交织——每 L1_TILE_N=256 列 = 128 列 gate + 128 列 up；scheduler 在
//                  全 n 上 tile，每 tile 只做 1 次 mmad，swiglu 在同 UB 行内读 gate / up 半。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsInterleaved = false>
__aicore__ inline void GroupMatmulSwigluQuant(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true, IsInterleaved>& epilogueOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t coreInGroup, uint32_t coresPerWave, int32_t& vecSetSyncCom,
    int waveTraceIdx)
{
    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = Get<N_VALUE>(problemShape);
    uint32_t k = Get<K_VALUE>(problemShape);
    uint32_t outputN = n / SWIGLU_N_HALF;  // swiglu 输出 N（两种分支都用它当地址基准）

    epilogueOp.UpdateNextProblem({m, outputN, k, 0});

    uint32_t dispatchBlockNumPerEP = params.tilingData->blockNumPerEP;

    GlobalTensor<int32_t> groupFlagListGm2;
    groupFlagListGm2.SetGlobalBuffer((__gm__ int32_t*)gmmAddrInfo.groupFlagList2);

    auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

    constexpr bool transA = false;
    constexpr bool transB = true; // 后续要改为从模板参数获取转置属性

    constexpr uint32_t c0SizeA = AuxGetC0Size<ElementA>();
    constexpr uint32_t c0SizeB = AuxGetC0Size<ElementB>();
    constexpr uint32_t c0SizeC = AuxGetC0Size<ElementC>();
    constexpr uint32_t c0SizeScale = 2U;

    using LayoutA = Std::conditional_t<transA, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutB = Std::conditional_t<transB, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutBias = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<c0SizeA>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<c0SizeB>>;
    using MakeLayoutScaleA = Std::conditional_t<
        transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutScaleB = Std::conditional_t<
        transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<c0SizeC>>;

    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleK);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleK, n);
    auto layoutBias = Te::MakeFrameLayout<LayoutBias>(1u, n); // block mmad需要传入bias
    auto layoutC = MakeLayoutC{}(L1_TILE_M, L1_TILE_N);

    using BiasType = float;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, BiasType, LayoutBias, void>;
    BlockMmad blockMmad;
    bool enableL0CPingPong = false;
    typename BlockMmad::L1Params l1Params{
        .kL1 = L1_TILE_K,
        .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE,
        .l1BufNum = 2 // 2: double buffer
    };
    typename BlockMmad::BlockShape l0TileShape{L1_TILE_M, L1_TILE_N, L0_TILE_K, 0};
    typename BlockMmad::TupleShape matmulShape{m, n, k};
    blockMmad.Init(matmulShape, l0TileShape, l1Params, false, enableL0CPingPong); // 当前固定无bias

    // GMM1 pingpong 开启条件：交织 + 两块 UbFirst/UbSecond 都能塞进 UB。
    // L1_TILE_M=128 时 2 * 128*256*4 = 256KB ≤ UB ✓；L1_TILE_M=256 时 2 * 256*256*4 = 512KB ✗。
    constexpr bool kGmm1Pingpong = IsInterleaved &&
        (MAX_SINGLE_MN_ALIGN32_NUM * sizeof(ElementC) * 2U <= UB_CAPACITY_BYTES);

    int64_t ubOffset = 0;
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);

    // 没启 pingpong 时 UbSecond 折叠到 0（与 UbFirst alias，永不访问），避免描述符越界
    int64_t ubSecondOffsetBytes = kGmm1Pingpong
        ? static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM * sizeof(ElementC))
        : static_cast<int64_t>(0);
    auto l0cOutUbSecond = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::UB, ElementC>(ubSecondOffsetBytes), layoutC);

    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layoutA);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layoutB);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
        layoutScaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
        layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layoutBias);

    // Scheduler 的 N 轴：
    //   非交织：在 swiglu 输出 N (= outputN = n/2) 上 tile，每个 scheduler tile = 1 个 swiglu 输出块（覆盖 B 的 2 个 256 列段）
    //   交织  ：在 mmad 输出 N (= n，全 B 宽) 上 tile，每个 scheduler tile = 1 次 mmad（256 列 = 128 gate + 128 up）
    uint32_t schedulerN;
    if constexpr (IsInterleaved) {
        schedulerN = n;
    } else {
        schedulerN = outputN;
    }
    BlockScheduler scheduler(
        {m, schedulerN, k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    // 本核负责本 wave 内 loopIdx ∈ {coreInGroup, coreInGroup + coresPerWave, ...} 的 tile
    //
    // 交织+pingpong 流水（IsInterleaved=true）：
    //   - mmad 输出在 UbFirst (UB[0..128KB]) / UbSecond (UB[128..256KB]) 间交替
    //   - 整套 swigluquant buffer（gluRes/quantOutput/scales）跟着所在 ping/pong 区整体偏移 128KB
    //   - cube/vec 信号 flag value = pingpongIdx (0/1)，2 个独立通道
    //   - AIC 第 3 次起 WaitForVector(pingpongIdx)：流水填满后才等
    // 非交织（IsInterleaved=false）：保持原来单通道串行（每 tile 2 次 mmad 占满两块 UB，无法 pingpong）
    uint16_t pingpongIdx = 0;
    for (uint32_t loopIdx = coreInGroup; loopIdx < tileNum; loopIdx += coresPerWave) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        bool isFirstTile = (loopIdx == coreInGroup);
        bool isLastTile = (loopIdx + coresPerWave >= tileNum);

        if constexpr (g_coreType == AscendC::AIC) {
            if (loopIdx == coreInGroup) {
                uint32_t targetValue = params.tilingData->epWorldSize * dispatchBlockNumPerEP;
                __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm2.GetPhyAddr();
                int64_t _spinStart = AscendC::GetSystemCycle();
                while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    if (AscendC::GetSystemCycle() - _spinStart > SPIN_TIMEOUT_CYCLES) {
                        int dbgBk = (int)(GetBlockIdx() / GetTaskRation());
                        int32_t dbgCur = AscendC::ReadGmByPassDCache(flagValueAddr);
                        AscendC::PRINTF("[static GMM1 AIC bk=%d] *** TIMEOUT spin dispatch flag target=%u cur=%d\n",
                                        dbgBk, targetValue, dbgCur);
                        break;
                    }
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {
                    }
                }
                // [TRACE] sp2: AIC 第一个 tile 等 dispatch flag 完成
                MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                              MegaMoeTrace::SP_GMM1_FIRST_WAIT_DONE);
            }

            // 等 AIV 释放 UB 槽：
            //   kGmm1Pingpong=true（L1_TILE_M=128）：第 3 次起才等（pingpong 流水填满后）
            //   kGmm1Pingpong=false：每次都等（单 UB，cube 与 vec 串行）
            if constexpr (kGmm1Pingpong) {
                if (vecSetSyncCom >= 2) {
                    WaitForVector(pingpongIdx);
                }
            } else {
                if (vecSetSyncCom) {
                    WaitForVector();
                }
            }

            auto gmBlockA = gmA.Slice(
                Te::MakeCoord(mLoc, kLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
            auto gmBlockScaleA = gmScaleA.Slice(
                Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));

            auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));

            typename BlockMmad::BlockShape singleShape{
                Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), Get<K_VALUE>(actualShape), 0};
            if constexpr (IsInterleaved) {
                // 交织：1 次 mmad；pingpong 时 ping=UbFirst, pong=UbSecond 交替；
                // 不 pingpong 时永远写 UbFirst（L1_TILE_M=256 即此分支）。
                auto gmBlockB = gmB.Slice(
                    Te::MakeCoord(kLoc, nLoc),
                    Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                auto gmBlockScaleB = gmScaleB.Slice(
                    Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                    Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
                if constexpr (kGmm1Pingpong) {
                    auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
                        Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                    auto tensorBlockUb = (pingpongIdx == 0) ? tensorBlockUbFirst : tensorBlockUbSecond;
                    blockMmad(
                        gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                        tensorBlockUb, singleShape);
                } else {
                    blockMmad(
                        gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                        tensorBlockUbFirst, singleShape);
                }
            } else {
                // 非交织：2 次 mmad，分别取 B 的 gate 段（nLoc）和 up 段（nLoc + outputN），
                // 输出到 UbFirst / UbSecond，swiglu 跨两个 UB 读。
                auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
                    Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                    auto gmBlockB = gmB.Slice(
                        Te::MakeCoord(kLoc, nLoc + weightBlock * outputN),
                        Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                    auto gmBlockScaleB = gmScaleB.Slice(
                        Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * outputN),
                        Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
                    blockMmad(
                        gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                        weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
                }
            }

            if constexpr (kGmm1Pingpong) {
                NotifyVector(pingpongIdx);
            } else {
                NotifyVector();
            }

            // [TRACE] sp3 / sp4: AIC GMM1 first / last tile 完成（mmad+NotifyVector 后）
            if (isFirstTile) {
                MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                              MegaMoeTrace::SP_GMM1_FIRST_TILE_DONE);
            }
            if (isLastTile) {
                MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                              MegaMoeTrace::SP_GMM1_LAST_TILE_DONE);
            }
        }

        // 计数器：pingpong 模式每 iter 递增（用于 EndSync 判断 drain 1/2 个 NotifyCube）；
        // 单 UB 模式（含非交织和交织+L1_TILE_M=256）置 1，EndSync 只 drain 一个。
        if constexpr (kGmm1Pingpong) {
            vecSetSyncCom++;
        } else {
            vecSetSyncCom = 1;
        }

        if constexpr (g_coreType == AscendC::AIV) {
            // 非交织：scheduler nLoc 已经走 swiglu 输出坐标，单 tile swiglu 输出宽 = L1_TILE_N (256)
            // 交织  ：scheduler nLoc 走 mmad N（B 全宽），swiglu 输出宽 = L1_TILE_N/2 (128)，
            //         A2 输出列号要除以 2
            uint32_t swigluNLoc;
            int64_t swigluN_tile;
            if constexpr (IsInterleaved) {
                swigluNLoc = nLoc / SWIGLU_N_HALF;
                swigluN_tile = static_cast<int64_t>(Get<N_VALUE>(actualShape)) / SWIGLU_N_HALF;
            } else {
                swigluNLoc = nLoc;
                swigluN_tile = static_cast<int64_t>(Get<N_VALUE>(actualShape));
            }
            Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{
                Get<M_VALUE>(actualShape), swigluN_tile, 0, 0};
            Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                mLoc * outputN + swigluNLoc,
                mLoc * CeilDiv(outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                    CeilDiv(swigluNLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                0, mLoc, 0, 0};
            if constexpr (kGmm1Pingpong) {
                WaitForCube(pingpongIdx);
                if (isFirstTile) {
                    MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                                  MegaMoeTrace::SP_GMM1_FIRST_WAIT_DONE);
                }
                AscendC::SetCtrlSpr<60, 60>(0);
                epilogueOp(epilogueShape, epilogueOffset, pingpongIdx);
                // ★ epi 末尾的 AtomicAdd(groupFlagListGm, 1) 在 scalar 流水, NotifyCube 在 PIPE_V。
                //   PIPE_V drain 不包含 scalar pipe → 不加 barrier 时 NotifyCube 可能在 atomic-add
                //   commit 到 HBM 之前触发, 跨核读 flag 看不到更新, 动态调度下其它核会卡在 spin。
                //   静态 cross-wave 下同 blockIdx 包揽 wave, 等的是自己核内 atomic-add (scalar 自序),
                //   所以不出问题; 动态调度跨核观察必须强 drain。
                PipeBarrier<PIPE_ALL>();
                NotifyCube(pingpongIdx);
            } else {
                // 单 UB（交织+L1_TILE_M=256 或 非交织）：单通道 cube↔vec sync
                WaitForCube();
                if (isFirstTile) {
                    MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                                  MegaMoeTrace::SP_GMM1_FIRST_WAIT_DONE);
                }
                AscendC::SetCtrlSpr<60, 60>(0);
                if constexpr (IsInterleaved) {
                    // 交织：epi 只读 UbFirst，固定传 pingpongIdx=0
                    epilogueOp(epilogueShape, epilogueOffset, static_cast<uint16_t>(0));
                } else {
                    // 非交织：epi 跨 UbFirst+UbSecond 读 gate/up
                    epilogueOp(epilogueShape, epilogueOffset);
                }
                // ★ 同上, 强 drain scalar pipe (atomic-add) 后再发 cube-vec flag
                PipeBarrier<PIPE_ALL>();
                NotifyCube();
            }

            // [TRACE] sp3 / sp4: AIV0 GMM1 epi first / last tile 完成（epi+NotifyCube 后）
            if (isFirstTile) {
                MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                              MegaMoeTrace::SP_GMM1_FIRST_TILE_DONE);
            }
            if (isLastTile) {
                MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                              MegaMoeTrace::SP_GMM1_LAST_TILE_DONE);
            }
        }

        // 切换 ping/pong（仅交织模式有意义；非交织无影响）
        pingpongIdx = 1U - pingpongIdx;
    }
}

// GroupMatmul2 - GMM2矩阵乘法，直接输出到GM（参考kernel_qbmm_mx.h的做法）
//   coreInGroup / coresPerWave 控制本核在 wave 内的分片任务；
//   groupIdx 即 expertIdx，传给 CombineTile 用于查 tokenPerExpert；
//   waveRowOffset 是本 wave 在 expert 内的起始行，用于把 wave 内 mLoc 翻译成 expert 内行号。
//   IsInterleaved 必须与 GMM1 保持一致——决定等待 GMM1 完成 flag 的目标计数：
//      非交织：targetLoops = (waveM/L1_TILE_M) * (outputN/L1_TILE_N)
//      交织  ：targetLoops = (waveM/L1_TILE_M) * (n/L1_TILE_N)  ，是非交织的 2 倍。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsInterleaved = false>
__aicore__ inline void GroupMatmul2(
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t coreInGroup, uint32_t coresPerWave,
    uint32_t groupIdx, int32_t rankId, uint64_t waveRowOffset,
    int waveTraceIdx)
{
    GlobalTensor<uint8_t> gmAGt, gmBGt, gmScaleAGt, gmScaleBGt;
    gmAGt.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gmmAddrInfo.aGlobal));
    gmBGt.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gmmAddrInfo.bGlobal));
    gmScaleAGt.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gmmAddrInfo.aScaleGlobal));
    gmScaleBGt.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gmmAddrInfo.bScaleGlobal));

    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = Get<K_VALUE>(problemShape);
    uint32_t k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;

    // 绝对的 cube 核 id，用于 rowReadyFlags 这种逐核计数器
    uint32_t blockIdx = GetBlockIdx() / GetTaskRation();

    GlobalTensor<int32_t> groupFlagListGm;
    groupFlagListGm.SetGlobalBuffer((__gm__ int32_t*)gmmAddrInfo.groupFlagList);

    // 信号量表，每个核心一个计数器
    GlobalTensor<int32_t> rowReadyFlags;
    rowReadyFlags.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrGmm2RowReady);

    auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

    constexpr bool transA = false;
    constexpr bool transB = true;

    constexpr uint32_t c0SizeA = AuxGetC0Size<ElementA>();
    constexpr uint32_t c0SizeB = AuxGetC0Size<ElementB>();
    constexpr uint32_t c0SizeC = AuxGetC0Size<ElementC>();
    constexpr uint32_t c0SizeScale = 2U;

    using LayoutA = Std::conditional_t<transA, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutB = Std::conditional_t<transB, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutBias = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<c0SizeA>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<c0SizeB>>;
    using MakeLayoutScaleA = Std::conditional_t<
        transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutScaleB = Std::conditional_t<
        transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<c0SizeC>>;

    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleK);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleK, n);
    auto layoutBias = Te::MakeFrameLayout<LayoutBias>(1U, n);
    auto layoutC = MakeLayoutC{}(m, n);  // 使用完整m,n而不是tile大小

    using BiasType = float;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, BiasType, LayoutBias, void>;
    BlockMmad blockMmad;
    bool enableL0CPingPong = false;
    typename BlockMmad::L1Params l1Params{
        .kL1 = L1_TILE_K,
        .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE,
        .l1BufNum = 2
    };
    typename BlockMmad::BlockShape l0TileShape{L1_TILE_M, L1_TILE_N, L0_TILE_K, 0};
    typename BlockMmad::TupleShape matmulShape{m, n, k};
    blockMmad.Init(matmulShape, l0TileShape, l1Params, false, enableL0CPingPong);

    auto gmC = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC*>(gmmAddrInfo.gmm2OutGlobal)), layoutC);

    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layoutA);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layoutB);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
        layoutScaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
        layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layoutBias);

    BlockScheduler scheduler(
        {m, n, k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    // 本核负责本 wave 内 loopIdx ∈ {coreInGroup, coreInGroup + coresPerWave, ...} 的 tile
    for (uint32_t loopIdx = coreInGroup; loopIdx < tileNum; loopIdx += coresPerWave) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        bool isFirstTile = (loopIdx == coreInGroup);
        bool isLastTile = (loopIdx + coresPerWave >= tileNum);

        if constexpr (g_coreType == AscendC::AIC) {
            // AIC: 执行gmm2计算
            if (loopIdx == coreInGroup) {
                // GMM1 实际 tile 数（用于 wait flag 目标计数）：
                //   非交织：GMM1 scheduler 在 {m, k=outputN, n=k_orig} 上 tile L1_TILE_M×L1_TILE_N
                //   交织  ：GMM1 scheduler 在 {m, k_full=outputN*2, n=k_orig} 上 tile，N 方向 tile 数翻倍
                int64_t schedulerN = static_cast<int64_t>(k);
                if constexpr (IsInterleaved) {
                    schedulerN = schedulerN * static_cast<int64_t>(SWIGLU_N_HALF);
                }
                BlockScheduler gmmBlockScheduler(
                    {static_cast<int64_t>(m), schedulerN, static_cast<int64_t>(n)},
                    BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M),
                        static_cast<int64_t>(L1_TILE_N))});
                uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
                __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm.GetPhyAddr();
                int64_t _spinStart = AscendC::GetSystemCycle();
                while (targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    if (AscendC::GetSystemCycle() - _spinStart > SPIN_TIMEOUT_CYCLES) {
                        int dbgBk = (int)(GetBlockIdx() / GetTaskRation());
                        int32_t dbgCur = AscendC::ReadGmByPassDCache(flagValueAddr);
                        AscendC::PRINTF("[static GMM2 AIC bk=%d] *** TIMEOUT swiglu->gmm2 flag target=%u cur=%d\n",
                                        dbgBk, targetLoops, dbgCur);
                        break;
                    }
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {
                    }
                }
                // [TRACE] sp7: AIC 第一个 tile 等到 GMM1 swiglu flag
                MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                              MegaMoeTrace::SP_GMM2_FIRST_WAIT_DONE);
            }

            auto gmBlockA = gmA.Slice(
                Te::MakeCoord(mLoc, kLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
            auto gmBlockScaleA = gmScaleA.Slice(
                Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));

            auto gmBlockC = gmC.Slice(
                Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));

            auto gmBlockB = gmB.Slice(
                Te::MakeCoord(kLoc, nLoc), Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = gmScaleB.Slice(
                Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));

            typename BlockMmad::BlockShape singleShape{
                Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), Get<K_VALUE>(actualShape), 0};
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, gmBlockC, singleShape);
            event_t eventIdFIX_S = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::FIX_S>());
            SetFlag<HardEvent::FIX_S>(eventIdFIX_S);
            WaitFlag<HardEvent::FIX_S>(eventIdFIX_S);
            AscendC::AtomicAdd((__gm__ int32_t*)rowReadyFlags.GetPhyAddr() + blockIdx, int32_t(1));
            // ★ 强制 atomic_add 的 PIPE_S commit 完成再返回, 否则下一个 task 的 mmad 流水可能
            //   抢在 commit 之前发出, 配对 AIV1 spin rowReadyFlags 读到旧值 → 整核错位。
            PipeBarrier<PIPE_ALL>();

            // [TRACE] sp8 / sp9: AIC GMM2 first / last tile 完成（mmad+atomic_add 后）
            if (isFirstTile) {
                MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                              MegaMoeTrace::SP_GMM2_FIRST_TILE_DONE);
            }
            if (isLastTile) {
                MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                              MegaMoeTrace::SP_GMM2_LAST_TILE_DONE);
            }
        } else {
            // AIV: 等待对应C核完成并combine发送
            {
                int64_t _spinStart = GetSystemCycle();
                while (ReadGmByPassDCache((__gm__ int32_t*)rowReadyFlags.GetPhyAddr() + blockIdx) < 1) {
                    if (GetSystemCycle() - _spinStart > SPIN_TIMEOUT_CYCLES) {
                        int dbgBk = (int)blockIdx;
                        unsigned dbgLoop = (unsigned)loopIdx;
                        int32_t dbgCur = ReadGmByPassDCache((__gm__ int32_t*)rowReadyFlags.GetPhyAddr() + blockIdx);
                        AscendC::PRINTF("[static GMM2 AIV1 bk=%d] *** TIMEOUT spin rowReadyFlags loopIdx=%u cur=%d\n",
                                        dbgBk, dbgLoop, dbgCur);
                        break;
                    }
                    int64_t st = GetSystemCycle();
                    while (GetSystemCycle() - st < 100) {}
                }
            }
            // [TRACE] sp7: AIV1 第一个 tile 等到 rowReady flag
            if (isFirstTile) {
                MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                              MegaMoeTrace::SP_GMM2_FIRST_WAIT_DONE);
            }
            AscendC::AtomicAdd((__gm__ int32_t*)rowReadyFlags.GetPhyAddr() + blockIdx, int32_t(-1));
            // 裸分配UB用于GM→UB→远端GM（与gmm1串行，offset 0不冲突）
            int64_t ubOffset = 0;
            LocalTensor<bfloat16_t> ubBuffer = LocalTensor<bfloat16_t>(TPosition::VECIN, ubOffset, L1_TILE_M * L1_TILE_N);
            MegaMoeCombineImpl::CombineTile<bfloat16_t, decltype(actualShape)>(
                groupIdx, mLoc, nLoc, n, rankId,
                gmmAddrInfo.gmm2OutGlobal, params, ubBuffer, actualShape, waveRowOffset);

            // [TRACE] sp8 / sp9: AIV1 GMM2 combine first / last tile 完成（CombineTile 返回后）
            if (isFirstTile) {
                MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                              MegaMoeTrace::SP_GMM2_FIRST_TILE_DONE);
            }
            if (isLastTile) {
                MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                              MegaMoeTrace::SP_GMM2_LAST_TILE_DONE);
            }
        }
    }
}



// ============================================================================
// SchedulerLoop — 单核 V 调度器主循环 (bk=0 AIV0)
// ============================================================================
// 任务：
//   1. 等本 expert 的 dispatch flag (epWorldSize * blockNumPerEP)
//   2. 维护每 wave 的 G1/G2 已派发计数; 监控 swiglu→gmm2 flag 判 G2 解锁
//   3. 扫描 worker 的 (slot_aic, slot_aiv) 对, 都为 0 时派发新 task 或 END
//   4. 全部 END 派发完 + 全部 slot 收到 ack 后退出
//
// task 编码用 PackDynamicTask, 每 task 写 2 个 slot:
//   slot[bk][channel][AIC]  和  slot[bk][channel][AIV]
// AtomicAdd 加上 packed (slot 此前 = 0)。同 channel 的两个 slot 都被 worker clear
// 回 0 后才能写下一 task。
//
// 跨 expert 隔离: 局部状态 dispatched_*[] / end_*[] 每 expert 重置。slot 由 worker
// clear 后保持 0，下一 expert scheduler 直接复用。
// ============================================================================
constexpr int32_t SCHED_MAX_WAVES = 64;     // waveCnt 上限 (m_i ≤ 16K token / 256 = 64)
constexpr int32_t SCHED_MAX_WORKERS = 64;   // blockNum 上限 (实际 32, 留余量)

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsInterleaved>
__aicore__ inline void SchedulerLoop(
    const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& baseOffset,
    int32_t rankId, uint32_t expertIdx, uint32_t blockNum,
    uint64_t waveCnt, uint64_t m_i,
    uint32_t tileNumGmm1, uint32_t tileNumGmm2)
{
    #if MEGA_MOE_DYN_DEBUG
    AscendC::PRINTF("[rank=%d SCHED] >>> ENTRY exp=%u tnG1=%u tnG2=%u waveCnt=%lu m_i=%lu blockNum=%u\n",
                    rankId, expertIdx, tileNumGmm1, tileNumGmm2, waveCnt, m_i, blockNum);
    #endif

    // -- Step 1: wait dispatch flag --
    __gm__ int32_t* dispatchFlagAddr = (__gm__ int32_t*)params.workspaceInfo.ptrFlagDispatchToGmm1 + expertIdx;
    uint32_t targetValue = params.tilingData->epWorldSize * params.tilingData->blockNumPerEP;
    {
        int64_t st = AscendC::GetSystemCycle();
        while (AscendC::ReadGmByPassDCache(dispatchFlagAddr) != static_cast<int32_t>(targetValue)) {
            if (AscendC::GetSystemCycle() - st > SPIN_TIMEOUT_CYCLES) {
                int32_t dbgCur = AscendC::ReadGmByPassDCache(dispatchFlagAddr);
                AscendC::PRINTF("[rank=%d SCHED] TIMEOUT dispatch flag exp=%u tgt=%u cur=%d\n",
                                rankId, expertIdx, targetValue, dbgCur);
                break;
            }
            int64_t ss = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - ss < 50) {}
        }
    }

    int64_t flagOffsetForExpert = Get<IDX_FLAG_OFFSET>(baseOffset);
    __gm__ int32_t* swigluFlagBase =
        (__gm__ int32_t*)params.workspaceInfo.ptrFlagSwiGluToGmm2 + flagOffsetForExpert;

    // -- Step 1.5: 强制 reset 所有 worker 的 slot 到 0 --
    //   防御上 expert 残留：若有 worker 在上 expert 超时退出而没 clear slot，slot 里会
    //   留下 packed 值。本 expert 进入后 scheduler 看到 slot != 0 会判它"被占用"永远不
    //   往里写新 task，造成卡死。SyncAll<true>() 已保证所有 worker 已退出上 expert，
    //   此处单核 reset 是安全的。
    {
        __gm__ int32_t* slotBase = (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot;
        for (uint32_t bk = 1; bk < blockNum; ++bk) {
            for (int32_t off = 0; off < 4; ++off) {
                __gm__ int32_t* p = slotBase + (bk * 4 + off);
                int32_t cur = AscendC::ReadGmByPassDCache(p);
                if (cur != 0) {
                    AscendC::AtomicAdd(p, -cur);
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    // -- Step 2: local dispatch state --
    int32_t dispatched_g1[SCHED_MAX_WAVES];
    int32_t dispatched_g2[SCHED_MAX_WAVES];
    for (int32_t i = 0; i < SCHED_MAX_WAVES; ++i) {
        dispatched_g1[i] = 0;
        dispatched_g2[i] = 0;
    }
    bool end_g1_sent[SCHED_MAX_WORKERS];
    bool end_g2_sent[SCHED_MAX_WORKERS];
    bool end_g1_acked[SCHED_MAX_WORKERS];
    bool end_g2_acked[SCHED_MAX_WORKERS];
    for (int32_t i = 0; i < SCHED_MAX_WORKERS; ++i) {
        end_g1_sent[i] = false;
        end_g2_sent[i] = false;
        end_g1_acked[i] = false;
        end_g2_acked[i] = false;
    }

    int32_t lastWaveM = static_cast<int32_t>(
        (m_i % L1_TILE_M != 0) ? (m_i % L1_TILE_M) : L1_TILE_M);
    int32_t lastWaveIdx = static_cast<int32_t>(waveCnt - 1);

    // -- Step 3: main dispatch + ack loop --
    int64_t loopStart = AscendC::GetSystemCycle();
    while (true) {
        bool allDone = true;
        for (uint32_t bk = 1; bk < blockNum; ++bk) {
            // === GMM1 channel ===
            if (!end_g1_acked[bk]) {
                allDone = false;
                __gm__ int32_t* sAIC = (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot
                                         + (bk * 4 + DYN_CHANNEL_GMM1 * 2 + DYN_SLOT_AIC);
                __gm__ int32_t* sAIV = (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot
                                         + (bk * 4 + DYN_CHANNEL_GMM1 * 2 + DYN_SLOT_AIV);
                int32_t vA = AscendC::ReadGmByPassDCache(sAIC);
                int32_t vB = AscendC::ReadGmByPassDCache(sAIV);
                if (vA == 0 && vB == 0) {
                    if (!end_g1_sent[bk]) {
                        // 找下一个有空 tile 的 wave
                        int32_t wDispatch = -1;
                        for (uint64_t w = 0; w < waveCnt; ++w) {
                            if (dispatched_g1[w] < static_cast<int32_t>(tileNumGmm1)) {
                                wDispatch = static_cast<int32_t>(w);
                                break;
                            }
                        }
                        if (wDispatch >= 0) {
                            DynamicTask t;
                            t.opType = DYN_OP_GMM1;
                            t.waveIdx = wDispatch;
                            t.nTileIdx = dispatched_g1[wDispatch];
                            t.waveM = (wDispatch == lastWaveIdx) ? lastWaveM : static_cast<int32_t>(L1_TILE_M);
                            int32_t packed = PackDynamicTask(t);
                            AscendC::AtomicAdd(sAIC, packed);
                            AscendC::AtomicAdd(sAIV, packed);
                            PipeBarrier<PIPE_ALL>();
                            dispatched_g1[wDispatch]++;
                        } else {
                            // 全部派完, 发 END
                            DynamicTask endT;
                            endT.opType = DYN_OP_END;
                            endT.waveIdx = 0;
                            endT.nTileIdx = 0;
                            endT.waveM = 0;
                            int32_t packed = PackDynamicTask(endT);
                            AscendC::AtomicAdd(sAIC, packed);
                            AscendC::AtomicAdd(sAIV, packed);
                            PipeBarrier<PIPE_ALL>();
                            end_g1_sent[bk] = true;
                        }
                    } else {
                        // END 已派发 + slot 都 0 = worker 已 ack END
                        end_g1_acked[bk] = true;
                    }
                }
            }
            // === GMM2 channel ===
            if (!end_g2_acked[bk]) {
                allDone = false;
                __gm__ int32_t* sAIC = (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot
                                         + (bk * 4 + DYN_CHANNEL_GMM2 * 2 + DYN_SLOT_AIC);
                __gm__ int32_t* sAIV = (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot
                                         + (bk * 4 + DYN_CHANNEL_GMM2 * 2 + DYN_SLOT_AIV);
                int32_t vA = AscendC::ReadGmByPassDCache(sAIC);
                int32_t vB = AscendC::ReadGmByPassDCache(sAIV);
                if (vA == 0 && vB == 0) {
                    if (!end_g2_sent[bk]) {
                        // 找下一个: wave 已解锁 (flag ≥ tnG1) 且还有空 tile
                        int32_t wDispatch = -1;
                        for (uint64_t w = 0; w < waveCnt; ++w) {
                            if (dispatched_g2[w] >= static_cast<int32_t>(tileNumGmm2)) continue;
                            int32_t flagVal = AscendC::ReadGmByPassDCache(swigluFlagBase + w);
                            if (flagVal < static_cast<int32_t>(tileNumGmm1)) continue;
                            wDispatch = static_cast<int32_t>(w);
                            break;
                        }
                        if (wDispatch >= 0) {
                            DynamicTask t;
                            t.opType = DYN_OP_GMM2;
                            t.waveIdx = wDispatch;
                            t.nTileIdx = dispatched_g2[wDispatch];
                            t.waveM = (wDispatch == lastWaveIdx) ? lastWaveM : static_cast<int32_t>(L1_TILE_M);
                            int32_t packed = PackDynamicTask(t);
                            AscendC::AtomicAdd(sAIC, packed);
                            AscendC::AtomicAdd(sAIV, packed);
                            PipeBarrier<PIPE_ALL>();
                            dispatched_g2[wDispatch]++;
                        } else {
                            // 检查所有 G2 是否派完, 是的话发 END
                            bool allG2Dispatched = true;
                            for (uint64_t w = 0; w < waveCnt; ++w) {
                                if (dispatched_g2[w] < static_cast<int32_t>(tileNumGmm2)) {
                                    allG2Dispatched = false;
                                    break;
                                }
                            }
                            if (allG2Dispatched) {
                                DynamicTask endT;
                                endT.opType = DYN_OP_END;
                                endT.waveIdx = 0;
                                endT.nTileIdx = 0;
                                endT.waveM = 0;
                                int32_t packed = PackDynamicTask(endT);
                                AscendC::AtomicAdd(sAIC, packed);
                                AscendC::AtomicAdd(sAIV, packed);
                                PipeBarrier<PIPE_ALL>();
                                end_g2_sent[bk] = true;
                            }
                            // else: 等 swiglu 解锁更多 wave; 跳过本 bk 本轮
                        }
                    } else {
                        end_g2_acked[bk] = true;
                    }
                }
            }
        }
        if (allDone) break;

        // 超时诊断 (主循环): 4 倍 SPIN_TIMEOUT 后印出每 wave 的状态, 帮助定位卡死
        if (AscendC::GetSystemCycle() - loopStart > SPIN_TIMEOUT_CYCLES * 40) {
            AscendC::PRINTF("[rank=%d SCHED] TIMEOUT main loop exp=%u\n", rankId, expertIdx);
            for (uint64_t w = 0; w < waveCnt; ++w) {
                int32_t dbgFlag = AscendC::ReadGmByPassDCache(swigluFlagBase + w);
                int32_t dbgD1 = dispatched_g1[w];
                int32_t dbgD2 = dispatched_g2[w];
                uint64_t dbgW = w;
                AscendC::PRINTF("  wave=%lu flag=%d d1=%d d2=%d\n", dbgW, dbgFlag, dbgD1, dbgD2);
            }
            // 只打没 ack 的 bk: 这些就是卡住的 bk, 让定位更准
            for (uint32_t bk = 1; bk < blockNum; ++bk) {
                int dbgE1s = end_g1_sent[bk] ? 1 : 0;
                int dbgE1a = end_g1_acked[bk] ? 1 : 0;
                int dbgE2s = end_g2_sent[bk] ? 1 : 0;
                int dbgE2a = end_g2_acked[bk] ? 1 : 0;
                if (dbgE1a == 1 && dbgE2a == 1) continue;  // 已 ack, 跳过
                int32_t s0 = AscendC::ReadGmByPassDCache(
                    (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot + (bk * 4 + 0));
                int32_t s1 = AscendC::ReadGmByPassDCache(
                    (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot + (bk * 4 + 1));
                int32_t s2 = AscendC::ReadGmByPassDCache(
                    (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot + (bk * 4 + 2));
                int32_t s3 = AscendC::ReadGmByPassDCache(
                    (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot + (bk * 4 + 3));
                AscendC::PRINTF("  STUCK bk=%u slot_g1aic=%d _aiv=%d slot_g2aic=%d _aiv=%d e1s=%d e1a=%d e2s=%d e2a=%d\n",
                                bk, s0, s1, s2, s3, dbgE1s, dbgE1a, dbgE2s, dbgE2a);
            }
            break;
        }
    }

    #if MEGA_MOE_DYN_DEBUG
    AscendC::PRINTF("[rank=%d SCHED] <<< EXIT exp=%u\n", rankId, expertIdx);
    #endif
}


// ============================================================================
// DynamicAicLoop — AIC worker 主循环 (bk ∈ [1, blockNum))
// ============================================================================
// 协议:
//   - 不再自取号; 由 scheduler 把 task 写到 slot_aic_g1/_g2 (HBM)
//   - 主循环轮询两个 slot, 拿到非 0 就 clear + 处理
//   - G1: 调静态 GroupMatmulSwigluQuant (单 tile), 内含 mmad + NotifyVector(0)
//     上一个 G1 task 的 AIV0 NotifyCube 用本核外层 WaitForVector(0) 在下一 G1 task
//     之前 drain, 保证 UB 复用安全
//   - G2: 调静态 GroupMatmul2 (单 tile), 内含 mmad + atomic_add rowReadyFlags
//   - END (op=2) 跳过工作; 退出时 drain 最后一发 G1 NotifyCube (若有)
// ============================================================================
template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsInterleaved>
__aicore__ inline void DynamicAicLoop(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true, IsInterleaved>& epilogueOp,
    const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& baseOffset,
    int32_t blockIdx, int32_t rankId, uint32_t expertIdx,
    uint32_t tileNumGmm1, uint32_t tileNumGmm2)
{
    #if MEGA_MOE_DYN_DEBUG
    if (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B) {
        AscendC::PRINTF("[rank=%d AIC bk=%d] >>> ENTRY exp=%u tnG1=%u tnG2=%u\n",
                        rankId, blockIdx, expertIdx, tileNumGmm1, tileNumGmm2);
    }
    #endif

    int64_t flagOffsetForExpert = Get<IDX_FLAG_OFFSET>(baseOffset);
    uint64_t n_full = static_cast<uint64_t>(Get<N_VALUE>(problemShape));
    uint64_t k_full = static_cast<uint64_t>(Get<K_VALUE>(problemShape));
    uint64_t outputN = n_full / SWIGLU_N_HALF;
    uint64_t scaleK_gmm1 = Ops::Base::CeilDiv(k_full, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                            MXFP_MULTI_BASE_SIZE;
    uint64_t scaleOutputN = Ops::Base::CeilDiv(outputN, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                             MXFP_MULTI_BASE_SIZE;

    __gm__ int32_t* slotAicG1 = (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot
                                  + (blockIdx * 4 + DYN_CHANNEL_GMM1 * 2 + DYN_SLOT_AIC);
    __gm__ int32_t* slotAicG2 = (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot
                                  + (blockIdx * 4 + DYN_CHANNEL_GMM2 * 2 + DYN_SLOT_AIC);

    // g1Pending: 是否有"还没被 WaitForVector drain"的 G1 NotifyCube
    bool g1Pending = false;
    bool g1End = false;
    bool g2End = false;
    int g1Seq = 0;
    int g2Seq = 0;

    int64_t loopStart = AscendC::GetSystemCycle();
    while (!g1End || !g2End) {
        // -- 1. 优先轮 G1 slot --
        if (!g1End) {
            int32_t packed = AscendC::ReadGmByPassDCache(slotAicG1);
            if (packed != 0) {
                DynamicTask task;
                UnpackDynamicTask(packed, task);
                ClearTaskSlot(params, blockIdx, DYN_CHANNEL_GMM1, DYN_SLOT_AIC, packed);

                if (task.opType == DYN_OP_END) {
                    g1End = true;
                    #if MEGA_MOE_DYN_DEBUG
                    if (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B) {
                        AscendC::PRINTF("[rank=%d AIC bk=%d] G1 END seen\n", rankId, blockIdx);
                    }
                    #endif
                    continue;
                }

                // 在下一个 G1 mmad 前 drain 上一 G1 task 的 AIV0 NotifyCube
                if (g1Pending) {
                    WaitForVector(0);
                    g1Pending = false;
                }

                g1Seq++;
                #if MEGA_MOE_DYN_DEBUG
                if (g1Seq <= 3 && (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B)) {
                    int32_t dbgW = task.waveIdx;
                    int32_t dbgN = task.nTileIdx;
                    int32_t dbgM = task.waveM;
                    int dbgSeq = g1Seq;
                    AscendC::PRINTF("[rank=%d AIC bk=%d] G1#%d w=%d nT=%d waveM=%d\n",
                                    rankId, blockIdx, dbgSeq, dbgW, dbgN, dbgM);
                }
                #endif

                auto taskProblemShape = problemShape;
                Get<M_VALUE>(taskProblemShape) = static_cast<int64_t>(task.waveM);
                int64_t waveRowOffset = static_cast<int64_t>(task.waveIdx) * static_cast<int64_t>(L1_TILE_M);

                GMMAddrInfo gmmAddrInfo;
                gmmAddrInfo.aGlobal = params.workspaceInfo.ptrA +
                    (Get<IDX_A_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(k_full)) *
                    sizeof(ElementA);
                gmmAddrInfo.aScaleGlobal = params.workspaceInfo.ptrAScale +
                    (Get<IDX_A_SCALE_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(scaleK_gmm1)) *
                    sizeof(ElementMxScaleA);
                gmmAddrInfo.bGlobal = params.bGmAddr +
                    Get<IDX_B_OFFSET>(baseOffset) * sizeof(ElementB);
                gmmAddrInfo.bScaleGlobal = params.bScaleGmAddr +
                    Get<IDX_B_SCALE_OFFSET>(baseOffset) * sizeof(ElementMxScaleB);
                gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params.workspaceInfo.ptrFlagSwiGluToGmm2 +
                    flagOffsetForExpert + static_cast<int64_t>(task.waveIdx);
                gmmAddrInfo.groupFlagList2 = (__gm__ int32_t*)params.workspaceInfo.ptrFlagDispatchToGmm1 + expertIdx;
                gmmAddrInfo.gmm2OutGlobal = nullptr;

                AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                    Get<IDX_C_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(outputN),
                    Get<IDX_C_SCALE_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(scaleOutputN),
                    Get<IDX_FLAG_OFFSET>(baseOffset) + static_cast<int64_t>(task.waveIdx),
                    0L, 0L, 0L};
                epilogueOp.UpdateGlobalAddr(vecBaseOffset);

                int32_t funcSync = 0;
                GroupMatmulSwigluQuant<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsInterleaved>(
                    epilogueOp, params, taskProblemShape, gmmAddrInfo,
                    static_cast<uint32_t>(task.nTileIdx), tileNumGmm1,
                    funcSync, 0);
                g1Pending = true;  // 静态函数内已发 NotifyVector(0)
                continue;
            }
        }
        // -- 2. 否则轮 G2 slot --
        if (!g2End) {
            int32_t packed = AscendC::ReadGmByPassDCache(slotAicG2);
            if (packed != 0) {
                DynamicTask task;
                UnpackDynamicTask(packed, task);
                ClearTaskSlot(params, blockIdx, DYN_CHANNEL_GMM2, DYN_SLOT_AIC, packed);

                if (task.opType == DYN_OP_END) {
                    g2End = true;
                    #if MEGA_MOE_DYN_DEBUG
                    if (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B) {
                        AscendC::PRINTF("[rank=%d AIC bk=%d] G2 END seen\n", rankId, blockIdx);
                    }
                    #endif
                    continue;
                }

                g2Seq++;
                #if MEGA_MOE_DYN_DEBUG
                if (g2Seq <= 3 && (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B)) {
                    int32_t dbgW = task.waveIdx;
                    int32_t dbgN = task.nTileIdx;
                    int32_t dbgM = task.waveM;
                    int dbgSeq = g2Seq;
                    AscendC::PRINTF("[rank=%d AIC bk=%d] G2#%d w=%d nT=%d waveM=%d\n",
                                    rankId, blockIdx, dbgSeq, dbgW, dbgN, dbgM);
                }
                #endif

                auto taskProblemShape = problemShape;
                Get<M_VALUE>(taskProblemShape) = static_cast<int64_t>(task.waveM);
                int64_t waveRowOffset = static_cast<int64_t>(task.waveIdx) * static_cast<int64_t>(L1_TILE_M);

                GMMAddrInfo gmmAddrInfo;
                gmmAddrInfo.aGlobal = params.workspaceInfo.ptrA2 +
                    (Get<IDX_C_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(outputN)) *
                    sizeof(ElementA);
                gmmAddrInfo.aScaleGlobal = params.workspaceInfo.ptrA2Scale +
                    (Get<IDX_C_SCALE_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(scaleOutputN)) *
                    sizeof(ElementMxScaleA);
                gmmAddrInfo.bGlobal = params.b2GmAddr +
                    Get<IDX_B2_OFFSET>(baseOffset) * sizeof(ElementB);
                gmmAddrInfo.bScaleGlobal = params.b2ScaleGmAddr +
                    Get<IDX_B2_SCALE_OFFSET>(baseOffset) * sizeof(ElementMxScaleB);
                gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params.workspaceInfo.ptrFlagSwiGluToGmm2 +
                    flagOffsetForExpert + static_cast<int64_t>(task.waveIdx);
                gmmAddrInfo.groupFlagList2 = nullptr;
                gmmAddrInfo.gmm2OutGlobal = params.workspaceInfo.ptrGmmOut +
                    (Get<IDX_Y2_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(k_full)) *
                    sizeof(bfloat16_t);

                GroupMatmul2<ElementA, ElementB, bfloat16_t, ElementMxScaleA, ElementMxScaleB, IsInterleaved>(
                    params, taskProblemShape, gmmAddrInfo,
                    static_cast<uint32_t>(task.nTileIdx), tileNumGmm2,
                    expertIdx, rankId, static_cast<uint64_t>(waveRowOffset), 0);
                continue;
            }
        }
        // 两个 slot 都空, 短暂等待 (scheduler 正在派或上一 task 还没消费完)
        // 超时诊断: 防止永久卡死
        if (AscendC::GetSystemCycle() - loopStart > SPIN_TIMEOUT_CYCLES * 40) {
            int32_t dbgS1 = AscendC::ReadGmByPassDCache(slotAicG1);
            int32_t dbgS2 = AscendC::ReadGmByPassDCache(slotAicG2);
            int dbgG1End = g1End ? 1 : 0;
            int dbgG2End = g2End ? 1 : 0;
            int dbgG1Pending = g1Pending ? 1 : 0;
            AscendC::PRINTF("[rank=%d AIC bk=%d] *** TIMEOUT idle exp=%u g1End=%d g2End=%d g1=%d g2=%d g1Pending=%d\n",
                            rankId, blockIdx, expertIdx, dbgG1End, dbgG2End,
                            dbgS1, dbgS2, dbgG1Pending);
            break;
        }
        int64_t ss = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - ss < 50) {}
    }

    // 最后一发 G1 NotifyCube 必须 drain (没 drain 的话退出 expert 后 flag 残留)
    if (g1Pending) {
        #if MEGA_MOE_DYN_DEBUG
        if (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B) {
            AscendC::PRINTF("[rank=%d AIC bk=%d] final WaitForVector(0) drain\n", rankId, blockIdx);
        }
        #endif
        WaitForVector(0);
    }

    #if MEGA_MOE_DYN_DEBUG
    if (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B) {
        AscendC::PRINTF("[rank=%d AIC bk=%d] <<< EXIT exp=%d (G1=%d G2=%d)\n",
                        rankId, blockIdx, expertIdx, g1Seq, g2Seq);
    }
    #endif
}

// ============================================================================
// DynamicAiv0Loop — AIV0 worker, GMM1 epi 端
// ============================================================================
// 协议: spin AIV slot → read & clear → 调静态 GroupMatmulSwigluQuant (AIV 分支:
//   WaitForCube(0) → epi → NotifyCube(0))。END 跳过函数, 不发 flag (AIC 不等)。
// ============================================================================
template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsInterleaved>
__aicore__ inline void DynamicAiv0Loop(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true, IsInterleaved>& epilogueOp,
    const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& baseOffset,
    int32_t blockIdx, int32_t rankId, uint32_t expertIdx, uint32_t tileNumGmm1)
{
    #if MEGA_MOE_DYN_DEBUG
    if (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B) {
        AscendC::PRINTF("[rank=%d AIV0 bk=%d] >>> ENTRY exp=%d\n", rankId, blockIdx, expertIdx);
    }
    #endif
    int64_t flagOffsetForExpert = Get<IDX_FLAG_OFFSET>(baseOffset);
    uint64_t k_full = static_cast<uint64_t>(Get<K_VALUE>(problemShape));
    uint64_t n_full = static_cast<uint64_t>(Get<N_VALUE>(problemShape));
    uint64_t outputN = n_full / SWIGLU_N_HALF;
    uint64_t scaleK_gmm1 = Ops::Base::CeilDiv(k_full, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                            MXFP_MULTI_BASE_SIZE;
    uint64_t scaleOutputN = Ops::Base::CeilDiv(outputN, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                             MXFP_MULTI_BASE_SIZE;

    __gm__ int32_t* slotPtr = (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot
                                + (blockIdx * 4 + DYN_CHANNEL_GMM1 * 2 + DYN_SLOT_AIV);

    int taskSeq = 0;
    while (true) {
        int32_t packed = 0;
        int64_t spinStart = AscendC::GetSystemCycle();
        bool timedOut = false;
        while (true) {
            packed = AscendC::ReadGmByPassDCache(slotPtr);
            if (packed != 0) break;
            if (AscendC::GetSystemCycle() - spinStart > SPIN_TIMEOUT_CYCLES * 40) {
                int dbgSeq = taskSeq + 1;
                AscendC::PRINTF("[rank=%d AIV0 bk=%d] *** TIMEOUT spin slot exp=%u seq=%d\n",
                                rankId, blockIdx, expertIdx, dbgSeq);
                timedOut = true;
                break;
            }
            int64_t ss = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - ss < 50) {}
        }
        if (timedOut) break;

        DynamicTask task;
        UnpackDynamicTask(packed, task);
        ClearTaskSlot(params, blockIdx, DYN_CHANNEL_GMM1, DYN_SLOT_AIV, packed);

        if (task.opType == DYN_OP_END) {
            break;
        }
        taskSeq++;
        #if MEGA_MOE_DYN_DEBUG
        if (taskSeq <= 3 && (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B)) {
            int32_t dbgW = task.waveIdx;
            int32_t dbgN = task.nTileIdx;
            int dbgSeq = taskSeq;
            AscendC::PRINTF("[rank=%d AIV0 bk=%d] #%d w=%d nT=%d\n",
                            rankId, blockIdx, dbgSeq, dbgW, dbgN);
        }
        #endif

        auto taskProblemShape = problemShape;
        Get<M_VALUE>(taskProblemShape) = static_cast<int64_t>(task.waveM);
        int64_t waveRowOffset = static_cast<int64_t>(task.waveIdx) * static_cast<int64_t>(L1_TILE_M);

        GMMAddrInfo gmmAddrInfo;
        gmmAddrInfo.aGlobal = params.workspaceInfo.ptrA +
            (Get<IDX_A_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(k_full)) *
            sizeof(ElementA);
        gmmAddrInfo.aScaleGlobal = params.workspaceInfo.ptrAScale +
            (Get<IDX_A_SCALE_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(scaleK_gmm1)) *
            sizeof(ElementMxScaleA);
        gmmAddrInfo.bGlobal = params.bGmAddr +
            Get<IDX_B_OFFSET>(baseOffset) * sizeof(ElementB);
        gmmAddrInfo.bScaleGlobal = params.bScaleGmAddr +
            Get<IDX_B_SCALE_OFFSET>(baseOffset) * sizeof(ElementMxScaleB);
        gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params.workspaceInfo.ptrFlagSwiGluToGmm2 +
            flagOffsetForExpert + static_cast<int64_t>(task.waveIdx);
        gmmAddrInfo.groupFlagList2 = (__gm__ int32_t*)params.workspaceInfo.ptrFlagDispatchToGmm1 + expertIdx;
        gmmAddrInfo.gmm2OutGlobal = nullptr;

        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            Get<IDX_C_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(outputN),
            Get<IDX_C_SCALE_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(scaleOutputN),
            Get<IDX_FLAG_OFFSET>(baseOffset) + static_cast<int64_t>(task.waveIdx),
            0L, 0L, 0L};
        epilogueOp.UpdateGlobalAddr(vecBaseOffset);

        int32_t funcSync = 0;
        GroupMatmulSwigluQuant<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsInterleaved>(
            epilogueOp, params, taskProblemShape, gmmAddrInfo,
            static_cast<uint32_t>(task.nTileIdx), tileNumGmm1,
            funcSync, 0);
    }
    #if MEGA_MOE_DYN_DEBUG
    if (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B) {
        AscendC::PRINTF("[rank=%d AIV0 bk=%d] <<< EXIT (tasks=%d)\n", rankId, blockIdx, taskSeq);
    }
    #endif
}

// ============================================================================
// DynamicAiv1Loop — AIV1 worker, GMM2 combine 端
// ============================================================================
// 协议: spin AIV slot → read & clear → 调静态 GroupMatmul2 (AIV 分支:
//   spin rowReadyFlags ≥ 1 → atomic_sub → CombineTile)。END 跳过函数。
// ============================================================================
template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsInterleaved>
__aicore__ inline void DynamicAiv1Loop(
    const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& baseOffset,
    int32_t blockIdx, int32_t rankId, uint32_t expertIdx, uint32_t tileNumGmm2)
{
    #if MEGA_MOE_DYN_DEBUG
    if (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B) {
        AscendC::PRINTF("[rank=%d AIV1 bk=%d] >>> ENTRY exp=%d\n", rankId, blockIdx, expertIdx);
    }
    #endif
    int64_t flagOffsetForExpert = Get<IDX_FLAG_OFFSET>(baseOffset);
    uint64_t k_full = static_cast<uint64_t>(Get<K_VALUE>(problemShape));
    uint64_t n_full = static_cast<uint64_t>(Get<N_VALUE>(problemShape));
    uint64_t outputN = n_full / SWIGLU_N_HALF;
    uint64_t scaleOutputN = Ops::Base::CeilDiv(outputN, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                             MXFP_MULTI_BASE_SIZE;

    __gm__ int32_t* slotPtr = (__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot
                                + (blockIdx * 4 + DYN_CHANNEL_GMM2 * 2 + DYN_SLOT_AIV);

    int taskSeq = 0;
    while (true) {
        int32_t packed = 0;
        int64_t spinStart = AscendC::GetSystemCycle();
        bool timedOut = false;
        while (true) {
            packed = AscendC::ReadGmByPassDCache(slotPtr);
            if (packed != 0) break;
            if (AscendC::GetSystemCycle() - spinStart > SPIN_TIMEOUT_CYCLES * 40) {
                int dbgSeq = taskSeq + 1;
                AscendC::PRINTF("[rank=%d AIV1 bk=%d] *** TIMEOUT spin slot exp=%u seq=%d\n",
                                rankId, blockIdx, expertIdx, dbgSeq);
                timedOut = true;
                break;
            }
            int64_t ss = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - ss < 50) {}
        }
        if (timedOut) break;

        DynamicTask task;
        UnpackDynamicTask(packed, task);
        ClearTaskSlot(params, blockIdx, DYN_CHANNEL_GMM2, DYN_SLOT_AIV, packed);

        if (task.opType == DYN_OP_END) {
            break;
        }
        taskSeq++;
        #if MEGA_MOE_DYN_DEBUG
        if (taskSeq <= 3 && (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B)) {
            int32_t dbgW = task.waveIdx;
            int32_t dbgN = task.nTileIdx;
            int dbgSeq = taskSeq;
            AscendC::PRINTF("[rank=%d AIV1 bk=%d] #%d w=%d nT=%d\n",
                            rankId, blockIdx, dbgSeq, dbgW, dbgN);
        }
        #endif

        auto taskProblemShape = problemShape;
        Get<M_VALUE>(taskProblemShape) = static_cast<int64_t>(task.waveM);
        int64_t waveRowOffset = static_cast<int64_t>(task.waveIdx) * static_cast<int64_t>(L1_TILE_M);

        GMMAddrInfo gmmAddrInfo;
        gmmAddrInfo.aGlobal = params.workspaceInfo.ptrA2 +
            (Get<IDX_C_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(outputN)) *
            sizeof(ElementA);
        gmmAddrInfo.aScaleGlobal = params.workspaceInfo.ptrA2Scale +
            (Get<IDX_C_SCALE_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(scaleOutputN)) *
            sizeof(ElementMxScaleA);
        gmmAddrInfo.bGlobal = params.b2GmAddr +
            Get<IDX_B2_OFFSET>(baseOffset) * sizeof(ElementB);
        gmmAddrInfo.bScaleGlobal = params.b2ScaleGmAddr +
            Get<IDX_B2_SCALE_OFFSET>(baseOffset) * sizeof(ElementMxScaleB);
        gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params.workspaceInfo.ptrFlagSwiGluToGmm2 +
            flagOffsetForExpert + static_cast<int64_t>(task.waveIdx);
        gmmAddrInfo.groupFlagList2 = nullptr;
        gmmAddrInfo.gmm2OutGlobal = params.workspaceInfo.ptrGmmOut +
            (Get<IDX_Y2_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(k_full)) *
            sizeof(ElementC);

        GroupMatmul2<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsInterleaved>(
            params, taskProblemShape, gmmAddrInfo,
            static_cast<uint32_t>(task.nTileIdx), tileNumGmm2,
            expertIdx, rankId, static_cast<uint64_t>(waveRowOffset), 0);
    }
    #if MEGA_MOE_DYN_DEBUG
    if (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B) {
        AscendC::PRINTF("[rank=%d AIV1 bk=%d] <<< EXIT (tasks=%d)\n", rankId, blockIdx, taskSeq);
    }
    #endif
}

} // namespace MegaMoeImpl

#endif