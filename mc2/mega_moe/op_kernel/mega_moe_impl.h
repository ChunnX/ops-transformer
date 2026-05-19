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
// DYN_SCHED_DESIGN — 去中心化原子认领 (Decentralized Atomic Work-Stealing)
// ============================================================================
//
// 解决问题：mid < hidden 不对称 case 下（如 tileNumGmm1=12, tileNumGmm2=28），静态
// cross-wave 在 Phase 1 让 4 个核闲着不能做 Phase 2 工作。动态调度让 32 个核全部
// 参与 task 认领，跨 group 跨 phase 自由分配，能省 1 round 关键路径。
//
// --- 任务空间 ---
//
// 每个 expert 内，task 分两类：
//   GMM1[w][n] : (waveIdx=w, nTileIdx=n)  n ∈ [0, tileNumGmm1)
//   GMM2[w][n] : (waveIdx=w, nTileIdx=n)  n ∈ [0, tileNumGmm2)
//
// 依赖：GMM2[w][*] 必须等 groupFlagListGm[w] == tileNumGmm1
//      (该 wave 全部 GMM1+epi 完成后由 AIV0 在 epi 内 atomic_add 累加触发)
//
// --- 认领协议 ---
//
// per (expert, wave) HBM 上各两个 int32 原子计数器：
//   gmm1NextTile[e * maxW + w]  AtomicAdd(_, 1) → 老值 = 本核拿到的 nTileIdx
//   gmm2NextTile[e * maxW + w]  同上
// 每 expert 起始 AIC 0 重置 0，SyncAll<false>() 让所有核见到。
//
// AIC 每核独立主循环（无 group 约束，所有 wave 都能抢）：
//   loop:
//     // 优先级 1: GMM2 已解锁的 wave
//     for w in 0..waveCnt-1:
//       if ReadGmByPassDCache(groupFlagListGm[w]) == tileNumGmm1:
//         t = AtomicAdd(gmm2NextTile[e][w], 1)
//         if t < tileNumGmm2: dispatch_gmm2(w, t); continue loop
//     // 优先级 2: GMM1
//     for w in 0..waveCnt-1:
//       t = AtomicAdd(gmm1NextTile[e][w], 1)
//       if t < tileNumGmm1: dispatch_gmm1(w, t); continue loop
//     break  // 本核所有 task 完成
//
// --- AIC ↔ AIV 配对：双 channel pingpong ---
//
// blockIdx 一致的 AIC + AIV0 + AIV1 仍是 1:1 pair (硬件约束)。
// AIC 抢到 task 后，通过 HBM 上的 taskSlot 把任务信息传给配对的 AIV：
//   taskSlot[blockIdx][channel=0=GMM1][pp=0..1]  ← AIC 写, AIV0 读
//   taskSlot[blockIdx][channel=1=GMM2][pp=0..1]  ← AIC 写, AIV1 读
//
// 每 channel 2 个 pingpong slot，让 AIC 可以提前一个 task 而不被 AIV 阻塞。
//
// Flag id 映射（参 mega_moe_base.h L91-L97）:
//   GMM1 channel : AIC NotifyVector(pp)     ↔ AIV0 WaitForCube(pp)
//                  AIV0 NotifyCube(pp)       ↔ AIC WaitForVector(pp)
//   GMM2 channel : AIC NotifyVectorAIV1(pp) ↔ AIV1 WaitForCube(pp)
//                  AIV1 NotifyCube(pp)       ↔ AIC WaitForVectorFromAIV1(pp)
// 物理上 GMM1 用 flag id {0,1}, GMM2 用 flag id {16,17}, 两组隔离。
//
// AIC 写 slot 流程：
//   if used[pp]: WaitForCube(pp) 或 WaitForVectorFromAIV1(pp)
//   WriteTaskSlot(...)
//   SyncFunc<MTE3_MTE2>()  // 保证 task 内容落 GM 后再发 NotifyVector
//   NotifyVector(pp) / NotifyVectorAIV1(pp)
//   used[pp] = true
//   pp ^= 1
//   do mmad of the task (与 NotifyVector 在 PIPE_FIX 上串行，自带 fixpipe drain)
//
// AIV 处理 slot 流程：
//   WaitForCube(pp)         // AIV0 / AIV1 都用 WaitForCube
//   ReadTaskSlot(...)
//   if task.opType == END: NotifyCube(pp); break
//   do_work(task)            // sw+quant 或 combine
//   NotifyCube(pp)
//   pp ^= 1
//
// --- END 终止 ---
//
// AIC 抢不到任何 task 后，分别给两个 channel 写 op=END 的 task → NotifyVector。
// AIV 见 END 就 NotifyCube 一次（让 AIC drain）然后退出循环。
// AIC 最后 drain 所有未消费的 NotifyCube (两 channel 各 1-2 个，按 pingpong FIFO 顺序)。
//
// --- 跨 expert ---
//
// 每 expert 结束时 AIC drain 完所有 pending NotifyCube → used[*]=false, pp reset 0。
// 下一 expert 起始 AIC 0 重置 counter，SyncAll → 干净开始。
//
// --- 对称 case 影响 ---
//
// 对称 (tileNumGmm1 == tileNumGmm2 == coresPerWave_equiv) 时，每核仍抢 5 个 task，
// 理论 wall 等于静态 cross-wave。多出的 overhead：
//   - 每 task ~300 cycle (atomic add + WriteTaskSlot + SyncFunc + NotifyVector)
//   - 全 kernel 5 task × 32 核 / 关键路径上的 5 task = ~5 × 300 = 1500 cycle ≈ 1us
//   - HBM atomic counter 争抢：32 核同时 add 同 1 个 counter 大约 50ns 序列化 × 16 次 ≈ 1us
// 总 overhead < 5us per kernel，相对 ~500us 量级 < 1%。
//
// ============================================================================

// ----------------------------------------------------------------------------
// 动态调度 helper：原子认领下一个 task
// ----------------------------------------------------------------------------
// 优先认领 GMM2 已解锁的 wave，再认领 GMM1。返回 false 表示无 task。
__aicore__ inline bool ClaimNextTask(
    const Params& params,
    uint32_t expertIdx, uint32_t maxWavesPerExpert,
    uint32_t tileNumGmm1, uint32_t tileNumGmm2,
    uint64_t waveCnt, uint64_t m_i,
    int64_t flagOffsetForExpert,  // = IDX_FLAG_OFFSET of this expert (来自 baseOffset_)
    DynamicTask& outTask)
{
    // Priority 1: GMM2 of unlocked waves
    for (uint64_t w = 0; w < waveCnt; ++w) {
        __gm__ int32_t* flagAddr = (__gm__ int32_t*)params.workspaceInfo.ptrFlagSwiGluToGmm2 +
                                    flagOffsetForExpert + static_cast<int64_t>(w);
        int32_t flagVal = AscendC::ReadGmByPassDCache(flagAddr);
        if (flagVal != static_cast<int32_t>(tileNumGmm1)) {
            continue;
        }
        __gm__ int32_t* counterAddr =
            (__gm__ int32_t*)params.workspaceInfo.ptrGmm2NextTile +
            expertIdx * maxWavesPerExpert + w;
        int32_t tile = AscendC::AtomicAdd(counterAddr, 1);  // fetch_and_add: 返回老值
        if (tile < static_cast<int32_t>(tileNumGmm2)) {
            outTask.opType    = DYN_OP_GMM2;
            outTask.waveIdx   = static_cast<int32_t>(w);
            outTask.nTileIdx  = tile;
            outTask.waveM     = static_cast<int32_t>((w == waveCnt - 1 && m_i % L1_TILE_M != 0) ?
                                                     (m_i % L1_TILE_M) : L1_TILE_M);
            return true;
        }
    }
    // Priority 2: GMM1
    for (uint64_t w = 0; w < waveCnt; ++w) {
        __gm__ int32_t* counterAddr =
            (__gm__ int32_t*)params.workspaceInfo.ptrGmm1NextTile +
            expertIdx * maxWavesPerExpert + w;
        int32_t tile = AscendC::AtomicAdd(counterAddr, 1);
        if (tile < static_cast<int32_t>(tileNumGmm1)) {
            outTask.opType    = DYN_OP_GMM1;
            outTask.waveIdx   = static_cast<int32_t>(w);
            outTask.nTileIdx  = tile;
            outTask.waveM     = static_cast<int32_t>((w == waveCnt - 1 && m_i % L1_TILE_M != 0) ?
                                                     (m_i % L1_TILE_M) : L1_TILE_M);
            return true;
        }
    }
    return false;
}

// ----------------------------------------------------------------------------
// taskSlot 读写: scalar 8 个 int32
// ----------------------------------------------------------------------------
// taskSlot offset = (blockIdx * 4 + channel * 2 + pp) * 32 byte = ... * 8 int32
__aicore__ inline void WriteTaskSlot(
    const Params& params, int32_t blockIdx, int32_t channel,
    int32_t pp, const DynamicTask& task)
{
    int32_t slotIdx = blockIdx * 4 + channel * 2 + pp;
    __gm__ int32_t* p = ((__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot) + slotIdx * DYN_TASK_INTS;
    p[0] = task.opType;
    p[1] = task.waveIdx;
    p[2] = task.nTileIdx;
    p[3] = task.waveM;
    p[4] = task.reserved[0];
    p[5] = task.reserved[1];
    p[6] = task.reserved[2];
    p[7] = task.reserved[3];
}

__aicore__ inline void ReadTaskSlot(
    const Params& params, int32_t blockIdx, int32_t channel,
    int32_t pp, DynamicTask& outTask)
{
    int32_t slotIdx = blockIdx * 4 + channel * 2 + pp;
    __gm__ int32_t* p = ((__gm__ int32_t*)params.workspaceInfo.ptrTaskSlot) + slotIdx * DYN_TASK_INTS;
    outTask.opType      = AscendC::ReadGmByPassDCache(p + 0);
    outTask.waveIdx     = AscendC::ReadGmByPassDCache(p + 1);
    outTask.nTileIdx    = AscendC::ReadGmByPassDCache(p + 2);
    outTask.waveM       = AscendC::ReadGmByPassDCache(p + 3);
    outTask.reserved[0] = AscendC::ReadGmByPassDCache(p + 4);
    outTask.reserved[1] = AscendC::ReadGmByPassDCache(p + 5);
    outTask.reserved[2] = AscendC::ReadGmByPassDCache(p + 6);
    outTask.reserved[3] = AscendC::ReadGmByPassDCache(p + 7);
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
                while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
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
                while (targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
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
            while (ReadGmByPassDCache((__gm__ int32_t*)rowReadyFlags.GetPhyAddr() + blockIdx) < 1) {
                int64_t st = GetSystemCycle();
                while (GetSystemCycle() - st < 100) {}
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
// DynamicAicLoop — AIC 端动态调度主循环
// ============================================================================
// 每个 AIC 独立运行；每次 ClaimNextTask 拿一个 task → 写 slot + NotifyVector → mmad。
// 内部建立两套 BlockMmad/Scheduler：一套给 GMM1（输出 float 到 UB），一套给 GMM2
// （输出 bfloat16 到 GM）。每 task 用 task.nTileIdx 反算 mLoc/nLoc/kLoc。
//
// 参数：
//   params       : 全局参数（含 workspaceInfo）
//   problemShape : (m_i, n_full, k_hidden, 0)，m=expert 总 token 数；GMM1 用 (m, n, k)，
//                  GMM2 用 (m, k_hidden, n=outputN)，n/k 维度通过参数 problemShape 推
//   baseOffset   : 当前 expert 的偏移（IDX_A_OFFSET, IDX_B_OFFSET, ..., IDX_FLAG_OFFSET）
//   blockIdx     : 本 AIC 的全局 blockIdx (GetBlockIdx() / GetTaskRation())
//   expertIdx, maxWavesPerExpert: 用于在 counter 数组里定位
//   waveCnt, m_i : 用于 wave 边界判断
//   tileNumGmm1, tileNumGmm2 : 任务上限
//
// 同步：
//   GMM1 task: AIC NotifyVector(pp)     → AIV0 WaitForCube(pp)
//   GMM2 task: AIC NotifyVectorAIV1(pp) → AIV1 WaitForCube(pp)
//   每 task 写 slot 后 SyncFunc<MTE3_MTE2>() 保证 task 内容先于 Notify 落 GM
// ----------------------------------------------------------------------------
template <typename ElementA, typename ElementB,
          typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsInterleaved>
__aicore__ inline void DynamicAicLoop(
    const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& baseOffset,
    int32_t blockIdx,
    uint32_t expertIdx, uint32_t maxWavesPerExpert,
    uint64_t waveCnt, uint64_t m_i,
    uint32_t tileNumGmm1, uint32_t tileNumGmm2)
{
    using QuantOutType = ElementA;  // = fp8_e5m2 / fp8_e4m3
    using ElementC_Gmm1 = float;
    using ElementC_Gmm2 = bfloat16_t;

    // ============================================================
    // 阶段 0: 等本 expert 的 dispatch flag (only once per expert)
    // ============================================================
    __gm__ int32_t* dispatchFlagAddr = (__gm__ int32_t*)params.workspaceInfo.ptrFlagDispatchToGmm1 + expertIdx;
    {
        uint32_t targetValue = params.tilingData->epWorldSize * params.tilingData->blockNumPerEP;
        while (targetValue != AscendC::ReadGmByPassDCache(dispatchFlagAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {}
        }
    }

    int64_t flagOffsetForExpert = Get<IDX_FLAG_OFFSET>(baseOffset);

    // ============================================================
    // 阶段 1: 共用 layout / 模板别名 (GMM1 / GMM2 都用)
    // ============================================================
    uint32_t n_full  = static_cast<uint32_t>(Get<N_VALUE>(problemShape));  // GMM1 mmad N (= 2*mid 交织 / mid 非交织)
    uint32_t k_full  = static_cast<uint32_t>(Get<K_VALUE>(problemShape));  // hidden
    uint32_t outputN = n_full / SWIGLU_N_HALF;
    uint32_t m_full  = static_cast<uint32_t>(m_i);                          // 本 expert 总 token 数

    auto scaleK_gmm1 = CeilDiv(k_full, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    auto scaleK_gmm2 = CeilDiv(outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

    constexpr bool transA = false;
    constexpr bool transB = true;
    constexpr uint32_t c0SizeA = AuxGetC0Size<ElementA>();
    constexpr uint32_t c0SizeB = AuxGetC0Size<ElementB>();
    constexpr uint32_t c0SizeCGmm1 = AuxGetC0Size<ElementC_Gmm1>();
    constexpr uint32_t c0SizeCGmm2 = AuxGetC0Size<ElementC_Gmm2>();
    constexpr uint32_t c0SizeScale = 2U;

    using LayoutA = Std::conditional_t<transA, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutB = Std::conditional_t<transB, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutBias = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;
    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<c0SizeA>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<c0SizeB>>;
    using MakeLayoutScaleA = Std::conditional_t<transA,
        Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutScaleB = Std::conditional_t<transB,
        Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutCGmm1 = Te::FrameLayoutFormat<LayoutC, Std::Int<c0SizeCGmm1>>;
    using MakeLayoutCGmm2 = Te::FrameLayoutFormat<LayoutC, Std::Int<c0SizeCGmm2>>;

    using BiasType = float;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;

    // ============================================================
    // 阶段 2a: GMM1 BlockMmad / scheduler / GM tensor 全套
    //   layout 用 expert 级 (m_full × k_full) 视图，per-task Slice。
    //   输出到 UB l0cOutUbFirst (与配对 AIV0 共享, 同 GroupMatmulSwigluQuant L290 视图)
    // ============================================================
    auto layoutA_gmm1      = MakeLayoutA{}(m_full, k_full);
    auto layoutB_gmm1      = MakeLayoutB{}(k_full, n_full);
    auto layoutScaleA_gmm1 = MakeLayoutScaleA{}(m_full, scaleK_gmm1);
    auto layoutScaleB_gmm1 = MakeLayoutScaleB{}(scaleK_gmm1, n_full);
    auto layoutBias_gmm1   = Te::MakeFrameLayout<LayoutBias>(1u, n_full);
    auto layoutC_gmm1      = MakeLayoutCGmm1{}(L1_TILE_M, L1_TILE_N);   // tile 视图 (UB)

    using BlockMmadGmm1 = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, ElementA, LayoutA, ElementB, LayoutB, ElementC_Gmm1, LayoutC, BiasType, LayoutBias, void>;
    BlockMmadGmm1 blockMmadGmm1;
    bool enableL0CPingPong_Gmm1 = false;
    typename BlockMmadGmm1::L1Params l1Params_Gmm1{
        .kL1 = L1_TILE_K, .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE, .l1BufNum = 2
    };
    typename BlockMmadGmm1::BlockShape l0TileShape_Gmm1{L1_TILE_M, L1_TILE_N, L0_TILE_K, 0};
    typename BlockMmadGmm1::TupleShape matmulShape_Gmm1{m_full, n_full, k_full};
    blockMmadGmm1.Init(matmulShape_Gmm1, l0TileShape_Gmm1, l1Params_Gmm1, false, enableL0CPingPong_Gmm1);

    // UB 输出区 (与 AIV0 epi 共享，AIV0 端有相同 layout 的 l0cOutUbFirst)
    constexpr bool kGmm1Pingpong = IsInterleaved &&
        (MAX_SINGLE_MN_ALIGN32_NUM * sizeof(ElementC_Gmm1) * 2U <= UB_CAPACITY_BYTES);
    int64_t ubOffset = 0;
    auto l0cOutUbFirst = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::UB, ElementC_Gmm1>(ubOffset), layoutC_gmm1);
    int64_t ubSecondOffsetBytes = kGmm1Pingpong
        ? static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM * sizeof(ElementC_Gmm1))
        : static_cast<int64_t>(0);
    auto l0cOutUbSecond = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::UB, ElementC_Gmm1>(ubSecondOffsetBytes), layoutC_gmm1);

    // GMM1 GM 基址 (expert 级，per-task 用 Slice 加 waveRowOffset)
    GM_ADDR gmm1Abase = params.workspaceInfo.ptrA +
        Get<IDX_A_OFFSET>(baseOffset) * sizeof(QuantOutType);
    GM_ADDR gmm1AScaleBase = params.workspaceInfo.ptrAScale +
        Get<IDX_A_SCALE_OFFSET>(baseOffset) * sizeof(ElementMxScaleA);
    GM_ADDR gmm1Bbase = params.bGmAddr + Get<IDX_B_OFFSET>(baseOffset) * sizeof(QuantOutType);
    GM_ADDR gmm1BScaleBase = params.bScaleGmAddr +
        Get<IDX_B_SCALE_OFFSET>(baseOffset) * sizeof(ElementMxScaleB);

    auto gmA_gmm1 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA*>(gmm1Abase)), layoutA_gmm1);
    auto gmB_gmm1 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB*>(gmm1Bbase)), layoutB_gmm1);
    auto gmScaleA_gmm1 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA*>(gmm1AScaleBase)),
        layoutScaleA_gmm1);
    auto gmScaleB_gmm1 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB*>(gmm1BScaleBase)),
        layoutScaleB_gmm1);
    auto gmBias_gmm1 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layoutBias_gmm1);

    uint32_t schedulerN_gmm1 = IsInterleaved ? n_full : outputN;
    BlockScheduler schedGmm1(
        {static_cast<int64_t>(L1_TILE_M), static_cast<int64_t>(schedulerN_gmm1), static_cast<int64_t>(k_full)},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M), static_cast<int64_t>(L1_TILE_N))});

    // ============================================================
    // 阶段 2b: GMM2 BlockMmad / scheduler / GM tensor 全套
    //   GMM2: m=m_full, k'=outputN, n'=k_full (hidden), 输出 bfloat16 到 GM
    // ============================================================
    uint32_t k_gmm2 = outputN;
    uint32_t n_gmm2 = k_full;

    auto layoutA_gmm2      = MakeLayoutA{}(m_full, k_gmm2);
    auto layoutB_gmm2      = MakeLayoutB{}(k_gmm2, n_gmm2);
    auto layoutScaleA_gmm2 = MakeLayoutScaleA{}(m_full, scaleK_gmm2);
    auto layoutScaleB_gmm2 = MakeLayoutScaleB{}(scaleK_gmm2, n_gmm2);
    auto layoutBias_gmm2   = Te::MakeFrameLayout<LayoutBias>(1U, n_gmm2);
    auto layoutC_gmm2      = MakeLayoutCGmm2{}(m_full, n_gmm2);  // 全 (m, n) 视图，per-task Slice

    using BlockMmadGmm2 = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, ElementA, LayoutA, ElementB, LayoutB, ElementC_Gmm2, LayoutC, BiasType, LayoutBias, void>;
    BlockMmadGmm2 blockMmadGmm2;
    bool enableL0CPingPong_Gmm2 = false;
    typename BlockMmadGmm2::L1Params l1Params_Gmm2{
        .kL1 = L1_TILE_K, .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE, .l1BufNum = 2
    };
    typename BlockMmadGmm2::BlockShape l0TileShape_Gmm2{L1_TILE_M, L1_TILE_N, L0_TILE_K, 0};
    typename BlockMmadGmm2::TupleShape matmulShape_Gmm2{m_full, n_gmm2, k_gmm2};
    blockMmadGmm2.Init(matmulShape_Gmm2, l0TileShape_Gmm2, l1Params_Gmm2, false, enableL0CPingPong_Gmm2);

    GM_ADDR gmm2Abase = params.workspaceInfo.ptrA2 +
        Get<IDX_C_OFFSET>(baseOffset) * sizeof(QuantOutType);
    GM_ADDR gmm2AScaleBase = params.workspaceInfo.ptrA2Scale +
        Get<IDX_C_SCALE_OFFSET>(baseOffset) * sizeof(ElementMxScaleA);
    GM_ADDR gmm2Bbase = params.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset) * sizeof(QuantOutType);
    GM_ADDR gmm2BScaleBase = params.b2ScaleGmAddr +
        Get<IDX_B2_SCALE_OFFSET>(baseOffset) * sizeof(ElementMxScaleB);
    GM_ADDR gmm2Cbase = params.workspaceInfo.ptrGmmOut +
        Get<IDX_Y2_OFFSET>(baseOffset) * sizeof(ElementC_Gmm2);

    auto gmA_gmm2 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA*>(gmm2Abase)), layoutA_gmm2);
    auto gmB_gmm2 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB*>(gmm2Bbase)), layoutB_gmm2);
    auto gmScaleA_gmm2 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA*>(gmm2AScaleBase)),
        layoutScaleA_gmm2);
    auto gmScaleB_gmm2 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB*>(gmm2BScaleBase)),
        layoutScaleB_gmm2);
    auto gmBias_gmm2 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layoutBias_gmm2);
    auto gmC_gmm2 = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC_Gmm2*>(gmm2Cbase)), layoutC_gmm2);

    BlockScheduler schedGmm2(
        {static_cast<int64_t>(L1_TILE_M), static_cast<int64_t>(n_gmm2), static_cast<int64_t>(k_gmm2)},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M), static_cast<int64_t>(L1_TILE_N))});

    // ============================================================
    // 阶段 3: pingpong 状态
    // ============================================================
    uint16_t gmm1Pp = 0, gmm2Pp = 0;
    bool gmm1Used[2] = {false, false};
    bool gmm2Used[2] = {false, false};

    // ============================================================
    // 阶段 4: 主循环 — claim → 写 slot → mmad → Notify
    //
    // 顺序设计:
    //   1. (若 slot 已用) Wait 配对 AIV NotifyCube 释放 slot
    //   2. WriteTaskSlot — 把 task 信息 (waveIdx, nTileIdx, waveM) 写到 GM slot
    //      用 scalar 写。这里发生在 mmad 之前，给写传播充足时间。
    //   3. blockMmad — 重活在 PIPE_FIX (~30us)。期间 scalar 写早已 commit。
    //   4. NotifyVector(AIV1) — PIPE_FIX，自带 fixpipe drain (mmad 完成 + GM 数据 commit)
    //      → 配对 AIV 端 WaitForCube 解锁，ReadGmByPassDCache 读到 fresh slot
    // ============================================================
    DynamicTask task;
    while (ClaimNextTask(params, expertIdx, maxWavesPerExpert,
                          tileNumGmm1, tileNumGmm2, waveCnt, m_i,
                          flagOffsetForExpert, task)) {
        int64_t waveRowOffset = static_cast<int64_t>(task.waveIdx) * static_cast<int64_t>(L1_TILE_M);

        if (task.opType == DYN_OP_GMM1) {
            // -- GMM1 task --
            auto blockCoord = schedGmm1.GetBlockCoord(static_cast<uint32_t>(task.nTileIdx));
            auto actualShape = schedGmm1.GetBlockShape(blockCoord);
            uint32_t mLoc = Get<M_VALUE>(blockCoord);  // 0 (schedGmm1 用 m=L1_TILE_M)
            uint32_t nLoc = Get<N_VALUE>(blockCoord);
            uint32_t kLoc = Get<K_VALUE>(blockCoord);  // 0 (BlockMmad 内部走 K)
            // ★ 关键：scheduler 用 m=L1_TILE_M 构建，actualShape.M 永远 = L1_TILE_M。
            //   末尾 partial wave 必须用 task.waveM 覆盖，否则 Slice 会越界到下一行/下一 expert。
            int64_t actualM = static_cast<int64_t>(task.waveM);
            int64_t actualN = static_cast<int64_t>(Get<N_VALUE>(actualShape));
            int64_t actualK = static_cast<int64_t>(Get<K_VALUE>(actualShape));

            // 1. 等 AIV0 释放上一次该 slot
            if (gmm1Used[gmm1Pp]) {
                WaitForVector(gmm1Pp);
            }
            // 2. 写 task slot (scalar)
            WriteTaskSlot(params, blockIdx, DYN_CHANNEL_GMM1, static_cast<int32_t>(gmm1Pp), task);

            // 3. mmad: A 走 expert 级地址 + 按 waveRowOffset Slice 行；B 是 expert 级权重，不分 wave
            int64_t rowStart = waveRowOffset + static_cast<int64_t>(mLoc);
            auto gmBlockA = gmA_gmm1.Slice(
                Te::MakeCoord(rowStart, static_cast<int64_t>(kLoc)),
                Te::MakeShape(actualM, actualK));
            auto gmBlockScaleA = gmScaleA_gmm1.Slice(
                Te::MakeCoord(rowStart, static_cast<int64_t>(kLoc) / MXFP_SCALE_GROUP_NUM),
                Te::MakeShape(actualM, CeilDiv(actualK, MXFP_SCALE_GROUP_NUM)));
            auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(actualM, actualN));

            typename BlockMmadGmm1::BlockShape singleShape{actualM, actualN, actualK, 0};
            if constexpr (IsInterleaved) {
                auto gmBlockB = gmB_gmm1.Slice(
                    Te::MakeCoord(static_cast<int64_t>(kLoc), static_cast<int64_t>(nLoc)),
                    Te::MakeShape(actualK, actualN));
                auto gmBlockScaleB = gmScaleB_gmm1.Slice(
                    Te::MakeCoord(static_cast<int64_t>(kLoc) / MXFP_SCALE_GROUP_NUM, static_cast<int64_t>(nLoc)),
                    Te::MakeShape(CeilDiv(actualK, MXFP_SCALE_GROUP_NUM), actualN));
                blockMmadGmm1(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias_gmm1,
                              tensorBlockUbFirst, singleShape);
            } else {
                auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
                    Te::MakeCoord(0, 0), Te::MakeShape(actualM, actualN));
                for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                    auto gmBlockB = gmB_gmm1.Slice(
                        Te::MakeCoord(static_cast<int64_t>(kLoc),
                                       static_cast<int64_t>(nLoc + weightBlock * outputN)),
                        Te::MakeShape(actualK, actualN));
                    auto gmBlockScaleB = gmScaleB_gmm1.Slice(
                        Te::MakeCoord(static_cast<int64_t>(kLoc) / MXFP_SCALE_GROUP_NUM,
                                       static_cast<int64_t>(nLoc + weightBlock * outputN)),
                        Te::MakeShape(CeilDiv(actualK, MXFP_SCALE_GROUP_NUM), actualN));
                    blockMmadGmm1(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias_gmm1,
                                  weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond,
                                  singleShape);
                }
            }

            // 4. NotifyVector(pp) — PIPE_FIX 自带 mmad 的 fixpipe drain → AIV0 解锁后读 slot + UB
            NotifyVector(gmm1Pp);
            gmm1Used[gmm1Pp] = true;
            gmm1Pp ^= 1;
        } else {  // DYN_OP_GMM2
            // -- GMM2 task --
            auto blockCoord = schedGmm2.GetBlockCoord(static_cast<uint32_t>(task.nTileIdx));
            auto actualShape = schedGmm2.GetBlockShape(blockCoord);
            uint32_t mLoc = Get<M_VALUE>(blockCoord);  // 0
            uint32_t nLoc = Get<N_VALUE>(blockCoord);
            uint32_t kLoc = Get<K_VALUE>(blockCoord);  // 0
            // ★ 同 GMM1：partial wave 末尾必须用 task.waveM 覆盖
            int64_t actualM = static_cast<int64_t>(task.waveM);
            int64_t actualN = static_cast<int64_t>(Get<N_VALUE>(actualShape));
            int64_t actualK = static_cast<int64_t>(Get<K_VALUE>(actualShape));

            if (gmm2Used[gmm2Pp]) {
                WaitForVectorFromAIV1(gmm2Pp);
            }
            WriteTaskSlot(params, blockIdx, DYN_CHANNEL_GMM2, static_cast<int32_t>(gmm2Pp), task);

            // GMM2 mmad: A=ptrA2[wave], B=b2, C=ptrGmmOut[wave]
            int64_t rowStart = waveRowOffset + static_cast<int64_t>(mLoc);
            auto gmBlockA = gmA_gmm2.Slice(
                Te::MakeCoord(rowStart, static_cast<int64_t>(kLoc)),
                Te::MakeShape(actualM, actualK));
            auto gmBlockScaleA = gmScaleA_gmm2.Slice(
                Te::MakeCoord(rowStart, static_cast<int64_t>(kLoc) / MXFP_SCALE_GROUP_NUM),
                Te::MakeShape(actualM, CeilDiv(actualK, MXFP_SCALE_GROUP_NUM)));
            auto gmBlockB = gmB_gmm2.Slice(
                Te::MakeCoord(static_cast<int64_t>(kLoc), static_cast<int64_t>(nLoc)),
                Te::MakeShape(actualK, actualN));
            auto gmBlockScaleB = gmScaleB_gmm2.Slice(
                Te::MakeCoord(static_cast<int64_t>(kLoc) / MXFP_SCALE_GROUP_NUM, static_cast<int64_t>(nLoc)),
                Te::MakeShape(CeilDiv(actualK, MXFP_SCALE_GROUP_NUM), actualN));
            auto gmBlockC = gmC_gmm2.Slice(
                Te::MakeCoord(rowStart, static_cast<int64_t>(nLoc)),
                Te::MakeShape(actualM, actualN));

            typename BlockMmadGmm2::BlockShape singleShape{actualM, actualN, actualK, 0};
            blockMmadGmm2(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias_gmm2,
                          gmBlockC, singleShape);

            // NotifyVectorAIV1 (PIPE_FIX) drain mmad 的 fixpipe → AIV1 解锁后读 GM combine
            NotifyVectorAIV1(gmm2Pp);
            gmm2Used[gmm2Pp] = true;
            gmm2Pp ^= 1;
        }
    }

    // ============================================================
    // 阶段 4: 发 END 到两个 channel + drain pending NotifyCube
    // ============================================================
    DynamicTask endTask{};
    endTask.opType = DYN_OP_END;

    if (gmm1Used[gmm1Pp]) {
        WaitForVector(gmm1Pp);
    }
    WriteTaskSlot(params, blockIdx, DYN_CHANNEL_GMM1, static_cast<int32_t>(gmm1Pp), endTask);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    NotifyVector(gmm1Pp);
    gmm1Used[gmm1Pp] = true;
    gmm1Pp ^= 1;

    if (gmm2Used[gmm2Pp]) {
        WaitForVectorFromAIV1(gmm2Pp);
    }
    WriteTaskSlot(params, blockIdx, DYN_CHANNEL_GMM2, static_cast<int32_t>(gmm2Pp), endTask);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    NotifyVectorAIV1(gmm2Pp);
    gmm2Used[gmm2Pp] = true;
    gmm2Pp ^= 1;

    // Drain 剩余 NotifyCube：两个 channel 各最多 2 个 (含 END)
    // pingpong FIFO 顺序：上一发先回，END 后回
    if (gmm1Used[gmm1Pp]) {
        WaitForVector(gmm1Pp);
    }
    gmm1Pp ^= 1;
    if (gmm1Used[gmm1Pp]) {
        WaitForVector(gmm1Pp);
    }
    if (gmm2Used[gmm2Pp]) {
        WaitForVectorFromAIV1(gmm2Pp);
    }
    gmm2Pp ^= 1;
    if (gmm2Used[gmm2Pp]) {
        WaitForVectorFromAIV1(gmm2Pp);
    }
}

// ============================================================================
// DynamicAiv0Loop — AIV0 (subBlock 0) 端 GMM1 epi (sw+quant) 主循环
// ============================================================================
// 协议: WaitForCube(pp) → ReadTaskSlot → if END break → 调 epi → NotifyCube(pp) → pp^=1
// 复用 epilogueOp_ 对象（caller 传入），但每 task 入口必须用 task.waveIdx 重设 GM 地址
// (UpdateGlobalAddr)，否则 epi 内部用的还是上一 task 的 wave 偏移。
template <typename ElementA, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsInterleaved>
__aicore__ inline void DynamicAiv0Loop(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true, IsInterleaved>& epilogueOp,
    const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& baseOffset,
    int32_t blockIdx)
{
    uint32_t n_full = static_cast<uint32_t>(Get<N_VALUE>(problemShape));
    uint32_t k_full = static_cast<uint32_t>(Get<K_VALUE>(problemShape));
    uint32_t outputN = n_full / SWIGLU_N_HALF;
    auto scaleOutputN = CeilDiv(outputN, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    // 用 GMM1 scheduler 反算 blockCoord
    uint32_t schedulerN_gmm1 = IsInterleaved ? n_full : outputN;
    BlockScheduler schedGmm1(
        {static_cast<int64_t>(L1_TILE_M), static_cast<int64_t>(schedulerN_gmm1), static_cast<int64_t>(k_full)},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M), static_cast<int64_t>(L1_TILE_N))});

    uint16_t pp = 0;
    DynamicTask task;
    while (true) {
        WaitForCube(pp);
        ReadTaskSlot(params, blockIdx, DYN_CHANNEL_GMM1, static_cast<int32_t>(pp), task);
        if (task.opType == DYN_OP_END) {
            NotifyCube(pp);
            break;
        }

        // ★ 每 task 用 task.waveM 重新设置 epi 的 problem M (处理末尾 partial wave)
        epilogueOp.UpdateNextProblem({static_cast<uint32_t>(task.waveM), outputN, k_full, 0});

        // 用 task.waveIdx 重设 epi 内部 GM 指针：ptrA2, ptrA2Scale, groupFlagListGm
        int64_t waveRowOffset = static_cast<int64_t>(task.waveIdx) * static_cast<int64_t>(L1_TILE_M);
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            Get<IDX_C_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(outputN),
            Get<IDX_C_SCALE_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(scaleOutputN),
            Get<IDX_FLAG_OFFSET>(baseOffset) + static_cast<int64_t>(task.waveIdx),
            0L, 0L, 0L};
        epilogueOp.UpdateGlobalAddr(vecBaseOffset);

        // 反算 mLoc/nLoc
        auto blockCoord = schedGmm1.GetBlockCoord(static_cast<uint32_t>(task.nTileIdx));
        auto actualShape = schedGmm1.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        uint32_t swigluNLoc;
        int64_t swigluN_tile;
        if constexpr (IsInterleaved) {
            swigluNLoc = nLoc / SWIGLU_N_HALF;
            swigluN_tile = static_cast<int64_t>(Get<N_VALUE>(actualShape)) / SWIGLU_N_HALF;
        } else {
            swigluNLoc = nLoc;
            swigluN_tile = static_cast<int64_t>(Get<N_VALUE>(actualShape));
        }
        // ★ M 用 task.waveM (处理末尾 partial wave; 中间 wave 这两个值相等)
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{
            static_cast<int64_t>(task.waveM), swigluN_tile, 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            mLoc * outputN + swigluNLoc,
            mLoc * CeilDiv(outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(swigluNLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            0, mLoc, 0, 0};

        AscendC::SetCtrlSpr<60, 60>(0);
        if constexpr (IsInterleaved) {
            epilogueOp(epilogueShape, epilogueOffset, static_cast<uint16_t>(0));
        } else {
            epilogueOp(epilogueShape, epilogueOffset);
        }
        // epi 内部最后 atomic_add(groupFlagListGm[waveIdx])，触发 GMM2 解锁

        NotifyCube(pp);
        pp ^= 1;
    }
}

// ============================================================================
// DynamicAiv1Loop — AIV1 (subBlock 1) 端 GMM2 combine 主循环
// ============================================================================
template <typename ElementC>  // = bfloat16_t
__aicore__ inline void DynamicAiv1Loop(
    const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& baseOffset,
    int32_t blockIdx, int32_t rankId, uint32_t expertIdx)
{
    uint32_t n_full = static_cast<uint32_t>(Get<N_VALUE>(problemShape));
    uint32_t k_full = static_cast<uint32_t>(Get<K_VALUE>(problemShape));
    uint32_t outputN = n_full / SWIGLU_N_HALF;

    // 用 GMM2 scheduler 反算 blockCoord
    BlockScheduler schedGmm2(
        {static_cast<int64_t>(L1_TILE_M), static_cast<int64_t>(k_full), static_cast<int64_t>(outputN)},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M), static_cast<int64_t>(L1_TILE_N))});

    uint16_t pp = 0;
    DynamicTask task;
    while (true) {
        WaitForCube(pp);  // AIV1 端: wait AIC NotifyVectorAIV1 → 底层 flag id = pp (AIV1 端不+16)
        ReadTaskSlot(params, blockIdx, DYN_CHANNEL_GMM2, static_cast<int32_t>(pp), task);
        if (task.opType == DYN_OP_END) {
            NotifyCube(pp);
            break;
        }

        // 反算 mLoc/nLoc
        auto blockCoord = schedGmm2.GetBlockCoord(static_cast<uint32_t>(task.nTileIdx));
        auto actualShape = schedGmm2.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        // ★ 覆盖 actualShape.M 为 task.waveM (partial wave 处理)
        // CombineTile 内部按 actualShape.M 拷数据，必须是真实 waveM，不能是 L1_TILE_M
        Get<M_VALUE>(actualShape) = static_cast<int64_t>(task.waveM);

        int64_t waveRowOffset = static_cast<int64_t>(task.waveIdx) * static_cast<int64_t>(L1_TILE_M);
        GM_ADDR gmm2OutGlobal = params.workspaceInfo.ptrGmmOut +
            (Get<IDX_Y2_OFFSET>(baseOffset) + waveRowOffset * static_cast<int64_t>(k_full)) * sizeof(ElementC);

        int64_t ubOffset = 0;
        LocalTensor<ElementC> ubBuffer = LocalTensor<ElementC>(TPosition::VECIN, ubOffset,
                                                                L1_TILE_M * L1_TILE_N);
        MegaMoeCombineImpl::CombineTile<ElementC, decltype(actualShape)>(
            expertIdx, mLoc, nLoc, k_full, rankId,
            gmm2OutGlobal, params, ubBuffer, actualShape,
            static_cast<uint64_t>(waveRowOffset));

        NotifyCube(pp);
        pp ^= 1;
    }
}

} // namespace MegaMoeImpl

#endif