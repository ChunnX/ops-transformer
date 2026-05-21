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
#define MEGA_MOE_DYN_DEBUG 0
// 只看两组 (AIC + AIV0 + AIV1) 的 PRINTF, 32 核全打日志噪音太大且会爆缓冲。
// 调试别的核改这两个即可。
constexpr int32_t DBG_BK_A = 0;
constexpr int32_t DBG_BK_B = 25;  // 之前是 30, 改成 25 因为日志显示 bk=25 出现 -2 over-clear, 重点观察

// spin 超时阈值 (单位: cycle); 超过这个时间没等到就打日志 + break, 不再死等。
// 1GHz 时钟下 5_000_000 cycle ≈ 5ms — 远超正常 spin (~几 us), 比 50ms 触发快 10x, 更快暴露问题。
// 真正卡死的话 5ms 也足够远超正常进度差。
constexpr int64_t SPIN_TIMEOUT_CYCLES = 5000000LL;

// ============================================================================
// DYN_SCHED_DESIGN — 单写者协议 (Single-Writer Protocol)
// ============================================================================
//
// 核心: 每个 HBM int32 地址只有一个核会 store。物理上不可能 race。无 atomic。
//
// --- 内存布局 (workspace, 一次 kernel 调用开始时清零) ---
//
//   taskData[bk*2 + ch]              : scheduler 写, AIC/AIV 只读
//                                       packed task (PackDynamicTask 编码 32 bit)
//   taskSeq [bk*2 + ch]              : scheduler 写, AIC/AIV 只读
//                                       单调递增计数器, 表示本队列总共下发了多少 task
//   taskAck [bk*4 + ch*2 + owner]    : worker (自己 owner) 写, scheduler 只读
//                                       worker 已处理到哪个 seq
//   ch ∈ {0=GMM1, 1=GMM2}, owner ∈ {0=AIC, 1=AIV}
//
// --- 派发协议 (per (bk, ch)) ---
//
//   scheduler 本地 my_seq[bk*2+ch]: 下一个待派 task 的 seq
//   每 iter: scheduler 用 DataCopyPad 一次性把整片 ack 数组从 HBM DMA 到 UB
//            (~500ns vs 124 次单点 read ~12μs, 快 ~24x)
//   派发条件: ack[bk*4+ch*2+0] == my_seq[bk*2+ch] && ack[bk*4+ch*2+1] == my_seq[bk*2+ch]
//            (AIC 和 AIV 都已确认上个 task)
//   派发动作:
//     1. write taskData[bk*2+ch] = packed
//     2. PipeBarrier (保证 taskData 落盘先于 taskSeq)
//     3. my_seq[bk*2+ch] += 1; write taskSeq[bk*2+ch] = my_seq[bk*2+ch]
//     4. PipeBarrier
//   batch: 一 iter 可对多个 (bk, ch) 同时派, 共享 2 次 PipeBarrier
//
// --- worker 处理 (per (bk, ch, owner)) ---
//
//   last_seen = 0
//   loop:
//     cur = ReadGmByPassDCache(taskSeq[bk*2+ch])
//     if cur > last_seen:
//       packed = ReadGmByPassDCache(taskData[bk*2+ch])
//       task = unpack(packed)
//       if task.op == END:
//         write taskAck[bk*4+ch*2+owner] = cur
//         break
//       process(task)
//       write taskAck[bk*4+ch*2+owner] = cur
//       last_seen = cur
//
// --- 跨 expert ---
//
//   taskSeq / taskAck / my_seq 全部跨 expert 单调递增, 永不回退。
//   不需要 expert 边界 reset HBM, 不需要 SyncAll。kernel 调用开始时一次性 reset 即可。
//
// ============================================================================

constexpr int32_t OWNER_AIC = 0;
constexpr int32_t OWNER_AIV = 1;

__aicore__ inline __gm__ int32_t* GetTaskDataPtr(const Params& p, int32_t bk, int32_t ch)
{
    return (__gm__ int32_t*)p.workspaceInfo.ptrTaskData + (bk * 2 + ch);
}
__aicore__ inline __gm__ int32_t* GetTaskSeqPtr(const Params& p, int32_t bk, int32_t ch)
{
    return (__gm__ int32_t*)p.workspaceInfo.ptrTaskSeq + (bk * 2 + ch);
}
__aicore__ inline __gm__ int32_t* GetTaskAckPtr(const Params& p, int32_t bk, int32_t ch, int32_t owner)
{
    return (__gm__ int32_t*)p.workspaceInfo.ptrTaskAck + (bk * 4 + ch * 2 + owner);
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
                // PipeBarrier<PIPE_ALL>();
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
                // 同上, atomic_add scalar pipe 必须 drain 后才发 cube-vec flag
                // PipeBarrier<PIPE_ALL>();
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
            // ★ atomic_add scalar pipe drain, 否则 AIV1 spin rowReadyFlags 读不到
            // PipeBarrier<PIPE_ALL>();

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
            //
            // ★ 关键: 这里 *绝对不能* 在 timeout 时 break 然后 atomic_sub。
            //   动态调度下, AIV1 收到 G2 task 后立刻进入这里 spin, AIC 可能还在啃 G1 backlog
            //   没轮到 G2, 几 ms 后才 atomic_add(+1)。若 break 后 atomic_sub(-1), rowReadyFlags
            //   被错误减成 -1, 同 bk 后续所有 G2 task 永远 spin 不到 >=1 → 级联崩盘 (print8.log 现象)。
            //   只打一次 warning 然后继续 spin, 直到 AIC 确实 +1。
            {
                int64_t _spinStart = GetSystemCycle();
                bool _warned = false;
                while (ReadGmByPassDCache((__gm__ int32_t*)rowReadyFlags.GetPhyAddr() + blockIdx) < 1) {
                    if (!_warned && GetSystemCycle() - _spinStart > SPIN_TIMEOUT_CYCLES) {
                        int dbgBk = (int)blockIdx;
                        unsigned dbgLoop = (unsigned)loopIdx;
                        int32_t dbgCur = ReadGmByPassDCache((__gm__ int32_t*)rowReadyFlags.GetPhyAddr() + blockIdx);
                        AscendC::PRINTF("[static GMM2 AIV1 bk=%d] WARN long spin rowReadyFlags loopIdx=%u cur=%d (keep spinning)\n",
                                        dbgBk, dbgLoop, dbgCur);
                        _warned = true;
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
// 单写者协议(见文件顶部 DYN_SCHED_DESIGN)。每 iter:
//   1. DataCopyPad 一次性 DMA 整片 ack 数组到 UB (~500ns)
//   2. 在 UB 内 scalar 比较 ack vs my_seq, 标记 idle (bk, ch)
//   3. 决策派发: 找下个 G1 / G2 task 或发 END, 暂存到 dispatch buf
//   4. Phase A: 批量写 taskData (无 PipeBarrier 间隔)
//   5. 1 次 PipeBarrier
//   6. Phase B: 批量写 taskSeq, 更新 my_seq[]
//   7. 1 次 PipeBarrier
// 退出条件: 所有 (bk, ch) end_acked。
//
// my_seq 由调用方维护, 跨 expert 单调递增不复位。 配合 worker last_seen 也单调递增,
// 整个 kernel 调用期间无需任何 reset / SyncAll。
// ============================================================================
constexpr int32_t SCHED_MAX_WAVES   = 64;   // waveCnt 上限
constexpr int32_t SCHED_MAX_WORKERS = 64;   // blockNum 上限

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
    uint32_t tileNumGmm1, uint32_t tileNumGmm2,
    int32_t* my_seq)  // [bk*2 + ch], 跨 expert 持续累积
{
    #if MEGA_MOE_DYN_DEBUG
    AscendC::PRINTF("[rank=%d SCHED] >>> ENTRY exp=%u tnG1=%u tnG2=%u waveCnt=%lu m_i=%lu\n",
                    rankId, expertIdx, tileNumGmm1, tileNumGmm2, waveCnt, m_i);
    #endif

    // --- 1. 等本 expert 的 dispatch flag ---
    __gm__ int32_t* dispatchFlagAddr = (__gm__ int32_t*)params.workspaceInfo.ptrFlagDispatchToGmm1 + expertIdx;
    int32_t targetValue = static_cast<int32_t>(params.tilingData->epWorldSize * params.tilingData->blockNumPerEP);
    {
        int64_t st = AscendC::GetSystemCycle();
        while (AscendC::ReadGmByPassDCache(dispatchFlagAddr) != targetValue) {
            if (AscendC::GetSystemCycle() - st > SPIN_TIMEOUT_CYCLES) {
                int32_t dbgCur = AscendC::ReadGmByPassDCache(dispatchFlagAddr);
                AscendC::PRINTF("[rank=%d SCHED] TIMEOUT dispatch flag exp=%u tgt=%d cur=%d\n",
                                rankId, expertIdx, targetValue, dbgCur);
                break;
            }
            int64_t ss = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - ss < 500) {}  // 拉长 spin 间隔 减小 HBM 争用
        }
    }

    int64_t flagOffsetForExpert = Get<IDX_FLAG_OFFSET>(baseOffset);
    __gm__ int32_t* swigluFlagBase =
        (__gm__ int32_t*)params.workspaceInfo.ptrFlagSwiGluToGmm2 + flagOffsetForExpert;

    // --- 2. 本 expert 局部状态 ---
    int32_t dispatched_g1[SCHED_MAX_WAVES];
    int32_t dispatched_g2[SCHED_MAX_WAVES];
    for (int32_t i = 0; i < SCHED_MAX_WAVES; ++i) {
        dispatched_g1[i] = 0;
        dispatched_g2[i] = 0;
    }
    bool end_sent [SCHED_MAX_WORKERS * 2];   // [bk*2 + ch]
    bool end_acked[SCHED_MAX_WORKERS * 2];
    for (int32_t i = 0; i < SCHED_MAX_WORKERS * 2; ++i) {
        end_sent[i]  = false;
        end_acked[i] = false;
    }

    // wave_unlocked[w]: sticky-true 缓存。一旦 swiglu flag[w] >= tnG1 就置 true,
    // 之后 G2 派发不再 ReadGmByPassDCache(swigluFlagBase+w) → scheduler 单轮 HBM 读
    // 从最坏 ~124 次降到最坏 waveCnt 次, 解锁完后降到 0 次。
    bool wave_unlocked[SCHED_MAX_WAVES];
    for (int32_t i = 0; i < SCHED_MAX_WAVES; ++i) wave_unlocked[i] = false;

    int32_t lastWaveM = static_cast<int32_t>(
        (m_i % L1_TILE_M != 0) ? (m_i % L1_TILE_M) : L1_TILE_M);
    int32_t lastWaveIdx = static_cast<int32_t>(waveCnt - 1);

    // --- 3. UB tensor 用来一次性 DMA ack 数组 ---
    int32_t ackElems = static_cast<int32_t>(blockNum * 4);
    int32_t ackElemsAligned = (ackElems + 7) & ~7;
    LocalTensor<int32_t> ubAck = LocalTensor<int32_t>(TPosition::VECIN, 0, ackElemsAligned);
    GlobalTensor<int32_t> gmAck;
    gmAck.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrTaskAck);

    // --- 4. 主循环 ---
    int64_t loopStart = AscendC::GetSystemCycle();
    while (true) {
        // 4-pre. 一次性刷新所有还没解锁的 wave 的 swiglu flag(每轮最多 waveCnt 次 HBM 读,
        //        解锁完后 0 次)。原版每个 (bk, G2) 迭代都重读 → 单轮可到 124 次。
        for (uint64_t w = 0; w < waveCnt; ++w) {
            if (!wave_unlocked[w]) {
                int32_t f = AscendC::ReadGmByPassDCache(swigluFlagBase + w);
                if (f >= static_cast<int32_t>(tileNumGmm1)) wave_unlocked[w] = true;
            }
        }

        // 4a. DMA: HBM ack 数组 → UB
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(ackElems * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
        DataCopyPad(ubAck, gmAck, copyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        // 4b. 扫描 + 决策 + 写 taskData (Phase A)
        bool allDone = true;
        int32_t numDispatched = 0;
        int32_t dispatchIdx[SCHED_MAX_WORKERS * 2];
        int32_t dispatchSeq[SCHED_MAX_WORKERS * 2];

        for (uint32_t bk = 1; bk < blockNum; ++bk) {
            for (int32_t ch = 0; ch < 2; ++ch) {
                int32_t idx = static_cast<int32_t>(bk) * 2 + ch;
                if (end_acked[idx]) continue;
                allDone = false;

                int32_t aicAck = ubAck.GetValue(bk * 4 + ch * 2 + OWNER_AIC);
                int32_t aivAck = ubAck.GetValue(bk * 4 + ch * 2 + OWNER_AIV);
                if (aicAck != my_seq[idx] || aivAck != my_seq[idx]) continue;   // worker busy

                if (end_sent[idx]) {                                            // END 已发 + 已 ack
                    end_acked[idx] = true;
                    continue;
                }

                // 找下一个 task
                int32_t wDispatch = -1;
                if (ch == 0) {                                                  // G1
                    for (uint64_t w = 0; w < waveCnt; ++w) {
                        if (dispatched_g1[w] < static_cast<int32_t>(tileNumGmm1)) {
                            wDispatch = static_cast<int32_t>(w); break;
                        }
                    }
                } else {                                                        // G2
                    for (uint64_t w = 0; w < waveCnt; ++w) {
                        if (dispatched_g2[w] >= static_cast<int32_t>(tileNumGmm2)) continue;
                        if (!wave_unlocked[w]) continue;   // 用本地缓存, 不再 ReadGm
                        wDispatch = static_cast<int32_t>(w); break;
                    }
                }

                int32_t packed;
                bool isEnd = false;
                if (wDispatch >= 0) {
                    DynamicTask t;
                    t.opType  = (ch == 0) ? DYN_OP_GMM1 : DYN_OP_GMM2;
                    t.waveIdx = wDispatch;
                    t.nTileIdx = (ch == 0) ? dispatched_g1[wDispatch] : dispatched_g2[wDispatch];
                    t.waveM   = (wDispatch == lastWaveIdx) ? lastWaveM : static_cast<int32_t>(L1_TILE_M);
                    packed = PackDynamicTask(t);
                    if (ch == 0) dispatched_g1[wDispatch]++;
                    else         dispatched_g2[wDispatch]++;
                } else {
                    // 是否全派完?
                    bool allDispatched = true;
                    if (ch == 0) {
                        for (uint64_t w = 0; w < waveCnt; ++w) {
                            if (dispatched_g1[w] < static_cast<int32_t>(tileNumGmm1)) { allDispatched = false; break; }
                        }
                    } else {
                        for (uint64_t w = 0; w < waveCnt; ++w) {
                            if (dispatched_g2[w] < static_cast<int32_t>(tileNumGmm2)) { allDispatched = false; break; }
                        }
                    }
                    if (!allDispatched) continue;                               // 等 swiglu 解锁
                    DynamicTask endT;
                    endT.opType = DYN_OP_END; endT.waveIdx = 0; endT.nTileIdx = 0; endT.waveM = 0;
                    packed = PackDynamicTask(endT);
                    isEnd = true;
                }

                // Phase A: 写 taskData
                __gm__ int32_t* tdPtr = GetTaskDataPtr(params, static_cast<int32_t>(bk), ch);
                AscendC::WriteGmByPassDCache(tdPtr, packed);
                dispatchIdx[numDispatched] = idx;
                dispatchSeq[numDispatched] = my_seq[idx] + 1;
                numDispatched++;
                if (isEnd) end_sent[idx] = true;
            }
        }

        if (allDone) break;

        // 4c. PipeBarrier + Phase B: 写 taskSeq
        if (numDispatched > 0) {
            PipeBarrier<PIPE_ALL>();  // Phase A → B 保序 + drain taskData 到 HBM
            for (int32_t i = 0; i < numDispatched; ++i) {
                int32_t idx = dispatchIdx[i];
                int32_t bkN = idx / 2;
                int32_t chN = idx % 2;
                int32_t newSeq = dispatchSeq[i];
                __gm__ int32_t* tsPtr = GetTaskSeqPtr(params, bkN, chN);
                AscendC::WriteGmByPassDCache(tsPtr, newSeq);
                my_seq[idx] = newSeq;
            }
            // PipeBarrier<PIPE_ALL>();  // 跨核可见性: worker spin 在等 taskSeq, 必须 drain 到 HBM
        }

        // 超时诊断
        if (AscendC::GetSystemCycle() - loopStart > SPIN_TIMEOUT_CYCLES * 40) {
            AscendC::PRINTF("[rank=%d SCHED] TIMEOUT exp=%u\n", rankId, expertIdx);
            for (uint64_t w = 0; w < waveCnt; ++w) {
                int32_t dbgFlag = AscendC::ReadGmByPassDCache(swigluFlagBase + w);
                int32_t dbgD1 = dispatched_g1[w];
                int32_t dbgD2 = dispatched_g2[w];
                uint64_t dbgW = w;
                AscendC::PRINTF("  wave=%lu flag=%d d1=%d d2=%d\n", dbgW, dbgFlag, dbgD1, dbgD2);
            }
            for (uint32_t bk = 1; bk < blockNum; ++bk) {
                for (int32_t ch = 0; ch < 2; ++ch) {
                    int32_t idx = static_cast<int32_t>(bk) * 2 + ch;
                    if (end_acked[idx]) continue;
                    int32_t myS = my_seq[idx];
                    int32_t aicA = AscendC::ReadGmByPassDCache(GetTaskAckPtr(params, static_cast<int32_t>(bk), ch, OWNER_AIC));
                    int32_t aivA = AscendC::ReadGmByPassDCache(GetTaskAckPtr(params, static_cast<int32_t>(bk), ch, OWNER_AIV));
                    int dbgEs = end_sent[idx] ? 1 : 0;
                    int dbgEa = end_acked[idx] ? 1 : 0;
                    AscendC::PRINTF("  STUCK bk=%u ch=%d myseq=%d aicAck=%d aivAck=%d es=%d ea=%d\n",
                                    bk, ch, myS, aicA, aivA, dbgEs, dbgEa);
                }
            }
            break;
        }
    }

    #if MEGA_MOE_DYN_DEBUG
    AscendC::PRINTF("[rank=%d SCHED] <<< EXIT exp=%u\n", rankId, expertIdx);
    #endif
}


// ============================================================================
// DynamicAicLoop — AIC worker, bk ∈ [1, blockNum)
// ============================================================================
// 单写者协议. 轮询 taskSeq[bk*2+0] (G1) 和 taskSeq[bk*2+1] (G2)。
//   cur = ReadGmByPassDCache(taskSeq)
//   if cur > last_seen[ch]:
//     packed = ReadGmByPassDCache(taskData)
//     if END: write taskAck = cur; mark end[ch]
//     else: process(packed); write taskAck = cur
//     last_seen[ch] = cur
// last_seen 由调用方维护, 跨 expert 单调累积。
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
    uint32_t tileNumGmm1, uint32_t tileNumGmm2,
    int32_t* last_seen)  // [0]=G1 channel, [1]=G2 channel; 跨 expert 持续
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

    __gm__ int32_t* seqG1 = GetTaskSeqPtr(params, blockIdx, 0);
    __gm__ int32_t* seqG2 = GetTaskSeqPtr(params, blockIdx, 1);
    __gm__ int32_t* dataG1 = GetTaskDataPtr(params, blockIdx, 0);
    __gm__ int32_t* dataG2 = GetTaskDataPtr(params, blockIdx, 1);
    __gm__ int32_t* ackG1 = GetTaskAckPtr(params, blockIdx, 0, OWNER_AIC);
    __gm__ int32_t* ackG2 = GetTaskAckPtr(params, blockIdx, 1, OWNER_AIC);

    bool g1Pending = false;  // 上次 G1 mmad 的 NotifyVector 没被 drain
    bool g1End = false, g2End = false;
    int g1Seq = 0, g2Seq = 0;

    int64_t loopStart = AscendC::GetSystemCycle();
    while (!g1End || !g2End) {
        // -- G2 channel(优先): AIV1 combine 是瓶颈,要让它尽早开始 + 持续有活。
        //    顺便: G1 mmad 之间的 WaitForVector(0) 间隙正好用 G2 mmad 填满。
        if (!g2End) {
            int32_t cur = AscendC::ReadGmByPassDCache(seqG2);
            if (cur > last_seen[1]) {
                int32_t packed = AscendC::ReadGmByPassDCache(dataG2);
                DynamicTask task;
                UnpackDynamicTask(packed, task);

                if (task.opType == DYN_OP_END) {
                    AscendC::WriteGmByPassDCache(ackG2, cur);
                    // PipeBarrier<PIPE_ALL>();  // 跨核可见性
                    last_seen[1] = cur;
                    g2End = true;
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

                AscendC::WriteGmByPassDCache(ackG2, cur);
                // PipeBarrier<PIPE_ALL>();  // 跨核可见性
                last_seen[1] = cur;
                continue;
            }
        }

        // -- G1 channel(G2 没活时才做): 推进 wave G1 mmad → epi → swiglu 解锁 → 后续 G2 派下来
        if (!g1End) {
            int32_t cur = AscendC::ReadGmByPassDCache(seqG1);
            if (cur > last_seen[0]) {
                int32_t packed = AscendC::ReadGmByPassDCache(dataG1);
                DynamicTask task;
                UnpackDynamicTask(packed, task);

                if (task.opType == DYN_OP_END) {
                    AscendC::WriteGmByPassDCache(ackG1, cur);
                    // PipeBarrier<PIPE_ALL>();  // 跨核可见性: scheduler DCP 读 ack 依赖 HBM commit
                    last_seen[0] = cur;
                    g1End = true;
                    continue;
                }

                // UB 屏障: 下一 G1 mmad 前等上次 AIV0 NotifyCube。如果中间做了 G2 mmad,
                // AIV0 多半已经 epi 完发了 NotifyCube,WaitForVector 立刻返回,等的时间近似 0。
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
                g1Pending = true;

                AscendC::WriteGmByPassDCache(ackG1, cur);
                // PipeBarrier<PIPE_ALL>();  // 跨核可见性
                last_seen[0] = cur;
                continue;
            }
        }

        // 两个 channel 都没新 task: 等一下
        if (AscendC::GetSystemCycle() - loopStart > SPIN_TIMEOUT_CYCLES * 40) {
            int32_t dbgSeqG1 = AscendC::ReadGmByPassDCache(seqG1);
            int32_t dbgSeqG2 = AscendC::ReadGmByPassDCache(seqG2);
            int32_t dbgLastG1 = last_seen[0];
            int32_t dbgLastG2 = last_seen[1];
            int dbgG1End = g1End ? 1 : 0;
            int dbgG2End = g2End ? 1 : 0;
            int dbgG1Pending = g1Pending ? 1 : 0;
            AscendC::PRINTF("[rank=%d AIC bk=%d] *** TIMEOUT idle exp=%u seqG1=%d seqG2=%d lastG1=%d lastG2=%d g1End=%d g2End=%d g1Pending=%d\n",
                            rankId, blockIdx, expertIdx, dbgSeqG1, dbgSeqG2, dbgLastG1, dbgLastG2,
                            dbgG1End, dbgG2End, dbgG1Pending);
            break;
        }
        int64_t ss = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - ss < 500) {}  // 拉长 spin 间隔 减小 HBM 争用
    }

    // 最后一发 G1 NotifyCube drain
    if (g1Pending) {
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
// DynamicAiv0Loop — AIV0 worker, G1 channel (epi)
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
    int32_t blockIdx, int32_t rankId, uint32_t expertIdx, uint32_t tileNumGmm1,
    int32_t* last_seen)  // 单元素, 跨 expert 持续
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

    __gm__ int32_t* seqPtr  = GetTaskSeqPtr(params, blockIdx, 0);
    __gm__ int32_t* dataPtr = GetTaskDataPtr(params, blockIdx, 0);
    __gm__ int32_t* ackPtr  = GetTaskAckPtr(params, blockIdx, 0, OWNER_AIV);

    int taskCnt = 0;
    int64_t loopStart = AscendC::GetSystemCycle();
    while (true) {
        int32_t cur = AscendC::ReadGmByPassDCache(seqPtr);
        if (cur > last_seen[0]) {
            int32_t packed = AscendC::ReadGmByPassDCache(dataPtr);
            DynamicTask task;
            UnpackDynamicTask(packed, task);

            if (task.opType == DYN_OP_END) {
                AscendC::WriteGmByPassDCache(ackPtr, cur);
                // PipeBarrier<PIPE_ALL>();  // 跨核可见性
                last_seen[0] = cur;
                break;
            }
            taskCnt++;
            #if MEGA_MOE_DYN_DEBUG
            if (taskCnt <= 3 && (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B)) {
                int32_t dbgW = task.waveIdx;
                int32_t dbgN = task.nTileIdx;
                int dbgSeq = taskCnt;
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

            AscendC::WriteGmByPassDCache(ackPtr, cur);
            // PipeBarrier<PIPE_ALL>();  // 跨核可见性
            last_seen[0] = cur;
            loopStart = AscendC::GetSystemCycle();  // reset timeout
            continue;
        }

        if (AscendC::GetSystemCycle() - loopStart > SPIN_TIMEOUT_CYCLES * 40) {
            int32_t dbgSeq = cur;
            int32_t dbgLast = last_seen[0];
            AscendC::PRINTF("[rank=%d AIV0 bk=%d] *** TIMEOUT exp=%u seq=%d last=%d\n",
                            rankId, blockIdx, expertIdx, dbgSeq, dbgLast);
            break;
        }
        int64_t ss = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - ss < 500) {}  // 拉长 spin 间隔 减小 HBM 争用
    }

    #if MEGA_MOE_DYN_DEBUG
    if (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B) {
        AscendC::PRINTF("[rank=%d AIV0 bk=%d] <<< EXIT (tasks=%d)\n", rankId, blockIdx, taskCnt);
    }
    #endif
}

// ============================================================================
// DynamicAiv1Loop — AIV1 worker, G2 channel (combine)
// ============================================================================
template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsInterleaved>
__aicore__ inline void DynamicAiv1Loop(
    const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& baseOffset,
    int32_t blockIdx, int32_t rankId, uint32_t expertIdx, uint32_t tileNumGmm2,
    int32_t* last_seen)
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

    __gm__ int32_t* seqPtr  = GetTaskSeqPtr(params, blockIdx, 1);
    __gm__ int32_t* dataPtr = GetTaskDataPtr(params, blockIdx, 1);
    __gm__ int32_t* ackPtr  = GetTaskAckPtr(params, blockIdx, 1, OWNER_AIV);

    int taskCnt = 0;
    int64_t loopStart = AscendC::GetSystemCycle();
    while (true) {
        int32_t cur = AscendC::ReadGmByPassDCache(seqPtr);
        if (cur > last_seen[0]) {
            int32_t packed = AscendC::ReadGmByPassDCache(dataPtr);
            DynamicTask task;
            UnpackDynamicTask(packed, task);

            if (task.opType == DYN_OP_END) {
                AscendC::WriteGmByPassDCache(ackPtr, cur);
                // PipeBarrier<PIPE_ALL>();  // 跨核可见性
                last_seen[0] = cur;
                break;
            }
            taskCnt++;
            #if MEGA_MOE_DYN_DEBUG
            if (taskCnt <= 3 && (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B)) {
                int32_t dbgW = task.waveIdx;
                int32_t dbgN = task.nTileIdx;
                int dbgSeq = taskCnt;
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

            AscendC::WriteGmByPassDCache(ackPtr, cur);
            // PipeBarrier<PIPE_ALL>();  // 跨核可见性
            last_seen[0] = cur;
            loopStart = AscendC::GetSystemCycle();
            continue;
        }

        if (AscendC::GetSystemCycle() - loopStart > SPIN_TIMEOUT_CYCLES * 40) {
            int32_t dbgSeq = cur;
            int32_t dbgLast = last_seen[0];
            AscendC::PRINTF("[rank=%d AIV1 bk=%d] *** TIMEOUT exp=%u seq=%d last=%d\n",
                            rankId, blockIdx, expertIdx, dbgSeq, dbgLast);
            break;
        }
        int64_t ss = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - ss < 500) {}  // 拉长 spin 间隔 减小 HBM 争用
    }

    #if MEGA_MOE_DYN_DEBUG
    if (blockIdx == DBG_BK_A || blockIdx == DBG_BK_B) {
        AscendC::PRINTF("[rank=%d AIV1 bk=%d] <<< EXIT (tasks=%d)\n", rankId, blockIdx, taskCnt);
    }
    #endif
}

} // namespace MegaMoeImpl

#endif