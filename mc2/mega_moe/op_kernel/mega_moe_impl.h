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
    int waveTraceIdx, uint16_t flagIdBase = 0U)
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
    // ★ HBM 解耦：layoutC 用全 (m, n) 视图——cube fixpipe 直写 GM，blockMmad 内部按 tile Slice。
    auto layoutC = MakeLayoutC{}(m, n);

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

    // ★ vec 端 UB staging 视图（cube 写 GM，vec MTE2 拉到 UB 再 epi）
    //   epi 内部按 UB[0..MAX_SINGLE_MN*sizeof(float)] 读 mmad 结果，行 stride = L1_TILE_N。
    auto ubLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<c0SizeC>>{}(L1_TILE_M, L1_TILE_N);
    int64_t ubOffset = 0;
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), ubLayoutC);

    // 没启 pingpong 时 UbSecond 折叠到 0（与 UbFirst alias，永不访问），避免描述符越界
    int64_t ubSecondOffsetBytes = kGmm1Pingpong
        ? static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM * sizeof(ElementC))
        : static_cast<int64_t>(0);
    auto l0cOutUbSecond = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::UB, ElementC>(ubSecondOffsetBytes), ubLayoutC);

    // ★ GMM1 输出 GM 视图（float），cube fixpipe 直接写这里。每个 tile 通过 Slice 切到对应位置。
    auto gmGmm1Out = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC*>(gmmAddrInfo.gmm1OutGlobal)),
        layoutC);

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
    // ★ HBM 解耦 + cross-wave flag rotation：
    //   - cube fixpipe 直接写 gmGmm1Out (HBM)，**不再** 等 vec 释放 UB
    //   - cube↔vec sync 仍走 NotifyVector/WaitForCube，但 flag id = pingpongIdx 在 0/1 之间轮换
    //     初值从 caller 传入 flagIdBase，让跨 wave 调用之间 (bi=0→0, bi=1→1) 不撞同槽
    //   - vec 端：WaitForCube → DataCopyPad GM→UB → SyncFunc<MTE2_V> → epi
    //     不发 NotifyCube（HBM 模式没有共享 UB 要释放）
    //   - 单 tile / call 时 pingpongIdx 不在 loop 内轮换，flag id 直接 = flagIdBase
    uint16_t pingpongIdx = flagIdBase;
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

            // ★ HBM 解耦：cube 不再 WaitForVector（不依赖 vec 释放 UB）
            //   cube/vec 1:1 强关系仍走 NotifyVector(pingpongIdx)，flag id 在 0/1 之间轮换避免单槽阻塞

            auto gmBlockA = gmA.Slice(
                Te::MakeCoord(mLoc, kLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
            auto gmBlockScaleA = gmScaleA.Slice(
                Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));

            typename BlockMmad::BlockShape singleShape{
                Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), Get<K_VALUE>(actualShape), 0};
            if constexpr (IsInterleaved) {
                // 交织：1 次 mmad，目标 GM = gmGmm1Out[mLoc..mLoc+M, nLoc..nLoc+N]
                auto gmBlockB = gmB.Slice(
                    Te::MakeCoord(kLoc, nLoc),
                    Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                auto gmBlockScaleB = gmScaleB.Slice(
                    Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                    Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
                auto gmBlockGmm1Out = gmGmm1Out.Slice(
                    Te::MakeCoord(mLoc, nLoc),
                    Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(
                    gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                    gmBlockGmm1Out, singleShape);
            } else {
                // 非交织：2 次 mmad，输出到 gmGmm1Out 的 [nLoc] 和 [nLoc + outputN] 两段
                for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                    auto gmBlockB = gmB.Slice(
                        Te::MakeCoord(kLoc, nLoc + weightBlock * outputN),
                        Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                    auto gmBlockScaleB = gmScaleB.Slice(
                        Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * outputN),
                        Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
                    auto gmBlockGmm1Out = gmGmm1Out.Slice(
                        Te::MakeCoord(mLoc, nLoc + weightBlock * outputN),
                        Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                    blockMmad(
                        gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                        gmBlockGmm1Out, singleShape);
                }
            }

            // NotifyVector(PIPE_FIX) 自带 fixpipe drain，无需额外 SetFlag<FIX_S>。
            // flag id 在 0/1 之间轮换：跨 wave 调用之间避免单槽阻塞。
            NotifyVector(pingpongIdx);

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

        // ★ vecSetSyncCom 已废：HBM 模式下 cube/vec UB 不共享，不存在"等 vec 释放 UB"。
        //   保留 0 让外层 EndSync() 走 early-return（避免误等不存在的 NotifyCube）。
        vecSetSyncCom = 0;

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
            // ★ HBM 解耦：vec 端 WaitForCube → MTE2 GM→UB → 调 epi
            //   1) 等同 pair cube 这一 tile 的 fixpipe 完成（cube 端 NotifyVector(pingpongIdx)）
            WaitForCube(pingpongIdx);
            if (isFirstTile) {
                MegaMoeTrace::RecordWavePoint(params.y2GmAddr, waveTraceIdx,
                                              MegaMoeTrace::SP_GMM1_FIRST_WAIT_DONE);
            }

            //   2) MTE2 GM → UB[0..MAX_SINGLE_MN*sizeof(float)]（与 l0cOutUbFirst_ 同位置）
            //      每个 tile = M_actual × N_actual 个 float
            //      交织：tile = (M, L1_TILE_N)；非交织：cube 已写 2 段 (gate + up)，分别拷到 UbFirst/UbSecond。
            int64_t copyM = Get<M_VALUE>(actualShape);
            int64_t copyN = Get<N_VALUE>(actualShape);
            LocalTensor<float> ubFirstFloat(TPosition::VECIN, 0, MAX_SINGLE_MN_ALIGN32_NUM);
            int64_t srcByteOffset = (static_cast<int64_t>(mLoc) * static_cast<int64_t>(n) +
                                     static_cast<int64_t>(nLoc)) * static_cast<int64_t>(sizeof(float));
            GlobalTensor<float> srcGm;
            srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                gmmAddrInfo.gmm1OutGlobal + srcByteOffset));
            // 行 stride：GM 跨度 n，UB 行连续紧凑（blockLen 个 float）
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(copyM),
                static_cast<uint32_t>(copyN * sizeof(float)),
                static_cast<uint32_t>((static_cast<int64_t>(n) - copyN) * sizeof(float)),
                0U, 0U};
            DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0.0f};
            DataCopyPad(ubFirstFloat, srcGm, copyParams, copyPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_V>();

            if constexpr (!IsInterleaved) {
                // 非交织：再拷 up 段（nLoc + outputN）到 l0cOutUbSecond 位置
                LocalTensor<float> ubSecondFloat(
                    TPosition::VECIN,
                    static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM * sizeof(float)),
                    MAX_SINGLE_MN_ALIGN32_NUM);
                int64_t srcByteOffsetUp = (static_cast<int64_t>(mLoc) * static_cast<int64_t>(n) +
                                           static_cast<int64_t>(nLoc + outputN)) *
                                          static_cast<int64_t>(sizeof(float));
                GlobalTensor<float> srcGmUp;
                srcGmUp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                    gmmAddrInfo.gmm1OutGlobal + srcByteOffsetUp));
                DataCopyPad(ubSecondFloat, srcGmUp, copyParams, copyPadParams);
                SyncFunc<AscendC::HardEvent::MTE2_V>();
            }

            //   3) 调 epi（API 不变，UB 内容已就位）
            AscendC::SetCtrlSpr<60, 60>(0);
            if constexpr (IsInterleaved) {
                epilogueOp(epilogueShape, epilogueOffset, static_cast<uint16_t>(0));
            } else {
                epilogueOp(epilogueShape, epilogueOffset);
            }
            // ★ 不发 NotifyCube：HBM 模式 cube 不依赖 vec 释放 UB

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

        // ★ flag id 在 0/1 之间轮换：单 tile / call 时不进这里轮换；多 tile 时跨 tile 轮换。
        //   跨 wave 调用之间的轮换由 caller 通过 flagIdBase 控制。
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

} // namespace MegaMoeImpl

#endif