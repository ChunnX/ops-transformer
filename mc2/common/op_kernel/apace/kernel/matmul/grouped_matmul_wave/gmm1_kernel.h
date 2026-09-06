/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef APACE_GMM1_KERNEL_H
#define APACE_GMM1_KERNEL_H

#include <type_traits>

#include "apace/kernel/matmul/grouped_matmul_wave/gmm_common.h"
#include "lib/matmul_intf.h"

namespace Apace {
namespace Gemm {
namespace Kernel {



template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorBias, typename TensorC>
struct Gmm1WorkSet {
    Scheduler &scheduler;
    TensorA &gmA;
    TensorB &gmB;
    TensorScaleA &gmScaleA;
    TensorScaleB &gmScaleB;
    TensorBias &gmBias;
    TensorC &gmC;
};

// 计算一个 Prefetch GMM1 tile 并直接写入 GM。
template <typename BlockMmad, bool IsGmm1Interleaved, typename Config, typename TensorB, typename TensorScaleB,
          typename TensorBias, typename ActualShape, typename TensorA, typename TensorScaleA, typename TensorC>
__aicore__ inline void Gmm1AicMmadTileToGmGeneric(BlockMmad &blockMmad, const Config &config, TensorB &gmB,
                                                  TensorScaleB &gmScaleB, TensorBias &gmBias, TensorC &gmC,
                                                  const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc,
                                                  uint32_t kLoc, TensorA &gmBlockA, TensorScaleA &gmBlockScaleA)
{
    typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                               Get<K_VALUE>(actualShape), 0};

    if constexpr (IsGmm1Interleaved) {
        auto gmBlockB =
            gmB.Slice(Te::MakeCoord(kLoc, nLoc), Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB =
            gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm =
            gmC.Slice(Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockGm, singleShape);
    } else {
        for (uint32_t weightBlock = 0; weightBlock < ACTIVATION_N_HALF; ++weightBlock) {
            uint32_t nOffset = nLoc + weightBlock * config.outputN;
            auto gmBlockB = gmB.Slice(Te::MakeCoord(kLoc, nOffset),
                                      Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB =
                gmScaleB.Slice(Te::MakeCoord(0, nOffset), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
            auto tensorBlockGm = gmC.Slice(Te::MakeCoord(mLoc, nOffset),
                                           Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockGm, singleShape);
        }
    }
}

// 计算一个 GMM1 tile 并写入固定 tile layout 的 UB ping-pong。
template <typename BlockMmad, bool IsGmm1Interleaved, typename Config, typename TensorB, typename TensorScaleB,
          typename TensorBias, typename ActualShape, typename TensorA, typename TensorScaleA, typename TensorC>
__aicore__ inline void Gmm1AicMmadTileToUbGeneric(BlockMmad &blockMmad, const Config &config, TensorB &gmB,
                                                  TensorScaleB &gmScaleB, TensorBias &gmBias,
                                                  const ActualShape &actualShape, uint32_t nLoc, uint32_t kLoc,
                                                  TensorA &gmBlockA, TensorScaleA &gmBlockScaleA,
                                                  TensorC &l0cOutUbFirst, TensorC &l0cOutUbSecond, uint16_t pingpongIdx)
{
    typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                               Get<K_VALUE>(actualShape), 0};

    if constexpr (IsGmm1Interleaved) {
        auto gmBlockB =
            gmB.Slice(Te::MakeCoord(kLoc, nLoc), Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB =
            gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorUb = pingpongIdx == 0U ? l0cOutUbFirst : l0cOutUbSecond;
        auto tensorBlockUb =
            tensorUb.Slice(Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockUb, singleShape);
    } else {
        auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
            Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
            Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        for (uint32_t weightBlock = 0; weightBlock < ACTIVATION_N_HALF; ++weightBlock) {
            auto gmBlockB = gmB.Slice(Te::MakeCoord(kLoc, nLoc + weightBlock * config.outputN),
                                      Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = gmScaleB.Slice(Te::MakeCoord(0, nLoc + weightBlock * config.outputN),
                                                Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                      weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
        }
    }
}

// AIC 执行一个 A8W4 GMM1 MMAD tile；不管理 Dispatch/GMM1 或 GMM1/SwiGLU 同步
template <typename BlockMmad, typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC,
          typename Config, typename ActualShape>
__aicore__ inline void Gmm1AicMmadTileA8W4(BlockMmad &blockMmad, TensorA &gmA, TensorScaleA &gmScaleA,
                                           TensorScaleB &gmScaleB, TensorC &l0cOutGm, const Config &config,
                                           const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc)
{
    auto gmBlockA = gmA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.k));
    auto gmBlockScaleA =
        gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));

    for (uint32_t weightBlock = 0; weightBlock < ACTIVATION_N_HALF; ++weightBlock) {
        uint32_t nOffset = nLoc + weightBlock * config.outputN;
        auto gmBlockScaleB =
            gmScaleB.Slice(Te::MakeCoord(0, nOffset), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nOffset),
                                            Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
    }
}

// 通用 GMM1 AIC tile 循环。
template <typename MmadContext, bool IsGmm1Interleaved, bool IsOutputToGm, typename SyncPolicy,
          typename WorkSet, typename Config>
__aicore__ inline void Gmm1AicMmadGeneric(WorkSet &workSet, const Config &config, uint32_t startLoopIdx,
                                           uint32_t tileNum, SyncPolicy &sync, MmadContext *blockMmadContext)
{
    using BlockMmad = decltype(blockMmadContext->blockMmad);
    InitBlockMmad(*blockMmadContext, config);
    auto &blockMmad = blockMmadContext->blockMmad;

    using KernelConfig = typename Config::KernelConfig;
    using ElementC = typename KernelConfig::ElementCType;
    using MakeLayoutC = typename KernelConfig::MakeLayoutC;
    uint32_t ubBufSize = config.tileM == L1_TILE_M_128 ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
    auto ubLayout = MakeLayoutC{}(config.tileM, L1_TILE_N);
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), ubLayout);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), ubLayout);

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);
        auto gmBlockA = workSet.gmA.Slice(Te::MakeCoord(mLoc, kLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        auto gmBlockScaleA =
            workSet.gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));
        sync.WaitUpstreamReady(mLoc);
        sync.WaitAivTileDone(loopIdx);
        if constexpr (IsOutputToGm) {
            Gmm1AicMmadTileToGmGeneric<BlockMmad, IsGmm1Interleaved>(blockMmad, config, workSet.gmB, workSet.gmScaleB,
                                                                     workSet.gmBias, workSet.gmC, actualShape, mLoc,
                                                                     nLoc, kLoc, gmBlockA, gmBlockScaleA);
        } else {
            Gmm1AicMmadTileToUbGeneric<BlockMmad, IsGmm1Interleaved>(
                blockMmad, config, workSet.gmB, workSet.gmScaleB, workSet.gmBias, actualShape, nLoc, kLoc, gmBlockA,
                gmBlockScaleA, l0cOutUbFirst, l0cOutUbSecond, sync.GetPingpongIdx());
        }
        sync.NotifyAivTileDone(loopIdx);
    }
}

// 执行 A8W4 GMM1 AIC tile 循环，并维持 Dispatch/GMM1 和 GMM1/SwiGLU 的同步关系。
template <typename BlockMmad, typename SyncPolicy, typename WorkSet, typename Config>
__aicore__ inline void Gmm1AicMmadA8W4(WorkSet &workSet, const Config &config, uint32_t startLoopIdx,
                                       uint32_t tileNum, SyncPolicy &sync)
{
    BlockMmad blockMmad{};
    typename BlockMmad::BlockShape l0TileShape{config.blockMmadTiling.tileM, config.blockMmadTiling.tileN, L0_TILE_K,
                                               0};
    typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
    blockMmad.Init(matmulShape, l0TileShape, config.blockMmadTiling.l1Params);

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        sync.WaitUpstreamReady(mLoc);

        Gmm1AicMmadTileA8W4(blockMmad, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.gmC, config,
                            actualShape, mLoc, nLoc);

        sync.NotifyAivTileDone(loopIdx);
    }
}

// 为一个 GMM1 逻辑 block 的 AIV0 路径展开两段 A8W4 权重。
template <typename BlockPrologue, typename Scheduler, typename TensorB, typename Config>
__aicore__ inline void Gmm1Aiv0PrologueA8W4(Scheduler &scheduler, TensorB &gmB, const Config &config,
                                            uint32_t startLoopIdx, uint32_t tileNum)
{
    BlockPrologue blockPrologue;
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        auto mL1Size = Get<M_VALUE>(actualShape);
        auto nL1Size = Get<N_VALUE>(actualShape);

        for (uint32_t weightBlock = 0; weightBlock < ACTIVATION_N_HALF; ++weightBlock) {
            auto nOffset = nLoc + weightBlock * config.outputN;
            blockPrologue(gmB, mL1Size, config.k, nL1Size, nOffset, config.n, config.blockMmadTiling.l1Params.kL1);
        }
    }
}

// AIV0 逐 tile 执行 epilogue
template <typename MakeLayoutC, typename WorkSet, typename Config, typename ActivationQuantOp, typename SyncPolicy>
__aicore__ inline void Gmm1Aiv0EpilogueGeneric(WorkSet &workSet, const Config &config, uint32_t startLoopIdx,
                                               uint32_t tileNum, ActivationQuantOp &activationQuantOp,
                                               SyncPolicy &sync)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        sync.WaitAicTileDone(loopIdx);
        activationQuantOp.template ProcessActivationTile<MakeLayoutC>(config, workSet.gmC, actualShape, mLoc, nLoc,
                                                                     sync);
        sync.NotifyAicTileDone(loopIdx);
    }
}

// 根据 GM 地址建立执行资源，并执行通用 GMM1/SwiGLU 阶段。
template <typename BlockMmad, typename ElementC, typename MakeLayoutC, typename SyncPolicy, typename EpilogueOp,
          bool IsOutputToGm = false, bool IsGmm1Interleaved = false, bool EnableL2CacheHint = false,
          typename Scheduler, typename Config>
__aicore__ inline void Gmm1ExecGeneric(Scheduler &scheduler, SyncPolicy &sync, const GmmTensorAddrs &tensorAddrs,
                                       const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                       EpilogueOp &activationQuantOp,
                                       BlockMmadContext<BlockMmad> *blockMmadContext, bool allowWeightL2Bypass)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(tensorAddrs.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(tensorAddrs.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(tensorAddrs.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(tensorAddrs.bScaleGlobal)),
        layouts.scaleB);
    if constexpr (EnableL2CacheHint && g_coreType == AscendC::AIC) {
        SetWaveWeightL2CacheHint<KernelConfig::IS_WEIGHT_NZ, KernelConfig>(config, allowWeightL2Bypass, gmB, gmScaleB);
    }
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);
    GM_ADDR cGlobal = 0UL;
    if constexpr (IsOutputToGm) {
        cGlobal = tensorAddrs.gmm1OutGlobal;
    }
    auto gmC =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(cGlobal)), layouts.c);

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB),
                                    decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};
    if constexpr (g_coreType == AscendC::AIC) {
        if (blockMmadContext != nullptr) {
            Gmm1AicMmadGeneric<BlockMmadContext<BlockMmad>, IsGmm1Interleaved, IsOutputToGm>(
                workSet, config, startLoopIdx, tileNum, sync, blockMmadContext);
        } else {
            BlockMmadContext<BlockMmad> localBlockMmadContext;
            Gmm1AicMmadGeneric<BlockMmadContext<BlockMmad>, IsGmm1Interleaved, IsOutputToGm>(
                workSet, config, startLoopIdx, tileNum, sync, &localBlockMmadContext);
        }
    } else {
        Gmm1Aiv0EpilogueGeneric<MakeLayoutC>(workSet, config, startLoopIdx, tileNum, activationQuantOp, sync);
    }
}

// AIV1 等待 AIC tile 就绪后消费 A8W4 GMM1 输出
template <typename MakeLayoutC, typename WorkSet, typename Config, typename ActivationQuantOp, typename SyncPolicy>
__aicore__ inline void Gmm1Aiv1EpilogueA8W4(WorkSet &workSet, const Config &config, uint32_t startLoopIdx,
                                            uint32_t tileNum, ActivationQuantOp &activationQuantOp,
                                            SyncPolicy &sync)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        sync.WaitAicTileDone(loopIdx);
        activationQuantOp.template ProcessActivationTileA8W4<MakeLayoutC>(config, workSet.gmC, actualShape, mLoc,
                                                                         nLoc, sync);
    }
}

// 根据 GM 地址建立执行资源，并执行 A8W4 GMM1/SwiGLU 阶段。
template <typename BlockMmad, typename BlockPrologue, typename ElementC, typename MakeLayoutC, typename SyncPolicy,
          typename EpilogueOp, typename Scheduler, typename Config>
__aicore__ inline void Gmm1ExecA8W4(Scheduler &scheduler, SyncPolicy &sync, const GmmTensorAddrs &tensorAddrs,
                                    const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                    EpilogueOp &activationQuantOp)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmC = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(tensorAddrs.gmm1OutGlobal)), layouts.c);
    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(tensorAddrs.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(tensorAddrs.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(tensorAddrs.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(tensorAddrs.bScaleGlobal)),
        layouts.scaleB);
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB),
                                    decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};
    if constexpr (g_coreType == AscendC::AIC) {
        Gmm1AicMmadA8W4<BlockMmad>(workSet, config, startLoopIdx, tileNum, sync);
    } else if (GetSubBlockIdx() == 0) {
        Gmm1Aiv0PrologueA8W4<BlockPrologue>(workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
    } else {
        Gmm1Aiv1EpilogueA8W4<MakeLayoutC>(workSet, config, startLoopIdx, tileNum, activationQuantOp, sync);
    }
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256, bool IsOutputToGm = false,
          bool IsGmm1Interleaved = false, bool EnableL2CacheHint = false, typename EpilogueOp, typename SyncPolicy>
__aicore__ inline void RunGmm1Generic(
    EpilogueOp &epilogueOp, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GmmTensorAddrs &tensorAddrs, uint32_t &startBlockIdx, const BlockJobContext &blockJob, SyncPolicy &sync,
    void *blockMmadContext = nullptr, bool allowWeightL2Bypass = false)
{
    using GmmConfig =
        Config<false, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                          IsWeightNZ, IsGmm1Interleaved>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM);

    BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        BlockScheduler::Params{Te::MakeCoord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if (GetSubBlockIdx() != 0) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }
    if constexpr (g_coreType == AIV) {
        epilogueOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MmadContext = BlockMmadContext<BlockMmad>;
    auto *typedBlockMmadContext = reinterpret_cast<MmadContext *>(blockMmadContext);
    using MakeLayoutC = typename GmmConfig::MakeLayoutC;
    Gmm1ExecGeneric<BlockMmad, ElementC, MakeLayoutC, SyncPolicy, EpilogueOp, IsOutputToGm,
                               IsGmm1Interleaved, EnableL2CacheHint>(scheduler, sync, tensorAddrs, config,
                                                                     startLoopIdx, tileNum, epilogueOp,
                                                                     typedBlockMmadContext, allowWeightL2Bypass);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256, bool IsOutputToGm = false,
          bool IsGmm1Interleaved = false, bool EnableL2CacheHint = false, typename EpilogueOp, typename SyncPolicy>
__aicore__ inline void RunGmm1Generic(
    EpilogueOp &epilogueOp, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GmmTensorAddrs &tensorAddrs, uint32_t &startBlockIdx, SyncPolicy &sync, void *blockMmadContext = nullptr,
    bool allowWeightL2Bypass = false)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    RunGmm1Generic<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsWeightNZ, Gmm1TileM,
                   IsOutputToGm, IsGmm1Interleaved, EnableL2CacheHint, EpilogueOp>(
        epilogueOp, problemShape, tensorAddrs, startBlockIdx, blockJob, sync, blockMmadContext, allowWeightL2Bypass);
}

// RunGmm1A8W4：执行 A8W4 prologue（W4→W8）、GMM1、SwiGLU 和量化。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, typename EpilogueOp, typename SyncPolicy>
__aicore__ inline void RunGmm1A8W4(
    EpilogueOp &activationQuantOp, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GmmTensorAddrs &tensorAddrs, uint32_t &startBlockIdx, const BlockJobContext &blockJob, SyncPolicy &sync)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using GmmConfig = Config<true, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM);

    if constexpr (g_coreType == AIV) {
        activationQuantOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    using BlockMmad = typename GmmConfig::BlockMmad;
    using BlockPrologue = typename GmmConfig::BlockPrologue;
    using MakeLayoutC = typename GmmConfig::MakeLayoutC;

    BlockScheduler scheduler({config.m, config.outputN, config.k},
                                        BlockScheduler::Params{Te::MakeCoord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    Gmm1ExecA8W4<BlockMmad, BlockPrologue, ElementC, MakeLayoutC>(
        scheduler, sync, tensorAddrs, config, startLoopIdx, tileNum, activationQuantOp);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, typename EpilogueOp, typename SyncPolicy>
__aicore__ inline void RunGmm1A8W4(
    EpilogueOp &activationQuantOp, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                   const GmmTensorAddrs &tensorAddrs, uint32_t &startBlockIdx, SyncPolicy &sync)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    RunGmm1A8W4<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, Gmm1TileM>(
        activationQuantOp, problemShape, tensorAddrs, startBlockIdx, blockJob, sync);
}

} // namespace Kernel
} // namespace Gemm
} // namespace Apace

#endif // APACE_RUN_GMM1_H
