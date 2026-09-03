/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef APACE_RUN_GMM2_H
#define APACE_RUN_GMM2_H

#include "apace/kernel/matmul/grouped_matmul/gmm_common.h"

namespace Apace {
namespace Gemm {
namespace Kernel {

// 执行通用 GMM2 tile 循环。同步细节（上游等待 / Combine 计数通知）在 SyncPolicy。
// CombineQuantMode 原始循环体内未使用（wave 量化 combine 计数已移交 wave 层），已删。
template <typename BlockMmad, typename WorkSet, typename Config, typename SyncPolicy>
__aicore__ inline void Gmm2AicMmadGeneric(BlockMmad &blockMmad, WorkSet &workSet, const Config &config,
                                          uint32_t startLoopIdx, uint32_t tileNum, SyncPolicy &sync)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        // Slice only builds tensor views. Keep it ahead of the synchronization waits so the
        // current tile's address calculation is not inserted into the compute critical path.
        auto gmBlockA = workSet.gmA.Slice(Te::MakeCoord(mLoc, kLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        /* E8M0 scales are padded to an even count because GM->L1 moves 64-K scale pairs as b16. */
        auto gmBlockScaleA =
            workSet.gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));

        sync.WaitUpstreamReady(mLoc);

        typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                   Get<K_VALUE>(actualShape), 0};

        auto gmBlockB = workSet.gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                          Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB =
            workSet.gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto gmBlockC = workSet.gmC.Slice(Te::MakeCoord(mLoc, nLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, gmBlockC, singleShape);
        sync.NotifyAivTileDone(mLoc);
    }
}

// 执行 A8W4 GMM2 tile 循环。
template <typename BlockMmad, typename Scheduler, typename TensorA, typename TensorScaleA, typename TensorScaleB,
          typename TensorC, typename Config, typename SyncPolicy>
__aicore__ inline void Gmm2AicMmadA8W4(BlockMmad &blockMmad, Scheduler &scheduler, TensorA &gmA, TensorScaleA &gmScaleA,
                                       TensorScaleB &gmScaleB, TensorC &l0cOutGm, const Config &config,
                                       uint32_t startLoopIdx, uint32_t tileNum, SyncPolicy &sync)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        sync.WaitUpstreamReady(mLoc);

        auto gmBlockA = gmA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.k));
        auto gmBlockScaleA =
            gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));

        auto gmBlockScaleB =
            gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                            Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
        sync.NotifyAivTileDone(mLoc);
    }
}

// 为一个 GMM2 逻辑 block 的 AIV0 路径展开一段 A8W4 权重。
template <typename BlockPrologue, typename Scheduler, typename TensorB, typename Config>
__aicore__ inline void Gmm2Aiv0PrologueA8W4(BlockPrologue &blockPrologue, Scheduler &scheduler, TensorB &gmB,
                                            const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        auto mL1Size = Get<M_VALUE>(actualShape);
        auto nL1Size = Get<N_VALUE>(actualShape);
        blockPrologue(gmB, mL1Size, config.k, nL1Size, nLoc, config.n, config.blockMmadTiling.l1Params.kL1);
    }
}

// 根据 GM 地址建立执行资源，并执行通用 GMM2 阶段。
template <typename BlockMmad, typename ElementC, typename SyncPolicy, bool EnableL2CacheHint = false,
          typename Scheduler, typename Config>
__aicore__ inline void Gmm2ExecGeneric(Scheduler &scheduler, SyncPolicy &sync, const GmmTensorAddrs &tensorAddrs,
                                       const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
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
    auto gmC = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(tensorAddrs.gmm2OutGlobal)), layouts.c);

    using WorkSetType = GroupMatmulWorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                           decltype(gmScaleB), decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};

    if constexpr (g_coreType == AscendC::AIC) {
        if (blockMmadContext != nullptr) {
            InitBlockMmad(*blockMmadContext, config);
            Gmm2AicMmadGeneric<BlockMmad>(blockMmadContext->blockMmad, workSet, config, startLoopIdx, tileNum, sync);
        } else {
            BlockMmadContext<BlockMmad> localBlockMmadContext;
            InitBlockMmad(localBlockMmadContext, config);
            Gmm2AicMmadGeneric<BlockMmad>(localBlockMmadContext.blockMmad, workSet, config, startLoopIdx, tileNum,
                                         sync);
        }
    }
}

// 根据 GM 地址建立执行资源，并执行 A8W4 GMM2 阶段。
template <typename BlockMmad, typename BlockPrologue, typename ElementC, typename SyncPolicy,
          typename Scheduler, typename Config>
__aicore__ inline void Gmm2ExecA8W4(Scheduler &scheduler, SyncPolicy &sync, const GmmTensorAddrs &tensorAddrs,
                                    const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmC = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(tensorAddrs.gmm2OutGlobal)), layouts.c);
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

    using WorkSetType = GroupMatmulWorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                           decltype(gmScaleB), decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};

    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad{};
        typename BlockMmad::BlockShape l0TileShape{config.blockMmadTiling.tileM, config.blockMmadTiling.tileN,
                                                   L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.blockMmadTiling.l1Params);
        Gmm2AicMmadA8W4<BlockMmad, decltype(workSet.scheduler), decltype(workSet.gmA), decltype(workSet.gmScaleA),
                        decltype(workSet.gmScaleB), std::remove_reference_t<decltype(workSet.gmC)>, Config>(
            blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.gmC, config,
            startLoopIdx, tileNum, sync);
    } else if (GetSubBlockIdx() == 0) {
        BlockPrologue blockPrologue;
        Gmm2Aiv0PrologueA8W4(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
    }
}


// RunGmm2Generic：执行 Generic GMM2，支持量化和非量化 Combine 模式。
// =================================================================================================
template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false,
          bool IsWaveFlagGrained = false, typename SyncPolicy>
__aicore__ inline void RunGmm2Generic(const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                      const GmmTensorAddrs &tensorAddrs, uint32_t &startBlockIdx,
                                      SyncPolicy &sync,
                                      void *blockMmadContext = nullptr, bool allowWeightL2Bypass = false)
{
    using GmmConfig =
        Config<false, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsWeightNZ>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape);

    BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        BlockScheduler::Params{Te::MakeCoord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MmadContext = BlockMmadContext<BlockMmad>;
    auto *typedBlockMmadContext = reinterpret_cast<MmadContext *>(blockMmadContext);
    Gmm2ExecGeneric<BlockMmad, ElementC, SyncPolicy, IsWaveFlagGrained>(
        scheduler, sync, tensorAddrs, config, startLoopIdx, tileNum, typedBlockMmadContext, allowWeightL2Bypass);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

// RunGmm2A8W4：执行 A8W4 prologue（W4→W8）和 GMM2；Combine 由调用路径独立调度。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, typename SyncPolicy>
__aicore__ inline void RunGmm2A8W4(const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                   const GmmTensorAddrs &tensorAddrs, uint32_t &startBlockIdx,
                                   SyncPolicy &sync)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using GmmConfig = Config<true, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape);

    using BlockMmad = typename GmmConfig::BlockMmad;
    using BlockPrologue = typename GmmConfig::BlockPrologue;
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

    Gmm2ExecA8W4<BlockMmad, BlockPrologue, ElementC, SyncPolicy, BlockScheduler,
                            decltype(config)>(scheduler, sync, tensorAddrs, config, startLoopIdx, tileNum);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

} // namespace Kernel
} // namespace Gemm
} // namespace Apace

#endif // APACE_RUN_GMM2_H
