/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef APACE_RUN_GMM2_GENERIC_H
#define APACE_RUN_GMM2_GENERIC_H

#include "apace/kernel/matmul/grouped_matmul/gmm_common.h"

namespace Apace {
namespace Gemm {
namespace Kernel {

// 执行通用 GMM2 tile 循环。
template <typename SyncPolicy, typename BlockMmad, typename WorkSet, typename Config>
__aicore__ inline void Gmm2AicMmadGeneric(BlockMmad &blockMmad, WorkSet &workSet, SyncPolicy &sync,
                                          const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
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
        sync.WaitAivTileDone(loopIdx);

        typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                   Get<K_VALUE>(actualShape), 0};

        auto gmBlockB = workSet.gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                          Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB =
            workSet.gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto gmBlockC = workSet.gmC.Slice(Te::MakeCoord(mLoc, nLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, gmBlockC, singleShape);

        sync.NotifyAivTileDone(loopIdx);
        sync.NotifyDownstreamReady(mLoc);
    }
}

// AIV1 等 AIC tile 就绪后执行 epilogue（统一调用，和 GMM1 一致）。
template <typename SyncPolicy, typename EpilogueOp, typename WorkSet, typename Config>
__aicore__ inline void Gmm2AivEpilogue(WorkSet &workSet, EpilogueOp &epilogueOp, SyncPolicy &sync,
                                        const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        sync.WaitAicTileDone(loopIdx);
        epilogueOp(workSet.gmC, actualShape, mLoc, nLoc, sync, loopIdx);
        sync.NotifyAicTileDone(loopIdx);
    }
}

// 根据 GM 地址建立执行资源，并执行通用 GMM2 阶段。
template <typename SyncPolicy, typename BlockMmad, typename ElementC, bool EnableL2CacheHint = false,
          typename Scheduler, typename Config>
__aicore__ inline void Gmm2ExecGeneric(Scheduler &scheduler, SyncPolicy &sync, const GmmTensorAddrs &gmmAddrInfo,
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
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);
    if constexpr (EnableL2CacheHint && g_coreType == AscendC::AIC) {
        SetWaveWeightL2CacheHint<KernelConfig::IS_WEIGHT_NZ, KernelConfig>(config, allowWeightL2Bypass, gmB, gmScaleB);
    }
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);
    auto gmC = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(gmmAddrInfo.gmm2OutGlobal)), layouts.c);

    using WorkSetType = GroupMatmulWorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                           decltype(gmScaleB), decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};

    if constexpr (g_coreType == AscendC::AIC) {
        if (blockMmadContext != nullptr) {
            InitBlockMmad(*blockMmadContext, config);
            Gmm2AicMmadGeneric<SyncPolicy, BlockMmad>(
                blockMmadContext->blockMmad, workSet, sync, config, startLoopIdx, tileNum);
        } else {
            BlockMmadContext<BlockMmad> localBlockMmadContext;
            InitBlockMmad(localBlockMmadContext, config);
            Gmm2AicMmadGeneric<SyncPolicy, BlockMmad>(
                localBlockMmadContext.blockMmad, workSet, sync, config, startLoopIdx, tileNum);
        }
    }
}

// RunGmm2Generic：执行 Generic GMM2，支持量化和非量化 Combine 模式。
template <typename SyncPolicy, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          bool EnableL2CacheHint = false>
class Gmm2GenericKernel {
public:
    using GmmConfig =
        Config<false, false, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsWeightNZ>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockMmad = typename GmmConfig::BlockMmad;

    struct Params {
        GmmTensorAddrs addrs;
        ProblemShape problemShape;
        uint32_t *startBlockIdx;
        void *blockMmadContext = nullptr;
        bool allowWeightL2Bypass = false;
    };

    __aicore__ inline void Init(const Params &params)
    {
        config_ = GmmConfig::BuildGmm2ProblemConfig(params.problemShape, Gmm1TileM);
    }

    __aicore__ inline void Run(SyncPolicy &sync, const Params &params)
    {
        BlockScheduler scheduler({config_.m, config_.schedulerN, config_.k},
                                 BlockScheduler::Params{Te::MakeCoord(
                                     static_cast<int64_t>(config_.tileM), static_cast<int64_t>(L1_TILE_N))});
        uint32_t tileNum = scheduler.GetTileNum();
        if constexpr (g_coreType == AscendC::AIV) {
            *params.startBlockIdx = (*params.startBlockIdx + tileNum) % config_.blockNum;
            return;
        }

        uint32_t startLoopIdx =
            (config_.blockIdx < *params.startBlockIdx ? config_.blockIdx + config_.blockNum : config_.blockIdx) -
            *params.startBlockIdx;
        using MmadContext = BlockMmadContext<BlockMmad>;
        auto *typedBlockMmadContext = reinterpret_cast<MmadContext *>(params.blockMmadContext);
        Gmm2ExecGeneric<SyncPolicy, BlockMmad, ElementC, EnableL2CacheHint>(
            scheduler, sync, params.addrs, config_, startLoopIdx, tileNum, typedBlockMmadContext,
            params.allowWeightL2Bypass);

        *params.startBlockIdx = (*params.startBlockIdx + tileNum) % config_.blockNum;
    }

    __aicore__ inline void operator()(SyncPolicy &sync, const Params &params)
    {
        Init(params);
        Run(sync, params);
    }

private:
    typename GmmConfig::ProblemConfig config_{};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Apace

#endif // APACE_RUN_GMM2_GENERIC_H
