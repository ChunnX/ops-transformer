/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef APACE_RUN_GMM1_A8W4_H
#define APACE_RUN_GMM1_A8W4_H

#include "apace/kernel/matmul/grouped_matmul/run_gmm1_generic.h"

namespace Apace {
namespace Gemm {
namespace Kernel {

// AIC 执行一个 A8W4 GMM1 MMAD tile；不管理 Dispatch/GMM1 或 GMM1/SwiGLU 同步，
// AIV0 与 AIC 的 Blaze 内部握手仍由 GMM 实现负责。
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

// 执行 A8W4 GMM1 AIC tile 循环，并维持 Dispatch/GMM1 和 GMM1/SwiGLU 的同步关系。
template <typename BlockMmad, typename SyncPolicy, typename WorkSet, typename Config>
__aicore__ inline void Gmm1AicMmadA8W4(WorkSet &workSet, SyncPolicy &sync, const Config &config,
                                       uint32_t startLoopIdx, uint32_t tileNum)
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
        sync.WaitAivTileDone(loopIdx);

        Gmm1AicMmadTileA8W4(blockMmad, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.gmC, config,
                            actualShape, mLoc, nLoc);

        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
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

// 根据 GM 地址建立执行资源，并执行 A8W4 GMM1/SwiGLU 阶段。
template <typename BlockMmad, typename BlockPrologue, typename ElementC, typename MakeLayoutC, typename SyncPolicy,
          typename EpilogueOp, typename Scheduler, typename Config>
__aicore__ inline void Gmm1ExecA8W4(Scheduler &scheduler, SyncPolicy &sync, const GmmTensorAddrs &gmmAddrInfo,
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
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(gmmAddrInfo.gmm1OutGlobal)), layouts.c);
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
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB),
                                    decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};
    if constexpr (g_coreType == AscendC::AIC) {
        Gmm1AicMmadA8W4<BlockMmad, SyncPolicy>(workSet, sync, config, startLoopIdx, tileNum);
    } else if (GetSubBlockIdx() == 0) {
        Gmm1Aiv0PrologueA8W4<BlockPrologue>(workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
    } else {
        Gmm1AivEpilogue<SyncPolicy>(
            workSet, sync, config, startLoopIdx, tileNum, activationQuantOp);
    }
}

// RunGmm1A8W4：执行 A8W4 prologue（W4→W8）、GMM1、SwiGLU 和量化。
template <typename SyncPolicy, typename EpilogueOp, typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB, uint32_t Gmm1TileM = L1_TILE_M_256>
class Gmm1A8W4Kernel {
public:
    using GmmConfig = Config<true, false, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using BlockPrologue = typename GmmConfig::BlockPrologue;
    using MakeLayoutC = typename GmmConfig::MakeLayoutC;

    struct Params {
        GmmTensorAddrs addrs;
        ProblemShape problemShape;
        uint32_t *startBlockIdx;
    };

    __aicore__ inline void Init(const Params &params)
    {
        static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
        static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");
        config_ = GmmConfig::BuildGmm1ProblemConfig(params.problemShape, Gmm1TileM);
    }

    __aicore__ inline void Run(SyncPolicy &sync, EpilogueOp &activationQuantOp, const Params &params)
    {
        BlockScheduler scheduler({config_.m, config_.outputN, config_.k},
                                 BlockScheduler::Params{Te::MakeCoord(
                                     static_cast<int64_t>(config_.tileM), static_cast<int64_t>(L1_TILE_N))});
        uint32_t tileNum = scheduler.GetTileNum();
        if constexpr (g_coreType == AIV) {
            activationQuantOp.UpdateNextProblem({config_.m, config_.outputN, config_.k, 0});
        }

        uint32_t startLoopIdx =
            (config_.blockIdx < *params.startBlockIdx ? config_.blockIdx + config_.blockNum : config_.blockIdx) -
            *params.startBlockIdx;
        if (startLoopIdx >= tileNum) {
            *params.startBlockIdx = (*params.startBlockIdx + tileNum) % config_.blockNum;
            return;
        }

        Gmm1ExecA8W4<BlockMmad, BlockPrologue, ElementC, MakeLayoutC, SyncPolicy, EpilogueOp,
                     BlockScheduler, decltype(config_)>(
            scheduler, sync, params.addrs, config_, startLoopIdx, tileNum, activationQuantOp);

        *params.startBlockIdx = (*params.startBlockIdx + tileNum) % config_.blockNum;
    }

    __aicore__ inline void operator()(SyncPolicy &sync, EpilogueOp &activationQuantOp, const Params &params)
    {
        Init(params);
        Run(sync, activationQuantOp, params);
    }

private:
    typename GmmConfig::ProblemConfig config_{};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Apace

#endif // APACE_RUN_GMM1_A8W4_H
