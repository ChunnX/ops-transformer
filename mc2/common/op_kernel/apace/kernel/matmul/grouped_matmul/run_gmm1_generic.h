/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef APACE_RUN_GMM1_GENERIC_H
#define APACE_RUN_GMM1_GENERIC_H

#include "apace/kernel/matmul/grouped_matmul/gmm_common.h"

namespace Apace {
namespace Gemm {
namespace Kernel {

constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_256 = 256U * 256U;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_128 = 128U * 256U;

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

// 计算一个非 Prefetch GMM1 tile 并写入固定 tile layout 的 UB。
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

// Prefetch 路径的 AIC 将 GMM1 tile 写入 GM，并通知 AIV 后处理。
template <typename MmadContext, typename SyncPolicy, bool IsGmm1Interleaved, typename WorkSet, typename Config>
__aicore__ inline void Gmm1AicPrefetchMmadGeneric(WorkSet &workSet, SyncPolicy &sync, const Config &config,
                                                  uint32_t startLoopIdx, uint32_t tileNum,
                                                  MmadContext *blockMmadContext)
{
    using BlockMmad = decltype(blockMmadContext->blockMmad);
    InitBlockMmad(*blockMmadContext, config);
    auto &blockMmad = blockMmadContext->blockMmad;

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
        Gmm1AicMmadTileToGmGeneric<BlockMmad, IsGmm1Interleaved>(blockMmad, config, workSet.gmB, workSet.gmScaleB,
                                                                 workSet.gmBias, workSet.gmC, actualShape, mLoc, nLoc,
                                                                 kLoc, gmBlockA, gmBlockScaleA);
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        sync.NotifyAivTileDone(loopIdx);
    }
}

// 非 Prefetch 路径的 AIC 将 GMM1 tile 写入 UB，并维持与 AIV 的 ping-pong 同步。
template <typename MmadContext, typename SyncPolicy, bool IsGmm1Interleaved, typename WorkSet, typename Config>
__aicore__ inline void Gmm1AicMmadGeneric(WorkSet &workSet, SyncPolicy &sync, const Config &config,
                                          uint32_t startLoopIdx, uint32_t tileNum, uint16_t &pingpongIdx,
                                          MmadContext *blockMmadContext)
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

        Gmm1AicMmadTileToUbGeneric<BlockMmad, IsGmm1Interleaved>(
            blockMmad, config, workSet.gmB, workSet.gmScaleB, workSet.gmBias, actualShape, nLoc, kLoc, gmBlockA,
            gmBlockScaleA, l0cOutUbFirst, l0cOutUbSecond, pingpongIdx);

        sync.NotifyAivTileDone(loopIdx);
    }
}

// AIV 等 AIC tile 就绪后执行 epilogue（Generic 和 A8W4 统一）。
template <typename SyncPolicy, typename WorkSet, typename Config, typename ActivationQuantOp>
__aicore__ inline void Gmm1AivEpilogue(WorkSet &workSet, SyncPolicy &sync, const Config &config,
                                        uint32_t startLoopIdx, uint32_t tileNum,
                                        ActivationQuantOp &activationQuantOp, uint16_t pingpongIdx = 0)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        sync.WaitAicTileDone(loopIdx);
        activationQuantOp(workSet.gmC, actualShape, mLoc, nLoc, sync, pingpongIdx);
        sync.NotifyAicTileDone(loopIdx);
    }
}

// 根据 GM 地址建立执行资源，并执行通用 GMM1/SwiGLU 阶段。
template <typename BlockMmad, typename ElementC, typename MakeLayoutC, typename SyncPolicy, typename EpilogueOp,
          bool PrefetchToGm, bool IsGmm1Interleaved = false, bool EnableL2CacheHint = false,
          typename Scheduler, typename Config>
__aicore__ inline void Gmm1ExecGeneric(Scheduler &scheduler, SyncPolicy &sync, const GmmTensorAddrs &gmmAddrInfo,
                                        const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                        EpilogueOp &activationQuantOp,
                                        uint16_t &pingpongIdx, BlockMmadContext<BlockMmad> *blockMmadContext,
                                        bool allowWeightL2Bypass)
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
    GM_ADDR cGlobal = 0UL;
    if constexpr (PrefetchToGm) {
        cGlobal = gmmAddrInfo.gmm1OutGlobal;
    }
    auto gmC =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(cGlobal)), layouts.c);

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB),
                                    decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};
    if constexpr (g_coreType == AscendC::AIC) {
        if (blockMmadContext != nullptr) {
            if constexpr (PrefetchToGm) {
                Gmm1AicPrefetchMmadGeneric<BlockMmadContext<BlockMmad>, SyncPolicy, IsGmm1Interleaved>(
                    workSet, sync, config, startLoopIdx, tileNum, blockMmadContext);
            } else {
                Gmm1AicMmadGeneric<BlockMmadContext<BlockMmad>, SyncPolicy, IsGmm1Interleaved>(
                    workSet, sync, config, startLoopIdx, tileNum, pingpongIdx, blockMmadContext);
            }
        } else {
            BlockMmadContext<BlockMmad> localBlockMmadContext;
            if constexpr (PrefetchToGm) {
                Gmm1AicPrefetchMmadGeneric<BlockMmadContext<BlockMmad>, SyncPolicy, IsGmm1Interleaved>(
                    workSet, sync, config, startLoopIdx, tileNum, &localBlockMmadContext);
            } else {
                Gmm1AicMmadGeneric<BlockMmadContext<BlockMmad>, SyncPolicy, IsGmm1Interleaved>(
                    workSet, sync, config, startLoopIdx, tileNum, pingpongIdx, &localBlockMmadContext);
            }
        }
    } else {
        Gmm1AivEpilogue<SyncPolicy>(
            workSet, sync, config, startLoopIdx, tileNum, activationQuantOp, pingpongIdx);
    }
}

template <typename SyncPolicy, typename EpilogueOp, typename ElementA, typename ElementB,
          typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool PrefetchToGm = false,
          bool IsGmm1Interleaved = false, bool EnableL2CacheHint = false>
class Gmm1GenericKernel {
public:
    using GmmConfig =
        Config<false, IsGmm1Interleaved, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsWeightNZ>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MakeLayoutC = typename GmmConfig::MakeLayoutC;

    struct Params {
        GmmTensorAddrs addrs;
        ProblemShape problemShape;
        uint32_t *startBlockIdx;
        uint16_t *pingpongIdx;
        void *blockMmadContext = nullptr;
        bool allowWeightL2Bypass = false;
    };

    __aicore__ inline void Init(const Params &params)
    {
        config_ = GmmConfig::BuildGmm1ProblemConfig(params.problemShape, Gmm1TileM);
    }

    __aicore__ inline void Run(SyncPolicy &sync, EpilogueOp &epilogueOp, const Params &params)
    {
        BlockScheduler scheduler({config_.m, config_.schedulerN, config_.k},
                                 BlockScheduler::Params{Te::MakeCoord(
                                     static_cast<int64_t>(config_.tileM), static_cast<int64_t>(L1_TILE_N))});
        uint32_t tileNum = scheduler.GetTileNum();
        if (GetSubBlockIdx() != 0) {
            *params.startBlockIdx = (*params.startBlockIdx + tileNum) % config_.blockNum;
            return;
        }
        if constexpr (g_coreType == AIV) {
            epilogueOp.UpdateNextProblem({config_.m, config_.outputN, config_.k, 0});
        }

        uint32_t startLoopIdx =
            (config_.blockIdx < *params.startBlockIdx ? config_.blockIdx + config_.blockNum : config_.blockIdx) -
            *params.startBlockIdx;
        using MmadContext = BlockMmadContext<BlockMmad>;
        auto *typedBlockMmadContext = reinterpret_cast<MmadContext *>(params.blockMmadContext);
        Gmm1ExecGeneric<BlockMmad, ElementC, MakeLayoutC, SyncPolicy, EpilogueOp, PrefetchToGm,
                       IsGmm1Interleaved, EnableL2CacheHint>(scheduler, sync, params.addrs, config_,
                                                              startLoopIdx, tileNum, epilogueOp,
                                                              *params.pingpongIdx,
                                                              typedBlockMmadContext, params.allowWeightL2Bypass);
        *params.startBlockIdx = (*params.startBlockIdx + tileNum) % config_.blockNum;
    }

    __aicore__ inline void operator()(SyncPolicy &sync, EpilogueOp &epilogueOp, const Params &params)
    {
        Init(params);
        Run(sync, epilogueOp, params);
    }

private:
    typename GmmConfig::ProblemConfig config_{};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Apace

#endif // APACE_RUN_GMM1_GENERIC_H
