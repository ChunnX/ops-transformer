/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_gmm2_kernel.h
 * \brief GMM2 kernel layer: AIC tile loop, AIV1 Combine epilogue, and entry functions.
 *        MoE-specific synchronization is injected via PipelineSync template parameter.
 *        A8W4 Combine epilogue work is delegated to EpilogueOp template parameter.
 */
#ifndef APACE_KERNEL_MATMUL_MOE_GROUPED_MATMUL_MEGA_MOE_GMM2_KERNEL_H
#define APACE_KERNEL_MATMUL_MOE_GROUPED_MATMUL_MEGA_MOE_GMM2_KERNEL_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"
#include "apace/kernel/matmul/moe_grouped_matmul/mega_moe_gmm_common.h"

namespace MegaMoeImpl {

using namespace AscendC;

// --- AIC tile loops ---

// 通用 GMM2 AIC tile 循环。同步通过 PipelineSync 注入。
template <uint8_t CombineQuantMode, typename BlockMmad, bool IsShared, bool IsLayered = false,
          typename WorkSet, typename Config, typename PipelineSync>
__aicore__ inline void Gmm2AicMmadGeneric(BlockMmad &blockMmad, WorkSet &workSet, PipelineSync &pipelineSync,
                                          const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
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

        if constexpr (Config::IS_WAVE_FLAG_GRAINED) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                pipelineSync.WaitGmm2Input(mLoc);
                lastWaveWaited = waveIdx;
            }
        } else if (loopIdx == startLoopIdx) {
            pipelineSync.WaitGmm2Input(mLoc);
        }

        typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                    Get<K_VALUE>(actualShape), 0};
        auto gmBlockB = workSet.gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                           Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB =
            workSet.gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto gmBlockC = workSet.gmC.Slice(Te::MakeCoord(mLoc, nLoc),
                                           Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, gmBlockC, singleShape);

        if constexpr ((CombineQuantMode != 0 || IsLayered) && !IsShared) {
            pipelineSync.NotifyGmm2ToEpilogue(mLoc);
        } else if constexpr (IsShared && !IsLayered) {
            pipelineSync.NotifyGmm2ToEpilogue(mLoc);
        }
    }
}

// A8W4 GMM2 AIC tile 循环。expertTokenCount 和 rowOffsetInExpert 用于 Combine counter 计算。
template <uint8_t CombineQuantMode, bool IsShared, bool IsLayered = false, typename BlockMmad, typename Scheduler,
          typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC, typename Config,
          typename PipelineSync>
__aicore__ inline void Gmm2AicMmadA8W4(BlockMmad &blockMmad, Scheduler &scheduler, TensorA &gmA, TensorScaleA &gmScaleA,
                                        TensorScaleB &gmScaleB, TensorC &l0cOutGm, PipelineSync &pipelineSync,
                                        int32_t &gmTileSequence, const Config &config, uint32_t startLoopIdx,
                                        uint32_t tileNum, uint32_t expertTokenCount, uint32_t rowOffsetInExpert)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        if constexpr (Config::IS_WAVE_FLAG_GRAINED) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                pipelineSync.WaitGmm2Input(mLoc);
                lastWaveWaited = waveIdx;
            }
        } else if (loopIdx == startLoopIdx) {
            pipelineSync.WaitGmm2Input(mLoc);
        }

        auto gmBlockA = gmA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.k));
        auto gmBlockScaleA =
            gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));
        auto gmBlockScaleB =
            gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                             Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);

        if constexpr ((CombineQuantMode != 0 || IsLayered) && !IsShared) {
            pipelineSync.NotifyGmm2ToEpilogue(rowOffsetInExpert + mLoc);
        } else if constexpr (IsShared) {
            pipelineSync.NotifyGmm2ToEpilogue(mLoc);
        }
        constexpr bool hasAiv1GmEpilogue = CombineQuantMode == 0 && !IsShared && !IsLayered;
        if constexpr (hasAiv1GmEpilogue) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            pipelineSync.NotifyGmm2ToEpilogue(gmTileSequence);
        }
    }
}

// AIV1 Combine epilogue：等待 AIC tile 就绪，执行 Combine。
template <typename WorkSet, typename Config, typename EpilogueOp, typename PipelineSync>
__aicore__ inline void Gmm2Aiv1EpilogueA8W4(WorkSet &workSet, PipelineSync &pipelineSync, const Config &config,
                                             uint32_t startLoopIdx, uint32_t tileNum, int32_t &gmTileSequence,
                                             EpilogueOp &epilogueOp)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t mLen = Get<M_VALUE>(actualShape);
        uint32_t nLen = Get<N_VALUE>(actualShape);

        int32_t expectedReadySequence = gmTileSequence + 1;
        pipelineSync.WaitGmm2ToEpilogue(expectedReadySequence);
        epilogueOp.Run(mLoc, nLoc, mLen, nLen, workSet.gmC);
        gmTileSequence = expectedReadySequence;
    }
}

// AIV0 prologue：展开 A8W4 权重。
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

// 通用 GMM2 执行分发器。
template <uint8_t CombineQuantMode, typename BlockMmad, typename ElementC, bool IsLayered = false,
          bool IsShared = false, typename Scheduler, typename Config, typename PipelineSync>
__aicore__ inline void Gmm2ExecGeneric(Scheduler &scheduler, const GmmParams &gmmParams, PipelineSync &pipelineSync,
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
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(gmmParams.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(gmmParams.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmParams.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmParams.bScaleGlobal)),
        layouts.scaleB);
    if constexpr (Config::IS_WAVE_FLAG_GRAINED && g_coreType == AscendC::AIC) {
        SetWaveWeightL2CacheHint<KernelConfig::IS_WEIGHT_NZ, KernelConfig>(config, allowWeightL2Bypass, gmB, gmScaleB);
    }
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);
    auto gmC = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(gmmParams.cGlobal)), layouts.c);

    using WorkSetType = GroupMatmulWorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                           decltype(gmScaleB), decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};

    if constexpr (g_coreType == AscendC::AIC) {
        if (blockMmadContext != nullptr) {
            InitBlockMmad(*blockMmadContext, config);
            Gmm2AicMmadGeneric<CombineQuantMode, BlockMmad, IsShared, IsLayered>(
                blockMmadContext->blockMmad, workSet, pipelineSync, config, startLoopIdx, tileNum);
        } else {
            BlockMmadContext<BlockMmad> localBlockMmadContext;
            InitBlockMmad(localBlockMmadContext, config);
            Gmm2AicMmadGeneric<CombineQuantMode, BlockMmad, IsShared, IsLayered>(
                localBlockMmadContext.blockMmad, workSet, pipelineSync, config, startLoopIdx, tileNum);
        }
    }
}

// A8W4 GMM2 执行分发器。
template <uint8_t CombineQuantMode, typename BlockMmad, typename BlockPrologue, typename ElementC, typename MakeLayoutC,
          bool IsShared, bool IsLayered = false, typename Scheduler, typename Config, typename EpilogueOp,
          typename PipelineSync>
__aicore__ inline void Gmm2ExecA8W4(Scheduler &scheduler, const GmmParams &gmmParams, PipelineSync &pipelineSync,
                                    const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                    int32_t &gmTileSequence, uint32_t expertTokenCount, uint32_t rowOffsetInExpert,
                                    EpilogueOp &epilogueOp)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmC = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(gmmParams.cGlobal)), layouts.c);
    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(gmmParams.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(gmmParams.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmParams.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmParams.bScaleGlobal)),
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
        Gmm2AicMmadA8W4<CombineQuantMode, IsShared, IsLayered, BlockMmad, decltype(workSet.scheduler),
                        decltype(workSet.gmA), decltype(workSet.gmScaleA), decltype(workSet.gmScaleB),
                        std::remove_reference_t<decltype(workSet.gmC)>, Config>(
            blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.gmC, pipelineSync,
            gmTileSequence, config, startLoopIdx, tileNum, expertTokenCount, rowOffsetInExpert);
    } else {
        if (GetSubBlockIdx() == 0) {
            BlockPrologue blockPrologue;
            Gmm2Aiv0PrologueA8W4(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
        } else {
            if constexpr (CombineQuantMode == 0 && !IsShared && !IsLayered) {
                Gmm2Aiv1EpilogueA8W4(workSet, pipelineSync, config, startLoopIdx, tileNum, gmTileSequence, epilogueOp);
            }
        }
    }
}

// RunGmm2Generic：执行通用 GMM2，无 AIV epilogue。
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false,
          bool IsLayered = false, uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false,
          typename PipelineSync = void>
__aicore__ inline void RunGmm2Generic(
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GmmParams &gmmParams, PipelineSync &pipelineSync, uint32_t &startBlockIdx,
    const BlockJobContext &blockJob, void *blockMmadContext = nullptr, bool allowWeightL2Bypass = false)
{
    using GmmConfig = Config<false, CombineQuantMode, ElementA, ElementB, ElementC, ElementMxScaleA,
                             ElementMxScaleB, IsWeightNZ, TopkWeightsPrefetch, IsShared, IsLayered,
                             IsGmm1Interleaved, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape, blockJob, Gmm1TileM);

    BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                              BlockScheduler::Params{Te::MakeCoord(
                                  static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if (GetSubBlockIdx() != 0) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MmadContext = BlockMmadContext<BlockMmad>;
    auto *typedBlockMmadContext = reinterpret_cast<MmadContext *>(blockMmadContext);
    Gmm2ExecGeneric<CombineQuantMode, BlockMmad, ElementC, IsLayered, IsShared>(
        scheduler, gmmParams, pipelineSync, config, startLoopIdx, tileNum, typedBlockMmadContext,
        allowWeightL2Bypass);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

// RunGmm2A8W4：执行 A8W4 GMM2，带 AIV1 Combine epilogue。
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB, typename EpilogueOp,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsLayered = false, bool IsWaveFlagGrained = false,
          typename PipelineSync = void>
__aicore__ inline void RunGmm2A8W4(
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GmmParams &gmmParams, PipelineSync &pipelineSync, uint32_t &startBlockIdx,
    int32_t &gmTileSequence, const BlockJobContext &blockJob,
    uint32_t expertTokenCount, uint32_t rowOffsetInExpert, EpilogueOp &epilogueOp)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using GmmConfig = Config<true, 0, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false,
                             TopkWeightsPrefetch, IsShared, IsLayered, false, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape, blockJob, Gmm1TileM);

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

    Gmm2ExecA8W4<CombineQuantMode, BlockMmad, BlockPrologue, ElementC, MakeLayoutC, IsShared, IsLayered>(
        scheduler, gmmParams, pipelineSync, config, startLoopIdx, tileNum, gmTileSequence,
        expertTokenCount, rowOffsetInExpert, epilogueOp);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

} // namespace MegaMoeImpl

#endif // APACE_KERNEL_MATMUL_MOE_GROUPED_MATMUL_MEGA_MOE_GMM2_KERNEL_H
