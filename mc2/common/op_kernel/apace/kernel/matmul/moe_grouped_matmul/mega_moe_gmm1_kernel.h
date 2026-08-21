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
 * \file mega_moe_gmm1_kernel.h
 * \brief GMM1 kernel layer: AIC tile loop, AIV epilogue tile loop, and entry functions.
 *        MoE-specific synchronization is injected via PipelineSync template parameter.
 *        Epilogue per-tile work is delegated to EpilogueOp template parameter.
 */
#ifndef APACE_KERNEL_MATMUL_GROUPED_MATMUL_MEGA_MOE_GMM1_KERNEL_H
#define APACE_KERNEL_MATMUL_GROUPED_MATMUL_MEGA_MOE_GMM1_KERNEL_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"
#include "apace/kernel/matmul/moe_grouped_matmul/mega_moe_gmm_common.h"

namespace Apace {
namespace Gemm {

using namespace AscendC;

constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_256 = 256U * 256U;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_128 = 128U * 256U;

template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorBias, typename TensorC, typename TensorMetaInfo>
struct Gmm1WorkSet {
    Scheduler &scheduler;
    TensorA &gmA;
    TensorB &gmB;
    TensorScaleA &gmScaleA;
    TensorScaleB &gmScaleB;
    TensorBias &gmBias;
    TensorC &gmC;
    TensorMetaInfo &metaInfoGm;
};

// 计算一个 Prefetch GMM1 tile 并直接写入 GM。
template <typename BlockMmad, bool IsGmm1Interleaved, typename Config, typename TensorB, typename TensorScaleB,
          typename TensorBias, typename ActualShape, typename TensorA, typename TensorScaleA, typename TensorC>
__aicore__ inline void Gmm1AicMmadTileToGmGeneric(
    BlockMmad &blockMmad, const Config &config, TensorB &gmB, TensorScaleB &gmScaleB, TensorBias &gmBias, TensorC &gmC,
    const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc, uint32_t kLoc, TensorA &gmBlockA,
    TensorScaleA &gmBlockScaleA)
{
    typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                               Get<K_VALUE>(actualShape), 0};

    if constexpr (IsGmm1Interleaved) {
        auto gmBlockB = gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                  Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB = gmScaleB.Slice(
            Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm = gmC.Slice(
            Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockGm, singleShape);
    } else {
        for (uint32_t weightBlock = 0; weightBlock < ACTIVATION_N_HALF; ++weightBlock) {
            uint32_t nOffset = nLoc + weightBlock * config.outputN;
            auto gmBlockB = gmB.Slice(Te::MakeCoord(kLoc, nOffset),
                                      Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = gmScaleB.Slice(
                Te::MakeCoord(0, nOffset), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
            auto tensorBlockGm = gmC.Slice(
                Te::MakeCoord(mLoc, nOffset), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockGm, singleShape);
        }
    }
}

// 计算一个非 Prefetch GMM1 tile 并写入固定 tile layout 的 UB。
template <typename BlockMmad, bool IsGmm1Interleaved, typename Config, typename TensorB, typename TensorScaleB,
          typename TensorBias, typename ActualShape, typename TensorA, typename TensorScaleA, typename TensorC>
__aicore__ inline void Gmm1AicMmadTileToUbGeneric(
    BlockMmad &blockMmad, const Config &config, TensorB &gmB, TensorScaleB &gmScaleB, TensorBias &gmBias,
    const ActualShape &actualShape, uint32_t nLoc, uint32_t kLoc, TensorA &gmBlockA, TensorScaleA &gmBlockScaleA,
    TensorC &l0cOutUbFirst, TensorC &l0cOutUbSecond, uint16_t pingpongIdx)
{
    typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                               Get<K_VALUE>(actualShape), 0};

    if constexpr (IsGmm1Interleaved) {
        auto gmBlockB = gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                  Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB = gmScaleB.Slice(
            Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorUb = pingpongIdx == 0U ? l0cOutUbFirst : l0cOutUbSecond;
        auto tensorBlockUb = tensorUb.Slice(
            Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockUb, singleShape);
    } else {
        auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
            Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
            Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        for (uint32_t weightBlock = 0; weightBlock < ACTIVATION_N_HALF; ++weightBlock) {
            auto gmBlockB = gmB.Slice(Te::MakeCoord(kLoc, nLoc + weightBlock * config.outputN),
                                      Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = gmScaleB.Slice(
                Te::MakeCoord(0, nLoc + weightBlock * config.outputN),
                Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                      weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
        }
    }
}

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
        auto tensorBlockGm = l0cOutGm.Slice(
            Te::MakeCoord(mLoc, nOffset), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
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
            blockPrologue(
                gmB, mL1Size, config.k, nL1Size, nOffset, config.n, config.blockMmadTiling.l1Params.kL1);
        }
    }
}

// Prefetch 路径的 AIC 将 GMM1 tile 写入 GM，并通知 AIV 后处理。
template <typename MmadContext, bool IsShared, bool IsGmm1Interleaved, typename WorkSet, typename Config,
          typename PipelineSync>
__aicore__ inline void Gmm1AicPrefetchMmadGeneric(
    WorkSet &workSet, PipelineSync &pipelineSync, const Config &config,
    uint32_t startLoopIdx, uint32_t tileNum, uint32_t expertIdx, MmadContext *blockMmadContext)
{
    using BlockMmad = decltype(blockMmadContext->blockMmad);
    InitBlockMmad(*blockMmadContext, config);
    auto &blockMmad = blockMmadContext->blockMmad;

    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);
        auto gmBlockA = workSet.gmA.Slice(
            Te::MakeCoord(mLoc, kLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        /*
         * E8M0 scales are stored in 64-K pairs. Keep the padded even scale span in the tensor view;
         * the QBMM GM->L1 copy consumes each pair as one 16-bit element.
         */
        auto gmBlockScaleA = workSet.gmScaleA.Slice(
            Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));
        uint32_t waveIdx = mLoc / config.tileM;
        if (waveIdx != lastWaveWaited) {
            pipelineSync.WaitGmm1Input(mLoc);
            lastWaveWaited = waveIdx;
        }
        Gmm1AicMmadTileToGmGeneric<BlockMmad, IsGmm1Interleaved>(
            blockMmad, config, workSet.gmB, workSet.gmScaleB, workSet.gmBias, workSet.gmC, actualShape, mLoc, nLoc,
            kLoc, gmBlockA, gmBlockScaleA);
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        pipelineSync.NotifyGmm1ToEpilogue(expertIdx, loopIdx);
    }
}

// 非 Prefetch 路径的 AIC 将 GMM1 tile 写入 UB，并维持与 AIV 的 ping-pong 同步。
template <typename MmadContext, bool IsShared, bool IsGmm1Interleaved, typename WorkSet, typename Config,
          typename PipelineSync>
__aicore__ inline void Gmm1AicMmadGeneric(
    WorkSet &workSet, PipelineSync &pipelineSync, const Config &config, uint32_t startLoopIdx,
    uint32_t tileNum, int32_t &vecSetSyncCom, uint16_t &pingpongIdx,
    MmadContext *blockMmadContext)
{
    using BlockMmad = decltype(blockMmadContext->blockMmad);
    InitBlockMmad(*blockMmadContext, config);
    auto &blockMmad = blockMmadContext->blockMmad;

    using KernelConfig = typename Config::KernelConfig;
    using ElementC = typename KernelConfig::ElementCType;
    using MakeLayoutC = typename KernelConfig::MakeLayoutC;
    uint32_t ubBufSize =
        config.tileM == L1_TILE_M_128 ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
    auto ubLayout = MakeLayoutC{}(config.tileM, L1_TILE_N);
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), ubLayout);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), ubLayout);

    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);
        auto gmBlockA = workSet.gmA.Slice(
            Te::MakeCoord(mLoc, kLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        auto gmBlockScaleA = workSet.gmScaleA.Slice(
            Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));
        uint32_t waveIdx = mLoc / config.tileM;
        if (waveIdx != lastWaveWaited) {
            pipelineSync.WaitGmm1Input(mLoc);
            lastWaveWaited = waveIdx;
        }
        if constexpr (IsGmm1Interleaved) {
            if (vecSetSyncCom >= 2) {
                pipelineSync.WaitEpilogueToGmm1(pingpongIdx);
            }
        } else if (vecSetSyncCom != 0) {
            pipelineSync.WaitEpilogueToGmm1();
        }

        Gmm1AicMmadTileToUbGeneric<BlockMmad, IsGmm1Interleaved>(
            blockMmad, config, workSet.gmB, workSet.gmScaleB, workSet.gmBias, actualShape, nLoc, kLoc, gmBlockA,
            gmBlockScaleA, l0cOutUbFirst, l0cOutUbSecond, pingpongIdx);

        if constexpr (IsGmm1Interleaved) {
            pipelineSync.NotifyGmm1ToEpilogue(pingpongIdx);
            vecSetSyncCom++;
            pingpongIdx = 1U - pingpongIdx;
        } else {
            pipelineSync.NotifyGmm1ToEpilogue();
            vecSetSyncCom = 1;
        }
    }
}

// 执行 A8W4 GMM1 AIC tile 循环，并维持 Dispatch/GMM1 和 GMM1/SwiGLU 的同步关系。
template <typename BlockMmad, bool IsShared, bool TopkWeightsPrefetch, typename WorkSet, typename Config,
          typename PipelineSync>
__aicore__ inline void Gmm1AicMmadA8W4(
    WorkSet &workSet, PipelineSync &pipelineSync, const Config &config,
    uint32_t startLoopIdx, uint32_t tileNum, int32_t &gmTileSequence, uint32_t expertIdx)
{
    BlockMmad blockMmad{};
    typename BlockMmad::BlockShape l0TileShape{config.blockMmadTiling.tileM, config.blockMmadTiling.tileN, L0_TILE_K,
                                                0};
    typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
    blockMmad.Init(matmulShape, l0TileShape, config.blockMmadTiling.l1Params);

    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t waveIdx = mLoc / config.tileM;
        if (waveIdx != lastWaveWaited) {
            pipelineSync.WaitGmm1Input(mLoc);
            lastWaveWaited = waveIdx;
        }

        Gmm1AicMmadTileA8W4(blockMmad, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.gmC, config,
                            actualShape, mLoc, nLoc);

        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        if constexpr (TopkWeightsPrefetch) {
            pipelineSync.NotifyGmm1ToEpilogue(expertIdx, loopIdx);
        } else {
            pipelineSync.NotifyGmm1ToEpilogue(gmTileSequence);
        }
    }
}

// Prefetch 路径的 AIV0 等待 GM tile 就绪，再执行 SwiGLU/量化。
// 非 Prefetch 路径的 AIV0 直接消费 AIC 的 UB tile，并维持 ping-pong 同步。
template <bool IsPrefetch, bool IsGmm1Interleaved, bool IsWaveFlagGrained,
           typename WorkSet, typename Config, typename EpilogueOp, typename PipelineSync>
__aicore__ inline void Gmm1Aiv0EpilogueGeneric(
    WorkSet &workSet, PipelineSync &pipelineSync, const Config &config,
    uint32_t startLoopIdx, uint32_t tileNum, EpilogueOp &epilogueOp, uint32_t expertBeforeCnt,
    uint32_t expertIdx, uint16_t &pingpongIdx)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t mLen = Get<M_VALUE>(actualShape);
        uint32_t nLen = Get<N_VALUE>(actualShape);
        if constexpr (IsPrefetch) {
            pipelineSync.WaitGmm1ToEpilogue(expertIdx, loopIdx);
            epilogueOp.template Run<false, IsPrefetch, IsGmm1Interleaved>(
                mLoc, nLoc, mLen, nLen, workSet.gmC, workSet.metaInfoGm);
        } else {
            if constexpr (IsGmm1Interleaved) {
                pipelineSync.WaitGmm1ToEpilogue(pingpongIdx);
                epilogueOp.template Run<false, IsPrefetch, IsGmm1Interleaved>(
                    mLoc, nLoc, mLen, nLen, workSet.gmC, workSet.metaInfoGm);
                pipelineSync.NotifyEpilogueToGmm1(pingpongIdx);
                pingpongIdx = 1U - pingpongIdx;
            } else {
                pipelineSync.WaitGmm1ToEpilogue();
                epilogueOp.template Run<false, IsPrefetch, IsGmm1Interleaved>(
                    mLoc, nLoc, mLen, nLen, workSet.gmC, workSet.metaInfoGm);
                pipelineSync.NotifyEpilogueToGmm1();
            }
        }
    }
}

// AIV1 epilogue：A8W4 路径。IsPrefetch 决定同步策略（WaitGmm1ToEpilogue tile-status vs GM sequence+gmTileSequence）。
template <bool IsPrefetch, bool IsWaveFlagGrained, typename WorkSet, typename Config,
           typename EpilogueOp, typename PipelineSync>
__aicore__ inline void Gmm1Aiv1EpilogueA8W4(
    WorkSet &workSet, PipelineSync &pipelineSync, const Config &config,
    uint32_t startLoopIdx, uint32_t tileNum, EpilogueOp &epilogueOp, uint32_t expertBeforeCnt,
    uint32_t expertIdx, int32_t &gmTileSequence)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t mLen = Get<M_VALUE>(actualShape);
        uint32_t nLen = Get<N_VALUE>(actualShape);
        if constexpr (IsPrefetch) {
            pipelineSync.WaitGmm1ToEpilogue(expertIdx, loopIdx);
            epilogueOp.template Run<true, IsPrefetch, false>(
                mLoc, nLoc, mLen, nLen, workSet.gmC, workSet.metaInfoGm);
        } else {
            int32_t expectedReadySequence = gmTileSequence + 1;
            pipelineSync.WaitGmm1ToEpilogue(expectedReadySequence);
            epilogueOp.template Run<true, IsPrefetch, false>(
                mLoc, nLoc, mLen, nLen, workSet.gmC);
            gmTileSequence = expectedReadySequence;
        }
    }
}

// 根据 GM 地址建立执行资源，并执行通用 GMM1/SwiGLU 阶段。
template <typename BlockMmad, bool TopkWeightsPrefetch,
           bool IsShared = false, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false,
           typename Scheduler, typename Config, typename EpilogueOp, typename PipelineSync>
__aicore__ inline void Gmm1ExecGeneric(
    Scheduler &scheduler, const GmmParams &gmmParams, PipelineSync &pipelineSync, const Config &config,
    uint32_t startLoopIdx, uint32_t tileNum, EpilogueOp &epilogueOp, int32_t &vecSetSyncCom,
    uint32_t expertBeforeCnt, uint32_t expertIdx, uint16_t &pingpongIdx,
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
    if constexpr (IsWaveFlagGrained && g_coreType == AscendC::AIC) {
        SetWaveWeightL2CacheHint<KernelConfig::IS_WEIGHT_NZ, KernelConfig>(config, allowWeightL2Bypass, gmB, gmScaleB);
    }
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);
    GM_ADDR cGlobal = 0UL;
    if constexpr (TopkWeightsPrefetch) {
        cGlobal = gmmParams.cGlobal;
    }
    using ElementC = typename KernelConfig::ElementCType;
    auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(cGlobal)),
                              layouts.c);
    AscendC::GlobalTensor<float> metaInfoGm;
    if constexpr (TopkWeightsPrefetch && g_coreType == AscendC::AIV) {
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gmmParams.metaInfoGlobal));
    }

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                    decltype(gmScaleB), decltype(gmBias), decltype(gmC), decltype(metaInfoGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC, metaInfoGm};
    if constexpr (g_coreType == AscendC::AIC) {
        if (blockMmadContext != nullptr) {
            if constexpr (TopkWeightsPrefetch) {
                Gmm1AicPrefetchMmadGeneric<BlockMmadContext<BlockMmad>, IsShared, IsGmm1Interleaved>(
                    workSet, pipelineSync, config, startLoopIdx, tileNum, expertIdx, blockMmadContext);
            } else {
                Gmm1AicMmadGeneric<BlockMmadContext<BlockMmad>, IsShared, IsGmm1Interleaved>(
                    workSet, pipelineSync, config, startLoopIdx, tileNum, vecSetSyncCom, pingpongIdx,
                    blockMmadContext);
            }
        } else {
            BlockMmadContext<BlockMmad> localBlockMmadContext;
            if constexpr (TopkWeightsPrefetch) {
                Gmm1AicPrefetchMmadGeneric<BlockMmadContext<BlockMmad>, IsShared, IsGmm1Interleaved>(
                    workSet, pipelineSync, config, startLoopIdx, tileNum, expertIdx,
                    &localBlockMmadContext);
            } else {
                Gmm1AicMmadGeneric<BlockMmadContext<BlockMmad>, IsShared, IsGmm1Interleaved>(
                    workSet, pipelineSync, config, startLoopIdx, tileNum, vecSetSyncCom, pingpongIdx,
                    &localBlockMmadContext);
            }
        }
    } else {
        Gmm1Aiv0EpilogueGeneric<TopkWeightsPrefetch, IsGmm1Interleaved, IsWaveFlagGrained>(
            workSet, pipelineSync, config, startLoopIdx, tileNum, epilogueOp, expertBeforeCnt,
            expertIdx, pingpongIdx);
    }
}

// 根据 GM 地址建立执行资源，并执行 A8W4 GMM1/SwiGLU 阶段。
template <typename BlockMmad, typename BlockPrologue, bool IsShared,
           typename Scheduler, typename Config, bool TopkWeightsPrefetch, bool IsWaveFlagGrained,
           typename EpilogueOp, typename PipelineSync>
__aicore__ inline void Gmm1ExecA8W4(
    Scheduler &scheduler, const GmmParams &gmmParams, PipelineSync &pipelineSync, const Config &config,
    uint32_t startLoopIdx, uint32_t tileNum, int32_t &gmTileSequence, EpilogueOp &epilogueOp,
    uint32_t expertBeforeCnt, uint32_t expertIdx)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;
    using ElementC = typename KernelConfig::ElementCType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                                  reinterpret_cast<__gm__ ElementC *>(gmmParams.cGlobal)),
                              layouts.c);
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
    AscendC::GlobalTensor<float> metaInfoGm;
    if constexpr (TopkWeightsPrefetch && g_coreType == AscendC::AIV) {
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gmmParams.metaInfoGlobal));
    }

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                    decltype(gmScaleB), decltype(gmBias), decltype(gmC), decltype(metaInfoGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC, metaInfoGm};
    if constexpr (g_coreType == AscendC::AIC) {
        Gmm1AicMmadA8W4<BlockMmad, IsShared, TopkWeightsPrefetch>(
            workSet, pipelineSync, config, startLoopIdx, tileNum, gmTileSequence, expertIdx);
    } else if (GetSubBlockIdx() == 0) {
        Gmm1Aiv0PrologueA8W4<BlockPrologue>(workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
    } else {
        Gmm1Aiv1EpilogueA8W4<TopkWeightsPrefetch, IsWaveFlagGrained>(
            workSet, pipelineSync, config, startLoopIdx, tileNum, epilogueOp, expertBeforeCnt,
            expertIdx, gmTileSequence);
    }
}

// RunGmm1Generic（带 BlockJobContext 参数版本）
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, typename EpilogueOp, bool IsWeightNZ = false,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false,
          typename PipelineSync = void>
__aicore__ inline void RunGmm1Generic(
    EpilogueOp &epilogueOp,
    const GmmParams &gmmParams, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    PipelineSync &pipelineSync, uint32_t &startBlockIdx, int32_t &vecSetSyncCom, const BlockJobContext &blockJob,
    uint32_t expertBeforeCnt, uint32_t expertIdx, uint16_t &pingpongIdx,
    void *blockMmadContext = nullptr, bool allowWeightL2Bypass = false)
{
    using GmmConfig = Config<false, 0, ElementA, ElementB, ElementC, ElementMxScaleA,
                                        ElementMxScaleB, IsWeightNZ, TopkWeightsPrefetch, IsShared,
                                        false, IsGmm1Interleaved, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM);

    BlockScheduler scheduler(
        {config.m, config.schedulerN, config.k},
        BlockScheduler::Params{
            Te::MakeCoord(static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if (GetSubBlockIdx() != 0) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }
    if constexpr (g_coreType == AscendC::AIV) {
        epilogueOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MmadContext = BlockMmadContext<BlockMmad>;
    auto *typedBlockMmadContext = reinterpret_cast<MmadContext *>(blockMmadContext);
    Gmm1ExecGeneric<BlockMmad, TopkWeightsPrefetch, IsShared,
                    IsGmm1Interleaved, IsWaveFlagGrained>(
        scheduler, gmmParams, pipelineSync, config, startLoopIdx, tileNum, epilogueOp, vecSetSyncCom, expertBeforeCnt,
        expertIdx, pingpongIdx, typedBlockMmadContext, allowWeightL2Bypass);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

// RunGmm1A8W4（带 BlockJobContext 参数版本）
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, typename EpilogueOp, uint32_t Gmm1TileM = L1_TILE_M_256,
          bool TopkWeightsPrefetch = false, bool IsShared = false, bool IsWaveFlagGrained = false,
          typename PipelineSync = void>
__aicore__ inline void RunGmm1A8W4(
    EpilogueOp &epilogueOp,
    const GmmParams &gmmParams, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    PipelineSync &pipelineSync, uint32_t &startBlockIdx, int32_t &gmTileSequence,
    const BlockJobContext &blockJob,
    uint32_t expertBeforeCnt, uint32_t expertIdx = 0)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using GmmConfig = Config<true, 0, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                                        false, TopkWeightsPrefetch, IsShared, false>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM);

    if constexpr (g_coreType == AscendC::AIV) {
        epilogueOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

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

    Gmm1ExecA8W4<BlockMmad, BlockPrologue, IsShared,
                 BlockScheduler, decltype(config), TopkWeightsPrefetch, IsWaveFlagGrained>(
        scheduler, gmmParams, pipelineSync, config, startLoopIdx, tileNum, gmTileSequence,
        epilogueOp, expertBeforeCnt, expertIdx);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

} // namespace Gemm
} // namespace Apace

#endif // APACE_KERNEL_MATMUL_GROUPED_MATMUL_MEGA_MOE_GMM1_KERNEL_H
