/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GROUP_MATMUL_COMPUTE_H
#define MEGA_MOE_GROUP_MATMUL_COMPUTE_H

// This header is included from mega_moe_impl.h inside MegaMoeImpl::Detail after
// the shared GMM policies and Config template have been defined.

// Prototype: AicComputeGeneric. Executes the generic GMM tile loop on one logical AIC block.
template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename ElementC,
    typename WorkSet, typename ExtraArgs>
__aicore__ inline void AicComputeGeneric(
    BlockMmad& blockMmad, WorkSet& workSet, uint32_t startLoopIdx, uint32_t tileNum, ExtraArgs& args)
{
    constexpr uint32_t ubBufSize = Policy::IS_GMM1
        ? MAX_SINGLE_MN_ALIGN32_NUM_256
        : ((CombineQuantMode == COMBINE_NO_QUANT) ? MAX_SINGLE_MN_ALIGN32_NUM_128 : 0);
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), workSet.layouts.c);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), workSet.layouts.c);

    const auto& config = workSet.config;
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        if constexpr (Policy::IS_GMM1) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                WaitForUpstreamReady<Policy>(workSet.gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
            if (args.vecSetSyncCom) {
                WaitForVector();
            }
        } else {
            if (loopIdx == startLoopIdx) {
                WaitForUpstreamReady<Policy>(workSet.gmmAddrInfo, config, mLoc);
            }
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                if (args.vecSetSyncCom2 >= 2) {
                    WaitForVector(args.pingpongIdx);
                }
            }
        }

        auto gmBlockA = workSet.gmA.Slice(
            Te::MakeCoord(mLoc, kLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        auto gmBlockScaleA = workSet.gmScaleA.Slice(
            Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
            Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));
        typename BlockMmad::BlockShape singleShape{
            Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), Get<K_VALUE>(actualShape), 0};

        if constexpr (Policy::IS_GMM1) {
            auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto gmBlockB = workSet.gmB.Slice(
                    Te::MakeCoord(kLoc, nLoc + weightBlock * config.outputN),
                    Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                auto gmBlockScaleB = workSet.gmScaleB.Slice(
                    Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * config.outputN),
                    Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias,
                    weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
            }
            NotifyVector();
            args.vecSetSyncCom = 1;
        } else {
            auto gmBlockB = workSet.gmB.Slice(
                Te::MakeCoord(kLoc, nLoc), Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = workSet.gmScaleB.Slice(
                Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                auto tensorUb = args.pingpongIdx == 0 ? l0cOutUbFirst : l0cOutUbSecond;
                auto tensorBlockUb = tensorUb.Slice(
                    Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, tensorBlockUb, singleShape);
                NotifyVector(args.pingpongIdx);
                args.vecSetSyncCom2++;
                args.pingpongIdx = 1 - args.pingpongIdx;
            } else {
                auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                    reinterpret_cast<__gm__ ElementC*>(workSet.gmmAddrInfo.gmm2OutGlobal)), workSet.layouts.c);
                auto gmBlockC = gmC.Slice(
                    Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, gmBlockC, singleShape);
                NotifyCombineTileComplete(mLoc, config.m, L1_TILE_M_256, config.blockNum * 2,
                    args.groupIdx, (__gm__ int32_t*)workSet.params.workspaceInfo.gmm2CombineSyncCounterPtr);
            }
        }
    }
}

// Prototype: AicComputeA8W4. Executes the A8W4 GMM tile loop and its AIC-AIV1 handshake.
template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename Scheduler, typename TensorA,
    typename TensorScaleA, typename TensorScaleB, typename TensorC, typename Config>
__aicore__ inline void AicComputeA8W4(
    BlockMmad& blockMmad, Scheduler& scheduler, TensorA& gmA, TensorScaleA& gmScaleA, TensorScaleB& gmScaleB,
    TensorC& l0cOutGm, const GMMAddrInfo& gmmAddrInfo, const Config& config,
    uint32_t startLoopIdx, uint32_t tileNum,
    uint32_t groupIdx = 0, __gm__ int32_t* gmm2CombineSyncCounterPtr = nullptr)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        bool shouldWait = false;
        uint32_t waveIdx = 0;
        if constexpr (Policy::IS_GMM1) {
            waveIdx = mLoc / L1_TILE_M_256;
            shouldWait = waveIdx != lastWaveWaited;
        } else {
            shouldWait = loopIdx == startLoopIdx;
        }
        if (shouldWait) {
            WaitForUpstreamReady<Policy>(gmmAddrInfo, config, mLoc);
            if constexpr (Policy::IS_GMM1) {
                lastWaveWaited = waveIdx;
            }
        }

        auto gmBlockA = gmA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.k));
        auto gmBlockScaleA =
            gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));

        if constexpr (Policy::IS_GMM1) {
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto nOffset = nLoc + weightBlock * config.outputN;
                auto gmBlockScaleB =
                    gmScaleB.Slice(Te::MakeCoord(0, nOffset), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
                auto tensorBlockGm = l0cOutGm.Slice(
                    Te::MakeCoord(mLoc, nOffset), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
            }
        } else {
            auto gmBlockScaleB =
                gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
            auto tensorBlockGm = l0cOutGm.Slice(
                Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
            if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
                NotifyCombineTileComplete<true>(mLoc, config.m, L1_TILE_M_256,
                    config.blockNum * 2, groupIdx, gmm2CombineSyncCounterPtr);
            }
        }
        constexpr bool hasAiv1GmEpilogue = Policy::IS_GMM1 || CombineQuantMode == COMBINE_NO_QUANT;
        if constexpr (hasAiv1GmEpilogue) {
            // Keep at most one AIC-to-AIV1 GM tile notification outstanding.
            NotifyAiv1GmTileReady();
            WaitForAiv1GmTileAccepted();
        }
    }
}

#endif // MEGA_MOE_GROUP_MATMUL_COMPUTE_H
