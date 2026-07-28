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
template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename ElementC, bool IsShared,
          bool TopkWeightsPrefetch, bool IsLayered = false, typename WorkSet, typename ExtraArgs>
__aicore__ inline void AicComputeGeneric(BlockMmad &blockMmad, WorkSet &workSet, uint32_t startLoopIdx,
                                         uint32_t tileNum, ExtraArgs &args)
{
    const auto &config = workSet.config;
    uint32_t ubBufSize;
    if constexpr (Policy::IS_GMM1) {
        ubBufSize = config.tileM == L1_TILE_M_128 ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
    } else {
        ubBufSize =
            (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) ? MAX_SINGLE_MN_ALIGN32_NUM_128 : 0;
    }
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), workSet.layouts.c);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), workSet.layouts.c);

    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    GroupSyncSlotLayout groupSyncSlotLayout{};
    if constexpr (!Policy::IS_GMM1 && (CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
        uint32_t logicalCoreCount = config.blockNum * 2U;
        groupSyncSlotLayout = CalcGroupSyncSlotLayout(config.m, logicalCoreCount);
    }

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
        auto gmBlockScaleA = workSet.gmScaleA.Slice(
            Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
            Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));

        if constexpr (Policy::IS_GMM1) {
            uint32_t waveIdx = mLoc / config.tileM;
            if (waveIdx != lastWaveWaited) {
                WaitForUpstreamReady<Policy, IsShared>(workSet.gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
            if constexpr (!TopkWeightsPrefetch) {
                if (args.vecSetSyncCom) {
                    WaitForVector();
                }
            }
        } else {
            if (loopIdx == startLoopIdx) {
                WaitForUpstreamReady<Policy, IsShared>(workSet.gmmAddrInfo, config, mLoc);
            }
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsLayered) {
                if (args.vecSetSyncCom2 >= 2) {
                    WaitForVector(args.pingpongIdx);
                }
            }
        }

        typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                   Get<K_VALUE>(actualShape), 0};

        if constexpr (Policy::IS_GMM1) {
            if constexpr (TopkWeightsPrefetch) {
                auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                                              reinterpret_cast<__gm__ ElementC *>(workSet.gmmAddrInfo.gmm1OutGlobal)),
                                          workSet.layouts.c);
                for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                    auto nOffset = nLoc + weightBlock * config.outputN;
                    auto gmBlockB =
                        workSet.gmB.Slice(Te::MakeCoord(kLoc, nOffset),
                                          Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                    auto gmBlockScaleB =
                        workSet.gmScaleB.Slice(Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nOffset),
                                               Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM),
                                                             Get<N_VALUE>(actualShape)));
                    auto tensorBlockGm = gmC.Slice(Te::MakeCoord(mLoc, nOffset),
                                                   Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                    blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, tensorBlockGm,
                              singleShape);
                }
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
                __gm__ int32_t *statusAddr =
                    reinterpret_cast<__gm__ int32_t *>(workSet.params.workspaceInfo.gmm1TileStatusPtr) +
                    static_cast<int64_t>(args.expertIdx) * workSet.params.tilingData->maxTilesPerExpert + loopIdx;
                AscendC::WriteGmByPassDCache(statusAddr, static_cast<int32_t>(args.expertIdx + 1));
            } else {
                auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
                    Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
                    Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                    auto gmBlockB =
                        workSet.gmB.Slice(Te::MakeCoord(kLoc, nLoc + weightBlock * config.outputN),
                                          Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                    auto gmBlockScaleB = workSet.gmScaleB.Slice(
                        Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * config.outputN),
                        Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM),
                                      Get<N_VALUE>(actualShape)));
                    blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias,
                              weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
                }
                NotifyVector();
                args.vecSetSyncCom = 1;
            }
        } else {
            auto gmBlockB = workSet.gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                              Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = workSet.gmScaleB.Slice(
                Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) {
                auto tensorUb = args.pingpongIdx == 0 ? l0cOutUbFirst : l0cOutUbSecond;
                auto tensorBlockUb = tensorUb.Slice(
                    Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, tensorBlockUb, singleShape);
                NotifyVector(args.pingpongIdx);
                args.vecSetSyncCom2++;
                args.pingpongIdx = 1 - args.pingpongIdx;
            } else {
                auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                                              reinterpret_cast<__gm__ ElementC *>(workSet.gmmAddrInfo.gmm2OutGlobal)),
                                          workSet.layouts.c);
                auto gmBlockC = gmC.Slice(Te::MakeCoord(mLoc, nLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, gmBlockC, singleShape);
                if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
                    NotifyCombineConsumersOfTileCompletion(mLoc, groupSyncSlotLayout,
                                                           workSet.gmmAddrInfo.gmm2CombineSyncCounter);
                }
            }
        }
    }
}

// Prototype: AicComputeA8W4. Executes the A8W4 GMM tile loop and its AIC-AIV1 handshake.
template <uint8_t CombineQuantMode, typename Policy, bool IsShared, bool IsLayered = false, typename BlockMmad,
          typename Scheduler, typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC,
          typename Config, bool TopkWeightsPrefetch>
__aicore__ inline void AicComputeA8W4(BlockMmad &blockMmad, Scheduler &scheduler, TensorA &gmA, TensorScaleA &gmScaleA,
                                      TensorScaleB &gmScaleB, TensorC &l0cOutGm, const GMMAddrInfo &gmmAddrInfo,
                                      const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                      const Params &params, uint32_t expertIdx)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    GroupSyncSlotLayout groupSyncSlotLayout{};
    if constexpr (!Policy::IS_GMM1 && (CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
        uint32_t logicalCoreCount = config.blockNum;
        groupSyncSlotLayout = CalcGroupSyncSlotLayout(config.m, logicalCoreCount);
    }
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        bool shouldWait = false;
        uint32_t waveIdx = 0;
        if constexpr (Policy::IS_GMM1) {
            waveIdx = mLoc / config.tileM;
            shouldWait = waveIdx != lastWaveWaited;
        } else {
            shouldWait = loopIdx == startLoopIdx;
        }
        if (shouldWait) {
            WaitForUpstreamReady<Policy, IsShared>(gmmAddrInfo, config, mLoc);
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
            auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                                Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
            if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
                NotifyCombineConsumersOfTileCompletion(mLoc, groupSyncSlotLayout, gmmAddrInfo.gmm2CombineSyncCounter);
            }
        }
        constexpr bool hasAiv1GmEpilogue =
            Policy::IS_GMM1 || (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered);
        if constexpr (hasAiv1GmEpilogue) {
            if constexpr (Policy::IS_GMM1 && TopkWeightsPrefetch) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
                __gm__ int32_t *statusAddr =
                    reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.gmm1TileStatusPtr) +
                    static_cast<int64_t>(expertIdx) * params.tilingData->maxTilesPerExpert + loopIdx;
                AscendC::WriteGmByPassDCache(statusAddr, static_cast<int32_t>(expertIdx + 1));
            } else {
                NotifyAiv1GmTileReady();
                WaitForAiv1GmTileAccepted();
            }
        }
    }
}

#endif // MEGA_MOE_GROUP_MATMUL_COMPUTE_H
