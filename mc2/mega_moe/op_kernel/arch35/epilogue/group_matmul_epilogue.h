/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GROUP_MATMUL_EPILOGUE_H
#define MEGA_MOE_GROUP_MATMUL_EPILOGUE_H

// This header is included from mega_moe_impl.h inside MegaMoeImpl::Detail after
// the AIC and AIV GMM implementation headers have been included.

// Prototype: GroupMatmulExecGeneric. Dispatches one generic GMM stage to its AIC or AIV implementation.
template <typename Policy, uint8_t CombineQuantMode, typename BlockMmad, typename ElementC, typename MakeLayoutC,
          bool TopkWeightsPrefetch, bool IsLayered = false, bool IsShared = false, typename WorkSet, typename ExtraArgs>
__aicore__ inline void GroupMatmulExecGeneric(WorkSet &workSet, uint32_t startLoopIdx, uint32_t tileNum,
                                              ExtraArgs &args)
{
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad;
        bool enableL0CPingPong = false;
        typename BlockMmad::BlockShape l0TileShape{workSet.config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{workSet.config.m, workSet.config.n, workSet.config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, workSet.config.l1Params, false, enableL0CPingPong);

        AicComputeGeneric<CombineQuantMode, Policy, BlockMmad, ElementC, IsShared, TopkWeightsPrefetch, IsLayered>(
            blockMmad, workSet, startLoopIdx, tileNum, args);
    } else {
        if constexpr (Policy::IS_GMM1) {
            AivGmm1PostGeneric<MakeLayoutC, ElementC, TopkWeightsPrefetch>(workSet, args, startLoopIdx, tileNum);
        } else if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) {
            AivGmm2PostGeneric<ElementC>(workSet, args, startLoopIdx, tileNum);
        }
        // GMM2 量化模式：AIV 不经此路径，由 ProcessCombine 独立处理
    }
}

// Prototype: GroupMatmulExecA8W4. Dispatches one A8W4 GMM stage across the block's AIC and two AIVs.
template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename BlockPrologue, typename ElementC,
          typename MakeLayoutC, bool IsShared, bool IsLayered = false, typename WorkSet, typename Config,
          bool TopkWeightsPrefetch, typename ExtraArgs>
__aicore__ inline void GroupMatmulExecA8W4(WorkSet &workSet, const Params &params, const GMMAddrInfo &gmmAddrInfo,
                                           const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                           ExtraArgs &args)
{
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad{};
        typename BlockMmad::BlockShape l0TileShape{config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.l1Params);
        if constexpr (Policy::IS_GMM1) {
            AicComputeA8W4<CombineQuantMode, Policy, IsShared, false, BlockMmad, decltype(workSet.scheduler),
                           decltype(workSet.gmA), decltype(workSet.gmScaleA), decltype(workSet.gmScaleB),
                           decltype(workSet.l0cOutGm), Config, TopkWeightsPrefetch>(
                blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.l0cOutGm,
                gmmAddrInfo, config, startLoopIdx, tileNum, params, args.expertIdx);
        } else {
            AicComputeA8W4<CombineQuantMode, Policy, IsShared, IsLayered, BlockMmad, decltype(workSet.scheduler),
                           decltype(workSet.gmA), decltype(workSet.gmScaleA), decltype(workSet.gmScaleB),
                           decltype(workSet.l0cOutGm), Config, TopkWeightsPrefetch>(
                blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.l0cOutGm,
                gmmAddrInfo, config, startLoopIdx, tileNum, params, 0);
        }
    } else {
        if (GetSubBlockIdx() == 0) {
            BlockPrologue blockPrologue;
            AivPrologueA8W4<Policy>(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
        } else {
            if constexpr (Policy::IS_GMM1) {
                AivGmm1PostA8W4<ElementC, MakeLayoutC, decltype(workSet.scheduler), decltype(workSet.l0cOutGm),
                                decltype(args.swigluQuantOp), Config, TopkWeightsPrefetch>(
                    args.swigluQuantOp, workSet.scheduler, workSet.l0cOutGm, config, startLoopIdx, tileNum, params,
                    args.expertBeforeCnt, args.expertIdx);
            } else {
                if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) {
                    AivGmm2PostA8W4<ElementC, MakeLayoutC>(workSet.scheduler, workSet.l0cOutGm, params, args.groupCnt,
                                                           config, startLoopIdx, tileNum);
                }
            }
        }
    }
}

#endif // MEGA_MOE_GROUP_MATMUL_EPILOGUE_H
