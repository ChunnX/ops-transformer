/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gmm_common.h
 * \brief Parameterized shared GMM infrastructure — pure matmul with generic PostOp callback
 *        for AIV post-processing. Replaces GetBlockIdx/GetBlockNum/GetSubBlockIdx with explicit
 *        tileIdx/totalCoreNum/subTileIdx. No SwiGLU or Combine implementation code — callers
 *        provide PostOp functors for AIV per-tile post-processing.
 */

#ifndef GMM_COMMON_H
#define GMM_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

#include "../mega_moe_impl_base.h"
#include "../mega_moe_base.h"
#include "../block_epilogue_swiglu_mx_quant.h"

#include "tensor_api/tensor.h"
#include "../blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "../blaze/gemm/block/block_scheduler_swizzle.h"
#include "../blaze/gemm/block/block_mmad_mx_fp8fp4.h"
#include "../blaze/prologue/block_prologue_mx_fp8fp4.h"

using namespace AscendC;

namespace GmmParam {

using BlockScheduler = typename Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 1>;

struct Gmm1Policy {
    static constexpr bool IS_GMM1 = true;
};

struct Gmm2Policy {
    static constexpr bool IS_GMM1 = false;
};

template <bool IsA8W4, typename C>
struct BlockMmadSelector;

template <typename C>
struct BlockMmadSelector<false, C> {
    using type = Blaze::Gemm::Block::BlockMmad<
        typename C::DispatchPolicy,
        typename C::ElementAType, typename C::LayoutA,
        typename C::ElementBType, typename C::LayoutB,
        typename C::ElementCType, typename C::LayoutC,
        typename C::BiasType, typename C::LayoutBias>;
};

template <typename C>
struct BlockMmadSelector<true, C> {
    using type = Blaze::Gemm::Block::BlockMmad<
        typename C::DispatchPolicy,
        AscendC::Std::tuple<typename C::ElementAType, typename C::ElementMxScaleAType>,
        AscendC::Std::tuple<typename C::MakeLayoutA, typename C::MakeLayoutScaleA>,
        AscendC::Std::tuple<typename C::ElementBType, typename C::ElementMxScaleBType>,
        AscendC::Std::tuple<typename C::MakeLayoutB, typename C::MakeLayoutScaleB>,
        typename C::ElementCType, typename C::MakeLayoutC, void, void>;
};

__aicore__ inline void ComputeCoreGrouping(uint32_t coreId, uint32_t numGroups, uint32_t totalCores,
    uint32_t& myGroup, uint32_t& myIdxInGrp, uint32_t& myGrpSize)
{
    uint32_t baseSize = totalCores / numGroups;
    uint32_t remainder = totalCores % numGroups;
    uint32_t boundary = remainder * (baseSize + 1);
    if (coreId < boundary) {
        myGroup = coreId / (baseSize + 1);
        myIdxInGrp = coreId % (baseSize + 1);
        myGrpSize = baseSize + 1;
    } else {
        uint32_t adjusted = coreId - boundary;
        myGroup = remainder + adjusted / baseSize;
        myIdxInGrp = adjusted % baseSize;
        myGrpSize = baseSize;
    }
}

__aicore__ inline void ComputeGroupRange(uint32_t groupIdx, uint32_t numGroups, uint32_t totalCores,
    uint32_t& grpCoreStart, uint32_t& grpCoreSize)
{
    uint32_t baseSize = totalCores / numGroups;
    uint32_t remainder = totalCores % numGroups;
    if (groupIdx < remainder) {
        grpCoreSize = baseSize + 1;
        grpCoreStart = groupIdx * (baseSize + 1);
    } else {
        grpCoreSize = baseSize;
        grpCoreStart = remainder * (baseSize + 1) + (groupIdx - remainder) * baseSize;
    }
}

template <bool IsA8W4 = false>
__aicore__ inline void NotifyGmm2TileGroupComplete(
    uint32_t mLoc, uint32_t m, uint32_t tileM,
    uint32_t blockAivNum, uint32_t groupIdx,
    __gm__ int32_t* counterPtr)
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
    uint32_t participatingCores = IsA8W4 ? (blockAivNum / 2) : blockAivNum;
    uint32_t tokenGroupsThisExpert = Ops::Base::CeilDiv(m, tileM);
    uint32_t tokenGroupIdx = mLoc / tileM;
    uint32_t grpCoreStart = 0, grpCoreSize = 0;
    ComputeGroupRange(tokenGroupIdx, tokenGroupsThisExpert, participatingCores, grpCoreStart, grpCoreSize);
    int64_t baseOffset = static_cast<int64_t>(groupIdx) * blockAivNum * INT_CACHELINE;
    for (uint32_t i = grpCoreStart; i < grpCoreStart + grpCoreSize; i++) {
        uint32_t physicalId = IsA8W4 ? (i * 2 + 1) : i;
        AscendC::AtomicAdd(counterPtr + baseOffset + physicalId * INT_CACHELINE, int32_t(1));
    }
}

template <typename Policy, typename Config>
__aicore__ inline void WaitForUpstreamReady(
    const MegaMoeImpl::GMMAddrInfo& gmmAddrInfo, const Config& config, uint32_t mLoc)
{
    if constexpr (Policy::IS_GMM1) {
        uint32_t waveIdx = mLoc / MegaMoeImpl::L1_TILE_M_256;
        uint32_t targetValue = (mLoc + MegaMoeImpl::L1_TILE_M_256 > config.m) ? (config.m - mLoc) : MegaMoeImpl::L1_TILE_M_256;
        __gm__ int32_t* flagValueAddr = gmmAddrInfo.dispatchToGmm1Flag + waveIdx;
        while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    } else {
        BlockScheduler gmmBlockScheduler(
            {config.m, config.k, config.n},
            BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(MegaMoeImpl::L1_TILE_M_256),
                static_cast<int64_t>(MegaMoeImpl::L1_TILE_N))});
        uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
        __gm__ int32_t* flagValueAddr = gmmAddrInfo.swigluToGmm2Flag;
        while (targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    }
}

template <bool IsA8W4, typename Policy, uint8_t CombineQuantMode, typename ElementA, typename ElementB,
    typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false>
struct Config {
    using ElementAType = ElementA;
    using ElementBType = ElementB;
    using ElementCType = ElementC;
    using ElementMxScaleAType = ElementMxScaleA;
    using ElementMxScaleBType = ElementMxScaleB;

    static constexpr uint32_t C0_SIZE_A = AuxGetC0Size<ElementA>();
    static constexpr uint32_t C0_SIZE_C = AuxGetC0Size<ElementC>();
    static constexpr uint32_t C0_SIZE_SCALE = 2U;
    static constexpr uint32_t C0_SIZE_B = IsA8W4 ? 32U : AuxGetC0Size<ElementB>();

    using LayoutA = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;
    using LayoutScaleA = Te::ScaleANDLayoutPtn;
    using LayoutScaleB = Te::ScaleBDNLayoutPtn;

    using BiasType = float;
    using LayoutBias = Te::NDExtLayoutPtn;
    using DispatchPolicy = Std::conditional_t<IsA8W4,
        Blaze::Gemm::MatmulMxFp8Fp4DynamicKL1TailResplit,
        Blaze::Gemm::MatmulWithScaleMx<>>;
    using LayoutB = Std::conditional_t<IsA8W4,
        Te::ZNLayoutPtn,
        Std::conditional_t<IsWeightNZ, Te::ZNLayoutPtn, Te::DNExtLayoutPtn>>;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE_A>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE_B>>;
    using MakeLayoutScaleA = Te::FrameLayoutFormat<LayoutScaleA, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutScaleB = Te::FrameLayoutFormat<LayoutScaleB, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE_C>>;

    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutAType = decltype(MakeLayoutA{}(uint32_t {}, uint32_t {}));
    using LayoutBType = decltype(MakeLayoutB{}(uint32_t {}, uint32_t {}));
    using LayoutScaleAType = decltype(MakeLayoutScaleA{}(uint32_t {}, uint32_t {}));
    using LayoutScaleBType = decltype(MakeLayoutScaleB{}(uint32_t {}, uint32_t {}));
    using LayoutCType = decltype(MakeLayoutC{}(uint32_t {}, uint32_t {}));
    using LayoutBiasType = decltype(Te::MakeFrameLayout<LayoutBias>(uint32_t {}, uint32_t {}));

    using BlockMmad = typename BlockMmadSelector<IsA8W4, Config>::type;
    using BlockPrologue = Std::conditional_t<IsA8W4,
        Blaze::Gemm::Prologue::BlockPrologue<DispatchPolicy, ElementA, ElementB>,
        void>;

    struct ProblemConfig {
        static __aicore__ inline typename BlockMmad::L1Params DefaultL1Params()
        {
            if constexpr (IsA8W4) {
                return typename BlockMmad::L1Params{.kL1 = MegaMoeImpl::L1_TILE_K, .scaleKL1 = 4096};
            } else {
                return typename BlockMmad::L1Params{
                    .kL1 = MegaMoeImpl::L1_TILE_K, .scaleKL1 = MegaMoeImpl::L1_TILE_K * MegaMoeImpl::SCALE_K_L1_RATE, .l1BufNum = 2};
            }
        }

        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        uint32_t outputN = 0;
        uint32_t blockNum = 0;
        uint32_t blockIdx = 0;
        uint32_t scaleK = 0;
        uint32_t tileM = 0;
        typename BlockMmad::L1Params l1Params = DefaultL1Params();
    };

    struct LayoutBundle {
        LayoutAType a;
        LayoutBType b;
        LayoutScaleAType scaleA;
        LayoutScaleBType scaleB;
        LayoutCType c;
        LayoutBiasType bias;
    };

    __aicore__ static inline ProblemConfig BuildProblemConfigParam(
        const ProblemShape& problemShape, uint32_t tileIdx, uint32_t totalCoreNum)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        if constexpr (Policy::IS_GMM1) {
            config.n = Get<N_VALUE>(problemShape);
            config.k = Get<K_VALUE>(problemShape);
        } else {
            config.n = Get<K_VALUE>(problemShape);
            config.k = Get<N_VALUE>(problemShape) / MegaMoeImpl::SWIGLU_N_HALF;
        }
        config.outputN = Policy::IS_GMM1 ? config.n / MegaMoeImpl::SWIGLU_N_HALF : config.n;
        config.blockNum = totalCoreNum;
        config.blockIdx = tileIdx;
        config.scaleK = CeilDiv(config.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        if constexpr (!IsA8W4) {
            if constexpr (Policy::IS_GMM1) {
                config.tileM = MegaMoeImpl::L1_TILE_M_256;
            } else {
                config.tileM = (CombineQuantMode == COMBINE_NO_QUANT) ? MegaMoeImpl::L1_TILE_M_128 : MegaMoeImpl::L1_TILE_M_256;
            }
        }
        return config;
    }

    __aicore__ static inline LayoutBundle BuildLayouts(const ProblemConfig& config)
    {
        LayoutBundle layouts;
        layouts.a = MakeLayoutA{}(config.m, config.k);
        layouts.b = MakeLayoutB{}(config.k, config.n);
        layouts.scaleA = MakeLayoutScaleA{}(config.m, config.scaleK);
        layouts.scaleB = MakeLayoutScaleB{}(config.scaleK, config.n);
        if constexpr (IsA8W4) {
            layouts.c = MakeLayoutC{}(config.m, config.n);
        } else {
            layouts.bias = Te::MakeFrameLayout<LayoutBias>(1U, config.n);
            if constexpr (Policy::IS_GMM1) {
                layouts.c = MakeLayoutC{}(MegaMoeImpl::L1_TILE_M_256, MegaMoeImpl::L1_TILE_N);
            } else {
                if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                    layouts.c = MakeLayoutC{}(MegaMoeImpl::L1_TILE_M_128, MegaMoeImpl::L1_TILE_N);
                } else {
                    layouts.c = MakeLayoutC{}(config.m, config.n);
                }
            }
        }
        return layouts;
    }
};

template <typename PostOp>
struct Gmm1ArgsGeneric {
    PostOp& postOp;
    int32_t& vecSetSyncCom;
};

template <typename PostOp>
struct Gmm2ArgsGeneric {
    PostOp& postOp;
    int32_t& vecSetSyncCom2;
    uint16_t& pingpongIdx;
    uint32_t groupIdx;
};

template <typename PostOp>
struct Gmm1ArgsA8W4 {
    PostOp& postOp;
    uint32_t groupIdx = 0;
};

template <typename PostOp>
struct Gmm2ArgsA8W4 {
    PostOp& postOp;
    uint32_t groupIdx = 0;
};

template <typename Scheduler, typename TensorA, typename TensorB,
    typename TensorScaleA, typename TensorScaleB, typename TensorBias,
    typename Config, typename LayoutBundle>
struct WorkSetGeneric {
    Scheduler& scheduler;
    TensorA& gmA;
    TensorB& gmB;
    TensorScaleA& gmScaleA;
    TensorScaleB& gmScaleB;
    TensorBias& gmBias;
    const MegaMoeImpl::GMMAddrInfo& gmmAddrInfo;
    const Params& params;
    const Config& config;
    const LayoutBundle& layouts;
};

template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
    typename TensorC>
struct WorkSetA8W4 {
    Scheduler& scheduler;
    TensorA& gmA;
    TensorB& gmB;
    TensorScaleA& gmScaleA;
    TensorScaleB& gmScaleB;
    TensorC& l0cOutGm;
};

template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename ElementC,
    typename WorkSet, typename ExtraArgs>
__aicore__ inline void AicComputeGeneric(
    BlockMmad& blockMmad, WorkSet& workSet, uint32_t startLoopIdx, uint32_t tileNum, ExtraArgs& args)
{
    constexpr uint32_t ubBufSize = Policy::IS_GMM1
        ? MegaMoeImpl::MAX_SINGLE_MN_ALIGN32_NUM_256
        : ((CombineQuantMode == COMBINE_NO_QUANT) ? MegaMoeImpl::MAX_SINGLE_MN_ALIGN32_NUM_128 : 0);
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
            uint32_t waveIdx = mLoc / MegaMoeImpl::L1_TILE_M_256;
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
            for (uint32_t weightBlock = 0; weightBlock < MegaMoeImpl::SWIGLU_N_HALF; ++weightBlock) {
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
                NotifyGmm2TileGroupComplete(mLoc, config.m, MegaMoeImpl::L1_TILE_M_256, config.blockNum * 2,
                    args.groupIdx, (__gm__ int32_t*)workSet.params.workspaceInfo.gmm2CombineSyncCounterPtr);
            }
        }
    }
}

template <typename PostOp, typename WorkSet>
__aicore__ inline void AivPostLoop(
    PostOp& postOp, WorkSet& workSet, uint32_t startLoopIdx, uint32_t tileNum)
{
    const auto& config = workSet.config;
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        WaitForCube();
        postOp(workSet, loopIdx);
        NotifyCube();
    }
}

template <typename ElementC, typename PostOp, typename WorkSet>
__aicore__ inline void AivPostLoopPingPong(
    PostOp& postOp, WorkSet& workSet, uint32_t startLoopIdx, uint32_t tileNum,
    uint16_t& pingpongIdx)
{
    constexpr uint32_t ubBufSize = MegaMoeImpl::MAX_SINGLE_MN_ALIGN32_NUM_128;
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
    LocalTensor<ElementC> l0cOutUbFirst(TPosition::VECIN, ubOffsetFirst,
        MegaMoeImpl::L1_TILE_M_128 * MegaMoeImpl::L1_TILE_N);
    LocalTensor<ElementC> l0cOutUbSecond(TPosition::VECIN, ubOffsetSecond,
        MegaMoeImpl::L1_TILE_M_128 * MegaMoeImpl::L1_TILE_N);

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += workSet.config.blockNum) {
        auto l0cOutUb = pingpongIdx == 0 ? l0cOutUbFirst : l0cOutUbSecond;
        WaitForCube(pingpongIdx);
        postOp(workSet, loopIdx, l0cOutUb);
        NotifyCube(pingpongIdx);
        pingpongIdx = 1 - pingpongIdx;
    }
}

template <typename Policy, uint8_t CombineQuantMode, typename BlockMmad, typename ElementC,
    bool IsLayered = false, typename WorkSet, typename ExtraArgs>
__aicore__ inline void GroupMatmulExecGeneric(
    WorkSet& workSet, uint32_t startLoopIdx, uint32_t tileNum, ExtraArgs& args)
{
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad;
        bool enableL0CPingPong = false;
        typename BlockMmad::BlockShape l0TileShape{workSet.config.tileM, MegaMoeImpl::L1_TILE_N, MegaMoeImpl::L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{workSet.config.m, workSet.config.n, workSet.config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, workSet.config.l1Params, false, enableL0CPingPong);

        AicComputeGeneric<CombineQuantMode, Policy, BlockMmad, ElementC>(
            blockMmad, workSet, startLoopIdx, tileNum, args);
    } else {
        if constexpr (Policy::IS_GMM1) {
            AivPostLoop(args.postOp, workSet, startLoopIdx, tileNum);
        } else if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            AivPostLoopPingPong<ElementC>(args.postOp, workSet, startLoopIdx, tileNum, args.pingpongIdx);
        }
    }
}

template <typename Policy, uint8_t CombineQuantMode, typename ElementA, typename ElementB,
    typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false,
    bool IsLayered = false, typename ExtraArgs>
__aicore__ inline void GroupMatmulImplGenericParam(
    uint32_t tileIdx, uint32_t totalCoreNum, uint32_t subTileIdx,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const MegaMoeImpl::GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, ExtraArgs& args)
{
    using ConfigType = Config<false, Policy, CombineQuantMode, ElementA, ElementB, ElementC,
        ElementMxScaleA, ElementMxScaleB, IsWeightNZ>;
    auto config = ConfigType::BuildProblemConfigParam(problemShape, tileIdx, totalCoreNum);

    BlockScheduler scheduler(
        {config.m, config.outputN, config.k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(config.tileM), static_cast<int64_t>(MegaMoeImpl::L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;

    auto layouts = ConfigType::BuildLayouts(config);

    if constexpr (Policy::IS_GMM1) {
        if (subTileIdx != 0) {
            startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
            return;
        }
    } else if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        if (subTileIdx != 0) return;
    }

    using BlockMmad = typename ConfigType::BlockMmad;
    using BiasType = typename ConfigType::BiasType;

    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);
    auto gmBias = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layouts.bias);

    using WorkSetType = WorkSetGeneric<BlockScheduler, decltype(gmA), decltype(gmB),
        decltype(gmScaleA), decltype(gmScaleB), decltype(gmBias),
        decltype(config), decltype(layouts)>;
    WorkSetType workSet{
        scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias,
        gmmAddrInfo, params, config, layouts
    };

    GroupMatmulExecGeneric<Policy, CombineQuantMode, BlockMmad, ElementC, IsLayered>(
        workSet, startLoopIdx, tileNum, args);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename Scheduler, typename TensorA,
    typename TensorScaleA, typename TensorScaleB, typename TensorC, typename ConfigType>
__aicore__ inline void AicComputeA8W4(
    BlockMmad& blockMmad, Scheduler& scheduler, TensorA& gmA, TensorScaleA& gmScaleA, TensorScaleB& gmScaleB,
    TensorC& l0cOutGm, const MegaMoeImpl::GMMAddrInfo& gmmAddrInfo, const ConfigType& config,
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
            waveIdx = mLoc / MegaMoeImpl::L1_TILE_M_256;
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
            for (uint32_t weightBlock = 0; weightBlock < MegaMoeImpl::SWIGLU_N_HALF; ++weightBlock) {
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
                NotifyGmm2TileGroupComplete<true>(mLoc, config.m, MegaMoeImpl::L1_TILE_M_256,
                    config.blockNum * 2, groupIdx, gmm2CombineSyncCounterPtr);
            }
        }
        NotifyVectorToCopyIn();
    }
}

template <typename Policy, typename BlockPrologue, typename Scheduler, typename TensorB, typename ConfigType>
__aicore__ inline void AivPrologueA8W4(BlockPrologue& blockPrologue, Scheduler& scheduler, TensorB& gmB,
    const ConfigType& config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        auto mL1Size = Get<M_VALUE>(actualShape);
        auto nL1Size = Get<N_VALUE>(actualShape);

        if constexpr (Policy::IS_GMM1) {
            for (uint32_t weightBlock = 0; weightBlock < MegaMoeImpl::SWIGLU_N_HALF; ++weightBlock) {
                auto nOffset = nLoc + weightBlock * config.outputN;
                blockPrologue(gmB, mL1Size, config.k, nL1Size, nOffset, config.n, config.l1Params.kL1);
            }
        } else {
            blockPrologue(gmB, mL1Size, config.k, nL1Size, nLoc, config.n, config.l1Params.kL1);
        }
    }
}



template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename BlockPrologue, typename ElementC,
    typename MakeLayoutC, typename WorkSet, typename ConfigType, typename ExtraArgs>
__aicore__ inline void GroupMatmulExecA8W4Param(
    uint32_t subTileIdx, WorkSet& workSet, const Params& params, const MegaMoeImpl::GMMAddrInfo& gmmAddrInfo,
    const ConfigType& config, uint32_t startLoopIdx, uint32_t tileNum, ExtraArgs& args)
{
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad{};
        typename BlockMmad::BlockShape l0TileShape{MegaMoeImpl::L1_TILE_M_256, MegaMoeImpl::L1_TILE_N, MegaMoeImpl::L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.l1Params);
        AicComputeA8W4<CombineQuantMode, Policy>(blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA,
            workSet.gmScaleB, workSet.l0cOutGm, gmmAddrInfo, config, startLoopIdx, tileNum,
            args.groupIdx, (__gm__ int32_t*)params.workspaceInfo.gmm2CombineSyncCounterPtr);
    } else {
        if (subTileIdx == 0) {
            BlockPrologue blockPrologue;
            AivPrologueA8W4<Policy>(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx,
                tileNum);
        } else {
            args.postOp(workSet, config, params, startLoopIdx, tileNum);
        }
    }
}

template <uint8_t CombineQuantMode, typename Policy, typename ElementA, typename ElementB, typename ElementC,
    typename ElementMxScaleA, typename ElementMxScaleB, typename ExtraArgs>
__aicore__ inline void GroupMatmulImplA8W4Param(
    uint32_t tileIdx, uint32_t totalCoreNum, uint32_t subTileIdx,
    const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape, const MegaMoeImpl::GMMAddrInfo& gmmAddrInfo,
    uint32_t& startBlockIdx, ExtraArgs& args)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using ConfigType = Config<true, Policy, 0, ElementA, ElementB, ElementC, ElementMxScaleA,
        ElementMxScaleB, false>;
    auto config = ConfigType::BuildProblemConfigParam(problemShape, tileIdx, totalCoreNum);

    auto layouts = ConfigType::BuildLayouts(config);
    using BlockMmad = typename ConfigType::BlockMmad;
    using BlockPrologue = typename ConfigType::BlockPrologue;
    using MakeLayoutC = typename ConfigType::MakeLayoutC;

    auto l0cOutGm = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC*>(
        Policy::IS_GMM1 ? gmmAddrInfo.gmm1OutGlobal : gmmAddrInfo.gmm2OutGlobal)), layouts.c);
    auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(
            reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(
            reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);

    BlockScheduler scheduler(
        {config.m, config.outputN, config.k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(MegaMoeImpl::L1_TILE_M_256), static_cast<int64_t>(MegaMoeImpl::L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    using WorkSetType = WorkSetA8W4<
        BlockScheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB), decltype(l0cOutGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, l0cOutGm};
    GroupMatmulExecA8W4Param<CombineQuantMode, Policy, BlockMmad, BlockPrologue, ElementC, MakeLayoutC>(
        subTileIdx, workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, args);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

} // namespace GmmParam

#endif
