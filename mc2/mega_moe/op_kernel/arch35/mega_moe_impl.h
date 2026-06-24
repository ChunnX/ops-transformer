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
 * \file mega_moe_impl.h
 * \brief
 */

#ifndef MEGA_MOE_IMPL_H
#define MEGA_MOE_IMPL_H

#include "kernel_operator.h"

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_base.h"

#include "tensor_api/tensor.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_swizzle.h"
#include "blaze/gemm/block/block_mmad_mx_fp8fp4.h"
#include "blaze/prologue/block_prologue_mx_fp8fp4.h"

#include "mega_moe_combine_send.h"

namespace MegaMoeImpl {
using BlockScheduler = typename Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 1>;  // 3: SwizzleOffset
constexpr uint32_t ALIGN32 = 32U;
constexpr uint32_t L1_TILE_M_256 = 256;
constexpr uint32_t L1_TILE_M_128 = 128;
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t L1_TILE_K = 256;
constexpr uint32_t L0_TILE_K = 128;
constexpr uint32_t SCALE_K_L1_RATE = 2;
constexpr uint32_t SWIGLU_N_HALF = 2;
constexpr uint32_t MAX_SINGLE_MN_256_256 = 256 * 256;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_256 = (MAX_SINGLE_MN_256_256 + 31U) / ALIGN32 * ALIGN32;
constexpr uint32_t MAX_SINGLE_MN_128_256 = 128 * 256;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_128 = (MAX_SINGLE_MN_128_256 + 31U) / ALIGN32 * ALIGN32;
constexpr uint32_t TRIPLE_TENSOR_ADDR = 200U * 1024U;  // triple tensor 在 UB 中的起始地址

// =================================================================================================
// ComputeCoreGrouping：计算当前 core 所属的 group 及其在 group 内的位置
// =================================================================================================
// 将 totalCores 个 core 均匀分配到 numGroups 个 group 中，余数分配给前 remainder 个 group。
__aicore__ inline void ComputeCoreGrouping(uint32_t coreId, uint32_t numGroups, uint32_t totalCores,
    uint32_t& myGroup, uint32_t& myIdxInGrp, uint32_t& myGrpSize)
{
    uint32_t baseSize = totalCores / numGroups;      // 每个 group 的基础 core 数
    uint32_t remainder = totalCores % numGroups;     // 余数，前 remainder 个 group 多分配 1 个 core
    uint32_t boundary = remainder * (baseSize + 1);  // 前 remainder 个 group 占用的 core 总数
    
    // 判断当前 core 是否在前 remainder 个 group 中（这些 group 有 baseSize+1 个 core）
    if (coreId < boundary) {
        myGroup = coreId / (baseSize + 1);           // 所属 group 索引
        myIdxInGrp = coreId % (baseSize + 1);        // 在 group 内的索引
        myGrpSize = baseSize + 1;                    // 当前 group 的 core 数
    } else {
        // 当前 core 在后面的 group 中（这些 group 只有 baseSize 个 core）
        uint32_t adjusted = coreId - boundary;       // 减去前 remainder 个 group 占用的 core 数
        myGroup = remainder + adjusted / baseSize;   // 所属 group 索引 = remainder + 偏移
        myIdxInGrp = adjusted % baseSize;            // 在 group 内的索引
        myGrpSize = baseSize;                        // 当前 group 的 core 数
    }
}

// =================================================================================================
// ComputeGroupRange：计算指定 group 包含的 core 范围
// =================================================================================================
// ComputeCoreGrouping 的逆操作：给定 groupIdx，返回该 group 的起始 core 和 core 数量。
// 用于 GMM2 量化路径中，AIC 计算完一个 tile 后，通知负责该 token group 的所有 AIV core。
__aicore__ inline void ComputeGroupRange(uint32_t groupIdx, uint32_t numGroups, uint32_t totalCores,
    uint32_t& grpCoreStart, uint32_t& grpCoreSize)
{
    uint32_t baseSize = totalCores / numGroups;      // 每个 group 的基础 core 数
    uint32_t remainder = totalCores % numGroups;     // 余数，前 remainder 个 group 多分配 1 个 core
    
    if (groupIdx < remainder) {
        // 当前 group 在前 remainder 个 group 中，有 baseSize+1 个 core
        grpCoreSize = baseSize + 1;
        grpCoreStart = groupIdx * (baseSize + 1);  // 起始 core = groupIdx * (baseSize+1)
    } else {
        // 当前 group 在后面的 group 中，只有 baseSize 个 core
        grpCoreSize = baseSize;
        // 起始 core = 前 remainder 个 group 占用的 core 数 + 偏移
        grpCoreStart = remainder * (baseSize + 1) + (groupIdx - remainder) * baseSize;
    }
}

// ==================================================================================
// A8W8/A4W4 configuration — shared type aliases and problem config for GMM1 / GMM2.
// ==================================================================================
namespace Detail {
struct Gmm1PolicyA8W8 {
    static constexpr bool IS_GMM1 = true;
};

struct Gmm2PolicyA8W8 {
    static constexpr bool IS_GMM1 = false;
};

template <typename Policy, uint8_t CombineQuantMode, typename ElementA, typename ElementB,
    typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
struct ConfigA8W8 {
    static constexpr bool TRANS_A = false;
    static constexpr bool TRANS_B = true;
    static constexpr uint32_t C0_SIZE_A = AuxGetC0Size<ElementA>();
    static constexpr uint32_t C0_SIZE_B = AuxGetC0Size<ElementB>();
    static constexpr uint32_t C0_SIZE_C = AuxGetC0Size<ElementC>();
    static constexpr uint32_t C0_SIZE_SCALE = 2U;

    using BiasType = float;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
    using LayoutA = Std::conditional_t<TRANS_A, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutB = Std::conditional_t<TRANS_B, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutBias = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE_A>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE_B>>;
    using MakeLayoutScaleA = Std::conditional_t<
        TRANS_A, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<C0_SIZE_SCALE>>,
        Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<C0_SIZE_SCALE>>>;
    using MakeLayoutScaleB = Std::conditional_t<
        TRANS_B, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<C0_SIZE_SCALE>>,
        Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<C0_SIZE_SCALE>>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE_C>>;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, BiasType, LayoutBias>;

    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutAType = decltype(MakeLayoutA{}(uint32_t {}, uint32_t {}));
    using LayoutBType = decltype(MakeLayoutB{}(uint32_t {}, uint32_t {}));
    using LayoutScaleAType = decltype(MakeLayoutScaleA{}(uint32_t {}, uint32_t {}));
    using LayoutScaleBType = decltype(MakeLayoutScaleB{}(uint32_t {}, uint32_t {}));
    using LayoutCType = decltype(MakeLayoutC{}(uint32_t {}, uint32_t {}));
    using LayoutBiasType = decltype(Te::MakeFrameLayout<LayoutBias>(uint32_t {}, uint32_t {}));

    struct ProblemConfig {
        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        uint32_t outputN = 0;
        uint32_t blockNum = 0;
        uint32_t blockIdx = 0;
        uint32_t scaleK = 0;
        uint32_t tileM = 0;
        typename BlockMmad::L1Params l1Params{
            .kL1 = L1_TILE_K, .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE, .l1BufNum = 2};
    };

    struct LayoutBundle {
        LayoutAType a;
        LayoutBType b;
        LayoutScaleAType scaleA;
        LayoutScaleBType scaleB;
        LayoutCType c;
        LayoutBiasType bias;
    };

    __aicore__ static inline ProblemConfig BuildProblemConfig(const ProblemShape& problemShape)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        if constexpr (Policy::IS_GMM1) {
            config.n = Get<N_VALUE>(problemShape);
            config.k = Get<K_VALUE>(problemShape);
            config.outputN = config.n / SWIGLU_N_HALF;
            config.tileM = L1_TILE_M_256;
        } else {
            config.n = Get<K_VALUE>(problemShape);
            config.k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;
            config.outputN = config.n;
            config.tileM = (CombineQuantMode == COMBINE_NO_QUANT) ? L1_TILE_M_128 : L1_TILE_M_256;
        }
        config.blockNum = GetBlockNum();
        config.blockIdx = GetBlockIdx() / GetTaskRation();
        config.scaleK = CeilDiv(config.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        return config;
    }

    __aicore__ static inline LayoutBundle BuildLayouts(const ProblemConfig& config)
    {
        LayoutBundle layouts;
        layouts.a = MakeLayoutA{}(config.m, config.k);
        layouts.b = MakeLayoutB{}(config.k, config.n);
        layouts.scaleA = MakeLayoutScaleA{}(config.m, config.scaleK);
        layouts.scaleB = MakeLayoutScaleB{}(config.scaleK, config.n);
        layouts.bias = Te::MakeFrameLayout<LayoutBias>(1U, config.n);
        if constexpr (Policy::IS_GMM1) {
            layouts.c = MakeLayoutC{}(L1_TILE_M_256, L1_TILE_N);
        } else {
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                layouts.c = MakeLayoutC{}(L1_TILE_M_128, L1_TILE_N);
            } else {
                layouts.c = MakeLayoutC{}(config.m, config.n);
            }
        }
        return layouts;
    }
};

template <typename Policy, typename Config>
__aicore__ inline void WaitForUpstreamA8W8(
    const GMMAddrInfo& gmmAddrInfo, const Config& config, uint32_t mLoc)
{
    if constexpr (Policy::IS_GMM1) {
        uint32_t waveIdx = mLoc / L1_TILE_M_256;
        uint32_t targetValue = (mLoc + L1_TILE_M_256 > config.m) ? (config.m - mLoc) : L1_TILE_M_256;
        __gm__ int32_t* flagValueAddr = gmmAddrInfo.dispatchToGmm1Flag + waveIdx;
        while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    } else {
        BlockScheduler gmmBlockScheduler(
            {config.m, config.k, config.n},
            BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M_256),
                static_cast<int64_t>(L1_TILE_N))});
        uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
        __gm__ int32_t* flagValueAddr = gmmAddrInfo.swigluToGmm2Flag;
        while (targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    }
}

template <typename BlockMmad, typename TensorA, typename TensorScaleA, typename TensorB,
    typename TensorScaleB, typename TensorBias, typename TensorC, typename BlockShape, typename Config>
__aicore__ inline void AicGmm1ComputeA8W8(
    BlockMmad& blockMmad, TensorA& gmA, TensorScaleA& gmScaleA, TensorB& gmB, TensorScaleB& gmScaleB,
    TensorBias& gmBias, TensorC& l0cOutUbFirst, TensorC& l0cOutUbSecond,
    const GMMAddrInfo& gmmAddrInfo, const Config& config,
    uint32_t mLoc, uint32_t nLoc, uint32_t kLoc, BlockShape actualShape,
    int32_t vecSetSyncCom, uint32_t& lastWaveWaited)
{
    uint32_t waveIdx = mLoc / L1_TILE_M_256;
    if (waveIdx != lastWaveWaited) {
        WaitForUpstreamA8W8<Gmm1PolicyA8W8>(gmmAddrInfo, config, mLoc);
        lastWaveWaited = waveIdx;
    }

    if (vecSetSyncCom) {
        WaitForVector();
    }

    auto gmBlockA = gmA.Slice(
        Te::MakeCoord(mLoc, kLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
    auto gmBlockScaleA = gmScaleA.Slice(
        Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
        Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));

    auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
        Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
    auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
        Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));

    typename BlockMmad::BlockShape singleShape{
        Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), Get<K_VALUE>(actualShape), 0};
    for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
        auto gmBlockB = gmB.Slice(
            Te::MakeCoord(kLoc, nLoc + weightBlock * config.outputN),
            Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB = gmScaleB.Slice(
            Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * config.outputN),
            Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
        blockMmad(
            gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
            weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
    }

    NotifyVector();
}

template <typename SwigluQuantOp, typename BlockShape, typename Config>
__aicore__ inline void AivGmm1PostA8W8(
    SwigluQuantOp& epilogueOp, uint32_t mLoc, uint32_t nLoc,
    BlockShape actualShape, const Config& config)
{
    Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{
        Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), 0, 0};
    Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
        mLoc * config.outputN + nLoc,
        mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
            CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
        0, 0, 0, 0};
    WaitForCube();
    AscendC::SetCtrlSpr<60, 60>(0);
    epilogueOp(epilogueShape, epilogueOffset);
    NotifyCube();
}

template <uint8_t CombineQuantMode, typename ElementC, typename BlockMmad, typename TensorA,
    typename TensorScaleA, typename TensorB, typename TensorScaleB, typename TensorBias,
    typename TensorC, typename BlockShape, typename LayoutCType, typename Config>
__aicore__ inline void AicGmm2ComputeA8W8(
    BlockMmad& blockMmad, TensorA& gmA, TensorScaleA& gmScaleA, TensorB& gmB, TensorScaleB& gmScaleB,
    TensorBias& gmBias, TensorC& l0cOutUbFirst, TensorC& l0cOutUbSecond,
    const GMMAddrInfo& gmmAddrInfo, const Params& params, const Config& config,
    uint32_t loopIdx, uint32_t startLoopIdx,
    uint32_t mLoc, uint32_t nLoc, uint32_t kLoc, BlockShape actualShape,
    int32_t vecSetSyncCom2, uint16_t pingpongIdx, uint32_t groupIdx, const LayoutCType& layoutC)
{
    if (loopIdx == startLoopIdx) {
        WaitForUpstreamA8W8<Gmm2PolicyA8W8>(gmmAddrInfo, config, mLoc);
    }

    auto gmBlockA = gmA.Slice(
        Te::MakeCoord(mLoc, kLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
    auto gmBlockScaleA = gmScaleA.Slice(
        Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
        Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));
    auto gmBlockB = gmB.Slice(
        Te::MakeCoord(kLoc, nLoc), Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
    auto gmBlockScaleB = gmScaleB.Slice(
        Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
        Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));

    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        if (vecSetSyncCom2 >= 2) {
            WaitForVector(pingpongIdx);
        }
        auto tensorUb = pingpongIdx == 0 ? l0cOutUbFirst : l0cOutUbSecond;
        auto tensorBlockUb = tensorUb.Slice(
            Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        
        typename BlockMmad::BlockShape singleShape{
            Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), Get<K_VALUE>(actualShape), 0};
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockUb, singleShape);
        NotifyVector(pingpongIdx);
    } else {
        uint32_t blockAivNum = config.blockNum * 2;
        GlobalTensor<int32_t> gmm2CombineSyncCounter;
        gmm2CombineSyncCounter.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.gmm2CombineSyncCounterPtr);
        auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
            reinterpret_cast<__gm__ ElementC*>(gmmAddrInfo.gmm2OutGlobal)), layoutC);
        auto gmBlockC = gmC.Slice(
            Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        
        typename BlockMmad::BlockShape singleShape{
            Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), Get<K_VALUE>(actualShape), 0};
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, gmBlockC, singleShape);
        SetFlag<HardEvent::FIX_S>(0);
        WaitFlag<HardEvent::FIX_S>(0);

        constexpr uint32_t TILE_M = L1_TILE_M_256;
        uint32_t tokenGroupIdx = mLoc / TILE_M;
        uint32_t tokenGroupsThisExpert = Ops::Base::CeilDiv(config.m, TILE_M);
        uint32_t grpCoreStart, grpCoreSize;
        ComputeGroupRange(tokenGroupIdx, tokenGroupsThisExpert, blockAivNum, grpCoreStart, grpCoreSize);
        int64_t baseOffset = static_cast<int64_t>(groupIdx) * blockAivNum * INT_CACHELINE;
        for (uint32_t aivIdx = grpCoreStart; aivIdx < grpCoreStart + grpCoreSize; aivIdx++) {
            AscendC::AtomicAdd((__gm__ int32_t*)gmm2CombineSyncCounter.GetPhyAddr()
                + baseOffset + aivIdx * INT_CACHELINE, int32_t(1));
        }
    }
}

template <typename ElementC, typename BlockShape, typename Config>
__aicore__ inline void AivGmm2PostA8W8(
    const Params& params, uint32_t mLoc, uint32_t nLoc, uint32_t groupCnt,
    uint16_t pingpongIdx, BlockShape actualShape, const Config& config,
    LocalTensor<ElementC>& l0cOutUbGMM2First, LocalTensor<ElementC>& l0cOutUbGMM2Second)
{
    auto l0cOutUbGMM2 = pingpongIdx == 0 ? l0cOutUbGMM2First : l0cOutUbGMM2Second;
    WaitForCube(pingpongIdx);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::GlobalTensor<int32_t> tripleGm;
    int32_t lenTile = Get<M_VALUE>(actualShape);
    LocalTensor<int32_t> tripleTensor = LocalTensor<int32_t>(
        TPosition::VECCALC, TRIPLE_TENSOR_ADDR, lenTile * TRIPLE_SIZE);
    tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.tripleInfoPtr +
        (groupCnt + mLoc) * TRIPLE_SIZE * sizeof(int32_t)));
    AscendC::DataCopy(tripleTensor, tripleGm, lenTile * TRIPLE_SIZE);
    MegaMoeCombineImpl::CombineTokens<ElementC, BlockShape>(
        mLoc, nLoc, config.n, tripleTensor, l0cOutUbGMM2, actualShape, params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    NotifyCube(pingpongIdx);
}

} // namespace Detail
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmulSwigluQuant(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true>& epilogueOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom)
{
    using Config = Detail::ConfigA8W8<Detail::Gmm1PolicyA8W8, COMBINE_NO_QUANT, ElementA, ElementB, ElementC,
        ElementMxScaleA, ElementMxScaleB>;
    auto config = Config::BuildProblemConfig(problemShape);
    auto layouts = Config::BuildLayouts(config);

    uint32_t m = config.m;
    uint32_t n = config.n;
    uint32_t k = config.k;
    uint32_t outputN = config.outputN;
    uint32_t blockNum = config.blockNum;
    uint32_t blockIdx = config.blockIdx;

    epilogueOp.UpdateNextProblem({m, outputN, k, 0});

    using BlockMmad = typename Config::BlockMmad;
    using BiasType = typename Config::BiasType;
    BlockMmad blockMmad;
    bool enableL0CPingPong = false;
    typename BlockMmad::BlockShape l0TileShape{L1_TILE_M_256, L1_TILE_N, L0_TILE_K, 0};
    typename BlockMmad::ProblemShape matmulShape{m, n, k, 0};
    blockMmad.Init(matmulShape, l0TileShape, config.l1Params, false, enableL0CPingPong);

    int64_t ubOffset = 0;
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layouts.c);

    ubOffset += MAX_SINGLE_MN_ALIGN32_NUM_256 * sizeof(ElementC);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layouts.c);

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

    BlockScheduler scheduler(
        {m, outputN, k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M_256), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;

    // wave-grain dispatch flag: 每 wave (L1_TILE_M_256 行) 一个槽,dispatch 完成的行数累加到该槽。
    // 目标值 = 该 wave 实际行数 = min(L1_TILE_M_256, m - mLoc) (尾 wave 可能 < L1_TILE_M_256)。
    // 仅在 wave 切换时等待;跨 nLoc 不同的同 mLoc tile 复用上次等待结果。
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        if constexpr (g_coreType == AscendC::AIC) {
            Detail::AicGmm1ComputeA8W8(
                blockMmad, gmA, gmScaleA, gmB, gmScaleB, gmBias, l0cOutUbFirst, l0cOutUbSecond,
                gmmAddrInfo, config, mLoc, nLoc, kLoc, actualShape, vecSetSyncCom, lastWaveWaited);
        }

        vecSetSyncCom = 1;

        if constexpr (g_coreType == AscendC::AIV) {
            Detail::AivGmm1PostA8W8(epilogueOp, mLoc, nLoc, actualShape, config);
        }
    }
    startBlockIdx = (startBlockIdx + tileNum) % blockNum;
}

// =================================================================================================
// GroupMatmul2：GMM2 矩阵乘法，支持量化和非量化模式
// =================================================================================================
// 量化模式：AIC 计算结果写入 GM，通过 AtomicAdd 通知 AIV；AIV 不参与计算
// 非量化模式：AIC 计算结果写入 UB，通过 pingpong 双缓冲通知 AIV；AIV 执行 CombineTokens
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmul2(
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
    int32_t& vecSetSyncCom2, uint32_t groupCnt, uint16_t& pingpongIdx, uint32_t groupIdx)
{
    // 非量化模式：仅 subBlockIdx_==0 的核参与
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        if (GetSubBlockIdx() != 0) return;
    }

    using Config = Detail::ConfigA8W8<Detail::Gmm2PolicyA8W8, CombineQuantMode, ElementA, ElementB, ElementC,
        ElementMxScaleA, ElementMxScaleB>;
    auto config = Config::BuildProblemConfig(problemShape);
    auto layouts = Config::BuildLayouts(config);

    uint32_t m = config.m;
    uint32_t n = config.n;
    uint32_t k = config.k;
    uint32_t blockNum = config.blockNum;
    uint32_t blockIdx = config.blockIdx;
    constexpr uint32_t TILE_M = (CombineQuantMode == COMBINE_NO_QUANT) ? L1_TILE_M_128 : L1_TILE_M_256;

    using BlockMmad = typename Config::BlockMmad;
    using BiasType = typename Config::BiasType;
    BlockMmad blockMmad;
    bool enableL0CPingPong = false;
    typename BlockMmad::BlockShape l0TileShape{TILE_M, L1_TILE_N, L0_TILE_K, 0};
    typename BlockMmad::ProblemShape matmulShape{m, n, k, 0};
    blockMmad.Init(matmulShape, l0TileShape, config.l1Params, false, enableL0CPingPong);

    // 非量化模式：分配 UB pingpong buffer
    LocalTensor<ElementC> l0cOutUbGMM2First, l0cOutUbGMM2Second;
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(0), layouts.c);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(0), layouts.c);
    
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        int64_t ubOffset = 0;
        l0cOutUbGMM2First = LocalTensor<ElementC>(TPosition::VECIN, ubOffset, L1_TILE_M_128 * L1_TILE_N);
        l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layouts.c);
        ubOffset += MAX_SINGLE_MN_ALIGN32_NUM_128 * sizeof(ElementC);
        l0cOutUbGMM2Second = LocalTensor<ElementC>(TPosition::VECIN, ubOffset, L1_TILE_M_128 * L1_TILE_N);
        l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layouts.c);
    }

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

    BlockScheduler scheduler(
        {m, n, k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(TILE_M), static_cast<int64_t>(L1_TILE_N))});
    uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;
    uint32_t tileNum = scheduler.GetTileNum();

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        if constexpr (g_coreType == AscendC::AIC) {
            Detail::AicGmm2ComputeA8W8<CombineQuantMode, ElementC>(
                blockMmad, gmA, gmScaleA, gmB, gmScaleB, gmBias, l0cOutUbFirst, l0cOutUbSecond,
                gmmAddrInfo, params, config, loopIdx, startLoopIdx,
                mLoc, nLoc, kLoc, actualShape, vecSetSyncCom2, pingpongIdx, groupIdx, layouts.c);
        }

        // 非量化模式：AIV CombineTokens + pingpong 切换
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            vecSetSyncCom2++;
            if constexpr (g_coreType == AscendC::AIV) {
                Detail::AivGmm2PostA8W8(
                    params, mLoc, nLoc, groupCnt, pingpongIdx, actualShape, config,
                    l0cOutUbGMM2First, l0cOutUbGMM2Second);
            }
            pingpongIdx = 1 - pingpongIdx;
        }
    }
    startBlockIdx = (startBlockIdx + tileNum) % blockNum;
}

// ==================================================================================
// A8W4 execution path — shared skeleton, policy-based dispatch for GMM1 / GMM2.
// ==================================================================================
namespace Detail {
struct Gmm1PolicyA8W4 {
    static constexpr bool IS_GMM1 = true;
};

struct Gmm2PolicyA8W4 {
    static constexpr bool IS_GMM1 = false;
};

template <typename SwigluQuantOp>
struct Gmm1ArgsA8W4 {
    SwigluQuantOp& swigluQuantOp;
};

struct Gmm2ArgsA8W4 {
    uint32_t groupCnt;
    uint16_t& pingpongIdx;
};

template <typename Policy, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
    typename ElementMxScaleB>
struct ConfigA8W4 {
    static constexpr uint32_t C0_SIZE_A = AuxGetC0Size<ElementA>();
    static constexpr uint32_t C0_SIZE_B = 32U;
    static constexpr uint32_t C0_SIZE_C = AuxGetC0Size<ElementC>();
    static constexpr uint32_t C0_SIZE_SCALE = 2U;

    using DispatchPolicy = Blaze::Gemm::MatmulMxFp8Fp4DynamicKL1TailResplit;
    using BlockPrologue = Blaze::Gemm::Prologue::BlockPrologue<DispatchPolicy, ElementA, ElementB>;
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::ZNLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutScaleA = AscendC::Te::ScaleANDLayoutPtn;
    using LayoutScaleB = AscendC::Te::ScaleBDNLayoutPtn;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE_A>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE_B>>;
    using MakeLayoutScaleA = Te::FrameLayoutFormat<LayoutScaleA, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutScaleB = Te::FrameLayoutFormat<LayoutScaleB, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE_C>>;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AscendC::Std::tuple<ElementA, ElementMxScaleA>,
        AscendC::Std::tuple<MakeLayoutA, MakeLayoutScaleA>, AscendC::Std::tuple<ElementB, ElementMxScaleB>,
        AscendC::Std::tuple<MakeLayoutB, MakeLayoutScaleB>, ElementC, MakeLayoutC, void, void>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutAType = decltype(MakeLayoutA{}(uint32_t {}, uint32_t {}));
    using LayoutBType = decltype(MakeLayoutB{}(uint32_t {}, uint32_t {}));
    using LayoutScaleAType = decltype(MakeLayoutScaleA{}(uint32_t {}, uint32_t {}));
    using LayoutScaleBType = decltype(MakeLayoutScaleB{}(uint32_t {}, uint32_t {}));
    using LayoutCType = decltype(MakeLayoutC{}(uint32_t {}, uint32_t {}));

    struct ProblemConfig {
        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        uint32_t outputN = 0;
        uint32_t blockNum = 0;
        uint32_t blockIdx = 0;
        uint32_t subBlockIdx = 0;
        uint32_t scaleK = 0;
        typename BlockMmad::L1Params l1Params{.kL1 = L1_TILE_K, .scaleKL1 = 4096};
    };

    struct LayoutBundle {
        LayoutAType a;
        LayoutBType b;
        LayoutScaleAType scaleA;
        LayoutScaleBType scaleB;
        LayoutCType c;
    };

    __aicore__ static inline ProblemConfig BuildProblemConfig(const ProblemShape& problemShape)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        if constexpr (Policy::IS_GMM1) {
            config.n = Get<N_VALUE>(problemShape);
            config.k = Get<K_VALUE>(problemShape);
        } else {
            config.n = Get<K_VALUE>(problemShape);
            config.k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;
        }
        config.outputN = Policy::IS_GMM1 ? config.n / SWIGLU_N_HALF : config.n;
        config.blockNum = GetBlockNum();
        config.blockIdx = GetBlockIdx() / GetTaskRation();
        config.subBlockIdx = GetSubBlockIdx();
        config.scaleK = CeilDiv(config.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        return config;
    }

    __aicore__ static inline LayoutBundle BuildLayouts(const ProblemConfig& config)
    {
        return LayoutBundle{
            MakeLayoutA{}(config.m, config.k),
            MakeLayoutB{}(config.k, config.n),
            MakeLayoutScaleA{}(config.m, config.scaleK),
            MakeLayoutScaleB{}(config.scaleK, config.n),
            MakeLayoutC{}(config.m, config.n)};
    }
};

template <typename Policy, typename Config>
__aicore__ inline void WaitForUpstreamReadyA8W4(
    const GMMAddrInfo& gmmAddrInfo, const Config& config, uint32_t mLoc)
{
    if constexpr (Policy::IS_GMM1) {
        uint32_t waveIdx = mLoc / L1_TILE_M_256;
        uint32_t targetValue = (mLoc + L1_TILE_M_256 > config.m) ? (config.m - mLoc) : L1_TILE_M_256;
        __gm__ int32_t* flagValueAddr = gmmAddrInfo.dispatchToGmm1Flag + waveIdx;
        while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    } else {
        BlockScheduler gmmBlockScheduler(
            {config.m, config.k, config.n},
            BlockScheduler::Params{
                Te::MakeCoord(static_cast<int64_t>(L1_TILE_M_256), static_cast<int64_t>(L1_TILE_N))});
        uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
        __gm__ int32_t* flagValueAddr = gmmAddrInfo.swigluToGmm2Flag;
        while (targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    }
}

template <typename Policy, typename BlockMmad, typename Scheduler, typename TensorA, typename TensorScaleA,
    typename TensorScaleB, typename TensorC, typename Config>
__aicore__ inline void AicComputeA8W4(
    BlockMmad& blockMmad, Scheduler& scheduler, TensorA& gmA, TensorScaleA& gmScaleA, TensorScaleB& gmScaleB,
    TensorC& l0cOutGm, const GMMAddrInfo& gmmAddrInfo, const Config& config,
    uint32_t startLoopIdx, uint32_t tileNum)
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
            WaitForUpstreamReadyA8W4<Policy>(gmmAddrInfo, config, mLoc);
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
                    gmScaleB.Slice(
                        Te::MakeCoord(0, nOffset), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
                auto tensorBlockGm = l0cOutGm.Slice(
                    Te::MakeCoord(mLoc, nOffset),
                    Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
            }
        } else {
            auto gmBlockScaleB =
                gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
            auto tensorBlockGm = l0cOutGm.Slice(
                Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
        }
        NotifyVectorToCopyIn();
    }
}

template <typename Policy, typename BlockPrologue, typename Scheduler, typename TensorB, typename Config>
__aicore__ inline void AivPrologueA8W4(BlockPrologue& blockPrologue, Scheduler& scheduler, TensorB& gmB,
    const Config& config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        auto mL1Size = Get<M_VALUE>(actualShape);
        auto nL1Size = Get<N_VALUE>(actualShape);

        if constexpr (Policy::IS_GMM1) {
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto nOffset = nLoc + weightBlock * config.outputN;
                blockPrologue(gmB, mL1Size, config.k, nL1Size, nOffset, config.n, config.l1Params.kL1);
            }
        } else {
            blockPrologue(gmB, mL1Size, config.k, nL1Size, nLoc, config.n, config.l1Params.kL1);
        }
    }
}

template <typename ElementC, typename MakeLayoutC, typename Scheduler, typename TensorC, typename SwigluQuantOp,
    typename Config>
__aicore__ inline void AivGmm1PostA8W4(SwigluQuantOp& swigluQuantOp, Scheduler& scheduler,
    TensorC& l0cOutGm, const Config& config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        WaitForCubeFinishCopyout();
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
        auto tensorBlockGmFirst = l0cOutGm.Slice(
            Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto tensorBlockGmSecond = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc + config.outputN),
            Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));

        auto layoutL0cUB = MakeLayoutC{}(L1_TILE_M_256, L1_TILE_N);
        int64_t ubOffsetFirst = 0;
        int64_t ubOffsetSecond = ubOffsetFirst + MAX_SINGLE_MN_ALIGN32_NUM_256 * sizeof(ElementC);
        auto tensorBlockUbFirst =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), layoutL0cUB);
        auto tensorBlockUbSecond =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), layoutL0cUB);
        auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGM2UB, tensorBlockUbFirst, tensorBlockGmFirst);
        AscendC::Te::Copy(copyGM2UB, tensorBlockUbSecond, tensorBlockGmSecond);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape),
                                                                     Get<N_VALUE>(actualShape), 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            mLoc * config.outputN + nLoc,
            mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            0, 0, 0, 0};

        AscendC::SetCtrlSpr<60, 60>(0);
        swigluQuantOp(epilogueShape, epilogueOffset);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
    }
}

template <typename ElementC, typename MakeLayoutC, typename Scheduler, typename TensorC, typename Config>
__aicore__ inline void AivGmm2PostA8W4(Scheduler& scheduler, TensorC& l0cOutGm, const Params& params,
    uint32_t groupCnt, const Config& config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        WaitForCubeFinishCopyout();
        auto tensorBlockGm = l0cOutGm.Slice(
            Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto layoutL0cUB = MakeLayoutC{}(L1_TILE_M_256, L1_TILE_N);
        int64_t ubOffset = 0;
        auto tensorBlockUb = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutL0cUB);
        LocalTensor<ElementC> l0cOutUbGMM2 =
            LocalTensor<ElementC>(TPosition::VECIN, ubOffset, L1_TILE_M_256 * L1_TILE_N);
        auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGM2UB, tensorBlockUb, tensorBlockGm);

        AscendC::GlobalTensor<int32_t> tripleGm;
        int32_t lenTile = Get<M_VALUE>(actualShape);
        LocalTensor<int32_t> tripleTensor = LocalTensor<int32_t>(TPosition::VECCALC, 200 * 1024, lenTile * 8);
        tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.tripleInfoPtr +
            (groupCnt + mLoc) * 32));
        AscendC::DataCopy(tripleTensor, tripleGm, lenTile * 8);
        MegaMoeCombineImpl::CombineTokens<ElementC, decltype(actualShape)>(
            mLoc, nLoc, config.n, tripleTensor, l0cOutUbGMM2, actualShape, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }
}

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

template <typename Policy, typename BlockMmad, typename BlockPrologue, typename ElementC, typename MakeLayoutC,
    typename WorkSet, typename Config, typename ExtraArgs>
__aicore__ inline void GroupMatmulExecA8W4(WorkSet& workSet, const Params& params, const GMMAddrInfo& gmmAddrInfo,
    const Config& config, uint32_t startLoopIdx, uint32_t tileNum, ExtraArgs& args)
{
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad{};
        typename BlockMmad::BlockShape l0TileShape{L1_TILE_M_256, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.l1Params);
        AicComputeA8W4<Policy>(blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB,
            workSet.l0cOutGm, gmmAddrInfo, config, startLoopIdx, tileNum);
    } else {
        if (config.subBlockIdx == 0) {
            BlockPrologue blockPrologue;
            AivPrologueA8W4<Policy>(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx,
                tileNum);
        } else {
            if constexpr (Policy::IS_GMM1) {
                AivGmm1PostA8W4<ElementC, MakeLayoutC>(
                    args.swigluQuantOp, workSet.scheduler, workSet.l0cOutGm, config, startLoopIdx, tileNum);
            } else {
                AivGmm2PostA8W4<ElementC, MakeLayoutC>(
                    workSet.scheduler, workSet.l0cOutGm, params, args.groupCnt, config,
                    startLoopIdx, tileNum);
            }
        }
    }
}

template <typename Policy, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
    typename ElementMxScaleB, typename ExtraArgs>
__aicore__ inline void GroupMatmulImplA8W4(const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape, const GMMAddrInfo& gmmAddrInfo,
    uint32_t& startBlockIdx, ExtraArgs& args)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using Config = ConfigA8W4<Policy, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB>;
    auto config = Config::BuildProblemConfig(problemShape);

    if constexpr (Policy::IS_GMM1) {
        args.swigluQuantOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    auto layouts = Config::BuildLayouts(config);
    using BlockMmad = typename Config::BlockMmad;
    using BlockPrologue = typename Config::BlockPrologue;
    using MakeLayoutC = typename Config::MakeLayoutC;

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
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M_256), static_cast<int64_t>(L1_TILE_N))});
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
    GroupMatmulExecA8W4<Policy, BlockMmad, BlockPrologue, ElementC, MakeLayoutC>(workSet, params, gmmAddrInfo,
        config, startLoopIdx, tileNum, args);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}
} // namespace Detail

// GroupMatmulSwigluQuantA8W4 — A8W4 prologue (W4→W8) + GMM1 + SwiGLU + Quant
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmulSwigluQuantA8W4(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true>& swigluQuantOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom)
{
    (void)vecSetSyncCom;
    using SwigluQuantOpType = std::remove_reference_t<decltype(swigluQuantOp)>;
    Detail::Gmm1ArgsA8W4<SwigluQuantOpType> args{swigluQuantOp};
    Detail::GroupMatmulImplA8W4<Detail::Gmm1PolicyA8W4, ElementA, ElementB, ElementC, ElementMxScaleA,
        ElementMxScaleB>(params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

// GroupMatmul2CombineA8W4 — A8W4 prologue (W4→W8) + GMM2 + Combine
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmul2CombineA8W4(
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom2, uint32_t groupCnt,
    uint16_t& pingpongIdx)
{
    (void)vecSetSyncCom2;
    Detail::Gmm2ArgsA8W4 args{groupCnt, pingpongIdx};
    Detail::GroupMatmulImplA8W4<Detail::Gmm2PolicyA8W4, ElementA, ElementB, ElementC, ElementMxScaleA,
        ElementMxScaleB>(params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

} // namespace MegaMoeImpl

#endif