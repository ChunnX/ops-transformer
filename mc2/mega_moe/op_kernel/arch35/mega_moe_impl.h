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

__aicore__ inline int64_t ComputeMaxRowGroupsStride(uint64_t maxOutputSize, uint32_t blockAivNum, uint32_t expertPerRank)
{
    int64_t maxTokensPerExpert = Ops::Base::CeilDiv(
        static_cast<int64_t>(maxOutputSize), static_cast<int64_t>(expertPerRank));
    int64_t maxRowGroupsPerExpert = Ops::Base::CeilDiv(maxTokensPerExpert, static_cast<int64_t>(L1_TILE_M_128));
    int64_t perCoreRowGroupsStride = Ops::Base::CeilAlign(
        static_cast<int64_t>(blockAivNum) * INT_CACHELINE, ALIGN_128);
    return Ops::Base::CeilAlign(maxRowGroupsPerExpert * perCoreRowGroupsStride, ALIGN_128);
}

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
    uint32_t& groupCoreStart, uint32_t& groupCoreSize)
{
    uint32_t baseSize = totalCores / numGroups;
    uint32_t remainder = totalCores % numGroups;
    if (groupIdx < remainder) {
        groupCoreSize = baseSize + 1;
        groupCoreStart = groupIdx * (baseSize + 1);
    } else {
        groupCoreSize = baseSize;
        groupCoreStart = remainder * (baseSize + 1) + (groupIdx - remainder) * baseSize;
    }
}

// GroupMatmulSwigluQuant - GMM1 + SwiGLU + Quant
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmulSwigluQuant(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true>& epilogueOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom)
{
    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = Get<N_VALUE>(problemShape);
    uint32_t k = Get<K_VALUE>(problemShape);
    uint32_t outputN = n / SWIGLU_N_HALF;

    epilogueOp.UpdateNextProblem({m, outputN, k, 0});

    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx() / GetTaskRation();

    GlobalTensor<int32_t> groupFlagListGm2;
    groupFlagListGm2.SetGlobalBuffer((__gm__ int32_t*)gmmAddrInfo.groupFlagList2);

    auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

    constexpr bool transA = false;
    constexpr bool transB = true; // 后续要改为从模板参数获取转置属性

    constexpr uint32_t c0SizeA = AuxGetC0Size<ElementA>();
    constexpr uint32_t c0SizeB = AuxGetC0Size<ElementB>();
    constexpr uint32_t c0SizeC = AuxGetC0Size<ElementC>();
    constexpr uint32_t c0SizeScale = 2U;

    using LayoutA = Std::conditional_t<transA, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutB = Std::conditional_t<transB, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutBias = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<c0SizeA>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<c0SizeB>>;
    using MakeLayoutScaleA = Std::conditional_t<
        transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutScaleB = Std::conditional_t<
        transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<c0SizeC>>;

    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleK);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleK, n);
    auto layoutBias = Te::MakeFrameLayout<LayoutBias>(1u, n); // block mmad需要传入bias
    auto layoutC = MakeLayoutC{}(L1_TILE_M_256, L1_TILE_N);

    using BiasType = float;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, BiasType, LayoutBias>;
    BlockMmad blockMmad;
    bool enableL0CPingPong = false;
    typename BlockMmad::L1Params l1Params{
        .kL1 = L1_TILE_K,
        .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE,
        .l1BufNum = 2 // 2: double buffer
    };
    typename BlockMmad::BlockShape l0TileShape{L1_TILE_M_256, L1_TILE_N, L0_TILE_K, 0};
    typename BlockMmad::ProblemShape matmulShape{m, n, k, 0};
    blockMmad.Init(matmulShape, l0TileShape, l1Params, false, enableL0CPingPong); // 当前固定无bias

    int64_t ubOffset = 0;
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);

    ubOffset += MAX_SINGLE_MN_ALIGN32_NUM_256 * sizeof(ElementC);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);

    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layoutA);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layoutB);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
        layoutScaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
        layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layoutBias);

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
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                uint32_t targetValue = (mLoc + L1_TILE_M_256 > m) ? (m - mLoc) : L1_TILE_M_256;
                __gm__ int32_t* flagValueAddr =
                    (__gm__ int32_t*)groupFlagListGm2.GetPhyAddr() + waveIdx;
                while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {
                    }
                }
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
                    Te::MakeCoord(kLoc, nLoc + weightBlock * outputN),
                    Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                auto gmBlockScaleB = gmScaleB.Slice(
                    Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * outputN),
                    Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
                blockMmad(
                    gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                    weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
            }

            NotifyVector();
        }

        vecSetSyncCom = 1;

        if constexpr (g_coreType == AscendC::AIV) {
            Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{
                Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), 0, 0};
            Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                mLoc * outputN + nLoc,
                mLoc * CeilDiv(outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                    CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                0, 0, 0, 0};
            WaitForCube();
            AscendC::SetCtrlSpr<60, 60>(0);
            epilogueOp(epilogueShape, epilogueOffset);
            NotifyCube();
        }
    }
    startBlockIdx = (startBlockIdx + tileNum) % blockNum;
}

// GroupMatmul2 - GMM2矩阵乘法，支持量化和非量化模式
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, 
          typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmul2(
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
    int32_t& vecSetSyncCom2, uint32_t groupCnt, uint16_t& pingpongIdx,
    uint32_t groupIdx, int64_t maxRowGroupsStride)
{
    // 非量化模式：仅 subBlockIdx_==0 的核参与
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        if (GetSubBlockIdx() != 0) return;
    }

    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = Get<K_VALUE>(problemShape);
    uint32_t k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;

    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx() / GetTaskRation();

    GlobalTensor<int32_t> groupFlagListGm;
    groupFlagListGm.SetGlobalBuffer((__gm__ int32_t*)gmmAddrInfo.groupFlagList);

    auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

    constexpr bool transA = false;
    constexpr bool transB = true;

    constexpr uint32_t c0SizeA = AuxGetC0Size<ElementA>();
    constexpr uint32_t c0SizeB = AuxGetC0Size<ElementB>();
    constexpr uint32_t c0SizeC = AuxGetC0Size<ElementC>();
    constexpr uint32_t c0SizeScale = 2U;

    using LayoutA = Std::conditional_t<transA, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutB = Std::conditional_t<transB, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutBias = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<c0SizeA>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<c0SizeB>>;
    using MakeLayoutScaleA = Std::conditional_t<
        transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutScaleB = Std::conditional_t<
        transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<c0SizeC>>;

    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleK);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleK, n);
    auto layoutBias = Te::MakeFrameLayout<LayoutBias>(1U, n);
    
    // 量化模式使用全矩阵 layout，非量化使用 tile 级 layout
    auto layoutC = MakeLayoutC{}(
        (CombineQuantMode == COMBINE_NO_QUANT) ? L1_TILE_M_128 : m,
        (CombineQuantMode == COMBINE_NO_QUANT) ? L1_TILE_N : n);

    using BiasType = float;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, BiasType, LayoutBias>;
    BlockMmad blockMmad;
    bool enableL0CPingPong = false;
    typename BlockMmad::L1Params l1Params{
        .kL1 = L1_TILE_K,
        .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE,
        .l1BufNum = 2
    };
    typename BlockMmad::BlockShape l0TileShape{L1_TILE_M_128, L1_TILE_N, L0_TILE_K, 0};
    typename BlockMmad::ProblemShape matmulShape{m, n, k, 0};
    blockMmad.Init(matmulShape, l0TileShape, l1Params, false, enableL0CPingPong);

    // 非量化模式：分配 UB pingpong buffer
    LocalTensor<ElementC> l0cOutUbGMM2First, l0cOutUbGMM2Second;
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(0), layoutC);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(0), layoutC);
    
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        int64_t ubOffset = 0;
        l0cOutUbGMM2First = LocalTensor<ElementC>(TPosition::VECIN, ubOffset, L1_TILE_M_128 * L1_TILE_N);
        l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);
        ubOffset += MAX_SINGLE_MN_ALIGN32_NUM_128 * sizeof(ElementC);
        l0cOutUbGMM2Second = LocalTensor<ElementC>(TPosition::VECIN, ubOffset, L1_TILE_M_128 * L1_TILE_N);
        l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);
    }

    // 量化模式：创建 rowGroupComplete
    GlobalTensor<int32_t> rowGroupComplete;
    uint32_t blockAivNum = blockNum * 2;

    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layoutA);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layoutB);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
        layoutScaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
        layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layoutBias);

    BlockScheduler scheduler(
        {m, n, k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M_128), static_cast<int64_t>(L1_TILE_N))});
    uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;
    uint32_t tileNum = scheduler.GetTileNum();

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        if constexpr (g_coreType == AscendC::AIC) {
            // 公共：GMM flag wait
            if (loopIdx == startLoopIdx) {
                BlockScheduler gmmBlockScheduler(
                    {m, k, n},
                    BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M_256),
                        static_cast<int64_t>(L1_TILE_N))});
                uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
                __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm.GetPhyAddr();
                while (targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {
                    }
                }
            }

            // 公共：切片 A/B/Scale
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
                // 非量化：输出到 UB
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
                // 量化：输出到 GM
                rowGroupComplete.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.rowGroupCompletePtr);
                auto gmC = Te::MakeTensor(
                    Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC*>(gmmAddrInfo.gmm2OutGlobal)), layoutC);
                auto gmBlockC = gmC.Slice(
                    Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                
                typename BlockMmad::BlockShape singleShape{
                    Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), Get<K_VALUE>(actualShape), 0};
                blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, gmBlockC, singleShape);
                SetFlag<HardEvent::FIX_S>(0);
                WaitFlag<HardEvent::FIX_S>(0);

                // AtomicAdd 通知 AIV
                uint32_t rowGroupIdx = mLoc / L1_TILE_M_128;
                uint32_t rowGroupsThisExpert = Ops::Base::CeilDiv(m, L1_TILE_M_128);
                uint32_t groupCoreStart, groupCoreSize;
                ComputeGroupRange(rowGroupIdx, rowGroupsThisExpert, blockAivNum, groupCoreStart, groupCoreSize);
                int64_t baseOffset = static_cast<int64_t>(groupIdx) * maxRowGroupsStride
                    + static_cast<int64_t>(rowGroupIdx) * blockAivNum * INT_CACHELINE;
                for (uint32_t aivIdx = groupCoreStart; aivIdx < groupCoreStart + groupCoreSize; aivIdx++) {
                    AscendC::AtomicAdd((__gm__ int32_t*)rowGroupComplete.GetPhyAddr()
                        + baseOffset + aivIdx * INT_CACHELINE, int32_t(1));
                }
            }
        }

        // 非量化模式：AIV CombineTokens + pingpong 切换
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            vecSetSyncCom2++;
            if constexpr (g_coreType == AscendC::AIV) {
                auto l0cOutUbGMM2 = pingpongIdx == 0 ? l0cOutUbGMM2First : l0cOutUbGMM2Second;
                WaitForCube(pingpongIdx);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
                AscendC::GlobalTensor<int32_t> tripleGm;
                int32_t lenTile = Get<M_VALUE>(actualShape);
                LocalTensor<int32_t> tripleTensor = LocalTensor<int32_t>(TPosition::VECCALC, 200 * 1024, lenTile * 8);
                tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.tripleInfoPtr +
                    (groupCnt + mLoc) * 32));
                AscendC::DataCopy(tripleTensor, tripleGm, lenTile * 8);
                MegaMoeCombineImpl::CombineTokens<ElementC, decltype(actualShape)>(
                    mLoc, nLoc, n, tripleTensor, l0cOutUbGMM2, actualShape, params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                NotifyCube(pingpongIdx);
            }
            pingpongIdx = 1 - pingpongIdx;
        }
    }
    startBlockIdx = (startBlockIdx + tileNum) % blockNum;
}

} // namespace MegaMoeImpl

#endif