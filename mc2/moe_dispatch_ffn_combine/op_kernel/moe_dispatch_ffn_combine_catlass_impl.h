/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef MOE_DISPATCH_FFN_COMBINE_CATLASS_IMPL_H
#define MOE_DISPATCH_FFN_COMBINE_CATLASS_IMPL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"

#include "gmmsq_base.h"
#include "moe_dispatch_ffn_combine_ops.h"

using namespace AscendC;
using namespace Catlass;

namespace MoeDispatchFFNCombineImpl {

using ArchTag = Arch::AtlasA5;
constexpr static bool enableUnitFlag = true;
using DispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag>;
using L1TileShape = tla::Shape<Int<256>, Int<256>, Int<448>>;
using L0TileShape = tla::Shape<Int<256>, Int<256>, Int<128>>;
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
constexpr uint32_t TILE_M = 256;
constexpr uint32_t TILE_N = 256;
constexpr uint32_t TILE_K = 448;
constexpr uint32_t SWIGLU_N_HALF = 2;
constexpr uint32_t MAX_SINGLE_MN = 256 * 256;

// GroupMatmulSwigluQuant - GMM1 + SwiGLU + Quant
template <typename ElementA, typename ElementB, typename ElementC, 
          typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmulSwigluQuant(
    BlockEpilogueSwigluQuant<L1TileShape, ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true>& epilogueOp,
    const Params& params,
    const Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo,
    uint32_t& startBlockIdx,
    int32_t& vecSetSyncCom)
{
    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = Get<N_VALUE>(problemShape);
    uint32_t k = Get<K_VALUE>(problemShape);
    uint32_t outputN = n / SWIGLU_N_HALF;

    epilogueOp.UpdateNextProblem({m, outputN, k, 0});

    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx() / GetTaskRation();
    uint32_t subBlockIdx = GetSubBlockIdx();
    uint32_t dispatchBlockNumPerEP = params.tilingData->blockNumPerEP;

    LocalTensor<ElementC> l0cOutUbFirst{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    LocalTensor<ElementC> l0cOutUbSecond{AscendC::TPosition::VECIN, MAX_SINGLE_MN * sizeof(ElementC), MAX_SINGLE_MN};

    GlobalTensor<int32_t> groupFlagListGm2;
    groupFlagListGm2.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrFlagList2);

    Arch::Resource<ArchTag> resource;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, void>;
    BlockMmad blockMmad(resource);

    auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    auto layoutA = tla::MakeLayout<ElementA, layout::RowMajor>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, layout::ColumnMajor>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(m, scaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, layout::ColumnMajor, true>(scaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(TILE_M, TILE_N);

    GlobalTensor<ElementA> aGlobalTensor;
    GlobalTensor<ElementB> bGlobalTensor;
    GlobalTensor<ElementMxScaleA> aScaleGlobalTensor;
    GlobalTensor<ElementMxScaleB> bScaleGlobalTensor;

    aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal));
    bGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal));
    aScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal));
    bScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal));

    auto tensorA = tla::MakeTensor(aGlobalTensor, layoutA, Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobalTensor, layoutB, Arch::PositionGM{});
    auto tensorMxScaleA = tla::MakeTensor(aScaleGlobalTensor, layoutMxScaleA, Arch::PositionGM{});
    auto tensorMxScaleB = tla::MakeTensor(bScaleGlobalTensor, layoutMxScaleB, Arch::PositionGM{});
    auto tensorUbFirst = tla::MakeTensor(l0cOutUbFirst, layoutC, Arch::PositionUB{});
    auto tensorUbSecond = tla::MakeTensor(l0cOutUbSecond, layoutC, Arch::PositionUB{});

    BlockScheduler scheduler({m, outputN, k}, MakeCoord(TILE_M, TILE_N));
    uint32_t coreLoops = scheduler.GetCoreLoops();
    uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
        GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        
        uint32_t mLoc = blockCoord.m() * TILE_M;
        uint32_t nLoc = blockCoord.n() * TILE_N;
        uint32_t kLoc = blockCoord.k() * TILE_K;

        if constexpr(g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
                uint32_t targetValue = params.tilingData->EP * dispatchBlockNumPerEP;
                __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm2.GetPhyAddr();
                while(targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {}
                }
            }
            if (vecSetSyncCom) {
                NotifyVector();
            }
            
            auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(mLoc, kLoc),
                tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockMxScaleA = GetTile(tensorMxScaleA,
                tla::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                tla::MakeShape(actualShape.m(), CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM)));
            auto tensorBlockUbFirst = GetTile(tensorUbFirst, tla::MakeCoord(0, 0),
                tla::MakeShape(actualShape.m(), actualShape.n()));
            auto tensorBlockUbSecond = GetTile(tensorUbSecond, tla::MakeCoord(0, 0),
                tla::MakeShape(actualShape.m(), actualShape.n()));

            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto tensorBlockB = GetTile(tensorB,
                    tla::MakeCoord(kLoc, nLoc + weightBlock * outputN),
                    tla::MakeShape(actualShape.k(), actualShape.n()));
                auto tensorBlockMxScaleB = GetTile(tensorMxScaleB,
                    tla::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * outputN),
                    tla::MakeShape(CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM), actualShape.n()));
                blockMmad(tensorBlockA, tensorBlockB, weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond,
                          actualShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
            }
            NotifyVector();
        }

        vecSetSyncCom = 1;
        
        if constexpr(g_coreType == AscendC::AIV) {
            if (subBlockIdx == 0) {
                Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{actualShape.m(), actualShape.n(), 0, 0};
                Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    mLoc * outputN + nLoc,
                    mLoc * CeilDiv(outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                        CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                    0, 0, 0, 0
                };
                WaitForCube();
                epilogueOp(epilogueShape, epilogueOffset);
                NotifyCube();
            }
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % blockNum;
}

// GroupMatmul2 - GMM2矩阵乘法 + Combine
template <typename ElementA, typename ElementB, typename ElementC, 
          typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmul2(
    const Params& params,
    const Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo,
    uint32_t& startBlockIdx,
    int32_t& vecSetSyncCom2,
    uint32_t groupIdx)
{
    static uint16_t pingpongIdx = 0;
    constexpr uint32_t GMM2_BLOCK_SIZE = TILE_M * TILE_N;

    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = Get<K_VALUE>(problemShape);
    uint32_t k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;

    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx() / GetTaskRation();
    uint32_t subBlockIdx = GetSubBlockIdx();

    LocalTensor<ElementC> l0cOutUbGMM2First{AscendC::TPosition::VECIN, 0, GMM2_BLOCK_SIZE};
    LocalTensor<ElementC> l0cOutUbGMM2Second{AscendC::TPosition::VECIN, GMM2_BLOCK_SIZE * sizeof(ElementC), GMM2_BLOCK_SIZE};

    GlobalTensor<int32_t> groupFlagListGm;
    groupFlagListGm.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrFlagList);

    Arch::Resource<ArchTag> resource;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, void>;
    BlockMmad blockMmad(resource);

    auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    auto layoutA = tla::MakeLayout<ElementA, layout::RowMajor>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, layout::ColumnMajor>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(m, scaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, layout::ColumnMajor, true>(scaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(TILE_M, TILE_N);

    GlobalTensor<ElementA> aGlobalTensor;
    GlobalTensor<ElementB> bGlobalTensor;
    GlobalTensor<ElementMxScaleA> aScaleGlobalTensor;
    GlobalTensor<ElementMxScaleB> bScaleGlobalTensor;

    aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal));
    bGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal));
    aScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal));
    bScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal));

    auto tensorA = tla::MakeTensor(aGlobalTensor, layoutA, Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobalTensor, layoutB, Arch::PositionGM{});
    auto tensorMxScaleA = tla::MakeTensor(aScaleGlobalTensor, layoutMxScaleA, Arch::PositionGM{});
    auto tensorMxScaleB = tla::MakeTensor(bScaleGlobalTensor, layoutMxScaleB, Arch::PositionGM{});
    auto tensorUbFirst = tla::MakeTensor(l0cOutUbGMM2First, layoutC, Arch::PositionUB{});
    auto tensorUbSecond = tla::MakeTensor(l0cOutUbGMM2Second, layoutC, Arch::PositionUB{});

    BlockScheduler scheduler({m, n, k}, MakeCoord(TILE_M, TILE_N));
    uint32_t coreLoops = scheduler.GetCoreLoops();
    uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
        GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        
        uint32_t mLoc = blockCoord.m() * TILE_M;
        uint32_t nLoc = blockCoord.n() * TILE_N;
        uint32_t kLoc = blockCoord.k() * TILE_K;

        auto tensorUb = pingpongIdx == 0 ? tensorUbFirst : tensorUbSecond;
        auto l0cOutUbGMM2 = pingpongIdx == 0 ? l0cOutUbGMM2First : l0cOutUbGMM2Second;

        if constexpr(g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
                BlockScheduler gmmBlockScheduler({m, k, n}, MakeCoord(TILE_M, TILE_N));
                uint32_t targetLoops = gmmBlockScheduler.GetCoreLoops();
                __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm.GetPhyAddr();
                while(targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {}
                }
            }
            if (vecSetSyncCom2 >= 2) {
                WaitForVector(pingpongIdx);
            }
            
            auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(mLoc, kLoc),
                tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockMxScaleA = GetTile(tensorMxScaleA,
                tla::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                tla::MakeShape(actualShape.m(), CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM)));
            auto tensorBlockUb = GetTile(tensorUb, tla::MakeCoord(0, 0),
                tla::MakeShape(actualShape.m(), actualShape.n()));
            auto tensorBlockB = GetTile(tensorB,
                tla::MakeCoord(kLoc, nLoc),
                tla::MakeShape(actualShape.k(), actualShape.n()));
            auto tensorBlockMxScaleB = GetTile(tensorMxScaleB,
                tla::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                tla::MakeShape(CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM), actualShape.n()));
            blockMmad(tensorBlockA, tensorBlockB, tensorBlockUb,
                      actualShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
            NotifyVector(pingpongIdx);
        }

        vecSetSyncCom2++;

        if constexpr(g_coreType == AscendC::AIV) {
            if (subBlockIdx == 0) {
                WaitForCube(pingpongIdx);
                MoeDispatchFFNCombineImpl::CombineTokens<ElementC>>(mLoc, nLoc, n, groupIdx, l0cOutUbGMM2, actualShape, params);
                NotifyCube(pingpongIdx);
            }
        }
        pingpongIdx = 1 - pingpongIdx;
    }
    startBlockIdx = (startBlockIdx + coreLoops) % blockNum;
}

}  // namespace MoeDispatchFFNCombineImpl

#endif
