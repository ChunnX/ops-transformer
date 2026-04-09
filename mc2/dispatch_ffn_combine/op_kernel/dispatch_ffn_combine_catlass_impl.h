/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef DISPATCH_FFN_COMBINE_CATLASS_IMPL_H
#define DISPATCH_FFN_COMBINE_CATLASS_IMPL_H

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
#include "dispatch_ffn_combine_ops.h"

using namespace AscendC;
using namespace Catlass;

namespace DispatchFFNCombineImpl {

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

// Gmm2CombineRunner - GMM2矩阵乘法 + Combine
// GMM1OutType: GMM1的输出类型(作为GMM2的A输入)
// WeightType: 权重类型
// WeightScaleType: scale类型
// OutputType: 最终输出类型
template <typename GMM1OutType, typename WeightType, typename WeightScaleType, typename OutputType>
struct Gmm2CombineRunner {
    using ElementA = GMM1OutType;  // GMM2的A输入是GMM1的输出
    using ElementB = WeightType;
    using ElementC = OutputType;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;
    
    using ElementMxScaleA = WeightScaleType;
    using ElementMxScaleB = WeightScaleType;

    __aicore__ inline static void RunTileGmm2(
        GM_ADDR aGm, GM_ADDR bGm, GM_ADDR aScaleGm, GM_ADDR bScaleGm, 
        GM_ADDR cUb, GM_ADDR outputGm,
        uint32_t m, uint32_t n, uint32_t k,
        uint32_t blockNum, uint32_t blockIdx, uint32_t startBlockIdx,
        uint32_t& vecSetSyncCom2, uint16_t& pingpongIdx,
        GM_ADDR notifyFn, GM_ADDR waitFn,
        GM_ADDR tokenPerExpertGm, GM_ADDR preSumBeforeRankGm, 
        GM_ADDR cumsumMMGm, uint32_t groupIdx,
        const Params& combineParams)
    {
        Arch::Resource<ArchTag> resource;
        using BlockMmad = Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, void>;
        
        BlockMmad blockMmad(resource);

        auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        auto layoutA = tla::MakeLayout<ElementA, LayoutA>(m, k);
        auto layoutB = tla::MakeLayout<ElementB, LayoutB>(k, n);
        auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutA, false>(m, scaleK);
        auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutB, true>(scaleK, n);
        auto layoutC = tla::MakeLayout<ElementC, LayoutC>(TILE_M, TILE_N);

        GlobalTensor<ElementA> aGlobal;
        GlobalTensor<ElementB> bGlobal;
        GlobalTensor<ElementMxScaleA> aScaleGlobal;
        GlobalTensor<ElementMxScaleB> bScaleGlobal;
        LocalTensor<ElementC> cLocal;
        GlobalTensor<ElementC> outputGlobal;
        GlobalTensor<int32_t> tokenPerExpertGm;
        GlobalTensor<int32_t> preSumBeforeRankGm;
        GlobalTensor<int32_t> cumsumMMGm;

        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(aGm));
        bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(bGm));
        aScaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleA*>(aScaleGm));
        bScaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleB*>(bScaleGm));
        cLocal.SetLocalBuffer(reinterpret_cast<__gm__ ElementC*>(cUb));
        outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(outputGm));
        tokenPerExpertGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenPerExpertGm));
        preSumBeforeRankGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(preSumBeforeRankGm));
        cumsumMMGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cumsumMMGm));

        auto tensorA = tla::MakeTensor(aGlobal, layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(bGlobal, layoutB, Arch::PositionGM{});
        auto tensorMxScaleA = tla::MakeTensor(aScaleGlobal, layoutMxScaleA, Arch::PositionGM{});
        auto tensorMxScaleB = tla::MakeTensor(bScaleGlobal, layoutMxScaleB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(cLocal, layoutC, Arch::PositionUB{});

        BlockScheduler scheduler({m, n, k}, MakeCoord(TILE_M, TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();
        uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;

        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            
            uint32_t mLoc = blockCoord.m() * TILE_M;
            uint32_t nLoc = blockCoord.n() * TILE_N;
            uint32_t kLoc = blockCoord.k() * TILE_K;

            LocalTensor<ElementC> cTile = pingpongIdx == 0 ? 
                reinterpret_cast<LocalTensor<ElementC>>(0x1000) : 
                reinterpret_cast<LocalTensor<ElementC>>(0x2000);
            auto tensorCTile = tla::MakeTensor(cTile, layoutC, Arch::PositionUB{});

            if constexpr(g_coreType == AscendC::AIC) {
                if (vecSetSyncCom2 >= 2) {
                    typedef void (*WaitVectorFn)(uint16_t);
                    (*(WaitVectorFn)waitFn)(pingpongIdx);
                }
                
                auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(mLoc, kLoc),
                    tla::MakeShape(actualShape.m(), actualShape.k()));
                auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(kLoc, nLoc),
                    tla::MakeShape(actualShape.k(), actualShape.n()));
                auto tensorBlockMxScaleA = GetTile(tensorMxScaleA,
                    tla::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                    tla::MakeShape(actualShape.m(), CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM)));
                auto tensorBlockMxScaleB = GetTile(tensorMxScaleB,
                    tla::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                    tla::MakeShape(CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM), actualShape.n()));

                blockMmad(tensorBlockA, tensorBlockB, tensorCTile, actualShape, tensorBlockMxScaleA, tensorBlockMxScaleB);

                typedef void (*NotifyVectorFn)(uint16_t);
                (*(NotifyVectorFn)notifyFn)(pingpongIdx);
            }

            if constexpr(g_coreType == AscendC::AIV) {
                typedef void (*WaitCubeFn)(uint16_t);
                (*(WaitCubeFn)waitFn)(pingpongIdx);
                
                // Combine: 将GMM2结果按token组合并分发到远端peermem
                CombineTokensToPeermem(cTile, actualShape, mLoc, nLoc, 
                                       combineParams, tokenPerExpertGm, preSumBeforeRankGm, 
                                       cumsumMMGm, groupIdx);
                
                pingpongIdx = 1 - pingpongIdx;

                typedef void (*NotifyCubeFn)(uint16_t);
                (*(NotifyCubeFn)notifyFn)(pingpongIdx);
            }
        }
        
        vecSetSyncCom2++;
    }
};

// GmmSwigluRunner - GMM1矩阵乘法 + SwiGLU
// XType: 输入X类型
// WeightType: 权重类型
// WeightScaleType: scale类型
// GMM1OutType: GMM1输出类型(SwiGLU输出)
template <typename XType, typename WeightType, typename WeightScaleType, typename GMM1OutType>
struct GmmSwigluRunner {
    using ElementA = XType;
    using ElementB = WeightType;
    using ElementC = GMM1OutType;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;
    
    using ElementMxScaleA = WeightScaleType;
    using ElementMxScaleB = WeightScaleType;

    struct GmmSwigluParams {
        GM_ADDR aGm;
        GM_ADDR bGm;
        GM_ADDR bScaleGm;
        GM_ADDR aScaleGm;
        GM_ADDR cUb;
        GM_ADDR cUb2;
        GM_ADDR outputUb;
        GM_ADDR outputScaleUb;
        uint32_t m;
        uint32_t n;
        uint32_t k;
        uint32_t blockNum;
        uint32_t blockIdx;
        uint32_t subBlockIdx;
        uint32_t startBlockIdx;
        bool isVecSetSyncCom;
    };

    __aicore__ inline static void RunTile(
        const GmmSwigluParams& params,
        LocalTensor<ElementC> l0cOutUbFirst,
        LocalTensor<ElementC> l0cOutUbSecond,
        uint16_t& pingpongIdx,
        bool& isVecSetSyncCom,
        uint32_t startBlockIdx,
        uint32_t blockNum,
        uint32_t blockIdx,
        GM_ADDR notifyFn, GM_ADDR waitFn)
    {
        Arch::Resource<ArchTag> resource;
        using BlockMmad = Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, void>;
        
        BlockMmad blockMmad(resource);

        uint32_t outputN = params.n / SWIGLU_N_HALF;

        auto scaleK = CeilDiv(params.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        auto layoutA = tla::MakeLayout<ElementA, LayoutA>(params.m, params.k);
        auto layoutB = tla::MakeLayout<ElementB, LayoutB>(params.k, params.n);
        auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutA, false>(params.m, scaleK);
        auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutB, true>(scaleK, params.n);
        auto layoutC = tla::MakeLayout<ElementC, LayoutC>(TILE_M, TILE_N);

        GlobalTensor<ElementA> aGlobal;
        GlobalTensor<ElementB> bGlobal;
        GlobalTensor<ElementMxScaleA> aScaleGlobal;
        GlobalTensor<ElementMxScaleB> bScaleGlobal;

        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.aGm));
        bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.bGm));
        aScaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleA*>(params.aScaleGm));
        bScaleGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleB*>(params.bScaleGm));

        auto tensorA = tla::MakeTensor(aGlobal, layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(bGlobal, layoutB, Arch::PositionGM{});
        auto tensorMxScaleA = tla::MakeTensor(aScaleGlobal, layoutMxScaleA, Arch::PositionGM{});
        auto tensorMxScaleB = tla::MakeTensor(bScaleGlobal, layoutMxScaleB, Arch::PositionGM{});

        BlockScheduler scheduler({params.m, outputN, params.k}, MakeCoord(TILE_M, TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();
        uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;

        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            
            uint32_t mLoc = blockCoord.m() * TILE_M;
            uint32_t nLoc = blockCoord.n() * TILE_N;
            uint32_t kLoc = blockCoord.k() * TILE_K;

            LocalTensor<ElementC> cLocal = pingpongIdx == 0 ? l0cOutUbFirst : l0cOutUbSecond;
            auto tensorC = tla::MakeTensor(cLocal, layoutC, Arch::PositionUB{});

            if constexpr(g_coreType == AscendC::AIC) {
                if (loopIdx == startLoopIdx && isVecSetSyncCom) {
                    typedef void (*WaitVectorFn)(uint16_t);
                    (*(WaitVectorFn)waitFn)(0);
                }
                
                auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(mLoc, kLoc),
                    tla::MakeShape(actualShape.m(), actualShape.k()));
                auto tensorBlockMxScaleA = GetTile(tensorMxScaleA,
                    tla::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                    tla::MakeShape(actualShape.m(), CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM)));
                
                uint32_t outputN = n / SWIGLU_N_HALF;
                for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                    LocalTensor<ElementC> cTile = weightBlock == 0 ? l0cOutUbFirst : l0cOutUbSecond;
                    auto tensorCTile = tla::MakeTensor(cTile, layoutC, Arch::PositionUB{});
                    
                    auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(kLoc, nLoc + weightBlock * outputN),
                        tla::MakeShape(actualShape.k(), actualShape.n()));
                    auto tensorBlockMxScaleB = GetTile(tensorMxScaleB,
                        tla::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * outputN),
                        tla::MakeShape(CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM), actualShape.n()));
                    
                    blockMmad(tensorBlockA, tensorBlockB, tensorCTile, actualShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
                }

                typedef void (*NotifyVectorFn)(uint16_t);
                (*(NotifyVectorFn)notifyFn)(0);
            }

            if constexpr(g_coreType == AscendC::AIV) {
                if (params.subBlockIdx == 0) {
                    typedef void (*WaitCubeFn)(uint16_t);
                    (*(WaitCubeFn)waitFn)(0);

                    // GMM1完成后，AIV执行SwiGLU+Quant (BlockEpilogue)
                    // TODO: 调用 epilogueOp(tensorC, shape, offset)
                    if (loopIdx == coreLoops - 1 || loopIdx + blockNum >= coreLoops) {
                        isVecSetSyncCom = true;
                    }
                    
                    pingpongIdx = 1 - pingpongIdx;
                    
                    typedef void (*NotifyCubeFn)(uint16_t);
                    (*(NotifyCubeFn)notifyFn)(0);
                }
            }

            startBlockIdx = (startBlockIdx + 1) % blockNum;
        }
    }
};

}  // namespace DispatchFFNCombineImpl

#endif