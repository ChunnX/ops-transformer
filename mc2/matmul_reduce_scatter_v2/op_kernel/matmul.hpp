/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul.hpp
 * \brief
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_REDUCE_SCATTER_AIV_MODE_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_REDUCE_SCATTER_AIV_MODE_HPP

#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/cross_core_sync.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/copy_gm_to_ub.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/copy_ub_to_gm.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/kernel/padding_matmul.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/catlass.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/coord.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm_coord.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/matrix_coord.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/resource.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tile/tile_copy.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tile/tile_mmad.hpp"
#include "matmul_reduce_scatter_aiv_mode_util.h"
#include "matmul_reduce_scatter_v2_aiv_mode_tiling.h"
#include "block_mmad_preload_fixpipe.h"

using namespace AscendC;
using namespace matmulReduceScatterV2_aivmode_tiling;
using namespace matmulReduceScatterV2_util;
namespace Catlass::Gemm::Kernel {
template <
    class PrologueA,
    class PrologueB,
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    bool HasBias
>
class MatmulReduceScatterAivMode : public CommBase {
public:
    using BlockMmad = BlockMmad_;
    using ComputePolicy = typename BlockMmad::ComputePolicy;
    using PlatformTag = typename BlockMmad::PlatformTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementScale = uint64_t;
    using ElemAInt8 = int8_t;
    using ElemBInt8 = int8_t;
    using ElemCHalf = half;
    using LayWA = typename BlockMmad::LayoutA;
    using LayWB = typename BlockMmad::LayoutB;
    using LayScale = typename layout::VectorLayout;
    template<bool condition, class mmad>
    struct BiasTypeHelper {
        using type = typename mmad::ElementBias;
    };

    template<class mmad>
    struct BiasTypeHelper<false, mmad> {
        using type = float;
    };

    template<class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template<>
    struct LayoutHelper<void> {
        using type = void;
    };
    using ElementBias = typename BiasTypeHelper<HasBias, BlockMmad>::type;
    using LayoutA = std::conditional_t<std::is_void_v<PrologueA>, LayWA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::conditional_t<std::is_void_v<PrologueB>, LayWB, typename LayoutHelper<PrologueB>::type>;

    using L1Tile = typename BlockMmad::L1Tile;
    using L0Tile = typename BlockMmad::L0Tile;
    using ElmtC = typename BlockMmad::ElmtC;
    using LayoutC = typename BlockMmad::LayoutC;

    using FixpipeBlockMM = Gemm::Block::FixpipeBlockMM<Gemm::MmadAtlasA2Preload<false, false>, L1Tile,
        L0Tile, LayoutA, LayoutB, LayoutC>;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrBias;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrScale;
        LayScale layoutScale;
        GM_ADDR ptrPeerMem;
        LayoutC layoutPeerMem;
        GM_ADDR ptrWorkSpace;
        int32_t pValue;
        int32_t swizzlCount;
        int32_t swizzlDirect;
        DequantType dequantType;
        int32_t rankIdx;
        int32_t rankSize;
        bool needFixpipe;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {
        }

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrBias_, GM_ADDR ptrC_,
               LayoutC layoutC_, GM_ADDR ptrScale_, LayScale layoutScale_, GM_ADDR ptrPeerMem_, LayoutC layoutPeerMem_,
               GM_ADDR ptrWorkSpace_, int32_t pValue_, int32_t swizzlCount_, int32_t swizzlDirect_, DequantType dequantType_,
               int32_t rankIdx_, int32_t rankSize_, bool needFixpipe_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrBias(ptrBias_), ptrC(ptrC_), layoutC(layoutC_), ptrScale(ptrScale_), layoutScale(layoutScale_),
              ptrPeerMem(ptrPeerMem_), layoutPeerMem(layoutPeerMem_), ptrWorkSpace(ptrWorkSpace_),
              pValue(pValue_), swizzlCount(swizzlCount_), swizzlDirect(swizzlDirect_), dequantType(dequantType_),
              rankIdx(rankIdx_), rankSize(rankSize_), needFixpipe(needFixpipe_) {}
    };

    // Methods
    CATLASS_DEVICE
    MatmulReduceScatterAivMode()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    inline __aicore__ void InitParams(Params const &params)
    {
        coreIdx = AscendC::GetBlockIdx();
        coreNum = AscendC::GetBlockNum();
        finalM = params.problemShape.m() / params.rankSize;
        mLoopPerRank = (finalM + L1Tile::M - 1) / L1Tile::M;
        mLoops = mLoopPerRank * params.rankSize;
        nLoops = (params.problemShape.n() + L1Tile::N - 1) / L1Tile::N;
        coreLoops = mLoops * nLoops;
        kAlignment = Block512B<ElementA>::AlignUp(params.problemShape.k());
        loopNumPerComm = params.pValue * coreNum;
        calCount = (coreLoops + loopNumPerComm - 1) / loopNumPerComm;
    }

    inline __aicore__ GemmCoord GetBlockSizeCoord(GemmCoord blockIdxCoord, GemmCoord blockLocCoord, int32_t mLoop,
                                                  int32_t mSize, int32_t nLoop, int32_t nSize, int32_t kSize)
    {
        uint32_t mActual = (blockIdxCoord.m() == (mLoop - 1)) ? (mSize - blockLocCoord.m()) : L1Tile::M;
        uint32_t nActual = (blockIdxCoord.n() == (nLoop - 1)) ? (nSize - blockLocCoord.n()) : L1Tile::N;
        uint32_t kActual = kSize;
        return GemmCoord{mActual, nActual, kActual};
    }

    inline __aicore__ GemmCoord GetBlockLocCoord(GemmCoord blockIdxCoord)
    {
        return GemmCoord{blockIdxCoord.m() * L1Tile::M, blockIdxCoord.n() * L1Tile::N,
                         blockIdxCoord.k() * L1Tile::K};
    }

    inline __aicore__ GemmCoord GetBlockIdCoord(int32_t loopOffset, int32_t mLoop, int32_t nLoop, int32_t swizzlDirect,
                                                int32_t swizzlCount)
    {
        uint32_t kIdx = 0;
        int64_t mIdx, nIdx;
        GetSwizzledBlockIdx(loopOffset, mLoop, nLoop, swizzlDirect, swizzlCount, mIdx, nIdx);
        return GemmCoord{static_cast<uint32_t>(mIdx), static_cast<uint32_t>(nIdx), kIdx}; // idx在uint32_t范围内
    }

    inline __aicore__ void FixpipeMatmul(Params const &params)
    {
        AscendC::GlobalTensor<ElemAInt8> gmAI8;
        AscendC::GlobalTensor<ElemBInt8> gmBI8;
        AscendC::GlobalTensor<ElementScale> gmScale;
        AscendC::GlobalTensor<ElemCHalf> gmCHalf;
        AscendC::GlobalTensor<ElemCHalf> gmPeerMemHalf;
        gmAI8.SetGlobalBuffer((__gm__ ElemAInt8 *)params.ptrA);
        gmBI8.SetGlobalBuffer((__gm__ ElemBInt8 *)params.ptrB);
        gmCHalf.SetGlobalBuffer((__gm__ ElemCHalf *)params.ptrC);
        gmPeerMemHalf.SetGlobalBuffer((__gm__ ElemCHalf *)params.ptrPeerMem);
        gmScale.SetGlobalBuffer((__gm__ ElementScale *)params.ptrScale);

        FixpipeBlockMM fixpipeBlockMmad(resource);
        int32_t blockSize = L1Tile::M * L1Tile::N;
        for (int32_t calIdx = 0; calIdx < calCount; calIdx++) {
            int32_t flagIdx = calIdx % MAX_BLOCK_COUNT;
            if (calIdx >= MAX_BLOCK_COUNT) {
                WaitEvent(flagIdx);
            }
            for (int32_t p = 0; p < params.pValue; p++) {
                int32_t loopIdx = calIdx * loopNumPerComm + p * coreNum + coreIdx;
                if (loopIdx >= coreLoops) {
                    break;
                }
                int32_t dstRankIdx = loopIdx % params.rankSize;
                int64_t gmABlockSt = dstRankIdx * finalM * kAlignment;
                int32_t inRankIdx = loopIdx / params.rankSize;
                GemmCoord blockIdxCoord = GetBlockIdCoord(inRankIdx, mLoopPerRank, nLoops,
                    params.swizzlDirect, params.swizzlCount);
                GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
                GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, 
                    mLoopPerRank, finalM, nLoops, params.problemShape.n(), params.problemShape.k());

                MatrixCoord offsetC{blockLocCoord.m(), blockLocCoord.n()};
                MatrixCoord offsetB{blockLocCoord.k(), blockLocCoord.n()};
                MatrixCoord offsetA{blockLocCoord.m(), blockLocCoord.k()};
                int64_t gmOffsetA = gmABlockSt + params.layoutA.GetOffset(offsetA);
                int64_t gmOffsetC;
                int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
                LayoutC layoutGmDst;
                AscendC::GlobalTensor<ElemCHalf> gmDstHalf;
                if (dstRankIdx == params.rankIdx && params.dequantType == DequantType::PER_CHANNEL) {
                    layoutGmDst = params.layoutC;
                    gmDstHalf = gmCHalf;
                    gmOffsetC = params.layoutC.GetOffset(offsetC);
                } else {
                    layoutGmDst = params.layoutPeerMem;
                    gmDstHalf = gmPeerMemHalf;
                    gmOffsetC = (flagIdx * loopNumPerComm + dstRankIdx * (loopNumPerComm / params.rankSize) +
                                 (loopIdx % loopNumPerComm) / params.rankSize) *
                                blockSize;
                }

                bool hasNextBlock = false;
                bool isFirstBlock = loopIdx == coreIdx;
                GemmCoord nextBlockIdCoord;
                GemmCoord nextBlockLocCoord;
                GemmCoord nextBlockSizeCoord;
                int32_t nextLoopIdx = loopIdx + coreNum;
                int32_t nextDstRankIdx = nextLoopIdx % params.rankSize;
                int32_t nextInRankIdx = nextLoopIdx / params.rankSize;
                if (nextLoopIdx < coreLoops) {
                    hasNextBlock = true;
                    nextBlockIdCoord = GetBlockIdCoord(nextInRankIdx, mLoopPerRank, nLoops, params.swizzlDirect,
                        params.swizzlCount);
                    nextBlockLocCoord = GetBlockLocCoord(nextBlockIdCoord);
                    nextBlockSizeCoord = GetBlockSizeCoord(nextBlockIdCoord, nextBlockLocCoord, mLoopPerRank,
                        finalM, nLoops, params.problemShape.n(), params.problemShape.k());
                }
                int64_t nextGmABlockSt = nextDstRankIdx * finalM * kAlignment;
                MatrixCoord offsetNextA{nextBlockLocCoord.m(), nextBlockLocCoord.k()};
                MatrixCoord offsetNextB{nextBlockLocCoord.k(), nextBlockLocCoord.n()};
                int64_t gmOffsetNextA = nextGmABlockSt + params.layoutA.GetOffset(offsetNextA);
                int64_t gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);

                int64_t gmOffsetScale = blockLocCoord.n();
                fixpipeBlockMmad(	
                    gmAI8[gmOffsetA], params.layoutA,	
                    gmBI8[gmOffsetB], params.layoutB,	
                    gmDstHalf[gmOffsetC], layoutGmDst,	
                    gmScale[gmOffsetScale], params.layoutScale,
                    gmAI8[gmOffsetNextA], gmBI8[gmOffsetNextB],
                    blockSizeCoord, nextBlockSizeCoord, isFirstBlock, hasNextBlock);
            }
            FFTSCrossCoreSync<PIPE_FIX, 2>(flagIdx);
        }
    }

    inline __aicore__ void Matmul(Params const &params)
    {
        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementB> gmB;
        AscendC::GlobalTensor<ElmtC> gmC;
        AscendC::GlobalTensor<ElementBias> gmBias;
        AscendC::GlobalTensor<ElmtC> gmPeerMem;
        AscendC::GlobalTensor<ElmtC> gmWorkSpace;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        gmC.SetGlobalBuffer((__gm__ ElmtC *)params.ptrC);
        if constexpr (HasBias) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);
        }
        gmPeerMem.SetGlobalBuffer((__gm__ ElmtC *)params.ptrPeerMem);
        gmWorkSpace.SetGlobalBuffer((__gm__ ElmtC *)params.ptrWorkSpace);

        BlockMmad blockMmad(resource);
        int32_t blockSize = L1Tile::M * L1Tile::N;
        for (int32_t calIdx = 0; calIdx < calCount; calIdx++) {
            int32_t flagIdx = calIdx % MAX_BLOCK_COUNT;
            if (calIdx >= MAX_BLOCK_COUNT) {
                WaitEvent(flagIdx);
            }
            for (int32_t p = 0; p < params.pValue; p++) {
                int32_t loopIdx = calIdx * loopNumPerComm + p * coreNum + coreIdx;
                if (loopIdx >= coreLoops) {
                    break;
                }
                int32_t dstRankIdx = loopIdx % params.rankSize;
                int32_t inRankIdx = loopIdx / params.rankSize;
                int64_t gmABlockSt = dstRankIdx * finalM * kAlignment;
                GemmCoord blockIdxCoord =
                    GetBlockIdCoord(inRankIdx, mLoopPerRank, nLoops, params.swizzlDirect, params.swizzlCount);
                GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
                GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, mLoopPerRank, finalM, nLoops,
                                                             params.problemShape.n(), params.problemShape.k());

                MatrixCoord offsetA{blockLocCoord.m(), blockLocCoord.k()};
                MatrixCoord offsetB{blockLocCoord.k(), blockLocCoord.n()};
                MatrixCoord offsetC{blockLocCoord.m(), blockLocCoord.n()};
                int64_t gmOffsetA = gmABlockSt + params.layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
                int64_t gmOffsetC;
                LayoutC layoutGmDst;
                AscendC::GlobalTensor<ElmtC> gmDst;
                if (std::is_same<ElmtC, int32_t>::value) {
                    gmDst = gmWorkSpace;
                    layoutGmDst = params.layoutPeerMem;
                    gmOffsetC = (flagIdx * loopNumPerComm + dstRankIdx * (loopNumPerComm / params.rankSize) +
                                 (loopIdx % loopNumPerComm) / params.rankSize) *
                                blockSize;
                } else if (dstRankIdx == params.rankIdx) {
                    gmDst = gmC;
                    layoutGmDst = params.layoutC;
                    gmOffsetC = params.layoutC.GetOffset(offsetC);
                } else {
                    gmDst = gmPeerMem;
                    layoutGmDst = params.layoutPeerMem;
                    gmOffsetC = (flagIdx * loopNumPerComm + dstRankIdx * (loopNumPerComm / params.rankSize) +
                                 (loopIdx % loopNumPerComm) / params.rankSize) *
                                blockSize;
                }
                if constexpr (HasBias) {
                    blockMmad(
                        gmA[gmOffsetA], params.layoutA,
                        gmB[gmOffsetB], params.layoutB,
                        gmC[gmOffsetC], params.layoutC,
                        gmBias[blockLocCoord.n()], blockSizeCoord);
                } else {
                    bool isFirstBlock = loopIdx == coreIdx;
                    bool hasNextBlock = false;
                    GemmCoord nextBlockLocCoord;
                    GemmCoord nextBlockSizeCoord;
                    GemmCoord nextBlockIdCoord;
                    int32_t nextLoopIdx = loopIdx + coreNum;
                    int32_t nextInRankIdx = nextLoopIdx / params.rankSize;
                    int32_t nextDstRankIdx = nextLoopIdx % params.rankSize;
                    if (nextLoopIdx < coreLoops) {
                        hasNextBlock = true;
                        nextBlockIdCoord = GetBlockIdCoord(nextInRankIdx, mLoopPerRank, nLoops, params.swizzlDirect,
                            params.swizzlCount);
                        nextBlockLocCoord = GetBlockLocCoord(nextBlockIdCoord);
                        nextBlockSizeCoord = GetBlockSizeCoord(nextBlockIdCoord, nextBlockLocCoord, mLoopPerRank,
                            finalM, nLoops, params.problemShape.n(), params.problemShape.k());
                    }
                    int32_t nextGmABlockSt = nextDstRankIdx * finalM * kAlignment;
                    MatrixCoord offsetNextB{nextBlockLocCoord.k(), nextBlockLocCoord.n()};
                    MatrixCoord offsetNextA{nextBlockLocCoord.m(), nextBlockLocCoord.k()};
                    int64_t gmOffsetNextA = nextGmABlockSt + params.layoutA.GetOffset(offsetNextA);
                    int64_t gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);
                    blockMmad(
                        gmA[gmOffsetA], params.layoutA,
                        gmB[gmOffsetB], params.layoutB,
                        gmDst[gmOffsetC], layoutGmDst,
                        gmA[gmOffsetNextA], gmB[gmOffsetNextB],
                        blockSizeCoord, nextBlockSizeCoord, isFirstBlock, hasNextBlock);
                }
            }
            FFTSCrossCoreSync<PIPE_FIX, 2>(flagIdx);
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPad);

        InitParams(params);

        if (params.needFixpipe) {
            FixpipeMatmul(params);
        } else {
            Matmul(params);
        }
    }

private:
    int32_t coreIdx;
    int32_t coreNum;
    int32_t mLoops;
    int32_t nLoops;
    int32_t coreLoops;
    int32_t calCount;
    int32_t finalM;
    int32_t kAlignment;
    int32_t mLoopPerRank;
    int32_t loopNumPerComm;
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID;
    Arch::CrossCoreFlag flagAivFinishPad{FLAG_AIV_FINISH_STORE};
    Arch::Resource<PlatformTag> resource;
};
} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_REDUCE_SCATTER_V2_HPP