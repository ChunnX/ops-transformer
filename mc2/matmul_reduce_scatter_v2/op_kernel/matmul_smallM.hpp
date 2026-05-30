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
 * \file matmul_smallM.hpp
 * \brief
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_REDUCE_SCATTER_AIV_MODE_SMALLM_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_REDUCE_SCATTER_AIV_MODE_SMALLM_HPP

#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/catlass.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/coord.hpp"
// smallM specific marker
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm_coord.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/matrix_coord.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/resource.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/cross_core_sync.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/copy_gm_to_ub.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/copy_ub_to_gm.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/kernel/padding_matmul.hpp"
#include "matmul_reduce_scatter_aiv_mode_util.h"
#include "matmul_reduce_scatter_v2_aiv_mode_tiling.h"
#include "block_mmad_preload_fixpipe.h"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tile/tile_copy.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tile/tile_mmad.hpp"

using namespace ascendc;
using namespace matmulReduceScatterV2_aivmode_tiling_utils;
using namespace matmulReduceScatterV2_utils;
namespace Catlass::Gemm::Kernel {
template <
    class PrologueA,
    class PrologueB,
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    bool HasBias
>

class MatmulReduceScatterAivModeSmallM : public CommBase {
public:
    using BlockMM = BlockMmad_;
    using ComputePolicy = typename BlockMM::DispatchPolicy;
    using PlatformTag = typename BlockMM::ArchTag;
    using ElemA = typename BlockMM::EA;
    using ElemB = typename BlockMM::ElementB;
    using ElemScale = uint64_t;
    using LayWA = typename BlockMM::LayoutA;
    using LayWB = typename BlockMM::LayoutB;
    using LayScl = typename layout::VectorLayout;
    template<bool condition, class mmad>
    struct BiasTpHlp {
        using type = typename mmad::ElementBias;
    };

    template<class mmad>
    struct BiasTpHlp<false, mmad> {
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
    using ElemBias = typename BiasTpHlp<HasBias, BlockMM>::type;
    using LayoutA = std::conditional_t<std::is_void_v<PrologueA>, LayoutWA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::conditional_t<std::is_void_v<PrologueB>, LayoutWB, typename LayoutHelper<PrologueB>::type>;
    using L1Tile = typename BlockMM::L1TileShape;
    using L0Tile = typename BlockMM::L0TileShape;
    using ElemC = typename BlockMM::ElemC;
    using LayC = typename BlockMM::LayoutC;

    using ElemAInt8 = int8_t;
    using ElemBInt8 = int8_t;
    using ElemCHalf = half;
    using FixpipeBlockMmad =
        Gemm::Block::FixpipeBlockMmad<Gemm::MmadAtlasA2Preload<false, false>, L1TileShape, L0TileShape, LayoutA,
            LayoutB, LayoutC>;

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
        GM_ADDR ptrScale;
        int32_t pValue;
        int32_t swizzlCount;
        int32_t swizzlDirect;
        int32_t rankIdx;
        int32_t rankSize;
        bool needFixpipe;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {
        }

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrBias_, GM_ADDR ptrC_, GM_ADDR ptrScale_, int32_t pValue_, int32_t swizzlCount_,
               int32_t swizzlDirect_, int32_t rankIdx_, int32_t rankSize_, bool needFixpipe_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrBias(ptrBias_), ptrC(ptrC_), ptrScale(ptrScale_), pValue(pValue_), swizzlCount(swizzlCount_),
              swizzlDirect(swizzlDirect_), rankIdx(rankIdx_), rankSize(rankSize_), needFixpipe(needFixpipe_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    MatmulReduceScatterAivModeSmallM(){}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    inline __aicore__ void InitArgs(Params const &params)
    {
        block_id = AscendC::GetBlockIdx();
        core_num = AscendC::GetBlockNum();

        m_loop = (params.problemShape.m() + L1TileShape::M - 1) / L1TileShape::M;
        n_loop = (params.problemShape.n() + L1TileShape::N - 1) / L1TileShape::N;
        loop_num_per_comm = params.pValue * core_num;
        coreLoops = m_loop * n_loop;
        kAlignment = Block512B<EA>::AlignUp(params.problemShape.k());
        calCount = (coreLoops + loop_num_per_comm - 1) / loop_num_per_comm;
    }

    inline __aicore__ GemmCoord GetBlockIdCoord(int32_t loopOffset, int32_t mLoop, int32_t nLoop, int32_t swizzlDirect,
                                                int32_t swizzlCount)
    {
        uint32_t kIdx = 0;
        int64_t mIdx, nIdx;
        GetSwizzledBlockIdx(loopOffset, mLoop, nLoop, swizzlDirect, swizzlCount, mIdx, nIdx);
        return GemmCoord{static_cast<uint32_t>(mIdx), static_cast<uint32_t>(nIdx), kIdx};
    }

    inline __aicore__ GemmCoord GetBlockLocCoord(GemmCoord blockIdxCoord)
    {
        return GemmCoord{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.n() * L1TileShape::N,
                         blockIdxCoord.k() * L1TileShape::K};
    }

    inline __aicore__ GemmCoord GetBlockSizeCoord(GemmCoord blockIdxCoord, GemmCoord blockLocCoord, int32_t mLoop,
                                                  int32_t mSize, int32_t nLoop, int32_t nSize, int32_t kSize)
    {
        uint32_t mActual = (blockIdxCoord.m() == (mLoop - 1)) ? (mSize - blockLocCoord.m()) : L1TileShape::M;
        uint32_t nActual = (blockIdxCoord.n() == (nLoop - 1)) ? (nSize - blockLocCoord.n()) : L1TileShape::N;
        uint32_t kActual = kSize;
        return GemmCoord{mActual, nActual, kActual};
    }

    inline __aicore__ void FixpipeMatmul(Params const &params)
    {
        AscendC::GlobalTensor<ElemScale> gmScale;
        AscendC::GlobalTensor<EAInt8> gmAI8;
        AscendC::GlobalTensor<ElementBInt8> gmBI8;
        AscendC::GlobalTensor<ElemCHalf> gmCHalf;
        gmScale.SetGlobalBuffer((__gm__ ElemScale *)params.ptrScale);
        gmAI8.SetGlobalBuffer((__gm__ EAInt8 *)params.ptrA);
        gmBI8.SetGlobalBuffer((__gm__ ElementBInt8 *)params.ptrB);
        gmCHalf.SetGlobalBuffer((__gm__ ElemCHalf *)params.ptrC);

        LayScl layoutScale{static_cast<uint32_t>(params.problemShape.n())};

        FixpipeBlockMmad fixpipeBlockMmad(resource);
        int32_t fixedBlkSz = L1TileShape::M * L1TileShape::N;
        CommColumnSplitter fixTaskSplit(loop_num_per_comm, m_loop,
                                     params.swizzlDirect ? params.swizzlCount : 1, n_loop);

        for (int32_t calcIdx = 0; calcIdx < calCount; calcIdx++) {
            int32_t colStartPos = fixTaskSplit.GetCulumnStartPos(calcIdx);
            int32_t colEndPos = calcIdx == calCount - 1 ? n_loop : fixTaskSplit.GetCulumnEndPos(calcIdx);
            int32_t nextColEndPos = calcIdx == calCount - 2 ? n_loop : fixTaskSplit.GetCulumnEndPos(calcIdx + 1);

            int32_t syncFlagIdx = calcIdx % MAX_BLOCK_COUNT_SM;
            for (int32_t p = 0; p < params.pValue; p++) {
                int32_t fixLoopIdx = calcIdx * loop_num_per_comm + p * core_num + block_id;
                if (fixLoopIdx >= coreLoops) {
                    break;
                }
                GemmCoord blockIdxCoord =
                    GetBlockIdCoord(fixLoopIdx, m_loop, n_loop, params.swizzlDirect, params.swizzlCount);
                GemmCoord blockLocCoord_ = GetBlockLocCoord(blockIdxCoord);
                GemmCoord blockSizeCoord_ =
                    GetBlockSizeCoord(blockIdxCoord, blockLocCoord_, m_loop, params.problemShape.m(), n_loop,
                                      params.problemShape.n(), params.problemShape.k());

                MatrixCoord offsetA_{blockLocCoord_.m(), blockLocCoord_.k()};
                MatrixCoord offsetB_{blockLocCoord_.k(), blockLocCoord_.n()};
                int64_t gmOffsetA_ = params.layoutA.GetOffset(offsetA_);
                int64_t gmOffsetB_ = params.layoutB.GetOffset(offsetB_);
                int64_t gmOffsetC_;

                int32_t nCommBlock = -1;
                LayoutC layoutTmp;
                // 计算切片的layout和offset
                if (blockIdxCoord.n() < colEndPos) {
                    //当前切片属于下一次通信的范围，放入colStartPos~colEndPos的通信矩阵切块中。colStartPos~colEndPos的通信矩阵切块是nd排布。
                    nCommBlock = colEndPos - colStartPos;
                    int32_t comm_fixLoopIdx = fixLoopIdx - m_loop * colStartPos;
                    layoutTmp = {static_cast<uint32_t>(params.problemShape.m()),
                                 static_cast<uint32_t>(nCommBlock * L1TileShape::N)};
                    GemmCoord comm_blockIdxCoord_ =
                        GetBlockIdCoord(comm_fixLoopIdx, m_loop, nCommBlock, params.swizzlDirect, params.swizzlCount);
                    if (params.swizzlDirect && ((colStartPos / params.swizzlCount) & 1)) {
                        comm_blockIdxCoord_.m() = m_loop - comm_blockIdxCoord_.m() - 1;
                    }
                    GemmCoord comm_blockLocCoord_ = GetBlockLocCoord(comm_blockIdxCoord_);
                    MatrixCoord offsetC_comm_{comm_blockLocCoord_.m(), comm_blockLocCoord_.n()};
                    gmOffsetC_ = L1TileShape::N * colStartPos * params.problemShape.m() + layoutTmp.GetOffset(offsetC_comm_);
                } else {
                    // 当前切片属于下下一次通信的范围，放入colEndPos~nextColEndPos的通信矩阵切块中。
                    nCommBlock = nextColEndPos - colEndPos;
                    int32_t comm_fixLoopIdx = fixLoopIdx - m_loop * colEndPos;
                    layoutTmp = {static_cast<uint32_t>(params.problemShape.m()),
                                 static_cast<uint32_t>(nCommBlock * L1TileShape::N)};
                    GemmCoord comm_blockIdxCoord_ =
                        GetBlockIdCoord(comm_fixLoopIdx, m_loop, nCommBlock, params.swizzlDirect, params.swizzlCount);
                    if (params.swizzlDirect && ((colEndPos / params.swizzlCount) & 1) & 1) {
                        comm_blockIdxCoord_.m() = m_loop - comm_blockIdxCoord_.m() - 1;
                    }
                    GemmCoord comm_blockLocCoord_ = GetBlockLocCoord(comm_blockIdxCoord_);
                    MatrixCoord offsetC_comm_{comm_blockLocCoord_.m(), comm_blockLocCoord_.n()};
                    gmOffsetC_ = L1TileShape::N * colEndPos * params.problemShape.m() + layoutTmp.GetOffset(offsetC_comm_);
                }

                bool isFirstBlock = fixLoopIdx == block_id;
                bool hasNextBlock = false;
                GemmCoord nextBlockIdCoord;
                GemmCoord nextBlockLocCoord_;
                GemmCoord nextBlockSizeCoord_;
                int32_t nextLoopIdx = fixLoopIdx + core_num;
                int32_t nextDstRankIdx = nextLoopIdx % params.rankSize;
                int32_t nextInRankIdx = nextLoopIdx / params.rankSize;
                if (nextLoopIdx < coreLoops) {
                    hasNextBlock = true;
                    nextBlockIdCoord =
                        GetBlockIdCoord(nextLoopIdx, m_loop, n_loop, params.swizzlDirect, params.swizzlCount);
                    nextBlockLocCoord_ = GetBlockLocCoord(nextBlockIdCoord);
                    nextBlockSizeCoord_ =
                        GetBlockSizeCoord(nextBlockIdCoord, nextBlockLocCoord_, m_loop, params.problemShape.m(), n_loop,
                                          params.problemShape.n(), params.problemShape.k());
                }
                MatrixCoord offsetNextA_{nextBlockLocCoord_.m(), nextBlockLocCoord_.k()};
                MatrixCoord offsetNextB_{nextBlockLocCoord_.k(), nextBlockLocCoord_.n()};
                int64_t gmOffsetNextA_ = params.layoutA.GetOffset(offsetNextA_);
                int64_t gmOffsetNextB_ = params.layoutB.GetOffset(offsetNextB_);

                int64_t scaleGmOffset = blockLocCoord_.n();
                fixpipeBlockMmad(gmAI8[gmOffsetA_], params.layoutA, gmBI8[gmOffsetB_], params.layoutB,
                                 gmCHalf[gmOffsetC_], layoutTmp, gmScale[scaleGmOffset], layoutScale,
                                 gmAI8[gmOffsetNextA_], gmBI8[gmOffsetNextB_], blockSizeCoord_, nextBlockSizeCoord_,
                                 isFirstBlock, hasNextBlock);
            }
            FFTSCrossCoreSync<PIPE_FIX, 2>(syncFlagIdx);
        }
    }

    inline __aicore__ void Matmul(Params const &params)
    {
        AscendC::GlobalTensor<EA> gmA;
        AscendC::GlobalTensor<ElementB> gmB;
        AscendC::GlobalTensor<ElemC> gmC;
        AscendC::GlobalTensor<ElementBias> gmBias;
        gmA.SetGlobalBuffer((__gm__ EA *)params.ptrA);
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        gmC.SetGlobalBuffer((__gm__ ElemC *)params.ptrC);
        if constexpr (HasBias) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);
        }

        BlockMmad blockMmad(resource);
        int32_t blockSize = L1TileShape::M * L1TileShape::N;
        CommColumnSplitter taskSplit(loop_num_per_comm, m_loop,
                                     params.swizzlDirect ? params.swizzlCount : 1, n_loop);

        for (int32_t calIdx = 0; calIdx < calCount; calIdx++) {
            int32_t colStartPos = fixTaskSplit.GetCulumnStartPos(calcIdx);
            int32_t colEndPos = calcIdx == calCount - 1 ? n_loop : fixTaskSplit.GetCulumnEndPos(calcIdx);
            int32_t nextColEndPos = calcIdx == calCount - 2 ? n_loop : fixTaskSplit.GetCulumnEndPos(calcIdx + 1);

            int32_t syncFlagIdx = calcIdx % MAX_BLOCK_COUNT_SM;
            for (int32_t p = 0; p < params.pValue; p++) {
                int32_t fixLoopIdx = calcIdx * loop_num_per_comm + p * core_num + block_id;
                if (fixLoopIdx >= coreLoops) {
                    break;
                }
                GemmCoord blockIdxCoord =
                    GetBlockIdCoord(loopIdx, m_loop, n_loop, params.swizzlDirect, params.swizzlCount);
                GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
                GemmCoord blockSizeCoord =
                    GetBlockSizeCoord(blockIdxCoord, blockLocCoord, m_loop, params.problemShape.m(), n_loop,
                                      params.problemShape.n(), params.problemShape.k());

                MatrixCoord offsetA{blockLocCoord.m(), blockLocCoord.k()};
                MatrixCoord offsetB{blockLocCoord.k(), blockLocCoord.n()};
                int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
                int64_t gmOffsetC;
                AscendC::GlobalTensor<ElemC> gmDst;
                gmDst = gmC;

                int32_t n_comm_block = -1;
                LayoutC layout_tmp;
                // 计算切片的layout和offset
                if (blockIdxCoord.n() < n_ed) {
                    //当前切片属于下一次通信的范围，放入n_st~n_ed的通信矩阵切块中。n_st~n_ed的通信矩阵切块是nd排布。
                    n_comm_block = n_ed - n_st;
                    int32_t comm_loopIdx = loopIdx - m_loop * n_st;
                    int32_t n_len = n_comm_block * L1TileShape::N;
                    layout_tmp = {static_cast<uint32_t>(params.problemShape.m()), static_cast<uint32_t>(n_len)};
                    GemmCoord comm_blockIdxCoord =
                        GetBlockIdCoord(comm_loopIdx, m_loop, n_comm_block, params.swizzlDirect, params.swizzlCount);
                    if (params.swizzlDirect && ((n_st / params.swizzlCount) & 1)) {
                        comm_blockIdxCoord.m() = m_loop - comm_blockIdxCoord.m() - 1;
                    }
                    GemmCoord comm_blockLocCoord = GetBlockLocCoord(comm_blockIdxCoord);
                    MatrixCoord offsetC_comm{comm_blockLocCoord.m(), comm_blockLocCoord.n()};
                    gmOffsetC = L1TileShape::N * n_st * params.problemShape.m() + layout_tmp.GetOffset(offsetC_comm);
                } else {
                    // 当前切片属于下下一次通信的范围，放入n_ed~n_nxt_ed的通信矩阵切块中。
                    n_comm_block = n_nxt_ed - n_ed;
                    int32_t comm_loopIdx = loopIdx - m_loop * n_ed;
                    int32_t n_len = n_comm_block * L1TileShape::N;
                    layout_tmp = {static_cast<uint32_t>(params.problemShape.m()), static_cast<uint32_t>(n_len)};
                    GemmCoord comm_blockIdxCoord =
                        GetBlockIdCoord(comm_loopIdx, m_loop, n_comm_block, params.swizzlDirect, params.swizzlCount);
                    if (params.swizzlDirect && ((n_ed / params.swizzlCount) & 1) & 1) {
                        comm_blockIdxCoord.m() = m_loop - comm_blockIdxCoord.m() - 1;
                    }
                    GemmCoord comm_blockLocCoord = GetBlockLocCoord(comm_blockIdxCoord);
                    MatrixCoord offsetC_comm{comm_blockLocCoord.m(), comm_blockLocCoord.n()};
                    gmOffsetC = L1TileShape::N * n_ed * params.problemShape.m() + layout_tmp.GetOffset(offsetC_comm);
                }

                if constexpr (HasBias) {
                    blockMmad(
                        gmA[gmOffsetA], params.layoutA,
                        gmB[gmOffsetB], params.layoutB,
                        gmC[gmOffsetC], layout_tmp,
                        gmBias[blockLocCoord.n()], blockSizeCoord);
                } else {

                    bool isFirstBlock = loopIdx == block_id;
                    bool hasNextBlock = false;
                    GemmCoord nextBlockIdCoord;
                    GemmCoord nextBlockLocCoord;
                    GemmCoord nextBlockSizeCoord;
                    int32_t nextLoopIdx = loopIdx + core_num;
                    int32_t nextDstRankIdx = nextLoopIdx % params.rankSize;
                    int32_t nextInRankIdx = nextLoopIdx / params.rankSize;
                    if (nextLoopIdx < coreLoops) {
                        hasNextBlock = true;
                        nextBlockIdCoord =
                            GetBlockIdCoord(nextLoopIdx, m_loop, n_loop, params.swizzlDirect, params.swizzlCount);
                        nextBlockLocCoord = GetBlockLocCoord(nextBlockIdCoord);
                        nextBlockSizeCoord =
                            GetBlockSizeCoord(nextBlockIdCoord, nextBlockLocCoord, m_loop, params.problemShape.m(),
                                n_loop, params.problemShape.n(), params.problemShape.k());
                    }
                    MatrixCoord offsetNextA{nextBlockLocCoord.m(), nextBlockLocCoord.k()};
                    MatrixCoord offsetNextB{nextBlockLocCoord.k(), nextBlockLocCoord.n()};
                    int64_t gmOffsetNextA = params.layoutA.GetOffset(offsetNextA);
                    int64_t gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);

                    blockMmad(gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmDst[gmOffsetC],
                              layout_tmp, gmA[gmOffsetNextA], gmB[gmOffsetNextB], blockSizeCoord, nextBlockSizeCoord,
                              isFirstBlock, hasNextBlock);
                }
            }
            FFTSCrossCoreSync<PIPE_FIX, 2>(flagIdx);
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);

        InitArgs(params);

        if (params.needFixpipe) {
            FixpipeMatmul(params);
        } else {
            Matmul(params);
        }
    }

private:
    int32_t coreLoops;
    int32_t calCount;
    int32_t finalM;
    int32_t kAlignment;
    int32_t mLoopPerRank;
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};
} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_REDUCE_SCATTER_AIV_MODE_SMALLM_HPP