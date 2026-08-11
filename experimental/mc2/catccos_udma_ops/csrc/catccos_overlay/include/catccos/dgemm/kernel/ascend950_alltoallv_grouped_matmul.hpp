/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALLV_GROUPED_MATMUL_HPP
#define CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALLV_GROUPED_MATMUL_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/catccos.hpp"
#include "catccos/layout/dist_matrix.hpp"
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif
namespace Catccos::DGemm::Kernel {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <
    class ProblemShape_,
    class BlockMmad_,
    class BlockComm_,
    class BlockMmadScheduler_,
    class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_,
    bool USE_UDMA_PEER_SERIAL_SCHEDULE_ = false
>
class Ascend950AllToAllVGroupedMatmul {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockComm = BlockComm_;
    using BlockCommParams = typename BlockComm::Params;

    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using LayoutTagC = typename BlockMmad::TileCopy::LayoutTagC;

    using BlockMmadScheduler = BlockMmadScheduler_;
    using BlockCommScheduler = BlockCommScheduler_;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr bool USE_UDMA_PEER_SERIAL_SCHEDULE = USE_UDMA_PEER_SERIAL_SCHEDULE_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        ProblemShape problemShape;
        uint32_t commInterval;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementC *ptrC;
        LayoutC layoutC;
        LayoutTagA layoutTagA;
        LayoutTagB layoutTagB;
        GM_ADDR ptrSymmetric;
        GM_ADDR syncMmadFinish;
        GM_ADDR syncCommFinish;
        BlockCommParams blockCommParams;

        CATLASS_HOST_DEVICE
        Params() {};

        CATLASS_HOST_DEVICE
        Params(
            ProblemShape const &problemShape_,
            uint32_t commInterval_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            LayoutTagA layoutTagA_, LayoutTagB layoutTagB_,
            GM_ADDR ptrSymmetric_, GM_ADDR syncMmadFinish_, GM_ADDR syncCommFinish_,
            BlockCommParams const &allToAllVGmmParams_
        ) : problemShape(problemShape_),
            commInterval(commInterval_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            layoutTagA(layoutTagA_), layoutTagB(layoutTagB_),
            ptrSymmetric(ptrSymmetric_), syncMmadFinish(syncMmadFinish_), syncCommFinish(syncCommFinish_),
            blockCommParams(allToAllVGmmParams_)
        {
        }
    };

    /// User-facing arguments
    struct Arguments {
        Catlass::GemmCoord gemmShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        uint32_t epSize;
        uint32_t expertNum;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrLocalTokensPerExpert;
        GM_ADDR ptrGlobalTokensPerLocalExpert;
        GM_ADDR ptrSymmetric;
        Catlass::MatrixCoord commBlockShape;
        Catlass::MatrixCoord commTileShape;
    };

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t *workspace = nullptr) {
        LayoutTagA tagA{args.gemmShape.m(), args.gemmShape.k()};
        LayoutTagB tagB{args.gemmShape.k(), args.gemmShape.n()};
        LayoutTagC tagC{args.gemmShape.m() * args.rankSize, args.gemmShape.n()};

        auto layoutA = tla::MakeLayoutFromTag(tagA);
        auto layoutB = tla::MakeLayoutFromTag(tagB);
        auto layoutC = tla::MakeLayoutFromTag(tagC);

        ProblemShape problemShape{
            args.gemmShape, args.rankSize, args.rankIdx, args.epSize, args.expertNum,
            args.ptrLocalTokensPerExpert, args.ptrGlobalTokensPerLocalExpert
        };

        typename BlockComm::TileRemoteCopy::Params tileParams{args.commTileShape};
        BlockCommParams blockCommParams{args.commBlockShape, tileParams};

        constexpr size_t IPC_BUFF_MAX_SIZE = 200 * 1024 * 1024 * sizeof(half);
        constexpr size_t SYNC_UNIT_SIZE = 4 * sizeof(int64_t);
        uint64_t symmetricOffset = 0;
        auto gmSymmetric = args.ptrSymmetric + symmetricOffset;
        symmetricOffset += IPC_BUFF_MAX_SIZE;
        auto syncMmadFinish = args.ptrSymmetric + symmetricOffset;
        symmetricOffset += SYNC_UNIT_SIZE;
        auto syncCommFinish = args.ptrSymmetric + symmetricOffset;

        return Params(
            problemShape,
            args.commInterval,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrC, layoutC,
            tagA, tagB,
            gmSymmetric, syncMmadFinish, syncCommFinish,
            blockCommParams
        );
    }

    CATLASS_DEVICE
    Ascend950AllToAllVGroupedMatmul()
    {
#ifdef ENABLE_TIMER
        __gm__ uint8_t* timer_buffer = GetTimerBuffer();
        if (timer_buffer != nullptr) {
            timer.Init(timer_buffer);
            timer.Tik();
        }
#endif
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishComm[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    CATLASS_DEVICE
    ~Ascend950AllToAllVGroupedMatmul()
    {
#ifdef ENABLE_TIMER
        timer.Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
#endif
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockMmad mmad{resource};

        GemmCoord blockShape = GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        uint32_t commShapeM = params.commInterval * L1_TILE_M;
        MatrixCoord commShape{commShapeM, params.problemShape.k()};
        BlockMmadScheduler scheduler{params.problemShape, commShapeM, blockShape.GetCoordMN()};

        auto rankSize = params.problemShape.rankSize();
        auto layoutTagSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * rankSize * commShapeM, params.problemShape.k());
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, rankSize, commShapeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);
        auto layoutSymmetric = tla::MakeLayoutFromTag(layoutTagSymmetric);

        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);

        auto tensorSymmetric = tla::MakeTensor(gmSymmetric, layoutSymmetric, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Catlass::Arch::PositionGM{});

        auto rankIdx = params.problemShape.rankIdx();

        auto commLoops = scheduler.GetCommLoops();
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            scheduler.UpdateCommContext(commIdx);
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;

            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishComm[stageIdx]);

#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIC);
#endif

            for (uint32_t localExpertIdx = 0; localExpertIdx < params.problemShape.localExpertNum(); ++localExpertIdx) {
                auto const &layoutB = params.layoutTagB;
                size_t localExpertOffsetB = localExpertIdx * layout::Capacity(layoutB);
                auto tensorB = tla::MakeTensor(gmB[localExpertOffsetB], params.layoutB, Catlass::Arch::PositionGM{});

                for (uint32_t srcRankIdx = 0; srcRankIdx < params.problemShape.rankSize(); ++srcRankIdx) {
                    scheduler.UpdateMmadContext(localExpertIdx, srcRankIdx);
                    auto remapperA = scheduler.GetRemapperA(commIdx, localExpertIdx, srcRankIdx);
                    auto remapperC = scheduler.GetRemapperC(commIdx, localExpertIdx, srcRankIdx);

                    for (auto iter = scheduler.Begin(); !iter.End(); iter.Next()) {
                        auto blockOffset = scheduler.GetBlockOffset(iter);

                        MatrixCoord commOffsetA{layoutSymmetricRow(Catlass::MakeCoord<int>(stageIdx, srcRankIdx, 0)), 0};
                        auto blockOffsetA = remapperA(blockOffset);
                        MatrixCoord blockOffsetB = blockOffset.GetCoordKN();
                        auto blockOffsetC = remapperC(blockOffset);
                        auto actualBlockShape = scheduler.RemapActualBlockShape(blockOffset, remapperA, remapperC);

                        auto tensorBlockA = GetTile(tensorSymmetric,
                                                    tla::MakeCoord(blockOffsetA.row() + commOffsetA.row(), blockOffsetA.column()),
                                                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                        auto tensorBlockB = GetTile(tensorB,
                                                    tla::MakeCoord(blockOffsetB.row(), blockOffsetB.column()),
                                                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                        auto tensorBlockC = GetTile(tensorC,
                                                    tla::MakeCoord(blockOffsetC.row(), blockOffsetC.column()),
                                                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                        mmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                    }
                }
            }

#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIC);
#endif
            if (commIdx < commLoops - WORKSPACE_STAGES && commLoops >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageIdx]);
            }
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        BlockComm blockRemoteCopy(resource, params.blockCommParams);

        uint32_t commShapeM = params.commInterval * L1_TILE_M;
        MatrixCoord commShape{commShapeM, params.problemShape.k()};
        MatrixCoord blockShape = params.blockCommParams.BlockShape();
        BlockCommScheduler scheduler{params.problemShape, commShapeM, blockShape};

        auto const &layoutA = params.layoutTagA;
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);

        auto rankSize = params.problemShape.rankSize();
        auto layoutTagSymmetric = layout::DistRowMajor::MakeAlignedLayout<ElementA>(commShape, rankSize);
        AscendC::GlobalTensor<ElementA> gmSymmetricList[WORKSPACE_STAGES];
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(ptrSymmetric + stageIdx * layout::Capacity(layoutTagSymmetric));
        }

        auto syncMmadFinish = reinterpret_cast<__gm__ int32_t *>(params.syncMmadFinish);
        auto syncCommFinish = reinterpret_cast<__gm__ int32_t *>(params.syncCommFinish);

        // 閫夋嫨鍞竴鐨?aicore锛屽褰撳墠 rank 鐨勪俊鍙峰湴鍧€杩涜鍒濆鍖?        bool isRootCore = (AscendC::GetBlockIdx() == 0);
        auto rankIdx = params.problemShape.rankIdx();
        if (isRootCore) {
            aclshmemx_signal_op(syncMmadFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
            aclshmemx_signal_op(syncCommFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
        }
        aclshmemx_barrier_all_vec();

        uint32_t receiveAccum = 0;
        auto commLoops = scheduler.GetCommLoops();
        blockRemoteCopy.InitBlockLoop();
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const &gmSymmetric = gmSymmetricList[stageIdx];

            // 棰勮鏈疆杩囧悗锛屽綋鍓?rank 灏嗙疮璁℃敹鍒板灏?token
            receiveAccum += scheduler.GetActualReceiveAccum(commIdx);
            auto remapperSrc = scheduler.GetRemapperSrc(commIdx);

            // 绛夊緟涓婁竴涓绠楄疆娆℃墍鏈?aicore 鐨勮绠楀畬鎴?            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);
                Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            }
#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIV);
#endif
            // 閫夋嫨鍞竴鐨?aicore锛屽皢褰撳墠 rank 鐨?syncMmadFinish 璁剧疆涓?commIdx + 1锛岃〃绀轰笂涓€涓€氫俊杞璁＄畻浠诲姟宸插畬鎴?            if (isRootCore) {
                aclshmemx_signal_op(syncMmadFinish, commIdx + 1, ACLSHMEM_SIGNAL_SET, rankIdx);
            }

            if constexpr (USE_UDMA_PEER_SERIAL_SCHEDULE) {
                // A UDMA queue is owned by one AIV subcore for each destination PE.
                // Sharing a PE and sync-id between many AIV cores can corrupt or
                // drop requests. Keep local MTE on subcore 0 and remote UDMA puts
                // on subcore 1, as done by the Ascend950 UDMA all-gather kernels.
                uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
                uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
                if (aicoreIdx < rankSize) {
                    uint32_t dstRankIdx = aicoreIdx;
                    bool isLocalCopy = dstRankIdx == rankIdx;
                    bool ownsPeer = (isLocalCopy && subcoreIdx == 0) ||
                                    (!isLocalCopy && subcoreIdx == 1);
                    if (ownsPeer) {
                        // Wait once, then submit this destination's blocks serially.
                        auto remoteSyncMmadFinish =
                            static_cast<__gm__ int32_t *>(shmem_ptr(syncMmadFinish, dstRankIdx));
                        aclshmem_signal_wait_until(remoteSyncMmadFinish, ACLSHMEM_CMP_EQ, commIdx + 1);

                        if (isLocalCopy) {
                            // The MTE path stages data through a bounded UB tile, so
                            // retain the regular communication-block granularity.
                            for (uint32_t blockIdx = 0; blockIdx < scheduler.GetBlockCount(); ++blockIdx) {
                                auto blockOffset = scheduler.GetBlockOffset(blockIdx, dstRankIdx);
                                auto blockOffsetSrc = remapperSrc(blockOffset);
                                auto blockOffsetDst = blockOffset.GetLocalCoord();
                                auto actualBlockShape = scheduler.RemapActualBlockShape(blockOffset, remapperSrc);
                                if (Numel(actualBlockShape) == 0) {
                                    continue;
                                }

                                auto gmBlockSrc = gmA[layoutA.GetOffset(blockOffsetSrc)];
                                auto gmBlockDst = gmSymmetric[layoutTagSymmetric.GetOffset(blockOffsetDst)];
                                auto layoutBlockSrc = layoutA.GetTileLayout(actualBlockShape);
                                auto layoutBlockDst = layoutTagSymmetric.GetTileLayout(actualBlockShape);

                                blockRemoteCopy(
                                    gmBlockSrc, layoutBlockSrc,
                                    gmBlockDst, layoutBlockDst,
                                    actualBlockShape, dstRankIdx
                                );
                                aclshmem_fence();
                                aclshmemx_signal_op(syncCommFinish, 1, ACLSHMEM_SIGNAL_ADD, dstRankIdx);
                            }
                        } else {
                            // UDMA does not need the MTE communication block. Submit
                            // the whole contiguous slice once, matching the proven
                            // Ascend950 UDMA AllGather schedule. Besides reducing WQE
                            // pressure, quiet now observes the complete peer slice.
                            auto commOffset = scheduler.GetBlockOffset(0, dstRankIdx);
                            auto actualCommShape = remapperSrc.GetResidueShape(commOffset);
                            if (Numel(actualCommShape) != 0) {
                                auto commOffsetSrc = remapperSrc(commOffset);
                                auto commOffsetDst = commOffset.GetLocalCoord();
                                auto gmCommSrc = gmA[layoutA.GetOffset(commOffsetSrc)];
                                auto gmCommDst = gmSymmetric[layoutTagSymmetric.GetOffset(commOffsetDst)];
                                auto layoutCommSrc = layoutA.GetTileLayout(actualCommShape);
                                auto layoutCommDst = layoutTagSymmetric.GetTileLayout(actualCommShape);

                                blockRemoteCopy(
                                    gmCommSrc, layoutCommSrc,
                                    gmCommDst, layoutCommDst,
                                    actualCommShape, dstRankIdx
                                );
                                aclshmem_fence();
                                auto completedBlocks = static_cast<int32_t>(
                                    Numel(CeilDiv(actualCommShape, blockShape)));
                                aclshmemx_signal_op(
                                    syncCommFinish, completedBlocks, ACLSHMEM_SIGNAL_ADD, dstRankIdx);
                            }
                        }
                    }
                }
            } else {
                for (auto iter = scheduler.Begin(commIdx); !iter.End(); iter.Next()) {
                    auto blockOffset = scheduler.GetBlockOffset(iter);

                    auto blockOffsetSrc = remapperSrc(blockOffset);
                    auto blockOffsetDst = blockOffset.GetLocalCoord();
                    auto actualBlockShape = scheduler.RemapActualBlockShape(blockOffset, remapperSrc);
                    if (Numel(actualBlockShape) == 0) {
                        continue;
                    }

                    auto gmBlockSrc = gmA[layoutA.GetOffset(blockOffsetSrc)];
                    auto gmBlockDst = gmSymmetric[layoutTagSymmetric.GetOffset(blockOffsetDst)];

                    auto layoutBlockSrc = layoutA.GetTileLayout(actualBlockShape);
                    auto layoutBlockDst = layoutTagSymmetric.GetTileLayout(actualBlockShape);

                    auto dstRankIdx = blockOffset.remote();
                    auto remoteSyncMmadFinish =
                        static_cast<__gm__ int32_t *>(shmem_ptr(syncMmadFinish, dstRankIdx));
                    aclshmem_signal_wait_until(remoteSyncMmadFinish, ACLSHMEM_CMP_EQ, commIdx + 1);
                    blockRemoteCopy(
                        gmBlockSrc, layoutBlockSrc,
                        gmBlockDst, layoutBlockDst,
                        actualBlockShape, dstRankIdx
                    );

                    aclshmem_fence();
                    aclshmemx_signal_op(syncCommFinish, 1, ACLSHMEM_SIGNAL_ADD, dstRankIdx);
                }
            }

            // 绛夊緟褰撳墠 commIdx 涓嬶紝鎵€鏈夊啓鍒板綋鍓?rank symmetric 涓婄殑鏁版嵁閮藉啓瀹?            if (isRootCore) {
                aclshmem_signal_wait_until(syncCommFinish, ACLSHMEM_CMP_EQ, receiveAccum);
            }
            // 绛夊緟褰撳墠閫氫俊杞涓嬫暟鎹帴鏀跺畬鎴愬悗锛屾墍鏈夌殑 aic 閫氱煡 aiv 寮€濮嬭绠楁湰杞殑
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIV);
#endif
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComm[stageIdx]);
        }
        blockRemoteCopy.FinalizeBlockLoop();
    }

private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishComm[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
#ifdef ENABLE_TIMER
    AscendTimerDevice timer;
#endif
};

}  // namespace Catccos::DGemm::Kernel

#endif
