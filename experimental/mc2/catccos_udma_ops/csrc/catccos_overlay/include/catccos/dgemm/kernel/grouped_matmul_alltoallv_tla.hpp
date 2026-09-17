/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_TLA_HPP
#define CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_TLA_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/catccos.hpp"
#include "catccos/dgemm/kernel/udma_workspace_layout.hpp"
#include "catccos/symm_coord.hpp"
#include "catccos/layout/dist_matrix.hpp"

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
    uint64_t SYMMETRIC_DATA_BYTES_ =
        OpsTransformer::CatccosUdma::UdmaWorkspaceLayout::kDataBytes,
    bool NO_REUSE_STAGING_ = false
>
class GroupedMatmulAllToAllVTla {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using LayoutTagC = typename BlockMmad::TileCopy::LayoutTagC;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockComm = BlockComm_;
    using BlockCommParams = typename BlockComm::Params;

    using ElementComm = typename BlockComm::ElementSrc;
    using LayoutTagComm = typename BlockComm::LayoutSrc;

    using BlockMmadScheduler = BlockMmadScheduler_;
    using BlockCommScheduler = BlockCommScheduler_;
    using BlockCommSchedulerParams = typename BlockCommScheduler::Params;
    using WorkspaceLayout =
        OpsTransformer::CatccosUdma::UdmaWorkspaceLayout;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    // FFTS exposes flag IDs 0..7. The 19/22-stage production profiles
    // still use only two flag IDs for the sequential AIC/AIV handshake.
    static constexpr uint32_t SYNC_FLAG_STAGES =
        WORKSPACE_STAGES < 2 ? WORKSPACE_STAGES : 2;
    static constexpr uint64_t SYMMETRIC_DATA_BYTES = SYMMETRIC_DATA_BYTES_;
    static constexpr bool NO_REUSE_STAGING = NO_REUSE_STAGING_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    using MoeConstraints = typename BlockMmadScheduler::MoeConstraints;

    struct Params {
        ProblemShape problemShape;
        uint32_t commInterval;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementC *ptrC;
        LayoutC layoutC;
        LayoutTagComm layoutTagComm;
        GM_ADDR ptrSymmetric;
        GM_ADDR syncMmadFinish;
        GM_ADDR syncCommFinish;
        BlockCommParams blockCommParams;
        BlockCommSchedulerParams blockCommSchedulerParams;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(
            ProblemShape const &problemShape_, uint32_t commInterval_,
            LayoutTagComm layoutTagComm_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR ptrSymmetric_, GM_ADDR syncMmadFinish_, GM_ADDR syncCommFinish_,
            BlockCommParams const &blockCommParams_,
            BlockCommSchedulerParams const &blockCommSchedulerParams_
        ) : problemShape(problemShape_), commInterval(commInterval_),
            layoutTagComm(layoutTagComm_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            ptrSymmetric(ptrSymmetric_), syncMmadFinish(syncMmadFinish_), syncCommFinish(syncCommFinish_),
            blockCommParams(blockCommParams_),
            blockCommSchedulerParams(blockCommSchedulerParams_)
        {
        }
    };

    /// User API arguments
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
        MatrixCoord commCoreSplit;
        MatrixCoord commBlockShape;
        MatrixCoord commTileShape;
        // The number of rows returned to this rank after AllToAllV. It is
        // independent of the number of input rows on an imbalanced route.
        // Zero preserves the legacy balanced-route layout. It is also safe for
        // a zero-token rank because the scheduler emits no C writes.
        uint32_t outputRows{0};
    };

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t *workspace = nullptr)
    {
        LayoutTagA layoutTagA{args.gemmShape.m(), args.gemmShape.k()};
        LayoutTagB layoutTagB{args.gemmShape.k(), args.gemmShape.n()};
        uint32_t outputRows = args.outputRows == 0 ? args.gemmShape.m() : args.outputRows;
        LayoutTagC layoutTagC{outputRows, args.gemmShape.n()};

        ProblemShape problemShape{
            args.gemmShape, args.rankSize, args.rankIdx, args.epSize, args.expertNum,
            args.ptrLocalTokensPerExpert, args.ptrGlobalTokensPerLocalExpert
        };

        auto layoutA = tla::MakeLayoutFromTag(layoutTagA);
        auto layoutB = tla::MakeLayoutFromTag(layoutTagB);
        auto layoutC = tla::MakeLayoutFromTag(layoutTagC);

        static_assert(
            SYMMETRIC_DATA_BYTES <= WorkspaceLayout::kDataBytes,
            "grouped-matmul AllToAllV staging exceeds the data plane");

        auto gmSymmetric = args.ptrSymmetric;
        auto syncMmadFinish =
            args.ptrSymmetric + WorkspaceLayout::kGmmA2avSyncMmadOffset;
        auto syncCommFinish =
            args.ptrSymmetric + WorkspaceLayout::kGmmA2avSyncCommOffset;

        typename BlockComm::TileRemoteCopy::Params tileParams{args.commTileShape};
        BlockCommParams blockCommParams{args.commBlockShape, tileParams};
        BlockCommSchedulerParams blockCommSchedulerParams{args.commCoreSplit};

        return Params(
            problemShape,
            args.commInterval,
            layoutTagC,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrC, layoutC,
            gmSymmetric, syncMmadFinish, syncCommFinish,
            blockCommParams,
            blockCommSchedulerParams
        );
    }

    CATLASS_DEVICE
    GroupedMatmulAllToAllVTla()
    {
        for (uint32_t stageIdx = 0; stageIdx < SYNC_FLAG_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishComm[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        uint32_t rankSize = params.problemShape.rankSize();
        uint32_t blockPerComm = AscendC::GetBlockNum() * params.commInterval;
        uint32_t tilePerCommInRank = blockPerComm / rankSize;
        uint32_t nLoops = CeilDiv(params.problemShape.n(), L1_TILE_N);
        uint32_t groupsPerCommInRank = Max<uint32_t>(1U, tilePerCommInRank / nLoops);
        GemmCoord mmadBlockShape{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        BlockMmadScheduler scheduler{
            params.problemShape, groupsPerCommInRank, mmadBlockShape.GetCoordMN()};

        uint32_t stageRows = groupsPerCommInRank * L1_TILE_M;
        auto layoutTagSymmetric = Catlass::layout::RowMajor::MakeLayout<ElementC>(
            stageRows * rankSize, params.problemShape.n());
        auto layoutSymmetric = tla::MakeLayoutFromTag(layoutTagSymmetric);
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric);
        AscendC::GlobalTensor<ElementC> gmSymmetricList[WORKSPACE_STAGES];
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(
                ptrSymmetric + stageIdx * layoutTagSymmetric.Capacity());
        }

        BlockMmad blockMmad(resource);
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Catlass::Arch::PositionGM{});

        uint32_t commLoops = scheduler.GetCommLoops();
        if constexpr (NO_REUSE_STAGING) {
            if (commLoops > WORKSPACE_STAGES) {
                return;
            }
        }
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            scheduler.UpdateCommContext(commIdx);
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            uint32_t flagIdx = commIdx % SYNC_FLAG_STAGES;
            if constexpr (!NO_REUSE_STAGING) {
                if (commIdx >= WORKSPACE_STAGES) {
                    Catlass::Arch::CrossCoreWaitFlag(
                        flagAivFinishComm[flagIdx]);
                }
            }

            auto tensorSymmetric = tla::MakeTensor(
                gmSymmetricList[stageIdx], layoutSymmetric, Catlass::Arch::PositionGM{});
            for (uint32_t localExpertIdx = 0;
                 localExpertIdx < params.problemShape.localExpertNum(); ++localExpertIdx) {
                scheduler.UpdateLocalExpertContext(localExpertIdx);
                auto remapperA = scheduler.GetRemapperA(commIdx, localExpertIdx);
                auto remapperC = scheduler.GetRemapperC(commIdx, localExpertIdx);
                size_t localExpertOffsetB = static_cast<size_t>(localExpertIdx) *
                    params.problemShape.k() * params.problemShape.n();
                auto tensorB = tla::MakeTensor(
                    gmB[localExpertOffsetB], params.layoutB, Catlass::Arch::PositionGM{});

                for (auto iter = scheduler.Begin(); !iter.End(); iter.Next()) {
                    DistGemmCoord blockOffset = scheduler.GetBlockOffset(iter);
                    uint32_t dstRankIdx = blockOffset.rank();
                    MatrixCoord stageRankOffset{dstRankIdx * stageRows, 0};
                    auto blockOffsetA = remapperA(blockOffset);
                    MatrixCoord blockOffsetB = blockOffset.GetCoordKN();
                    auto blockOffsetC = remapperC(blockOffset);
                    auto actualBlockShape = scheduler.RemapActualBlockShape(
                        blockOffset, remapperA, remapperC);

                    auto tensorBlockA = GetTile(
                        tensorA, tla::MakeCoord(blockOffsetA.row(), blockOffsetA.column()),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockB = GetTile(
                        tensorB, tla::MakeCoord(blockOffsetB.row(), blockOffsetB.column()),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockC = GetTile(
                        tensorSymmetric,
                        tla::MakeCoord(
                            blockOffsetC.row() + stageRankOffset.row(),
                            blockOffsetC.column()),
                        tla::MakeShape(
                            actualBlockShape.m(), actualBlockShape.n()));
                    blockMmad(
                        tensorBlockA, tensorBlockB, tensorBlockC,
                        actualBlockShape);
                }
            }
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad.SynchronizeBlock();
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[flagIdx]);
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        BlockComm blockRemoteCopy(resource, params.blockCommParams);
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
        uint32_t rankSize = params.problemShape.rankSize();
        uint32_t rankIdx = params.problemShape.rankIdx();
        uint32_t srcRankIdx = aicoreIdx;
        uint32_t srcEpIdx = srcRankIdx % params.problemShape.epSize();

        uint32_t blockPerComm = AscendC::GetBlockNum() * params.commInterval;
        uint32_t tilePerCommInRank = blockPerComm / rankSize;
        uint32_t nLoops = CeilDiv(params.problemShape.n(), L1_TILE_N);
        uint32_t groupsPerCommInRank = Max<uint32_t>(1U, tilePerCommInRank / nLoops);
        uint32_t stageRows = groupsPerCommInRank * L1_TILE_M;
        GemmCoord mmadBlockShape{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        MatrixCoord commBlockShape = params.blockCommParams.BlockShape();
        MatrixCoord commCoreSplit = params.blockCommSchedulerParams.CoreSplit();
        BlockCommScheduler scheduler{
            params.problemShape, groupsPerCommInRank, srcEpIdx,
            mmadBlockShape.GetCoordMN(), commBlockShape, commCoreSplit};
        auto layoutSymmetric = Catlass::layout::RowMajor(stageRows, params.problemShape.n());
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric);
        AscendC::GlobalTensor<ElementC> gmSymmetricList[WORKSPACE_STAGES];
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            uint32_t stageRankIdx = stageIdx * rankSize + rankIdx;
            gmSymmetricList[stageIdx].SetGlobalBuffer(
                ptrSymmetric + stageRankIdx * layout::Capacity(layoutSymmetric));
        }

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
        auto const &layoutC = params.layoutTagComm;

        bool isSyncCore = (subcoreIdx == 0) && (srcRankIdx == rankIdx);
        bool isCommCore = (subcoreIdx == 0) && (srcRankIdx < rankSize);
        auto syncMmadFinish = reinterpret_cast<__gm__ int32_t *>(params.syncMmadFinish);
        if (isCommCore) {
            syncMmadFinish = static_cast<__gm__ int32_t *>(
                aclshmem_ptr(params.syncMmadFinish, srcRankIdx));
        }
        auto syncCommFinish = reinterpret_cast<__gm__ int32_t *>(params.syncCommFinish);

        // All ranks must have left the previous kernel launch before any rank
        // clears the reused synchronization counters.  Otherwise a fast rank
        // can reset state that a slow rank still consumes, or republish the
        // same commIdx value and trigger a cross-launch ABA match.
        aclshmemx_barrier_all_vec();
        if (isSyncCore) {
            aclshmemx_signal_op(syncMmadFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
            if constexpr (!NO_REUSE_STAGING) {
                aclshmemx_signal_op(
                    syncCommFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
            }
        }
        // Publish the cleared state before this launch starts producing data.
        aclshmemx_barrier_all_vec();

        uint32_t receiveAccum = 0;
        uint32_t commLoops = scheduler.GetCommLoops();
        if constexpr (NO_REUSE_STAGING) {
            if (commLoops > WORKSPACE_STAGES) {
                return;
            }
        }
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            uint32_t flagIdx = commIdx % SYNC_FLAG_STAGES;
            scheduler.UpdateCommContext(commIdx);
            if constexpr (!NO_REUSE_STAGING) {
                receiveAccum += scheduler.GetActualReceiveAccum(commIdx);
            }
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[flagIdx]);
            // PIPE_FIX hands off the AIC stores; all local AIVs rendezvous
            // before the sync core publishes producer readiness to remote PEs.
            AscendC::SyncAll<true>();
            if (isSyncCore) {
                aclshmemx_signal_op(
                    syncMmadFinish, commIdx + 1, ACLSHMEM_SIGNAL_SET, rankIdx);
            }

            bool hasInput = scheduler.HasInput();
            uint32_t actualRows = scheduler.GetActualRows();
            uint32_t outputRowOffset = scheduler.GetOutputRowOffset();
            if (isCommCore && hasInput) {
                aclshmem_signal_wait_until(
                    syncMmadFinish, ACLSHMEM_CMP_EQ, commIdx + 1);

                MatrixCoord actualShape{actualRows, params.problemShape.n()};
                auto const &gmBlockSrc = gmSymmetricList[stageIdx];
                auto layoutBlockSrc = layoutSymmetric.GetTileLayout(actualShape);
                MatrixCoord outputOffset{outputRowOffset, 0};
                auto gmBlockDst = gmC[layoutC.GetOffset(outputOffset)];
                auto layoutBlockDst = layoutC.GetTileLayout(actualShape);

                blockRemoteCopy.InitBlockLoop();
                blockRemoteCopy(
                    gmBlockSrc, layoutBlockSrc,
                    gmBlockDst, layoutBlockDst,
                    actualShape, srcRankIdx);
                blockRemoteCopy.FinalizeBlockLoop();

                if constexpr (!NO_REUSE_STAGING) {
                    aclshmem_fence();
                    aclshmemx_signal_op(
                        syncCommFinish, 1,
                        ACLSHMEM_SIGNAL_ADD, srcRankIdx);
                }
            }

            if constexpr (!NO_REUSE_STAGING) {
                if (isSyncCore) {
                    aclshmem_signal_wait_until(
                        syncCommFinish, ACLSHMEM_CMP_EQ, receiveAccum);
                }
            }

            if constexpr (!NO_REUSE_STAGING) {
                if (commIdx + WORKSPACE_STAGES < commLoops) {
                    aclshmemx_barrier_all_vec();
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                        flagAivFinishComm[flagIdx]);
                }
            }
        }

        if constexpr (NO_REUSE_STAGING) {
            // Each commIdx owns a distinct producer stage, so AIC never needs
            // a per-round remote-consumer acknowledgement. Every active AIV
            // GET has already completed its local quiet before reaching this
            // point. Retire all remote reads once, immediately before any PE
            // may leave the kernel and reuse the symmetric data plane.
            aclshmemx_barrier_all_vec();
        }
    }

private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[SYNC_FLAG_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishComm[SYNC_FLAG_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

}  // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_TLA_HPP
