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

#include <type_traits>

#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/catccos.hpp"
#include "catccos/dgemm/kernel/udma_workspace_layout.hpp"
#include "catccos/dgemm/moe_route_metadata.hpp"
#include "catccos/layout/dist_matrix.hpp"

namespace Catccos::DGemm::Kernel {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <bool Enabled, class BlockExport>
class AllToAllVExportPipeline {
public:
    template <class Resource, class BlockCommParams>
    CATLASS_DEVICE
    AllToAllVExportPipeline(Resource &, BlockCommParams const &) {}
};

template <class BlockExport>
class AllToAllVExportPipeline<true, BlockExport> {
public:
    template <class Resource, class BlockCommParams>
    CATLASS_DEVICE
    AllToAllVExportPipeline(Resource &resource, BlockCommParams const &commParams) :
        blockExport(resource, MakeParams(commParams))
    {
    }

    CATLASS_DEVICE
    void InitBlockLoop()
    {
        blockExport.InitBlockLoop();
    }

    CATLASS_DEVICE
    void FinalizeBlockLoop()
    {
        blockExport.FinalizeBlockLoop();
    }

    CATLASS_DEVICE
    BlockExport &GetBlockExport()
    {
        return blockExport;
    }

private:
    template <class BlockCommParams>
    CATLASS_DEVICE
    static typename BlockExport::Params MakeParams(BlockCommParams const &commParams)
    {
        typename BlockExport::TileRemoteCopy::Params exportTileParams{
            commParams.TileShape()};
        return typename BlockExport::Params{
            commParams.BlockShape(), exportTileParams};
    }

    BlockExport blockExport;
};

template <bool EXPORT_ALLTOALLV_, class ElementA_>
struct AllToAllVExportParams {
    CATLASS_HOST_DEVICE
    AllToAllVExportParams(GM_ADDR = nullptr) {}
};

template <class ElementA_>
struct AllToAllVExportParams<true, ElementA_> {
    __gm__ ElementA_ *ptrAllToAllVOutput;

    CATLASS_HOST_DEVICE
    AllToAllVExportParams(GM_ADDR ptrAllToAllVOutput_ = nullptr) :
        ptrAllToAllVOutput(reinterpret_cast<__gm__ ElementA_ *>(ptrAllToAllVOutput_))
    {
    }
};

template <
    class ProblemShape_,
    class BlockMmad_,
    class BlockComm_,
    class BlockMmadScheduler_,
    class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_,
    bool EXPORT_ALLTOALLV_ = false,
    class BlockExport_ = void
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
    using BlockExport = BlockExport_;
    using WorkspaceLayout =
        OpsTransformer::CatccosUdma::UdmaWorkspaceLayout;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr bool EXPORT_ALLTOALLV = EXPORT_ALLTOALLV_;
    static constexpr uint32_t READY_SLOT_RANKS =
        WorkspaceLayout::kMaxRankSize;
    static_assert(!EXPORT_ALLTOALLV || WORKSPACE_STAGES >= 2,
        "Pipelined AllToAllV export requires at least two staging buffers");
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    using ExportParams = AllToAllVExportParams<EXPORT_ALLTOALLV, ElementA>;

    struct Params : ExportParams {
        ProblemShape problemShape;
        uint32_t commInterval;
        uint32_t stageRows;
        __gm__ MoeRouteMetadata const *routeMetadata;
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
        GM_ADDR syncCommReady;
        uint32_t exportCoreCount;
        uint32_t selfCopyCores;
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
            GM_ADDR ptrSymmetric_, GM_ADDR syncMmadFinish_, GM_ADDR syncCommReady_,
            BlockCommParams const &allToAllVGmmParams_,
            GM_ADDR ptrAllToAllVOutput_ = nullptr,
            GM_ADDR ptrRouteMetadata_ = nullptr,
            uint32_t stageRows_ = 0,
            uint32_t exportCoreCount_ = 0,
            uint32_t selfCopyCores_ = 0
        ) : ExportParams(ptrAllToAllVOutput_), problemShape(problemShape_),
            commInterval(commInterval_), stageRows(stageRows_),
            routeMetadata(reinterpret_cast<__gm__ MoeRouteMetadata const *>(ptrRouteMetadata_)),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            layoutTagA(layoutTagA_), layoutTagB(layoutTagB_),
            ptrSymmetric(ptrSymmetric_), syncMmadFinish(syncMmadFinish_),
            syncCommReady(syncCommReady_),
            exportCoreCount(exportCoreCount_), selfCopyCores(selfCopyCores_),
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
        // Exact route-dependent GMM output capacity supplied by the PyTorch wrapper.
        uint32_t outputRows;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrLocalTokensPerExpert;
        GM_ADDR ptrGlobalTokensPerLocalExpert;
        GM_ADDR ptrSymmetric;
        Catlass::MatrixCoord commBlockShape;
        Catlass::MatrixCoord commTileShape;
        GM_ADDR ptrAllToAllVOutput{nullptr};
        GM_ADDR ptrRouteMetadata{nullptr};
        uint32_t stageRows{0};
        uint32_t exportCoreCount{0};
        uint32_t selfCopyCores{0};
    };

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t *workspace = nullptr) {
        LayoutTagA tagA{args.gemmShape.m(), args.gemmShape.k()};
        LayoutTagB tagB{args.gemmShape.k(), args.gemmShape.n()};
        LayoutTagC tagC{args.outputRows, args.gemmShape.n()};

        auto layoutA = tla::MakeLayoutFromTag(tagA);
        auto layoutB = tla::MakeLayoutFromTag(tagB);
        auto layoutC = tla::MakeLayoutFromTag(tagC);

        ProblemShape problemShape{
            args.gemmShape, args.rankSize, args.rankIdx, args.epSize, args.expertNum,
            args.ptrLocalTokensPerExpert, args.ptrGlobalTokensPerLocalExpert
        };

        typename BlockComm::TileRemoteCopy::Params tileParams{args.commTileShape};
        BlockCommParams blockCommParams{args.commBlockShape, tileParams};

        constexpr uint64_t IPC_BUFF_MAX_SIZE =
            200UL * 1024 * 1024 * sizeof(half);
        static_assert(
            IPC_BUFF_MAX_SIZE <= WorkspaceLayout::kDataBytes,
            "AllToAllV grouped-matmul staging exceeds the data plane");
        static_assert(
            WORKSPACE_STAGES <= WorkspaceLayout::kA2avGmmReadyStages,
            "AllToAllV grouped-matmul ready slots are undersized");

        auto gmSymmetric = args.ptrSymmetric;
        auto syncMmadFinish =
            args.ptrSymmetric + WorkspaceLayout::kA2avGmmSyncMmadOffset;
        auto syncCommReady =
            args.ptrSymmetric + WorkspaceLayout::kA2avGmmReadyOffset;

        return Params(
            problemShape,
            args.commInterval,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrC, layoutC,
            tagA, tagB,
            gmSymmetric, syncMmadFinish, syncCommReady,
            blockCommParams,
            args.ptrAllToAllVOutput,
            args.ptrRouteMetadata,
            args.stageRows,
            args.exportCoreCount,
            args.selfCopyCores
        );
    }

    CATLASS_DEVICE
    Ascend950AllToAllVGroupedMatmul()
    {
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
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
        BlockMmad mmad{resource};

        GemmCoord blockShape = GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        uint32_t commShapeM = params.stageRows == 0 ?
            params.commInterval * L1_TILE_M : params.stageRows;
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


            for (uint32_t localExpertIdx = 0; localExpertIdx < params.problemShape.localExpertNum(); ++localExpertIdx) {
                auto const &layoutB = params.layoutTagB;
                size_t localExpertOffsetB = localExpertIdx * layout::Capacity(layoutB);
                auto tensorB = tla::MakeTensor(gmB[localExpertOffsetB], params.layoutB, Catlass::Arch::PositionGM{});

                // Aggregate every source-rank fragment of one expert into one
                // AIC task domain.  Small per-peer GEMMs can then occupy all
                // cores while every task still reuses the same expert weight.
                scheduler.UpdateMmadContext(localExpertIdx);
                auto remapperA = scheduler.GetRemapperA(commIdx, localExpertIdx);
                auto remapperC = scheduler.GetRemapperC(commIdx, localExpertIdx);

                for (auto iter = scheduler.Begin(); !iter.End(); iter.Next()) {
                    auto blockOffset = scheduler.GetBlockOffset(iter);

                    MatrixCoord commOffsetA{
                        layoutSymmetricRow(Catlass::MakeCoord<int>(
                            stageIdx, blockOffset.rank(), 0)),
                        0};
                    auto blockOffsetA = remapperA(blockOffset);
                    MatrixCoord blockOffsetB = blockOffset.GetCoordKN();
                    auto blockOffsetC = remapperC(blockOffset);
                    auto actualBlockShape =
                        scheduler.RemapActualBlockShape(blockOffset, remapperA, remapperC);

                    auto tensorBlockA = GetTile(
                        tensorSymmetric,
                        tla::MakeCoord(
                            blockOffsetA.row() + commOffsetA.row(),
                            blockOffsetA.column()),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockB = GetTile(
                        tensorB,
                        tla::MakeCoord(blockOffsetB.row(), blockOffsetB.column()),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockC = GetTile(
                        tensorC,
                        tla::MakeCoord(blockOffsetC.row(), blockOffsetC.column()),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                    mmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                }
            }

            // Close every stage-ready handshake, including the final rolling
            // stages.  Earlier versions omitted the final WORKSPACE_STAGES
            // acknowledgements because those buffers are not reused inside
            // this invocation.  That also left AIV without proof that the
            // final AIC consumers had retired before the mixed kernel could
            // close and a following launch could reuse the rolling stages.
            // AIV drains these final acknowledgements once at kernel exit,
            // outside the steady-state communication/GMM pipeline.
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                flagAicFinishStore[stageIdx]);
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        BlockComm blockRemoteCopy(resource, params.blockCommParams);
        AllToAllVExportPipeline<EXPORT_ALLTOALLV, BlockExport> exportPipeline{
            resource, params.blockCommParams};

        uint32_t commShapeM = params.stageRows == 0 ?
            params.commInterval * L1_TILE_M : params.stageRows;
        MatrixCoord commShape{commShapeM, params.problemShape.k()};
        MatrixCoord blockShape = params.blockCommParams.BlockShape();
        BlockCommScheduler scheduler{params.problemShape, commShapeM, blockShape};

        auto const &layoutA = params.layoutTagA;
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        auto rankSize = params.problemShape.rankSize();
        auto rankIdx = params.problemShape.rankIdx();
        auto layoutTagSymmetric = layout::DistRowMajor::MakeAlignedLayout<ElementA>(commShape, rankSize);
        AscendC::GlobalTensor<ElementA> gmSymmetricList[WORKSPACE_STAGES];
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric);
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(
                ptrSymmetric + stageIdx * layout::Capacity(layoutTagSymmetric));
        }

        auto syncMmadFinish = reinterpret_cast<__gm__ int32_t *>(params.syncMmadFinish);
        auto syncCommReady = reinterpret_cast<__gm__ uint64_t *>(params.syncCommReady);
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreCount = AscendC::GetBlockNum();
        uint32_t exportCoreCount = params.exportCoreCount == 0 ?
            coreCount : Min(params.exportCoreCount, coreCount);
        uint32_t availableSelfCores = coreCount - rankSize + 1;
        uint32_t selfCopyCores = params.selfCopyCores == 0 ? 1U :
            Min(params.selfCopyCores, availableSelfCores);
        bool isExportWorker = EXPORT_ALLTOALLV && subcoreIdx == 0 &&
            aicoreIdx < exportCoreCount;
        bool isRemoteSender = rankSize > 1 && subcoreIdx == 1 &&
            aicoreIdx == (rankIdx + 1) % rankSize;
        bool isExtraSelfCopyCore = aicoreIdx >= rankSize &&
            aicoreIdx < rankSize + selfCopyCores - 1;
        uint32_t selfWorkerIdx = isExtraSelfCopyCore ? aicoreIdx - rankSize + 1 : 0;
        bool isSelfCopyWorker = subcoreIdx == (EXPORT_ALLTOALLV ? 1U : 0U) &&
            (aicoreIdx == rankIdx || isExtraSelfCopyCore);
        bool isRootCore = (AscendC::GetBlockIdx() == 0);

        // Close the previous invocation before clearing reusable generations.
        // Each sender quiets its requests before reaching the next entry barrier.
        aclshmemx_barrier_all_vec();
        if (isRootCore) {
            aclshmemx_signal_op(syncMmadFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
            auto readyWords = reinterpret_cast<__gm__ int32_t *>(syncCommReady);
            constexpr uint32_t READY_WORDS = WORKSPACE_STAGES * READY_SLOT_RANKS *
                sizeof(uint64_t) / sizeof(int32_t);
            for (uint32_t wordIdx = 0; wordIdx < READY_WORDS; ++wordIdx) {
                aclshmemx_signal_op(readyWords + wordIdx, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
            }
        }
        aclshmemx_barrier_all_vec();

        if constexpr (EXPORT_ALLTOALLV) {
            static_assert(!std::is_void_v<BlockExport>,
                "The AllToAllV export kernel requires a local GM copy block");
            if (isExportWorker) {
                exportPipeline.InitBlockLoop();
            }
        }
        auto commLoops = scheduler.GetCommLoops();
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const &gmSymmetric = gmSymmetricList[stageIdx];
            auto remapperSrc = scheduler.GetRemapperSrc(commIdx);

            // AIC must retire the old stage before any PE can overwrite it.
            // Export of that stage already joined an earlier loop-end barrier.
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);
                Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            }
            if (isRootCore) {
                aclshmemx_signal_op(syncMmadFinish, commIdx + 1, ACLSHMEM_SIGNAL_SET, rankIdx);
            }

            if constexpr (EXPORT_ALLTOALLV) {
                // Export keeps the original subcore-0 worker pool and overlaps
                // communication into the current stage with reads of the previous one.
                if (commIdx > 0 && isExportWorker) {
                    uint32_t exportIdx = commIdx - 1;
                    ExportCurrentStage(params, exportIdx, commShapeM,
                        gmSymmetricList[exportIdx % WORKSPACE_STAGES], layoutTagSymmetric,
                        exportPipeline.GetBlockExport());
                }
            }

            if (isRemoteSender) {
                // One AIV owns every remote queue. Complete each entire peer
                // slice before visiting the next rank in cyclic order.
                for (uint32_t step = 1; step < rankSize; ++step) {
                    uint32_t dstRankIdx = (rankIdx + step) % rankSize;
                    auto commOffset = scheduler.GetBlockOffset(0, dstRankIdx);
                    auto actualCommShape = remapperSrc.GetResidueShape(commOffset);
                    if (Numel(actualCommShape) == 0) {
                        continue;
                    }
                    auto remoteSyncMmadFinish = static_cast<__gm__ int32_t *>(
                        shmem_ptr(syncMmadFinish, dstRankIdx));
                    aclshmem_signal_wait_until(
                        remoteSyncMmadFinish, ACLSHMEM_CMP_EQ, commIdx + 1);

                    auto commOffsetSrc = remapperSrc(commOffset);
                    auto commOffsetDst = commOffset.GetLocalCoord();
                    auto gmCommSrc = gmA[layoutA.GetOffset(commOffsetSrc)];
                    auto gmCommDst = gmSymmetric[layoutTagSymmetric.GetOffset(commOffsetDst)];
                    auto layoutCommSrc = layoutA.GetTileLayout(actualCommShape);
                    auto layoutCommDst = layoutTagSymmetric.GetTileLayout(actualCommShape);
                    // The slot identifies the source rank, independently of
                    // which AIV submits the WRITE_WITH_NOTIFY request.
                    auto readyAddress = syncCommReady + stageIdx * READY_SLOT_RANKS + rankIdx;
                    blockRemoteCopy.PutRemoteContiguousWithSignalAndQuiet(
                        gmCommSrc, layoutCommSrc, gmCommDst, layoutCommDst,
                        actualCommShape, readyAddress,
                        static_cast<uint64_t>(commIdx + 1), dstRankIdx);
                }
            }

            if (isSelfCopyWorker) {
                // Preserve the existing MTE worker pool and block distribution.
                // These AIVs never submit remote UDMA requests.
                blockRemoteCopy.InitBlockLoop();
                aclshmem_signal_wait_until(syncMmadFinish, ACLSHMEM_CMP_EQ, commIdx + 1);
                for (uint32_t blockIdx = selfWorkerIdx;
                     blockIdx < scheduler.GetBlockCount(); blockIdx += selfCopyCores) {
                    auto blockOffset = scheduler.GetBlockOffset(blockIdx, rankIdx);
                    auto actualBlockShape = scheduler.RemapActualBlockShape(blockOffset, remapperSrc);
                    if (Numel(actualBlockShape) == 0) {
                        continue;
                    }
                    auto blockOffsetSrc = remapperSrc(blockOffset);
                    auto blockOffsetDst = blockOffset.GetLocalCoord();
                    auto gmBlockSrc = gmA[layoutA.GetOffset(blockOffsetSrc)];
                    auto gmBlockDst = gmSymmetric[layoutTagSymmetric.GetOffset(blockOffsetDst)];
                    auto layoutBlockSrc = layoutA.GetTileLayout(actualBlockShape);
                    auto layoutBlockDst = layoutTagSymmetric.GetTileLayout(actualBlockShape);
                    blockRemoteCopy(gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst,
                        actualBlockShape, rankIdx);
                }
                blockRemoteCopy.FinalizeBlockLoop();
            }

            // Outbound quiet does not prove inbound completion. Root waits for
            // exactly the remote sources that have payload in this round.
            if (isRootCore) {
                for (uint32_t srcRankIdx = 0; srcRankIdx < rankSize; ++srcRankIdx) {
                    uint32_t receiveRows = scheduler.GetActualReceiveRows(commIdx, srcRankIdx);
                    if (srcRankIdx == rankIdx || receiveRows == 0) {
                        continue;
                    }
                    WaitForStrongReady(
                        syncCommReady + stageIdx * READY_SLOT_RANKS + srcRankIdx,
                        static_cast<uint64_t>(commIdx + 1), rankIdx, srcRankIdx,
                        commIdx, stageIdx, receiveRows);
                }
            }
            if constexpr (EXPORT_ALLTOALLV) {
                if (isExportWorker && commIdx > 0) {
                    exportPipeline.FinalizeBlockLoop();
                    exportPipeline.InitBlockLoop();
                }
            }
            // Every AIV participates, including idle workers. Join self MTE,
            // all outbound quiets, inbound readiness and previous-stage export.
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComm[stageIdx]);
        }
        if constexpr (EXPORT_ALLTOALLV) {
            if (isExportWorker) {
                if (commLoops > 0) {
                    uint32_t exportIdx = commLoops - 1;
                    ExportCurrentStage(params, exportIdx, commShapeM,
                        gmSymmetricList[exportIdx % WORKSPACE_STAGES], layoutTagSymmetric,
                        exportPipeline.GetBlockExport());
                }
                exportPipeline.FinalizeBlockLoop();
            }
        }

        // Close the final AIC consumers before a following invocation can reuse GM.
        uint32_t finalStageCount = Min(commLoops, WORKSPACE_STAGES);
        uint32_t firstFinalComm = commLoops - finalStageCount;
        for (uint32_t commIdx = firstFinalComm; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);
        }
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
    }

private:
    CATLASS_DEVICE
    static void WaitForStrongReady(
        __gm__ volatile uint64_t *readyAddress,
        uint64_t expectedGeneration,
        uint32_t rankIdx,
        uint32_t srcRankIdx,
        uint32_t commIdx,
        uint32_t stageIdx,
        uint32_t receiveRows)
    {
#if CATCCOS_UDMA_MSSANITIZER
        // GetSystemCycle uses a fixed 50 MHz clock: 420 s (7 min) * 50,000,000.
        constexpr int64_t READY_WAIT_TIMEOUT_CYCLES = 21000000000LL;
        // Reading the clock less often keeps the diagnostic close to the
        // original SHMEM wait loop, whose only operations are DCCI + GM load.
        constexpr uint32_t CLOCK_CHECK_POLL_MASK = 0xFFFU;
        int64_t startCycle = AscendC::GetSystemCycle();
        uint32_t pollCount = 0;

        while (true) {
            dcci_cacheline((__gm__ uint8_t *)readyAddress);
            uint64_t observedGeneration = *readyAddress;
            if (observedGeneration == expectedGeneration) {
                return;
            }
            if ((++pollCount & CLOCK_CHECK_POLL_MASK) != 0) {
                continue;
            }

            int64_t elapsedCycles =
                AscendC::GetSystemCycle() - startCycle;
            if (elapsedCycles < READY_WAIT_TIMEOUT_CYCLES) {
                continue;
            }

            uint64_t readyAddressValue =
                reinterpret_cast<uint64_t>(readyAddress);
            ascendc_assert(
                observedGeneration == expectedGeneration,
                "[CATCCOS_UDMA_A2AV_GMM_READY_TIMEOUT] "
                "rank=%u source=%u comm=%u stage=%u rows=%u "
                "observed=0x%x expected=0x%x "
                "address=0x%x elapsed_ms=%u\n",
                rankIdx, srcRankIdx, commIdx, stageIdx, receiveRows,
                observedGeneration, expectedGeneration, readyAddressValue,
                static_cast<uint32_t>(elapsedCycles / 50000LL));
            // A failed device assertion emits one diagnostic record and
            // terminates this root AIV. Keep the return for compiler flow
            // analysis even though the failed assertion does not return.
            return;
        }
#else
        aclshmemi_signal_wait_until_eq<uint64_t>(
            readyAddress, expectedGeneration);
#endif
    }

    template <class ExportCopy>
    CATLASS_DEVICE
    void ExportCurrentStage(
        Params const &params,
        uint32_t commIdx,
        uint32_t commShapeM,
        AscendC::GlobalTensor<ElementA> const &gmSymmetric,
        layout::DistRowMajor const &layoutTagSymmetric,
        ExportCopy &blockExport)
    {
        static_assert(EXPORT_ALLTOALLV, "AllToAllV export is only available in the export kernel variant");

        AscendC::GlobalTensor<ElementA> gmOutput;
        gmOutput.SetGlobalBuffer(params.ptrAllToAllVOutput);
        Catlass::layout::RowMajor layoutOutput{
            tla::get<0>(params.layoutC.shape()),
            params.problemShape.k()};

        auto blockShape = params.blockCommParams.BlockShape();
        uint32_t aivSubcoreCount = AscendC::GetSubBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx() / aivSubcoreCount;
        // GetBlockNum() is already the number of physical AI cores for this
        // mixed AIC/AIV launch; only GetBlockIdx() contains the subcore index.
        uint32_t coreCount = AscendC::GetBlockNum();
        uint32_t exportCoreCount = params.exportCoreCount == 0 ?
            coreCount : Min(params.exportCoreCount, coreCount);
        if (AscendC::GetSubBlockIdx() != 0 || coreCount == 0 ||
            coreIdx >= exportCoreCount) {
            return;
        }

        uint64_t sourceOffsets[
            BlockMmadScheduler::MoeConstraints::RANK_SIZE_LIMIT]{0};
        uint64_t outputOffset = 0;
        uint64_t windowStart = static_cast<uint64_t>(commIdx) * commShapeM;
        uint64_t windowEnd = windowStart + commShapeM;
        uint32_t taskIdx = 0;
        for (uint32_t localExpertIdx = 0;
             localExpertIdx < params.problemShape.localExpertNum(); ++localExpertIdx) {
            for (uint32_t srcRankIdx = 0;
                 srcRankIdx < params.problemShape.rankSize(); ++srcRankIdx) {
                uint64_t sourceOffset = sourceOffsets[srcRankIdx];
                uint64_t tokenCount = params.problemShape.globalTokensPerLocalExpert(
                    srcRankIdx, localExpertIdx);
                uint64_t segmentOutputOffset = outputOffset;
                if (params.routeMetadata != nullptr) {
                    uint32_t segmentIdx =
                        localExpertIdx * params.problemShape.rankSize() + srcRankIdx;
                    sourceOffset = params.routeMetadata->segments[segmentIdx].sourceRowOffset;
                    tokenCount = params.routeMetadata->segments[segmentIdx].rows;
                    segmentOutputOffset =
                        params.routeMetadata->segments[segmentIdx].outputRowOffset;
                }
                uint64_t copyBegin = Max(sourceOffset, windowStart);
                uint64_t copyEnd = Min(sourceOffset + tokenCount, windowEnd);
                copyEnd = Max(copyBegin, copyEnd);
                uint32_t rows = static_cast<uint32_t>(copyEnd - copyBegin);
                uint64_t stageTokenOffset = copyBegin - windowStart;
                uint64_t outputTokenOffset =
                    segmentOutputOffset + copyBegin - sourceOffset;
                MatrixCoord segmentShape{rows, params.problemShape.k()};
                MatrixCoord blockGrid = CeilDiv(segmentShape, blockShape);
                uint32_t segmentTasks = Numel(blockGrid);
                for (uint32_t segmentTask = 0; segmentTask < segmentTasks;
                     ++segmentTask, ++taskIdx) {
                    if (taskIdx % exportCoreCount != coreIdx) {
                        continue;
                    }
                    MatrixCoord blockCoord{
                        segmentTask / blockGrid.column(),
                        segmentTask % blockGrid.column()};
                    MatrixCoord blockOffset = blockCoord * blockShape;
                    MatrixCoord actualBlockShape = Min(
                        blockShape, ClipSub(segmentShape, blockOffset));

                    DistMatrixCoord srcOffset{
                        static_cast<uint32_t>(stageTokenOffset) + blockOffset.row(),
                        blockOffset.column(), srcRankIdx};
                    MatrixCoord dstOffset{
                        static_cast<uint32_t>(outputTokenOffset) + blockOffset.row(),
                        blockOffset.column()};

                    auto gmSrc = gmSymmetric[layoutTagSymmetric.GetOffset(srcOffset)];
                    auto gmDst = gmOutput[layoutOutput.GetOffset(dstOffset)];
                    auto layoutSrc = layoutTagSymmetric.GetTileLayout(actualBlockShape);
                    auto layoutDst = layoutOutput.GetTileLayout(actualBlockShape);
                    blockExport(
                        gmSrc, layoutSrc, gmDst, layoutDst,
                        actualBlockShape);
                }
                if (params.routeMetadata == nullptr) {
                    sourceOffsets[srcRankIdx] += tokenCount;
                    outputOffset += tokenCount;
                }
            }
        }
    }

    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishComm[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

}  // namespace Catccos::DGemm::Kernel

#endif
