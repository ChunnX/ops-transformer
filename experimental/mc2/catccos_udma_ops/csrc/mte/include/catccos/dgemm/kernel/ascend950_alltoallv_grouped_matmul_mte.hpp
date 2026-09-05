/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALLV_GROUPED_MATMUL_MTE_HPP
#define CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALLV_GROUPED_MATMUL_MTE_HPP

#include <type_traits>

#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/catccos.hpp"
#include "catccos/comm/mte_barrier.hpp"
#include "catccos/comm/mte_peer_completion.hpp"
#include "catccos/dgemm/moe_route_metadata.hpp"
#include "catccos/layout/dist_matrix.hpp"
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif

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
            commParams.BlockShape(), exportTileParams, commParams.transport};
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
class Ascend950AllToAllVGroupedMatmulMte {
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
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr bool EXPORT_ALLTOALLV = EXPORT_ALLTOALLV_;
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
        GM_ADDR ptrControl;
        GM_ADDR syncMmadFinish;
        GM_ADDR syncCommFinish;
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
            GM_ADDR ptrSymmetric_, GM_ADDR syncMmadFinish_, GM_ADDR syncCommFinish_,
            BlockCommParams const &allToAllVGmmParams_,
            GM_ADDR ptrAllToAllVOutput_ = nullptr,
            GM_ADDR ptrRouteMetadata_ = nullptr,
            uint32_t stageRows_ = 0,
            uint32_t exportCoreCount_ = 0,
            uint32_t selfCopyCores_ = 0,
            GM_ADDR ptrControl_ = nullptr
        ) : ExportParams(ptrAllToAllVOutput_), problemShape(problemShape_),
            commInterval(commInterval_), stageRows(stageRows_),
            routeMetadata(reinterpret_cast<__gm__ MoeRouteMetadata const *>(ptrRouteMetadata_)),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            layoutTagA(layoutTagA_), layoutTagB(layoutTagB_),
            ptrSymmetric(ptrSymmetric_), ptrControl(ptrControl_), syncMmadFinish(syncMmadFinish_), syncCommFinish(syncCommFinish_),
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
        OpsTransformer::CatccosMte::MteCommConfig transport{};
        GM_ADDR ptrControl{nullptr};
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
        BlockCommParams blockCommParams{args.commBlockShape, tileParams, args.transport};

        using namespace OpsTransformer::CatccosMte;
        auto gmSymmetric = args.ptrSymmetric;
        auto syncMmadFinish = args.ptrControl + kMteMmadSignalOffset;
        auto syncCommFinish = args.ptrControl + kMteCommSignalOffset;

        return Params(
            problemShape,
            args.commInterval,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrC, layoutC,
            tagA, tagB,
            gmSymmetric, syncMmadFinish, syncCommFinish,
            blockCommParams,
            args.ptrAllToAllVOutput,
            args.ptrRouteMetadata,
            args.stageRows,
            args.exportCoreCount,
            args.selfCopyCores,
            args.ptrControl
        );
    }

    CATLASS_DEVICE
    Ascend950AllToAllVGroupedMatmulMte()
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
    ~Ascend950AllToAllVGroupedMatmulMte()
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
        uint32_t commShapeM = params.stageRows == 0 ?
            params.commInterval * L1_TILE_M : params.stageRows;
        MatrixCoord commShape{commShapeM, params.problemShape.k()};
        BlockMmadScheduler scheduler{params.problemShape, commShapeM, blockShape.GetCoordMN()};

        auto rankSize = params.problemShape.rankSize();
        // AIC must read the same padded row stride used by the MTE writers.
        // The frozen K=2048 profiles keep their original contiguous layout.
        auto stageLayout = layout::DistRowMajor::MakeAlignedLayout<ElementA>(commShape, rankSize);
        auto layoutTagSymmetric = stageLayout.GetTileLayout(MatrixCoord{
            WORKSPACE_STAGES * rankSize * commShapeM, params.problemShape.k()});
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

#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIC);
#endif
            if (commLoops >= WORKSPACE_STAGES && commIdx < commLoops - WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageIdx]);
            }
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        BlockComm blockCopy(resource, params.blockCommParams);
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
        const uint32_t rankSize = params.problemShape.rankSize();
        const uint32_t rankIdx = params.problemShape.rankIdx();
        auto layoutSymmetric = layout::DistRowMajor::MakeAlignedLayout<ElementA>(commShape, rankSize);
        AscendC::GlobalTensor<ElementA> stages[WORKSPACE_STAGES];
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric);
        for (uint32_t stage = 0; stage < WORKSPACE_STAGES; ++stage) {
            stages[stage].SetGlobalBuffer(ptrSymmetric + stage * layout::Capacity(layoutSymmetric));
        }
        auto syncMmadFinish = reinterpret_cast<__gm__ int32_t *>(params.syncMmadFinish);
        auto syncCommFinish = reinterpret_cast<__gm__ int32_t *>(params.syncCommFinish);
        Catccos::Comm::MtePeerCompletion completion{params.ptrControl};
        const auto transport = params.blockCommParams.transport;
        const uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
        const uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const uint32_t coreCount = AscendC::GetBlockNum();
        const uint32_t selfCores = params.selfCopyCores == 0 ? 1 : params.selfCopyCores;
        const uint32_t exportCores = params.exportCoreCount == 0 ? coreCount : params.exportCoreCount;
        const bool isExport = EXPORT_ALLTOALLV && subcoreIdx == 0 && coreIdx < exportCores;
        const bool extraSelf = coreIdx >= rankSize && coreIdx < rankSize + selfCores - 1;
        const bool selfSlot = coreIdx == rankIdx || extraSelf;
        const uint32_t selfLane = extraSelf ? coreIdx - rankSize + 1 : 0;
        const bool isSelf = selfSlot && subcoreIdx == (EXPORT_ALLTOALLV ? 1 : 0);
        // Compact physical slots around the fixed self-copy cores. R-1
        // default workers retain the original per-peer core assignment.
        const uint32_t worker = coreIdx < rankIdx ? coreIdx :
            (coreIdx < rankSize ? coreIdx - 1 : coreIdx - selfCores);
        const bool isRemote = !selfSlot && subcoreIdx == 1 && worker < transport.commAivCount;
        const uint32_t groups = transport.commAivCount / transport.workersPerPeer;
        const bool isRoot = AscendC::GetBlockIdx() == 0;

        // Join the previous invocation before clearing remotely observed
        // state: uneven routes need not have the same loop count on all PEs.
        Catccos::Comm::MteAllRankBarrier(params.ptrControl, rankSize);
        if (isRoot) {
            aclshmemx_signal_op(syncMmadFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
            aclshmemx_signal_op(syncCommFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
            completion.Reset(rankSize, transport.workersPerPeer);
        }
        Catccos::Comm::MteAllRankBarrier(params.ptrControl, rankSize);
        uint32_t receiveAccum = 0;
        const uint32_t commLoops = scheduler.GetCommLoops();
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            const uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const &gmSymmetric = stages[stageIdx];
            receiveAccum += scheduler.GetActualReceiveAccum(commIdx);
            auto remapper = scheduler.GetRemapperSrc(commIdx);
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);
                Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            }
            if (isRoot) {
                aclshmemx_signal_op(syncMmadFinish, commIdx + 1, ACLSHMEM_SIGNAL_SET, rankIdx);
            }
            if constexpr (EXPORT_ALLTOALLV) {
                if (commIdx > 0 && isExport) {
                    exportPipeline.InitBlockLoop();
                    ExportCurrentStage(params, commIdx - 1, commShapeM,
                        stages[(commIdx - 1) % WORKSPACE_STAGES], layoutSymmetric,
                        exportPipeline.GetBlockExport());
                    exportPipeline.FinalizeBlockLoop();
                }
            }
            if (isSelf) {
                auto peerOffset = scheduler.GetBlockOffset(0, rankIdx);
                auto peerShape = remapper.GetResidueShape(peerOffset);
                if (Numel(peerShape) != 0) {
                    aclshmem_signal_wait_until(syncMmadFinish, ACLSHMEM_CMP_EQ, commIdx + 1);
                    blockCopy.SetWorker(0, 1);
                    blockCopy.InitBlockLoop();
                    for (uint32_t blockIdx = selfLane; blockIdx < scheduler.GetBlockCount();
                         blockIdx += selfCores) {
                        auto offset = scheduler.GetBlockOffset(blockIdx, rankIdx);
                        auto actual = scheduler.RemapActualBlockShape(offset, remapper);
                        if (Numel(actual) == 0) { continue; }
                        auto source = gmA[layoutA.GetOffset(remapper(offset))];
                        auto target = gmSymmetric[layoutSymmetric.GetOffset(offset.GetLocalCoord())];
                        blockCopy(source, layoutA.GetTileLayout(actual), target,
                                  layoutSymmetric.GetTileLayout(actual), actual, rankIdx);
                    }
                    blockCopy.FinalizeBlockLoop();
                    if (selfLane == 0) {
                        // The loop-end AIV barrier joins all self-copy workers
                        // before releasing AIC, including workers with no data.
                        aclshmemx_signal_op(syncCommFinish,
                            static_cast<int32_t>(Numel(CeilDiv(peerShape, blockShape))),
                            ACLSHMEM_SIGNAL_ADD, rankIdx);
                    }
                }
            }
            if (isRemote) {
                const uint32_t lane = worker % transport.workersPerPeer;
                blockCopy.SetWorker(lane, transport.workersPerPeer);
                for (uint32_t ordinal = worker / transport.workersPerPeer;
                     ordinal < rankSize - 1; ordinal += groups) {
                    const uint32_t peer = ordinal < rankIdx ? ordinal : ordinal + 1;
                    auto offset = scheduler.GetBlockOffset(0, peer);
                    auto actual = remapper.GetResidueShape(offset);
                    if (Numel(actual) == 0) { continue; }
                    auto remoteReady = static_cast<__gm__ int32_t *>(aclshmem_ptr(syncMmadFinish, peer));
                    aclshmem_signal_wait_until(remoteReady, ACLSHMEM_CMP_EQ, commIdx + 1);
                    auto source = gmA[layoutA.GetOffset(remapper(offset))];
                    auto target = gmSymmetric[layoutSymmetric.GetOffset(offset.GetLocalCoord())];
                    blockCopy.InitBlockLoop();
                    blockCopy(source, layoutA.GetTileLayout(actual), target,
                              layoutSymmetric.GetTileLayout(actual), actual, peer);
                    blockCopy.FinalizeBlockLoop();
                    // Even a lane with zero chunks must take part in the join.
                    completion.MarkDone(peer, lane, commIdx + 1);
                    if (lane == 0) {
                        completion.Wait(peer, transport.workersPerPeer, commIdx + 1);
                        aclshmemx_signal_op(syncCommFinish,
                            static_cast<int32_t>(Numel(CeilDiv(actual, blockShape))),
                            ACLSHMEM_SIGNAL_ADD, peer);
                    }
                }
            }
            if (isRoot) {
                aclshmem_signal_wait_until(syncCommFinish, ACLSHMEM_CMP_EQ, receiveAccum);
            }
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComm[stageIdx]);
        }
        if constexpr (EXPORT_ALLTOALLV) {
            if (isExport && commLoops > 0) {
                exportPipeline.InitBlockLoop();
                ExportCurrentStage(params, commLoops - 1, commShapeM,
                    stages[(commLoops - 1) % WORKSPACE_STAGES], layoutSymmetric,
                    exportPipeline.GetBlockExport());
                exportPipeline.FinalizeBlockLoop();
            }
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        }
    }

private:
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
#ifdef ENABLE_TIMER
    AscendTimerDevice timer;
#endif
};

}  // namespace Catccos::DGemm::Kernel

#endif
