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
    bool USE_UDMA_PEER_SERIAL_SCHEDULE_ = false,
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
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr bool USE_UDMA_PEER_SERIAL_SCHEDULE = USE_UDMA_PEER_SERIAL_SCHEDULE_;
    static constexpr bool EXPORT_ALLTOALLV = EXPORT_ALLTOALLV_;
    static_assert(!EXPORT_ALLTOALLV || WORKSPACE_STAGES >= 2,
        "Pipelined AllToAllV export requires at least two staging buffers");
    static_assert(!EXPORT_ALLTOALLV || USE_UDMA_PEER_SERIAL_SCHEDULE,
        "Pipelined AllToAllV export requires the two-subcore UDMA schedule");
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    using ExportParams = AllToAllVExportParams<EXPORT_ALLTOALLV, ElementA>;

    struct Params : ExportParams {
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
            BlockCommParams const &allToAllVGmmParams_,
            GM_ADDR ptrAllToAllVOutput_ = nullptr
        ) : ExportParams(ptrAllToAllVOutput_), problemShape(problemShape_),
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
        GM_ADDR ptrAllToAllVOutput{nullptr};
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
            blockCommParams,
            args.ptrAllToAllVOutput
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
        AllToAllVExportPipeline<EXPORT_ALLTOALLV, BlockExport> exportPipeline{
            resource, params.blockCommParams};

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

        // 闁瀚ㄩ崬顖欑閻?aicore閿涘苯顕ぐ鎾冲 rank 閻ㄥ嫪淇婇崣宄版勾閸р偓鏉╂稖顢戦崚婵嗩潗閸?
        bool isRootCore = (AscendC::GetBlockIdx() == 0);
        auto rankIdx = params.problemShape.rankIdx();
        if (isRootCore) {
            aclshmemx_signal_op(syncMmadFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
            aclshmemx_signal_op(syncCommFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
        }
        aclshmemx_barrier_all_vec();

        uint32_t receiveAccum = 0;
        auto commLoops = scheduler.GetCommLoops();
        if constexpr (EXPORT_ALLTOALLV) {
            static_assert(!std::is_void_v<BlockExport>,
                "The AllToAllV export kernel requires a local GM copy block");
        }
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
        if constexpr (EXPORT_ALLTOALLV) {
            // Keep communication and export on different AIV subcores. UB and
            // MTE event IDs are private to an AIV subcore, so the two blocks
            // can now make progress independently without aliasing resources.
            if (subcoreIdx == 0) {
                exportPipeline.InitBlockLoop();
            } else {
                blockRemoteCopy.InitBlockLoop();
            }
        } else {
            blockRemoteCopy.InitBlockLoop();
        }
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const &gmSymmetric = gmSymmetricList[stageIdx];

            // 妫板嫯顓搁張顒冪枂鏉╁洤鎮楅敍灞界秼閸?rank 鐏忓棛鐤拋鈩冩暪閸掓澘顦跨亸?token
            receiveAccum += scheduler.GetActualReceiveAccum(commIdx);
            auto remapperSrc = scheduler.GetRemapperSrc(commIdx);
            // 缁涘绶熸稉濠佺娑擃亣顓哥粻妤勭枂濞嗏剝澧嶉張?aicore 閻ㄥ嫯顓哥粻妤€鐣幋?
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);
                Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            }
#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIV);
#endif
            // 闁瀚ㄩ崬顖欑閻?aicore閿涘苯鐨㈣ぐ鎾冲 rank 閻?syncMmadFinish 鐠佸墽鐤嗘稉?commIdx + 1閿涘矁銆冪粈杞扮瑐娑撯偓娑擃亪鈧矮淇婃潪顔筋偧鐠侊紕鐣绘禒璇插瀹告彃鐣幋?
            if (isRootCore) {
                aclshmemx_signal_op(syncMmadFinish, commIdx + 1, ACLSHMEM_SIGNAL_SET, rankIdx);
            }

            if constexpr (EXPORT_ALLTOALLV) {
                // Pipeline the previous stage's ordinary-GM export with the
                // current stage's communication. Both consumers only read the
                // staging buffers, and the per-loop barrier below guarantees
                // that export(i - 1) finishes before its stage can be reused.
                if (commIdx > 0 && subcoreIdx == 0) {
                    uint32_t exportIdx = commIdx - 1;
                    uint32_t exportStageIdx = exportIdx % WORKSPACE_STAGES;
                    ExportCurrentStage(params, exportIdx, commShapeM,
                        gmSymmetricList[exportStageIdx], layoutTagSymmetric,
                        exportPipeline.GetBlockExport());
                }
            }

            if constexpr (USE_UDMA_PEER_SERIAL_SCHEDULE) {
                // A UDMA queue is owned by one AIV subcore for each destination PE.
                // Sharing a PE and sync-id between many AIV cores can corrupt or
                // drop requests. The regular variant keeps local MTE on subcore 0
                // and remote UDMA puts on subcore 1; the export variant reserves
                // subcore 0 for the preceding stage's export.
                uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
                if (aicoreIdx < rankSize) {
                    uint32_t dstRankIdx = aicoreIdx;
                    bool isLocalCopy = dstRankIdx == rankIdx;
                    bool ownsPeer;
                    if constexpr (EXPORT_ALLTOALLV) {
                        // Subcore 0 is dedicated to export. Subcore 1 owns both
                        // the self MTE copy and remote UDMA submission so the
                        // complete next communication round overlaps export.
                        ownsPeer = subcoreIdx == 1;
                    } else {
                        ownsPeer = (isLocalCopy && subcoreIdx == 0) ||
                                   (!isLocalCopy && subcoreIdx == 1);
                    }
                    if (ownsPeer) {
                        auto commOffset = scheduler.GetBlockOffset(0, dstRankIdx);
                        auto actualCommShape = remapperSrc.GetResidueShape(commOffset);
                        if (Numel(actualCommShape) != 0) {
                            // Uneven AllToAllV routes can give ranks different communication-loop counts.
                            // Wait only when this loop has payload for the destination peer.
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

            // 缁涘绶熻ぐ鎾冲 commIdx 娑撳绱濋幍鈧張澶婂晸閸掓澘缍嬮崜?rank symmetric 娑撳﹦娈戦弫鐗堝祦闁棄鍟撶€?
            if (isRootCore) {
                aclshmem_signal_wait_until(syncCommFinish, ACLSHMEM_CMP_EQ, receiveAccum);
            }
            // 缁涘绶熻ぐ鎾冲闁矮淇婃潪顔筋偧娑撳鏆熼幑顔藉复閺€璺虹暚閹存劕鎮楅敍灞惧閺堝娈?aic 闁氨鐓?aiv 瀵偓婵顓哥粻妤佹拱鏉烆喚娈?            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            if constexpr (EXPORT_ALLTOALLV) {
                // Export reads stage(i - 1), so it does not depend on the
                // current receive-counter wait. Drain export(i - 1) and
                // communication(i) independently, then use one barrier to
                // publish both completions and protect staging reuse.
                if (subcoreIdx == 0) {
                    if (commIdx > 0) {
                        exportPipeline.FinalizeBlockLoop();
                        exportPipeline.InitBlockLoop();
                    }
                } else {
                    blockRemoteCopy.FinalizeBlockLoop();
                    if (commIdx + 1 < commLoops) {
                        blockRemoteCopy.InitBlockLoop();
                    }
                }
                Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            } else {
                Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            }
#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIV);
#endif
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComm[stageIdx]);

        }
        if constexpr (EXPORT_ALLTOALLV) {
            // Drain the final stage; there is no communication(i + 1) with
            // which to overlap it. AIC can still compute the same stage in
            // parallel because both operations are read-only on staging GM.
            if (subcoreIdx == 0) {
                if (commLoops > 0) {
                    uint32_t exportIdx = commLoops - 1;
                    uint32_t exportStageIdx = exportIdx % WORKSPACE_STAGES;
                    ExportCurrentStage(params, exportIdx, commShapeM,
                        gmSymmetricList[exportStageIdx], layoutTagSymmetric,
                        exportPipeline.GetBlockExport());
                }
                exportPipeline.FinalizeBlockLoop();
            }
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        } else {
            blockRemoteCopy.FinalizeBlockLoop();
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
        if (AscendC::GetSubBlockIdx() != 0 || coreCount == 0) {
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
                uint64_t copyBegin = Max(sourceOffset, windowStart);
                uint64_t copyEnd = Min(sourceOffset + tokenCount, windowEnd);
                copyEnd = Max(copyBegin, copyEnd);
                uint32_t rows = static_cast<uint32_t>(copyEnd - copyBegin);
                uint64_t stageTokenOffset = copyBegin - windowStart;
                uint64_t outputTokenOffset = outputOffset + copyBegin - sourceOffset;
                MatrixCoord segmentShape{rows, params.problemShape.k()};
                MatrixCoord blockGrid = CeilDiv(segmentShape, blockShape);
                uint32_t segmentTasks = Numel(blockGrid);
                for (uint32_t segmentTask = 0; segmentTask < segmentTasks;
                     ++segmentTask, ++taskIdx) {
                    if (taskIdx % coreCount != coreIdx) {
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
                sourceOffsets[srcRankIdx] += tokenCount;
                outputOffset += tokenCount;
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
