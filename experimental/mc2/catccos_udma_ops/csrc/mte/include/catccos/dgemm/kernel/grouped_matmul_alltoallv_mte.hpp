/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_MTE_HPP
#define CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_MTE_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/catccos.hpp"
#include "catccos/symm_coord.hpp"
#include "catccos/layout/dist_matrix.hpp"
#include "catccos/comm/mte_transport_params.hpp"
#include "catccos/comm/mte_peer_completion.hpp"
#include "catccos/comm/mte_barrier.hpp"

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
    uint64_t SYMMETRIC_DATA_BYTES_ = 1000UL * 1024 * 1024
>
class GroupedMatmulAllToAllVMte {
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

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr uint64_t SYMMETRIC_DATA_BYTES = SYMMETRIC_DATA_BYTES_;
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
        GM_ADDR ptrControl;
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
            BlockCommSchedulerParams const &blockCommSchedulerParams_,
            GM_ADDR ptrControl_
        ) : problemShape(problemShape_), commInterval(commInterval_),
            layoutTagComm(layoutTagComm_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            ptrSymmetric(ptrSymmetric_), syncMmadFinish(syncMmadFinish_), syncCommFinish(syncCommFinish_),
            ptrControl(ptrControl_),
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
        OpsTransformer::CatccosMte::MteCommConfig transport;
        GM_ADDR ptrControl;
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

        auto gmSymmetric = args.ptrSymmetric;
        auto syncMmadFinish = args.ptrControl;
        auto syncCommFinish = args.ptrControl + 64;

        typename BlockComm::TileRemoteCopy::Params tileParams{args.commTileShape};
        BlockCommParams blockCommParams{args.commBlockShape, tileParams, args.transport};
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
            blockCommSchedulerParams, args.ptrControl
        );
    }

    CATLASS_DEVICE
    GroupedMatmulAllToAllVMte()
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
        RunTargetOrderAic(params);
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        RunTargetOrderAiv(params);
    }

private:
    CATLASS_DEVICE
    void RunTargetOrderAic(Params const &params)
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
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            scheduler.UpdateCommContext(commIdx);
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishComm[stageIdx]);
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
                            blockOffsetC.row() + stageRankOffset.row(), blockOffsetC.column()),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                }
            }
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad.SynchronizeBlock();
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageIdx]);
        }
    }

    // The producer's full-N stage geometry is unchanged. Reconstruct the
    // consumer prefix for each assigned peer so a worker group can service
    // more than one peer without keeping a scheduler instance per peer.
    CATLASS_DEVICE
    void UpdatePeerContext(BlockCommScheduler &scheduler, uint32_t commIdx,
                           uint32_t srcPeer)
    {
        scheduler.srcEpIdx = srcPeer;
        scheduler.actualRows = 0;
        scheduler.consumedRows = 0;
        uint32_t skipGroups = commIdx * scheduler.groupsPerCommInRank;
        for (uint32_t expert = 0;
             expert < scheduler.problemShape.localExpertNum() && skipGroups != 0;
             ++expert) {
            uint32_t groups = Min(skipGroups,
                scheduler.inputGroupNumInLocalExpert[srcPeer][expert]);
            uint32_t tokens = scheduler.problemShape.localTokensPerExpert(srcPeer, expert);
            scheduler.consumedRows += Min(groups * scheduler.blockM, tokens);
            skipGroups -= groups;
        }
        scheduler.UpdateCommContext(commIdx);
    }

    CATLASS_DEVICE
    void CopyPeer(Params const &params, BlockCommScheduler &scheduler,
                  BlockComm &blockCopy, Catccos::Comm::MtePeerCompletion &completion,
                  AscendC::GlobalTensor<ElementC> const &gmStage,
                  LayoutTagComm const &layoutStage,
                  AscendC::GlobalTensor<ElementC> const &gmOutput,
                  uint32_t commIdx, uint32_t peer, uint32_t lane, uint32_t workers,
                  bool sequentialPeer)
    {
        if (sequentialPeer) {
            // Self and single-peer groups keep the baseline incremental prefix
            // update. Only a group that changes peers needs reconstruction.
            scheduler.srcEpIdx = peer;
            scheduler.UpdateCommContext(commIdx);
        } else {
            UpdatePeerContext(scheduler, commIdx, peer);
        }
        if (!scheduler.HasInput()) {
            return;
        }
        auto remoteReady = static_cast<__gm__ int32_t *>(
            aclshmem_ptr(params.syncMmadFinish, peer));
        aclshmem_signal_wait_until(remoteReady, ACLSHMEM_CMP_EQ, commIdx + 1);
        MatrixCoord shape{scheduler.GetActualRows(), params.problemShape.n()};
        auto layoutSrc = layoutStage.GetTileLayout(shape);
        auto dst = gmOutput[params.layoutTagComm.GetOffset(
            MatrixCoord{scheduler.GetOutputRowOffset(), 0})];
        auto layoutDst = params.layoutTagComm.GetTileLayout(shape);
        blockCopy.SetWorker(lane, workers);
        blockCopy.InitBlockLoop();
        blockCopy(gmStage, layoutSrc, dst, layoutDst, shape, peer);
        blockCopy.FinalizeBlockLoop();

        // A lane with no chunks must also participate. Acknowledging the peer
        // before the last lane drains would let its producer overwrite stage.
        completion.MarkDone(peer, lane, commIdx + 1);
        if (lane == 0) {
            completion.Wait(peer, workers, commIdx + 1);
            aclshmem_fence();
            aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(params.syncCommFinish),
                               1, ACLSHMEM_SIGNAL_ADD, peer);
        }
    }

    CATLASS_DEVICE
    void RunTargetOrderAiv(Params const &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
        uint32_t rankSize = params.problemShape.rankSize();
        uint32_t rankIdx = params.problemShape.rankIdx();
        const auto &transport = params.blockCommParams.transport;
        bool isSelfWorker = subcoreIdx == 0 && aicoreIdx == rankIdx;
        uint32_t remoteWorker = aicoreIdx < rankIdx ? aicoreIdx : aicoreIdx - 1;
        bool isRemoteWorker = subcoreIdx == 0 && aicoreIdx != rankIdx &&
                              remoteWorker < transport.commAivCount;

        uint32_t blockPerComm = AscendC::GetBlockNum() * params.commInterval;
        uint32_t tilePerCommInRank = blockPerComm / rankSize;
        uint32_t nLoops = CeilDiv(params.problemShape.n(), L1_TILE_N);
        uint32_t groupsPerCommInRank = Max<uint32_t>(1U, tilePerCommInRank / nLoops);
        uint32_t stageRows = groupsPerCommInRank * L1_TILE_M;
        GemmCoord shapeMmad{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        BlockCommScheduler scheduler{
            params.problemShape, groupsPerCommInRank, rankIdx,
            shapeMmad.GetCoordMN(), params.blockCommParams.BlockShape(),
            params.blockCommSchedulerParams.CoreSplit()};
        BlockComm blockCopy(resource, params.blockCommParams);
        Catccos::Comm::MtePeerCompletion completion{params.ptrControl};

        auto layoutStage = Catlass::layout::RowMajor(stageRows, params.problemShape.n());
        auto ptrStage = reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric);
        AscendC::GlobalTensor<ElementC> gmStages[WORKSPACE_STAGES];
        for (uint32_t stage = 0; stage < WORKSPACE_STAGES; ++stage) {
            gmStages[stage].SetGlobalBuffer(ptrStage +
                (stage * rankSize + rankIdx) * layout::Capacity(layoutStage));
        }
        AscendC::GlobalTensor<ElementC> gmOutput;
        gmOutput.SetGlobalBuffer(params.ptrC);
        auto syncReady = reinterpret_cast<__gm__ int32_t *>(params.syncMmadFinish);
        auto syncDone = reinterpret_cast<__gm__ int32_t *>(params.syncCommFinish);

        // All ranks leave the preceding collective before any rank resets the
        // reused counters. Both barriers are required across repeated calls.
        Catccos::Comm::MteAllRankBarrier(params.ptrControl, rankSize);
        if (isSelfWorker) {
            aclshmemx_signal_op(syncReady, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
            aclshmemx_signal_op(syncDone, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
            completion.Reset(rankSize, transport.workersPerPeer);
        }
        Catccos::Comm::MteAllRankBarrier(params.ptrControl, rankSize);

        uint32_t receiveAccum = 0;
        uint32_t commLoops = scheduler.GetCommLoops();
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stage = commIdx % WORKSPACE_STAGES;
            receiveAccum += scheduler.GetActualReceiveAccum(commIdx);
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stage]);
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            if (isSelfWorker) {
                aclshmemx_signal_op(syncReady, commIdx + 1, ACLSHMEM_SIGNAL_SET, rankIdx);
                CopyPeer(params, scheduler, blockCopy, completion, gmStages[stage],
                         layoutStage, gmOutput, commIdx, rankIdx, 0, 1, true);
            } else if (isRemoteWorker) {
                uint32_t lane = remoteWorker % transport.workersPerPeer;
                uint32_t group = remoteWorker / transport.workersPerPeer;
                uint32_t groupCount = transport.commAivCount / transport.workersPerPeer;
                bool sequentialPeer = group + groupCount >= rankSize - 1;
                for (uint32_t ordinal = group; ordinal + 1 < rankSize; ordinal += groupCount) {
                    uint32_t peer = ordinal < rankIdx ? ordinal : ordinal + 1;
                    CopyPeer(params, scheduler, blockCopy, completion, gmStages[stage],
                             layoutStage, gmOutput, commIdx, peer, lane,
                             transport.workersPerPeer, sequentialPeer);
                }
            }
            if (isSelfWorker) {
                aclshmem_signal_wait_until(syncDone, ACLSHMEM_CMP_EQ, receiveAccum);
            }
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            if (commIdx + WORKSPACE_STAGES < commLoops) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComm[stage]);
            }
        }
        Catccos::Comm::MteAllRankBarrier(params.ptrControl, rankSize);
    }

    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishComm[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

}  // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_GROUPED_MATMUL_ALLTOALLV_MTE_HPP
