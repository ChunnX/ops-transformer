/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_FP8_TLA_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_FP8_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Gemm::Kernel {

// Template for grouped matmul kernel. Compute grouped C = A * B
template <
    class BlockMmadTla_,
    class BlockEpilogue_,
    class BlockScheduler_,
    uint32_t WORKSPACE_STAGES_,
    class ElementGroupList_
>
class GroupedMatmulSliceMPerTokenPerChannelMultiStageTla {
public:
    using BlockMmadTla = BlockMmadTla_;
    using ArchTag = typename BlockMmadTla::ArchTag;
    using L1TileShape = typename BlockMmadTla::L1TileShape;
    using ElementA = typename BlockMmadTla::ElementA;
    using LayoutA = typename BlockMmadTla::LayoutA;
    using ElementB = typename BlockMmadTla::ElementB;
    using LayoutB = typename BlockMmadTla::LayoutB;
    using ElementC = typename BlockMmadTla::ElementC;
    using LayoutC = typename BlockMmadTla::LayoutC;
    using ElementAccumulator = typename BlockMmadTla::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using ElementScale = typename BlockEpilogue::ElementScale;
    using LayoutScale = typename BlockEpilogue::LayoutScale;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;    

    using ElementGroupList = ElementGroupList_;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList; // int64_t
        __gm__ ElementA *ptrA; // fp8_e5m2
        LayoutA layoutA; // {m, k}
        __gm__ ElementB *ptrB; // fp8_e5m2
        LayoutB layoutB; // {k, n}
        __gm__ ElementScale *ptrScale; // fp8_e5m2
        LayoutScale layoutScale; // {n}
        __gm__ ElementPerTokenScale *ptrPerTokenScale; // fp_e5m2
        LayoutPerTokenScale layoutPerTokenScale; // {m}
        __gm__ ElementD *ptrD; 
        LayoutD layoutD; // {m, n}
        layout::RowMajor layoutTagA;
        layout::RowMajor layoutTagC;
        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrScale_, LayoutScale layoutScale_,
            GM_ADDR ptrPerTokenScale_, LayoutPerTokenScale layoutPerTokenScale_,
            GM_ADDR ptrD_, LayoutD const &layoutD_, layout::RowMajor const &layoutTagA_, layout::RowMajor const &layoutTagC_,
            GM_ADDR ptrWorkspace_
        ) : problemShape(problemShape_),
            problemCount(problemCount_), ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrScale(reinterpret_cast<__gm__ ElementScale *>(ptrScale_)), layoutScale(layoutScale_),
            ptrPerTokenScale(reinterpret_cast<__gm__ ElementPerTokenScale *>(ptrPerTokenScale_)),
            layoutPerTokenScale(layoutPerTokenScale_),
            ptrD(reinterpret_cast<__gm__ ElementD *>(ptrD_)), layoutD(layoutD_), layoutTagA(layoutTagA_),layoutTagC(layoutTagC_),
            ptrWorkspace(ptrWorkspace_)
        {
        }
    };
    struct Arguments {
        GemmCoord problemShape;
        uint32_t problemCount;
        uint32_t aicCoreNum;
        uint8_t *ptrGroupList;
        uint8_t *ptrA; LayoutA layoutA;
        uint8_t *ptrB; LayoutB layoutB;
        uint8_t *ptrScale; LayoutScale layoutScale;
        uint8_t *ptrPerTokenScale; LayoutPerTokenScale layoutPerTokenScale;
        uint8_t *ptrD; LayoutD layoutD;
    };
    static bool CanImplement(const Arguments &args)
    {
        return true;
    }
    static size_t GetWorkspaceSize(const Arguments &args)
    {
        size_t lenWorkspace = static_cast<size_t>(L1_TILE_M) * L1_TILE_N *
            args.aicCoreNum * WORKSPACE_STAGES;
        size_t sizeWorkspace = lenWorkspace * sizeof(uint32_t);
        return sizeWorkspace;
    }
    static Params ToUnderlyingArguments(const Arguments &args, uint8_t* workspace)
    {
        uint32_t m = args.problemShape.m();
        uint32_t n = args.problemShape.n();
        uint32_t k = args.problemShape.k();
        layout::RowMajor layoutTagA{m, k};
        layout::RowMajor layoutTagC{m, n};

        Params params{args.problemShape, args.problemCount, args.ptrGroupList,
            args.ptrA, args.layoutA,
            args.ptrB, args.layoutB,
            args.ptrScale, args.layoutScale,
            args.ptrPerTokenScale, args.layoutPerTokenScale,
            args.ptrD, args.layoutD, layoutTagA, layoutTagC,
            workspace};
        return params;
    }
    // Methods
    CATLASS_DEVICE
    GroupedMatmulSliceMPerTokenPerChannelMultiStageTla() 
    {
        Arch::FlagID flagId = 0;
        for (uint32_t stageId = 0; stageId < WORKSPACE_STAGES; ++stageId) {
            flagAicFinishStoreList[stageId] = Arch::CrossCoreFlag(flagId++);
            flagAivFinishComputeList[stageId] = Arch::CrossCoreFlag(flagId++);\
            aicWaitFuncList[stageId] = {this, stageId};
            aicSetFuncList[stageId] = {this, stageId};
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        // cce::printf("[debug] 进了cube\n");
        AscendC::ICachePreLoad(1);
        BlockScheduler blockScheduler;
        BlockMmadTla blockMmadTla(resource); 

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        
        AscendC::GlobalTensor<ElementC> gmC; // 先输出到workspace上
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutTagC = layout::RowMajor{L1_TILE_M * coreNum * WORKSPACE_STAGES, L1_TILE_N};
        auto layoutC = tla::MakeLayoutFromTag(layoutTagC);       

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t stageId = 0;
        uint32_t stageUsed = 0;
        uint32_t startCoreIdx = 0;

        auto tensorC = tla::MakeTensor(gmC, layoutC, Arch::PositionGM{});
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = groupList.GetValue(groupIdx);
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            // 输出到workspace
            auto layoutA = params.layoutTagA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            auto layoutTileA = tla::MakeLayoutFromTag(layoutA);
            auto layoutB = params.layoutB;

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            // 每个专家的权重不一样，需要设置GlobalTensor
            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(params.ptrB + gmGroupOffsetB);
            
            auto strideA = layoutA.stride(0); // 获取每列的步长
            auto strideC = params.layoutTagC.stride(0);

            if (CeilDiv(currentM, L1_TILE_M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }


            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            
            auto tensorA = tla::MakeTensor(gmA, layoutTileA, Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(gmB, layoutB, Arch::PositionGM{});
            // Loop through the matmul of each groupIdx
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                Callback callbackBeforeFixpipe{};
                if (stageUsed == WORKSPACE_STAGES) {
                    callbackBeforeFixpipe = MakeCallback(&aicWaitFuncList[stageId]);
                } else {
                    ++stageUsed;
                }
                Callback callbackAfterFixpipe = MakeCallback(&aicSetFuncList[stageId]);

                MatrixCoord offsetA{blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);

                // Make tiled views
                auto tensorBlockA = GetTile(tensorA,
                                            tla::MakeCoord((gmGroupOffsetA + gmOffsetA) / strideA, (gmGroupOffsetA + gmOffsetA) % strideA),
                                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB = GetTile(tensorB,
                                            tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                                            tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockC = GetTile(tensorC,
                                            tla::MakeCoord((stageId * coreNum + coreIdx) * L1_TILE_M, 0),
                                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                // Compute block-scoped matrix multiply-add
                if constexpr (BlockMmadTla::DispatchPolicy::ASYNC) {
                    blockMmadTla(tensorBlockA, tensorBlockB, tensorBlockC,
                    actualBlockShape, callbackBeforeFixpipe, callbackAfterFixpipe);
                } else {
                    callbackBeforeFixpipe();
                    blockMmadTla(
                        tensorBlockA, tensorBlockB, tensorBlockC,
                        actualBlockShape
                    );
                    callbackAfterFixpipe();
                }
                stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
            }
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (BlockMmadTla::DispatchPolicy::ASYNC) {
            blockMmadTla.template SynchronizeBlock<decltype(tensorC)>();
        }
        while (stageUsed > 0) {
            uint32_t aivComputeStageId = (stageId >= stageUsed) ?
                (stageId - stageUsed) : (stageId + WORKSPACE_STAGES - stageUsed);
            Arch::CrossCoreWaitFlag(flagAivFinishComputeList[aivComputeStageId]);
            --stageUsed;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        AscendC::ICachePreLoad(1);
        BlockScheduler blockScheduler;
        BlockEpilogue blockEpilogue(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetScale = 0;
        int64_t gmGroupOffsetPerTokenScale = 0;
        int64_t gmGroupOffsetD = 0;

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{L1_TILE_M * coreNum * WORKSPACE_STAGES, L1_TILE_N};

        uint32_t stageId = 0;
        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = groupList.GetValue(groupIdx);
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            LayoutScale layoutScale = params.layoutScale;
            LayoutPerTokenScale layoutPerTokenScale =
                params.layoutPerTokenScale.GetTileLayout(inGroupProblemShape.template GetCoordByAxis<0>());
            LayoutD layoutD = params.layoutD.GetTileLayout(inGroupProblemShape.GetCoordMN());

            EpilogueParams epilogueParams{
                params.ptrScale + gmGroupOffsetScale, layoutScale,
                params.ptrPerTokenScale + gmGroupOffsetPerTokenScale, layoutPerTokenScale,
                params.ptrD + gmGroupOffsetD, layoutD
            };

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
            blockEpilogue.UpdateParams(epilogueParams);
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            GemmCoord blockShapeMNK = MakeCoord(L1_TILE_M, L1_TILE_N, L1_TILE_K);
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoordMNK);

                MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1_TILE_M, 0};
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                auto gmBlockC = gmC[gmOffsetC];
                auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());

                Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
                blockEpilogue(blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);

                stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
            }

            gmGroupOffsetScale += inGroupProblemShape.n();
            gmGroupOffsetPerTokenScale += inGroupProblemShape.m();
            gmGroupOffsetD += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    friend struct AicWaitFunc;
    friend struct AicSetFunc;

    struct AicWaitFunc {
        using MatmulKernel = 
            Gemm::Kernel::GroupedMatmulSliceMPerTokenPerChannelMultiStageTla<BlockMmadTla, 
                BlockEpilogue, BlockScheduler, WORKSPACE_STAGES, ElementGroupList>;

        CATLASS_DEVICE
        AicWaitFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreWaitFlag(ptr->flagAivFinishComputeList[stageId]);
        }

        MatmulKernel *ptr{nullptr};
        uint32_t stageId;
    };

    struct AicSetFunc {
        using MatmulKernel = 
            Gemm::Kernel::GroupedMatmulSliceMPerTokenPerChannelMultiStageTla<BlockMmadTla, 
                BlockEpilogue, BlockScheduler, WORKSPACE_STAGES, ElementGroupList>;

        CATLASS_DEVICE
        AicSetFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(ptr->flagAicFinishStoreList[stageId]);
        }

        MatmulKernel *ptr{nullptr};
        uint32_t stageId;
    };

    Arch::CrossCoreFlag flagAicFinishStoreList[WORKSPACE_STAGES];
    Arch::CrossCoreFlag flagAivFinishComputeList[WORKSPACE_STAGES];

    AicWaitFunc aicWaitFuncList[WORKSPACE_STAGES];
    AicSetFunc aicSetFuncList[WORKSPACE_STAGES];
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_GROUPED_MATMUL_SLICE_M_FP8_TLA_HPP