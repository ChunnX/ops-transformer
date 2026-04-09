/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_MIX_FIXPIPE_OPTI_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_MIX_FIXPIPE_OPTI_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

template <
    class ProblemShape_,
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class KernelMatmulMixFixpipeOpti {
public:
    CATLASS_DEVICE 
    KernelMatmulMixFixpipeOpti()
    {
        if ASCEND_IS_AIV {
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);      // ping
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + 1);  // pong
        }
    }
    CATLASS_DEVICE
    ~KernelMatmulMixFixpipeOpti()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG);                    // ping
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);      // ping
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + 1);                // pong
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + 1 + FLAG_ID_MAX);  // pong
        }
    }

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using LayoutTagA = typename BlockMmad::LayoutTagA;
    using LayoutTagB = typename BlockMmad::LayoutTagB;
    using LayoutTagC = typename BlockMmad::LayoutTagC;

    // Block Argument / Params
    using BlockMmadArguments = typename BlockMmad::Arguments;
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockSchedulerArguments = typename BlockScheduler::Arguments;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    using TupleShape = tla::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = tla::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = tla::Coord<int64_t, int64_t, int64_t, int64_t>;
    
    AscendC::GlobalTensor<ElementA> aGlobal_;
    AscendC::GlobalTensor<ElementB> bGlobal_;
    AscendC::GlobalTensor<ElementC> cGlobal_;

    // shape
    TupleShape problemShape_{};

    constexpr static uint64_t ML1_ = BlockMmad::ML1_;
    constexpr static uint64_t NL1_ = BlockMmad::NL1_;
    constexpr static uint64_t KL1_ = BlockMmad::KL1_;

    constexpr static uint64_t ML0_ = BlockMmad::ML0_;
    constexpr static uint64_t NL0_ = BlockMmad::NL0_;
    constexpr static uint64_t KL0_ = BlockMmad::KL0_;

    constexpr static uint64_t UB_TILE_SIZE = ArchTag::UB_SIZE / 2;

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        BlockEpilogueArguments epilogueArgs;
        BlockSchedulerArguments schArgs;
        Arguments() = default;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    CATLASS_DEVICE 
    static TupleShape ToShapeTuple(ProblemShape const &shape)
    {
        return {shape.m, shape.n, shape.k, shape.b};
    }

    CATLASS_DEVICE 
    void Init(Params const &params)
    {
        problemShape_ = ToShapeTuple(params.problemShape);
        BlockMmadParams blockMmadParams = params.mmadParams;
        BlockEpilogueParams blockEpilogueParams = params.epilogueParams;
        m_ = tla::get<0>(problemShape_);
        n_ = tla::get<1>(problemShape_);
        k_ = tla::get<2>(problemShape_);
        // Init GlobalTensor
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(blockMmadParams.aGmAddr), m_ * k_);
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(blockMmadParams.bGmAddr), k_ * n_);
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(blockEpilogueParams.outGmAddr), m_ * n_);
        // Init UbTensor
        for (uint32_t i = 0; i < NUM_TWO; ++i) {
            ubTensorList_[i] = resource.ubBuf.template GetBufferByByte<ElementC>(UB_TILE_SIZE * i);
        }
    }

    CATLASS_DEVICE 
    void operator()(Params const &params)
    {
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        // Init
        Init(params);
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m_, k_);
        auto aTlaTensor = tla::MakeTensor(aGlobal_, layoutA, Arch::PositionGM{});

        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k_, n_);
        auto bTlaTensor = tla::MakeTensor(bGlobal_, layoutB, Arch::PositionGM{});

        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m_, n_);
        auto cTlaTensor = tla::MakeTensor(cGlobal_, layoutC, Arch::PositionGM{});

        bool enableDualDst = (AscendC::IsSameType<ElementC, float>::value);
        if ASCEND_IS_AIV {
            if (!enableDualDst && AscendC::GetSubBlockIdx() > 0) {
                return;
            }
            curBlockIdx /= AscendC::GetTaskRation();
        }
        // Instantiate
        BlockMmad blockMmadOp(resource, problemShape_);
        BlockEpilogue epilogueOp(params.epilogueParams, problemShape_);
        BlockScheduler bs(params.problemShape, curBlockIdx, blockNum, params.schParams);

        int64_t tileNum = bs.GetTileNum();
        int64_t realBlockNum = bs.GetBlockNum(params.problemShape, blockNum);
        bool isHf32 = bs.GetHf32Flag();
        bool enableUbDB = bs.GetUbDB();
        if (curBlockIdx >= realBlockNum) {
            return;
        }

        if (isHf32) {
            AscendC::SetHF32Mode(1);
            AscendC::SetHF32TransMode(1);
        }

        uint64_t cvIndex = 0;
        uint64_t cvPingPongBit = enableUbDB ? 1 : 0;
        
        // Process tiles in ping-pong mode
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            auto blockCoord = bs.GetBlockCoord(tileIdx);
            TupleShape tileShape = bs.GetBlockShape(tileIdx);
            uint64_t mMaxLimit = tla::get<0>(tileShape);
            uint64_t nMaxLimit = tla::get<1>(tileShape);
            // mIter
            for (uint64_t mOffset = 0; mOffset < mMaxLimit; mOffset += ML0_) {
                // nIter
                for (uint64_t nOffset = 0; nOffset < nMaxLimit; nOffset += NL0_) {
                    BlockShape blockShape = bs.GetBlockShape(tileIdx, mOffset, nOffset);
                    if (tla::get<0>(blockShape) <= 0 || tla::get<1>(blockShape) <= 0) {
                        if (isHf32) {
                            AscendC::SetHF32Mode(0);
                        }
                        return;
                    }
                    
                    int64_t mCoord = tla::get<0>(blockCoord) + mOffset;
                    int64_t nCoord = tla::get<1>(blockCoord) + nOffset;
                    int64_t kCoord = tla::get<2>(blockCoord);
                    int64_t curML0 = tla::get<4>(blockShape);
                    int64_t curNL0 = tla::get<5>(blockShape);

                    auto aTileTensor = GetTile(
                        aTlaTensor,
                        tla::MakeCoord(mCoord, kCoord),
                        tla::MakeShape(curML0, k_)
                    );

                    auto bTileTensor = GetTile(
                        bTlaTensor,
                        tla::MakeCoord(kCoord, nCoord),
                        tla::MakeShape(k_, curNL0)
                    );

                    auto cTileTensor = GetTile(
                        cTlaTensor,
                        tla::MakeCoord(mCoord, nCoord),
                        tla::MakeShape(curML0, curNL0)
                    );
                    
                    uint16_t pingPongIdx = cvIndex & cvPingPongBit;                    
                    int64_t alignNL0 = RoundUp(curNL0, static_cast<int64_t>(Catlass::BYTE_PER_BLK / sizeof(ElementC)));
                    auto ubLayout = tla::MakeLayout<ElementC, LayoutTagC>(curML0, alignNL0);
                    auto cLocalTensor = tla::MakeTensor(ubTensorList_[pingPongIdx], ubLayout, Arch::PositionUB{});

                    if ASCEND_IS_AIC {
                        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + (pingPongIdx));
                        if (enableDualDst) {
                            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                                AIV_SYNC_AIC_FLAG + (pingPongIdx) + FLAG_ID_MAX);
                        }

                        blockMmadOp(
                            cLocalTensor,
                            aTileTensor,
                            bTileTensor,
                            blockShape,
                            tileIdx == curBlockIdx);
                        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + (pingPongIdx));
                        if (enableDualDst) {
                            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                                AIC_SYNC_AIV_FLAG + (pingPongIdx) + FLAG_ID_MAX);
                        }
                    }
                    if ASCEND_IS_AIV {
                        // Synchronize with aic
                        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + (pingPongIdx));
                        // Calulate epilogue
                        epilogueOp(cTileTensor, cLocalTensor, enableDualDst);
                        // Notify aic
                        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + (pingPongIdx));
                    }
                    cvIndex++;
                }
            }
        }
        if (isHf32) {
            AscendC::SetHF32Mode(0);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_HOST_DEVICE 
    static bool CanImplement(Arguments const &args)
    { 
        return true;
    }

    CATLASS_HOST_DEVICE 
    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    CATLASS_HOST_DEVICE 
    static Params ToUnderlyingArguments(Arguments const &args, GM_ADDR workspace)
    {
        return {args.problemShape, args.mmadArgs, args.epilogueArgs, args.schArgs};
    }

    static int64_t GetBlockNum(ProblemShape shape)
    {
        return BlockScheduler::GetBlockNum(shape);
    }

private:
    Arch::Resource<ArchTag> resource;
    constexpr static uint16_t NUM_TWO = 2;
    AscendC::LocalTensor<ElementC> ubTensorList_[NUM_TWO];
    constexpr static uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;
    constexpr static uint32_t MATRIX_INNER_DIM_LIMIT_SIZE = 65536;
    constexpr static int32_t MATMUL_MNK_ALIGN = 16;
    constexpr static int16_t FLAG_ID_MAX = 16;
    int64_t m_{1};
    int64_t n_{1};
    int64_t k_{1};
};

}  // namespace Catlass::Gemm::Kernel
#endif //CATLASS_GEMM_KERNEL_MATMUL_MIX_FIXPIPE_OPTI_HPP