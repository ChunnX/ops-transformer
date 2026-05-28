/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_matmul_basic.h
 * \brief
 */

#pragma once

#define ASCENDC_CUBE_ONLY

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "../../epilogue/block/block_epilogue_empty.h"
#include "../block/block_mmad.h"
#include "../block/block_mmad_matmul_basic.h"
#include "../utils/common_utils.h"
#include "kernel_universal.h"
#include "../../../tensor_api/include/tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<KernelMmadMultiBlockBasic, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal()
    {}
    __aicore__ inline ~GemmUniversal()
    {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;
    static constexpr bool weightNZFormat = BlockMmad::weightNZFormat;
    // mmad
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using TupleShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<AscendC::AuxGetC0Size<AType>()>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<AscendC::AuxGetC0Size<BType>()>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::AuxGetC0Size<CType>()>>;
    using MakeLayoutBias =
        AscendC::Te::FrameLayoutFormat<LayoutBias, AscendC::Std::Int<AscendC::AuxGetC0Size<BiasType>()>>;
    static constexpr bool isFp32 = (std::is_same_v<BType, float>);

    // shape
    TupleShape problemShape_{};
    BlockMmadParams blockMmadParams_{};
    bool isBias_ = false;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // 可选输入，直接初始化

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schedulerParams;
        Params() = default;
    };

    __aicore__ inline void Init(Params const& params)
    {
        problemShape_ = params.problemShape;
        blockMmadParams_ = params.mmadParams;
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
        if (blockMmadParams_.biasGmAddr != nullptr) {
            isBias_ = true;
            biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
        }
    }

    __aicore__ inline void UnsetHf32(bool isHf32)
    {
        if (isHf32) {
            AscendC::SetHF32Mode(0);
        }
    }

    __aicore__ inline void operator()(Params const& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        // 初始化mmad
        BlockMmad blockMmad;
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        Init(params);

        // 初始化blockScheduler
        BlockScheduler bs(params.problemShape, curBlockIdx, blockNum, params.schedulerParams, isFp32, !weightNZFormat);

        int64_t tileNum = bs.GetTileNum();
        TupleShape tileL1 = bs.GetTileL1Shape();
        TupleShape tileL0 = bs.GetTileL0Shape();
        int64_t realBlockNum = bs.GetBlockNum(params.problemShape, blockNum);
        if (curBlockIdx >= realBlockNum) {
            return;
        }
        bool isHf32 = bs.Gethf32Flag();
        if (isHf32) {
            AscendC::SetHF32Mode(1);
            AscendC::SetHF32TransMode(1);
        }
        SetMMLayoutTransform(true); // Set Mmad output as cloumn major for Fixpipe
        blockMmad.Init(problemShape_, tileL1, tileL0, isBias_, bs.GetL1BuferNum_(), bs.GetL0cDB());

        int64_t m = AscendC::Te::Get<MNK_M>(problemShape_);
        int64_t n = AscendC::Te::Get<MNK_N>(problemShape_);
        int64_t k = AscendC::Te::Get<MNK_K>(problemShape_);

        // 默认ND Format
        auto layoutA = MakeLayoutA{}(m, k);        // ND layout for A
        auto layoutB = MakeLayoutB{}(k, n);        // ND layout for B
        auto layoutC = MakeLayoutC{}(m, n);        // ND layout for C
        auto layoutBias = MakeLayoutBias{}(1L, n); // ND layout for Bias
        // A,B,C Gm Tensor
        auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
        auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB);
        auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);
        auto gmBias =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmAddr_), layoutBias);

        // 使能双页表
        if (bs.GetBL2CacheDisable()) {
            gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
        }
        if (bs.GetAL2CacheDisable()) {
            gmA.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
        }

        // Process tiles in ping-pong mode
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            auto tileShape = bs.template GetBlockShape<transB, BType>(tileIdx); // 非全载
            auto tileCoord = bs.GetBlockCoord(tileIdx);                         // (m, n, k, b)
            auto coordM = AscendC::Te::Get<MNK_M>(tileCoord);
            auto coordN = AscendC::Te::Get<MNK_N>(tileCoord);
            auto shapeM = AscendC::Te::Get<MNK_M>(tileShape);
            auto shapeN = AscendC::Te::Get<MNK_N>(tileShape);
            auto shapeK = AscendC::Te::Get<MNK_K>(tileShape);
            // Block offset
            auto gmBlockA = gmA.Slice(AscendC::MakeCoord(coordM, 0L), AscendC::MakeShape(shapeM, shapeK));
            auto gmBlockB = gmB.Slice(AscendC::MakeCoord(0L, coordN), AscendC::MakeShape(shapeK, shapeN));
            auto gmBlockC = gmC.Slice(AscendC::MakeCoord(coordM, coordN), AscendC::MakeShape(shapeM, shapeN));
            auto gmBlockBias = gmBias.Slice(AscendC::MakeCoord(0L, coordN), AscendC::MakeShape(1L, shapeN));
            blockMmad(gmBlockC, gmBlockA, gmBlockB, gmBlockBias, tileShape);
        }
        SetMMLayoutTransform(false);
        UnsetHf32(isHf32);
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
