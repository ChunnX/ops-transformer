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
 * \file kernel_matmul_b_full_load.h
 * \brief
 */

#pragma once

#include "kernel_basic_intf.h"

#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_b_fullLoad_fixpipe_opti.h"
#include "blaze/gemm/utils/common_utils.h"
#include "kernel_universal.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMmadMultiBlockBFullLoad, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
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
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<AType>>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<BType>>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<CType>>>;
    using MakeLayoutBias = AscendC::Te::FrameLayoutFormat<LayoutBias,
                                                          AscendC::Std::Int<AscendC::Te::C0_ELEMENT<BiasType>>>;

    static constexpr bool TRANS_B = BlockMmad::TRANS_B;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>>;

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline void operator()(Params const& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        Init(params);

        // 初始化blockScheduler
        BlockScheduler bs(params.problemShape, params.schParams);
        int64_t curBlockIdx = AscendC::GetBlockIdx();

        int64_t realCoreNums = bs.GetCoreNums();
        if (curBlockIdx >= realCoreNums) {
            return;
        }
        Blaze::Gemm::SetHF32(params.schParams.isHf32);

        BlockMmad blockMmad;
        blockMmad.Init(params.mmadParams);
        MatmulProcess(params, blockMmad, bs, curBlockIdx, AscendC::GetBlockNum(), bs.GetBlockNums());
        Blaze::Gemm::UnsetHF32(params.schParams.isHf32);
    }

private:
    __aicore__ inline void MatmulProcess(Params const& params, BlockMmad& blockMmad, BlockScheduler& bs,
                                         int64_t curBlockIdx, int64_t coreNums, int64_t totalBlockNums)
    {
        // 默认ND Format
        auto layoutA = MakeLayoutA{}(m_, k_);       // ND layout for A
        auto layoutB = MakeLayoutB{}(k_, n_);       // ND layout for B
        auto layoutC = MakeLayoutC{}(m_, n_);       // ND layout for C
        auto layoutBias = MakeLayoutBias{}(1L, n_); // ND layout for Bias
        // A,B,C Gm Tensor
        auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
        auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB);
        auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);
        auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmAddr_),
                                              layoutBias);

        // 使能双页表
        SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);

        uint64_t preBatchIdx = 0;
        // Process tiles in ping-pong mode
        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
            auto tileShape = bs.template GetBlockShape<TRANS_B, BType>(blockIdx);
            auto tileCoord = bs.GetBlockCoord(blockIdx);
            auto coordM = AscendC::Te::Get<MNK_M>(tileCoord);
            // auto coordN = AscendC::Te::Get<MNK_N>(tileCoord);
            auto shapeM = AscendC::Te::Get<MNK_M>(tileShape);
            auto shapeN = AscendC::Te::Get<MNK_N>(tileShape);
            auto shapeK = AscendC::Te::Get<MNK_K>(tileShape);
            curBatchIdx_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_B>(tileCoord));
            // Block offset
            if (preBatchIdx != curBatchIdx_) {
                UpdateBatchOffset(params);
                gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
                gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);
                preBatchIdx = curBatchIdx_;
                SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);
            }
            // Block offset
            auto gmBlockA = gmA.Slice(AscendC::MakeCoord(coordM, 0L), AscendC::MakeShape(shapeM, shapeK));
            auto gmBlockC = gmC.Slice(AscendC::MakeCoord(coordM, 0), AscendC::MakeShape(shapeM, n_));
            blockMmad(gmBlockA, gmB, gmBias, gmBlockC, tileShape);
        }
    }
    __aicore__ inline void Init(Params const& params)
    {
        auto blockMmadParams = params.mmadParams;
        m_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_K>(params.problemShape));
        kAlign_ = Blaze::Gemm::CeilAlign(k_, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        isBias_ = blockMmadParams.biasGmAddr != nullptr;
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams.biasGmAddr);
    }

    __aicore__ inline void UpdateBatchOffset(Params const& params)
    {
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr) + curBatchIdx_ * m_ * k_;
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr) + curBatchIdx_ * m_ * n_;
    }

    template <typename TensorA, typename TensorB>
    __aicore__ inline void SetL2Cache(TensorA& gmA, TensorB& gmB, uint32_t l2CacheMode)
    {
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == B_L2_CACHE_DISABLE) {
            gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
        }
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == A_L2_CACHE_DISABLE) {
            gmA.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
        }
    }

private:
    static constexpr bool IS_FP32 = (AscendC::Std::is_same_v<BType, float>);
    static constexpr int64_t C0_SIZE = IS_FP32 ? C0_SIZE_fp32 : C0_SIZE_fp16;
    static constexpr bool TRANS_A = BlockMmad::TRANS_A;
    static constexpr bool WEIGHTNZ_FORMAT = BlockMmad::WEIGHTNZ_FORMAT;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // 可选输入，直接初始化

    uint64_t curBatchIdx_ = {0};
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t kAlign_{1};
    bool isBias_{false};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
