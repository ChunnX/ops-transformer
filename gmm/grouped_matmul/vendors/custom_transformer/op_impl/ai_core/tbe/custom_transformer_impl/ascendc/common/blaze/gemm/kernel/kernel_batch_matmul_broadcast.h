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
 * \file kernel_batch_matmul_broadcast.h
 * \brief
 */

#pragma once

#include "kernel_basic_intf.h"

#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/utils/common_utils.h"
#include "kernel_universal.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMmadMultiBlockBmmBroadcast, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
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

    struct BatchInfo {
        uint32_t aBatchDim0 = 1UL;
        uint32_t bBatchDim0 = 1UL;
        uint32_t aBatchDim1 = 1UL;
        uint32_t bBatchDim1 = 1UL;
        uint32_t cBatchDim1 = 1UL;
        uint32_t aBatchDim2 = 1UL;
        uint32_t bBatchDim2 = 1UL;
        uint32_t cBatchDim2 = 1UL;
        uint32_t aBatchDim3 = 1UL;
        uint32_t bBatchDim3 = 1UL;
        uint32_t cBatchDim3 = 1UL;
        uint32_t biasBatchDimAll = 1UL;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        BatchInfo batchInfo;
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
        int64_t realCoreNums = bs.GetCoreNums(); // 实际需要的核数
        if (curBlockIdx >= realCoreNums) {
            return;
        }

        Blaze::Gemm::SetHF32(params.schParams.isHf32);

        BlockMmad blockMmad;
        blockMmad.Init(params.mmadParams);

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

        uint64_t preBatchIdx = 0;
        int64_t totalBlockNums = bs.GetBlockNums(); // 切分总块数
        int64_t coreNums = AscendC::GetBlockNum();  // 实际启用的物理核数
        // Process tiles in ping-pong mode
        for (int64_t tileIdx = curBlockIdx; tileIdx < totalBlockNums; tileIdx += coreNums) {
            auto tileShape = bs.template GetBlockShape<TRANS_B, BType>(tileIdx); // 非全载
            auto tileCoord = bs.GetBlockCoord(tileIdx);                          // (m, n, k, b)
            auto coordM = AscendC::Te::Get<MNK_M>(tileCoord);
            auto coordN = AscendC::Te::Get<MNK_N>(tileCoord);
            auto shapeM = AscendC::Te::Get<MNK_M>(tileShape);
            auto shapeN = AscendC::Te::Get<MNK_N>(tileShape);
            auto shapeK = AscendC::Te::Get<MNK_K>(tileShape);
            curBatchIdx_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_B>(tileCoord));

            if (preBatchIdx != curBatchIdx_) {
                uint64_t batchC1Index = curBatchIdx_ / (params.batchInfo.cBatchDim1 * params.batchInfo.cBatchDim2 *
                                                        params.batchInfo.cBatchDim3);
                uint64_t batchC2Index = curBatchIdx_ %
                                        (params.batchInfo.cBatchDim1 * params.batchInfo.cBatchDim2 *
                                         params.batchInfo.cBatchDim3) /
                                        (params.batchInfo.cBatchDim2 * params.batchInfo.cBatchDim3);
                uint64_t batchC3Index = curBatchIdx_ % (params.batchInfo.cBatchDim2 * params.batchInfo.cBatchDim3) /
                                        (params.batchInfo.cBatchDim3);
                uint64_t batchC4Index = curBatchIdx_ % params.batchInfo.cBatchDim3;

                uint64_t batchA1Index = batchC1Index % params.batchInfo.aBatchDim0;
                uint64_t batchA2Index = batchC2Index % params.batchInfo.aBatchDim1;
                uint64_t batchA3Index = batchC3Index % params.batchInfo.aBatchDim2;
                uint64_t batchA4Index = batchC4Index % params.batchInfo.aBatchDim3;
                batchAIndex_ = batchA1Index * (params.batchInfo.aBatchDim1 * params.batchInfo.aBatchDim2 *
                                               params.batchInfo.aBatchDim3) +
                               batchA2Index * (params.batchInfo.aBatchDim2 * params.batchInfo.aBatchDim3) +
                               batchA3Index * params.batchInfo.aBatchDim3 + batchA4Index;

                uint64_t batchB1Index = batchC1Index % params.batchInfo.bBatchDim0;
                uint64_t batchB2Index = batchC2Index % params.batchInfo.bBatchDim1;
                uint64_t batchB3Index = batchC3Index % params.batchInfo.bBatchDim2;
                uint64_t batchB4Index = batchC4Index % params.batchInfo.bBatchDim3;
                batchBIndex_ = batchB1Index * (params.batchInfo.bBatchDim1 * params.batchInfo.bBatchDim2 *
                                               params.batchInfo.bBatchDim3) +
                               batchB2Index * (params.batchInfo.bBatchDim2 * params.batchInfo.bBatchDim3) +
                               batchB3Index * params.batchInfo.bBatchDim3 + batchB4Index;

                UpdateBatchOffset(params);
                gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
                gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB);
                gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);
                if (params.batchInfo.biasBatchDimAll != 1UL) {
                    gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmAddr_),
                                                     layoutBias);
                }
                preBatchIdx = curBatchIdx_;
            }
            // Block offset
            auto gmBlockA = gmA.Slice(AscendC::MakeCoord(coordM, 0L), AscendC::MakeShape(shapeM, shapeK));
            auto gmBlockB = gmB.Slice(AscendC::MakeCoord(0L, coordN), AscendC::MakeShape(shapeK, shapeN));
            auto gmBlockC = gmC.Slice(AscendC::MakeCoord(coordM, coordN), AscendC::MakeShape(shapeM, shapeN));
            auto gmBlockBias = gmBias.Slice(AscendC::MakeCoord(0L, coordN), AscendC::MakeShape(1L, shapeN));
            blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, tileShape);
        }

        Blaze::Gemm::UnsetHF32(params.schParams.isHf32);
    }

private:
    __aicore__ inline void Init(Params const& params)
    {
        auto blockMmadParams = params.mmadParams;
        m_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_K>(params.problemShape));
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams.biasGmAddr);
    }

    __aicore__ inline void UpdateBatchOffset(Params const& params)
    {
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr) + batchAIndex_ * m_ * k_;
        if (!WEIGHT_NZ_FORMAT) {
            bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr) + batchBIndex_ * k_ * n_;
        } else {
            bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr) +
                       Blaze::Gemm::CalWeightNZGmAddrOffset(TRANS_B, batchBIndex_, n_, k_, C0_SIZE);
        }
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr) + curBatchIdx_ * m_ * n_;
        if (params.batchInfo.biasBatchDimAll != 1UL) {
            biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr) + curBatchIdx_ * n_;
        }
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
    static constexpr bool TRANS_B = BlockMmad::TRANS_B;
    static constexpr bool WEIGHT_NZ_FORMAT = BlockMmad::WEIGHT_NZ_FORMAT;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // 可选输入，直接初始化

    uint64_t curBatchIdx_ = {0};
    uint64_t batchAIndex_ = {0};
    uint64_t batchBIndex_ = {0};
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
