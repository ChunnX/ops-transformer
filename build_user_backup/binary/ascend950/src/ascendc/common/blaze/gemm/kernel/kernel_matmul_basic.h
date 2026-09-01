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

#include "kernel_basic_intf.h"

#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_mmad_matmul_al1_full_load.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "kernel_universal.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMmadMultiBlockBasic, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
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
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<BType>>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<CType>>>;
    using MakeLayoutBias = AscendC::Te::FrameLayoutFormat<LayoutBias,
                                                          AscendC::Std::Int<AscendC::Te::C0_ELEMENT<BiasType>>>;
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline void operator()(Params& params)
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
        if constexpr (AscendC::Std::is_same_v<
                          MatmulMultiBlockBasicSplitK<0, 1, Blaze::Gemm::KernelMmadMultiBlockBasic, 0>,
                          typename BlockMmad_::DispatchPolicy>) {
            params.mmadParams.k = k_;
        }
        blockMmad.Init(params.mmadParams);

        if constexpr (NON_CONTIGUOUS_TYPE == static_cast<uint64_t>(NoContiguousType::NON_CONTIGUOUS_TYPE_SLICE)) {
            MatmulSliceProcess(params, blockMmad, bs, curBlockIdx, AscendC::GetBlockNum(), bs.GetBlockNums());
        } else {
            MatmulProcess(params, blockMmad, bs, curBlockIdx, AscendC::GetBlockNum(), bs.GetBlockNums());
        }

        Blaze::Gemm::UnsetHF32(params.schParams.isHf32);
    }

private:
    __aicore__ inline auto MakeLayoutAGm(Params const& params)
    {
        uint64_t aBatch = IS_A_FULL_LOAD ? 1UL : batch_;
        if constexpr (!TRANS_A) {
            // 连续场景下rowStride表示k或1, 非连续场景下表示m轴的stride
            uint64_t rowStride = params.mmadParams.rowStride == 0 ? k_ : params.mmadParams.rowStride;
            uint64_t batchStride = m_ * rowStride;
            return AscendC::Te::MakePatternLayout<LayoutA, AscendC::Te::LayoutTraitDefault<>>(
                AscendC::Te::MakeShape(aBatch, AscendC::Te::MakeShape(AscendC::Te::MakeShape(AscendC::Te::_1{}, m_),
                                                                      AscendC::Te::MakeShape(AscendC::Te::_1{}, k_))),
                AscendC::Te::MakeStride(
                    batchStride,
                    AscendC::Te::MakeStride(AscendC::Te::MakeStride(AscendC::Te::_0{}, rowStride),
                                            AscendC::Te::MakeStride(AscendC::Te::_0{}, AscendC::Te::_1{}))));
        } else {
            return MakeLayoutA{}(aBatch, m_, k_);
        }
    }

    __aicore__ inline void MatmulProcess(Params const& params, BlockMmad& blockMmad, BlockScheduler& bs,
                                         int64_t curBlockIdx, int64_t coreNums, int64_t totalBlockNums)
    {
        auto layoutA = MakeLayoutAGm(params);
        auto layoutB = MakeLayoutB{}(batch_, k_, n_); // ND layout for B
        auto layoutC = MakeLayoutC{}(batch_, m_, n_); // ND layout for C
        auto layoutBias = MakeLayoutBias{}(1L, n_);   // ND layout for Bias
        // A,B,C Gm Tensor
        auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
        auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB);
        auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);
        auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmAddr_),
                                              layoutBias);

        // 使能双页表
        SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);

        // Process tiles in ping-pong mode
        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
            auto blockShape = bs.template GetBlockShape<TRANS_B, BType>(blockIdx); // (m, n, k, b)
            auto blockCoord = bs.GetBlockCoord(blockIdx);                          // (m, n, k, b)
            auto coordM = AscendC::Te::Get<MNK_M>(blockCoord);
            auto coordN = AscendC::Te::Get<MNK_N>(blockCoord);
            auto shapeM = AscendC::Te::Get<MNK_M>(blockShape);
            auto shapeN = AscendC::Te::Get<MNK_N>(blockShape);
            auto shapeK = AscendC::Te::Get<MNK_K>(blockShape);
            curBatchIdx_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_B>(blockCoord));

            // Block offset
            // A full-load shares one complete A matrix across batches; other modes select the current A tile.
            auto batchIdxA = IS_A_FULL_LOAD ? 0L : curBatchIdx_;
            auto coordMA = IS_A_FULL_LOAD ? 0L : coordM;
            auto shapeMA = IS_A_FULL_LOAD ? m_ : shapeM;
            auto subTensorA = gmA.Slice(AscendC::MakeCoord(batchIdxA, AscendC::MakeCoord(coordMA, 0L)),
                                        AscendC::MakeShape(1L, AscendC::MakeShape(shapeMA, shapeK)));
            auto gmBlockA = AscendC::Te::Squeeze<0>(subTensorA);
            auto subTensorB = gmB.Slice(AscendC::MakeCoord(curBatchIdx_, AscendC::MakeCoord(0L, coordN)),
                                        AscendC::MakeShape(1L, AscendC::MakeShape(shapeK, shapeN)));
            auto gmBlockB = AscendC::Te::Squeeze<0>(subTensorB);
            auto subTensorC = gmC.Slice(AscendC::MakeCoord(curBatchIdx_, AscendC::MakeCoord(coordM, coordN)),
                                        AscendC::MakeShape(1L, AscendC::MakeShape(shapeM, shapeN)));
            auto gmBlockC = AscendC::Te::Squeeze<0>(subTensorC);
            auto gmBlockBias = gmBias.Slice(AscendC::MakeCoord(0L, coordN), AscendC::MakeShape(1L, shapeN));
            blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, blockShape);
        }
    }

    __aicore__ inline auto MakeLayoutA3DForSlice(Params const& params)
    {
        int64_t sliceM = static_cast<int64_t>(params.schParams.sliceM);
        int64_t sliceBatch = static_cast<int64_t>(m_) / sliceM;
        int64_t srcNdStride = static_cast<int64_t>(params.schParams.srcNdStride);

        return AscendC::Te::MakePatternLayout<
            NDSliceLayoutPtn, AscendC::Te::LayoutTrait<AType, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<AType>>>>(
            AscendC::Te::MakeShape(sliceBatch, AscendC::Te::MakeShape(sliceM, k_)),
            AscendC::Te::MakeStride(srcNdStride, AscendC::Te::MakeStride(k_, 1L)));
    }

    __aicore__ inline void MatmulSliceProcess(Params const& params, BlockMmad& blockMmad, BlockScheduler& bs,
                                              int64_t curBlockIdx, int64_t coreNums, int64_t totalBlockNums)
    {
        auto layoutA = MakeLayoutA3DForSlice(params);
        auto layoutB = MakeLayoutB{}(k_, n_);
        auto layoutC = MakeLayoutC{}(m_, n_);
        auto layoutBias = MakeLayoutBias{}(1L, n_);

        auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
        auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB);
        auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);
        auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmAddr_),
                                              layoutBias);
        SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);

        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
            auto blockShape = bs.template GetBlockShape<TRANS_B, BType>(blockIdx);
            auto blockCoord = bs.GetBlockCoord(blockIdx);
            auto coordM = AscendC::Te::Get<MNK_M>(blockCoord);
            auto coordN = AscendC::Te::Get<MNK_N>(blockCoord);
            auto shapeM = AscendC::Te::Get<MNK_M>(blockShape);
            auto shapeN = AscendC::Te::Get<MNK_N>(blockShape);
            auto shapeK = AscendC::Te::Get<MNK_K>(blockShape);

            int64_t sliceM = static_cast<int64_t>(params.schParams.sliceM);
            auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(coordM / sliceM, AscendC::Te::MakeCoord(0L, 0L)),
                                      AscendC::Te::MakeShape(shapeM / sliceM, AscendC::Te::MakeShape(sliceM, shapeK)));
            auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(0L, coordN), AscendC::Te::MakeShape(shapeK, shapeN));
            auto gmBlockC = gmC.Slice(AscendC::Te::MakeCoord(coordM, coordN), AscendC::Te::MakeShape(shapeM, shapeN));
            auto gmBlockBias = gmBias.Slice(AscendC::Te::MakeCoord(0L, coordN), AscendC::Te::MakeShape(1L, shapeN));
            blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, blockShape);
        }
    }

    __aicore__ inline void Init(Params const& params)
    {
        auto blockMmadParams = params.mmadParams;
        m_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_K>(params.problemShape));
        batch_ = static_cast<uint64_t>(AscendC::Std::max(AscendC::Te::Get<MNK_B>(params.problemShape), 1L));
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams.biasGmAddr);
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
    static constexpr bool TRANS_A = BlockMmad::TRANS_A;
    static constexpr bool TRANS_B = BlockMmad::TRANS_B;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = BlockMmad::NON_CONTIGUOUS_TYPE;
    static constexpr bool IS_A_FULL_LOAD = BlockMmad::DispatchPolicy::FULL_LOAD_MODE == A_FULL_LOAD_MODE;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // 可选输入，直接初始化

    uint64_t curBatchIdx_ = {0};
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t batch_{1};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
