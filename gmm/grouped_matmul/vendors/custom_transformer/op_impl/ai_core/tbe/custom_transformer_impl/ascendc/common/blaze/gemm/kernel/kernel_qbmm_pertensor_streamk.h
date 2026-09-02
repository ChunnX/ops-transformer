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
 * \file kernel_qbmm_pertensor_streamk.h
 * \brief GemmUniversal specialization for QBMM per-tensor StreamK.
 */

#pragma once

#include "kernel_universal.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

/**
 * QBMM per-tensor StreamK specialization.
 *
 * DP blocks use fixpipe dequantization and write C directly. StreamK blocks
 * write raw accumulation partials to workspace for AIV reduction.
 */
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<KernelQbmmPertensorMultiBlockStreamK,
                                                      typename BlockMmad_::DispatchPolicy::ScheduleType> &&
                              AscendC::Std::is_same_v<KernelQbmmPertensorMultiBlockStreamK,
                                                      typename BlockEpilogue_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using X2ScaleType = typename BlockMmad::X2ScaleType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using WorkspaceType = typename BlockEpilogue::WorkspaceType;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Te::LayoutTraitDefault<AType>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Te::LayoutTraitDefault<BType>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Te::LayoutTraitDefault<CType>>;
    using MakeLayoutBias = AscendC::Te::FrameLayoutFormat<LayoutBias, AscendC::Te::LayoutTraitDefault<BiasType>>;

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams blockMmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline void operator()(Params const& params)
    {
        if (params.schParams.usedCoreNum <= 0 || AscendC::Te::Get<MNK_B>(params.problemShape) != 1) {
            return;
        }
        Init(params);

        BlockScheduler bs(params.problemShape, params.schParams);
        mL1_ = params.schParams.baseM;
        nL1_ = params.schParams.baseN;
        mBlockNums_ = Blaze::Gemm::CeilDiv(m_, mL1_);
        nBlockNums_ = Blaze::Gemm::CeilDiv(n_, nL1_);
        skBlockNums_ = Blaze::Gemm::CeilDiv(k_, params.schParams.singleCoreK);

        if ASCEND_IS_AIC {
            ProcessOnAic(params, bs);
        }

        if ASCEND_IS_AIV {
            ProcessOnAiv(params, bs);
        }
    }

private:
    __aicore__ inline void ProcessOnAic(Params const& params, BlockScheduler& bs)
    {
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        if (curBlockIdx < bs.GetCoreNums()) {
            ProcessAicBlocks(params, bs, curBlockIdx);
        }
        SignalAicFinish();
    }

    __aicore__ inline void ProcessAicBlocks(Params const& params, BlockScheduler& bs, int64_t curBlockIdx)
    {
        int64_t usedCoreNum = params.schParams.usedCoreNum;
        BlockMmad blockMmad;
        BlockShape l0BlockShape{params.schParams.baseM, params.schParams.baseN, params.schParams.baseK, 0};
        blockMmad.Init(params.problemShape, l0BlockShape, params.schParams.kL1, params.schParams.kL1, 2UL,
                       QuantMode::PERTENSOR_MODE, params.blockMmadParams.biasGmAddr != nullptr, false);

        int64_t totalBlockNums = bs.GetBlockNums();
        int64_t tailSKTotalBlockNums = (mBlockNums_ * nBlockNums_) % usedCoreNum * skBlockNums_;
        int64_t totalMNBlockNumsInDP = mBlockNums_ * nBlockNums_ - (mBlockNums_ * nBlockNums_) % usedCoreNum;

        auto layoutA = MakeLayoutA{}(m_, k_);
        auto layoutB = MakeLayoutB{}(k_, n_);
        auto layoutC = MakeLayoutC{}(m_, n_);
        auto layoutBias = MakeLayoutBias{}(1L, n_);
        auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
        auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB);
        auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);
        auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmAddr_),
                                              layoutBias);

        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += usedCoreNum) {
            int64_t actualBlockIdx = GetActualBlockIdx(bs, blockIdx, totalBlockNums, tailSKTotalBlockNums, usedCoreNum);
            ProcessAicBlock(params, bs, blockMmad, gmA, gmB, gmBias, gmC, actualBlockIdx, totalMNBlockNumsInDP);
        }
    }

    __aicore__ inline void SignalAicFinish() const
    {
        AscendC::CrossCoreSetFlag<AIC_ONLY_SYNC_MODE, PIPE_FIX>(AIC_ONLY_SYNC_FLAG);
        AscendC::CrossCoreWaitFlag(AIC_ONLY_SYNC_FLAG);
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
    }

    __aicore__ inline int64_t GetActualBlockIdx(BlockScheduler& bs, int64_t blockIdx, int64_t totalBlockNums,
                                                int64_t tailSKTotalBlockNums, int64_t usedCoreNum) const
    {
        if (bs.CheckIsSkScene(0)) {
            return blockIdx;
        }
        bool preloadSK = blockIdx % usedCoreNum < tailSKTotalBlockNums &&
                         Blaze::Gemm::CeilDiv(blockIdx + 1, usedCoreNum) ==
                             Blaze::Gemm::CeilDiv(totalBlockNums, usedCoreNum) - 1;
        if (preloadSK) {
            return blockIdx + usedCoreNum;
        }
        bool moveBackSK = blockIdx % usedCoreNum < tailSKTotalBlockNums &&
                          Blaze::Gemm::CeilDiv(blockIdx + 1, usedCoreNum) ==
                              Blaze::Gemm::CeilDiv(totalBlockNums, usedCoreNum);
        return moveBackSK ? blockIdx - usedCoreNum : blockIdx;
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void ProcessAicBlock(Params const& params, BlockScheduler& bs, BlockMmad& blockMmad, TensorA& gmA,
                                           TensorB& gmB, TensorBias& gmBias, TensorC& gmC, int64_t blockIdx,
                                           int64_t totalMNBlockNumsInDP)
    {
        BlockShape singleCoreShape = bs.GetBlockShape(blockIdx);
        BlockShape singleCoreCoord = bs.GetBlockCoord(blockIdx);
        bool isSkBlock = bs.CheckIsSkScene(blockIdx);
        int64_t kSingleCore = isSkBlock ? params.schParams.singleCoreK : k_;
        int64_t offsetWorkspace = (isSkBlock ? (blockIdx - totalMNBlockNumsInDP) : 0) * BLOCK_BASE_M * BLOCK_BASE_N;
        auto gmWorkSpace = MakeWorkspaceTensor(singleCoreShape, offsetWorkspace);

        auto gmBlockA = gmA.Slice(
            AscendC::Te::MakeCoord(AscendC::Te::Get<MNK_M>(singleCoreCoord) * mL1_,
                                   AscendC::Te::Get<MNK_K>(singleCoreCoord) * kSingleCore),
            AscendC::Te::MakeShape(AscendC::Te::Get<MNK_M>(singleCoreShape), AscendC::Te::Get<MNK_K>(singleCoreShape)));
        auto gmBlockB = gmB.Slice(
            AscendC::Te::MakeCoord(AscendC::Te::Get<MNK_K>(singleCoreCoord) * kSingleCore,
                                   AscendC::Te::Get<MNK_N>(singleCoreCoord) * nL1_),
            AscendC::Te::MakeShape(AscendC::Te::Get<MNK_K>(singleCoreShape), AscendC::Te::Get<MNK_N>(singleCoreShape)));
        auto gmBlockC = gmC.Slice(
            AscendC::Te::MakeCoord(AscendC::Te::Get<MNK_M>(singleCoreCoord) * mL1_,
                                   AscendC::Te::Get<MNK_N>(singleCoreCoord) * nL1_),
            AscendC::Te::MakeShape(AscendC::Te::Get<MNK_M>(singleCoreShape), AscendC::Te::Get<MNK_N>(singleCoreShape)));
        auto gmBlockBias = gmBias.Slice(AscendC::Te::MakeCoord(0L, AscendC::Te::Get<MNK_N>(singleCoreCoord) * nL1_),
                                        AscendC::Te::MakeShape(1L, AscendC::Te::Get<MNK_N>(singleCoreShape)));

        blockMmad(gmBlockA, gmBlockB, scaleScalar_, gmBlockBias, gmBlockC, gmWorkSpace, singleCoreShape,
                  AscendC::Te::Get<MNK_K>(singleCoreCoord), isSkBlock);
    }

    __aicore__ inline void ProcessOnAiv(Params const& params, BlockScheduler& bs)
    {
        uint64_t curBlockIdxInAiv = AscendC::GetBlockIdx();
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_MTE2>(AIC_SYNC_AIV_FLAG);
        int64_t usedCoreNum = params.schParams.usedCoreNum;
        uint64_t lastLoopTotalCnt = static_cast<uint64_t>((mBlockNums_ * nBlockNums_) % usedCoreNum * skBlockNums_);
        if (curBlockIdxInAiv >= lastLoopTotalCnt * AscendC::GetTaskRation()) {
            return;
        }

        BlockEpilogue epilogueOp;
        BlockShape l1Block = {params.schParams.baseM, params.schParams.baseN, params.schParams.kL1, 1};
        epilogueOp.Init(params.epilogueParams, params.problemShape, l1Block,
                        {mBlockNums_, nBlockNums_, skBlockNums_, 1}, usedCoreNum, bs.CheckIsSkScene(0));
        epilogueOp();
    }

    __aicore__ inline void Init(Params const& params)
    {
        m_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_K>(params.problemShape));
        if ASCEND_IS_AIC {
            auto mmParams = params.blockMmadParams;
            aGmAddr_ = reinterpret_cast<__gm__ AType*>(mmParams.aGmAddr);
            bGmAddr_ = reinterpret_cast<__gm__ BType*>(mmParams.bGmAddr);
            cGmAddr_ = reinterpret_cast<__gm__ CType*>(mmParams.cGmAddr);
            workspaceGmAddr_ = reinterpret_cast<__gm__ WorkspaceType*>(params.epilogueParams.workspaceGmAddr);
            biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(mmParams.biasGmAddr);
            InitScale(params);
        }
    }

    __aicore__ inline void InitScale(Params const& params)
    {
        if (params.epilogueParams.perTokenScaleGmAddr != nullptr) {
            AscendC::GlobalTensor<float> x1Scale;
            AscendC::GlobalTensor<float> x2Scale;
            x1Scale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.epilogueParams.perTokenScaleGmAddr));
            x2Scale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.epilogueParams.scaleGmAddr));
            float dequantScale = x1Scale.GetValue(0) * x2Scale.GetValue(0);
            uint32_t scaleBits = Blaze::Gemm::Float32ToBits(dequantScale);
            scaleScalar_ = static_cast<uint64_t>(scaleBits & DEQ_SCALE_MUL_MASK);
        } else if constexpr (AscendC::IsSameType<X2ScaleType, uint64_t>::value ||
                             AscendC::IsSameType<X2ScaleType, int64_t>::value) {
            AscendC::GlobalTensor<uint64_t> x2Scale;
            x2Scale.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(params.epilogueParams.scaleGmAddr));
            scaleScalar_ = x2Scale.GetValue(0);
        } else if constexpr (AscendC::IsSameType<X2ScaleType, bfloat16_t>::value) {
            AscendC::GlobalTensor<uint16_t> x2Scale;
            x2Scale.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(params.epilogueParams.scaleGmAddr));
            uint32_t scaleBits = static_cast<uint32_t>(x2Scale.GetValue(0)) << BF16_SHIFT;
            scaleScalar_ = static_cast<uint64_t>(scaleBits & DEQ_SCALE_MUL_MASK);
        } else {
            AscendC::GlobalTensor<float> x2Scale;
            x2Scale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.epilogueParams.scaleGmAddr));
            scaleScalar_ = static_cast<uint64_t>(Blaze::Gemm::Float32ToBits(x2Scale.GetValue(0)) & DEQ_SCALE_MUL_MASK);
        }
    }

    template <typename SingleCoreShape>
    __aicore__ inline auto MakeWorkspaceTensor(const SingleCoreShape& singleCoreShape, int64_t offsetWorkspace)
    {
        auto workspaceStrideColumn = Blaze::Gemm::CeilAlign(
            AscendC::Te::Get<MNK_N>(singleCoreShape),
            static_cast<int64_t>(AscendC::GetVecLen() / sizeof(WorkspaceType)));
        auto workspaceShape = AscendC::Te::MakeShape(
            AscendC::Te::MakeShape(AscendC::Te::_1{}, AscendC::Te::Get<MNK_M>(singleCoreShape)),
            AscendC::Te::MakeShape(AscendC::Te::_1{}, workspaceStrideColumn));
        auto workspaceStride = AscendC::Te::MakeStride(
            AscendC::Te::MakeStride(AscendC::Te::_0{}, workspaceStrideColumn),
            AscendC::Te::MakeStride(AscendC::Te::_0{}, AscendC::Te::_1{}));
        auto layoutWorkspace = AscendC::Te::MakePatternLayout<AscendC::Te::NDExtLayoutPtn,
                                                              AscendC::Te::LayoutTraitDefault<WorkspaceType>>(
            workspaceShape, workspaceStride);
        return AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(workspaceGmAddr_ + offsetWorkspace), layoutWorkspace);
    }

private:
    static constexpr uint8_t AIC_ONLY_SYNC_MODE = 0;
    static constexpr uint8_t AIC_SYNC_AIV_MODE = 4;
    static constexpr uint16_t AIC_ONLY_SYNC_FLAG = 7;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 8;
    static constexpr uint16_t FLAG_ID_MAX = 16;
    static constexpr uint16_t BLOCK_BASE_M = 256;
    static constexpr uint16_t BLOCK_BASE_N = 256;
    static constexpr uint32_t BF16_SHIFT = 16U;
    static constexpr uint32_t DEQ_SCALE_MUL_MASK = 0xFFFFE000U;

    __gm__ AType* aGmAddr_{nullptr};
    __gm__ BType* bGmAddr_{nullptr};
    __gm__ CType* cGmAddr_{nullptr};
    __gm__ BiasType* biasGmAddr_{nullptr};
    __gm__ WorkspaceType* workspaceGmAddr_{nullptr};
    uint64_t scaleScalar_{0UL};

    int64_t m_{1};
    int64_t n_{1};
    int64_t k_{1};
    int64_t mL1_{0};
    int64_t nL1_{0};
    int64_t mBlockNums_{0};
    int64_t nBlockNums_{0};
    int64_t skBlockNums_{0};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
