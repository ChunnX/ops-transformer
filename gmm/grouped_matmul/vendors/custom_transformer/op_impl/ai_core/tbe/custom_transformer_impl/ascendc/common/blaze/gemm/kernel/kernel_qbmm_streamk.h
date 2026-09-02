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
 * \file kernel_qbmm_streamk.h
 * \brief GemmUniversal specialization for QBMM MX StreamK.
 */

#pragma once

#include "kernel_universal.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/epilogue/block/block_epilogue_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<KernelQbmmMultiBlockStreamK, typename BlockMmad_::DispatchPolicy::ScheduleType> &&
        AscendC::Std::is_same_v<KernelMultiBlockStreamK, typename BlockEpilogue_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    struct QBMMStreamKParams {
        uint32_t scaleKL1;
        uint32_t dbL0C;
        uint32_t bMustHitL2 = 1U;
    };

    struct Params {
        ProblemShape_ problemShape;
        typename BlockMmad_::Params mmadParams;
        typename BlockEpilogue_::Params epilogueParams;
        typename BlockScheduler_::Params schParams;
        QBMMStreamKParams qbmmParams;
    };

    __aicore__ inline void operator()(Params const& params)
    {
        Init(params);
        if (!IsValidProblem()) {
            return;
        }

        BlockScheduler_ bs(problemShape_, params.schParams);
        if ASCEND_IS_AIC {
            ProcessAic(params, bs);
        }

        if ASCEND_IS_AIV {
            ProcessAiv(params, bs);
        }
    }

private:
    using ProblemShape = ProblemShape_;
    using BlockMmadOp = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;

    using BlockMmadParams = typename BlockMmadOp::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    using AType = typename BlockMmadOp::AType;
    using BType = typename BlockMmadOp::BType;
    using CType = typename BlockMmadOp::CType;
    using BiasType = typename BlockMmadOp::BiasType;
    using LayoutA = typename BlockMmadOp::LayoutA;
    using LayoutB = typename BlockMmadOp::LayoutB;
    using LayoutC = typename BlockMmadOp::LayoutC;

    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr int32_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<
        LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<CType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        TRANS_B, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    __aicore__ inline void Init(Params const& params)
    {
        problemShape_ = params.problemShape;
        usedCoreNums_ = params.schParams.usedCoreNum;
        blockMmadParams_ = params.mmadParams;

        int64_t m = AscendC::Te::Get<MNK_M>(problemShape_);
        int64_t n = AscendC::Te::Get<MNK_N>(problemShape_);
        int64_t k = AscendC::Te::Get<MNK_K>(problemShape_);

        mBlockNums_ = Blaze::Gemm::CeilDiv(m, params.schParams.baseM);
        nBlockNums_ = Blaze::Gemm::CeilDiv(n, params.schParams.baseN);
        skBlockNums_ = Blaze::Gemm::CeilDiv(k, params.schParams.singleCoreK);

        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams_.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams_.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams_.cGmAddr);
        scaleAGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(blockMmadParams_.scaleAGmAddr);
        scaleBGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(blockMmadParams_.scaleBGmAddr);
        workspaceGmAddr_ = reinterpret_cast<__gm__ float*>(params.epilogueParams.workspaceGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams_.biasGmAddr);
    }

    __aicore__ inline bool IsValidProblem() const
    {
        return usedCoreNums_ > 0 && AscendC::Te::Get<MNK_B>(problemShape_) == 1;
    }

    __aicore__ inline int64_t GetActualBlockIdx(
        BlockScheduler& bs, int64_t blockIdx, int64_t blockNums, int64_t tailSKTotalBlockNums) const
    {
        if (bs.CheckIsSkScene(0)) {
            return blockIdx;
        }
        bool preloadSK = blockIdx % usedCoreNums_ < tailSKTotalBlockNums &&
                         CeilDiv(blockIdx + 1, usedCoreNums_) == CeilDiv(blockNums, usedCoreNums_) - 1;
        if (preloadSK) {
            return blockIdx + usedCoreNums_;
        }
        bool moveBackSK = blockIdx % usedCoreNums_ < tailSKTotalBlockNums &&
                          CeilDiv(blockIdx + 1, usedCoreNums_) == CeilDiv(blockNums, usedCoreNums_);
        return moveBackSK ? blockIdx - usedCoreNums_ : blockIdx;
    }

    __aicore__ inline void SignalAicFinish() const
    {
        AscendC::CrossCoreSetFlag<AIC_ONLY_SYNC_MODE, PIPE_FIX>(AIC_ONLY_SYNC_FLAG);
        AscendC::CrossCoreWaitFlag(AIC_ONLY_SYNC_FLAG);
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
    }

    template <typename BlockShape>
    __aicore__ inline auto MakeWorkspaceTensor(BlockShape const& singleCoreShape, int64_t offsetWorkspace) const
    {
        auto workspaceStrideColumn0 =
            BlockEpilogue::DispatchPolicy::FIXP_OPTI == MatMulL0C2Out::ND_FIXPIPE_1_2 ?
                CeilAlign(static_cast<uint64_t>(AscendC::Te::Get<MNK_N>(singleCoreShape)),
                          static_cast<uint64_t>(BLOCK_BYTE_SIZE)) :
                AscendC::Te::Get<MNK_N>(singleCoreShape);
        auto workspaceShape = AscendC::Te::MakeShape(
            AscendC::Te::MakeShape(AscendC::Te::_1{}, AscendC::Te::Get<MNK_M>(singleCoreShape)),
            AscendC::Te::MakeShape(AscendC::Te::_1{}, AscendC::Te::Get<MNK_N>(singleCoreShape)));
        auto workspaceStride = AscendC::Te::MakeStride(
            AscendC::Te::MakeStride(AscendC::Te::_0{}, workspaceStrideColumn0),
            AscendC::Te::MakeStride(AscendC::Te::_0{}, AscendC::Te::_1{}));
        auto layoutWorkspace = AscendC::Te::MakePatternLayout<
            AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>(workspaceShape, workspaceStride);
        return AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(workspaceGmAddr_ + offsetWorkspace), layoutWorkspace);
    }

    template <typename TensorB, typename BlockShape>
    __aicore__ inline void SetBL2Cache(Params const& params, BlockShape const& currentBasicBlock, TensorB& gmB) const
    {
        // 0xff: 256 cache line alignment for FP4 B matrix GM streaming
        // 0x7f: 128 cache line alignment for FP8 B matrix GM streaming
        constexpr uint64_t cacheLineAlignMask = IsFp4<BType>() ? 0xffUL : 0x7fUL;
        const bool isCurrentNAligned =
            TRANS_B || (AscendC::Te::Get<MNK_N>(currentBasicBlock) & cacheLineAlignMask) == 0UL;
        const bool disableWeightL2 = params.qbmmParams.bMustHitL2 == 0U &&
                                     AscendC::Te::Get<MNK_M>(currentBasicBlock) >=
                                         AscendC::Te::Get<MNK_M>(problemShape_) &&
                                     isCurrentNAligned;
        gmB.SetL2CacheHint(disableWeightL2 ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                            AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
    }

    template <typename TensorA, typename TensorScaleA, typename TensorB, typename TensorScaleB,
              typename TensorBias, typename TensorC>
    __aicore__ inline void ProcessAicBlock(
        Params const& params, BlockScheduler& bs, BlockMmadOp& blockMmadOp, TensorA& gmA, TensorScaleA& gmScaleA,
        TensorB& gmB, TensorScaleB& gmScaleB, TensorBias& gmBias, TensorC& gmC, int64_t tmpBlockIdx,
        int64_t mL1, int64_t nL1, ProblemShape const& l0BlockShape, uint64_t scaleKL1)
    {
        auto singleCoreShape = bs.GetBlockShape(tmpBlockIdx);
        auto singleCoreCoord = bs.GetBlockCoord(tmpBlockIdx);
        SetBL2Cache(params, singleCoreShape, gmB);
        bool isSkScene = bs.CheckIsSkScene(tmpBlockIdx);
        int64_t kSingleCore = isSkScene ? params.schParams.singleCoreK : AscendC::Te::Get<MNK_K>(problemShape_);
        int64_t offsetWorkspace = (((tmpBlockIdx % usedCoreNums_) / skBlockNums_) * skBlockNums_ +
                                   AscendC::Te::Get<MNK_K>(singleCoreCoord)) * BLOCK_BASE_M * BLOCK_BASE_N;
        auto gmWorkSpace = MakeWorkspaceTensor(singleCoreShape, offsetWorkspace);
        auto scaleKL1LenSingleCore = CeilDiv(kSingleCore, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                                     MXFP_MULTI_BASE_SIZE;
        auto gmBlockA = gmA.Slice(
            AscendC::Te::MakeCoord(AscendC::Te::Get<MNK_M>(singleCoreCoord) * mL1,
                                   AscendC::Te::Get<MNK_K>(singleCoreCoord) * kSingleCore),
            AscendC::Te::MakeShape(AscendC::Te::Get<MNK_M>(singleCoreShape),
                                   AscendC::Te::Get<MNK_K>(singleCoreShape)));
        auto gmBlockScaleA = gmScaleA.Slice(
            AscendC::Te::MakeCoord(AscendC::Te::Get<MNK_M>(singleCoreCoord) * mL1,
                                   AscendC::Te::Get<MNK_K>(singleCoreCoord) * scaleKL1LenSingleCore),
            AscendC::Te::MakeShape(AscendC::Te::Get<MNK_M>(singleCoreShape), scaleKL1LenSingleCore));
        auto gmBlockB = gmB.Slice(
            AscendC::Te::MakeCoord(AscendC::Te::Get<MNK_K>(singleCoreCoord) * kSingleCore,
                                   AscendC::Te::Get<MNK_N>(singleCoreCoord) * nL1),
            AscendC::Te::MakeShape(AscendC::Te::Get<MNK_K>(singleCoreShape),
                                   AscendC::Te::Get<MNK_N>(singleCoreShape)));
        auto gmBlockScaleB = gmScaleB.Slice(
            AscendC::Te::MakeCoord(AscendC::Te::Get<MNK_K>(singleCoreCoord) * scaleKL1LenSingleCore,
                                   AscendC::Te::Get<MNK_N>(singleCoreCoord) * nL1),
            AscendC::Te::MakeShape(scaleKL1LenSingleCore, AscendC::Te::Get<MNK_N>(singleCoreShape)));
        auto gmBlockBias = gmBias.Slice(AscendC::Te::MakeCoord(0L, AscendC::Te::Get<MNK_N>(singleCoreCoord) * nL1),
                                        AscendC::Te::MakeShape(1L, AscendC::Te::Get<MNK_N>(singleCoreShape)));
        auto gmBlockC = gmC.Slice(
            AscendC::Te::MakeCoord(AscendC::Te::Get<MNK_M>(singleCoreCoord) * mL1,
                                   AscendC::Te::Get<MNK_N>(singleCoreCoord) * nL1),
            AscendC::Te::MakeShape(AscendC::Te::Get<MNK_M>(singleCoreShape),
                                   AscendC::Te::Get<MNK_N>(singleCoreShape)));
        bool isBiasForBlock = biasGmAddr_ != nullptr && AscendC::Te::Get<MNK_K>(singleCoreCoord) == 0;
        blockMmadOp.Init(
            singleCoreShape, l0BlockShape,
            {static_cast<uint64_t>(params.schParams.kL1), scaleKL1, DOUBLE_BUFFER_COUNT},
            isBiasForBlock, params.qbmmParams.dbL0C > 1);
        if (isSkScene) {
            blockMmadOp(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmWorkSpace, singleCoreShape);
        } else {
            blockMmadOp(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleCoreShape);
        }
    }

    __aicore__ inline void ProcessAicBlocks(Params const& params, BlockScheduler& bs, int64_t curBlockIdx)
    {
        int64_t m = AscendC::Te::Get<MNK_M>(problemShape_);
        int64_t n = AscendC::Te::Get<MNK_N>(problemShape_);
        int64_t k = AscendC::Te::Get<MNK_K>(problemShape_);

        int64_t mnBlockNums = mBlockNums_ * nBlockNums_;
        int64_t tailSKTotalBlockNums = static_cast<int64_t>((mnBlockNums % usedCoreNums_) * skBlockNums_);
        int64_t blockNums = bs.GetBlockNums();
        auto scaleKLen = CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_),
                                           MakeLayoutA{}(m, k));
        auto gmScaleA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleAGmAddr_),
                                                MakeLayoutScaleA{}(m, scaleKLen));
        auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_),
                                           MakeLayoutB{}(k, n));
        auto gmScaleB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBGmAddr_),
                                                MakeLayoutScaleB{}(scaleKLen, n));
        auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmAddr_),
                                              AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1L, n));
        auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_),
                                           MakeLayoutC{}(m, n));
        BlockMmadOp blockMmadOp;
        auto scaleKL1 = static_cast<uint64_t>(params.qbmmParams.scaleKL1);
        ProblemShape l0BlockShape = {params.schParams.baseM, params.schParams.baseN, params.schParams.baseK, 1};
        for (int64_t blockIdx = curBlockIdx; blockIdx < blockNums; blockIdx += usedCoreNums_) {
            int64_t tmpBlockIdx = GetActualBlockIdx(bs, blockIdx, blockNums, tailSKTotalBlockNums);
            ProcessAicBlock(params, bs, blockMmadOp, gmA, gmScaleA, gmB, gmScaleB, gmBias, gmC, tmpBlockIdx,
                            params.schParams.baseM, params.schParams.baseN, l0BlockShape, scaleKL1);
        }
    }

    __aicore__ inline void ProcessAic(Params const& params, BlockScheduler& bs)
    {
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        if (curBlockIdx < usedCoreNums_) {
            ProcessAicBlocks(params, bs, curBlockIdx);
        }
        SignalAicFinish();
    }

    __aicore__ inline void ProcessAiv(Params const& params, BlockScheduler& bs)
    {
        int64_t mnBlockNums = mBlockNums_ * nBlockNums_;
        uint64_t lastLoopTotalCnt = (mnBlockNums % usedCoreNums_) * skBlockNums_;
        uint64_t curBlockIdxInAiv = AscendC::GetBlockIdx();
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_MTE2>(AIC_SYNC_AIV_FLAG);
        if (curBlockIdxInAiv >= lastLoopTotalCnt * AscendC::GetTaskRation()) {
            return;
        }

        ProblemShape l1BlockShape = {params.schParams.baseM, params.schParams.baseN, params.schParams.kL1, 1};
        BlockEpilogue epilogueOp;
        epilogueOp.Init(
            params.epilogueParams, problemShape_, l1BlockShape,
            {mBlockNums_, nBlockNums_, skBlockNums_, 1}, usedCoreNums_,
            bs.CheckIsSkScene(0));
        epilogueOp();
    }

    static constexpr uint8_t AIC_ONLY_SYNC_MODE = 0;
    static constexpr uint8_t AIC_SYNC_AIV_MODE = 4;
    static constexpr uint16_t AIC_ONLY_SYNC_FLAG = 7;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 8;
    static constexpr uint16_t FLAG_ID_MAX = 16;
    static constexpr uint16_t BLOCK_BASE_M = 256;
    static constexpr uint16_t BLOCK_BASE_N = 256;

    ProblemShape problemShape_{};
    int64_t usedCoreNums_{0};
    int64_t mBlockNums_{0};
    int64_t nBlockNums_{0};
    int64_t skBlockNums_{0};
    BlockMmadParams blockMmadParams_{};

    __gm__ AType* aGmAddr_{nullptr};
    __gm__ BType* bGmAddr_{nullptr};
    __gm__ CType* cGmAddr_{nullptr};
    __gm__ BiasType* biasGmAddr_{nullptr};
    __gm__ fp8_e8m0_t* scaleAGmAddr_{nullptr};
    __gm__ fp8_e8m0_t* scaleBGmAddr_{nullptr};
    __gm__ float* workspaceGmAddr_{nullptr};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
