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
 * \file kernel_qbmm_mix_without_batch.h
 * \brief MIX template kernel without batch: AIC cube compute + AIV dequant epilogue (Tensor API).
 */
#pragma once

#include "kernel_universal.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define QBMM_MIX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_MIX_WITHOUT_BATCH_KERNEL_TEM_PARAMS                       \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,            \
        AscendC::Std::enable_if_t<                                     \
            AscendC::Std::is_same_v<KernelMmadWithScaleMixWithoutBatch, typename BlockMmad::DispatchPolicy::ScheduleType>>

QBMM_MIX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<QBMM_MIX_WITHOUT_BATCH_KERNEL_TEM_PARAMS> {
public:
    __aicore__ inline GemmUniversal()
    {}
    __aicore__ inline ~GemmUniversal()
    {}

    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using L0CType = typename BlockMmad::L0CType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using EpilogueParams = typename BlockEpilogue::Params;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    struct QBMMTiling {
        uint32_t bMustHitL2 = 1U;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmParams;
        BlockSchedulerParams schParams;
        EpilogueParams epilogueParams;
        QBMMTiling qbmmParams;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        BlockScheduler bs(params.problemShape, params.schParams);
        if ASCEND_IS_AIC {
            mmOp_.Init(params.mmParams);
        }
        if ASCEND_IS_AIV {
            epilogueOp_.Init(params.epilogueParams);
        }
        Run(params, bs);
    }

private:
    template <typename TensorB>
    __aicore__ inline void SetBL2Cache(const ProblemShape& problemShape, uint64_t currentBasicBlockM,
                                      uint64_t currentBasicBlockN, uint32_t bMustHitL2, TensorB& gmB)
    {
        if ASCEND_IS_AIC {
            // 0x7f: 128-element alignment for 128-byte B matrix GM streaming
            constexpr uint64_t cacheLineAlignMask = 0x7fUL;
            const bool isCurrentNAligned = TRANS_B || (currentBasicBlockN & cacheLineAlignMask) == 0UL;
            const bool disableWeightL2 = bMustHitL2 == 0U &&
                                         currentBasicBlockM >= AscendC::Te::Get<MNK_M>(problemShape) &&
                                         isCurrentNAligned;
            gmB.SetL2CacheHint(disableWeightL2 ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                                AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        }
    }

    // Process one block on AIC(cube) and AIV(dequant), keeping Run compact.
    // hasBlock is only used by AIC WaitForVector; AIV does not read it.
    template <class GmTensorA, class GmTensorB>
    __aicore__ inline void ProcessOneBlock(
        const GmTensorA& gmA, const GmTensorB& gmB, const BlockShape& singleShape, int64_t mPos, int64_t nPos,
        int64_t curM, int64_t curN, int64_t k, int64_t n, int64_t l0cUbBaseOffset, bool hasBlock)
    {
        constexpr int64_t kPos = 0;
        if ASCEND_IS_AIC {
            if (hasBlock) {
                WaitForVector();
            }
            auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(curM, k));
            auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(k, curN));

            // DATA_BLOCK=32 matches BlockEpilogueDequant::DATA_BLOCK.
            constexpr int64_t l0cAlign = BLOCK_BYTE_SIZE / sizeof(L0CType);
            const int64_t curNAligned = Blaze::Gemm::CeilAlign(curN, l0cAlign);
            const int64_t curMAligned = Blaze::Gemm::CeilAlign(curM, static_cast<int64_t>(2));
            auto layoutUbC = AscendC::Te::MakeFrameLayout<AscendC::Te::NDLayoutPtn>(curMAligned, curNAligned);
            auto ubC = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, L0CType>(l0cUbBaseOffset * sizeof(L0CType)),
                layoutUbC);
            mmOp_(gmBlockA, gmBlockB, ubC, singleShape);
            NotifyVector();
        }
        if ASCEND_IS_AIV {
            WaitForCube();
            epilogueOp_(curM, curN, nPos, mPos, nPos, mPos * n + nPos, l0cUbBaseOffset);
            NotifyCube();
        }
    }

    __aicore__ inline void Run(const Params& params, BlockScheduler& bs)
    {
        const int64_t m = AscendC::Te::Get<MNK_M>(params.problemShape);
        const int64_t n = AscendC::Te::Get<MNK_N>(params.problemShape);
        const int64_t k = AscendC::Te::Get<MNK_K>(params.problemShape);

        auto layoutA = MakeLayoutA{}(m, k);
        auto layoutB = MakeLayoutB{}(k, n);
        auto gmA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ AType*>(params.mmParams.aGmAddr)),
            layoutA);
        auto gmB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ BType*>(params.mmParams.bGmAddr)),
            layoutB);

        if ((bs.GetEndBlockIdx() + 1) * params.schParams.mTailTile * params.schParams.nTailTile <=
            AscendC::GetBlockNum()) {
            bs.UpdateTailTile(params.schParams.mTailTile, params.schParams.nTailTile);
        }

        BlockCoord blockCoord;
        int64_t mPos = 0;
        int64_t nPos = 0;
        bool hasBlock = false;
        while (bs.GetTileIdx(blockCoord)) {
            BlockShape singleShape =
                bs.template GetBlockShape<QuantMode::DEFAULT, QuantMode::DEFAULT, WEIGHT_NZ>(blockCoord);
            if (AscendC::Te::Get<IDX_M_TILEIDX>(singleShape) <= 0 ||
                AscendC::Te::Get<IDX_N_TILEIDX>(singleShape) <= 0) {
                break;
            }
            bs.GetTileCoord(blockCoord, mPos, nPos);
            const int64_t curM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
            const int64_t curN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
            const int64_t l0cUbBaseOffset = 0;
            SetBL2Cache(params.problemShape, curM, curN, params.qbmmParams.bMustHitL2, gmB);
            ProcessOneBlock(gmA, gmB, singleShape, mPos, nPos, curM, curN, k, n, l0cUbBaseOffset, hasBlock);
            hasBlock = true;
        }
        if ASCEND_IS_AIC {
            if (hasBlock) {
                WaitForVector();
            }
        }
    }

    BlockMmad mmOp_;
    BlockEpilogue epilogueOp_;

    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr int64_t C0_SIZE = AscendC::Te::C0_ELEMENT<AType>;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
