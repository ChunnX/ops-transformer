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
 * \file matmul_reduce_scatter_v2_qbmm_mx_kernel.h
 * \brief APACE MXFP8/MXFP4 QBMM kernel for MatmulReduceScatterV2.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "include/tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

// The scheduler sees rankDim * sliceM as one logical M. Each scheduler block
// is explicitly split at a rank boundary and then passed to the same standard
// MX BlockMmad used by AllToAllMatmul. This keeps all ranks in one scheduler
// without requiring FragmentTensor or a fragment-MMAD dependency.
template <class ProblemShape, class BlockMmad, class BlockScheduler>
class MatmulReduceScatterV2QbmmMxKernel {
public:
    static constexpr bool weightNz = BlockMmad::WEIGHT_NZ;
    static constexpr bool transA = BlockMmad::TRANS_A;
    static constexpr bool transB = BlockMmad::TRANS_B;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;

    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t CACHE_LINE_ALIGN_MASK = IsFp4<AType>() ? 0xff : 0x7f;
    static constexpr int32_t SCALE_C0 = 2;

    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        transA,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        transB,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    struct QBMMTiling {
        enum BiasMode : uint32_t {
            BIAS_DISABLED = 0,
            BIAS_ENABLED = 1
        };
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0C;
        uint32_t isBias;
    };

    struct SliceParams {
        uint32_t rankDim;
        uint64_t perRankM;
        uint64_t sliceStart;
        uint64_t sliceM;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmParams;
        L1Params l1Params;
        BlockSchedulerParams schedulerParams;
        QBMMTiling qbmmParams;
        SliceParams sliceParams;
    };

    __aicore__ inline void Run(const Params &params)
    {
        if ASCEND_IS_AIV {
            return;
        }

        const auto &sp = params.sliceParams;
        const uint64_t fullLogicalM = sp.perRankM * sp.rankDim;
        if (sp.rankDim == 0 || sp.sliceM == 0 || sp.perRankM == 0 ||
            sp.sliceStart + sp.sliceM > sp.perRankM ||
            static_cast<uint64_t>(AscendC::Te::Get<MNK_M>(params.problemShape)) != fullLogicalM) {
            return;
        }

        const bool isBias = params.qbmmParams.isBias == QBMMTiling::BIAS_ENABLED;
        ProblemShape sliceProblemShape{static_cast<int64_t>(sp.sliceM * sp.rankDim),
                                       AscendC::Te::Get<MNK_N>(params.problemShape),
                                       AscendC::Te::Get<MNK_K>(params.problemShape),
                                       AscendC::Te::Get<MNK_B>(params.problemShape)};
        BlockSchedulerParams sliceSchedulerParams;
        SetupSliceSchedulerParams(params.schedulerParams, sliceSchedulerParams);
        BlockScheduler scheduler(sliceProblemShape, sliceSchedulerParams);
        BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN,
                               params.qbmmParams.baseK, 0};
        mmOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias,
                   params.qbmmParams.dbL0C > 1, 1);
        Process(params, sliceProblemShape, sliceSchedulerParams, scheduler);
    }

    __aicore__ inline void operator()(const Params &params)
    {
        Run(params);
    }

    // Compute one ordered communication-slice sequence through one QBMM-kernel
    // invocation and one complete-M QuantMatmul tiling. The communication
    // policy owns the physical partition representation; Matmul only consumes
    // {sliceStart, sliceM} and does not distinguish main and tail groups.
    // BlockMmad is initialized once; each communication slice still owns a
    // scheduler completion boundary.
    // CommPolicy::IssueSlice() is called only after every AIC has completed
    // the current slice, preserving the compute(i) / communication(i-1)
    // overlap required by MatmulReduceScatterV2.
    template <typename CommPolicy>
    __aicore__ inline void RunSlices(const Params &params, CommPolicy &commPolicy)
    {
        if ASCEND_IS_AIV {
            return;
        }
        const auto &sp = params.sliceParams;
        const uint64_t fullLogicalM = sp.perRankM * sp.rankDim;
        if (sp.rankDim == 0 || sp.perRankM == 0 || commPolicy.GetSliceCount() == 0 ||
            static_cast<uint64_t>(AscendC::Te::Get<MNK_M>(params.problemShape)) != fullLogicalM) {
            return;
        }

        const bool isBias = params.qbmmParams.isBias == QBMMTiling::BIAS_ENABLED;
        BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN,
                               params.qbmmParams.baseK, 0};
        mmOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias,
                   params.qbmmParams.dbL0C > 1, 1);
        RunSliceSequence(params, commPolicy);
    }

private:
    __aicore__ inline void SetupSliceSchedulerParams(
        const BlockSchedulerParams &fullSchedulerParams,
        BlockSchedulerParams &sliceSchedulerParams) const
    {
        sliceSchedulerParams = fullSchedulerParams;
        // The serialized scheduler parameters were generated for fullLogicalM.
        // M-tail merging/splitting cannot be reused for a shorter communication
        // slice. Keep baseM/baseN/baseK and the N-tail policy, but use the
        // conservative one-block M-tail policy for every slice scheduler.
        sliceSchedulerParams.mTailTile = 1;
        sliceSchedulerParams.mBaseTailSplitCnt = 1;
        sliceSchedulerParams.mTailMain = 0;
    }

    template <typename CommPolicy>
    __aicore__ inline void RunSliceSequence(const Params &baseParams, CommPolicy &commPolicy)
    {
        const auto &baseSlice = baseParams.sliceParams;
        if (baseSlice.rankDim == 0 || baseSlice.perRankM == 0) {
            return;
        }

        const uint32_t sliceCount = commPolicy.GetSliceCount();
        uint64_t coveredM = 0;
        for (uint32_t sliceIdx = 0; sliceIdx < sliceCount; ++sliceIdx) {
            uint64_t sliceStart = 0;
            uint64_t sliceM = 0;
            if (!commPolicy.GetSliceInfo(sliceIdx, sliceStart, sliceM) ||
                sliceStart != coveredM || sliceStart + sliceM > baseSlice.perRankM) {
                return;
            }

            Params params = baseParams;
            params.sliceParams.sliceStart = sliceStart;
            params.sliceParams.sliceM = sliceM;

            ProblemShape sliceProblemShape{static_cast<int64_t>(sliceM * baseSlice.rankDim),
                                           AscendC::Te::Get<MNK_N>(params.problemShape),
                                           AscendC::Te::Get<MNK_K>(params.problemShape),
                                           AscendC::Te::Get<MNK_B>(params.problemShape)};
            BlockSchedulerParams sliceSchedulerParams;
            SetupSliceSchedulerParams(params.schedulerParams, sliceSchedulerParams);

            BlockScheduler scheduler(sliceProblemShape, sliceSchedulerParams);
            Process(params, sliceProblemShape, sliceSchedulerParams, scheduler);
            AscendC::SyncAll();
            commPolicy.IssueSlice(sliceStart, sliceM);
            coveredM += sliceM;
        }
    }

    template <typename TensorScaleB>
    __aicore__ inline void SetScaleL2Cache(
        const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB &gmScaleB)
    {
        if (AscendC::Te::Get<MNK_B>(problemShape) != 1) {
            return;
        }
        if constexpr (transB) {
            const int64_t scaleKRowBytes =
                CeilDiv(AscendC::Te::Get<MNK_K>(problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                MXFP_MULTI_BASE_SIZE;
            const int64_t scaleKL1RowBytes =
                CeilDiv(scaleKL1, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
            const bool aligned = (scaleKRowBytes & CACHE_LINE_ALIGN_MASK) == 0 &&
                                 (scaleKL1RowBytes & CACHE_LINE_ALIGN_MASK) == 0;
            gmScaleB.SetL2CacheHint(
                aligned ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE : AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        } else {
            const int64_t scaleNStrideBytes = AscendC::Te::Get<MNK_N>(problemShape) * MXFP_MULTI_BASE_SIZE;
            const int64_t scaleBaseNStrideBytes = baseN * MXFP_MULTI_BASE_SIZE;
            const bool aligned = (scaleNStrideBytes & CACHE_LINE_ALIGN_MASK) == 0 &&
                                 (scaleBaseNStrideBytes & CACHE_LINE_ALIGN_MASK) == 0;
            gmScaleB.SetL2CacheHint(
                aligned ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE : AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        }
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2Cache(
        const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN,
        uint64_t scaleKL1, TensorB &gmB, TensorScaleB &gmScaleB)
    {
        if (curBaseM < static_cast<uint64_t>(AscendC::Te::Get<MNK_M>(problemShape))) {
            return;
        }

        SetScaleL2Cache(problemShape, baseN, scaleKL1, gmScaleB);
        if constexpr (weightNz) {
            gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
        } else if constexpr (transB) {
            const bool aligned = (AscendC::Te::Get<MNK_K>(problemShape) & CACHE_LINE_ALIGN_MASK) == 0;
            gmB.SetL2CacheHint(
                aligned ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE : AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        } else {
            const bool aligned = (AscendC::Te::Get<MNK_N>(problemShape) & CACHE_LINE_ALIGN_MASK) == 0 &&
                                 (baseN & CACHE_LINE_ALIGN_MASK) == 0;
            gmB.SetL2CacheHint(
                aligned ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE : AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        }
    }

    __aicore__ inline void Process(const Params &params, const ProblemShape &sliceProblemShape,
                                   const BlockSchedulerParams &sliceSchedulerParams,
                                   BlockScheduler &scheduler)
    {
        using AscendC::Te::Get;
        const auto &sp = params.sliceParams;
        const int64_t logicalM = Get<MNK_M>(sliceProblemShape);
        const int64_t n = Get<MNK_N>(sliceProblemShape);
        const int64_t k = Get<MNK_K>(sliceProblemShape);
        const int64_t totalInputM = static_cast<int64_t>(sp.perRankM * sp.rankDim);
        const int64_t scaleK =
            CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

        auto gmA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ AType *>(params.mmParams.aGmAddr)),
            MakeLayoutA{}(totalInputM, k));
        auto gmScaleA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(params.mmParams.scaleAGmAddr)),
            MakeLayoutScaleA{}(totalInputM, scaleK));
        auto gmB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ BType *>(params.mmParams.bGmAddr)),
            MakeLayoutB{}(k, n));
        auto gmScaleB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(params.mmParams.scaleBGmAddr)),
            MakeLayoutScaleB{}(scaleK, n));
        auto gmBias = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ BiasType *>(params.mmParams.biasGmAddr)),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1L, n));
        auto gmC = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                reinterpret_cast<__gm__ CType *>(params.mmParams.cGmAddr)),
            MakeLayoutC{}(totalInputM, n));

        const auto &mTailTile = sliceSchedulerParams.mTailTile;
        const auto &nTailTile = sliceSchedulerParams.nTailTile;
        if ((scheduler.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= AscendC::GetBlockNum()) {
            scheduler.UpdateTailTile(mTailTile, nTailTile);
        }
        SetL2Cache(sliceProblemShape, params.qbmmParams.baseM, params.qbmmParams.baseN,
                   params.l1Params.scaleKL1, gmB, gmScaleB);

        BlockCoord blockIdx;
        int64_t mPos = 0;
        int64_t nPos = 0;
        while (scheduler.GetTileIdx(blockIdx)) {
            BlockShape singleShape = scheduler.template GetBlockShape<
                QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
            const int64_t currentM = Get<IDX_M_TILEIDX>(singleShape);
            const int64_t currentN = Get<IDX_N_TILEIDX>(singleShape);
            if (currentM <= 0 || currentN <= 0) {
                return;
            }

            scheduler.GetTileCoord(blockIdx, mPos, nPos);
            auto blockB = gmB.Slice(
                AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(k, currentN));
            auto blockScaleB = gmScaleB.Slice(
                AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(scaleK, currentN));
            auto blockBias = gmBias.Slice(
                AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(1L, currentN));

            int64_t logicalMPos = mPos;
            int64_t remainingM = currentM;
            while (remainingM > 0) {
                const uint64_t rankIdx = static_cast<uint64_t>(logicalMPos) / sp.sliceM;
                const uint64_t rankSliceOffset = static_cast<uint64_t>(logicalMPos) % sp.sliceM;
                const int64_t currentRankM = static_cast<int64_t>(sp.sliceM - rankSliceOffset);
                const int64_t subM = remainingM < currentRankM ? remainingM : currentRankM;
                const int64_t inputMPos = static_cast<int64_t>(
                    rankIdx * sp.perRankM + sp.sliceStart + rankSliceOffset);

                auto blockA = gmA.Slice(
                    AscendC::Te::MakeCoord(inputMPos, 0L), AscendC::Te::MakeShape(subM, k));
                auto blockScaleA = gmScaleA.Slice(
                    AscendC::Te::MakeCoord(inputMPos, 0L), AscendC::Te::MakeShape(subM, scaleK));
                const int64_t workspaceMPos = static_cast<int64_t>(
                    sp.sliceStart * sp.rankDim + static_cast<uint64_t>(logicalMPos));
                auto blockC = gmC.Slice(
                    AscendC::Te::MakeCoord(workspaceMPos, nPos), AscendC::Te::MakeShape(subM, currentN));
                BlockShape subShape{subM, currentN, k, 0};
                mmOp_(blockA, blockB, blockScaleA, blockScaleB, blockBias, blockC, subShape, 0);

                logicalMPos += subM;
                remainingM -= subM;
            }
        }
    }

private:
    BlockMmad mmOp_;
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
