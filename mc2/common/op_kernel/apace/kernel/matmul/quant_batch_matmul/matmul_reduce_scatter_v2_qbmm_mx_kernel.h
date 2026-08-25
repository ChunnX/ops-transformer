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
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "include/tensor_api/tensor.h"
#include "../../../block/mmad/qmm_mx_block_mmad_fragment.h"
#include "../../../basic/fragment_tensor/fragment_tensor_api.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

// The scheduler sees rankDim * sliceM as one logical M. FragmentTensor maps
// that logical M to one non-contiguous GM fragment per rank, so a scheduler
// block can cross rank boundaries without being split into several MMAD calls.
template <class ProblemShape, class BlockMmad, class BlockScheduler>
class MatmulReduceScatterV2QbmmMxKernel {
public:
    static constexpr bool weightNz = BlockMmad::weightNz;
    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

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
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<
        LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<CType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        transA,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        transB,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    static constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;
    static constexpr uint64_t MXFP_DATA_NUM_PER_BYTE = 2UL;
    using FragTensorA = Apace::Basic::FragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>;
    using FragScaleA = Apace::Basic::FragmentTensor<
        2, MAX_FRAG, MakeLayoutScaleA, AscendC::fp8_e8m0_t>;
    using FragTensorC = Apace::Basic::FragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>;

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
        if (sp.rankDim == 0 || sp.rankDim > MAX_FRAG || sp.sliceM == 0 || sp.perRankM == 0 ||
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
        BuildSliceFragmentTensors(params, sliceProblemShape);
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
        if (sp.rankDim == 0 || sp.rankDim > MAX_FRAG || sp.perRankM == 0 ||
            commPolicy.GetSliceCount() == 0 ||
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
            BuildSliceFragmentTensors(params, sliceProblemShape);
            Process(params, sliceProblemShape, sliceSchedulerParams, scheduler);
            AscendC::SyncAll();
            commPolicy.IssueSlice(sliceStart, sliceM);
            coveredM += sliceM;
        }
    }

    __aicore__ inline Apace::Basic::FragmentParam<2> MakeFragParam(
        uint64_t fragmentM, uint32_t fragmentCount, uint64_t secondAxis) const
    {
        Apace::Basic::FragmentParam<2> param{};
        param.assembleAxis = 0;
        param.assembledShape[0] = fragmentM * fragmentCount;
        param.assembledShape[1] = secondAxis;
        param.fragmentSize = fragmentM;
        param.realFragmentSize = fragmentM;
        param.fragmentCnt = fragmentCount;
        return param;
    }

    __aicore__ inline void BuildSliceFragmentTensors(
        const Params &params, const ProblemShape &sliceProblemShape)
    {
        using AscendC::Te::Get;
        const auto &sp = params.sliceParams;
        const uint64_t k = static_cast<uint64_t>(Get<MNK_K>(sliceProblemShape));
        const uint64_t n = static_cast<uint64_t>(Get<MNK_N>(sliceProblemShape));
        const uint64_t scaleK =
            CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        const uint64_t aRowBytes = IsFp4<AType>() ?
            CeilDiv(k, MXFP_DATA_NUM_PER_BYTE) : k * sizeof(AType);
        const uint64_t scaleARowBytes = scaleK * sizeof(AscendC::fp8_e8m0_t);
        const uint64_t cRowBytes = n * sizeof(CType);
        const uint64_t workspaceSliceRow = sp.sliceStart * sp.rankDim;

        for (uint32_t rankIdx = 0; rankIdx < sp.rankDim; ++rankIdx) {
            const uint64_t inputRow =
                static_cast<uint64_t>(rankIdx) * sp.perRankM + sp.sliceStart;
            const uint64_t workspaceRow =
                workspaceSliceRow + static_cast<uint64_t>(rankIdx) * sp.sliceM;
            sliceAddrListA_[rankIdx] = params.mmParams.aGmAddr + inputRow * aRowBytes;
            sliceAddrListScaleA_[rankIdx] =
                params.mmParams.scaleAGmAddr + inputRow * scaleARowBytes;
            sliceAddrListC_[rankIdx] = params.mmParams.cGmAddr + workspaceRow * cRowBytes;
        }

        sliceFragA_ = Apace::Basic::MakeFragmentTensor<
            2, MAX_FRAG, MakeLayoutA, AType>(
            MakeFragParam(sp.sliceM, sp.rankDim, k), sliceAddrListA_);
        sliceFragScaleA_ = Apace::Basic::MakeFragmentTensor<
            2, MAX_FRAG, MakeLayoutScaleA, AscendC::fp8_e8m0_t>(
            MakeFragParam(sp.sliceM, sp.rankDim, scaleK), sliceAddrListScaleA_);
        sliceFragC_ = Apace::Basic::MakeFragmentTensor<
            2, MAX_FRAG, MakeLayoutC, CType>(
            MakeFragParam(sp.sliceM * sp.rankDim, 1, n), sliceAddrListC_);
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
        const int64_t n = Get<MNK_N>(sliceProblemShape);
        const int64_t k = Get<MNK_K>(sliceProblemShape);
        const int64_t scaleK =
            CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

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
            auto blockA = sliceFragA_.Slice(
                AscendC::Te::MakeCoord(mPos, 0L), AscendC::Te::MakeShape(currentM, k));
            auto blockScaleA = sliceFragScaleA_.Slice(
                AscendC::Te::MakeCoord(mPos, 0L), AscendC::Te::MakeShape(currentM, scaleK));
            auto blockC = sliceFragC_.Slice(
                AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(currentM, currentN));

            // FragmentTensor presents all rank-local input slices as one
            // logical M. The fragment MMAD gathers A/ScaleA into L1 and
            // scatters C back to the slice-major workspace fragments.
            mmOp_(blockA, blockB, blockScaleA, blockScaleB, blockBias,
                  sliceAddrListC_, sp.perRankM, sp.sliceM, 1, 0,
                  sp.rankDim, n, singleShape, mPos, nPos, 0, blockC);
        }
    }

private:
    BlockMmad mmOp_;
    GM_ADDR sliceAddrListA_[MAX_FRAG]{};
    GM_ADDR sliceAddrListScaleA_[MAX_FRAG]{};
    GM_ADDR sliceAddrListC_[MAX_FRAG]{};
    FragTensorA sliceFragA_{};
    FragScaleA sliceFragScaleA_{};
    FragTensorC sliceFragC_{};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
