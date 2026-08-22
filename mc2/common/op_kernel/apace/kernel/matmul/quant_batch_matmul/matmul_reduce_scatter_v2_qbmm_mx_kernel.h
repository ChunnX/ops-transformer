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
 * \brief APACE MXFP8/MXFP4 fragment QBMM kernel for MatmulReduceScatterV2.
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
#include "apace/block/mmad/qmm_mx_block_mmad_fragment.h"
#include "apace/basic/fragment_tensor/fragment_tensor.h"
#include "apace/basic/fragment_tensor/fragment_tensor_api.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

// A ReduceScatter slice consists of rankDim discontiguous M fragments in the
// original [rankM, K] input.  FragmentTensor assembles those fragments as one
// logical M dimension, allowing one block scheduler to distribute all ranks'
// work across the AIC cores instead of launching one small QBMM per rank.
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
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t CACHE_LINE_ALIGN_MASK = IsFp4<AType>() ? 0xff : 0x7f;
    static constexpr int32_t SCALE_C0 = 2;
    static constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;

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

    using FragTensorA = Apace::Basic::FragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>;
    using FragScaleA = Apace::Basic::FragmentTensor<2, MAX_FRAG, MakeLayoutScaleA, AscendC::fp8_e8m0_t>;
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

    struct FragmentParams {
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
        FragmentParams fragmentParams;
    };

    __aicore__ inline void Run(const Params &params)
    {
        if ASCEND_IS_AIV {
            return;
        }

        const auto &fp = params.fragmentParams;
        const uint64_t totalLogicalM = fp.sliceM * fp.rankDim;
        if (fp.rankDim == 0 || fp.rankDim > MAX_FRAG || fp.sliceM == 0 ||
            static_cast<uint64_t>(AscendC::Te::Get<MNK_M>(params.problemShape)) != totalLogicalM) {
            return;
        }

        const bool isBias = params.qbmmParams.isBias == QBMMTiling::BIAS_ENABLED;
        BlockScheduler scheduler(params.problemShape, params.schedulerParams);
        BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN,
                               params.qbmmParams.baseK, 0};
        mmOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias,
                   params.qbmmParams.dbL0C > 1, 1);
        Process(params, scheduler);
    }

    __aicore__ inline void operator()(const Params &params)
    {
        Run(params);
    }

private:
    __aicore__ inline Apace::Basic::FragmentParam<2> MakeFragParam(
        uint64_t fragmentSize, uint32_t fragmentCnt, uint64_t shape1) const
    {
        Apace::Basic::FragmentParam<2> param{};
        param.assembledShape[0] = fragmentSize * fragmentCnt;
        param.assembledShape[1] = shape1;
        param.fragmentSize = fragmentSize;
        param.realFragmentSize = fragmentSize;
        param.assembleAxis = 0;
        param.fragmentCnt = fragmentCnt;
        return param;
    }

    __aicore__ inline void BuildFragmentTensors(const Params &params)
    {
        const auto &fp = params.fragmentParams;
        const uint64_t k = static_cast<uint64_t>(AscendC::Te::Get<MNK_K>(params.problemShape));
        const uint64_t n = static_cast<uint64_t>(AscendC::Te::Get<MNK_N>(params.problemShape));
        const uint64_t aRowBytes = IsFp4<AType>() ? CeilDiv(k, 2UL) : k * sizeof(AType);
        const uint64_t scaleK = CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

        for (uint32_t rankIdx = 0; rankIdx < fp.rankDim; ++rankIdx) {
            const uint64_t globalMOffset = static_cast<uint64_t>(rankIdx) * fp.perRankM + fp.sliceStart;
            aFragAddrs_[rankIdx] = params.mmParams.aGmAddr + globalMOffset * aRowBytes;
            scaleAFragAddrs_[rankIdx] = params.mmParams.scaleAGmAddr +
                globalMOffset * scaleK * sizeof(AscendC::fp8_e8m0_t);
            // Inputs stay rank-major. C is slice-major so all rank fragments
            // consumed by one ReduceScatter are adjacent in the workspace:
            // [slice][rank][sliceM][N]. Different slices remain disjoint while
            // their asynchronous communication is in flight.
            const uint64_t sliceWorkspaceM = fp.sliceStart * fp.rankDim +
                static_cast<uint64_t>(rankIdx) * fp.sliceM;
            cFragAddrs_[rankIdx] = params.mmParams.cGmAddr + sliceWorkspaceM * n * sizeof(CType);
        }

        const bool sameShape = fragmentReady_ && fragmentRankDim_ == fp.rankDim &&
            fragmentSliceM_ == fp.sliceM && fragmentK_ == k && fragmentN_ == n;
        if (sameShape) {
            // The head slices have the same fragment shape.  Match the APACE
            // AllGather implementation and only update their GM address lists
            // instead of rebuilding three FragmentTensor objects every round.
            gmA_.UpdateAddrList(aFragAddrs_);
            gmScaleA_.UpdateAddrList(scaleAFragAddrs_);
            gmC_.UpdateAddrList(cFragAddrs_);
            return;
        }

        gmA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
            MakeFragParam(fp.sliceM, fp.rankDim, k), aFragAddrs_);
        gmScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutScaleA, AscendC::fp8_e8m0_t>(
            MakeFragParam(fp.sliceM, fp.rankDim, scaleK), scaleAFragAddrs_);
        gmC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
            MakeFragParam(fp.sliceM, fp.rankDim, n), cFragAddrs_);
        fragmentReady_ = true;
        fragmentRankDim_ = fp.rankDim;
        fragmentSliceM_ = fp.sliceM;
        fragmentK_ = k;
        fragmentN_ = n;
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

    __aicore__ inline void Process(const Params &params, BlockScheduler &scheduler)
    {
        using AscendC::Te::Get;
        const int64_t n = Get<MNK_N>(params.problemShape);
        const int64_t k = Get<MNK_K>(params.problemShape);
        const int64_t scaleK = CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

        BuildFragmentTensors(params);

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

        auto &mTailTile = params.schedulerParams.mTailTile;
        auto &nTailTile = params.schedulerParams.nTailTile;
        if ((scheduler.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= AscendC::GetBlockNum()) {
            scheduler.UpdateTailTile(mTailTile, nTailTile);
        }
        SetL2Cache(params.problemShape, params.qbmmParams.baseM, params.qbmmParams.baseN,
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
            auto blockA = gmA_.Slice(
                AscendC::Te::MakeCoord(mPos, 0L), AscendC::Te::MakeShape(currentM, k));
            auto blockScaleA = gmScaleA_.Slice(
                AscendC::Te::MakeCoord(mPos, 0L), AscendC::Te::MakeShape(currentM, scaleK));
            auto blockB = gmB.Slice(
                AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(k, currentN));
            auto blockScaleB = gmScaleB.Slice(
                AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(scaleK, currentN));
            auto blockBias = gmBias.Slice(
                AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(1L, currentN));
            auto blockC = gmC_.Slice(
                AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(currentM, currentN));

            // The extra shape arguments are retained by the shared fragment
            // MMAD interface. FragmentTensor already contains the exact MMRS
            // source and destination addresses for this communication slice.
            mmOp_(blockA, blockB, blockScaleA, blockScaleB, blockBias, cFragAddrs_,
                  params.fragmentParams.perRankM, params.fragmentParams.sliceM, 1, 0,
                  params.fragmentParams.rankDim, n, singleShape, mPos, nPos, 0, blockC);
        }
    }

private:
    BlockMmad mmOp_;
    FragTensorA gmA_;
    FragScaleA gmScaleA_;
    FragTensorC gmC_;
    GM_ADDR aFragAddrs_[MAX_FRAG]{};
    GM_ADDR scaleAFragAddrs_[MAX_FRAG]{};
    GM_ADDR cFragAddrs_[MAX_FRAG]{};
    bool fragmentReady_{false};
    uint32_t fragmentRankDim_{0};
    uint64_t fragmentSliceM_{0};
    uint64_t fragmentK_{0};
    uint64_t fragmentN_{0};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
