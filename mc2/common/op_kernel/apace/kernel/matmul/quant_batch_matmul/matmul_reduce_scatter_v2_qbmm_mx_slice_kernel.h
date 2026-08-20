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
 * \file matmul_reduce_scatter_v2_qbmm_mx_slice_kernel.h
 * \brief Slice-producing APACE MXFP8/MXFP4 Matmul kernel for MatmulReduceScatterV2.
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
#include "../../../basic/fragment_tensor/fragment_tensor.h"
#include "../../../basic/fragment_tensor/fragment_tensor_api.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape, class BlockMmad, class BlockScheduler>
class MatmulReduceScatterV2QbmmMxSliceKernel {
public:
    using SourceBlockMmad = BlockMmad;
    using AType = typename SourceBlockMmad::AType;
    using BType = typename SourceBlockMmad::BType;
    using CType = typename SourceBlockMmad::CType;
    using BiasType = typename SourceBlockMmad::BiasType;
    using LayoutA = typename SourceBlockMmad::LayoutA;
    using LayoutB = typename SourceBlockMmad::LayoutB;
    using LayoutC = typename SourceBlockMmad::LayoutC;
    // MatmulReduceScatterV2 currently constrains C and bias to ND.  The
    // generic Blaze BlockMmad does not expose LayoutBias on every toolchain,
    // while LayoutC is the same ND pattern for this fusion.
    using LayoutBias = LayoutC;

    static constexpr bool transA = SourceBlockMmad::TRANS_A;
    static constexpr bool transB = SourceBlockMmad::TRANS_B;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t CACHE_LINE_ALIGN_MASK = IsFp4<AType>() ? 0xff : 0x7f;
    static constexpr int32_t SCALE_C0 = 2;
    static constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;

    using FragmentBlockMmad = Blaze::Gemm::Block::QmmMxBlockMmadFragment<
        0, false, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
    static constexpr bool weightNz = FragmentBlockMmad::weightNz;
    using BlockMmadParams = typename FragmentBlockMmad::Params;
    using L1Params = typename FragmentBlockMmad::L1Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using BlockShape = typename FragmentBlockMmad::BlockShape;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

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

    struct SliceParams {
        uint32_t rankDim;
        uint32_t sliceM;
        uint32_t paddedSliceM;
        uint64_t perRankM;
        uint64_t sliceStart;
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

        const bool isBias = params.qbmmParams.isBias == QBMMTiling::BIAS_ENABLED;
        BlockScheduler scheduler(params.problemShape, params.schedulerParams);
        BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN,
                               params.qbmmParams.baseK, 0};
        FragmentBlockMmad mmOp;
        mmOp.Init(params.problemShape, l0TileShape, params.l1Params, isBias,
                  params.qbmmParams.dbL0C > 1, 1);
        BuildFragmentTensors(params);
        Process(params, scheduler, mmOp);
    }

    __aicore__ inline void operator()(const Params &params)
    {
        Run(params);
    }

private:
    __aicore__ inline Apace::Basic::FragmentParam<2> MakeFragParam(
        uint64_t fragmentSize, uint64_t realFragmentSize, uint32_t fragmentCnt, uint64_t shape1) const
    {
        Apace::Basic::FragmentParam<2> param{};
        param.assembleAxis = 0;
        param.assembledShape[0] = fragmentSize * fragmentCnt;
        param.assembledShape[1] = shape1;
        param.fragmentSize = fragmentSize;
        param.realFragmentSize = realFragmentSize;
        param.fragmentCnt = fragmentCnt;
        return param;
    }

    __aicore__ inline void BuildFragmentTensors(const Params &params)
    {
        using AscendC::Te::Get;
        const auto &slice = params.sliceParams;
        const uint64_t k = static_cast<uint64_t>(Get<MNK_K>(params.problemShape));
        const uint64_t n = static_cast<uint64_t>(Get<MNK_N>(params.problemShape));
        const uint64_t scaleK = CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        const uint64_t aRowBytes = IsFp4<AType>() ? CeilDiv(k, 2UL) : k * sizeof(AType);
        const uint64_t scaleARowBytes = scaleK * sizeof(AscendC::fp8_e8m0_t);
        const uint64_t cRowBytes = n * sizeof(CType);

        for (uint32_t rank = 0; rank < slice.rankDim; ++rank) {
            const uint64_t globalM = static_cast<uint64_t>(rank) * slice.perRankM + slice.sliceStart;
            aFragAddrs_[rank] = params.mmParams.aGmAddr + globalM * aRowBytes;
            scaleAFragAddrs_[rank] = params.mmParams.scaleAGmAddr + globalM * scaleARowBytes;
            cFragAddrs_[rank] = params.mmParams.cGmAddr + globalM * cRowBytes;
        }

        fragA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
            MakeFragParam(slice.paddedSliceM, slice.sliceM, slice.rankDim, k), aFragAddrs_);
        fragScaleA_ = Apace::Basic::MakeFragmentTensor<
            2, MAX_FRAG, MakeLayoutScaleA, AscendC::fp8_e8m0_t>(
            MakeFragParam(slice.paddedSliceM, slice.sliceM, slice.rankDim, scaleK), scaleAFragAddrs_);
        fragC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
            MakeFragParam(slice.paddedSliceM, slice.sliceM, slice.rankDim, n), cFragAddrs_);
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

    __aicore__ inline void Process(const Params &params, BlockScheduler &scheduler, FragmentBlockMmad &mmOp)
    {
        using AscendC::Te::Get;
        const int64_t n = Get<MNK_N>(params.problemShape);
        const int64_t k = Get<MNK_K>(params.problemShape);
        const int64_t scaleK = CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

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
            auto blockA = fragA_.Slice(AscendC::Te::MakeCoord(mPos, 0L), AscendC::Te::MakeShape(currentM, k));
            auto blockScaleA = fragScaleA_.Slice(
                AscendC::Te::MakeCoord(mPos, 0L), AscendC::Te::MakeShape(currentM, scaleK));
            auto blockB = gmB.Slice(AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(k, currentN));
            auto blockScaleB = gmScaleB.Slice(
                AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(scaleK, currentN));
            auto blockBias = gmBias.Slice(
                AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(1L, currentN));
            auto blockC = fragC_.Slice(
                AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(currentM, currentN));

            mmOp(blockA, blockB, blockScaleA, blockScaleB, blockBias, cFragAddrs_,
                 params.sliceParams.perRankM, params.sliceParams.paddedSliceM, 1,
                 params.sliceParams.sliceM, params.sliceParams.rankDim, n,
                 singleShape, mPos, nPos, 0, blockC);
        }
    }

private:
    GM_ADDR aFragAddrs_[MAX_FRAG]{};
    GM_ADDR scaleAFragAddrs_[MAX_FRAG]{};
    GM_ADDR cFragAddrs_[MAX_FRAG]{};
    FragTensorA fragA_{};
    FragScaleA fragScaleA_{};
    FragTensorC fragC_{};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
