/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_v2_mxquant.h
 * \brief
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_MXQUANT_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_MXQUANT_H

#include "cgmct/kernel/kernel_gmm_swiglu_mxquant.h"
#include "cgmct/block/block_mx_mm_aic_to_aiv_builder.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "tensor_api/tensor.h"

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;

namespace Cgmct {
namespace Gemm {
namespace Block {
namespace {
constexpr uint64_t GMMSQ_MX_SCALE_FACTOR_BIT = 8UL;
constexpr uint64_t GMMSQ_MX_SCALE_FACTOR_MASK = 0xFFUL;
} // namespace

template <class BlockMmadOp_, class LayoutA_, class LayoutB_>
class BlockMxTensorApiConcatToUbAdapter {
public:
    using BlockMmadOp = BlockMmadOp_;
    using AType = typename BlockMmadOp::AType;
    using BType = typename BlockMmadOp::BType;
    using CType = typename BlockMmadOp::CType;
    using BiasType = typename BlockMmadOp::BiasType;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    static constexpr uint32_t C0_SIZE = Blaze::Gemm::IsFp4<AType>() ? Blaze::Gemm::C0_SIZE_B4 :
                                                                     Blaze::Gemm::C0_SIZE_B8;

    using TensorLayoutA = AscendC::Std::conditional_t<transA, AscendC::Te::DNExtLayoutPtn,
                                                      AscendC::Te::NDExtLayoutPtn>;
    using TensorLayoutB = AscendC::Std::conditional_t<transB, AscendC::Te::DNExtLayoutPtn,
                                                      AscendC::Te::NDExtLayoutPtn>;
    using TensorLayoutC = AscendC::Te::NDExtLayoutPtn;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<TensorLayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<TensorLayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<Blaze::Gemm::SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<Blaze::Gemm::SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<Blaze::Gemm::SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<Blaze::Gemm::SCALE_C0>>>;

    __aicore__ inline void Init(TCubeTiling *__restrict matmulTiling, AscendC::TPipe *tpipe = nullptr)
    {
        (void)tpipe;
        m_ = matmulTiling->M;
        n_ = matmulTiling->N;
        k_ = matmulTiling->Ka;
        scaleK_ = GetScaleK(k_);
        uint64_t scaleFactorA = matmulTiling->mxTypePara & GMMSQ_MX_SCALE_FACTOR_MASK;
        uint64_t scaleFactorB =
            (matmulTiling->mxTypePara >> GMMSQ_MX_SCALE_FACTOR_BIT) & GMMSQ_MX_SCALE_FACTOR_MASK;
        uint64_t kAL1 = static_cast<uint64_t>(matmulTiling->stepKa) * static_cast<uint64_t>(matmulTiling->baseK);
        uint64_t kBL1 = static_cast<uint64_t>(matmulTiling->stepKb) * static_cast<uint64_t>(matmulTiling->baseK);
        uint64_t scaleKL1 = Cgmct::Gemm::Min(
            Cgmct::Gemm::Max(scaleFactorA * kAL1, scaleFactorB * kBL1), static_cast<uint64_t>(matmulTiling->Ka));
        uint64_t concatBaseN = Cgmct::Gemm::Min(static_cast<uint64_t>(matmulTiling->baseN) * 2UL,
                                                static_cast<uint64_t>(matmulTiling->N));
        typename BlockMmadOp::BlockShape l0TileShape{static_cast<int64_t>(matmulTiling->baseM),
                                                     static_cast<int64_t>(concatBaseN),
                                                     static_cast<int64_t>(matmulTiling->baseK), 0};
        typename BlockMmadOp::L1Params l1Params{kAL1, kBL1, scaleKL1};
        typename BlockMmadOp::MmadParams params{l0TileShape, l1Params, false, false};
        typename BlockMmadOp::ProblemShape problemShape{m_, n_, k_, 0};
        if ASCEND_IS_AIC {
            blockMmad_.Init(problemShape, params);
        }
    }

    template <typename SingleShape>
    __aicore__ inline void RunSwigluConcat(const AscendC::GlobalTensor<AType> &aGlobal,
                                           const AscendC::GlobalTensor<BType> &bGlobal,
                                           const AscendC::GlobalTensor<AscendC::fp8_e8m0_t> &scaleAGlobal,
                                           const AscendC::GlobalTensor<AscendC::fp8_e8m0_t> &scaleBGlobal,
                                           int64_t aOffset, int64_t bLeftOffset, int64_t bRightOffset,
                                           int64_t scaleAOffset, int64_t scaleBLeftOffset,
                                           int64_t scaleBRightOffset,
                                           const AscendC::LocalTensor<CType> &cLocal, const SingleShape &singleShape,
                                           bool transAIn, bool transBIn)
    {
        (void)transAIn;
        (void)transBIn;
        RunConcat(reinterpret_cast<__gm__ AType *>(aGlobal[aOffset].GetPhyAddr()),
                  reinterpret_cast<__gm__ BType *>(bGlobal[bLeftOffset].GetPhyAddr()),
                  reinterpret_cast<__gm__ BType *>(bGlobal[bRightOffset].GetPhyAddr()),
                  reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(scaleAGlobal[scaleAOffset].GetPhyAddr()),
                  reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(scaleBGlobal[scaleBLeftOffset].GetPhyAddr()),
                  reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(scaleBGlobal[scaleBRightOffset].GetPhyAddr()),
                  cLocal, singleShape);
    }

private:
    __aicore__ inline int64_t GetScaleK(int64_t k) const
    {
        return ((k + Blaze::Gemm::MXFP_DIVISOR_SIZE - 1) >> Blaze::Gemm::MXFP_DIVISOR_SHIFT)
               << Blaze::Gemm::MXFP_MULTI_BASE_SHIFT;
    }

    template <typename SingleShape>
    __aicore__ inline void RunConcat(__gm__ AType *aPtr, __gm__ BType *bLeftPtr, __gm__ BType *bRightPtr,
                                     __gm__ AscendC::fp8_e8m0_t *scaleAPtr,
                                     __gm__ AscendC::fp8_e8m0_t *scaleBLeftPtr,
                                     __gm__ AscendC::fp8_e8m0_t *scaleBRightPtr,
                                     const AscendC::LocalTensor<CType> &cLocal, const SingleShape &singleShape)
    {
        int64_t curM = Get<0>(singleShape);
        int64_t curN = Get<1>(singleShape);
        int64_t curK = Get<2>(singleShape);
        auto layoutA = MakeLayoutA{}(curM, k_);
        auto layoutB = MakeLayoutB{}(k_, n_);
        auto layoutScaleA = MakeLayoutScaleA{}(curM, scaleK_);
        auto layoutScaleB = MakeLayoutScaleB{}(scaleK_, n_);
        auto layoutBias = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(static_cast<int64_t>(1),
                                                                                   curN * 2);
        auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aPtr), layoutA);
        auto gmBLeft =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bLeftPtr), layoutB);
        auto gmBRight =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bRightPtr), layoutB);
        auto gmScaleA =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleAPtr), layoutScaleA);
        auto gmScaleBLeft =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBLeftPtr),
                                    layoutScaleB);
        auto gmScaleBRight =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBRightPtr),
                                    layoutScaleB);
        auto gmBias = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(nullptr)),
            layoutBias);
        typename BlockMmadOp::BlockShape blockShape{curM, curN, curK, 0};
        blockMmad_.RunSwigluConcat(gmA, gmBLeft, gmBRight, gmScaleA, gmScaleBLeft, gmScaleBRight, gmBias,
                                   const_cast<AscendC::LocalTensor<CType> &>(cLocal), blockShape);
    }

    BlockMmadOp blockMmad_;
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t scaleK_{0};
};

template <class AType_, class LayoutA_, class BType_, class LayoutB_, class BiasType_, class CType_, class LayoutC_,
          class L1TileShape_, class L0TileShape_, class BlockScheduler_,
          class BlockMatmulPolicy_ = QuantMatmulWithTileMultiBlock<>, class TileCopyParam_ = void,
          typename Enable_ = void>
class BlockMxTensorApiConcatMmAicToAivBuilder {
    static_assert(AscendC::Std::always_false_v<BlockMatmulPolicy_>,
                  "BlockMxTensorApiConcatMmAicToAivBuilder is not implemented for this BlockMatmulPolicy");
};

template <class AType_, class LayoutA_, class BType_, class LayoutB_, class BiasType_, class CType_, class LayoutC_,
          class L1TileShape_, class L0TileShape_, class BlockScheduler_, class BlockMatmulPolicy_,
          class TileCopyParam_>
class BlockMxTensorApiConcatMmAicToAivBuilder<
    AType_, LayoutA_, BType_, LayoutB_, BiasType_, CType_, LayoutC_, L1TileShape_, L0TileShape_, BlockScheduler_,
    BlockMatmulPolicy_, TileCopyParam_,
    AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<QuantMatmulWithTileMultiBlock<>, BlockMatmulPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using BlockMatmulPolicy = BlockMatmulPolicy_;
    using TileCopyParam = TileCopyParam_;

    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    static constexpr CubeFormat formatA = TagToFormat<LayoutA>::format;
    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;
    static constexpr CubeFormat formatC = TagToFormat<LayoutC>::format;

    using TensorLayoutA = AscendC::Std::conditional_t<transA, AscendC::Te::DNExtLayoutPtn,
                                                      AscendC::Te::NDExtLayoutPtn>;
    using TensorLayoutB = AscendC::Std::conditional_t<transB, AscendC::Te::DNExtLayoutPtn,
                                                      AscendC::Te::NDExtLayoutPtn>;
    using TensorLayoutC = AscendC::Te::NDExtLayoutPtn;
    using TensorBlockMmadPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
    using TensorBlockMmad = Blaze::Gemm::Block::BlockMmad<TensorBlockMmadPolicy, AType, TensorLayoutA, BType,
                                                          TensorLayoutB, CType, TensorLayoutC, BiasType,
                                                          TensorLayoutC>;
    using BlockMmadOp = Block::BlockMxTensorApiConcatToUbAdapter<TensorBlockMmad, LayoutA, LayoutB>;
    static constexpr bool useSingleL0c2Ub = true;

    static constexpr int64_t l1M = GetIntegralConstant<MNK_M, L1TileShape>();
    static constexpr int64_t l1N = GetIntegralConstant<MNK_N, L1TileShape>();
    static constexpr int64_t l1K = GetIntegralConstant<MNK_K, L1TileShape>();

    struct Arguments {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
    };

    using Params = Arguments;
};
} // namespace Block
} // namespace Gemm
} // namespace Cgmct

template <typename layoutA, typename layoutB>
__aicore__ inline void GmmSwigluAswt(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                     GM_ADDR weightAssistanceMatrix, GM_ADDR smoothScale, GM_ADDR groupList,
                                     GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingDataParams, gmmSwigluQuantParams, gmmSwigluQuantParams_, tiling);       \
    GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingDataParams, mmTilingData, mmTilingData_, tiling);                       \
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    // 定义矩阵的类型和布局
    using AType = DTYPE_X;
    using BType = DTYPE_WEIGHT;
    using CType = DTYPE_Y;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajorAlign;
    using weightscaleType = AscendC::fp8_e8m0_t;
    using BiasType = float;
    // 定义scheduler类型
    using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;
    // 定义MMAD类型
    using C1Type = float;
    // 定义BlockEpilogue类型
    using BlockEpilogue = Block::BlockEpilogueSwigluQuant<L0TileShape, CType, C1Type, weightscaleType, weightscaleType,
                                                          true>;
    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;
    using BlockMmad = Block::BlockMxMmAicToAivBuilder<AType, LayoutA, BType, LayoutB, BiasType, C1Type, LayoutC, L1TileShape,
                                                 L0TileShape, BlockScheduler, QuantMatmulWithTileMultiBlock<>,
                        Tile::TileCopy<Arch::DAV3510, Tile::CopyInAndCopyOutSplitMWithParams>>;
    using QGmmKernel =
    Kernel::KernelGmmSwiGluMixOnlineDynamic<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QGmmKernel::Params;
    using GMMTiling = typename QGmmKernel::GMMTiling;
    GMMTiling gmmParams{gmmSwigluQuantParams_.groupNum, gmmSwigluQuantParams_.groupListType, mmTilingData_.baseM,
                        mmTilingData_.baseN, mmTilingData_.baseK};
    gmmParams.matmulTiling = &mmTilingData_;
    Params params = {// template shape, gmm shape can not get now
                    {1, 1, 1, 1},
                    // mmad args
                    {x, weight, weightScale, xScale, y, groupList},
                    {y, yScale, nullptr, nullptr, nullptr, static_cast<uint32_t>(mmTilingData_.baseM),
                        static_cast<uint32_t>(mmTilingData_.baseN)},
                    // gmm tiling data
                    gmmParams};
    QGmmKernel op;
    op(params);
}

template <typename layoutA, typename layoutB>
__aicore__ inline void GmmSwigluMxFp8NdTensorApi(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                                 GM_ADDR weightAssistanceMatrix, GM_ADDR smoothScale,
                                                 GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingDataParams, gmmSwigluQuantParams, gmmSwigluQuantParams_, tiling);       \
    GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingDataParams, mmTilingData, mmTilingData_, tiling);                       \
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    using AType = DTYPE_X;
    using BType = DTYPE_WEIGHT;
    using CType = DTYPE_Y;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajorAlign;
    using WeightScaleType = AscendC::fp8_e8m0_t;
    using BiasType = float;
    using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;
    using C1Type = float;
    using BlockEpilogue = Block::BlockEpilogueSwigluQuant<L0TileShape, CType, C1Type, WeightScaleType,
                                                          WeightScaleType, true>;
    using ProblemShape = MatmulShape;
    using BlockMmad = Block::BlockMxTensorApiConcatMmAicToAivBuilder<
        AType, LayoutA, BType, LayoutB, BiasType, C1Type, LayoutC, L1TileShape, L0TileShape, BlockScheduler,
        QuantMatmulWithTileMultiBlock<>, Tile::TileCopy<Arch::DAV3510, Tile::CopyInAndCopyOutSplitMWithParams>>;
    using QGmmKernel =
        Kernel::KernelGmmSwiGluMixOnlineDynamic<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QGmmKernel::Params;
    using GMMTiling = typename QGmmKernel::GMMTiling;
    GMMTiling gmmParams{gmmSwigluQuantParams_.groupNum, gmmSwigluQuantParams_.groupListType, mmTilingData_.baseM,
                        mmTilingData_.baseN, mmTilingData_.baseK};
    gmmParams.matmulTiling = &mmTilingData_;
    Params params = {{1, 1, 1, 1},
                     {x, weight, weightScale, xScale, y, groupList},
                     {y, yScale, nullptr, nullptr, nullptr, static_cast<uint32_t>(mmTilingData_.baseM),
                      static_cast<uint32_t>(mmTilingData_.baseN)},
                     gmmParams};
    QGmmKernel op;
    op(params);
}

#endif
