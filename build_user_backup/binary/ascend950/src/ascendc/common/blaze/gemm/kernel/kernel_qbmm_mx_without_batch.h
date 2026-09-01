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
 * \file kernel_qbmm_mx_without_batch.h
 * \brief
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
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {
#define QBMM_MX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_MX_WITHOUT_BATCH_KERNEL_TEM_PARAMS                                              \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,                                  \
        AscendC::Std::enable_if_t<AscendC::Std::is_same_v<KernelMmadWithScaleMxWithoutBatch, \
                                                          typename BlockMmad::DispatchPolicy::ScheduleType>>

QBMM_MX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<QBMM_MX_WITHOUT_BATCH_KERNEL_TEM_PARAMS> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockSchedulerParams = typename BlockScheduler::Params;

    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
        uint32_t bMustHitL2 = 1U;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_ATOMIC_ADD = BlockMmad::DispatchPolicy::IS_ATOMIC_ADD;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;

    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<CType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        TRANS_B, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void Process(const Params& params, BlockScheduler& bs);

    // Process one block, keeping Process compact.
    template <class GmTensorA, class GmTensorB, class GmTensorScaleA, class GmTensorScaleB, class GmTensorBias,
              class GmTensorC>
    __aicore__ inline void ProcessOneBlock(const GmTensorA& gmA, const GmTensorB& gmB, const GmTensorScaleA& gmScaleA,
                                           const GmTensorScaleB& gmScaleB, const GmTensorBias& gmBias,
                                           const GmTensorC& gmC, const BlockShape& singleShape, int64_t mPos,
                                           int64_t nPos, int64_t baseM, int64_t baseN, int64_t k, int64_t scaleKLen);

    template <typename TensorB>
    __aicore__ inline void SetBL2Cache(const ProblemShape& problemShape, uint64_t currentBasicBlockM,
                                       uint64_t currentBasicBlockN, uint32_t bMustHitL2, TensorB& gmB);

    BlockMmad mmadOp_;

    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // optional input
    __gm__ AscendC::fp8_e8m0_t* scaleAGmAddr_;
    __gm__ AscendC::fp8_e8m0_t* scaleBGmAddr_;
};

QBMM_MX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_WITHOUT_BATCH_KERNEL_TEM_PARAMS>::Run(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    if constexpr (IS_ATOMIC_ADD) {
        AscendC::SetAtomicAdd<float>();
    }
    const auto& problemShape = params.problemShape;
    const auto& qbmmParams = params.qbmmParams;
    const bool isBias = qbmmParams.isBias == 1;
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
    if (isBias) {
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }

    BlockScheduler bs(problemShape, params.schParams);

    const BlockShape l0BlockShape{qbmmParams.baseM, qbmmParams.baseN, qbmmParams.baseK, 0};
    mmadOp_.Init(problemShape, l0BlockShape, params.l1Params, isBias, qbmmParams.dbL0C > 1);

    Process(params, bs);
    if constexpr (IS_ATOMIC_ADD) {
        AscendC::SetAtomicNone();
    }
}

QBMM_MX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB>
__aicore__ inline void GemmUniversal<QBMM_MX_WITHOUT_BATCH_KERNEL_TEM_PARAMS>::SetBL2Cache(
    const ProblemShape& problemShape, uint64_t currentBasicBlockM, uint64_t currentBasicBlockN, uint32_t bMustHitL2,
    TensorB& gmB)
{
    // 0xff: 256 cache line alignment for FP4 B matrix GM streaming
    // 0x7f: 128 cache line alignment for FP8 B matrix GM streaming
    constexpr uint64_t cacheLineAlignMask = IsFp4<BType>() ? 0xffUL : 0x7fUL;
    const bool isCurrentNAligned = TRANS_B || (currentBasicBlockN & cacheLineAlignMask) == 0UL;
    const bool disableWeightL2 = bMustHitL2 == 0U && currentBasicBlockM >= AscendC::Te::Get<MNK_M>(problemShape) &&
                                 isCurrentNAligned;
    gmB.SetL2CacheHint(disableWeightL2 ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                         AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
}

QBMM_MX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
template <class GmTensorA, class GmTensorB, class GmTensorScaleA, class GmTensorScaleB, class GmTensorBias,
          class GmTensorC>
__aicore__ inline void GemmUniversal<QBMM_MX_WITHOUT_BATCH_KERNEL_TEM_PARAMS>::ProcessOneBlock(
    const GmTensorA& gmA, const GmTensorB& gmB, const GmTensorScaleA& gmScaleA, const GmTensorScaleB& gmScaleB,
    const GmTensorBias& gmBias, const GmTensorC& gmC, const BlockShape& singleShape, int64_t mPos, int64_t nPos,
    int64_t baseM, int64_t baseN, int64_t k, int64_t scaleKLen)
{
    constexpr int64_t kPos = 0L; // K is not split, so the K coordinate is 0.
    auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(baseM, k));
    auto gmBlockScaleA = gmScaleA.Slice(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(baseM, scaleKLen));
    auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(k, baseN));
    auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(scaleKLen, baseN));
    auto gmBlockBias = gmBias.Slice(AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(1L, baseN));
    auto gmBlockC = gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(baseM, baseN));

    mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
}

QBMM_MX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_WITHOUT_BATCH_KERNEL_TEM_PARAMS>::Process(const Params& params,
                                                                                       BlockScheduler& bs)
{
    const auto& problemShape = params.problemShape;
    const auto m = AscendC::Te::Get<MNK_M>(problemShape);
    const auto n = AscendC::Te::Get<MNK_N>(problemShape);
    const auto k = AscendC::Te::Get<MNK_K>(problemShape);
    const auto scaleKLen = Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleKLen);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, n);
    auto layoutBias = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1L, n);
    auto layoutC = MakeLayoutC{}(m, n);
    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
    auto gmScaleA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleAGmAddr_),
                                            layoutScaleA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB);
    auto gmScaleB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBGmAddr_),
                                            layoutScaleB);
    auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmAddr_), layoutBias);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);
    if constexpr (IS_ATOMIC_ADD) {
        gmC.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
    }

    const auto mTailTile = params.schParams.mTailTile;
    const auto nTailTile = params.schParams.nTailTile;
    if ((bs.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= AscendC::GetBlockNum()) {
        bs.UpdateTailTile(mTailTile, nTailTile);
    }

    BlockCoord blockCoord;
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    while (bs.GetTileIdx(blockCoord)) {
        BlockShape singleShape = bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE,
                                                           WEIGHT_NZ>(blockCoord);
        const auto baseM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
        const auto baseN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
        if (baseM <= 0 || baseN <= 0) {
            break;
        }
        bs.GetTileCoord(blockCoord, mPos, nPos);
        SetBL2Cache(problemShape, baseM, baseN, params.qbmmParams.bMustHitL2, gmB);
        ProcessOneBlock(gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC, singleShape, mPos, nPos, baseM, baseN, k, scaleKLen);
    }
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
