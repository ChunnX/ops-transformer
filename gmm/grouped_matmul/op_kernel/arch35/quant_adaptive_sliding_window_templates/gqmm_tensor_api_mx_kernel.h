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
 * \file gqmm_tensor_api_mx_kernel.h
 * \brief
 */

#ifndef GQMM_TENSOR_API_MX_KERNEL_H
#define GQMM_TENSOR_API_MX_KERNEL_H

#include <stdio.h>

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_mx.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "cgmct/block/block_scheduler_policy.h"
#include "cgmct/epilogue/block_epilogue_empty.h"
#include "cgmct/kernel/kernel_qgmm_mx.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

using GMMQuantParams = GroupedMatmulTilingData::GMMQuantParams;
using QuantBasicApiMMTiling = GroupedMatmulTilingData::QuantBasicApiMMTiling;
using namespace Cgmct::Gemm;

template <class InnerBlockMmad>
class BlazeBlockMmadMxAdapter {
public:
    using AType = typename InnerBlockMmad::AType;
    using BType = typename InnerBlockMmad::BType;
    using CType = typename InnerBlockMmad::CType;
    using BiasType = typename InnerBlockMmad::BiasType;
    using LayoutA = typename InnerBlockMmad::LayoutA;
    using LayoutB = typename InnerBlockMmad::LayoutB;
    using LayoutC = typename InnerBlockMmad::LayoutC;
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    static constexpr bool transA = InnerBlockMmad::transA;
    static constexpr bool transB = InnerBlockMmad::transB;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kAL1;
        uint64_t kBL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    __aicore__ inline void Init(const AscendC::Shape<int64_t, int64_t, int64_t>& problemShape,
                                const AscendC::Shape<int64_t, int64_t, int64_t>& l0TileShape,
                                const L1Params& l1Params, bool isBias, bool dbL0C)
    {
        innerProblemShape_ = {static_cast<uint64_t>(AscendC::Get<0>(problemShape)),
                              static_cast<uint64_t>(AscendC::Get<1>(problemShape)),
                              static_cast<uint64_t>(AscendC::Get<2>(problemShape)), 0UL};
        innerBlockShape_ = {AscendC::Get<0>(l0TileShape), AscendC::Get<1>(l0TileShape), AscendC::Get<2>(l0TileShape),
                            0};
        innerL1Params_ = {l1Params.kAL1, l1Params.scaleKL1, l1Params.l1BufNum};
        isBias_ = isBias;
        dbL0C_ = dbL0C;
        inner_.Init(innerProblemShape_, innerBlockShape_, innerL1Params_, isBias_, dbL0C_);
    }

    __aicore__ inline void UpdateParamsForNextProblem(const AscendC::Shape<int64_t, int64_t, int64_t>& problemShape)
    {
        innerProblemShape_ = {static_cast<uint64_t>(AscendC::Get<0>(problemShape)),
                              static_cast<uint64_t>(AscendC::Get<1>(problemShape)),
                              static_cast<uint64_t>(AscendC::Get<2>(problemShape)), 0UL};
        inner_.Init(innerProblemShape_, innerBlockShape_, innerL1Params_, isBias_, dbL0C_);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
              typename TensorC>
    __aicore__ inline void operator()(TensorA const& gmA, TensorB const& gmB, TensorScaleA const& gmScaleA,
                                      TensorScaleB const& gmScaleB, TensorBias const& gmBias, TensorC const& gmC,
                                      const AscendC::Shape<int64_t, int64_t, int64_t>& blockShape)
    {
        inner_(gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC,
               {AscendC::Get<0>(blockShape), AscendC::Get<1>(blockShape), AscendC::Get<2>(blockShape), 0});
    }

private:
    InnerBlockMmad inner_;
    typename InnerBlockMmad::ProblemShape innerProblemShape_{};
    typename InnerBlockMmad::BlockShape innerBlockShape_{};
    typename InnerBlockMmad::L1Params innerL1Params_{};
    bool isBias_{false};
    bool dbL0C_{false};
};

template <class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType, class xLayout,
          class wLayout, class yLayout, class l0cType>
__aicore__ inline void GmmTensorApiMxKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList,
                                            GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                            const GMMQuantParams* gmmBaseParamsIn,
                                            const QuantBasicApiMMTiling* mmTilingDataIn, AscendC::TPipe* que)
{
    (void)workspace;
    (void)que;
    if ASCEND_IS_AIV {
        return;
    }
    if ASCEND_IS_AIC {
        if (AscendC::GetBlockIdx() == 0) {
            AscendC::printf(
                "[zzzlog][gqmm_tensor_api][L42] raw enter x=%p w=%p bias=%p scale=%p groupList=%p perTokenScale=%p "
                "y=%p workspace=%p gmmBaseParamsIn=%p mmTilingDataIn=%p que=%p\n",
                x, weight, bias, scale, groupList, perTokenScale, y, workspace, gmmBaseParamsIn, mmTilingDataIn, que);
        }
    }
    using AType = xType;
    using BType = wType;
    using BiasType = float;
    using YType = yType;
    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using ProblemShape = MatmulShape;
    using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;
    using BlockMmadPolicy = Blaze::Gemm::MatmulWithScaleMx<0>;
    using TensorBlockMmad =
        Blaze::Gemm::Block::BlockMmad<BlockMmadPolicy, AType, LayoutA, BType, LayoutB, YType, LayoutC, BiasType, LayoutC>;
    using BlockMmad = BlazeBlockMmadMxAdapter<TensorBlockMmad>;
    using BlockEpilogue = Block::BlockEpilogueEmpty;
    using GmmKernel = Kernel::KernelQGmmMx<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;

    GMMTiling gmmParams{gmmBaseParamsIn->groupNum, mmTilingDataIn->m,          mmTilingDataIn->n,
                        mmTilingDataIn->k,         mmTilingDataIn->baseM,      mmTilingDataIn->baseN,
                        mmTilingDataIn->baseK,     mmTilingDataIn->kAL1,       mmTilingDataIn->kBL1,
                        mmTilingDataIn->scaleKAL1, mmTilingDataIn->scaleKBL1,  mmTilingDataIn->isBias,
                        mmTilingDataIn->dbL0C,     gmmBaseParamsIn->groupType, gmmBaseParamsIn->groupListType};

    if ASCEND_IS_AIC {
        if (AscendC::GetBlockIdx() == 0) {
            AscendC::printf(
                "[zzzlog][gqmm_tensor_api][L74] enter groupNum=%u m=%u n=%u k=%u baseM=%u baseN=%u baseK=%u "
                "kAL1=%u kBL1=%u scaleKAL1=%u scaleKBL1=%u isBias=%u dbL0C=%u groupType=%d groupListType=%u "
                "x=%p w=%p bias=%p scale=%p groupList=%p perTokenScale=%p y=%p\n",
                gmmParams.groupNum, gmmParams.m, gmmParams.n, gmmParams.k, gmmParams.baseM, gmmParams.baseN,
                gmmParams.baseK, gmmParams.kAL1, gmmParams.kBL1, gmmParams.scaleKAL1, gmmParams.scaleKBL1,
                gmmParams.isBias, gmmParams.dbL0C, static_cast<int>(gmmParams.groupType), gmmParams.groupListType, x,
                weight, bias, scale, groupList, perTokenScale, y);
        }
    }

    Params params = {
        {1, 1, 1, 1},
        {x, weight, scale, perTokenScale, y, bias, groupList},
        gmmParams};

    if ASCEND_IS_AIC {
        if (AscendC::GetBlockIdx() == 0) {
            AscendC::printf("[zzzlog][gqmm_tensor_api][L111] before gmm(params)\n");
        }
    }
    GmmKernel gmm;
    gmm(params);
    if ASCEND_IS_AIC {
        if (AscendC::GetBlockIdx() == 0) {
            AscendC::printf("[zzzlog][gqmm_tensor_api][L118] after gmm(params)\n");
        }
    }
}

#endif
