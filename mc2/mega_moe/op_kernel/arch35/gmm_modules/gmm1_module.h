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
 * \file gmm1_module.h
 * \brief GMM1 single-core module — pure GroupMatmul, no SwiGLU/Combine code inside.
 *        AIV post-processing is provided via PostOp functor by the caller.
 *        All multi-core identity (tileIdx, totalCoreNum, subTileIdx) passed as explicit parameters.
 */

#ifndef GMM1_MODULE_H
#define GMM1_MODULE_H

#include "gmm_common.h"

namespace Gmm1Module {

template <typename PostOp, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false>
__aicore__ inline void GroupMatmul1(
    uint32_t tileIdx, uint32_t totalCoreNum, uint32_t subTileIdx,
    PostOp& postOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const MegaMoeImpl::GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom)
{
    GmmParam::Gmm1ArgsGeneric<PostOp> args{postOp, vecSetSyncCom};
    GmmParam::GroupMatmulImplGenericParam<GmmParam::Gmm1Policy, COMBINE_NO_QUANT, ElementA, ElementB, ElementC,
        ElementMxScaleA, ElementMxScaleB, IsWeightNZ>(
        tileIdx, totalCoreNum, subTileIdx, params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

template <typename PostOp, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmul1A8W4(
    uint32_t tileIdx, uint32_t totalCoreNum, uint32_t subTileIdx,
    PostOp& postOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const MegaMoeImpl::GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom)
{
    (void)vecSetSyncCom;
    GmmParam::Gmm1ArgsA8W4<PostOp> args{postOp};
    GmmParam::GroupMatmulImplA8W4Param<COMBINE_NO_QUANT, GmmParam::Gmm1Policy, ElementA, ElementB, ElementC,
        ElementMxScaleA, ElementMxScaleB>(
        tileIdx, totalCoreNum, subTileIdx, params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

} // namespace Gmm1Module

#endif
