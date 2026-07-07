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
 * \file gmm2_module.h
 * \brief GMM2 single-core module — pure GroupMatmul2, no Combine code inside.
 *        AIV post-processing (Combine for COMBINE_NO_QUANT) is provided via PostOp functor by the caller.
 *        All multi-core identity (tileIdx, totalCoreNum, subTileIdx) passed as explicit parameters.
 *        ProcessCombine for quant modes stays in the multi-core orchestrator (mega_moe.h).
 */

#ifndef GMM2_MODULE_H
#define GMM2_MODULE_H

#include "gmm_common.h"

namespace Gmm2Module {

template <uint8_t CombineQuantMode, typename PostOp, typename ElementA, typename ElementB, typename ElementC,
           typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false,
           bool IsLayered = false>
__aicore__ inline void GroupMatmul2(
    uint32_t tileIdx, uint32_t totalCoreNum, uint32_t subTileIdx,
    PostOp& postOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const MegaMoeImpl::GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
    int32_t& vecSetSyncCom2, uint16_t& pingpongIdx, uint32_t groupIdx)
{
    GmmParam::Gmm2ArgsGeneric<PostOp> args{postOp, vecSetSyncCom2, pingpongIdx, groupIdx};
    GmmParam::GroupMatmulImplGenericParam<GmmParam::Gmm2Policy, CombineQuantMode, ElementA, ElementB, ElementC,
        ElementMxScaleA, ElementMxScaleB, IsWeightNZ, IsLayered>(
        tileIdx, totalCoreNum, subTileIdx, params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

template <uint8_t CombineQuantMode, typename PostOp, typename ElementA, typename ElementB, typename ElementC,
    typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmul2A8W4(
    uint32_t tileIdx, uint32_t totalCoreNum, uint32_t subTileIdx,
    PostOp& postOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const MegaMoeImpl::GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom2, uint32_t groupCnt,
    uint16_t& pingpongIdx, uint32_t groupIdx = 0)
{
    (void)vecSetSyncCom2;
    (void)groupCnt;
    (void)pingpongIdx;
    GmmParam::Gmm2ArgsA8W4<PostOp> args{postOp, groupIdx};
    GmmParam::GroupMatmulImplA8W4Param<CombineQuantMode, GmmParam::Gmm2Policy, ElementA, ElementB, ElementC,
        ElementMxScaleA, ElementMxScaleB>(
        tileIdx, totalCoreNum, subTileIdx, params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

} // namespace Gmm2Module

#endif
