/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the software repository for the full text of the License.
 */

/*!
 * \file swiglu_quant.h
 * \brief Single-core SwiGLU+MX quantization epilogue wrapper. All multi-core
 *        primitives (GetBlockIdx, GetBlockNum, GetSubBlockIdx, blockIdx_,
 *        aivCoreIdx_, subBlockIdx_, blockAivNum_, blockNum_) are replaced by
 *        TileInfo passed explicitly.
 */

#ifndef MEGA_MOE_SINGLE_CORE_SWIGLU_QUANT_H
#define MEGA_MOE_SINGLE_CORE_SWIGLU_QUANT_H

#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "block_epilogue_swiglu_mx_quant.h"

namespace SwigluQuant {

template <typename SwigluQuantOutType, typename QuantScaleOutType>
using BlockEpilogue = BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t,
    QuantScaleOutType, QuantScaleOutType, true>;

template <typename SwigluQuantOutType, typename QuantScaleOutType>
__aicore__ inline void InitEpilogue(
    BlockEpilogue<SwigluQuantOutType, QuantScaleOutType>& op,
    GM_ADDR yGmAddr, GM_ADDR yScaleGmAddr, GM_ADDR groupFlagListGmAddr,
    uint32_t baseM, uint32_t baseN, float clampLimit)
{
    typename BlockEpilogue<SwigluQuantOutType, QuantScaleOutType>::Params params{
        yGmAddr, yScaleGmAddr, groupFlagListGmAddr, nullptr, nullptr, nullptr,
        baseM, baseN, clampLimit};
    op.Init(params);
}

template <typename SwigluQuantOutType, typename QuantScaleOutType>
__aicore__ inline void UpdateNextProblem(
    BlockEpilogue<SwigluQuantOutType, QuantScaleOutType>& op,
    const typename BlockEpilogue<SwigluQuantOutType, QuantScaleOutType>::ProblemShape& problemShape)
{
    op.UpdateNextProblem(problemShape);
}

template <typename SwigluQuantOutType, typename QuantScaleOutType>
__aicore__ inline void RunEpilogue(
    BlockEpilogue<SwigluQuantOutType, QuantScaleOutType>& op,
    const typename BlockEpilogue<SwigluQuantOutType, QuantScaleOutType>::BlockShape& blockShape,
    const typename BlockEpilogue<SwigluQuantOutType, QuantScaleOutType>::BlockCoord& blockCoord,
    const MegaMoeModules::TileInfo& tile)
{
    if constexpr (g_coreType == AIV) {
        if (tile.subTileIdx == 0) {
            op(blockShape, blockCoord);
        }
    }
}

} // namespace SwigluQuant

#endif // MEGA_MOE_SINGLE_CORE_SWIGLU_QUANT_H
