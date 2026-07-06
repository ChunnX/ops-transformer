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
 * \file swiglu_quant.h
 * \brief Single-core SwigluQuant module: SwiGLU activation + MX quantization epilogue operator
 *        and its single-core interface wrapper.
 */

#ifndef MEGA_MOE_SWIGLU_QUANT_H
#define MEGA_MOE_SWIGLU_QUANT_H

#include "kernel_operator.h"
#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"

using namespace AscendC;

namespace MegaMoeModules {
namespace SwigluQuant {

template <typename SwigluQuantOutType, typename QuantScaleOutType>
struct SwigluQuantState {
    using BlockEpilogue = BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t,
        QuantScaleOutType, QuantScaleOutType, true>;
    BlockEpilogue epilogueOp;

    Params params;
};

template <typename SwigluQuantOutType, typename QuantScaleOutType>
__aicore__ inline void InitEpilogue(
    SwigluQuantState<SwigluQuantOutType, QuantScaleOutType> &state,
    const Params &params, uint32_t hiddenDim, float clampLimit)
{
    state.params = params;
    state.epilogueOp.Init({params.workspaceInfo.swigluQuantDataPtr, params.workspaceInfo.swigluQuantScalePtr,
        params.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr,
        ALIGN_256, ALIGN_256, clampLimit});
}

template <typename SwigluQuantOutType, typename QuantScaleOutType>
__aicore__ inline void UpdateNextProblem(
    SwigluQuantState<SwigluQuantOutType, QuantScaleOutType> &state,
    const TupleShape &problemShape)
{
    state.epilogueOp.UpdateNextProblem(problemShape);
}

template <typename SwigluQuantOutType, typename QuantScaleOutType>
__aicore__ inline void UpdateGlobalAddr(
    SwigluQuantState<SwigluQuantOutType, QuantScaleOutType> &state,
    const BlockOffset &baseOffset)
{
    if constexpr(g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            Get<IDX_C_OFFSET>(baseOffset), Get<IDX_C_SCALE_OFFSET>(baseOffset),
            Get<IDX_FLAG_OFFSET>(baseOffset), 0L, 0L, 0L};
        state.epilogueOp.UpdateGlobal_addr(vecBaseOffset);
    }
}

template <typename SwigluQuantOutType, typename QuantScaleOutType>
__aicore__ inline void RunEpilogue(
    SwigluQuantState<SwigluQuantOutType, QuantScaleOutType> &state,
    const TileInfo &tile,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &blockShape,
    const AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> &blockCoord)
{
    if constexpr(g_coreType == AIV) {
        if (tile.subTileIdx != 0) {
            return;
        }
        AscendC::SetCtrlSpr<60, 60>(0);
        state.epilogueOp(blockShape, blockCoord);
    }
}

} // namespace SwigluQuant
} // namespace MegaMoeModules

#endif
