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
 * \file moe_distribute_combine_v2.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "moe_distribute_combine_v2_tiling_key.h"
#include "moe_distribute_combine_v2_tiling.h"
#include "moe_distribute_combine_v2.h"

using namespace MoeDistributeCombineV2Impl;
using namespace Mc2Kernel;
using namespace Mc2Tiling;
using namespace AscendC;

namespace {
template <CombineMC2TypeClass>
__aicore__ inline void ExecMoeDistributeCombineV2(GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds,
                                                GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, 
                                                GM_ADDR tpSendCount, GM_ADDR scales, GM_ADDR xActiveMask, 
                                                GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX, 
                                                GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2,
                                                GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut, 
                                                GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe *pipePtr)
{
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
    MoeDistributeCombineV2<CombineMC2TypeFunc> op;
    op.Init(mc2Context, expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, nullptr, nullptr,
            scales, xActiveMask, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, 
            constExpertAlpha2, constExpertV, performanceInfo, nullptr, nullptr, XOut, workspaceGM, pipePtr, &tilingData);
    op.Process();
}
}
template<bool HasTp, uint8_t QuantMode, uint8_t LayeredMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_combine_v2(GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                                                    GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount,
                                                    GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale,
                                                    GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR sharedExpertX, 
                                                    GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
                                                    GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, 
                                                    GM_ADDR performanceInfo, GM_ADDR XOut, GM_ADDR workspaceGM, 
                                                    GM_ADDR tilingGM)

{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData);
    TPipe pipe;

#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    if constexpr ((ArchTag == TILINGKEY_TPL_A5) && (LayeredMode == TILINGKEY_TPL_MTE)) {
        ExecMoeDistributeCombineV2<DTYPE_EXPAND_X, DTYPE_X, int32_t, HasTp, QuantMode, false>(
        mc2Context, expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX,
        elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM,
        tilingGM, &pipe);
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
#endif
#endif
}