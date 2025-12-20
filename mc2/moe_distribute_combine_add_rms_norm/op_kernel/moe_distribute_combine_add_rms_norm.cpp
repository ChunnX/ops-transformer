/**
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_add_rms_norm.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "moe_distribute_combine_add_rms_norm.h"
#include "moe_distribute_combine_add_rms_norm_tiling_key.h"
#if __has_include("../moe_distribute_combine_v2/moe_distribute_combine_v2_tiling.h")
#include "../moe_distribute_combine_v2/moe_distribute_combine_v2_tiling.h"
#else
#include "../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2_tiling.h"
#endif

using namespace AscendC;
using namespace MoeDistributeCombineAddRmsNormImpl;
using namespace Mc2Tiling;

namespace {
template <TemplateMC2TypeClass>
__aicore__ inline void ExecMoeDistributeCombineAddRmsNorm(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, GM_ADDR tpSendCount,
    GM_ADDR residualX, GM_ADDR gamma, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo,
    GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR YOut,
    GM_ADDR dynamicScaleOut, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe* pipePtr)
{
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
    MoeDistributeCombineAddRmsNorm<TemplateMC2TypeFunc> op;
    op.Init(
        expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, residualX, gamma, scales, xActiveMask,
        sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, YOut, dynamicScaleOut,
        XOut, workspaceGM, pipePtr, &tilingData);
    op.Process();
    }
}

template<bool HasTp, int QuantMode> __global__ __aicore__ void moe_distribute_combine_add_rms_norm(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR residualX,
    GM_ADDR gamma, GM_ADDR tpSendCount, GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale,
    GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1, 
    GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR YOut, GM_ADDR dynamicScaleOut, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
        ExecMoeDistributeCombineAddRmsNorm<DTYPE_EXPAND_X, DTYPE_X, int32_t, HasTp, QuantMode == TILINGKEY_INT8_QUANT>(
            expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, residualX, gamma, scales, xActiveMask,
            sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, YOut, dynamicScaleOut, XOut, workspaceGM, tilingGM, &pipe);
#endif
}