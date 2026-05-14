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
 * \file moe_init_routing_v2.cpp
 * \brief Non-quant variant of MoeInitRoutingV2 — used by the unified
 *        DispatchFFNCombine kernel's BF16/FP16 path where the input X is
 *        forwarded to remote ranks WITHOUT being quantized to INT8.
 *
 *        Ported (and re-namespaced) from
 *        `mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/moe_init_routing_v2.cpp`,
 *        adapted to:
 *          1. Live in `namespace MoeInitRoutingQuantV2` alongside the existing
 *             quant variant, so it can share the unified sort / expert-token /
 *             srcToDst / gather helpers (all of which are in that namespace).
 *          2. Consume `InnerMoeInitRoutingV2TilingData*` (the type produced
 *             by the unified host tiling when `info.isQuantRouting == 0`).
 *          3. Match the tilingKey scheme produced by
 *             `MoeInitRoutingQuantV2TilingBase::GetTilingKey()` with
 *             `quantMode == 0`, namely:
 *               20000 — fullload (high performance)
 *               10000 — one-core sort, dropless
 *               10010 — multi-core sort, dropless
 *               10100 — one-core sort, drop
 *               10110 — multi-core sort, drop
 */

#ifndef MOE_INIT_ROUTING_V2_HPP
#define MOE_INIT_ROUTING_V2_HPP

#include "moe_v2_sort_one_core.h"
#include "moe_v2_sort_multi_core.h"
#include "moe_v2_mrgsort_out.h"
#include "moe_v2_mrgsort.h"
#include "moe_v2_expert_token_out.h"
#include "moe_v2_src_to_dst_op.h"
#include "moe_v2_src_to_dst_with_capacity.h"
#include "moe_v2_init_routing_fullload.h"
#include "moe_v2_gather_out.h"

using namespace AscendC;
using namespace MoeInitRoutingQuantV2;
using namespace optiling;

template <class DTYPE_X = bfloat16_t>
__aicore__ inline void moe_init_routing_v2(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace,
    const InnerMoeInitRoutingV2TilingData* tilingData, uint64_t tilingKey)
{
    if (g_coreType == AIC) {
        return;
    }
    if (workspace == nullptr) {
        return;
    }

    // ----------------------------------------------------------------
    // 1. Full-load fast path (small inputs that fit entirely in UB).
    //    tilingKey 20000 is produced by `GetTilingKey()` when
    //    `IsFullLoad() == true` (quant base, quantMode=0).
    // ----------------------------------------------------------------
    if (tilingKey == 20000) {
        TPipe sortPipe;
        MoeV2FullLoad<DTYPE_X> op;
        op.Init(x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum,
                workspace, tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
        return;
    }

    // ----------------------------------------------------------------
    // 2. Generic sort path.
    //    `*0` -> single-core sort, `*10` -> multi-core sort (the
    //    `TILING_KEY_SORT_BASE = 10` bit).
    // ----------------------------------------------------------------
    if (tilingKey == 10000 || tilingKey == 10100) {
        TPipe sortPipe;
        MoeV2SortOneCore op;
        op.Init<InnerMoeInitRoutingV2TilingData>(expertIdx, expertTokensCountOrCumsum,
                                                 expertTokensBeforeCapacity, workspace,
                                                 tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    } else if (tilingKey == 10010 || tilingKey == 10110) {
        TPipe sortPipe;
        MoeV2SortMultiCore op;
        op.Init<InnerMoeInitRoutingV2TilingData>(expertIdx, expertTokensCountOrCumsum,
                                                 expertTokensBeforeCapacity, workspace,
                                                 tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    }

    // ----------------------------------------------------------------
    // 3. ExpertTokenOut + (SrcToDstOp | SrcToDstWithCapacity).
    //    The `*100` bit (`TILING_KEY_DROP_MODE_BASE = 100`) selects the
    //    drop variant.
    // ----------------------------------------------------------------
    if (tilingKey == 10000 || tilingKey == 10010) {
        // dropless
        if (tilingData->expertTokensCountOrCumsumFlag != EXERPT_TOKENS_NONE) {
            TPipe expertTokenOutPipe;
            MoeV2ExpertTokenOut expertTokenOutOp;
            expertTokenOutOp.Init<InnerMoeInitRoutingV2TilingData>(
                expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                expandedRowIdx, workspace, tilingData, &expertTokenOutPipe);
            expertTokenOutOp.Process();
            expertTokenOutPipe.Destroy();
        }
        TPipe srcToDstPipe;
        MoeV2SrcToDstOp srcToDstOp;
        srcToDstOp.Init<InnerMoeInitRoutingV2TilingData>(expandedRowIdx, workspace, tilingData, &srcToDstPipe);
        srcToDstOp.Process();
        srcToDstPipe.Destroy();
    } else if (tilingKey == 10100 || tilingKey == 10110) {
        // drop
        TPipe expertTokenOutPipe;
        MoeV2ExpertTokenOut expertTokenOutOp;
        expertTokenOutOp.Init<InnerMoeInitRoutingV2TilingData>(
            expertTokensCountOrCumsum, expertTokensBeforeCapacity,
            expandedRowIdx, workspace, tilingData, &expertTokenOutPipe);
        expertTokenOutOp.Process();
        expertTokenOutPipe.Destroy();

        TPipe srcToDstPipe;
        MoeV2SrcToDstWithCapacity<DTYPE_X, InnerMoeInitRoutingV2TilingData> srcToDstWithCapacityOp;
        srcToDstWithCapacityOp.Init(expandedRowIdx, expandedX, workspace, tilingData, &srcToDstPipe);
        srcToDstWithCapacityOp.Process();
        srcToDstPipe.Destroy();
    }

    // ----------------------------------------------------------------
    // 4. Gather (NON-quant — emits the X dtype directly, no per-token
    //    int8 quantization). This is the key behavioral difference vs.
    //    the quant variant which uses MoeV2GatherQuant / MoeV2GatherDynamicQuant.
    // ----------------------------------------------------------------
    TPipe gatherPipe;
    MoeV2GatherOut<DTYPE_X> gatherOp;
    gatherOp.Init(x, expandedRowIdx, expandedX, workspace, tilingData, &gatherPipe);
    gatherOp.Process();
    gatherPipe.Destroy();
}

#endif  // MOE_INIT_ROUTING_V2_HPP
