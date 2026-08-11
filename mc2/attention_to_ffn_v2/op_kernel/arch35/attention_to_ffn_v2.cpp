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
 * \file attention_to_ffn_v2.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#if __has_include("../../attention_to_ffn/attention_to_ffn.h")
#include "../../attention_to_ffn/attention_to_ffn.h"
#else
#include "../../../attention_to_ffn/op_kernel/attention_to_ffn.h"
#endif
#include "../attention_to_ffn_v2_tiling.h"
#include "../attention_to_ffn_v2_tiling_key.h"
#include <cstddef>

using namespace AscendC;
using namespace AttentionToFFNImpl;
using namespace Mc2Tiling;

static_assert(offsetof(AttentionToFfnV2Info, quantDtype) == sizeof(AttentionToFFNInfo),
              "AttentionToFFN V1 tiling info must remain a prefix of V2");
static_assert(offsetof(AttentionToFfnV2TilingData, attentionToFfnV2Info) ==
                  offsetof(AttentionToFFNTilingData, attentionToFFNInfo),
              "AttentionToFFN V1/V2 tiling info offsets must match");

template <uint8_t QuantMode, uint8_t QuantDtype, bool ScaleMode, bool isSync, bool isActiveMask, uint8_t ArchTag>
__global__ __aicore__ void attention_to_ffn_v2(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR sessionId, GM_ADDR microBatchId,
                                            GM_ADDR layerId, GM_ADDR expertIds, GM_ADDR expertRankTable,
                                            GM_ADDR scales, GM_ADDR active_mask, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AttentionToFfnV2TilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A5", AttentionToFfnV2TilingData);
    TPipe pipe;

    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        constexpr bool isQuant = QuantMode != ATTN_FFN_TILINGKEY_NO_QUANT;
        GET_TILING_DATA_WITH_STRUCT(AttentionToFfnV2TilingData, tilingData, tilingGM);
        const auto *v1TilingData = reinterpret_cast<const AttentionToFFNTilingData *>(&tilingData);
        AttentionToFFN<DTYPE_X, isQuant, isSync, isActiveMask, AttentionToFFNContextType::MC2_CONTEXT> op;
        op.Init(mc2Context, x, sessionId, microBatchId, layerId, expertIds, expertRankTable, scales, active_mask,
                workspaceGM, &pipe, v1TilingData);
        op.Process();
    }
}
