/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file fused_infer_attention_score_tiling_arch38.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling_arch38.h"
#include "../../../incre_flash_attention/op_host/incre_flash_attention_tiling_arch38.h"
#include "../../../prompt_flash_attention/op_host/prompt_flash_attention_tiling_arch38.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "../../op_kernel/fused_infer_attention_score_tilingkey.h"

using namespace ge;
using namespace AscendC;
using namespace optiling::arch38;
namespace optiling {
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_10000000000000090, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000090, FlashAttentionScoreSimplifiedTilingData)

REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002122000000021012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001121000000021012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002312000000021012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001311000000021012, FlashAttentionScoreSimplifiedTilingData)

REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002312000040001212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002312000040021212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001311000000001212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001311000000021212, FlashAttentionScoreSimplifiedTilingData)

// Inputs Index
constexpr uint32_t QUERY_DIM_0 = 0;
constexpr uint32_t QUERY_DIM_1 = 1;
constexpr uint32_t QUERY_DIM_2 = 2;
constexpr uint32_t QUERY_DIM_3 = 3;
constexpr uint32_t QUERY_DIM_4 = 4;
constexpr uint32_t VALUE_DIM_2 = 2;
constexpr uint32_t VALUE_DIM_3 = 3;
constexpr uint32_t KV_DIM_0 = 0;
constexpr uint32_t KV_DIM_2 = 2;
constexpr uint32_t KV_DIM_3 = 3;
constexpr uint32_t OUT_DIM_1 = 1;
constexpr uint32_t OUT_DIM_2 = 2;
constexpr uint32_t OUT_DIM_3 = 3;
// Output Index
constexpr uint32_t DLIMIT = 512;
constexpr uint32_t BLIMIT = 65536;

constexpr uint32_t QK_D_PFA_MLA = 192;
constexpr uint32_t V_D_PFA_MLA = 128;

constexpr uint32_t SHAPE_INDEX_TWO = 2;

constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t ATTENTION_OUT_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t PSE_SHIFT_INDEX = 3;
constexpr uint32_t ATTEN_MASK_INDEX = 4;
constexpr uint32_t FROM_FUSED_FLAG = 71;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX = 6;
constexpr uint32_t ANTIQUANT_SCALE_INDEX = 12;
constexpr uint32_t ANTIQUANT_OFFSET_INDEX = 13;
constexpr uint32_t QUANT_SCALE1_INDEX = 8;
constexpr uint32_t QUANT_SCALE2_INDEX = 10;
constexpr uint32_t QUANT_OFFSET2_INDEX = 11;
constexpr uint32_t ATTR_N_INDEX = 0;
constexpr uint32_t ATTR_SCALE_INDEX = 1;
constexpr uint32_t ATTR_PRE_TOKEN_INDEX = 2;
constexpr uint32_t ATTR_NEXT_TOKEN_INDEX = 3;
constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 4;
constexpr uint32_t ATTR_NUM_KV_HEADS_INDEX = 5;

constexpr uint32_t DEQUANT_SCALE1_INDEX = 7;
constexpr uint32_t DEQUANT_SCALE2_INDEX = 9;
constexpr uint32_t BLOCK_TABLE_INDEX = 14;
constexpr uint32_t QUERY_PADDING_SIZE_INDEX = 15;
constexpr uint32_t KV_PADDING_SIZE_INDEX = 16;
constexpr uint32_t KEY_ANTIQUANT_SCALE_INDEX = 17;
constexpr uint32_t KEY_ANTIQUANT_OFFSET_INDEX = 18;
constexpr uint32_t VALUE_ANTIQUANT_SCALE_INDEX = 19;
constexpr uint32_t VALUE_ANTIQUANT_OFFSET_INDEX = 20;
constexpr uint32_t KEY_SHARED_PREFIX_INDEX = 21;
constexpr uint32_t VALUE_SHARED_PREFIX_INDEX = 22;
constexpr uint32_t ACTUAL_SHARED_PREFIX_LEN_INDEX = 23;
constexpr uint32_t QUERY_ROPE_INDEX = 24;
constexpr uint32_t KEY_ROPE_INDEX = 25;
constexpr uint32_t DEQUANT_SCALE_QUERY_INDEX = 27;
constexpr uint32_t LEARNABLE_SINK_INDEX = 28;
constexpr uint32_t Q_START_IDX_INDEX = 29;
constexpr uint32_t KV_START_IDX_INDEX = 30;

constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 6;
constexpr uint32_t ATTR_INNER_PRECISE_INDEX = 7;
constexpr uint32_t ATTR_BLOCK_SIZE_INDEX = 8;
constexpr uint32_t ANTIQUANT_MODE_INDEX = 9;
constexpr uint32_t SOFTMAX_LSE_FLAG_INDEX = 10;
constexpr uint32_t KEY_ANTIQUANT_MODE_INDEX = 11;
constexpr uint32_t VALUE_ANTIQUANT_MODE_INDEX = 12;
constexpr uint32_t QUERY_QUANT_MODE_INDEX = 13;
constexpr uint32_t PSE_TYPE_INDEX = 14;

constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

static bool GetMaxWorkspaceFlag(gert::TilingContext &context)
{
    bool res = (context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX) != nullptr &&
        context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX)->GetData<int64_t>() == nullptr) ||
        (context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX) != nullptr &&
        context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX)->GetData<int64_t>() == nullptr);
    return res;
}

ge::graphStatus TilingFusedInferAttentionScoreArch38(gert::TilingContext *context)
{
    FusedInferAttentionScoreTilingArch38 FIATilingArch38(context);
    auto ret = FIATilingArch38.DoTiling(nullptr);
    return ret;
}

ge::graphStatus FusedInferAttentionScoreTilingArch38::DoOpTiling()
{
    if (context_ == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    bool isMaxWorkspace = GetMaxWorkspaceFlag(*context_);

    return ge::GRAPH_FAILED;
}
} // namespace optiling