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

    return ge::GRAPH_FAILED

    // auto tempQ = context_->GetInputShape(QUERY_INDEX);
    // auto tempV = context_->GetDynamicInputShape(VALUE_INDEX, 0);
    // auto tempOut = context_->GetOutputShape(ATTENTION_OUT_INDEX);
    // auto tempLse = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
    // bool qOutEmptyTensor = false;
    // uint32_t queryD = 1U;
    // uint32_t valueD = 1U;
    // OP_CHECK_IF((tempQ == nullptr),
    //     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "Query input is null pointer!"), return ge::GRAPH_FAILED);
    // OP_CHECK_IF((tempV == nullptr),
    //     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "Value input is null pointer!"), return ge::GRAPH_FAILED);
    // OP_CHECK_IF((tempOut == nullptr),
    //     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "AttentionOut is null pointer!"), return ge::GRAPH_FAILED);
    // auto attrs = context_->GetAttrs();
    // OP_CHECK_IF((tempQ->GetStorageShape().GetShapeSize() == 0 && tempOut->GetStorageShape().GetShapeSize() != 0) ||
    //     (tempQ->GetStorageShape().GetShapeSize() != 0 && tempOut->GetStorageShape().GetShapeSize() == 0),
    //     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //     "query shape size is %llu byte, but attention Out shape size is %llu byte, they cannot be empty while the "
    //     "other is not",
    //     tempQ->GetStorageShape().GetShapeSize(), tempOut->GetStorageShape().GetShapeSize()),
    //     return ge::GRAPH_FAILED);
    // if (tempQ->GetStorageShape().GetShapeSize() == 0 && tempOut->GetStorageShape().GetShapeSize() == 0) {
    //     qOutEmptyTensor = true;
    // }

    // OP_CHECK_IF((tempQ->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
    //     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "Get the shape size of Query failed!"),
    //     return ge::GRAPH_FAILED);
    // OP_CHECK_IF(attrs == nullptr,
    //     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "Attributes returned from GetAttrs() is a nullptr!"),
    //     return ge::GRAPH_FAILED);

    // uint32_t tempN = *attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX);
    // uint32_t tempKVN = *attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX);
    // OP_CHECK_IF(tempN == 0, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "Q numhead is 0!"),
    //     return ge::GRAPH_FAILED);
    // if (tempKVN == 0U) {
    //     tempKVN = tempN;
    // }
    // const string inputLayoutStr = string(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    // int64_t s = 0;
    // int64_t b = tempQ->GetStorageShape().GetDim(QUERY_DIM_0);
    // int64_t t = 0;
    // bool lseFlag = *attrs->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    // bool usingIFA = false;
    // if (inputLayoutStr == "BNSD" || inputLayoutStr == "BNSD_BSND") {
    //     s = tempQ->GetStorageShape().GetDim(QUERY_DIM_2);
    // } else if (inputLayoutStr == "TND" || inputLayoutStr == "TND_NTD") {
    //     if (isMaxWorkspace) {
    //         t = tempQ->GetStorageShape().GetDim(QUERY_DIM_0);
    //         s = tempQ->GetStorageShape().GetDim(QUERY_DIM_0);
    //     } else {
    //         const gert::Tensor *actualSeqLength = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    //         int64_t actSeqLenDims = (actualSeqLength != nullptr) ? actualSeqLength->GetShapeSize() : 0;
    //         OP_CHECK_IF(((actualSeqLength == nullptr) || (actSeqLenDims == 0) ||
    //             (actualSeqLength->GetData<int64_t>() == nullptr)),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "When layout is TND/TND_NTD, actualSequenceLengthQ is required!"),
    //             return ge::GRAPH_FAILED);
    //         s = actualSeqLength->GetData<int64_t>()[0];
    //         for (int i = 1; i < actualSeqLength->GetShapeSize(); ++i) {
    //             s = std::max(s, actualSeqLength->GetData<int64_t>()[i] - actualSeqLength->GetData<int64_t>()[i - 1]);
    //         }
    //         t = actualSeqLength->GetData<int64_t>()[actualSeqLength->GetShapeSize() - 1];
    //     }
    // } else if (inputLayoutStr == "NTD" || inputLayoutStr == "NTD_TND") {
    //     if (isMaxWorkspace) {
    //         t = tempQ->GetStorageShape().GetDim(QUERY_DIM_1);
    //         s = tempQ->GetStorageShape().GetDim(QUERY_DIM_1);
    //     } else {
    //         const gert::Tensor *actualSeqLength = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    //         int64_t actSeqLenDims = (actualSeqLength != nullptr) ? actualSeqLength->GetShapeSize() : 0;
    //         OP_CHECK_IF(((actualSeqLength == nullptr) || (actSeqLenDims == 0) ||
    //             (actualSeqLength->GetData<int64_t>() == nullptr)),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "When layout is NTD/NTD_TND, actualSequenceLengthQ is required!"),
    //             return ge::GRAPH_FAILED);
    //         s = actualSeqLength->GetData<int64_t>()[0];
    //         for (int i = 1; i < actualSeqLength->GetShapeSize(); ++i) {
    //             s = std::max(s, actualSeqLength->GetData<int64_t>()[i] - actualSeqLength->GetData<int64_t>()[i - 1]);
    //         }
    //         t = actualSeqLength->GetData<int64_t>()[actualSeqLength->GetShapeSize() - 1];
    //     }
    // } else {
    //     s = tempQ->GetStorageShape().GetDim(1);
    // }
    // if (inputLayoutStr == "NSD") { // 当前已没有NSD, 回主线后在FIA tiling Arch38内删除
    //     b = 1;
    //     OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "input shape dim should be 3!"),
    //         return ge::GRAPH_FAILED);
    //     queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_2);
    //     valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_2);
    //     OP_CHECK_IF(((queryD == valueD) && (tempQ->GetStorageShape() != tempOut->GetStorageShape())),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "Layout is NSD and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, %ld]!",
    //         tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
    //         tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempOut->GetStorageShape().GetDim(0),
    //         tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(OUT_DIM_2)),
    //         return ge::GRAPH_FAILED);
    // } else if (inputLayoutStr == "TND" || inputLayoutStr == "TND_NTD") {
    //     OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "The current layout is %s, input query shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
    //         tempQ->GetStorageShape().GetDimNum()),
    //         return ge::GRAPH_FAILED);
    //     OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_3),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "The current layout is %s, attention out shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
    //         tempOut->GetStorageShape().GetDimNum()),
    //         return ge::GRAPH_FAILED);
    //     queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_2);
    //     valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_2);
    //     if (inputLayoutStr == "TND") {
    //         OP_CHECK_IF(((queryD == valueD) && (tempQ->GetStorageShape() != tempOut->GetStorageShape())),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "Layout is TND and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, "
    //             "%ld]!",
    //             tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
    //             tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempOut->GetStorageShape().GetDim(0),
    //             tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(OUT_DIM_2)),
    //             return ge::GRAPH_FAILED);
    //     } else {
    //         OP_CHECK_IF(
    //             ((queryD == valueD) && ((tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(1)) ||
    //             (tempQ->GetStorageShape().GetDim(1) != tempOut->GetStorageShape().GetDim(0)) ||
    //             (tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != tempOut->GetStorageShape().GetDim(QUERY_DIM_2)))),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "Layout is TND_NTD and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, "
    //             "%ld, %ld]!",
    //             tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
    //             tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempOut->GetStorageShape().GetDim(0),
    //             tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(OUT_DIM_2)),
    //             return ge::GRAPH_FAILED);
    //     }
    // } else if (inputLayoutStr == "NTD" || inputLayoutStr == "NTD_TND") {
    //     OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "The current layout is %s, input query shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
    //         tempQ->GetStorageShape().GetDimNum()),
    //         return ge::GRAPH_FAILED);
    //     OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_3),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "The current layout is %s, attention out shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
    //         tempOut->GetStorageShape().GetDimNum()),
    //         return ge::GRAPH_FAILED);
    //     queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_2);
    //     valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_2);
    //     if (inputLayoutStr == "NTD") {
    //         OP_CHECK_IF(((queryD == valueD) && (tempQ->GetStorageShape() != tempOut->GetStorageShape())),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "Layout is NTD and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, "
    //             "%ld]!",
    //             tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
    //             tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempOut->GetStorageShape().GetDim(0),
    //             tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(OUT_DIM_2)),
    //             return ge::GRAPH_FAILED);
    //     } else if (inputLayoutStr == "NTD_TND") {
    //         OP_CHECK_IF(
    //             ((queryD == valueD) && ((tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(1)) ||
    //             (tempQ->GetStorageShape().GetDim(1) != tempOut->GetStorageShape().GetDim(0)) ||
    //             (tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != tempOut->GetStorageShape().GetDim(OUT_DIM_2)))),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "Layout is NTD_TND and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, "
    //             "%ld, %ld]!",
    //             tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
    //             tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempOut->GetStorageShape().GetDim(0),
    //             tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(OUT_DIM_2)),
    //             return ge::GRAPH_FAILED);
    //     }
    // } else if (inputLayoutStr == "BSH") {
    //     OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "The current layout is %s, input query shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
    //         tempQ->GetStorageShape().GetDimNum()),
    //         return ge::GRAPH_FAILED);
    //     OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_3),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "The current layout is %s, attention out shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
    //         tempOut->GetStorageShape().GetDimNum()),
    //         return ge::GRAPH_FAILED);
    //     queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_2) / tempN;
    //     valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_2) / tempKVN;
    //     OP_CHECK_IF(((queryD == valueD) && (tempQ->GetStorageShape() != tempOut->GetStorageShape())),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "Layout is BSH and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, %ld]!",
    //         tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
    //         tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempOut->GetStorageShape().GetDim(0),
    //         tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(OUT_DIM_2)),
    //         return ge::GRAPH_FAILED);
    // } else if (inputLayoutStr == "BNSD_BSND" || inputLayoutStr == "BNSD" || inputLayoutStr == "BSND" ||
    //     inputLayoutStr == "BSND_BNSD") {
    //     OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_4),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "The current layout is %s, input query shape dim(%zu) should be 4!", inputLayoutStr.c_str(),
    //         tempQ->GetStorageShape().GetDimNum()),
    //         return ge::GRAPH_FAILED);
    //     OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_4),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "The current layout is %s, attention out shape dim(%zu) should be 4!", inputLayoutStr.c_str(),
    //         tempOut->GetStorageShape().GetDimNum()),
    //         return ge::GRAPH_FAILED);
    //     queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_3);
    //     valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_3);
    //     if (inputLayoutStr == "BNSD_BSND" || inputLayoutStr == "BSND_BNSD") {
    //         OP_CHECK_IF(
    //             ((queryD == valueD) && ((tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(0)) ||
    //             (tempQ->GetStorageShape().GetDim(1) != tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
    //             (tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != tempOut->GetStorageShape().GetDim(1)) ||
    //             (tempQ->GetStorageShape().GetDim(QUERY_DIM_3) != tempOut->GetStorageShape().GetDim(OUT_DIM_3)))),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "Layout is %s and Query shape size[%ld, %ld, %ld, %ld] does NOT match Attention Out shape size[%ld, "
    //             "%ld, %ld, %ld]!",
    //             inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
    //             tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempQ->GetStorageShape().GetDim(QUERY_DIM_3),
    //             tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
    //             tempOut->GetStorageShape().GetDim(OUT_DIM_2), tempOut->GetStorageShape().GetDim(OUT_DIM_3)),
    //             return ge::GRAPH_FAILED);
    //     } else if (inputLayoutStr == "BNSD") {
    //         OP_CHECK_IF(((queryD == valueD) && (tempQ->GetStorageShape() != tempOut->GetStorageShape())),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "Layout is BNSD and Query shape size[%ld, %ld, %ld, %ld] does NOT match Attention Out shape size[%ld, "
    //             "%ld, %ld, %ld]!",
    //             tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
    //             tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempQ->GetStorageShape().GetDim(QUERY_DIM_3),
    //             tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
    //             tempOut->GetStorageShape().GetDim(OUT_DIM_2), tempOut->GetStorageShape().GetDim(OUT_DIM_3)),
    //             return ge::GRAPH_FAILED);
    //     }
    // } else if (inputLayoutStr == "BSH_BNSD" || inputLayoutStr == "BSH_NBSD") {
    //     OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "The current layout is %s, input query shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
    //         tempQ->GetStorageShape().GetDimNum()),
    //         return ge::GRAPH_FAILED);
    //     OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_4),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "The current layout is %s, attention out shape dim(%zu) should be 4!", inputLayoutStr.c_str(),
    //         tempOut->GetStorageShape().GetDimNum()),
    //         return ge::GRAPH_FAILED);
    //     queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_2) / tempN;
    //     valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_2) / tempKVN;
    //     if (inputLayoutStr == "BSH_BNSD") {
    //         OP_CHECK_IF(((queryD == valueD) &&
    //             ((tempQ->GetStorageShape().GetDim(QUERY_DIM_0) != tempOut->GetStorageShape().GetDim(0)) ||
    //             (tempQ->GetStorageShape().GetDim(QUERY_DIM_1) != tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
    //             queryD != tempOut->GetStorageShape().GetDim(OUT_DIM_3)) ||
    //             tempN != tempOut->GetStorageShape().GetDim(OUT_DIM_1)),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "Layout is %s and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, "
    //             "%ld, %ld]!",
    //             inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDim(QUERY_DIM_0),
    //             tempQ->GetStorageShape().GetDim(QUERY_DIM_1), tempQ->GetStorageShape().GetDim(QUERY_DIM_2),
    //             tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
    //             tempOut->GetStorageShape().GetDim(OUT_DIM_2), tempOut->GetStorageShape().GetDim(OUT_DIM_3)),
    //             return ge::GRAPH_FAILED);
    //     } else if (inputLayoutStr == "BSH_NBSD") {
    //         OP_CHECK_IF(((queryD == valueD) &&
    //             ((tempQ->GetStorageShape().GetDim(QUERY_DIM_0) != tempOut->GetStorageShape().GetDim(OUT_DIM_1)) ||
    //             (tempQ->GetStorageShape().GetDim(QUERY_DIM_1) != tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
    //             queryD != tempOut->GetStorageShape().GetDim(OUT_DIM_3)) ||
    //             tempN != tempOut->GetStorageShape().GetDim(0)),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "Layout is %s and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, "
    //             "%ld, %ld]!",
    //             inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDim(QUERY_DIM_0),
    //             tempQ->GetStorageShape().GetDim(QUERY_DIM_1), tempQ->GetStorageShape().GetDim(QUERY_DIM_2),
    //             tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
    //             tempOut->GetStorageShape().GetDim(OUT_DIM_2), tempOut->GetStorageShape().GetDim(OUT_DIM_3)),
    //             return ge::GRAPH_FAILED);
    //     }
    // } else if (inputLayoutStr == "BSND_NBSD" || inputLayoutStr == "BNSD_NBSD") {
    //     OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_4),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "The current layout is %s, input query shape dim(%zu) should be 4!", inputLayoutStr.c_str(),
    //         tempQ->GetStorageShape().GetDimNum()),
    //         return ge::GRAPH_FAILED);
    //     OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_4),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //         "The current layout is %s, attention out shape dim(%zu) should be 4!", inputLayoutStr.c_str(),
    //         tempOut->GetStorageShape().GetDimNum()),
    //         return ge::GRAPH_FAILED);
    //     queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_3);
    //     valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_3);
    //     if (inputLayoutStr == "BSND_NBSD") {
    //         OP_CHECK_IF(((queryD == valueD) &&
    //             ((tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(OUT_DIM_1)) ||
    //             (tempQ->GetStorageShape().GetDim(QUERY_DIM_1) != tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
    //             (tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != tempOut->GetStorageShape().GetDim(0)) ||
    //             (tempQ->GetStorageShape().GetDim(QUERY_DIM_3) != tempOut->GetStorageShape().GetDim(OUT_DIM_3)))),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "Layout is %s and Query shape size[%ld, %ld, %ld, %ld] does NOT match Attention Out shape size[%ld, "
    //             "%ld, %ld, %ld]!",
    //             inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
    //             tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempQ->GetStorageShape().GetDim(QUERY_DIM_3),
    //             tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
    //             tempOut->GetStorageShape().GetDim(OUT_DIM_2), tempOut->GetStorageShape().GetDim(OUT_DIM_3)),
    //             return ge::GRAPH_FAILED);
    //     } else if (inputLayoutStr == "BNSD_NBSD") {
    //         OP_CHECK_IF(((queryD == valueD) &&
    //             ((tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(OUT_DIM_1)) ||
    //             (tempQ->GetStorageShape().GetDim(QUERY_DIM_1) != tempOut->GetStorageShape().GetDim(0)) ||
    //             (tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
    //             (tempQ->GetStorageShape().GetDim(QUERY_DIM_3) != tempOut->GetStorageShape().GetDim(OUT_DIM_3)))),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "Layout is %s and Query shape size[%ld, %ld, %ld, %ld] does NOT match Attention Out shape size[%ld, "
    //             "%ld, %ld, %ld]!",
    //             inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
    //             tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempQ->GetStorageShape().GetDim(QUERY_DIM_3),
    //             tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
    //             tempOut->GetStorageShape().GetDim(OUT_DIM_2), tempOut->GetStorageShape().GetDim(OUT_DIM_3)),
    //             return ge::GRAPH_FAILED);
    //     }
    // } else {
    //     OP_LOGE(context_->GetNodeName(),
    //         "Invalid input layout:%s. Currently only "
    //         "TND/NTD/BSH/BNSD/BSND/BSND_BNSD/BNSD_BSND/BSH_BNSD/BSND_NBSD/BNSD_NBSD/BSH_NBSD layout are supported!",
    //         inputLayoutStr.c_str());
    //     return ge::GRAPH_FAILED;
    // }
    // OP_CHECK_IF((queryD > DLIMIT),
    //     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //     "D of query should be less than or equal to 512, but d = %u!", queryD),
    //     return ge::GRAPH_FAILED);
    // bool inputOutputIsNullPtr = (context_->GetInputDesc(QUERY_INDEX) == nullptr) ||
    //     (context_->GetInputDesc(KEY_INDEX) == nullptr) || (context_->GetInputDesc(VALUE_INDEX) == nullptr) ||
    //     (context_->GetOutputDesc(ATTENTION_OUT_INDEX) == nullptr) ||
    //     (context_->GetInputShape(QUERY_INDEX) == nullptr) || (context_->GetInputShape(KEY_INDEX) == nullptr) ||
    //     (context_->GetInputShape(VALUE_INDEX) == nullptr) || (context_->GetOutputShape(ATTENTION_OUT_INDEX) == nullptr);
    // OP_CHECK_IF(inputOutputIsNullPtr,
    //     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "q, k, v or attenOut is nullptr!"),
    //     return ge::GRAPH_FAILED);
    // auto qDType = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
    // auto kDType = context_->GetInputDesc(KEY_INDEX)->GetDataType();
    // // IFA非MLA或伪量化场景走IFA模板
    // // IFA的MLA或PFA非伪量化场景走PFA模板
    // if ((qDType != kDType) ||
    //     ((s == 1) && ((inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") || (inputLayoutStr == "BSND")) &&
    //     ((context_->GetOptionalInputShape(QUERY_ROPE_INDEX) == nullptr) &&
    //     (context_->GetOptionalInputShape(KEY_ROPE_INDEX) == nullptr)))) {
    //     auto platformInfoPtr = context_->GetPlatformInfo();
    //     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    //     if (qDType == kDType) {
    //         usingIFA = false;
    //     } else {
    //         usingIFA = true;
    //     }
    // }
    // if (usingIFA) {
    //     // IFA tiling path
    //     IncreFlashAttentionContext ifaContext{};
    //     auto ret = ConvertContextToParamsIFA(*context_, ifaContext, isMaxWorkspace);
    //     if (ret != ge::GRAPH_SUCCESS) {
    //         OP_LOGE(context_->GetNodeName(), "Error occored while convert tilingContext to ifa context!");
    //         return ret;
    //     }
    //     IFATilingArch38 ifaTilingArch38(context_);
    //     ret = ifaTilingArch38.DoSubOpTiling(ifaContext);
    //     OP_CHECK_IF(ret == ge::GRAPH_FAILED,
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "failed in FIA DoSubOpTiling"),
    //         return ge::GRAPH_FAILED);
    //     return ret;
    // } else {
    //     // PFA tiling process
    //     constexpr int64_t D_ALIGN_32 = 32;
    //     constexpr int64_t D_ALIGN_16 = 16;

    //     PromptFlashAttentionTilingData pfaTilingData;
    //     PromptFlashAttentionTilingArch38 pfa_tiling(context_);
    //     ContextParamsForPFATiling contextParamsForPFATiling;
    //     PromptFlashAttentionCompileInfo tempCompileInfoPtr;

    //     OP_CHECK_IF((attrs->GetAttrPointer<uint64_t>(ANTIQUANT_MODE_INDEX) != nullptr) &&
    //         (*attrs->GetAttrPointer<uint64_t>(ANTIQUANT_MODE_INDEX) != 0),
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "antiquant_mode is not supported!"),
    //         return ge::GRAPH_FAILED);

    //     auto platformInfoPtr = context_->GetPlatformInfo();
    //     OP_CHECK_IF(platformInfoPtr == nullptr,
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "platformInfoPtr is null!"), return ge::GRAPH_FAILED);
    //     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    //     tempCompileInfoPtr.aivNum = ascendcPlatform.GetCoreNumAiv();
    //     tempCompileInfoPtr.aicNum = ascendcPlatform.GetCoreNumAic();
    //     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tempCompileInfoPtr.ubSize);
    //     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, tempCompileInfoPtr.l1Size);
    //     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, tempCompileInfoPtr.l0CSize);
    //     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, tempCompileInfoPtr.l0ASize);
    //     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, tempCompileInfoPtr.l0BSize);
    //     tempCompileInfoPtr.socShortName = ascendcPlatform.GetSocVersion();
    //     if (tempCompileInfoPtr.socShortName == platform_ascendc::SocVersion::ASCEND310P) {
    //         // sys workspace size default value
    //         tempCompileInfoPtr.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    //     } else {
    //         tempCompileInfoPtr.defaultSysWorkspaceSize = 0;
    //     }

    //     contextParamsForPFATiling.compileInfoPtr = &tempCompileInfoPtr;
        // auto ret = ConvertContextToParamsPFA(context_, contextParamsForPFATiling, isMaxWorkspace);
    //     if (ret != ge::GRAPH_SUCCESS) {
    //         OP_LOGE(context_->GetNodeName(), "Error occored while convert tilingContext to PFA context!");
    //         return ret;
    //     }
    //     if (lseFlag != false) {
    //         if (!pfa_tiling.CheckNonEmptyShapeExceptions(contextParamsForPFATiling,
    //             contextParamsForPFATiling.lseoutputShape, "softmaxLse")) {
    //             return ge::GRAPH_FAILED;
    //         }
    //         OP_CHECK_IF(((tempLse == nullptr)),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "SoftmaxLse shape is null, but SoftmaxLseFlag is true!"),
    //             return ge::GRAPH_FAILED);

    //         if (!qOutEmptyTensor) { // q、out为空时，lse为空则不输出，不为空则输出inf，不做拦截
    //             if (inputLayoutStr == "TND" || inputLayoutStr == "TND_NTD") {
    //                 OP_CHECK_IF(((tempLse->GetStorageShape().GetDimNum() != QUERY_DIM_3)), // 3：lse shape TN1
    //                     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //                     "Layout is TND/TND_NTD SoftmaxLse shape dim should be 3, but got %zu!",
    //                     tempLse->GetStorageShape().GetDimNum()),
    //                     return ge::GRAPH_FAILED);
    //                 OP_CHECK_IF((((tempLse->GetStorageShape().GetDim(QUERY_DIM_0) != t) ||
    //                     (tempLse->GetStorageShape().GetDim(QUERY_DIM_1) !=
    //                     tempN) || // 0: the first dimension 1: the second dimension
    //                     (tempLse->GetStorageShape().GetDim(QUERY_DIM_2) != 1))), // 2: the third dimension
    //                     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //                     "Layout is TND SoftmaxLse shape size[%ld, %ld, %ld] does not match TN1[%ld, %u, 1]!",
    //                     tempLse->GetStorageShape().GetDim(QUERY_DIM_0),
    //                     tempLse->GetStorageShape().GetDim(
    //                     QUERY_DIM_1), // 0: the first dimension 1: the second dimension
    //                     tempLse->GetStorageShape().GetDim(QUERY_DIM_2), t, tempN), // 2: the third dimension
    //                     return ge::GRAPH_FAILED);
    //             } else if (inputLayoutStr == "NTD" || inputLayoutStr == "NTD_TND") {
    //                 OP_CHECK_IF(((tempLse->GetStorageShape().GetDimNum() != QUERY_DIM_3)), // 3：lse shape TN1
    //                     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //                     "Layout is NTD/NTD_TND SoftmaxLse shape dim should be 3, but got %zu!",
    //                     tempLse->GetStorageShape().GetDimNum()),
    //                     return ge::GRAPH_FAILED);
    //                 OP_CHECK_IF((((tempLse->GetStorageShape().GetDim(QUERY_DIM_1) != tempN) ||
    //                     (tempLse->GetStorageShape().GetDim(QUERY_DIM_0) !=
    //                     t) || // 0: the first dimension 1: the second dimension
    //                     (tempLse->GetStorageShape().GetDim(QUERY_DIM_2) != 1))), // 2: the third dimension
    //                     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //                     "Layout is NTD SoftmaxLse shape size[%ld, %ld, %ld] does not match TN1[%ld, %u, 1]!",
    //                     tempLse->GetStorageShape().GetDim(QUERY_DIM_0),
    //                     tempLse->GetStorageShape().GetDim(
    //                     QUERY_DIM_1), // 0: the first dimension 1: the second dimension
    //                     tempLse->GetStorageShape().GetDim(QUERY_DIM_2), t, tempN), // 2: the third dimension
    //                     return ge::GRAPH_FAILED);
    //             } else {
    //                 OP_CHECK_IF(((tempLse->GetStorageShape().GetDimNum() != QUERY_DIM_4)), // 4：lse shape BNS1
    //                     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //                     "Layout is %s, SoftmaxLse shape dim should be 4, but got %zu!", inputLayoutStr.c_str(),
    //                     tempLse->GetStorageShape().GetDimNum()),
    //                     return ge::GRAPH_FAILED);
    //                 OP_CHECK_IF((((tempLse->GetStorageShape().GetDim(QUERY_DIM_0) != b) ||
    //                     (tempLse->GetStorageShape().GetDim(QUERY_DIM_1) !=
    //                     tempN) || // 0: the first dimension 1: the second dimension
    //                     (tempLse->GetStorageShape().GetDim(QUERY_DIM_2) != s) ||
    //                     (tempLse->GetStorageShape().GetDim(QUERY_DIM_3) !=
    //                     1))), // 2: the third dimension 3: the fourth dimension
    //                     OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //                     "SoftmaxLse shape size[%ld, %ld, %ld, %ld] does not match BNS1[%ld, %u, %ld, 1]!",
    //                     tempLse->GetStorageShape().GetDim(QUERY_DIM_0),
    //                     tempLse->GetStorageShape().GetDim(
    //                     QUERY_DIM_1), // 0: the first dimension 1: the second dimension
    //                     tempLse->GetStorageShape().GetDim(QUERY_DIM_2), tempLse->GetStorageShape().GetDim(QUERY_DIM_3),
    //                     b, tempN, s), // 2: the third dimension 3: the fourth dimension
    //                     return ge::GRAPH_FAILED);
    //             }
    //         }
    //     }
    //     if (ascendcPlatform.GetCurNpuArch() != NpuArch::DAV_5102) {
    //         OP_CHECK_IF((((contextParamsForPFATiling.inputDataType == ge::DT_INT8) ||
    //             (contextParamsForPFATiling.kDataType == ge::DT_INT8) ||
    //             (contextParamsForPFATiling.outputDataType == ge::DT_INT8)) &&
    //             (queryD % D_ALIGN_32 != 0)),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "D(%u) of query should be 32 elements aligned when int8 is involved!", queryD),
    //             return ge::GRAPH_FAILED);
    //         OP_CHECK_IF((queryD % D_ALIGN_16 != 0),
    //             OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
    //             "D(%u) should be 16 elements aligned when FP16/BF16 dtype!", queryD),
    //             return ge::GRAPH_FAILED);
    //     }
    //     ret = pfa_tiling.DoSubOpTiling(pfaTilingData, contextParamsForPFATiling);
    //     OP_CHECK_IF(ret == ge::GRAPH_FAILED,
    //         OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "failed in FIA DoSubOpTiling"),
    //         return ge::GRAPH_FAILED);
    //     return ret;
    // }
}
} // namespace optiling