/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t GLI_QUERY_INDEX = 0;
constexpr uint32_t ATTR_TOPK = 0;
constexpr uint32_t ATTR_CMP_RATIO = 1;
constexpr uint32_t ATTR_LAYOUT_Q = 2;
constexpr uint32_t ATTR_LAYOUT_K = 3;
constexpr uint32_t ATTR_SPARSE_MODE = 4;
constexpr uint32_t ATTR_ACT_TYPE = 5;
constexpr uint32_t ATTR_HAS_WEIGHTED_SUM = 6;
constexpr uint32_t ATTR_EXPAND_INDICES = 7;
constexpr uint32_t ATTR_APPEND_TAIL = 8;
constexpr uint32_t ATTR_QUANT_MODE = 9;
constexpr uint32_t ATTR_RETURN_VALUE = 10;
constexpr uint32_t ATTR_SELECT_MODE = 11;
constexpr uint32_t ATTR_KEY_REDUCE_MODE = 12;
constexpr uint32_t GLI_KEY_INDEX = 1;
constexpr uint32_t GLI_BLOCK_TABLE_INDEX = 8;
constexpr int64_t SCORE_BLOCK_ALIGN = 16;

constexpr int64_t TOPK_MIN = 1;
constexpr int64_t TOPK_MAX_CONTINUOUS = 2048;
constexpr int64_t CMP_RATIO_MIN = 1;
constexpr int64_t CMP_RATIO_MAX = 128;

static bool IsTopkValid(int64_t topk)
{
    return (topk >= TOPK_MIN && topk <= TOPK_MAX_CONTINUOUS) || topk == 3072 || topk == 4096 || topk == 5120 ||
           topk == 6144 || topk == 7168 || topk == 8192;
}

static int64_t IndicesLastDim(int64_t topk, int64_t cmpRatio, bool expand, bool appendTail)
{
    if (!expand) {
        return topk;
    }
    int64_t expanded = topk * cmpRatio;
    return appendTail ? (expanded + cmpRatio - 1) : expanded;
}

static int64_t RoundUpTo(int64_t value, int64_t align)
{
    return (align == 0) ? value : ((value + align - 1) / align * align);
}

static ge::graphStatus InferShapeGenericLightningIndexer(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GenericLightningIndexer", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(GLI_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    gert::Shape *sparseIndicesShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseIndicesShape);
    gert::Shape *sparseValuesShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseValuesShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t *topkPtr = attrs->GetInt(ATTR_TOPK);
    const int64_t *cmpRatioPtr = attrs->GetInt(ATTR_CMP_RATIO);
    const char *layoutQPtr = attrs->GetAttrPointer<char>(ATTR_LAYOUT_Q);
    const char *layoutKPtr = attrs->GetAttrPointer<char>(ATTR_LAYOUT_K);
    const int64_t *sparseModePtr = attrs->GetInt(ATTR_SPARSE_MODE);
    const int64_t *actTypePtr = attrs->GetInt(ATTR_ACT_TYPE);
    const bool *hasWeightedPtr = attrs->GetAttrPointer<bool>(ATTR_HAS_WEIGHTED_SUM);
    const bool *expandPtr = attrs->GetAttrPointer<bool>(ATTR_EXPAND_INDICES);
    const bool *appendPtr = attrs->GetAttrPointer<bool>(ATTR_APPEND_TAIL);
    const int64_t *quantModePtr = attrs->GetInt(ATTR_QUANT_MODE);
    const bool *returnValuePtr = attrs->GetAttrPointer<bool>(ATTR_RETURN_VALUE);
    const int64_t *selectModePtr = attrs->GetInt(ATTR_SELECT_MODE);
    const int64_t *keyReducePtr = attrs->GetInt(ATTR_KEY_REDUCE_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmpRatioPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutQPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutKPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, actTypePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, hasWeightedPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, appendPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyReducePtr);

    const int64_t topk = *topkPtr;
    const int64_t cmpRatio = *cmpRatioPtr;
    const std::string layoutQ(layoutQPtr);
    const std::string layoutK(layoutKPtr);
    const int64_t sparseMode = *sparseModePtr;
    const int64_t actType = *actTypePtr;
    const bool hasWeighted = *hasWeightedPtr;
    const bool expand = *expandPtr;
    const bool appendTail = *appendPtr;
    const int64_t quantMode = *quantModePtr;
    const bool returnValue = *returnValuePtr;
    const int64_t selectMode = *selectModePtr;
    const int64_t keyReduceMode = *keyReducePtr;

    OP_CHECK_IF(layoutQ != "BSND" && layoutQ != "TND",
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("GenericLightningIndexer", "layout_q", layoutQ.c_str(),
                                                      "layout_q should be BSND or TND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(layoutK != "BSND" && layoutK != "TND" && layoutK != "PA_BBND" && layoutK != "PA_BSND" &&
                    layoutK != "BBND" && layoutK != "BNBD",
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("GenericLightningIndexer", "layout_k", layoutK.c_str(),
                                                      "layout_k should be BSND, TND, PA_BBND, PA_BSND, BBND or BNBD"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        sparseMode != 0 && sparseMode != 3 && sparseMode != 4,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("GenericLightningIndexer", "sparse_mode",
                                              std::to_string(sparseMode).c_str(), "sparse_mode should be 0, 3 or 4"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(actType != 0 && actType != 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("GenericLightningIndexer", "act_type",
                                                      std::to_string(actType).c_str(), "act_type should be 0 or 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsTopkValid(topk),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    "GenericLightningIndexer", "topk", std::to_string(topk).c_str(),
                    "topk must be in [1, 2048] or one of {3072, 4096, 5120, 6144, 7168, 8192}"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        cmpRatio < CMP_RATIO_MIN || cmpRatio > CMP_RATIO_MAX,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("GenericLightningIndexer", "cmp_ratio", std::to_string(cmpRatio).c_str(),
                                              "cmp_ratio must be in [1, 128]"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(appendTail && !expand,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("GenericLightningIndexer", "append_tail",
                                                         "append_tail requires expand_indices=true"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        quantMode != -1 && quantMode != 0 && quantMode != 1 && quantMode != 2 && quantMode != 3 && quantMode != 4 &&
            quantMode != 5,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("GenericLightningIndexer", "quant_mode",
                                              std::to_string(quantMode).c_str(), "quant_mode should be -1 or 0..5"),
        return ge::GRAPH_FAILED);

    const bool isMsa = (selectMode == 1 || keyReduceMode == 1 || actType == 1);
    if (isMsa) {
        OP_CHECK_IF(selectMode != 1 || keyReduceMode != 1 || actType != 1,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        "GenericLightningIndexer", "msa combo",
                        "MSA requires act_type=1, key_reduce_mode=1 and select_mode=1 together"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hasWeighted,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("GenericLightningIndexer", "has_weighted_sum",
                                                             "MSA requires has_weighted_sum=false"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(expand || appendTail,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("GenericLightningIndexer", "expand_indices",
                                                             "MSA forbids expand_indices/append_tail"),
                    return ge::GRAPH_FAILED);
    }

    (void)hasWeighted;
    if (isMsa) {
        const gert::Shape *keyShape = context->GetInputShape(GLI_KEY_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
        int64_t nBlocks = 1;
        if (layoutK == "PA_BBND" || layoutK == "PA_BSND" || layoutK == "BBND" || layoutK == "BNBD") {
            const gert::Shape *btShape = context->GetOptionalInputShape(GLI_BLOCK_TABLE_INDEX);
            OP_CHECK_IF(btShape == nullptr, OP_LOGE(context, "PA layout requires block_table for MSA"),
                        return ge::GRAPH_FAILED);
            nBlocks = btShape->GetDim(1);
        } else if (layoutK == "TND") {
            nBlocks = (keyShape->GetDim(0) + cmpRatio - 1) / cmpRatio;
        } else {
            nBlocks = (keyShape->GetDim(1) + cmpRatio - 1) / cmpRatio;
        }
        if (nBlocks <= 0) {
            nBlocks = 1;
        }
        nBlocks = RoundUpTo(nBlocks, SCORE_BLOCK_ALIGN);
        sparseIndicesShape->SetDimNum(1);
        sparseIndicesShape->SetDim(0, 0);
        if (layoutQ == "BSND") {
            OP_CHECK_IF(queryShape->GetDimNum() != 4,
                        OP_LOGE(context, "Layout BSND, queryDims (%zu) must be 4!", queryShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
            sparseValuesShape->SetDimNum(4);
            sparseValuesShape->SetDim(0, queryShape->GetDim(0));
            sparseValuesShape->SetDim(1, queryShape->GetDim(1));
            sparseValuesShape->SetDim(2, queryShape->GetDim(2));
            sparseValuesShape->SetDim(3, nBlocks);
        } else {
            OP_CHECK_IF(queryShape->GetDimNum() != 3,
                        OP_LOGE(context, "Layout TND, queryDims (%zu) must be 3!", queryShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
            // 对齐 MsaIndexScore：TND 输出 [N1, T1, nOut]
            sparseValuesShape->SetDimNum(3);
            sparseValuesShape->SetDim(0, queryShape->GetDim(1));
            sparseValuesShape->SetDim(1, queryShape->GetDim(0));
            sparseValuesShape->SetDim(2, nBlocks);
        }
        OP_LOGI(context->GetNodeName(), "GenericLightningIndexer InferShape MSA nBlocks=%ld", nBlocks);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t outLastDim = IndicesLastDim(topk, cmpRatio, expand, appendTail);

    if (layoutQ == "BSND") {
        OP_CHECK_IF(queryShape->GetDimNum() != 4,
                    OP_LOGE(context, "Layout BSND, queryDims (%zu) must be 4!", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        sparseIndicesShape->SetDimNum(3);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0));
        sparseIndicesShape->SetDim(1, queryShape->GetDim(1));
        sparseIndicesShape->SetDim(2, outLastDim);
    } else {
        OP_CHECK_IF(queryShape->GetDimNum() != 3,
                    OP_LOGE(context, "Layout TND, queryDims (%zu) must be 3!", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        sparseIndicesShape->SetDimNum(2);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0));
        sparseIndicesShape->SetDim(1, outLastDim);
    }

    if (returnValue) {
        if (layoutQ == "BSND") {
            sparseValuesShape->SetDimNum(3);
            sparseValuesShape->SetDim(0, queryShape->GetDim(0));
            sparseValuesShape->SetDim(1, queryShape->GetDim(1));
            sparseValuesShape->SetDim(2, topk);
        } else {
            sparseValuesShape->SetDimNum(2);
            sparseValuesShape->SetDim(0, queryShape->GetDim(0));
            sparseValuesShape->SetDim(1, topk);
        }
    } else {
        sparseValuesShape->SetDimNum(1);
        sparseValuesShape->SetDim(0, 0);
    }

    OP_LOGI(context->GetNodeName(), "GenericLightningIndexer InferShape end. L_out=%ld", outLastDim);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGenericLightningIndexer(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GenericLightningIndexer", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GenericLightningIndexer)
    .InferShape(InferShapeGenericLightningIndexer)
    .InferDataType(InferDataTypeGenericLightningIndexer);
} // namespace ops
