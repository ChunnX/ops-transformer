/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "log/error_code.h"

using namespace ge;

namespace ops {
static constexpr uint32_t FIA_LAYOUT_DIM0 = 0;
static constexpr uint32_t FIA_LAYOUT_DIM1 = 1;
static constexpr uint32_t FIA_LAYOUT_DIM2 = 2;
static constexpr uint32_t FIA_LAYOUT_DIM3 = 3;
static constexpr uint32_t FIA_LAYOUT_DIM4 = 4;
static constexpr uint32_t FIA_LAYOUT_DIM_NUMS_1 = 1;
static constexpr uint32_t FIA_LAYOUT_DIM_NUMS_3 = 3;
static constexpr uint32_t FIA_LAYOUT_DIM_NUMS_4 = 4;
static constexpr uint32_t LAYOUT_SH_DIM_NUMS = 2;
static constexpr uint32_t LAYOUT_BSH_DIM_NUMS = 3;
static constexpr uint32_t LAYOUT_NSD_DIM_NUMS = 3;
static constexpr uint32_t LAYOUT_BNSD_DIM_NUMS = 4;
static constexpr uint32_t LAYOUT_BSND_DIM_NUMS = 4;
static constexpr uint32_t LAYOUT_BNSD_BSND_DIM_NUMS = 4;
static constexpr uint32_t LAYOUT_TND_DIM_NUMS = 3;
static constexpr int32_t FIA_UNKNOWN_DIMS = -2;
static constexpr uint32_t LAYOUT_PA_BBH_DIM_NUMS = 3;
static constexpr uint32_t LAYOUT_PA_BNBD_DIM_NUMS = 4;
static constexpr uint32_t LAYOUT_PA_NZ_DIM_NUMS = 5;
static constexpr uint32_t NUM_0 = 0;
static constexpr uint32_t NUM_1 = 1;
static constexpr uint32_t FIA_QUERY_INDEX = 0;
static constexpr uint32_t FIA_VALUE_INDEX = 2;
static constexpr uint32_t FIA_DYNAMIC_VALUE_INDEX = 0;
static constexpr uint32_t FIA_BLOCK_TABLE_INDEX = 14;
static constexpr uint32_t FIA_QUANT_SCALE2_INDEX = 10;
static constexpr uint32_t FIA_ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t FIA_SOFTMAX_LSE_INDEX = 1;
static constexpr uint32_t FIA_ATTR_NUM_HEADS_INDEX = 0;
static constexpr uint32_t FIA_ATTR_NUM_KV_HEADS_INDEX = 5;
static constexpr uint32_t FIA_ATTR_INPUT_LAYOUT_INDEX = 4;
static constexpr uint32_t FIA_INPUT_ACTUAL_SEQ_LENGTHS_INDEX = 5;
static constexpr uint32_t FIA_INPUT_ACTUAL_SEQ_LENGTHS_KV_INDEX = 6;
static constexpr uint32_t FIA_ATTR_INPUT_SOFTMAX_LSE_FLAG_INDEX = 10;
static constexpr uint32_t FIA_INPUT_QUERY_PADDING_SIZE_INDEX = 15;
static constexpr uint32_t FIA_INPUT_KV_PADDING_SIZE_INDEX = 16;
static constexpr uint32_t FIA_INPUT_ACTUAL_SHARED_PREFIX_LEN_INDEX = 23;
static constexpr uint32_t FIA_QUERY_ROPE_INDEX = 24;
static constexpr uint32_t FIA_OUT_DTYPE_INDEX = 15;

static const std::map<int64_t, ge::DataType> TORCH_DTYPE_ENUM_VALUE_TO_GE_DTYPE_MAP = {
    {5,  ge::DT_FLOAT16}, 
    {15, ge::DT_BF16},
    {23, ge::DT_FLOAT8_E5M2},
    {24, ge::DT_FLOAT8_E4M3FN},
    {290, ge::DT_HIFLOAT8}};

static ge::graphStatus InferShapeFusedInferAttentionScore(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter FusedInferAttentionScore InferShape impl.");
    // query shape
    const gert::Shape *queryShape = context->GetInputShape(FIA_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    // value shape
    const gert::Shape *valueShape = context->GetDynamicInputShape(FIA_VALUE_INDEX, FIA_DYNAMIC_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    // Page Attention
    bool isPageAttention = (context->GetOptionalInputShape(FIA_BLOCK_TABLE_INDEX) == nullptr) ? false : true;

    // attentionOut
    gert::Shape *attentionOutShape = context->GetOutputShape(FIA_ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    gert::Shape *softmaxLseShape = context->GetOutputShape(FIA_SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);

    // Get attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutPtr = attrs->GetAttrPointer<char>(FIA_ATTR_INPUT_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutPtr);
    const int64_t *numHeadsPtr = attrs->GetInt(FIA_ATTR_NUM_HEADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numHeadsPtr);
    const int64_t *numKeyValueHeadsPtr = attrs->GetInt(FIA_ATTR_NUM_KV_HEADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numKeyValueHeadsPtr);

    // KV_N除零保护, 当KV_N为零时KV_N = Q_N
    if (*numHeadsPtr == 0) {
        OP_LOGE(context->GetNodeName(), "numHeads can not be 0!");
        return ge::GRAPH_FAILED;
    }
    int64_t numKeyValueHeads = (*numKeyValueHeadsPtr == 0) ? *numHeadsPtr : *numKeyValueHeadsPtr;

    int64_t qSeqSize = 1;
    int64_t batchOfQ = 1;

    // set AttentionOut shape
    *attentionOutShape = *queryShape;

    // UNKNOWN DIM
    if (((queryShape->GetDimNum() == FIA_LAYOUT_DIM_NUMS_1) && (queryShape->GetDim(FIA_LAYOUT_DIM0) == FIA_UNKNOWN_DIMS))||
        ((valueShape->GetDimNum() == FIA_LAYOUT_DIM_NUMS_1) && (valueShape->GetDim(FIA_LAYOUT_DIM0) == FIA_UNKNOWN_DIMS))) {
        attentionOutShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_1);
        (*attentionOutShape)[FIA_LAYOUT_DIM0] = FIA_UNKNOWN_DIMS;
        softmaxLseShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_1);
        (*softmaxLseShape)[FIA_LAYOUT_DIM0] = FIA_UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    if (strcmp(inputLayoutPtr, "SH") == 0) {
        if (queryShape->GetDimNum() != LAYOUT_SH_DIM_NUMS) {
            OP_LOGE(context->GetNodeName(), "Layout SH, queryDims(%zu) must be 2!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        qSeqSize = (*queryShape)[FIA_LAYOUT_DIM0]; // already ensure not nullptr and out of bound.
    } else if (strcmp(inputLayoutPtr, "BSH") == 0) {
        if (queryShape->GetDimNum() != LAYOUT_BSH_DIM_NUMS) {
            OP_LOGE(context->GetNodeName(), "Layout BSH, queryDims(%zu) must be 3!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (valueShape->GetDim(FIA_LAYOUT_DIM2) != -1) {
            int64_t outputH =  0;
            if (!isPageAttention) {
                if (valueShape->GetDimNum() != LAYOUT_BSH_DIM_NUMS) {
                    OP_LOGE(context->GetNodeName(), "Layout BSH, valueDims(%zu) must be 3!", valueShape->GetDimNum());
                    return ge::GRAPH_FAILED;
                }
                outputH = (*valueShape)[FIA_LAYOUT_DIM2] / numKeyValueHeads * (*numHeadsPtr);
                outputH = (outputH == 0 || (*queryShape)[FIA_LAYOUT_DIM2] == 0) ? (*queryShape)[FIA_LAYOUT_DIM2] : outputH;
            } else {
                if (valueShape->GetDimNum() == LAYOUT_PA_NZ_DIM_NUMS) {    // NZ [blockNum, N, D/16, blockSize, 16]
                    outputH = (*valueShape)[FIA_LAYOUT_DIM2] * (*valueShape)[FIA_LAYOUT_DIM4] * (*numHeadsPtr);
                    outputH = (outputH == 0 || (*queryShape)[FIA_LAYOUT_DIM2] == 0) ? (*queryShape)[FIA_LAYOUT_DIM2] : outputH;
                } else if (valueShape->GetDimNum() == LAYOUT_PA_BBH_DIM_NUMS) { // BBH
                    outputH = (*valueShape)[FIA_LAYOUT_DIM2] / numKeyValueHeads * (*numHeadsPtr);
                    outputH = (outputH == 0 || (*queryShape)[FIA_LAYOUT_DIM2] == 0) ? (*queryShape)[FIA_LAYOUT_DIM2] : outputH;
                } else {
                    OP_LOGE(context->GetNodeName(), "Layout BSH with page attention,  valueDims(%zu) must be 3 or 5!", valueShape->GetDimNum());
                    return ge::GRAPH_FAILED;
                }
            }
            batchOfQ = (*queryShape)[FIA_LAYOUT_DIM0];
            qSeqSize = (*queryShape)[FIA_LAYOUT_DIM1];
            (*attentionOutShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM0];
            (*attentionOutShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM1];
            (*attentionOutShape)[FIA_LAYOUT_DIM2] = outputH;
        }
    } else if (strcmp(inputLayoutPtr, "BSND") == 0) {
        if (queryShape->GetDimNum() != LAYOUT_BSND_DIM_NUMS) {
            OP_LOGE(context->GetNodeName(), "Layout BSND, queryDims(%zu) must be 4!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        int64_t outputD = 0;
        if (!isPageAttention) {
            if (valueShape->GetDimNum() != LAYOUT_BSND_DIM_NUMS) {
                OP_LOGE(context->GetNodeName(), "Layout BSND, valueDims(%zu) must be 4!", valueShape->GetDimNum());
                return ge::GRAPH_FAILED;
            }
            outputD = (*valueShape)[FIA_LAYOUT_DIM3];
            outputD = (outputD == 0 || (*queryShape)[FIA_LAYOUT_DIM3] == 0) ? (*queryShape)[FIA_LAYOUT_DIM3] : outputD;
        } else {
            if (valueShape->GetDimNum() == LAYOUT_PA_BBH_DIM_NUMS) { // BBH
                outputD = (*valueShape)[FIA_LAYOUT_DIM2] / numKeyValueHeads;
                outputD = (outputD == 0 || (*queryShape)[FIA_LAYOUT_DIM3] == 0) ? (*queryShape)[FIA_LAYOUT_DIM3] : outputD;
            } else if (valueShape->GetDimNum() == LAYOUT_PA_NZ_DIM_NUMS) { // NZ [blockNum, N, D/16, blockSize, 16]
                outputD = (*valueShape)[FIA_LAYOUT_DIM2] * (*valueShape)[FIA_LAYOUT_DIM4];
                outputD = (outputD == 0 || (*queryShape)[FIA_LAYOUT_DIM3] == 0) ? (*queryShape)[FIA_LAYOUT_DIM3] : outputD;
            } else {
                OP_LOGE(context->GetNodeName(), "Layout BSND with page attention,  valueDims(%zu) must be 3 or 5!", valueShape->GetDimNum());
                return ge::GRAPH_FAILED;
            }
        }
        batchOfQ = (*queryShape)[FIA_LAYOUT_DIM0];
        qSeqSize = (*queryShape)[FIA_LAYOUT_DIM1];
        (*attentionOutShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM0];
        (*attentionOutShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM1];
        (*attentionOutShape)[FIA_LAYOUT_DIM2] = (*queryShape)[FIA_LAYOUT_DIM2];
        (*attentionOutShape)[FIA_LAYOUT_DIM3] = outputD;
    } else if (strcmp(inputLayoutPtr, "NSD") == 0) {
        if (queryShape->GetDimNum() != LAYOUT_NSD_DIM_NUMS) {
            OP_LOGE(context->GetNodeName(), "Layout NSD, queryDims(%zu) must be 3!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        qSeqSize = (*queryShape)[FIA_LAYOUT_DIM1];
    } else if (strcmp(inputLayoutPtr, "BNSD") == 0) {
        if (queryShape->GetDimNum() != LAYOUT_BNSD_DIM_NUMS) {
            OP_LOGE(context->GetNodeName(), "Layout BNSD, queryDims(%zu) must be 4!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        int64_t outputD =  0;
        if (!isPageAttention) {
            if (valueShape->GetDimNum() != LAYOUT_BNSD_DIM_NUMS) {
                OP_LOGE(context->GetNodeName(), "Layout BNSD, valueDims(%zu) must be 4!", valueShape->GetDimNum());
                return ge::GRAPH_FAILED;
            }
            outputD = (*valueShape)[FIA_LAYOUT_DIM3];
            outputD = (outputD == 0 || (*queryShape)[FIA_LAYOUT_DIM3] == 0) ? (*queryShape)[FIA_LAYOUT_DIM3] : outputD;
        } else {
            if (valueShape->GetDimNum() == LAYOUT_PA_BBH_DIM_NUMS) {
                outputD = (*valueShape)[FIA_LAYOUT_DIM2] / numKeyValueHeads;
                outputD = (outputD == 0 || (*queryShape)[FIA_LAYOUT_DIM3] == 0) ? (*queryShape)[FIA_LAYOUT_DIM3] : outputD;
            } else if (valueShape->GetDimNum() == LAYOUT_PA_BNBD_DIM_NUMS) {
                outputD = (*valueShape)[FIA_LAYOUT_DIM3];
                outputD = (outputD == 0 || (*queryShape)[FIA_LAYOUT_DIM3] == 0) ? (*queryShape)[FIA_LAYOUT_DIM3] : outputD;
            } else if (valueShape->GetDimNum() == LAYOUT_PA_NZ_DIM_NUMS) { // NZ [blockNum, N, D/16, blockSize, 16]
                outputD = (*valueShape)[FIA_LAYOUT_DIM2] * (*valueShape)[FIA_LAYOUT_DIM4];
                outputD = (outputD == 0 || (*queryShape)[FIA_LAYOUT_DIM3] == 0) ? (*queryShape)[FIA_LAYOUT_DIM3] : outputD;
            } else {
                OP_LOGE(context->GetNodeName(), "Layout BNSD with page attention,  valueDims(%zu) must be 3 or 4 or 5!", valueShape->GetDimNum());
                return ge::GRAPH_FAILED;
            }
        }
        batchOfQ = (*queryShape)[FIA_LAYOUT_DIM0];
        qSeqSize = (*queryShape)[FIA_LAYOUT_DIM2];
        (*attentionOutShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM0];
        (*attentionOutShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM1];
        (*attentionOutShape)[FIA_LAYOUT_DIM2] = (*queryShape)[FIA_LAYOUT_DIM2];
        (*attentionOutShape)[FIA_LAYOUT_DIM3] = outputD;
    } else if (strcmp(inputLayoutPtr, "BNSD_BSND") == 0) {
        if (queryShape->GetDimNum() != LAYOUT_BNSD_BSND_DIM_NUMS) {
            OP_LOGE(context->GetNodeName(), "Layout BNSD_BSND, queryDims(%zu) must be 4!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        int64_t outputD =  0;
        if (!isPageAttention) {
            if (valueShape->GetDimNum() != LAYOUT_BNSD_BSND_DIM_NUMS) {
                OP_LOGE(context->GetNodeName(), "Layout BNSD_BSND, valueDims(%zu) must be 4!", valueShape->GetDimNum());
                return ge::GRAPH_FAILED;
            }
            outputD = (*valueShape)[FIA_LAYOUT_DIM3];
            outputD = (outputD == 0 || (*queryShape)[FIA_LAYOUT_DIM3] == 0) ? (*queryShape)[FIA_LAYOUT_DIM3] : outputD; 
        } else {
            if (valueShape->GetDimNum() == LAYOUT_PA_BNBD_DIM_NUMS) {
                outputD = (*valueShape)[FIA_LAYOUT_DIM3];
                outputD = (outputD == 0 || (*queryShape)[FIA_LAYOUT_DIM3] == 0) ? (*queryShape)[FIA_LAYOUT_DIM3] : outputD;
            } else if (valueShape->GetDimNum() == LAYOUT_PA_BBH_DIM_NUMS) {
                outputD = (*valueShape)[FIA_LAYOUT_DIM2] / numKeyValueHeads;
                outputD = (outputD == 0 || (*queryShape)[FIA_LAYOUT_DIM3] == 0) ? (*queryShape)[FIA_LAYOUT_DIM3] : outputD;
            } else {
                OP_LOGE(context->GetNodeName(), "Layout BNSD_BSND with page attention,  valueDims(%zu) must be 3 or 4!", valueShape->GetDimNum());
                return ge::GRAPH_FAILED;
            }
        }
        batchOfQ = (*queryShape)[FIA_LAYOUT_DIM0];
        qSeqSize = (*queryShape)[FIA_LAYOUT_DIM2];
        (*attentionOutShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM0];
        (*attentionOutShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM2];
        (*attentionOutShape)[FIA_LAYOUT_DIM2] = (*queryShape)[FIA_LAYOUT_DIM1];
        (*attentionOutShape)[FIA_LAYOUT_DIM3] = outputD;
    } else if (strcmp(inputLayoutPtr, "TND") == 0) {
        if (queryShape->GetDimNum() != LAYOUT_TND_DIM_NUMS) {
            OP_LOGE(context->GetNodeName(), "Layout TND, queryDims(%zu) must be 3!",
                queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        int64_t outputD =  0;
        if (!isPageAttention) {
            if (valueShape->GetDimNum() != LAYOUT_TND_DIM_NUMS) {
                OP_LOGE(context->GetNodeName(), "Layout TND, valueDims(%zu) must be 3!", valueShape->GetDimNum());
                return ge::GRAPH_FAILED;
            }
            outputD = (*valueShape)[FIA_LAYOUT_DIM2];
        } else {
            if (valueShape->GetDimNum() == LAYOUT_PA_BNBD_DIM_NUMS) {
                outputD = (*valueShape)[FIA_LAYOUT_DIM3];
            } else if (valueShape->GetDimNum() == LAYOUT_PA_BBH_DIM_NUMS) {
                outputD = (*valueShape)[FIA_LAYOUT_DIM2] / numKeyValueHeads;
            } else if (valueShape->GetDimNum() == LAYOUT_PA_NZ_DIM_NUMS) {
                outputD = (*valueShape)[FIA_LAYOUT_DIM2] * (*valueShape)[FIA_LAYOUT_DIM4];
            } else {
                OP_LOGE(context->GetNodeName(), "Layout TND,  valueDims(%zu) must be 3 or 4 or 5!", valueShape->GetDimNum());
                return ge::GRAPH_FAILED;
            }
        }
        (*attentionOutShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM0];
        (*attentionOutShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM1];
        (*attentionOutShape)[FIA_LAYOUT_DIM2] = outputD;
    } else if (strcmp(inputLayoutPtr, "BNSD_NBSD") == 0) {
        if (queryShape->GetDimNum() != LAYOUT_BNSD_DIM_NUMS) {
            OP_LOGE(context->GetNodeName(), "Layout BNSD_NBSD, queryDims(%zu) must be 4!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        batchOfQ = (*queryShape)[FIA_LAYOUT_DIM0];
        qSeqSize = (*queryShape)[FIA_LAYOUT_DIM2];
        (*attentionOutShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM1];
        (*attentionOutShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM0];
        (*attentionOutShape)[FIA_LAYOUT_DIM2] = (*queryShape)[FIA_LAYOUT_DIM2];
        (*attentionOutShape)[FIA_LAYOUT_DIM3] = (*queryShape)[FIA_LAYOUT_DIM3];
    } else if (strcmp(inputLayoutPtr, "BSND_NBSD") == 0) {
        if (queryShape->GetDimNum() != LAYOUT_BSND_DIM_NUMS) {
            OP_LOGE(context->GetNodeName(), "Layout BSND_NBSD, queryDims(%zu) must be 4!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        batchOfQ = (*queryShape)[FIA_LAYOUT_DIM0];
        qSeqSize = (*queryShape)[FIA_LAYOUT_DIM1];
        (*attentionOutShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM2];
        (*attentionOutShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM0];
        (*attentionOutShape)[FIA_LAYOUT_DIM2] = (*queryShape)[FIA_LAYOUT_DIM1];
        (*attentionOutShape)[FIA_LAYOUT_DIM3] = (*queryShape)[FIA_LAYOUT_DIM3];
    } else if (strcmp(inputLayoutPtr, "BSH_NBSD") == 0) {
        if (queryShape->GetDimNum() != LAYOUT_BSH_DIM_NUMS) {
            OP_LOGE(context->GetNodeName(), "Layout BSH_NBSD, queryDims(%zu) must be 3!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        attentionOutShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_4);
        batchOfQ = (*queryShape)[FIA_LAYOUT_DIM0];
        qSeqSize = (*queryShape)[FIA_LAYOUT_DIM1];
        (*attentionOutShape)[FIA_LAYOUT_DIM0] = *numHeadsPtr;
        (*attentionOutShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM0];
        (*attentionOutShape)[FIA_LAYOUT_DIM2] = (*queryShape)[FIA_LAYOUT_DIM1];
        (*attentionOutShape)[FIA_LAYOUT_DIM3] = (*queryShape)[FIA_LAYOUT_DIM2] / (*numHeadsPtr);
    } else if (strcmp(inputLayoutPtr, "TND_NTD") == 0) {
        if (queryShape->GetDimNum() != LAYOUT_TND_DIM_NUMS) {
            OP_LOGE(context->GetNodeName(), "Layout TND_NTD, queryDims(%zu) must be 3!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        (*attentionOutShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM1];
        (*attentionOutShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM0];
        (*attentionOutShape)[FIA_LAYOUT_DIM2] = (*queryShape)[FIA_LAYOUT_DIM2];
    } else if (strcmp(inputLayoutPtr, "NTD_TND") == 0) {
        if (queryShape->GetDimNum() != LAYOUT_TND_DIM_NUMS) {
            OP_LOGE(context->GetNodeName(), "Layout NTD_TND, queryDims(%zu) must be 3!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }

        if (isPageAttention) {
            if (valueShape->GetDimNum() == FIA_LAYOUT_DIM3) {
                (*attentionOutShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM1];
                (*attentionOutShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM0];
                (*attentionOutShape)[FIA_LAYOUT_DIM2] = (*valueShape)[FIA_LAYOUT_DIM2] / numKeyValueHeads;
            } else if (valueShape->GetDimNum() == FIA_LAYOUT_DIM4) {
                (*attentionOutShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM1];
                (*attentionOutShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM0];
                (*attentionOutShape)[FIA_LAYOUT_DIM2] = (*valueShape)[FIA_LAYOUT_DIM3];
            } else if (valueShape->GetDimNum() == LAYOUT_PA_NZ_DIM_NUMS) {
                (*attentionOutShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM1];
                (*attentionOutShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM0];
                (*attentionOutShape)[FIA_LAYOUT_DIM2] = (*valueShape)[FIA_LAYOUT_DIM2] * (*valueShape)[FIA_LAYOUT_DIM4];
            } else {
                OP_LOGE(context->GetNodeName(), "Layout NTD_TND, keyValueDims(%zu) must be 3/4/5!", valueShape->GetDimNum());
                return ge::GRAPH_FAILED;
            }
        } else {
            (*attentionOutShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM1];
            (*attentionOutShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM0];
            (*attentionOutShape)[FIA_LAYOUT_DIM2] = (*valueShape)[FIA_LAYOUT_DIM2];
        }
    } else {
        // not support layout
        OP_LOGE(context->GetNodeName(), "Invalid input layout: %s, not support!", inputLayoutPtr);
        return ge::GRAPH_FAILED;
    }

    const bool *softmaxLsePtr = attrs->GetAttrPointer<bool>(FIA_ATTR_INPUT_SOFTMAX_LSE_FLAG_INDEX);
    bool softmaxLseFlag = (softmaxLsePtr != nullptr) ? *softmaxLsePtr : false;
    if (softmaxLseFlag) {
        if (strcmp(inputLayoutPtr, "TND") == 0 || strcmp(inputLayoutPtr, "TND_NTD") == 0) { // TN1
            softmaxLseShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_3);
            if (!isPageAttention) { // PFA
                (*softmaxLseShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM0];
                (*softmaxLseShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM1];
                (*softmaxLseShape)[FIA_LAYOUT_DIM2] = NUM_1;
            } else { //IFA
                (*softmaxLseShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM0];
                (*softmaxLseShape)[FIA_LAYOUT_DIM1] = *numHeadsPtr;
                (*softmaxLseShape)[FIA_LAYOUT_DIM2] = NUM_1;
            }
        } else if (strcmp(inputLayoutPtr, "NTD_TND") == 0) {
            softmaxLseShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_3);
            if (!isPageAttention) { // PFA
                (*softmaxLseShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM1];
                (*softmaxLseShape)[FIA_LAYOUT_DIM1] = (*queryShape)[FIA_LAYOUT_DIM0];
                (*softmaxLseShape)[FIA_LAYOUT_DIM2] = NUM_1;
            } else { //IFA
                (*softmaxLseShape)[FIA_LAYOUT_DIM0] = (*queryShape)[FIA_LAYOUT_DIM1];
                (*softmaxLseShape)[FIA_LAYOUT_DIM1] = *numHeadsPtr;
                (*softmaxLseShape)[FIA_LAYOUT_DIM2] = NUM_1;
            }
        } else { // BNS1
            softmaxLseShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_4);
            (*softmaxLseShape)[FIA_LAYOUT_DIM0] = batchOfQ;
            (*softmaxLseShape)[FIA_LAYOUT_DIM1] = *numHeadsPtr;
            (*softmaxLseShape)[FIA_LAYOUT_DIM2] = qSeqSize;
            (*softmaxLseShape)[FIA_LAYOUT_DIM3] = NUM_1;
        }
    } else {
        softmaxLseShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_1);
        (*softmaxLseShape)[FIA_LAYOUT_DIM0] = NUM_0;
    }
    OP_LOGD(context->GetNodeName(), "FusedInferAttentionScore InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFusedInferAttentionScore(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter FusedInferAttentionScore InferDataType impl.");
    // default set q's dtype as fia's output type
    ge::DataType outputType = context->GetInputDataType(FIA_QUERY_INDEX);
    // 10 is quant_scale2's index, if not instantiated or illegal return ge::DT_UNDEFINED
    if (context->GetOptionalInputDataType(FIA_QUANT_SCALE2_INDEX) != ge::DT_UNDEFINED) {
        outputType = ge::DT_INT8;

        auto attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const int64_t *outTypePtr = attrs->GetInt(FIA_OUT_DTYPE_INDEX);
        if (outTypePtr != nullptr) {
            auto iter = TORCH_DTYPE_ENUM_VALUE_TO_GE_DTYPE_MAP.find(*outTypePtr);
            if (iter != TORCH_DTYPE_ENUM_VALUE_TO_GE_DTYPE_MAP.end()) {
                outputType = iter->second;
            }
        }
    } else if (context->GetInputDataType(FIA_QUERY_INDEX) == ge::DT_INT8 ||
        context->GetInputDataType(FIA_QUERY_INDEX) == ge::DT_FLOAT8_E5M2 ||
        context->GetInputDataType(FIA_QUERY_INDEX) == ge::DT_FLOAT8_E4M3FN ||
        context->GetInputDataType(FIA_QUERY_INDEX) == ge::DT_HIFLOAT8) {
        // 1. MLA: if the dtype of input query is int8, the dtype of output is same as the dtype of input query_rope.
        // 2. GQA: the int8 dtype of input query is not supported.
        outputType = context->GetOptionalInputDataType(FIA_QUERY_ROPE_INDEX);
        if (outputType == ge::DT_UNDEFINED) {
            outputType = ge::DT_FLOAT16;
        }
    }
    // attention_out, outidx:0
    context->SetOutputDataType(FIA_ATTENTION_OUT_INDEX, outputType);
    context->SetOutputDataType(FIA_SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "FusedInferAttentionScore InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedInferAttentionScore)
    .InferShape(InferShapeFusedInferAttentionScore)
    .InferDataType(InferDataTypeFusedInferAttentionScore)
    .InputsDataDependency({FIA_INPUT_ACTUAL_SEQ_LENGTHS_INDEX, FIA_INPUT_ACTUAL_SEQ_LENGTHS_KV_INDEX,
                           FIA_INPUT_QUERY_PADDING_SIZE_INDEX, FIA_INPUT_KV_PADDING_SIZE_INDEX,
                           FIA_INPUT_ACTUAL_SHARED_PREFIX_LEN_INDEX});
} // namespace ops