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
 * \file sparse_flash_mla_grad_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
namespace sfmg {
constexpr uint32_t SELECTED_BLOCK_SIZE = 1;
enum class InputIndex : uint32_t {
    Q = 0,
    ORI_KV,
    CMP_KV,
    D_OUT,
    OUT,
    ORI_SPARSE_INDICES,
    CMP_SPARSE_INDICES,
    CU_SEQLENS_Q,
    CU_SEQLENS_ORI_KV,
    CU_SEQLENS_CMP_KV,
    SEQUSED_Q,
    SEQUSED_ORI_KV,
    SEQUSED_CMP_KV,
    CMP_RESIDUAL_KV,
    ORI_TOPK_LENGTH,
    CMP_TOPK_LENGTH,
    SOFTMAX_LSE,
    SINKS,
    METADATA,
};

enum class OutputIndex : uint32_t {
    DQ = 0,
    DORI_KV,
    DCMP_KV,
    DSINKS,
    ORI_SOFTMAX_L1,
    CMP_SOFTMAX_L1
};

enum class AttrIndex : uint32_t {
    SOFTMAX_SCALE = 0,
    CMP_RATIO,
    ORI_MASK_MODE,
    CMP_MASK_MODE,
    ORI_WIN_LEFT,
    ORI_WIN_RIGHT,
    LAYOUT_Q,
    LAYOUT_KV,
    DETERMINISTIC
};

ge::graphStatus InferShape4SparseFlashMlaGrad(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseAttnSharedkvGrad", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferShape4SparseAttnSharedkvGrad.");

    const gert::Shape *queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::Q));
    const gert::Shape *oriKVShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::ORI_KV));
    const gert::Shape *cmpKVShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::CMP_KV));
    const gert::Shape *sinksShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SINKS));
    const gert::Shape *oriSparseIndicesShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::ORI_SPARSE_INDICES));
    const gert::Shape *cmpSparseIndicesShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::CMP_SPARSE_INDICES));
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, oriKVShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmpKVShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinksShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, oriSparseIndicesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmpSparseIndicesShape);

    auto attrs = context->GetAttrs();
    auto scaleValue = attrs->GetInt(static_cast<size_t>(AttrIndex::SOFTMAX_SCALE));
    const char *inputLayout = attrs->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::LAYOUT_Q));
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleValue);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayout);

    std::string inputLayoutSasg = std::string(inputLayout);
    for (auto &c : inputLayoutSasg) {
        c = toupper(c);
    }
    if (inputLayoutSasg != "BSND" && inputLayoutSasg != "TND") {
        OP_LOGE(context, "The SparseAttnSharedkvGrad inputLayout should be BSND/TND, but got %s.",
                  inputLayoutSasg.c_str());
        return ge::GRAPH_FAILED;
    }

    gert::Shape *dqShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DQ));
    OP_CHECK_NULL_WITH_CONTEXT(context, dqShape);
    *dqShape = *queryShape;
    
    if (oriKVShape != nullptr) {
        gert::Shape *doriKVShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DORI_KV));
        OP_CHECK_NULL_WITH_CONTEXT(context, doriKVShape);
        *doriKVShape = *oriKVShape;
    }
    if (cmpKVShape != nullptr) {
        gert::Shape *dcmpKVShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DCMP_KV));
        OP_CHECK_NULL_WITH_CONTEXT(context, dcmpKVShape);
        *dcmpKVShape = *cmpKVShape;
    }
    if (sinksShape != nullptr) {
        gert::Shape *dsinksShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DSINKS));
        OP_CHECK_NULL_WITH_CONTEXT(context, dsinksShape);
        *dsinksShape = *sinksShape;
    }
    if (oriSparseIndicesShape != nullptr) {
        gert::Shape *doriSoftmaxL1Shape = context->GetOutputShape(static_cast<size_t>(OutputIndex::ORI_SOFTMAX_L1));
        OP_CHECK_NULL_WITH_CONTEXT(context, doriSoftmaxL1Shape);
        *doriSoftmaxL1Shape = *oriSparseIndicesShape;
    }
    if (cmpSparseIndicesShape != nullptr) {
        gert::Shape *dcmpSoftmaxL1Shape = context->GetOutputShape(static_cast<size_t>(OutputIndex::CMP_SOFTMAX_L1));
        OP_CHECK_NULL_WITH_CONTEXT(context, dcmpSoftmaxL1Shape);
        *dcmpSoftmaxL1Shape = *cmpSparseIndicesShape;
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4SparseFlashMlaGrad(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseAttnSharedkvGrad", "InferDataTypeContext is nullptr"),
               return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferDataType4SparseAttnSharedkvGrad.");

    auto dtype = context->GetInputDataType(static_cast<size_t>(InputIndex::Q));
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DQ), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DORI_KV), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DCMP_KV), dtype);
    auto dtypeFp32 = context->GetInputDataType(static_cast<size_t>(InputIndex::SOFTMAX_LSE));
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DSINKS), dtypeFp32);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::ORI_SOFTMAX_L1), dtypeFp32);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::ORI_SOFTMAX_L1), dtypeFp32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseFlashMlaGrad)
    .InferShape(InferShape4SparseFlashMlaGrad)
    .InferDataType(InferDataType4SparseFlashMlaGrad);
} // namespace sfmg
} // namespace ops
