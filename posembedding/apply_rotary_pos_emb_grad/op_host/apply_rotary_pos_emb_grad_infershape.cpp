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
 * \file apply_rotary_pos_emb_grad_infershape.cpp
 * \brief 形状推导
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t IGQ = 0; // grad_query_embed
static constexpr size_t IGK = 1; // grad_key_embed
static constexpr size_t ICS = 2; // cos
static constexpr size_t ISN = 3; // sin
static constexpr size_t OGQ = 0; // grad_query
static constexpr size_t OGK = 1; // grad_key
static constexpr size_t OGC = 2; // grad_cos
static constexpr size_t OGS = 3; // grad_sin

static ge::graphStatus InferShapeForApplyRotaryPosEmbGrad(gert::InferShapeContext* context)
{
    const gert::Shape* gq = context->GetInputShape(IGQ); OP_CHECK_NULL_WITH_CONTEXT(context, gq);
    const gert::Shape* gk = context->GetInputShape(IGK); OP_CHECK_NULL_WITH_CONTEXT(context, gk);
    const gert::Shape* cs = context->GetInputShape(ICS); OP_CHECK_NULL_WITH_CONTEXT(context, cs);
    const gert::Shape* sn = context->GetInputShape(ISN); OP_CHECK_NULL_WITH_CONTEXT(context, sn);

    gert::Shape* ogq = context->GetOutputShape(OGQ); OP_CHECK_NULL_WITH_CONTEXT(context, ogq);
    gert::Shape* ogk = context->GetOutputShape(OGK); OP_CHECK_NULL_WITH_CONTEXT(context, ogk);
    gert::Shape* ogc = context->GetOutputShape(OGC);
    gert::Shape* ogs = context->GetOutputShape(OGS);

    *ogq = *gq;
    *ogk = *gk;
    if (ogc != nullptr) {
        *ogc = *cs;
    }
    if (ogs != nullptr) {
        *ogs = *sn;
    }
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForApplyRotaryPosEmbGrad(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OGQ, context->GetInputDataType(IGQ));
    context->SetOutputDataType(OGK, context->GetInputDataType(IGK));
    if (context->GetOutputShape(OGC) != nullptr) {
        context->SetOutputDataType(OGC, context->GetInputDataType(ICS));
    }
    if (context->GetOutputShape(OGS) != nullptr) {
        context->SetOutputDataType(OGS, context->GetInputDataType(ISN));
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyRotaryPosEmbGrad)
    .InferShape(InferShapeForApplyRotaryPosEmbGrad)
    .InferDataType(InferDataTypeForApplyRotaryPosEmbGrad);
} // namespace ops
