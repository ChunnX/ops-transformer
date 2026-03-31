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
 * \file flash_attn_metadata_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../op_kernel_aicpu/flash_attn_metadata.h"
#include "flash_attn_metadata_proto.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeFlashAttnMetadata(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "InferShapeFlashAttnMetadata");

    gert::Shape* oShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, oShape);
    oShape->SetDimNum(1);
    oShape->SetDim(0, optiling::FA_META_SIZE);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeFlashAttnMetadata(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "InferDtypeFlashAttnMetadata");

    context->SetOutputDataType(0, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlashAttnMetadata)
    .InferShape(InferShapeFlashAttnMetadata)
    .InferDataType(InferDtypeFlashAttnMetadata);
} // namespace ops