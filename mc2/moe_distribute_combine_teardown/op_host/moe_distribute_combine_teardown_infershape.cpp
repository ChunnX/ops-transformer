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
 * \file moe_distribute_combine_teardown_infershape.cpp
 * \brief
 */
#include "runtime_util.h"
#include "mc2_log.h"
#include "../../common/op_host/mc2_combine_infershape.h"
#include "platform/platform_info.h"

namespace ops {

using namespace ge;

static constexpr size_t COMBINE_INPUT_EXPAND_X_INDEX = 0;
static constexpr size_t COMBINE_INPUT_QUANT_EXPAND_X_INDEX = 1;
static constexpr size_t COMBINE_INPUT_EXPERT_IDS_INDEX = 2;
static constexpr size_t COMBINE_INPUT_EXPAND_IDX_INDEX = 3;
static constexpr size_t COMBINE_INPUT_EXPERT_SCALES_INDEX = 4;
static constexpr size_t COMBINE_INPUT_COMM_CMD_INFO_INDEX = 5;
static constexpr size_t COMBINE_OUTPUT_X_INDEX = 0;

static ge::graphStatus InferShapeMoeDistributeCombineTeardown(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDistributeCombineTeardown.");
    // These required inputs do not determine the output dimensions, but must still be present.
    const gert::Shape *quantExpandXShape = context->GetInputShape(COMBINE_INPUT_QUANT_EXPAND_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantExpandXShape);
    const gert::Shape *expandIdxShape = context->GetInputShape(COMBINE_INPUT_EXPAND_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandIdxShape);
    const gert::Shape *expertScalesShape = context->GetInputShape(COMBINE_INPUT_EXPERT_SCALES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertScalesShape);
    const gert::Shape *commCmdInfoShape = context->GetInputShape(COMBINE_INPUT_COMM_CMD_INFO_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commCmdInfoShape);
    if (Mc2InferShape::InferCombineOutputShape(context, COMBINE_INPUT_EXPAND_X_INDEX,
                                                COMBINE_INPUT_EXPERT_IDS_INDEX, COMBINE_OUTPUT_X_INDEX) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDistributeCombineTeardown.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDistributeCombineTeardown(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeCombineTeardown.");
    Mc2InferShape::InferCombineOutputXDataType(context, COMBINE_INPUT_EXPAND_X_INDEX, COMBINE_OUTPUT_X_INDEX);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDistributeCombineTeardown.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDistributeCombineTeardown)
    .InferShape(InferShapeMoeDistributeCombineTeardown)
    .InferDataType(InferDataTypeMoeDistributeCombineTeardown);
} // namespace ops
