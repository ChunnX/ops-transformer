/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "runtime/rt_external_base.h"
#include "platform/soc_spec.h"

using namespace ge;
namespace ops {

static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;

static constexpr size_t MOE_DISPATCH_FFN_COMBINE_INPUT_CONTEXT_INDEX = 0;
static constexpr size_t MOE_DISPATCH_FFN_COMBINE_INPUT_X_INDEX = 1;
static constexpr size_t MOE_DISPATCH_FFN_COMBINE_INPUT_TOPK_IDS_INDEX = 2;
static constexpr size_t MOE_DISPATCH_FFN_COMBINE_INPUT_TOPK_WEIGHTS_INDEX = 3;
// weight1 starts at index 4 (dynamic input)
// weight2 follows weight1 (dynamic input)
// x_active_mask, weight_scales1, weight_scales2, scales are optional/dynamic inputs

static constexpr size_t MOE_DISPATCH_FFN_COMBINE_OUTPUT_Y_INDEX = 0;
static constexpr size_t MOE_DISPATCH_FFN_COMBINE_OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 1;

static constexpr size_t MOE_DISPATCH_FFN_COMBINE_ATTR_MOE_EXPERT_NUM_INDEX = 0;
static constexpr size_t MOE_DISPATCH_FFN_COMBINE_ATTR_EP_WORLD_SIZE_INDEX = 1;
static constexpr size_t MOE_DISPATCH_FFN_COMBINE_ATTR_CCL_BUFFER_SIZE_INDEX = 2;
static constexpr size_t MOE_DISPATCH_FFN_COMBINE_ATTR_MAX_RECV_TOKEN_NUM_INDEX = 3;
static constexpr size_t MOE_DISPATCH_FFN_COMBINE_ATTR_DISPATCH_QUANT_MODE_INDEX = 4;
static constexpr size_t MOE_DISPATCH_FFN_COMBINE_ATTR_DISPATCH_QUANT_OUT_TYPE_INDEX = 5;
static constexpr size_t MOE_DISPATCH_FFN_COMBINE_ATTR_COMBINE_QUANT_MODE_INDEX = 6;
static constexpr size_t MOE_DISPATCH_FFN_COMBINE_ATTR_COMM_ALG_INDEX = 7;
static constexpr size_t MOE_DISPATCH_FFN_COMBINE_ATTR_GLOBAL_BS_INDEX = 8;

static ge::graphStatus InferShapeMoeDispatchFFNCombine(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDispatchFFNCombine.");

    const gert::Shape *xShape = context->GetInputShape(MOE_DISPATCH_FFN_COMBINE_INPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape *yShape = context->GetOutputShape(MOE_DISPATCH_FFN_COMBINE_OUTPUT_Y_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape *expertTokenNumsShape = context->GetOutputShape(MOE_DISPATCH_FFN_COMBINE_OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto moeExpertNum = attrs->GetAttrPointer<int64_t>(MOE_DISPATCH_FFN_COMBINE_ATTR_MOE_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, moeExpertNum);

    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(MOE_DISPATCH_FFN_COMBINE_ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);

    int64_t bs = xShape->GetDim(0);
    int64_t h = xShape->GetDim(1);

    yShape->SetDimNum(DIM_TWO);
    yShape->SetDim(0U, bs);
    yShape->SetDim(1U, h);

    expertTokenNumsShape->SetDimNum(DIM_ONE);
    expertTokenNumsShape->SetDim(0U, *moeExpertNum / *epWorldSize);

    OP_LOGD(context->GetNodeName(), "y shape is [%ld, %ld] after infershape.", bs, h);
    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDispatchFFNCombine.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDispatchFFNCombine(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDispatchFFNCombine.");

    auto xDtype = context->GetInputDataType(MOE_DISPATCH_FFN_COMBINE_INPUT_X_INDEX);
    context->SetOutputDataType(MOE_DISPATCH_FFN_COMBINE_OUTPUT_Y_INDEX, xDtype);
    context->SetOutputDataType(MOE_DISPATCH_FFN_COMBINE_OUTPUT_EXPERT_TOKEN_NUMS_INDEX, ge::DT_INT32);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDispatchFFNCombine.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDispatchFFNCombine)
    .InferShape(InferShapeMoeDispatchFFNCombine)
    .InferDataType(InferDataTypeMoeDispatchFFNCombine);
}  // namespace ops
