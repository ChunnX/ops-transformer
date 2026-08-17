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
 * \file ffn_worker_scheduler_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {

constexpr size_t INPUT_IDX_SCHEDULE_CONTEXT = 0;
constexpr size_t OUTPUT_IDX_SCHEDULE_CONTEXT = 0;

graphStatus InferShape4FfnWorkerScheduler(gert::InferShapeContext *context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShape4FfnWorkerScheduler");

    const gert::Shape *scheduleContextInputShape = context->GetInputShape(INPUT_IDX_SCHEDULE_CONTEXT);
    OP_CHECK_NULL_WITH_CONTEXT(context, scheduleContextInputShape);

    gert::Shape *scheduleContextOutputShape = context->GetOutputShape(OUTPUT_IDX_SCHEDULE_CONTEXT);
    OP_CHECK_NULL_WITH_CONTEXT(context, scheduleContextOutputShape);

    // 输出 shape 与输入 shape 一致
    *scheduleContextOutputShape = *scheduleContextInputShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShape4FfnWorkerScheduler");
    return GRAPH_SUCCESS;
}

graphStatus InferDtype4FfnWorkerScheduler(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "InferDtype4FfnWorkerScheduler enter");

    ge::DataType inputDtype = context->GetInputDataType(INPUT_IDX_SCHEDULE_CONTEXT);
    context->SetOutputDataType(OUTPUT_IDX_SCHEDULE_CONTEXT, inputDtype);

    OP_LOGD(context->GetNodeName(), "InferDtype4FfnWorkerScheduler end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FfnWorkerScheduler)
    .InferShape(InferShape4FfnWorkerScheduler)
    .InferDataType(InferDtype4FfnWorkerScheduler);

} // namespace ops
