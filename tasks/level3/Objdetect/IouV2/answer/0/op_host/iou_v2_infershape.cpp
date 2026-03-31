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
 * \file stack_group_points_infershape.cpp
 * \brief
 */
#include <cstdio>

#include "register/op_impl_registry.h"

// #include "log/log.h"
#define OP_LOGE(nodeName, fmt, ...) \
    do { \
        std::printf(fmt, ##__VA_ARGS__); \
        std::printf("\n"); \
    } while (0)

using namespace ge;
using namespace std;

namespace
{
    const int64_t INPUT_INDEX_BBOXES = 0;
    const int64_t INPUT_INDEX_GTBOXES = 1;
    const int64_t OUTPUT_INDEX_OVERLAP = 0;
    const int64_t IOUS_DIM = 2;
    const int64_t ALIGNED_INFO_IDX = 2;
} // namespace

namespace ops
{
    static ge::graphStatus InferShapeForIouV2(gert::InferShapeContext *context)
    {
        auto overlapShape = context->GetOutputShape(OUTPUT_INDEX_OVERLAP);
        auto const bboxesShape = context->GetInputShape(INPUT_INDEX_BBOXES);
        auto const gtboxesShape = context->GetInputShape(INPUT_INDEX_GTBOXES);

        auto attrs = context->GetAttrs();
        const bool *aligned = attrs->GetAttrPointer<bool>(ALIGNED_INFO_IDX);

        if (bboxesShape == nullptr || gtboxesShape == nullptr || attrs == nullptr)
        {
            return ge::GRAPH_FAILED;
        }

        // update output shape.
        overlapShape->SetDimNum(IOUS_DIM); // the output dimensions are 2.
        if (*aligned)
        {
            int64_t const bboxesNum = bboxesShape->GetDim(1);
            int64_t const gtboxesNum = gtboxesShape->GetDim(1);
            if (bboxesNum != gtboxesNum)
            {
                OP_LOGE(
                    context, "Parameter aligned is true, the num of bboxes and gtboxes must be same.");
                return ge::GRAPH_FAILED;
            }
            overlapShape->SetDim(0, gtboxesNum);
            overlapShape->SetDim(1, 1);
        }
        else
        {
            int64_t const bboxesNum = bboxesShape->GetDim(0);
            int64_t const gtboxesNum = gtboxesShape->GetDim(0);
            overlapShape->SetDim(0, gtboxesNum);
            overlapShape->SetDim(1, bboxesNum);
        }
        return ge::GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataTypeForIouV2(gert::InferDataTypeContext *context)
    {
        const ge::DataType feature_dtype = context->GetInputDataType(INPUT_INDEX_BBOXES);
        context->SetOutputDataType(OUTPUT_INDEX_OVERLAP, feature_dtype);
        return GRAPH_SUCCESS;
    }

    IMPL_OP_INFERSHAPE(IouV2)
        .InferShape(InferShapeForIouV2)
        .InferDataType(InferDataTypeForIouV2);
} // namespace ops