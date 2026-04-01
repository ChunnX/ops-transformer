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
#include "register/op_impl_registry.h"

using namespace ge;
using namespace std;

namespace
{
    const uint8_t FEATURE_INDEX = 0;
    const uint8_t INDICES_INDEX = 2;
    const uint8_t OUTPUT_INDEX = 0;

    const uint8_t N_INDEX = 0;
    const uint8_t C_INDEX = 1;
    const uint8_t M_INDEX = 0;
    const uint8_t NSAMPLE_INDEX = 1;

    const uint8_t FIRST_DIM_INDEX = 0;
    const uint8_t SECOND_DIM_INDEX = 1;
    const uint8_t THIRD_DIM_INDEX = 2;
} // namespace

namespace ops
{
    static ge::graphStatus InferShapeForStackGroupPoints(gert::InferShapeContext *context)
    {
        const gert::Shape *feture_shape = context->GetInputShape(FEATURE_INDEX);
        const gert::Shape *indices_shape = context->GetInputShape(INDICES_INDEX);
        gert::Shape *output_shape = context->GetOutputShape(OUTPUT_INDEX);
        if (feture_shape == nullptr || indices_shape == nullptr || output_shape == nullptr)
        {
            return ge::GRAPH_FAILED;
        }

        int32_t m = indices_shape->GetDim(M_INDEX);
        int32_t nsample = indices_shape->GetDim(NSAMPLE_INDEX);
        int32_t c = feture_shape->GetDim(C_INDEX);

        int8_t output_shape_length = 3;
        output_shape->SetDimNum(output_shape_length);
        output_shape->SetDim(FIRST_DIM_INDEX, m);
        output_shape->SetDim(SECOND_DIM_INDEX, c);
        output_shape->SetDim(THIRD_DIM_INDEX, nsample);

        return GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataTypeForStackGroupPoints(gert::InferDataTypeContext *context)
    {
        const ge::DataType feature_dtype = context->GetInputDataType(FEATURE_INDEX);
        context->SetOutputDataType(OUTPUT_INDEX, feature_dtype);
        return GRAPH_SUCCESS;
    }

    IMPL_OP_INFERSHAPE(StackGroupPoints)
        .InferShape(InferShapeForStackGroupPoints)
        .InferDataType(InferDataTypeForStackGroupPoints);
} // namespace ops