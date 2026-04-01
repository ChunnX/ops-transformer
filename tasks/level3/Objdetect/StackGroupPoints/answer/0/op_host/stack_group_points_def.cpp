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
 * \file stack_group_points_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops
{
    class StackGroupPoints : public OpDef
    {
    public:
        explicit StackGroupPoints(const char *name) : OpDef(name)
        {
            this->Input("features")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("features_batch_cnt")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("indices")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("indices_batch_cnt")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910_93");
            this->AICore().AddConfig("kirinx90");
        }
    };

    OP_ADD(StackGroupPoints);
} // namespace ops