/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_bicubic2d_aa_grad_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> xDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
static const std::vector<ge::Format> xFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class UpsampleBicubic2dAAGrad : public OpDef {
public:
    explicit UpsampleBicubic2dAAGrad(const char *name) : OpDef(name)
    {
        this->Input("grad_output")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat);
        this->Output("grad_input")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat);
        this->Attr("output_size").AttrType(REQUIRED).ListInt();
        this->Attr("input_size").AttrType(REQUIRED).ListInt();
        this->Attr("align_corners").AttrType(OPTIONAL).Bool(false);
        this->Attr("scales_h").AttrType(OPTIONAL).Float();
        this->Attr("scales_w").AttrType(OPTIONAL).Float();
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(UpsampleBicubic2dAAGrad);
}  // namespace ops
