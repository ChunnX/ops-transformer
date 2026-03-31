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
 * \file roi_align_rotated_grad.cpp
 * \brief
 */
#include "register/op_def_registry.h"

using namespace ge;
using namespace std;

namespace ops
{
    class RoiAlignRotatedGrad : public OpDef
    {
    public:
        explicit RoiAlignRotatedGrad(const char *name) : OpDef(name)
        {
            this->Input("x_grad")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("rois")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
            this->Attr("y_grad_shape").AttrType(REQUIRED).ListInt();
            this->Attr("pooled_h").AttrType(REQUIRED).Int();
            this->Attr("pooled_w").AttrType(REQUIRED).Int();
            this->Attr("spatial_scale").AttrType(REQUIRED).Float();
            this->Attr("sampling_ratio").AttrType(REQUIRED).Int();
            this->Attr("aligned").AttrType(REQUIRED).Bool();
            this->Attr("clockwise").AttrType(REQUIRED).Bool();
            this->Output("y_grad")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910_93");
            this->AICore().AddConfig("ascend310p");
        }
    };

    OP_ADD(RoiAlignRotatedGrad);
}