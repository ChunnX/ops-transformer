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
 * \file kda_input_proj_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class KdaInputProj : public OpDef {
public:
    explicit KdaInputProj(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight_qkv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight_beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight_gate")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight_g")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("scale_qkv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("qkv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Output("gate")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("g")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});

        this->Attr("trans_weight_qkv").AttrType(OPTIONAL).Bool(true);
        this->Attr("trans_weight_beta").AttrType(OPTIONAL).Bool(true);
        this->Attr("trans_weight_gate").AttrType(OPTIONAL).Bool(true);
        this->Attr("trans_weight_g").AttrType(OPTIONAL).Bool(true);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(KdaInputProj);
} // namespace ops
