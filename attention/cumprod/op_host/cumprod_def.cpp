/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cumprod_def.cpp
 * \brief Cumprod op definition: y = cumprod(x, axis). Signature mirrors the
 *        built-in Cumprod IR (x + axis tensor inputs, exclusive/reverse attrs)
 *        so the custom package coexists with / overrides the built-in op.
 */

#include "register/op_def_registry.h"

namespace ops {
class Cumprod : public OpDef {
public:
    explicit Cumprod(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // axis mirrors the built-in Cumprod IR (INPUT axis, IndexNumberType): the aclnn layer
        // materializes the api scalar into an int64 const tensor input. ValueDepend(REQUIRED)
        // is the official declaration for value-dependent tensor inputs (SplitV2 size_splits /
        // split_dim pattern): the host tiling reads the value to fold the scan axes.
        this->Input("axis")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("exclusive").AttrType(OPTIONAL).Bool(false);
        this->Attr("reverse").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "false")
            .ExtendCfgInfo("opFile.value", "cumprod");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(Cumprod);
} // namespace ops
