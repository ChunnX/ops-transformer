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
 * \file moe_tutel_dispatch_def.cpp
 * \brief MoeTutelDispatch op def, Ascend950 (arch35) only
 */

#include "register/op_def_registry.h"

namespace ops {
// 与 REG_OP(MoeTutelDispatch) 的 TensorType 严格一致：float16 / float32 / bfloat16
static const std::vector<ge::DataType> FLOAT_DTYPE = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
static const std::vector<ge::DataType> INDEX_DTYPE = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
static const std::vector<ge::Format> ND_FORMAT = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class MoeTutelDispatch : public OpDef {
public:
    explicit MoeTutelDispatch(const char *name)
        : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(FLOAT_DTYPE)
            .Format(ND_FORMAT)
            .UnknownShapeFormat(ND_FORMAT);
        this->Input("gates")
            .ParamType(REQUIRED)
            .DataType(FLOAT_DTYPE)
            .Format(ND_FORMAT)
            .UnknownShapeFormat(ND_FORMAT);
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType(INDEX_DTYPE)
            .Format(ND_FORMAT)
            .UnknownShapeFormat(ND_FORMAT);
        this->Input("locations")
            .ParamType(REQUIRED)
            .DataType(INDEX_DTYPE)
            .Format(ND_FORMAT)
            .UnknownShapeFormat(ND_FORMAT);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(FLOAT_DTYPE)
            .Format(ND_FORMAT)
            .UnknownShapeFormat(ND_FORMAT);
        this->Attr("capacity").AttrType(REQUIRED).Int();

        OpAICoreConfig config950;
        config950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "moe_tutel_dispatch");
        this->AICore().AddConfig("ascend950", config950);
    }
};

OP_ADD(MoeTutelDispatch);
} // namespace ops
