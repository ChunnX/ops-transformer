/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tutel_dispatch_def.cpp
 * \brief MoeTutelDispatch op def, Ascend 950 only
 */

#include "register/op_def_registry.h"

namespace ops {
class MoeTutelDispatch : public OpDef {
public:
    explicit MoeTutelDispatch(const char *name)
        : OpDef(name)
    {
        // 与 REG_OP(MoeTutelDispatch) 的 TensorType 严格一致。
        const std::vector<ge::DataType> floatDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
        const std::vector<ge::DataType> indexDtype = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
        const std::vector<ge::Format> ndFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("x").ParamType(REQUIRED).DataType(floatDtype).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("gates").ParamType(REQUIRED).DataType(floatDtype).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("indices").ParamType(REQUIRED).DataType(indexDtype).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Input("locations").ParamType(REQUIRED).DataType(indexDtype).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Output("y").ParamType(REQUIRED).DataType(floatDtype).Format(ndFormat).UnknownShapeFormat(ndFormat);
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
