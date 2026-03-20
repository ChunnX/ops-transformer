/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotate_dynamic_quant_def.cpp
 * \brief 算子定义文件，定义算子的输入输出、数据类型、格式等
 */
#include "register/op_def_registry.h"

namespace ops {
class RotateDynamicQuant : public OpDef {
public:
    explicit RotateDynamicQuant(const char* name) : OpDef(name)
    {
        // 输入 x: [M, N], DT_FLOAT16/DT_BFLOAT16, FORMAT_ND
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BFLOAT16, ge::DT_BFLOAT16, ge::DT_BFLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 输入 rot: [N, N], DT_FLOAT16/DT_BFLOAT16, FORMAT_ND (与x保持一致)
        this->Input("rot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BFLOAT16, ge::DT_BFLOAT16, ge::DT_BFLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 输出 yOut: [M, N], DT_INT8/DT_INT32, FORMAT_ND
        this->Output("yOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT32, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 输出 scaleOut: [M], DT_FLOAT, FORMAT_ND
        this->Output("scaleOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 属性 dst_type: 指定输出y的数据类型
        this->Attr("dst_type")
            .AttrType(OPTIONAL)
            .Int(0);

        // OpAICoreConfig原型注册与管理
        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false);

        // 添加支持的芯片型号
        this->AICore().AddConfig("ascend910b", aicConfig);
    }
};

// 注册算子
OP_ADD(RotateDynamicQuant);

} // namespace ops
