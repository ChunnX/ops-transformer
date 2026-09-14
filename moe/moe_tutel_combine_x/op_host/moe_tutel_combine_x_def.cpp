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
 * \file moe_tutel_combine_x_def.cpp
 * \brief OpDef definition of the MoeTutelCombineX operator.
 */
#include "register/op_def_registry.h"

namespace ops {

class MoeTutelCombineX : public OpDef {
public:
    explicit MoeTutelCombineX(const char *name)
        : OpDef(name)
    {
        // —— Input 声明：ParamType + DataType（3 组组合列展开，列 i 对应第 i 组组合；
        //    indices / locations 恒 int32）+ Format + UnknownShapeFormat，所有 IO 等长 ——
        this->Input("y_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("locations")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // —— Output 声明：同 Input 结构；x_grad 与 y_grad / gates 同 dtype（组合列对位），
        //    shape = gates.shape + (y_grad.shape[-1],)，aliasing: none ——
        this->Output("x_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // —— Attr 声明 ——
        // 无任何 ATTR：capacity C 不走算子属性，由 EC = y_grad.shape[-2]、
        // E = max(indices)+1、C = floor(EC/E) 在 tiling / kernel 侧推导。

        // —— AICore 配置（Ascend950PR / Ascend950DT 同 3510 架构，统一 soc 名 ascend950）——
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig
            .DynamicCompileStaticFlag(
                true)                      // 静态 shape 统一走动态 kernel（binary json shape [-2]），避免 per-shape JIT
            .DynamicFormatFlag(false)      // 固定 ND，不支持动态格式转换
            .DynamicRankSupportFlag(true)  // 支持双 rank 形态：y_grad 2D/3D、元数据 1D/2D
            .DynamicShapeSupportFlag(true) // S / H / EC / B 每步可变（动态 shape）
            .NeedCheckSupportFlag(false)   // 跳过框架 support check（rank/同形/dtype 契约由 L2 aclnn 自检）
            .PrecisionReduceFlag(false)    // 禁止隐式降精度：fp32 主梯度值域可超 fp16 表示域（±65504），
                                           // 降档会破坏数值契约；3 dtype 均有精确 kernel，无需降级回退
            .ExtendCfgInfo("opFile.value",
                           "moe_tutel_combine_x_apt"); // 指向 kernel 入口文件 moe_tutel_combine_x_apt.cpp
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};

OP_ADD(MoeTutelCombineX);
} // namespace ops
