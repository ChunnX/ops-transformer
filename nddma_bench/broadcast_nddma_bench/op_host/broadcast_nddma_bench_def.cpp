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
 * \file broadcast_nddma_bench_def.cpp
 * \brief BroadcastNddmaBench ascendc op def（仅 ascend950 / arch35）
 *
 * Attr 声明顺序即 tiling/aclnn 的 attr index 顺序：is_mte3, scene, dim2, dim3, dim4。
 * 广播算子输入 shape 的广播轴折叠为 1，输出满 5D 的 dim2/dim3/dim4 无法从 input 恢复，
 * 故通过 Attr 传入；scene 选择四种广播模式（0=AB/1=BA/2=BAB/3=ABA）。
 */

#include "register/op_def_registry.h"

namespace ops {
class BroadcastNddmaBench : public OpDef {
public:
    explicit BroadcastNddmaBench(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("is_mte3").AttrType(OPTIONAL).Int(1);
        this->Attr("scene").AttrType(OPTIONAL).Int(0);
        this->Attr("dim2").AttrType(OPTIONAL).Int(0);
        this->Attr("dim3").AttrType(OPTIONAL).Int(0);
        this->Attr("dim4").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig config_95;
        config_95.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "broadcast_nddma_bench_apt");

        this->AICore().AddConfig("ascend950", config_95);
    }
};

OP_ADD(BroadcastNddmaBench);
}  // namespace ops
