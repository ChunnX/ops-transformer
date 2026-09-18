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
 * \file moe_distribute_combine_add_rms_norm_def.cpp
 * \brief
 */

#include "../../common/op_host/mc2_combine_registration.h"

namespace ops {
class MoeDistributeCombineAddRmsNorm : public OpDef {
public:
    explicit MoeDistributeCombineAddRmsNorm(const char *name)
        : OpDef(name)
    {
        DefineRequiredInputs();
        DefineOptionalInputs();
        DefineOutputs();
        DefineAttributes();

        // A3 (arch22): _a3 entry (opFile -> arch22/..._a3.cpp). Reuses the class-level I/O.
        auto aicore_config_a3 = Mc2OpDef::BuildCombineAICoreConfig("static_true", "moe_distribute_combine_add_rms_norm_a3");

        // A5 (arch35): dedicated kernel entry (opFile -> arch35/..._apt.cpp).
        auto aicore_config_apt = Mc2OpDef::BuildCombineAICoreConfig("static_true", "moe_distribute_combine_add_rms_norm_apt");

        this->AICore().AddConfig("ascend910_93", aicore_config_a3);
        this->AICore().AddConfig("ascend950", aicore_config_apt);
        this->MC2().HcclGroup({"group_ep", "group_tp"});
    }

private:
    void DefineRequiredInputs()
    {
        this->Input("expand_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("assist_info_for_combine")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ep_send_counts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_scales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("residual_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
    }

    void DefineOptionalInput(const char *name, ge::DataType dataType)
    {
        this->Input(name)
            .ParamType(OPTIONAL)
            .DataType({dataType})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
    }

    void DefineOptionalInputs()
    {
        DefineOptionalInput("tp_send_counts", ge::DT_INT32);
        DefineOptionalInput("x_active_mask", ge::DT_BOOL);
        DefineOptionalInput("activation_scale", ge::DT_FLOAT);
        DefineOptionalInput("weight_scale", ge::DT_FLOAT);
        DefineOptionalInput("group_list", ge::DT_INT64);
        DefineOptionalInput("expand_scales", ge::DT_FLOAT);
        DefineOptionalInput("shared_expert_x", ge::DT_BF16);
        DefineOptionalInput("elastic_info", ge::DT_INT32);
        DefineOptionalInput("ori_x", ge::DT_BF16);
        DefineOptionalInput("const_expert_alpha_1", ge::DT_BF16);
        DefineOptionalInput("const_expert_alpha_2", ge::DT_BF16);
        DefineOptionalInput("const_expert_v", ge::DT_BF16);
    }

    void DefineOutputs()
    {
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("rstdOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
    }

    void DefineAttributes()
    {
        Mc2OpDef::DefineCombineAttributes(*this, 0);
        this->Attr("comm_alg").AttrType(OPTIONAL).String("");
        this->Attr("norm_eps").AttrType(OPTIONAL).Float(1e-6f);
        Mc2OpDef::DefineCombineSpecialExpertAttributes(*this);
}
};

OP_ADD(MoeDistributeCombineAddRmsNorm);

} // namespace ops
