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
 * \file moe_distribute_combine_v3_def.cpp
 * \brief
 */

#include "../../common/op_host/mc2_combine_registration.h"
#include "../../common/op_host/mc2_moe_input_registration.h"

namespace ops {
class MoeDistributeCombineV3 : public OpDef {
public:
    explicit MoeDistributeCombineV3(const char *name)
        : OpDef(name)
    {
        this->Input("context")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        Mc2OpDef::DefineCombineRequiredInputs(*this);

        Mc2OpDef::DefineCombineOptionalInputs(*this);

        Mc2OpDef::DefineCombineOutputX(*this);
        DefineAttributes();

        // A3 (arch22): _a3 entry (opFile -> arch22/..._a3.cpp). Reuses the class-level I/O.
        auto aicore_config_a3 = Mc2OpDef::BuildCombineAICoreConfig("static_true", "moe_distribute_combine_v3_a3");

        // A5 (arch35): dedicated kernel entry (opFile -> arch35/..._apt.cpp).
        auto aicore_config_apt = Mc2OpDef::BuildCombineAICoreConfig("static_true", "moe_distribute_combine_v3_apt");

        this->AICore().AddConfig("ascend910_93", aicore_config_a3);
        this->AICore().AddConfig("ascend950", aicore_config_apt);
    }

private:
    void DefineAttributes()
    {
        Mc2OpDef::DefineCombineEpAttributes(*this);
        this->Attr("ccl_buffer_size").AttrType(REQUIRED).Int();
        Mc2OpDef::DefineCombineTpAttributes(*this);
        Mc2OpDef::DefineCombineExpertAttributes(*this, 1);
        Mc2OpDef::DefineCombineOutputAttributes(*this);
        this->Attr("comm_alg").AttrType(OPTIONAL).String("");
        Mc2OpDef::DefineCombineSpecialExpertAttributes(*this);
}
};

OP_ADD(MoeDistributeCombineV3);

} // namespace ops
