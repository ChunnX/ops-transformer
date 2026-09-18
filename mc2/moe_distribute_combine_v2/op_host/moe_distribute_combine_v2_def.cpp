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
 * \file moe_distribute_combine_v2_def.cpp
 * \brief
 */

#include "../../common/op_host/mc2_combine_registration.h"
#include "../../common/op_host/mc2_moe_input_registration.h"

namespace ops {
class MoeDistributeCombineV2 : public OpDef {
public:
    explicit MoeDistributeCombineV2(const char *name)
        : OpDef(name)
    {
        Mc2OpDef::DefineCombineRequiredInputs(*this);
        Mc2OpDef::DefineCombineOptionalInputs(*this);

        Mc2OpDef::DefineCombineOutputX(*this);
        DefineAttributes();

        // A2 (arch22): _a2 entry (opFile -> arch22/..._a2.cpp).
        this->AICore().AddConfig("ascend910b", Mc2OpDef::BuildCombineAICoreConfig("static_false", "moe_distribute_combine_v2_a2"));
        // A3 (arch22): _a3 entry (opFile -> arch22/..._a3.cpp).
        this->AICore().AddConfig("ascend910_93", Mc2OpDef::BuildCombineAICoreConfig("static_true", "moe_distribute_combine_v2_a3"));
        // A5 (arch35): _apt entry (opFile -> arch35/..._apt.cpp).
        this->AICore().AddConfig("ascend950", Mc2OpDef::BuildCombineAICoreConfig("static_true", "moe_distribute_combine_v2_apt"));

        this->MC2().HcclGroup({"group_ep", "group_tp"});
    }

private:
    void DefineAttributes()
    {
        Mc2OpDef::DefineCombineAttributes(*this, 1);
        this->Attr("comm_alg").AttrType(OPTIONAL).String("");
        Mc2OpDef::DefineCombineSpecialExpertAttributes(*this);
}
};

OP_ADD(MoeDistributeCombineV2);

} // namespace ops
