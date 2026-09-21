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
 * \file attention_to_ffn_v2_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "../../common/op_host/attention_ffn_op_def.h"

namespace ops {
class AttentionToFfnV2 : public OpDef {
public:
    explicit AttentionToFfnV2(const char *name)
        : OpDef(name)
    {
        this->Input("context")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        AttentionFfnOpDef::DefineXInput(*this);
        AttentionFfnOpDef::DefineAttentionFfRequiredInput(*this);
        AttentionFfnOpDef::DefineOptionalInputs(*this);
        AttentionFfnOpDef::DefineAttributes(*this);
        this->Attr("ccl_buffer_size").AttrType(REQUIRED).Int();

        AttentionFfnOpDef::AddAICoreConfig(*this, "ascend950");
    }
};

OP_ADD(AttentionToFfnV2);
} // namespace ops
