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
 * \file ffn_to_attention_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "../../common/op_host/attention_ffn_op_def.h"

namespace ops {
class FFNToAttention : public OpDef {
public:
    explicit FFNToAttention(const char *name)
        : OpDef(name)
    {
        DefineRequiredInputs();
        this->Input("attn_rank_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        DefineAttributes();

        AttentionFfnOpDef::AddAICoreConfig(*this, "ascend910_93");
        this->MC2().HcclGroup({"group"});
    }

private:
    void DefineRequiredInputs()
    {
        AttentionFfnOpDef::DefineXInput(*this);
        this->Input("session_ids")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("micro_batch_ids")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("token_ids")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_offsets")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_token_num")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
    }

    void DefineAttributes()
    {
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("world_size").AttrType(REQUIRED).Int();
        this->Attr("token_info_table_shape").AttrType(REQUIRED).ListInt();
        this->Attr("token_data_shape").AttrType(REQUIRED).ListInt();
    }
};

OP_ADD(FFNToAttention);
} // namespace ops
