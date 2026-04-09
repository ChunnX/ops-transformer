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
 * \file attention_pioneer_metadata_def.cpp
 * \brief OpDef registration for AttentionPioneerMetadata AICPU operator.
 */
#include "register/op_def_registry.h"

namespace ops {
class AttentionPioneerMetadata : public OpDef {
public:
    AttentionPioneerMetadata(const char *name) : OpDef(name)
    {
        this->Input("actual_seq_lengths_q")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("actual_seq_lengths_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Output("metaData_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Attr("num_heads_q").AttrType(REQUIRED).Int(0);
        this->Attr("num_heads_kv").AttrType(REQUIRED).Int(1);
        this->Attr("head_dim_qk").AttrType(REQUIRED).Int(0);
        this->Attr("head_dim_v").AttrType(REQUIRED).Int(0);
        this->Attr("batch_size").Int(0);
        this->Attr("sparse_mode").Int(0);
        this->Attr("pre_tokens").Int(2147483647);
        this->Attr("next_tokens").Int(2147483647);
        this->Attr("input_layout").String("TND");
        this->Attr("k_sink_num").Int(0);
        this->Attr("rope_head_dim").Int(0);
        this->Attr("block_size").Int(0);
        this->Attr("soc_version").AttrType(REQUIRED).String("ascend950");
        this->Attr("aic_core_num").AttrType(REQUIRED).Int(36);
        this->Attr("aiv_core_num").AttrType(REQUIRED).Int(72);

        this->SetInferShape(ge::InferShape);
    }
};

OP_ADD(AttentionPioneerMetadata);
} // namespace ops
