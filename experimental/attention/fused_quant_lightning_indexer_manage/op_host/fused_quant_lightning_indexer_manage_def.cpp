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
 * \file fused_quant_lightning_indexer_manage_def.cpp
 * \brief Operator definition for fused quantized Lightning Indexer cache management.
 */

#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class FusedQuantLightningIndexerManage : public OpDef {
public:
    explicit FusedQuantLightningIndexerManage(const char *name) : OpDef(name)
    {
        const std::vector<ge::Format> nd = {ge::FORMAT_ND};
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        this->Input("index_weights").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format(nd).IgnoreContiguous();
        this->Input("query_dequant_scale").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format(nd);
        this->Input("query").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN}).Format(nd);
        this->Input("index_key_dequant_scale").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format(nd).IgnoreContiguous();
        this->Input("index_key_cache").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN}).Format(nd).IgnoreContiguous();
        this->Input("index_block_table").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Input("actual_seq_lengths_query").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Input("actual_seq_lengths_key").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Input("offload_seq_lengths_key").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Input("num_cache_tokens").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Input("request_state").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Input("req_pool_entries").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Input("cache_slots_pool").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Attr("key_stride").Int();
        this->Attr("scale_stride").Int();
        this->Attr("weight_stride").Int();
        this->Output("topk_src_ids").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Output("topk_dst_slots").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Output("topk_miss_counts").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Output("miss_src_ids").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Output("miss_dst_slots").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Output("miss_counts").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->Output("cache_slots_pool_out").ParamType(REQUIRED).DataType(ints).Format(nd);
        this->AICore().AddConfig("ascend950");
    }
};
OP_ADD(FusedQuantLightningIndexerManage);
} // namespace ops
