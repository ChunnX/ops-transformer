/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_def_registry.h"

namespace ops {
class TurboQuantAttentionScore : public OpDef {
public:
    explicit TurboQuantAttentionScore(const char *name) : OpDef(name)
    {
        for (const char *input : {"query_rotated", "query_qjl"}) {
            Input(input).ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        }
        for (const char *input : {"key_cache_idx", "key_cache_qjl"}) {
            Input(input).ParamType(REQUIRED).DataType({ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        }
        for (const char *input : {"key_cache_norm", "key_cache_gamma"}) {
            Input(input).ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        }
        Input("block_table").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        Input("seq_lens").ParamType(REQUIRED).ValueDepend(OPTIONAL, DependScope::TILING)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        Output("attn_scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        Attr("mse_bits").AttrType(OPTIONAL).Int(3);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(false);
        AICore().AddConfig("ascend910b", config);
    }
};
OP_ADD(TurboQuantAttentionScore);
} // namespace ops
