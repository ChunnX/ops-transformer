/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class GenericLightningIndexer : public OpDef {
public:
    explicit GenericLightningIndexer(const char *name) : OpDef(name)
    {
        // A2/A3: LI FP16/BF16 ± FP32 weights, QLI INT8, MSA FP16×INT8.
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("weights")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("actual_seq_k")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("seqused_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("seqused_k")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("cmp_residual_k")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("q_descale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k_descale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("output_idx_offset")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("metadata")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("start_loc")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Output("sparse_indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("sparse_values")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        this->Attr("topk").AttrType(OPTIONAL).Int(2048);
        this->Attr("cmp_ratio").AttrType(OPTIONAL).Int(1);
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_k").AttrType(OPTIONAL).String("BSND");
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("act_type").AttrType(OPTIONAL).Int(0); // 0 relu, 1 identity
        this->Attr("has_weighted_sum").AttrType(OPTIONAL).Bool(true);
        this->Attr("expand_indices").AttrType(OPTIONAL).Bool(false);
        this->Attr("append_tail").AttrType(OPTIONAL).Bool(false);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(-1);
        this->Attr("return_value").AttrType(OPTIONAL).Bool(false);
        this->Attr("select_mode").AttrType(OPTIONAL).Int(0);     // 0 topk, 1 none (step2)
        this->Attr("key_reduce_mode").AttrType(OPTIONAL).Int(0); // 0 none, 1 maxpool (step2)
        this->Attr("score_scale").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("key_stride0").AttrType(OPTIONAL).Int(-1);
        this->Attr("k_descale_stride0").AttrType(OPTIONAL).Int(-1);
        this->Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1);
        this->Attr("init_blocks").AttrType(OPTIONAL).Int(0);
        this->Attr("local_blocks").AttrType(OPTIONAL).Int(0);

        // 仅 950。A2/A3 不注册。
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_INT8,
                       ge::DT_INT8, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_INT8,
                       ge::DT_INT8, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_INT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicoreConfig950.Input("weights")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("q_descale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("k_descale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};
OP_ADD(GenericLightningIndexer);
} // namespace ops
