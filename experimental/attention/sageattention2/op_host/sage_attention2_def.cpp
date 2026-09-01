/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software and is licensed under the CANN Open Software License Agreement
 * Version 2.0. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND.
 */

#include "register/op_def_registry.h"

namespace ops {
class SageAttention2 : public OpDef {
public:
    explicit SageAttention2(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .Follow("query")
            .AutoContiguous();
        this->Input("value")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .Follow("query")
            .AutoContiguous();

        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .Follow("query");
        // A zero-sized FP32 tensor is used when return_lse is false. Keeping the output present makes the
        // generated aclnn ABI stable and avoids a second kernel signature.
        this->Output("softmax_lse")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        this->Attr("tensor_layout").AttrType(OPTIONAL).String("HND");
        this->Attr("is_causal").AttrType(OPTIONAL).Bool(false);
        this->Attr("qk_quant_gran").AttrType(OPTIONAL).String("per_thread");
        // A non-positive value means 1 / sqrt(original head_dim).
        this->Attr("sm_scale").AttrType(OPTIONAL).Float(0.0F);
        this->Attr("pv_accum_dtype").AttrType(OPTIONAL).String("fp32+fp16");
        this->Attr("smooth_k").AttrType(OPTIONAL).Bool(true);
        this->Attr("smooth_v").AttrType(OPTIONAL).Bool(false);
        this->Attr("return_lse").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(SageAttention2);
} // namespace ops
