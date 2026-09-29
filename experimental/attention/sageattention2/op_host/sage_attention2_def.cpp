/*
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
class SageAttention2 : public OpDef {
public:
    explicit SageAttention2(const char *name)
        : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("lse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("sm_scale").AttrType(REQUIRED).Float();
        this->Attr("smooth_k").AttrType(REQUIRED).Bool();
        this->Attr("is_causal").AttrType(REQUIRED).Bool();
        this->Attr("return_lse").AttrType(REQUIRED).Bool();
        // Encoded enums from the aclnn string parameters; see SageQuantGran
        // and SagePvAccum in op_kernel/params.h.
        this->Attr("qk_quant_gran").AttrType(REQUIRED).Int();
        this->Attr("pv_accum_dtype").AttrType(REQUIRED).Int();
        this->Attr("smooth_v").AttrType(REQUIRED).Bool();
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicShapeSupportFlag(true)
            .DynamicRankSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "sageattention2")
            .ExtendCfgInfo("opInterface.value", "sage_attention2");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(SageAttention2);
} // namespace ops
