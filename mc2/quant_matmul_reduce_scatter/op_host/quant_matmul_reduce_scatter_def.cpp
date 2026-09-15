/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_reduce_scatter_def.cpp
 * \brief OpDef definition for QuantMatmulReduceScatter (MX-quant FP8, apace UDMA path, compute-first)
 */

#include <register/op_def_registry.h>

namespace ops {
class QuantMatmulReduceScatter : public OpDef {
public:
    explicit QuantMatmulReduceScatter(const char *name)
        : OpDef(name)
    {
        this->Input("context")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("x1_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x2_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("hccl_buffer_size").AttrType(REQUIRED).Int();
        this->Attr("is_trans_a").AttrType(OPTIONAL).Bool(false);
        this->Attr("is_trans_b").AttrType(OPTIONAL).Bool(false);
        this->Attr("rank_size").AttrType(OPTIONAL).Int(0);
        this->Attr("group_size").AttrType(OPTIONAL).Int(0);
        this->Attr("y_dtype").AttrType(OPTIONAL).Int(static_cast<int64_t>(ge::DT_UNDEFINED));
        this->Attr("comm_mode").AttrType(OPTIONAL).String("ai_cpu");

        OpAICoreConfig aicoreConfig_950;
        aicoreConfig_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false") // 动态shape,复用二进制,后续图支持后修改
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel")
            .ExtendCfgInfo("opFile.value", "quant_matmul_reduce_scatter_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig_950);
    }
};

OP_ADD(QuantMatmulReduceScatter);
} // namespace ops
