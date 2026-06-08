/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_ffn_combine_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
class DispatchFFNCombine : public OpDef {
 public:
  explicit DispatchFFNCombine(const char *name) : OpDef(name) {
    this->Input("context")
        .ParamType(REQUIRED)
        .DataTypeList({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("topk_ids")
        .ParamType(REQUIRED)
        .DataTypeList({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("topk_weights")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("weight1")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ});
    this->Input("weight2")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ});
    this->Input("weight_scales1")
        .ParamType(DYNAMIC)
        .DataTypeList({ge::DT_INT64})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("weight_scales2")
        .ParamType(DYNAMIC)
        .DataTypeList({ge::DT_INT64})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("bias1")
        .ParamType(DYNAMIC)
        .DataTypeList({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("bias2")
        .ParamType(DYNAMIC)
        .DataTypeList({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("x_active_mask")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_INT8})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("scales")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();

    // 输出
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("expert_token_nums")
        .ParamType(REQUIRED)
        .DataTypeList({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND});

    this->Attr("moe_expert_num").AttrType(REQUIRED).Int();
    this->Attr("ep_world_size").AttrType(REQUIRED).Int();
    this->Attr("ccl_buffer_size").AttrType(REQUIRED).Int();
    this->Attr("max_recv_token_num").AttrType(OPTIONAL).Int(0);
    this->Attr("dispatch_quant_mode").AttrType(OPTIONAL).Int(0);
    this->Attr("dispatch_quant_out_type").AttrType(OPTIONAL).Int(static_cast<int>(ge::DT_UNDEFINED));
    this->Attr("combine_quant_mode").AttrType(OPTIONAL).Int(0);
    this->Attr("comm_alg").AttrType(OPTIONAL).String("");
    this->Attr("num_max_token_per_rank").AttrType(OPTIONAL).Int(0);
    this->Attr("activation").AttrType(OPTIONAL).String("");
    this->Attr("activation_clamp").AttrType(OPTIONAL).Float(0.0f);
    this->Attr("gmm2_x_dtype").AttrType(OPTIONAL).Int(static_cast<int>(ge::DT_UNDEFINED));
    this->Attr("transpose_weight1").AttrType(OPTIONAL).Bool(false);
    this->Attr("transpose_weight2").AttrType(OPTIONAL).Bool(false);
    this->Attr("weight1_interleave").AttrType(OPTIONAL).Int(0);

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("prebuildPattern.value", "Opaque")
        .ExtendCfgInfo("jitCompile.flag", "static_false")
        .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
    this->AICore().AddConfig("ascend910_93", aicore_config);
    this->AICore().AddConfig("ascend910b", aicore_config);
  }
};

OP_ADD(DispatchFFNCombine);
}  // namespace ops