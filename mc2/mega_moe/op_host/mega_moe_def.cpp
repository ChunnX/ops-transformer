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
 * \file MegaMoe.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class MegaMoe : public OpDef {
public:
  explicit MegaMoe(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("expert_ids")
        .ParamType(REQUIRED)
        .DataTypeList({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("weight1")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("weightscales1")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_FLOAT8_E8M0})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("weight2")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("weightscales2")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_FLOAT8_E8M0})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();

    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND});
        
    this->Attr("group_ep").AttrType(REQUIRED).String();
    this->Attr("ep_world_size").AttrType(REQUIRED).Int();
    this->Attr("max_output_size").AttrType(REQUIRED).Int();


    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("prebuildPattern.value", "Opaque")
        .ExtendCfgInfo("jitCompile.flag", "static_true")
        .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");

    this->AICore().AddConfig("ascend950", aicore_config);
    this->MC2().HcclGroup({"group_ep"});
  }
};

OP_ADD(MegaMoe);

} // namespace ops
