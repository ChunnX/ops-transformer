/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
class BandwidthTest : public OpDef {
public:
  explicit BandwidthTest(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("expert_id")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .AutoContiguous();

    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});
    this->Output("receive_cnt")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND});

    this->Attr("group").AttrType(REQUIRED).String();
    this->Attr("world_size").AttrType(REQUIRED).Int();
    this->Attr("rank_id").AttrType(REQUIRED).Int();
    this->Attr("max_bs").AttrType(REQUIRED).Int();
    this->Attr("mode").AttrType(REQUIRED).Int();

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("prebuildPattern.value", "Opaque")
        .ExtendCfgInfo("jitCompile.flag", "static_false");

    this->AICore().AddConfig("ascend950", aicore_config);
    this->MC2().HcclGroup({"group"});
  }
};

OP_ADD(BandwidthTest);

} // namespace ops