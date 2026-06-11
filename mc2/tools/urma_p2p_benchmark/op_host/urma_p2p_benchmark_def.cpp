/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
class UrmaP2pBenchmark : public OpDef {
public:
  explicit UrmaP2pBenchmark(const char* name) : OpDef(name) {
    // 输入: x(发送数据)
    // context输入已移除：HCCL上下文由MC2框架基于.MC2().HcclGroup声明自动创建
    // dst_rank_id使用属性而非输入，P2P场景目标rank为固定值
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .AutoContiguous();

    // 输出: y(接收数据), receive_cnt(各rank接收token计数)
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});
    this->Output("receive_cnt")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND});
    this->Output("performance_info")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND});

    // 属性: group(通信域), world_size(rank数), max_bs(最大batch), mode(测试模式),
    //        ccl_buffer_size(HCCL缓冲区), dst_rank_id(P2P目标rank), aiv_num(AIV核数)
    // 注意: 删除comm_alg属性(固定为urma via SQE级), 新增dst_rank_id属性
    this->Attr("group").AttrType(REQUIRED).String();
    this->Attr("world_size").AttrType(REQUIRED).Int();
    this->Attr("max_bs").AttrType(REQUIRED).Int();
    this->Attr("mode").AttrType(REQUIRED).Int();
    this->Attr("ccl_buffer_size").AttrType(REQUIRED).Int();
    this->Attr("dst_rank_id").AttrType(OPTIONAL).Int(-1);
    this->Attr("aiv_num").AttrType(OPTIONAL).Int(-1);

    // SQE级HcclCombinOpParam URMA通信：声明MC2框架自动创建HCCL上下文
    this->MC2().HcclGroup({"group"});

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
  }
};

OP_ADD(UrmaP2pBenchmark);

} // namespace ops