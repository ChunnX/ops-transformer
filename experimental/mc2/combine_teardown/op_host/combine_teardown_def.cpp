/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  combine_teardown_def.cpp —— CombineTeardown(mxfp8 多核) 算子定义（aclnn 注册式，纯 AIV）。
 *  mxfp8：combine 数据区 = fp8(e4m3fn) payload + e8m0 scale；gather k 份复制体 → 逐组(32)反量化 → 加权和 → x[bs,h] bf16。
 *  纯 AIV、不挂 MC2().HcclGroup()。
 */
#include "register/op_def_registry.h"

namespace ops {
class CombineTeardown : public OpDef {
 public:
  explicit CombineTeardown(const char *name) : OpDef(name) {
    this->Input("combine_data")        // [GE*bs, h] fp8_e4m3fn combine 数据区 payload(每全局专家一块,块 bs 槽,块内 expand_idx 定位)
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT8_E4M3FN})
        .FormatList({ge::FORMAT_ND});
    this->Input("combine_scale")       // [GE*bs, h/32] e8m0 combine 数据区 scale(与 payload 同块结构)
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT8_E8M0})
        .FormatList({ge::FORMAT_ND});
    this->Input("expert_scales")       // [bs, topk] fp32 每 token 对 k 专家的权重
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND});
    this->Input("expand_idx")          // [bs, topk] int32 块内位置(dispatch_teardown 输出)
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND});
    this->Input("expert_ids")          // [bs, topk] int32 全局专家 id(定位块)。
                                       // 【调用方契约】同一 token 的 topk 个专家各不相同(distinct) 且 id∈[0,GE=R*E)（隐含 topk≤GE）；
                                       //   teardown 假定合法、不校验：负 id 会致 ge*bs+pos 越界读 combine 数据区。暂不支持 -1(丢弃 token)。
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND});

    this->Output("x")                  // [bs, h] bf16 加权求和还原
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .FormatList({ge::FORMAT_ND});

    this->Attr("rank_num").AttrType(REQUIRED).Int();          // R
    this->Attr("expert_per_rank").AttrType(REQUIRED).Int();   // E

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("jitCompile.flag", "static_false")
        .ExtendCfgInfo("opFile.value", "combine_teardown");
    this->AICore().AddConfig("ascend950", aicore_config);        // a5
    this->AICore().AddConfig("ascend910_96", aicore_config);     // a6
  }
};

OP_ADD(CombineTeardown);
}  // namespace ops
