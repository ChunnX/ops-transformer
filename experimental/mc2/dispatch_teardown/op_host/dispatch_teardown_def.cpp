/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  dispatch_teardown_def.cpp —— DispatchTeardown(mxfp8 多核) 算子定义。
 *  mxfp8：token = fp8(e4m3fn) payload + e8m0 scale，两者各独立 GM、随 token 一起重排（无算术，纯字节搬）。
 *  纯 AIV、不挂 MC2().HcclGroup()。opFile.value="dispatch_teardown" 关联 kernel。
 */
#include "register/op_def_registry.h"

namespace ops {
class DispatchTeardown : public OpDef {
 public:
  explicit DispatchTeardown(const char *name) : OpDef(name) {
    // ---- 输入 ----
    this->Input("token_data")          // [R*E*bs, h] fp8_e4m3fn 收到的数据区 payload(expert-major 块序 le*R+src, le∈[0,E)本地专家 src∈[0,R)源卡, 每块 bs 槽; 与 expand_x 同序, teardown 无转置)
        .ParamType(REQUIRED).DataType({ge::DT_FLOAT8_E4M3FN}).FormatList({ge::FORMAT_ND});
    this->Input("token_scale")         // [R*E*bs, h/32] e8m0 收到的数据区 scale(与 payload 同块结构, expert-major)
        .ParamType(REQUIRED).DataType({ge::DT_FLOAT8_E8M0}).FormatList({ge::FORMAT_ND});
    this->Input("wish_cnt")            // [R*E] int32 每块实收 token 数(expert-major: idx=le*R+src, 与数据区块同序)
        .ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    this->Input("expert_ids")          // [bs, topk] int32 本卡 token 的全局专家选择。
                                       // 【调用方契约】同一 token 的 topk 个专家各不相同(distinct) 且 id∈[0,GE=R*E)（隐含 topk≤GE）；
                                       //   teardown 假定合法、不校验：负 id 会致 counter[ge] 越界，重复会破坏出现次数语义。暂不支持 -1(丢弃 token)。
        .ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

    // ---- 输出 ----
    this->Output("expand_x")           // [A, h] fp8_e4m3fn 紧密重排(expert-major) payload
        .ParamType(REQUIRED).DataType({ge::DT_FLOAT8_E4M3FN}).FormatList({ge::FORMAT_ND});
    this->Output("expand_scale")       // [A, h/32] e8m0 紧密重排 scale(随 payload 同序)
        .ParamType(REQUIRED).DataType({ge::DT_FLOAT8_E8M0}).FormatList({ge::FORMAT_ND});
    this->Output("send_counts")        // [R*E] int32 inclusive 前缀和(expert-major 序)
        .ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    this->Output("expand_idx")         // [bs, topk] int32 出现次数(0-based)
        .ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    this->Output("expert_token_nums")  // [E] int32 本卡每个本地专家收到 token 总数
        .ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

    // ---- attr ----
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
        .ExtendCfgInfo("opFile.value", "dispatch_teardown");
    this->AICore().AddConfig("ascend950", aicore_config);        // a5
    this->AICore().AddConfig("ascend910_96", aicore_config);     // a6
  }
};

OP_ADD(DispatchTeardown);
}  // namespace ops
