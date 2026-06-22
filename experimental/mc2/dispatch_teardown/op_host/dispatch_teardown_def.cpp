/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  dispatch_teardown_def.cpp —— DispatchTeardown 算子定义（aclnn 注册式，纯 AIV）。
 *  对标 all_gather_matmul_v2_def.cpp 的 OpDef 写法，裁剪为最小：
 *    - 1 输入 status：本卡状态区 GM（CCU dispatch_setup 用 LocalCopyNb 写 1 的同一块），int32；
 *    - 1 输出 done  ：AIV 轮询到 1 后写回的完成标志（供 host 校验），int32；
 *    - 无 attr；
 *    - 纯 AIV 计算，【不】挂 this->MC2().HcclGroup()（本算子只读 GM、不走通信引擎）。
 *  opFile.value = "dispatch_teardown" 关联 op_kernel/dispatch_teardown.cpp 的核函数。
 */
#include "register/op_def_registry.h"

namespace ops {
class DispatchTeardown : public OpDef {
 public:
  explicit DispatchTeardown(const char *name) : OpDef(name) {
    this->Input("status")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND});

    this->Output("done")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND});

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
    this->AICore().AddConfig("ascend950", aicore_config);
  }
};

OP_ADD(DispatchTeardown);
}  // namespace ops
