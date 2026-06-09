/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_fetch_wait_def.cpp
 * \brief EngramFetchWait 算子定义（极简骨架版）
 */

#include "register/op_def_registry.h"

namespace ops {
class EngramFetchWait : public OpDef {
public:
    explicit EngramFetchWait(const char *name) : OpDef(name)
    {
        this->Input("handle").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(false)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_true");

        this->AICore().AddConfig("ascend910b", aicore_config);
    }
};

OP_ADD(EngramFetchWait);

} // namespace ops