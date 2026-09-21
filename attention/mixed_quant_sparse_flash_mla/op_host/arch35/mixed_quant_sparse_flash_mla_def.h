/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_ARCH35_DEF_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_ARCH35_DEF_H

#include "register/op_def_registry.h"

namespace ops {
namespace arch35 {
inline void AddConfig(OpDef &op)
{
    OpAICoreConfig aicore_config_950;
    aicore_config_950.Input("q")
        .ParamType(REQUIRED)
        .DataTypeList({ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    aicore_config_950.Input("ori_kv")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_FLOAT8_E4M3FN})
        .FormatList({ge::FORMAT_ND})
        .IgnoreContiguous();
    aicore_config_950.Input("cmp_kv")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_FLOAT8_E4M3FN})
        .FormatList({ge::FORMAT_ND})
        .IgnoreContiguous();
    aicore_config_950.Output("attn_out")
        .ParamType(REQUIRED)
        .DataTypeList({ge::DT_BF16})
        .FormatList({ge::FORMAT_ND});
    aicore_config_950.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
    op.AICore().AddConfig("ascend950", aicore_config_950);
}
} // namespace arch35
} // namespace ops

#endif
