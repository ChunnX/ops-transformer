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
 * \file kv_cache_turbo_quant_def.cpp
 * \brief KvCacheTurboQuant 算子原型定义
 */
#include "register/op_def_registry.h"

namespace ops {
class KvCacheTurboQuant : public OpDef {
public:
    explicit KvCacheTurboQuant(const char *name)
        : OpDef(name)
    {
        this->Input("kv_vectors")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rotation_matrix")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("qjl_matrix")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("quant_idx")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND});
        this->Output("quant_qjl")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND});
        this->Output("quant_norm")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Output("quant_gamma")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});

        this->Attr("mse_bits").AttrType(OPTIONAL).Int(3);
        // quant_norm/quant_gamma 元素类型：0=bfloat16（默认），1=float16
        this->Attr("out_dtype").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", aicore_config);
        // ascend910_93 = Atlas 800I A2 推理系列（同为 arch2201/dav_c220），与 ascend910b（800T A2 训练）
        // 共用同一套 A2 形态内核；仓内 experimental/attention 各算子均同时声明两个 SOC。
        // A3 不在支持范围（README 产品表 ×）。
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};
OP_ADD(KvCacheTurboQuant);
} // namespace ops
