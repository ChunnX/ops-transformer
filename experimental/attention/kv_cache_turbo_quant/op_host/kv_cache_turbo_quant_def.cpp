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
 * \brief KvCacheTurboQuant op definition
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
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rotation_matrix")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("qjl_matrix")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("quant_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND});
        this->Output("quant_qjl")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND});
        this->Output("quant_norm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Output("quant_gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});

        this->Attr("mse_bits").AttrType(OPTIONAL).Int(3);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false);
        // NOTE: no "aclnnSupport.value" cfg here - the op ships a hand-written
        // op_api (op_api/aclnn_kv_cache_turbo_quant.cpp); the cfg would make the
        // build ALSO autogenerate nnopbase aclnn glue and the two collide at
        // link time (duplicate aclnnKvCacheTurboQuant* symbols).
        this->AICore().AddConfig("ascend910b", aicore_config);
    }
};
OP_ADD(KvCacheTurboQuant);
} // namespace ops
