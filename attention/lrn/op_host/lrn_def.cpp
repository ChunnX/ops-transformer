/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lrn_def.cpp
 * \brief LRN op definition: TF across-channels local response
 * normalization on 4-D NCHW tensors (PyPTO-Pro kernel delivery).
 */
#include "register/op_def_registry.h"

namespace ops {
class LRN : public OpDef {
public:
    explicit LRN(const char *name) : OpDef(name)
    {
        // Inputs. Dtype lists are positional combinations (column-wise), giving the
        // 2 supported same-dtype signatures: (fp16, fp16) and (fp32, fp32).
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // Attr order is fixed — the host tiling reads by index
        // (0=depth_radius, 1=bias, 2=alpha, 3=beta, 4=norm_region).
        this->Attr("depth_radius").AttrType(OPTIONAL).Int(5);
        this->Attr("bias").AttrType(OPTIONAL).Float(1.0);
        this->Attr("alpha").AttrType(OPTIONAL).Float(1.0);
        this->Attr("beta").AttrType(OPTIONAL).Float(0.5);
        this->Attr("norm_region").AttrType(OPTIONAL).String("ACROSS_CHANNELS");
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "lrn")
            .ExtendCfgInfo("opInterface.value", "lrn");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(LRN);
} // namespace ops
