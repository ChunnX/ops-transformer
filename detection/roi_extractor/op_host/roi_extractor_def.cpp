/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_extractor_def.cpp
 * \brief RoiExtractor op definition: 8 OPTIONAL features (padded fixed-arity
 *        TensorList expansion), REQUIRED rois, OPTIONAL index, OUTPUT y; attr
 *        order frozen by SPEC §5.
 */

#include "register/op_def_registry.h"

namespace ops {
class RoiExtractor : public OpDef {
public:
    explicit RoiExtractor(const char *name) : OpDef(name)
    {
        this->Input("features_0")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("features_1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("features_2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("features_3")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("features_4")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("features_5")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("features_6")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("features_7")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rois")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("index")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // attr order is frozen by SPEC §5 (canndev op_proto order)
        this->Attr("finest_scale").AttrType(OPTIONAL).Int(56);
        this->Attr("roi_scale_factor").AttrType(OPTIONAL).Float(0.0);
        this->Attr("spatial_scale").AttrType(OPTIONAL).ListFloat({0.25, 0.125, 0.0625, 0.03125});
        this->Attr("pooled_height").AttrType(OPTIONAL).Int(7);
        this->Attr("pooled_width").AttrType(OPTIONAL).Int(7);
        this->Attr("sample_num").AttrType(OPTIONAL).Int(0);
        this->Attr("pool_mode").AttrType(OPTIONAL).String("avg");
        this->Attr("aligned").AttrType(OPTIONAL).Bool(true);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "roi_extractor")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(RoiExtractor);
} // namespace ops
