/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_paged_kv_def.cpp
 * \brief StemOamPrepPagedKV OpDef registration
 */
#include "register/op_def_registry.h"

namespace ops {

class StemOamPrepPagedKV : public OpDef {
public:
    explicit StemOamPrepPagedKV(const char *name) : OpDef(name)
    {
        this->Input("KCache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("VCache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("KvIndices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("KvSeqLens").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("KScaleCache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("VScale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("KFlat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BFLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("VBias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("LambdaMag").AttrType(OPTIONAL).Float(0.3);
        this->Attr("CacheLayout").AttrType(OPTIONAL).Int(0);
        this->Attr("KvBlockSize").AttrType(OPTIONAL).Int(64);
        this->Attr("StemBlockSize").AttrType(OPTIONAL).Int(128);
        this->Attr("StemStride").AttrType(OPTIONAL).Int(16);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "stem_oam_prep_paged_kv_apt");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(StemOamPrepPagedKV);

} // namespace ops