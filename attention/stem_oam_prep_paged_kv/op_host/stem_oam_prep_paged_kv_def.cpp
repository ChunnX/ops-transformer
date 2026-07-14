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

class StemOamPrepPagedKv : public OpDef {
public:
    explicit StemOamPrepPagedKv(const char *name) : OpDef(name)
    {
        this->Input("kCache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("vCache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("kvIndices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("kvSeqLens").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("kScaleCache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("vScale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("kFlat")
            .OutputShapeDependOnCompute()
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .OutputShapeDependOnCompute()
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("vBias")
            .OutputShapeDependOnCompute()
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .OutputShapeDependOnCompute()
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("lambdaMag").AttrType(OPTIONAL).Float(0.3);
        this->Attr("cacheLayout").AttrType(OPTIONAL).Int(0);
        this->Attr("kvBlockSize").AttrType(OPTIONAL).Int(64);
        this->Attr("stemBlockSize").AttrType(OPTIONAL).Int(128);
        this->Attr("stemStride").AttrType(OPTIONAL).Int(16);
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

OP_ADD(StemOamPrepPagedKv);

} // namespace ops