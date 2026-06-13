/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cast_def.cpp
 * \brief Cast op definition for ascend310p (phase 1: hardware DIRECT + TWO_CAST via f16)
 */

#include "register/op_def_registry.h"

namespace ops {
// 45 dtype pairs: phase1 32 + phase2 i64 family 5 + bf16 input family 8
static const std::vector<ge::DataType> xDataType310p = {
    // f16 -> f32 i32 i8 u8 i16 bool
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    // f32 -> f16 i32 i16 i8 u8 bool
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    // i32 -> f32 f16 i16 i8 u8 bool
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    // i8 -> f16 f32 i32 u8
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    // u8 -> f16 f32 i32 i8
    ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8,
    // i16 -> f16 f32
    ge::DT_INT16, ge::DT_INT16,
    // bool -> f16 f32 i32 u8
    ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL,
    // phase2 i64 family: i64 -> i32 f32 f16; i32/f32 -> i64
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_FLOAT,
    // phase2 bf16 input only: bf16 -> f32 f16 i32 i8 u8 bool i16 i64
    ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16
};
static const std::vector<ge::DataType> yDataType310p = {
    ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_BOOL,
    ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL,
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT8,
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT,
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT8,
    ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT64, ge::DT_INT64,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL, ge::DT_INT16, ge::DT_INT64
};
static const std::vector<ge::Format> castFormat310p(xDataType310p.size(), ge::FORMAT_ND);

class Cast : public OpDef {
public:
    explicit Cast(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(xDataType310p)
            .Format(castFormat310p)
            .UnknownShapeFormat(castFormat310p);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(yDataType310p)
            .Format(castFormat310p)
            .UnknownShapeFormat(castFormat310p);
        this->Attr("dst_type")
            .AttrType(REQUIRED)
            .Int();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("opFile.value", "cast");
        this->AICore().AddConfig("ascend310p", aicoreConfig);
    }
};

OP_ADD(Cast);
}  // namespace ops
