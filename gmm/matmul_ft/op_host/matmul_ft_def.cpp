/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

/*!
 * \file matmul_ft_def.cpp
 * \brief MatmulFt operator definition.
 */
#include "register/op_def_registry.h"

namespace ops {
namespace {
const std::vector<ge::DataType> kPrecisionDtype = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
const std::vector<ge::DataType> kCeInputDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT};
const std::vector<ge::DataType> kFloatDtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
const std::vector<ge::DataType> kUint8Dtype = {ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8};
const std::vector<ge::Format> kNdFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
} // namespace

class MatmulFt : public OpDef {
public:
    explicit MatmulFt(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType(kPrecisionDtype)
            .Format(kNdFormat)
            .UnknownShapeFormat(kNdFormat)
            .AutoContiguous();
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType(kPrecisionDtype)
            .Format(kNdFormat)
            .UnknownShapeFormat(kNdFormat)
            .AutoContiguous();
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(kCeInputDtype)
            .Format(kNdFormat)
            .UnknownShapeFormat(kNdFormat)
            .AutoContiguous();
        this->Input("xv")
            .ParamType(REQUIRED)
            .DataType(kPrecisionDtype)
            .Format(kNdFormat)
            .UnknownShapeFormat(kNdFormat)
            .AutoContiguous();
        this->Input("vx_for_ae")
            .ParamType(REQUIRED)
            .DataType(kPrecisionDtype)
            .Format(kNdFormat)
            .UnknownShapeFormat(kNdFormat)
            .AutoContiguous();

        AddOutput("z_row", kFloatDtype);
        AddOutput("z_col", kFloatDtype);
        AddOutput("d_row", kFloatDtype);
        AddOutput("d_col", kFloatDtype);
        AddOutput("comp_row", kUint8Dtype);
        AddOutput("comp_col", kUint8Dtype);
        AddOutput("c", kFloatDtype);
        AddOutput("threshold", kFloatDtype);
        AddOutput("b_mean_abs", kFloatDtype);
        AddOutput("b_mean_square", kFloatDtype);
        AddOutput("b_var", kFloatDtype);
        AddOutput("be", kPrecisionDtype);
        AddOutput("be_for_aiv", kCeInputDtype);
        AddOutput("b_max_slice", kFloatDtype);
        AddOutput("b_min_slice", kFloatDtype);
        AddOutput("a_max", kFloatDtype);
        AddOutput("a_mean", kFloatDtype);
        AddOutput("a_min", kFloatDtype);

        this->Attr("e_max").AttrType(OPTIONAL).Float(0.001);
        this->Attr("reduce_cores").AttrType(OPTIONAL).Int(8);
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "matmul_ft");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }

private:
    void AddOutput(const char* name, const std::vector<ge::DataType>& dtypes)
    {
        this->Output(name)
            .ParamType(REQUIRED)
            .DataType(dtypes)
            .Format(kNdFormat)
            .UnknownShapeFormat(kNdFormat)
            .AutoContiguous();
    }
};

OP_ADD(MatmulFt);
} // namespace ops
