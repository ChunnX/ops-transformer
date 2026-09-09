/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
constexpr int32_t DEFAULT_DST_TYPE = 36; // FLOAT8_E4M3FN
constexpr int32_t DEFAULT_SCALE_ALG = 1;

// 8 dtype branches: fp16->e2m1, bf16->e2m1, fp16->e1m2, bf16->e1m2,
//                   fp16->e4m3fn, bf16->e4m3fn, fp16->e5m2, bf16->e5m2
class ClaGateMxQuantWithDualAxis : public OpDef {
public:
    explicit ClaGateMxQuantWithDualAxis(const char *name)
        : OpDef(name)
    {
        const std::initializer_list<ge::DataType> inputTypes = {ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                                                                ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                                                                ge::DT_FLOAT16, ge::DT_BF16};
        const std::initializer_list<ge::Format> formats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                           ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("global_out").ParamType(REQUIRED).DataType(inputTypes).Format(formats).AutoContiguous();
        this->Input("local_out").ParamType(REQUIRED).DataType(inputTypes).Format(formats).AutoContiguous();
        this->Input("global_gate_logits").ParamType(REQUIRED).DataType(inputTypes).Format(formats).AutoContiguous();
        this->Input("local_gate_logits").ParamType(REQUIRED).DataType(inputTypes).Format(formats).AutoContiguous();
        const std::initializer_list<ge::DataType> dataTypes = {
            ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E1M2,
            ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2};
        const std::initializer_list<ge::DataType> scaleTypes = {
            ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
            ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0};
        this->Output("row_data").ParamType(REQUIRED).DataType(dataTypes).Format(formats);
        this->Output("row_scale").ParamType(REQUIRED).DataType(scaleTypes).Format(formats);
        this->Output("col_data").ParamType(REQUIRED).DataType(dataTypes).Format(formats);
        this->Output("col_scale").ParamType(REQUIRED).DataType(scaleTypes).Format(formats);
        this->Attr("dst_type").AttrType(OPTIONAL).Int(DEFAULT_DST_TYPE);
        this->Attr("scale_alg").AttrType(OPTIONAL).Int(DEFAULT_SCALE_ALG);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(0);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true).ExtendCfgInfo(
            "opFile.value", "cla_gate_mx_quant_with_dual_axis_apt");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(ClaGateMxQuantWithDualAxis);
} // namespace ops
