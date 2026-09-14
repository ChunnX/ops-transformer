/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * OpDef for the Ascend 950-only MXFP8 weight-NZ V3 path.
 */

#include "register/op_def_registry.h"

namespace ops {
class GroupedMatmulSwigluQuantV3 : public OpDef {
public:
    explicit GroupedMatmulSwigluQuantV3(const char *name)
        : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND});
        this->Input("x_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND});
        this->Input("group_list")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_FRACTAL_NZ});
        this->Input("weight_scale")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND});
        this->Input("weight_assist_matrix")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("smooth_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND});
        this->Output("y_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND});

        this->Attr("dequant_mode").AttrType(OPTIONAL).Int(2);
        this->Attr("dequant_dtype").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(2);
        this->Attr("quant_dtype").AttrType(OPTIONAL).Int(0);
        this->Attr("transpose_weight").AttrType(OPTIONAL).Bool(0);
        this->Attr("group_list_type").AttrType(OPTIONAL).Int(0);
        this->Attr("tuning_config").AttrType(OPTIONAL).ListInt({0});
        this->Attr("activate_dim").AttrType(OPTIONAL).Int(-1);
        this->Attr("activate_left").AttrType(OPTIONAL).Bool(1);
        this->Attr("swiglu_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("clamp_limit").AttrType(OPTIONAL).Float(7.0F);
        this->Attr("glu_alpha").AttrType(OPTIONAL).Float(1.702F);
        this->Attr("glu_bias").AttrType(OPTIONAL).Float(1.0F);
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("scale_alg").AttrType(OPTIONAL).Int(0);
        this->Attr("dst_type_max").AttrType(OPTIONAL).Float(0.0F);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "grouped_matmul_swiglu_quant_v3_apt");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(GroupedMatmulSwigluQuantV3);
} // namespace ops
