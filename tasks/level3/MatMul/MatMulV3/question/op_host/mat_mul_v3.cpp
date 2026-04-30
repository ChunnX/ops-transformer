#include "register/op_def_registry.h"

namespace ops {
class MatMulV3 : public OpDef {
public:
    explicit MatMulV3(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("offset_w")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("transpose_x1")
            .AttrType(OPTIONAL)
            .Bool(false);
        this->Attr("transpose_x2")
            .AttrType(OPTIONAL)
            .Bool(false);
        this->Attr("offset_x")
            .AttrType(OPTIONAL)
            .Int(0);
        this->Attr("opImplMode")
            .AttrType(OPTIONAL)
            .Int(1);
        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(false)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false);
        this->AICore().AddConfig("ascend910b", aicConfig);
        this->AICore().AddConfig("ascend910_93", aicConfig);
    }
};

OP_ADD(MatMulV3);
}

namespace optiling {
static ge::graphStatus MatmulV3TilingFunc(gert::TilingContext *context) {
    context->SetBlockDim(1);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(MatMulV3)
    .Tiling(MatmulV3TilingFunc);
}
