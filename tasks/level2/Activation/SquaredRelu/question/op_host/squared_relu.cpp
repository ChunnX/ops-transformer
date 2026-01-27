
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ops {

class SquaredRelu : public OpDef {
public:
explicit SquaredRelu(const char* name) : OpDef(name) {
    this->Input("input")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

    this->Output("output")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");

    OpAICoreConfig config_kirin = GetKirinCoreConfig();
    this->AICore().AddConfig("kirinx90", config_kirin);
}

private:
OpAICoreConfig GetKirinCoreConfig() const
{
    OpAICoreConfig config_kirin;
    config_kirin.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true);
    config_kirin.Input("input")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    config_kirin.Output("output")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    return config_kirin;
}
};

OP_ADD(SquaredRelu);
}  // namespace ops