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
class QuantizeAddLayerNorm : public OpDef {
public:
    explicit QuantizeAddLayerNorm(const char* name) : OpDef(name)
    {
        this->Input("x1").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("x2").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("scales").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("zero_points").ParamType(OPTIONAL).DataType({ge::DT_FLOAT, ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Output("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Attr("dtype").AttrType(REQUIRED).Int(ge::DT_INT8);
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5f);
        this->Attr("additional_output").AttrType(OPTIONAL).Bool(false);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(QuantizeAddLayerNorm);
}
