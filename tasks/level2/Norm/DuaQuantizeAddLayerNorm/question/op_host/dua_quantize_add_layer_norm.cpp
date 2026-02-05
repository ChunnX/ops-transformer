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
class DuaQuantizeAddLayerNorm : public OpDef {
public:
    explicit DuaQuantizeAddLayerNorm(const char* name) : OpDef(name)
    {
        this->Input("x1").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("x2").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("scales1").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("scales2").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Output("y1").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Output("y2").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Output("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5f);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(DuaQuantizeAddLayerNorm);
} // namespace ops
