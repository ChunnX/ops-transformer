#include "register/op_def_registry.h"

namespace optiling {

// 占位 tiling 函数
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    context->SetBlockDim(1);

    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {

// OpDef 骨架: tiling 由 IMPL_OP_OPTILING 注册, 不使用 .SetTiling()
class FlatQuant : public OpDef {
public:
    explicit FlatQuant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("kronecker_p1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("kronecker_p2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT4, ge::DT_INT4})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("quant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("clip_ratio").AttrType(OPTIONAL).Float(1.0);
        this->Attr("dst_dtype").AttrType(OPTIONAL).Int(ge::DT_INT4);

        // 注意: 使用 IMPL_OP_OPTILING 模式, 不写 .SetTiling()
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(FlatQuant);

}  // namespace ops
