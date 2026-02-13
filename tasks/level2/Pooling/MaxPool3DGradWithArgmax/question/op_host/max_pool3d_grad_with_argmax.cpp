
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
class MaxPool3DGradWithArgmax : public OpDef {
public:
    explicit MaxPool3DGradWithArgmax(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .AutoContiguous();
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .AutoContiguous();
        this->Input("argmax")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW});
        this->Attr("ksize").AttrType(REQUIRED).ListInt();
        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(REQUIRED).ListInt();
        this->Attr("dilation").AttrType(OPTIONAL).ListInt({1, 1, 1});
        this->Attr("ceil_mode").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "max_pool3d_grad_with_argmax")
            .ExtendCfgInfo("opInterface.value", "max_pool3d_grad_with_argmax")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(MaxPool3DGradWithArgmax);
} // namespace ops