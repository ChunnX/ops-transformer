
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

static const std::vector<ge::Format> formats = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};
static const std::vector<ge::DataType> tokenTypes = {
    ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT
};

class MoeTokenUnpermuteGrad : public OpDef
{
public:
    explicit MoeTokenUnpermuteGrad(const char* name) : OpDef(name)
    {
        this->Input("permuted_tokens")
            .ParamType(REQUIRED)
            .DataType(tokenTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("unpermuted_tokens_grad")
            .ParamType(REQUIRED)
            .DataType(tokenTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("sorted_indices")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                 ge::DT_INT32, ge::DT_INT32})
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("probs")
            .ParamType(OPTIONAL)
            .DataType(
                {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                 ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Output("permuted_tokens_grad")
            .ParamType(REQUIRED)
            .DataType(tokenTypes)
            .Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("probs_grad")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                 ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format(formats)
            .UnknownShapeFormat(formats);
        this->Attr("padded_mode").AttrType(OPTIONAL).Bool(false);
        this->Attr("restore_shape").AttrType(OPTIONAL).ListInt({1, 1});

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(MoeTokenUnpermuteGrad);
} // namespace ops