
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
class MoeReRouting : public OpDef {
public:
    explicit MoeReRouting(const char *name) : OpDef(name)
    {
        this->Input("tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_token_num_per_rank")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("per_token_scales")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("permute_tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("permute_per_token_scales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("permute_token_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("expert_token_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("expert_token_num_type").AttrType(OPTIONAL).Int(1);
        this->Attr("idx_type").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig membaseCfg;
        membaseCfg.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "moe_re_routing");
        this->AICore().AddConfig("ascend910b", membaseCfg);
        this->AICore().AddConfig("ascend910_93", membaseCfg);

        OpAICoreConfig regbaseCfg;
        regbaseCfg.Input("tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_INT8,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_INT8,
                ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E4M3FN,
                ge::DT_FLOAT8_E4M3FN,
                ge::DT_FLOAT8_E4M3FN,
                ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .AutoContiguous();
        regbaseCfg.Input("expert_token_num_per_rank")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT64,
                ge::DT_INT64,
                ge::DT_INT64,
                ge::DT_INT32,
                ge::DT_INT64,
                ge::DT_INT32,
                ge::DT_INT64,
                ge::DT_INT32,
                ge::DT_INT64,
                ge::DT_INT32,
                ge::DT_INT64})
            .Format({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .AutoContiguous();
        regbaseCfg.Input("per_token_scales")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT8_E8M0,
                ge::DT_FLOAT8_E8M0,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT8_E8M0,
                ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .AutoContiguous();
        regbaseCfg.Output("permute_tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_INT8,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_INT8,
                ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E4M3FN,
                ge::DT_FLOAT8_E4M3FN,
                ge::DT_FLOAT8_E4M3FN,
                ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND});
        regbaseCfg.Output("permute_per_token_scales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT8_E8M0,
                ge::DT_FLOAT8_E8M0,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT8_E8M0,
                ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND});
        regbaseCfg.Output("permute_token_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32})
            .Format({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND});
        regbaseCfg.Output("expert_token_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT64,
                ge::DT_INT64,
                ge::DT_INT64,
                ge::DT_INT32,
                ge::DT_INT64,
                ge::DT_INT32,
                ge::DT_INT64,
                ge::DT_INT32,
                ge::DT_INT64,
                ge::DT_INT32,
                ge::DT_INT64})
            .Format({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND});

        regbaseCfg.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "moe_re_routing_apt");

        this->AICore().AddConfig("ascend950", regbaseCfg);

        OpAICoreConfig config_kirin = GetKirinCoreConfig();
        this->AICore().AddConfig("kirinx90", config_kirin);
        this->AICore().AddConfig("kirin9030", config_kirin);
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
        config_kirin.Input("tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("expert_token_num_per_rank")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("per_token_scales")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Output("permute_tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_kirin.Output("permute_per_token_scales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_kirin.Output("permute_token_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_kirin.Output("expert_token_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        return config_kirin;
    }
};

OP_ADD(MoeReRouting);
}  // namespace ops