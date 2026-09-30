#include "register/op_def_registry.h"

namespace ops {

class SageAttention2 : public OpDef {
public:
    explicit SageAttention2(const char *name) : OpDef(name)
    {
        DeclareTensorInputs();
        DeclareTensorOutputs();
        DeclareOperatorAttributes();
        DeclareHardwareTarget();
    }

private:
    void DeclareTensorInputs()
    {
        const std::vector<ge::DataType> supportedTypes = {ge::DT_FLOAT16, ge::DT_BF16};
        const std::vector<ge::Format> supportedFormats = {ge::FORMAT_ND, ge::FORMAT_ND};

        Input("q").ParamType(REQUIRED).DataType(supportedTypes).Format(supportedFormats);
        Input("k").ParamType(REQUIRED).DataType(supportedTypes).Format(supportedFormats);
        Input("v").ParamType(REQUIRED).DataType(supportedTypes).Format(supportedFormats);
    }

    void DeclareTensorOutputs()
    {
        const std::vector<ge::DataType> outTypes = {ge::DT_FLOAT16, ge::DT_BF16};
        const std::vector<ge::DataType> lseTypes = {ge::DT_FLOAT, ge::DT_FLOAT};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND, ge::FORMAT_ND};

        Output("out").ParamType(REQUIRED).DataType(outTypes).Format(formats);
        Output("lse").ParamType(REQUIRED).DataType(lseTypes).Format(formats);
    }

    void DeclareOperatorAttributes()
    {
        Attr("sm_scale").AttrType(REQUIRED).Float();
        Attr("smooth_k").AttrType(REQUIRED).Bool();
        Attr("is_causal").AttrType(REQUIRED).Bool();
        Attr("return_lse").AttrType(REQUIRED).Bool();
        Attr("qk_quant_gran").AttrType(REQUIRED).Int();
        Attr("pv_accum_dtype").AttrType(REQUIRED).Int();
        Attr("smooth_v").AttrType(REQUIRED).Bool();
    }

    void DeclareHardwareTarget()
    {
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicShapeSupportFlag(true)
            .DynamicRankSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "sageattention2")
            .ExtendCfgInfo("opInterface.value", "sage_attention2");
        AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(SageAttention2);

} // namespace ops
