#include "register/op_def_registry.h"

namespace ops {

namespace {

struct OperatorDescriptorBuilder {
    static void RegisterIo(OpDef &opDef)
    {
        const std::vector<ge::DataType> fpTypes = {ge::DT_FLOAT16, ge::DT_BF16};
        const std::vector<ge::Format> ndFormats = {ge::FORMAT_ND, ge::FORMAT_ND};

        for (const auto &tag : {"q", "k", "v"}) {
            opDef.Input(tag).ParamType(REQUIRED).DataType(fpTypes).Format(ndFormats);
        }
        opDef.Output("out").ParamType(REQUIRED).DataType(fpTypes).Format(ndFormats);
        opDef.Output("lse").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format(ndFormats);
    }

    static void RegisterAttributes(OpDef &opDef)
    {
        auto bindBool = [&opDef](const char *attrName) {
            opDef.Attr(attrName).AttrType(REQUIRED).Bool();
        };
        auto bindInt = [&opDef](const char *attrName) {
            opDef.Attr(attrName).AttrType(REQUIRED).Int();
        };

        opDef.Attr("sm_scale").AttrType(REQUIRED).Float();
        bindBool("smooth_k");
        bindBool("is_causal");
        bindBool("return_lse");
        bindInt("qk_quant_gran");
        bindInt("pv_accum_dtype");
        bindBool("smooth_v");
    }

    static void RegisterDeviceTarget(OpDef &opDef)
    {
        OpAICoreConfig socCfg;
        socCfg.DynamicCompileStaticFlag(true).DynamicShapeSupportFlag(true).DynamicRankSupportFlag(true).NeedCheckSupportFlag(false);
        socCfg.ExtendCfgInfo("opFile.value", "sageattention2").ExtendCfgInfo("opInterface.value", "sage_attention2");
        opDef.AICore().AddConfig("ascend950", socCfg);
    }
};

} // namespace

class SageAttention2 : public OpDef {
public:
    explicit SageAttention2(const char *name) : OpDef(name)
    {
        OperatorDescriptorBuilder::RegisterIo(*this);
        OperatorDescriptorBuilder::RegisterAttributes(*this);
        OperatorDescriptorBuilder::RegisterDeviceTarget(*this);
    }
};

OP_ADD(SageAttention2);

} // namespace ops
