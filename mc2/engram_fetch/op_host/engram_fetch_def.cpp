#include "register/op_def_registry.h"

namespace ops {
class EngramFetch : public OpDef {
public:
    explicit EngramFetch(const char *name) : OpDef(name)
    {
        this->Input("bufferhandle")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("fetched")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        this->Attr("num_entries_per_rank").AttrType(REQUIRED).Int();
        this->Attr("hidden_size").AttrType(REQUIRED).Int();
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("output_dtype").AttrType(OPTIONAL).Int(1);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("jitCompile.flag", "static_false")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");

        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_95", aicore_config);
        this->MC2().HcclGroup({"group"});
    }
};
OP_ADD(EngramFetch);
} // namespace ops