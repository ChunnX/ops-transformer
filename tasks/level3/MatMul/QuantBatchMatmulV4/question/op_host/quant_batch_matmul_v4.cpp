#include "register/op_def_registry.h"

namespace ops {
class QuantBatchMatmulV4 : public OpDef {
public:
    explicit QuantBatchMatmulV4(const char *name) : OpDef(name)
    {
        this->Input("x1").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
        this->Input("x2").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("x1_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("x2_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("y_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("x1_offset").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("x2_offset").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("y_offset").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("x2_table").ParamType(OPTIONAL).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Attr("dtype").AttrType(REQUIRED).Int(-1);
        this->Attr("compute_type").AttrType(OPTIONAL).Int(-1);
        this->Attr("transpose_x1").AttrType(OPTIONAL).Bool(false);
        this->Attr("transpose_x2").AttrType(OPTIONAL).Bool(false);
        this->Attr("group_size").AttrType(OPTIONAL).Int(-1);

        OpAICoreConfig config_910BC;
        config_910BC.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config_910BC);
        this->AICore().AddConfig("ascend910_93", config_910BC);
    }
};

OP_ADD(QuantBatchMatmulV4);
}  // namespace ops
