#include "register/op_def_registry.h"

namespace ops {
class WeightQuantBatchMatmulV2 : public OpDef
{
public:
    WeightQuantBatchMatmulV2(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_INT8, ge::DT_INT4}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("antiquant_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("antiquant_offset").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("quant_scale").ParamType(OPTIONAL).DataType({ge::DT_FLOAT, ge::DT_UINT64}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("quant_offset").ParamType(OPTIONAL).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("transpose_x").AttrType(OPTIONAL).Bool(false);
        this->Attr("transpose_weight").AttrType(OPTIONAL).Bool(false);
        this->Attr("antiquant_group_size").AttrType(OPTIONAL).Int(0);
        this->Attr("dtype").AttrType(OPTIONAL).Int(-1);
        this->Attr("inner_precise").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(WeightQuantBatchMatmulV2);
} // namespace ops
