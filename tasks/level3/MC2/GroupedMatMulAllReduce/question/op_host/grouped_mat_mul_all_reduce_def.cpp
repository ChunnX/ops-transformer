/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * LLM question template — OpDef 与 answer 一致；tiling 由 IMPL_OP_OPTILING 注册，此处不写 SetTiling。
 */
#include "register/op_def_registry.h"

namespace ops {
class GroupedMatMulAllReduce : public OpDef
{
public:
    explicit GroupedMatMulAllReduce(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("group_list")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("splitItem").AttrType(OPTIONAL).Int(0);
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("reduceOp").AttrType(OPTIONAL).String("sum");
        this->Attr("commTurn").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config_910;
        aicore_config_910.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend910b", aicore_config_910);
        this->AICore().AddConfig("ascend910_93", aicore_config_910);
        this->MC2().HcclGroup("group");
    }
};

OP_ADD(GroupedMatMulAllReduce);
} // namespace ops
