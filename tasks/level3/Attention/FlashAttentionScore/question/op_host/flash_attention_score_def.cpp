/**
 * 题目占位：最小 OpDef；完整定义见 answer/0。
 */
#include "register/op_def_registry.h"

namespace ops {

class FlashAttentionScore : public OpDef {
public:
    explicit FlashAttentionScore(const char *name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat(
            {ge::FORMAT_ND});
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat(
            {ge::FORMAT_ND});
        this->Input("value").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat(
            {ge::FORMAT_ND});
        this->Output("softmax_max").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("softmax_sum").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("softmax_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("attention_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("head_num").AttrType(REQUIRED).Int();
        this->Attr("input_layout").AttrType(REQUIRED).String();
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(FlashAttentionScore);

} // namespace ops
