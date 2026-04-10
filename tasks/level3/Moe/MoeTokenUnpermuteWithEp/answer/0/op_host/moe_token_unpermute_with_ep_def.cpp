#include "register/op_def_registry.h"

namespace ops {
class MoeTokenUnpermuteWithEp : public OpDef {
 public:
  explicit MoeTokenUnpermuteWithEp(const char* name) : OpDef(name) {
    this->Input("permuted_tokens")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                   ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("sorted_indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                   ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("probs")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                   ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("unpermuted_tokens")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                   ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("num_topk").AttrType(OPTIONAL).Int(1);
    this->Attr("range").AttrType(OPTIONAL).ListInt({0, 0});
    this->Attr("padded_mode").AttrType(OPTIONAL).Bool(false);
    this->Attr("restore_shape").AttrType(OPTIONAL).ListInt();

    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(MoeTokenUnpermuteWithEp);
}  // namespace ops
