/**
 * Copyright (c) 2026 fk2323
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the repository.
 */

#include "register/op_def_registry.h"

namespace ops {
class Fmm016StreamingTopKBlockedV4 : public OpDef {
public:
    explicit Fmm016StreamingTopKBlockedV4(const char *name) : OpDef(name)
    {
        Input("q_index").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Input("k_index_transposed").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Input("index_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("selected_index").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("selected_valid").ParamType(REQUIRED).DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Attr("top_k").AttrType(OPTIONAL).Int(1);
        AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Fmm016StreamingTopKBlockedV4);
}
