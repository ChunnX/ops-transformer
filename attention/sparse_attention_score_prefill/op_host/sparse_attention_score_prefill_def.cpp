/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "register/op_def_registry.h"

namespace ops {

class SparseAttentionScorePrefill : public OpDef {
public:
    explicit SparseAttentionScorePrefill(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("selectIdx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("blockTable")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("kvToQSelectIdx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("kvToQSelectNum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("kvToQSlotIdx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("kvBlockValidSize")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("actualSeqLengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("actualSeqLengthsKv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Output("attentionOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});

        this->Attr("numKeyValueHeads").AttrType(OPTIONAL).Int(1);
        this->Attr("scaleValue").AttrType(OPTIONAL).Float(0.0);
        this->Attr("blockSize").AttrType(OPTIONAL).Int(128);
        this->Attr("topK").AttrType(OPTIONAL).Int(8);
        this->Attr("innerPrecise").AttrType(OPTIONAL).Int(4);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(SparseAttentionScorePrefill);

}  // namespace ops
