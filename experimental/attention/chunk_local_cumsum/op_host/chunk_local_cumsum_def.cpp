
#include "register/op_def_registry.h"

namespace ops {
class ChunkLocalCumsum : public OpDef {
   public:
    explicit ChunkLocalCumsum(const char *name) : OpDef(name) {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cu_seqlens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("chunkSize").Int();
        this->Attr("cuSeqlenDict").ListInt();
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ChunkLocalCumsum);
}  // namespace ops
