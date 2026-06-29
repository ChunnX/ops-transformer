/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "register/op_def_registry.h"

namespace ops {
class CausalConv1dDecode : public OpDef {
public:
    explicit CausalConv1dDecode(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("conv_states")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("query_start_loc")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("cache_indices")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("num_accepted_tokens")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Attr("activation_mode").AttrType(REQUIRED).Int(); // 0 none, 1 silu
        this->Attr("pad_slot_id").AttrType(REQUIRED).Int();     // skip invalid slot
        this->Attr("conv_mode").AttrType(REQUIRED).Int();       // v1 supports 0

        OpAICoreConfig config950;
        config950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "causal_conv1d_decode_apt");
        this->AICore().AddConfig("ascend950", config950);
    }
};

OP_ADD(CausalConv1dDecode);
} // namespace ops
