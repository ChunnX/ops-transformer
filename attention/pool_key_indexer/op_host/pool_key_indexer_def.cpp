/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pool_key_indexer_def.cpp
 * \brief PoolKeyIndexer operator definition
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class PoolKeyIndexer : public OpDef {
public:
    explicit PoolKeyIndexer(const char *name) : OpDef(name)
    {
        // ---- Base config: A2/A3 (ascend910b, ascend910_93) ----
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pool_key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pool_tail_k")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_k")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("q_descale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k_descale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Output("sparse_indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("sparse_values")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Attr("topk").AttrType(OPTIONAL).Int(2048);
        this->Attr("pool_size").AttrType(OPTIONAL).Int(16);
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_k").AttrType(OPTIONAL).String("BSND");
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(3);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(-1);
        this->Attr("return_value").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);

        // ---- 950 config: adds FP8 combinations ----
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("pool_key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        aicoreConfig950.Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("pool_tail_k")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("actual_seq_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("actual_seq_k")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("q_descale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("k_descale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicoreConfig950.Output("sparse_indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        aicoreConfig950.Output("sparse_values")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};
OP_ADD(PoolKeyIndexer);
} // namespace ops
