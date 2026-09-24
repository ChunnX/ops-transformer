/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file quant_sparse_flash_mla_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class QuantSparseFlashMla : public OpDef {
public:
    explicit QuantSparseFlashMla(const char *name) : OpDef(name)
    {
        const auto addQuantSparseMlaControlInput = [this](const char *inputName) {
            this->Input(inputName)
                .ParamType(OPTIONAL)
                .DataTypeList({ge::DT_INT32})
                .FormatList({ge::FORMAT_ND})
                .AutoContiguous();
        };
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_HIFLOAT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ori_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_HIFLOAT8})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("cmp_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_HIFLOAT8})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("q_descale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ori_kv_descale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cmp_kv_descale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        const char *const quantSparseMlaControlInputs[] = {
            "ori_sparse_indices", "cmp_sparse_indices", "ori_block_table", "cmp_block_table", "cu_seqlens_q",
            "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "seqused_q", "seqused_ori_kv", "seqused_cmp_kv",
            "cmp_residual_kv", "ori_topk_length", "cmp_topk_length"};
        for (const char *inputName : quantSparseMlaControlInputs) {
            addQuantSparseMlaControlInput(inputName);
        }
        this->Input("sinks")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        addQuantSparseMlaControlInput("metadata");
        this->Output("attn_out")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("softmax_lse")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        const auto addQuantSparseMlaIntAttribute = [this](const char *attributeName, int64_t defaultValue) {
            this->Attr(attributeName).AttrType(OPTIONAL).Int(defaultValue);
        };
        const auto addQuantSparseMlaLayoutAttribute = [this](const char *attributeName) {
            this->Attr(attributeName).AttrType(OPTIONAL).String("BSND");
        };
        this->Attr("quant_mode").AttrType(REQUIRED).Int();
        this->Attr("softmax_scale").AttrType(OPTIONAL).Float();
        addQuantSparseMlaIntAttribute("cmp_ratio", 1);
        addQuantSparseMlaIntAttribute("ori_mask_mode", 0);
        addQuantSparseMlaIntAttribute("cmp_mask_mode", 0);
        addQuantSparseMlaIntAttribute("ori_win_left", -1);
        addQuantSparseMlaIntAttribute("ori_win_right", -1);
        addQuantSparseMlaLayoutAttribute("layout_q");
        addQuantSparseMlaLayoutAttribute("layout_kv");
        addQuantSparseMlaIntAttribute("topk_value_mode", 1);
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicore_config;
        const auto configureQuantSparseMlaAicore = [](OpAICoreConfig &config) {
            config.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(true)
                .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        };
        configureQuantSparseMlaAicore(aicore_config);
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(QuantSparseFlashMla);
} // namespace ops
