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
* \file sparse_flash_mla_def.cpp
* \brief
*/

#include "register/op_def_registry.h"

namespace ops {
class SparseFlashMla : public OpDef {
public:
    explicit SparseFlashMla(const char *name) : OpDef(name)
    {
        const auto addSparseMlaControlInput = [this](const char *inputName) {
            this->Input(inputName)
                .ParamType(OPTIONAL)
                .DataTypeList({ge::DT_INT32})
                .FormatList({ge::FORMAT_ND})
                .AutoContiguous();
        };
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ori_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("cmp_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        const char *const sparseMlaControlInputs[] = {
            "ori_sparse_indices", "cmp_sparse_indices", "ori_block_table", "cmp_block_table", "cu_seqlens_q",
            "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "seqused_q", "seqused_ori_kv", "seqused_cmp_kv",
            "cmp_residual_kv", "ori_topk_length", "cmp_topk_length"};
        for (const char *inputName : sparseMlaControlInputs) {
            addSparseMlaControlInput(inputName);
        }
        this->Input("sinks")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        addSparseMlaControlInput("metadata");
        this->Output("attn_out") // SMLA: output config.
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("softmax_lse")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        const auto addSparseMlaIntAttribute = [this](const char *attributeName, int64_t defaultValue) {
            this->Attr(attributeName).AttrType(OPTIONAL).Int(defaultValue);
        };
        const auto addSparseMlaLayoutAttribute = [this](const char *attributeName) {
            this->Attr(attributeName).AttrType(OPTIONAL).String("BSND");
        };
        this->Attr("softmax_scale").AttrType(OPTIONAL).Float(1.0);
        addSparseMlaIntAttribute("cmp_ratio", 1);
        addSparseMlaIntAttribute("ori_mask_mode", 0);
        addSparseMlaIntAttribute("cmp_mask_mode", 0);
        addSparseMlaIntAttribute("ori_win_left", -1);
        addSparseMlaIntAttribute("ori_win_right", -1);
        addSparseMlaLayoutAttribute("layout_q");
        addSparseMlaLayoutAttribute("layout_kv");
        addSparseMlaIntAttribute("topk_value_mode", 1);
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicore_config;
        const auto configureSparseMlaAicore = [](OpAICoreConfig &config) {
            config.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(true)
                .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        };
        configureSparseMlaAicore(aicore_config);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(SparseFlashMla);
} // namespace ops
