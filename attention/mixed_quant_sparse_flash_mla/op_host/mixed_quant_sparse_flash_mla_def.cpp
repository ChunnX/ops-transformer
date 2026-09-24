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
 * \file mixed_quant_sparse_flash_mla_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class MixedQuantSparseFlashMla : public OpDef {
public:
    explicit MixedQuantSparseFlashMla(const char *name) : OpDef(name)
    {
        const auto addMixedQuantMlaControlInput = [this](const char *inputName) {
            this->Input(inputName)
                .ParamType(OPTIONAL)
                .DataTypeList({ge::DT_INT32})
                .FormatList({ge::FORMAT_ND})
                .AutoContiguous();
        };
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ori_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("cmp_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        const char *const mixedQuantMlaControlInputs[] = {
            "ori_sparse_indices", "cmp_sparse_indices", "ori_block_table", "cmp_block_table", "cu_seqlens_q",
            "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "seqused_q", "seqused_ori_kv", "seqused_cmp_kv",
            "cmp_residual_kv", "ori_topk_length", "cmp_topk_length"};
        for (const char *inputName : mixedQuantMlaControlInputs) {
            addMixedQuantMlaControlInput(inputName);
        }
        this->Input("sinks")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        addMixedQuantMlaControlInput("metadata");
        this->Output("attn_out") // MQSMLA: output config.
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("softmax_lse")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        const auto addMixedQuantMlaIntAttribute = [this](const char *attributeName, int64_t defaultValue) {
            this->Attr(attributeName).AttrType(OPTIONAL).Int(defaultValue);
        };
        const auto addMixedQuantMlaLayoutAttribute = [this](const char *attributeName) {
            this->Attr(attributeName).AttrType(OPTIONAL).String("BSND");
        };
        this->Attr("quant_mode").AttrType(OPTIONAL).Int();
        this->Attr("rope_head_dim").AttrType(OPTIONAL).Int();
        this->Attr("softmax_scale").AttrType(OPTIONAL).Float();
        this->Attr("cmp_ratio").AttrType(OPTIONAL).Int();
        addMixedQuantMlaIntAttribute("ori_mask_mode", 0);
        addMixedQuantMlaIntAttribute("cmp_mask_mode", 0);
        addMixedQuantMlaIntAttribute("ori_win_left", -1);
        addMixedQuantMlaIntAttribute("ori_win_right", -1);
        addMixedQuantMlaLayoutAttribute("layout_q");
        addMixedQuantMlaLayoutAttribute("layout_kv");
        addMixedQuantMlaIntAttribute("topk_value_mode", 1);
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicore_config;
        const auto configureMixedQuantMlaAicore = [](OpAICoreConfig &config) {
            config.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(true)
                .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        };
        configureMixedQuantMlaAicore(aicore_config);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);

        OpAICoreConfig aicore_config_950;
        aicore_config_950.Input("q")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("ori_kv")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicore_config_950.Input("cmp_kv")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicore_config_950.Input("ori_sparse_indices")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("cmp_sparse_indices")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("ori_block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("cmp_block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("cu_seqlens_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("cu_seqlens_ori_kv")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("cu_seqlens_cmp_kv")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("seqused_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("seqused_ori_kv")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("seqused_cmp_kv")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("cmp_residual_kv")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("ori_topk_length")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("cmp_topk_length")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("sinks")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("metadata")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Output("attn_out")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        aicore_config_950.Output("softmax_lse")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        aicore_config_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicore_config_950);
    }
};
OP_ADD(MixedQuantSparseFlashMla);
} // namespace ops
