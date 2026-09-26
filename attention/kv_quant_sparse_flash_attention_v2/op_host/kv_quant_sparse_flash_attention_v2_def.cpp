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
 * \file kv_quant_sparse_flash_attention_v2_def.cpp
 * \brief
 */

#include <utility>
#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class KvQuantSparseFlashAttentionV2 : public OpDef {
public:
    explicit KvQuantSparseFlashAttentionV2(const char *name)
        : OpDef(name)
    {
        const auto addRegularInput = [this](const char *inputName, Option paramType,
                                            std::vector<ge::DataType> dataTypes) {
            this->Input(inputName)
                .ParamType(paramType)
                .DataType(std::move(dataTypes))
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
        };
        addRegularInput("query", REQUIRED, {ge::DT_FLOAT16, ge::DT_BF16});
        addRegularInput("key", REQUIRED, {ge::DT_INT8, ge::DT_INT8});
        this->Input("value").ParamType(REQUIRED).Follow("key").AutoContiguous();
        addRegularInput("sparse_indices", REQUIRED, {ge::DT_INT32, ge::DT_INT32});
        addRegularInput("key_dequant_scale", OPTIONAL, {ge::DT_FLOAT, ge::DT_FLOAT});
        addRegularInput("value_dequant_scale", OPTIONAL, {ge::DT_FLOAT, ge::DT_FLOAT});
        addRegularInput("block_table", OPTIONAL, {ge::DT_INT32, ge::DT_INT32});
        addRegularInput("actual_seq_lengths_query", OPTIONAL, {ge::DT_INT32, ge::DT_INT32});
        addRegularInput("actual_seq_lengths_kv", OPTIONAL, {ge::DT_INT32, ge::DT_INT32});
        addRegularInput("sinks", OPTIONAL, {ge::DT_FLOAT, ge::DT_FLOAT});
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("softmax_max")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("softmax_sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        const auto addIntAttribute = [this](const char *attributeName, Option paramType, int64_t defaultValue) {
            this->Attr(attributeName).AttrType(paramType).Int(defaultValue);
        };
        const auto addLayoutAttribute = [this](const char *attributeName) {
            this->Attr(attributeName).AttrType(OPTIONAL).String("BSND");
        };
        this->Attr("scale_value").AttrType(REQUIRED).Float(1.0);
        addIntAttribute("key_quant_mode", REQUIRED, 1);
        addIntAttribute("value_quant_mode", REQUIRED, 1);
        addIntAttribute("sparse_block_size", OPTIONAL, 1);
        addLayoutAttribute("layout_query");
        addLayoutAttribute("layout_kv");
        addIntAttribute("sparse_mode", OPTIONAL, 3);
        addIntAttribute("pre_tokens", OPTIONAL, INT64_MAX);
        addIntAttribute("next_tokens", OPTIONAL, INT64_MAX);
        addIntAttribute("attention_mode", OPTIONAL, 0);
        addIntAttribute("quant_scale_repo_mode", OPTIONAL, 1);
        addIntAttribute("tile_size", OPTIONAL, 128);
        addIntAttribute("rope_head_dim", OPTIONAL, 64);
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicore_config_95;
        const auto addAscend950Input = [&aicore_config_95](const char *inputName, Option paramType,
                                                            std::vector<ge::DataType> dataTypes) {
            aicore_config_95.Input(inputName)
                .ParamType(paramType)
                .DataTypeList(std::move(dataTypes))
                .FormatList({ge::FORMAT_ND})
                .AutoContiguous();
        };
        const auto addAscend950RequiredInput = [&aicore_config_95](const char *inputName,
                                                                     std::vector<ge::DataType> dataTypes,
                                                                     bool ignoreContiguous) {
            OpParamDef &input = aicore_config_95.Input(inputName)
                                    .ParamType(REQUIRED)
                                    .DataType(std::move(dataTypes))
                                    .FormatList({ge::FORMAT_ND});
            if (ignoreContiguous) {
                input.IgnoreContiguous();
            } else {
                input.AutoContiguous();
            }
        };
        addAscend950RequiredInput(
            "query", {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, false);
        const std::vector<ge::DataType> packedKvDataTypes = {
            ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_INT8,
            ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_INT8};
        addAscend950RequiredInput("key", packedKvDataTypes, true);
        addAscend950RequiredInput("value", packedKvDataTypes, true);
        addAscend950Input("sparse_indices", REQUIRED, {ge::DT_INT32});
        addAscend950Input("key_dequant_scale", OPTIONAL, {ge::DT_FLOAT});
        addAscend950Input("value_dequant_scale", OPTIONAL, {ge::DT_FLOAT});
        addAscend950Input("block_table", OPTIONAL, {ge::DT_INT32});
        addAscend950Input("actual_seq_lengths_query", OPTIONAL, {ge::DT_INT32});
        addAscend950Input("actual_seq_lengths_kv", OPTIONAL, {ge::DT_INT32});
        addAscend950Input("sinks", OPTIONAL, {ge::DT_FLOAT});
        aicore_config_95.Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        aicore_config_95.Output("softmax_max")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        aicore_config_95.Output("softmax_sum")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        const auto configureV2Ascend950 = [](OpAICoreConfig &config) {
            config.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(true);
        };
        configureV2Ascend950(aicore_config_95);
        this->AICore().AddConfig("ascend950", aicore_config_95); // KV-SFA V2: tile config.
    }
};
OP_ADD(KvQuantSparseFlashAttentionV2);
} // namespace ops
