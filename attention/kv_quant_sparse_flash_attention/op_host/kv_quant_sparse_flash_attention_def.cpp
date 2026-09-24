/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_flash_attention_def.cpp
 * \brief
 */

#include <initializer_list>
#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class KvQuantSparseFlashAttention : public OpDef {
public:
    explicit KvQuantSparseFlashAttention(const char *name)
        : OpDef(name)
    {
        const auto addV1Input = [this](const char *inputName, Option paramType,
                                       std::initializer_list<ge::DataType> dataTypes) {
            this->Input(inputName)
                .ParamType(paramType)
                .DataType(std::vector<ge::DataType>(dataTypes))
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
        };
        addV1Input("query", REQUIRED, {ge::DT_FLOAT16, ge::DT_BF16});
        addV1Input("key", REQUIRED, {ge::DT_INT8, ge::DT_INT8});
        this->Input("value").ParamType(REQUIRED).Follow("key").AutoContiguous();
        addV1Input("sparse_indices", REQUIRED, {ge::DT_INT32, ge::DT_INT32});
        addV1Input("key_dequant_scale", OPTIONAL, {ge::DT_FLOAT, ge::DT_FLOAT});
        addV1Input("value_dequant_scale", OPTIONAL, {ge::DT_FLOAT, ge::DT_FLOAT});
        addV1Input("block_table", OPTIONAL, {ge::DT_INT32, ge::DT_INT32});
        addV1Input("actual_seq_lengths_query", OPTIONAL, {ge::DT_INT32, ge::DT_INT32});
        addV1Input("actual_seq_lengths_kv", OPTIONAL, {ge::DT_INT32, ge::DT_INT32});
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        struct IntAttributeSpec {
            const char *name;
            Option paramType;
            int64_t defaultValue;
        };
        const IntAttributeSpec v1PreLayoutIntAttributes[] = {
            {"key_quant_mode", REQUIRED, 1}, {"value_quant_mode", REQUIRED, 1},
            {"sparse_block_size", OPTIONAL, 1}};
        const IntAttributeSpec v1PostLayoutIntAttributes[] = {
            {"sparse_mode", OPTIONAL, 3},
            {"pre_tokens", OPTIONAL, INT64_MAX}, {"next_tokens", OPTIONAL, INT64_MAX},
            {"attention_mode", OPTIONAL, 0}, {"quant_scale_repo_mode", OPTIONAL, 1},
            {"tile_size", OPTIONAL, 128}, {"rope_head_dim", OPTIONAL, 64}};
        this->Attr("scale_value").AttrType(REQUIRED).Float(1.0);
        for (const IntAttributeSpec &attribute : v1PreLayoutIntAttributes) {
            this->Attr(attribute.name).AttrType(attribute.paramType).Int(attribute.defaultValue);
        }
        this->Attr("layout_query").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("BSND");
        for (const IntAttributeSpec &attribute : v1PostLayoutIntAttributes) {
            this->Attr(attribute.name).AttrType(attribute.paramType).Int(attribute.defaultValue);
        }
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config); // KV-SFA: query config.
        this->AICore().AddConfig("ascend910_93", aicore_config);

        OpAICoreConfig aicore_config_95;
        const auto addV1Ascend950Input = [&aicore_config_95](const char *inputName, Option paramType,
                                                              std::initializer_list<ge::DataType> dataTypes) {
            aicore_config_95.Input(inputName)
                .ParamType(paramType)
                .DataTypeList(std::vector<ge::DataType>(dataTypes))
                .FormatList({ge::FORMAT_ND})
                .AutoContiguous();
        };
        aicore_config_95.Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        const std::vector<ge::DataType> v1PackedKvDataTypes = {
            ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_INT8,
            ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_INT8};
        aicore_config_95.Input("key")
            .ParamType(REQUIRED)
            .DataType(v1PackedKvDataTypes)
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicore_config_95.Input("value")
            .ParamType(REQUIRED)
            .DataType(v1PackedKvDataTypes)
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        struct Ascend950OptionalInputSpec {
            const char *name;
            Option paramType;
            ge::DataType dataType;
        };
        const Ascend950OptionalInputSpec v1OptionalInputs[] = {
            {"sparse_indices", REQUIRED, ge::DT_INT32}, {"key_dequant_scale", OPTIONAL, ge::DT_FLOAT},
            {"value_dequant_scale", OPTIONAL, ge::DT_FLOAT}, {"block_table", OPTIONAL, ge::DT_INT32},
            {"actual_seq_lengths_query", OPTIONAL, ge::DT_INT32}, {"actual_seq_lengths_kv", OPTIONAL, ge::DT_INT32}};
        for (const Ascend950OptionalInputSpec &input : v1OptionalInputs) {
            addV1Ascend950Input(input.name, input.paramType, {input.dataType});
        }
        aicore_config_95.Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        const auto configureV1Ascend950 = [](OpAICoreConfig &config) {
            config.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(true);
        };
        configureV1Ascend950(aicore_config_95);
        this->AICore().AddConfig("ascend950", aicore_config_95); // KV-SFA: tile config.
    }
};
OP_ADD(KvQuantSparseFlashAttention);
} // namespace ops
