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
 * \file sparse_flash_attention_def.cpp
 * \brief
 */

#include <initializer_list>
#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class SparseFlashAttention : public OpDef {
public:
    explicit SparseFlashAttention(const char *name) : OpDef(name)
    {
        const auto addSfaInput = [this](const char *inputName, Option paramType,
                                        std::initializer_list<ge::DataType> dataTypes) {
            this->Input(inputName)
                .ParamType(paramType)
                .DataType(std::vector<ge::DataType>(dataTypes))
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
        };
        addSfaInput("query", REQUIRED, {ge::DT_FLOAT16, ge::DT_BF16});
        addSfaInput("key", REQUIRED, {ge::DT_FLOAT16, ge::DT_BF16});
        addSfaInput("value", REQUIRED, {ge::DT_FLOAT16, ge::DT_BF16});
        addSfaInput("sparse_indices", REQUIRED, {ge::DT_INT32, ge::DT_INT32});
        addSfaInput("block_table", OPTIONAL, {ge::DT_INT32, ge::DT_INT32});
        addSfaInput("actual_seq_lengths_query", OPTIONAL, {ge::DT_INT32, ge::DT_INT32});
        addSfaInput("actual_seq_lengths_kv", OPTIONAL, {ge::DT_INT32, ge::DT_INT32});
        addSfaInput("query_rope", OPTIONAL, {ge::DT_FLOAT16, ge::DT_BF16});
        addSfaInput("key_rope", OPTIONAL, {ge::DT_FLOAT16, ge::DT_BF16});
        addSfaInput("sinks", OPTIONAL, {ge::DT_FLOAT, ge::DT_FLOAT});
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
        struct SfaIntAttribute {
            const char *name;
            int64_t defaultValue;
        };
        const SfaIntAttribute sfaPostLayoutIntAttributes[] = {
            {"sparse_mode", 3}, {"pre_tokens", INT64_MAX}, {"next_tokens", INT64_MAX}, {"attention_mode", 0}};
        this->Attr("scale_value").AttrType(REQUIRED).Float(1.0);
        this->Attr("sparse_block_size").AttrType(OPTIONAL).Int(1);
        this->Attr("layout_query").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("BSND");
        for (const SfaIntAttribute &attribute : sfaPostLayoutIntAttributes) {
            this->Attr(attribute.name).AttrType(OPTIONAL).Int(attribute.defaultValue);
        }
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig aicore_config;
        const auto configureSfaAicore = [](OpAICoreConfig &config) {
            config.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(true);
        };
        configureSfaAicore(aicore_config);
        this->AICore().AddConfig("ascend910b", aicore_config); // SFA: KV config.
        this->AICore().AddConfig("ascend910_93", aicore_config);

        OpAICoreConfig aicore_config_95;
        const auto addSfa950KvInput = [&aicore_config_95](const char *inputName, Option paramType) {
            aicore_config_95.Input(inputName)
                .ParamType(paramType)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .IgnoreContiguous();
        };
        addSfa950KvInput("key", REQUIRED);
        addSfa950KvInput("value", REQUIRED);
        addSfa950KvInput("key_rope", OPTIONAL);
        configureSfaAicore(aicore_config_95);
        this->AICore().AddConfig("ascend950", aicore_config_95); // SFA: tile config.
    }
};
OP_ADD(SparseFlashAttention);
} // namespace ops
