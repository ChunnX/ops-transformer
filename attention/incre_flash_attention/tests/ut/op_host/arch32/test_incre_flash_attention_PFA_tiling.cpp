/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../../../../../prompt_flash_attention/op_host/prompt_flash_attention_tiling_compile_info.h"
#include "tiling_case_executor.h"
#include "op_host_csv_case_loader.h"
#include "softmax_tiling_mocker.h"

namespace {

class IncreFlashAttentionPfaTiling : public testing::Test {
protected:
    void SetUp() override
    {
        SoftmaxTilingMocker::GetInstance().SetSocVersion("Ascend910B");
    }

    void TearDown() override
    {
        SoftmaxTilingMocker::GetInstance().Reset();
    }
};

} // namespace

TEST_F(IncreFlashAttentionPfaTiling, IncreFlashAttention_PFA_tiling_0)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actualSeqKv[] = {256};
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            TD_DEFAULT,
            TD_DEFAULT,
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            TD_DEFAULT,
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            TD_DEFAULT,
        },
        {
            {{{1, 16, 1024}, {1, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo,
        "Ascend910B",
        64,
        262144,
        16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(IncreFlashAttentionPfaTiling, IncreFlashAttention_PFA_tiling_1)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
            {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 8, 128, 128}, {4, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
            TD_DEFAULT,
        },
        {
            {{{1, 8, 16, 128}, {1, 8, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo,
        "Ascend910B",
        64,
        262144,
        16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
