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

class IncreFlashAttentionPfaArch38Tiling : public testing::Test {
protected:
    void SetUp() override
    {
        SoftmaxTilingMocker::GetInstance().SetSocVersion("mc62cm12a");
    }

    void TearDown() override
    {
        SoftmaxTilingMocker::GetInstance().Reset();
    }
};

} // namespace

TEST_F(IncreFlashAttentionPfaArch38Tiling, IncreFlashAttention_PFA_arch38_tiling_0)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::MC62CM12A};
    int64_t actualSeqKv[] = {2048};
    gert::TilingContextPara tilingContextPara(
        "IncreFlashAttention",
        {
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_INT8, ge::FORMAT_ND},    // key input1
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
            TD_DEFAULT,                                                               // pse_shift input3
            TD_DEFAULT,                                                               // atten_mask input4
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv},             // actual_seq_lengths_kv
            TD_DEFAULT,                                                               // deq_scale1 input6
            TD_DEFAULT,                                                               // quant_scale1 input7
            TD_DEFAULT,                                                               // deq_scale2 input8
            TD_DEFAULT,                                                               // quant_scale2 input9
            TD_DEFAULT,                                                               // quant_offset2 input10
            TD_DEFAULT,                                                               // antiquant_scale input11
            TD_DEFAULT,                                                               // antiquant_offset input12
            TD_DEFAULT,                                                               // block_table input13
            TD_DEFAULT,                                                               // kv_padding_size input14
        },
        {
            {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo,
        "mc62cm12a",
        64,
        262144,
        16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
