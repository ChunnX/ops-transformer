/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/prompt_flash_attention_tiling_compile_info.h"
#include "../../../../op_host/prompt_flash_attention_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "softmax_tiling_mocker.h"
using namespace std;

std::string PromptFlashAttention_A2SocInfo = "{\n"
                                   "  \"hardware_info\": {\n"
                                   "    \"BT_SIZE\": 0,\n"
                                   "    \"load3d_constraints\": \"1\",\n"
                                   "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                                   "    \"Intrinsic_data_move_l12ub\": true,\n"
                                   "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                                   "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                                   "    \"UB_SIZE\": 196608,\n"
                                   "    \"L2_SIZE\": 201326592,\n"
                                   "    \"L1_SIZE\": 524288,\n"
                                   "    \"L0A_SIZE\": 65536,\n"
                                   "    \"L0B_SIZE\": 65536,\n"
                                   "    \"L0C_SIZE\": 131072,\n"
                                   "    \"vector_core_cnt\": 40,\n"
                                   "    \"cube_core_cnt\": 20,\n"
                                   "    \"socVersion\": \"Ascend910_B3\"\n"
                                   "  }\n"
                                   "}";


class PromptFlashAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PromptFlashAttentionTiling SetUp" << std::endl;
        SoftmaxTilingMocker::GetInstance().SetSocVersion("Ascend910B");
    }

    static void TearDownTestCase()
    {
        std::cout << "PromptFlashAttentionTiling TearDown" << std::endl;
        SoftmaxTilingMocker::GetInstance().Reset();
    }
};

// BNSD
TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_0)
{
    optiling::PromptFlashAttentionCompileInfo compileInfo = {64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    int64_t actual_seq_qlist[] = {556, 732, 637, 573, 149, 158, 278, 1011, 623, 683, 680, 449, 538, 920, 396, 322, 268, 153, 452, 458, 821, 1001, 744};
    int64_t actual_seq_kvlist[] = {556, 732, 637, 573, 149, 158, 278, 1011, 623, 683, 680, 449, 538, 920, 396, 322, 268, 153, 452, 458, 821, 1001, 744};
    gert::TilingContextPara tilingContextPara(
        "PromptFlashAttention",
        {
            {{{23, 40, 1024, 128}, {23, 40, 1024, 128}}, ge::DT_INT8, ge::FORMAT_ND},  // query input0
            {{{23, 40, 9088, 128}, {23, 40, 9088, 128}}, ge::DT_INT8, ge::FORMAT_ND},  // key input1
            {{{23, 40, 9088, 128}, {23, 40, 9088, 128}}, ge::DT_INT8, ge::FORMAT_ND},  // value input2
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // pse_shift input3
            {{{23, 1024, 9088}, {23, 1024, 9088}}, ge::DT_BOOL, ge::FORMAT_ND},    // atten_mask input4
            {{{23}, {23}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
            {{{23}, {23}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
            {{{40, 128}, {40, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
        },
        {
            {{{23, 40, 1024, 128}, {23, 40, 1024, 128}}, ge::DT_INT8, ge::FORMAT_ND}
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(488)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-127)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}
        },
        &compileInfo, "Ascend910B", PromptFlashAttention_A2SocInfo, 4096);
    int64_t expectTilingKey = 4423156480;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// // // BSH
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_1)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t* actual_seq_qlist = nullptr;
// //     int64_t* actual_seq_kvlist = nullptr;
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
// //             {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1
// //             {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // pse_shift input3
// //             {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},    // atten_mask input4
// //             {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
// //             {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
// //         },
// //         {
// //             {{{256, 14, 5120}, {256, 14, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0884f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 266601217;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // BSND
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_2)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048, 2048, 1024, 1024, 2048, 2028, 2048, 1024};
// //     int64_t actual_seq_kvlist[] = {2048, 1024, 2048, 2048, 2048, 1024, 1024, 1024};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND},  // query input0
// //             {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND},  // key input1
// //             {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND},  // value input2
// //             {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},    // pse_shift input3
// //             {{{8, 1, 2048, 2048}, {8, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},    // atten_mask input4
// //             {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
// //             {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
// //             {{{1, 1, 40, 128}, {1, 1, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // quant_scale2 input8
// //             {{{1, 1, 40, 128}, {1, 1, 40, 128}}, ge::DT_BF16, ge::FORMAT_ND}    // quant_offset2 input9
// //         },
// //         {
// //             {{{8, 2048, 40, 128}, {8, 2048, 40, 128}}, ge::DT_INT8, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0884f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1000)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 266601217;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // TND
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_3)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
    
//     int64_t actual_seq_qlist[] = {409, 818, 1227, 1636, 2045, 2454, 2863, 3272, 3681, 4090, 4499, 4908, 5317, 5726, 6135, 6544, 6953, 7362, 7771, 8180, 8589,
//                 8998, 9407, 9816, 10225, 10634, 11043, 11452, 11861, 12270, 12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
//     int64_t actual_seq_kvlist[] = {409, 818, 1227, 1636, 2045, 2454, 2863, 3272, 3681, 4090, 4499, 4908, 5317, 5726, 6135, 6544, 6953, 7362, 7771, 8180, 8589,
//                 8998, 9407, 9816, 10225, 10634, 11043, 11452, 11861, 12270, 12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},  // query input0
//             {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},  // key input1
//             {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},  // value input2
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // pse_shift input3
//             {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},    // atten_mask input4
//             {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
//             {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
//         },
//         {
//             {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
//             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878364870323f)},
//             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
//             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
//             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
//             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
//             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 4000000000000000000;
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }

// // // k = 0/v = 0/out = 0
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_4)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
// //             {{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1
// //             {{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
// //             {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},    // pse_shift input3
// //             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},    // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
// //         },
// //         {
// //             {{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 1000000000000000020;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// // }

// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_5)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
// //             {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},    // pse_shift input3
// //             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},    // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 132383488;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // DT_HIFLOAT8
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_6)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},  // key input1
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
// //             {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},    // pse_shift input3
// //             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},    // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 132383488;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // num_heads < 0 || num_key_value_heads < 0
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_7)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
// //             {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},    // pse_shift input3
// //             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},    // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 132383488;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // sparse_mode = 0
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_8)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
// //             {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},    // pse_shift input3
// //             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},    // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 132383488;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // //  enableIFA = true
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_9)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {    // 硬件参数
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {};
// //     int64_t actual_seq_kvlist[] = {};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 5, 1, 128}, {1, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // pse_shift input3
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // atten_mask input4
// //             {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
// //             {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_kv 
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},    // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 5, 1, 128}, {1, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 1206124800;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // DT_FLOAT input type is not supported
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_10)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},  // query input0 DT_FLOAT not supported
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},  // key input1
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},  // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                              // pse_shift input3
// //             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                 // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // innerPrecise = 4 (APPROXIMATE_COMPUTATION) not supported on ASCEND910B
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_11)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // pse_shift input3
// //             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},       // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},      // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // TND layout does not support PSE shift
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_12)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
//     int64_t actual_seq_qlist[] = {8};
//     int64_t actual_seq_kvlist[] = {8};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{8, 8, 128}, {8, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},             // query input0 T=8,N=8,D=128
//             {{{8, 8, 128}, {8, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},             // key input1
//             {{{8, 8, 128}, {8, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},             // value input2
//             {{{1, 8, 8, 8}, {1, 8, 8, 8}}, ge::DT_BF16, ge::FORMAT_ND},          // pse_shift input3 (non-null, TND not supported)
//             {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},                               // atten_mask input4
//             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},    // actual_seq_lengths_q
//             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},   // actual_seq_lengths_kv
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale1 input5
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale1 input6
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale2 input7
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale2 input8
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // quant_offset2 input9
//         },
//         {
//             {{{8, 8, 128}, {8, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 0;
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// }

// // // BF16 input with FP16 attention mask is not supported
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_13)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_BF16, ge::FORMAT_ND},         // query input0
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_BF16, ge::FORMAT_ND},         // key input1
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_BF16, ge::FORMAT_ND},         // value input2
// //             {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                       // pse_shift input3
// //             {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // atten_mask input4 (FP16 mask, not supported with BF16)
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},             // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},            // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                      // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                      // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                        // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_BF16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_14)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {};
// //     int64_t actual_seq_kvlist[] = {};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 4096, 640}, {1, 4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // query input0
// //             {{{1, 4096, 640}, {1, 4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key input1
// //             {{{1, 4096, 640}, {1, 4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                       // pse_shift input3
// //             {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // atten_mask input4 (FP16 mask, not supported with BF16)
// //             {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},             // actual_seq_lengths_q
// //             {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},            // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                      // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                      // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                        // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 4096, 640}, {1, 4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // FP16 high-precision mode with FP16 attention mask is not supported
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_15)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},      // query input0
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},      // key input1
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},      // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                    // pse_shift input3
// //             {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // atten_mask input4 (FP16 mask)
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},             // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},            // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                      // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                      // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                        // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // BNSD layout with mismatched D dimensions between query and key/value
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_16)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0 D=128
// //             {{{1, 5, 2048, 64}, {1, 5, 2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key input1 D=64 mismatch
// //             {{{1, 5, 2048, 64}, {1, 5, 2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // value input2 D=64 mismatch
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // pse_shift input3
// //             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},       // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},      // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // INT8 input with empty deq_scale1 is not allowed
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_17)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_INT8, ge::FORMAT_ND},    // query input0
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_INT8, ge::FORMAT_ND},    // key input1
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_INT8, ge::FORMAT_ND},    // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // pse_shift input3
// //             {{{1, 1, 2048, 2048}, {1, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},  // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},        // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},       // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                 // deq_scale1 empty (required for INT8)
// //             {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // quant_scale1 input6
// //             {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},                              // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                   // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // key dtype and value dtype must be consistent
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_18)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_INT8, ge::FORMAT_ND},    // key input1 (INT8)
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2 (FP16) dtype mismatch
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // pse_shift input3
// //             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},       // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},      // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // batch size exceeds BLIMIT (65536) triggers GRAPH_FAILED for empty KV tensors
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_19)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t *actual_seq_qlist = nullptr;
// //     int64_t *actual_seq_kvlist = nullptr;
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{65537, 5, 1, 128}, {65537, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0 B=65537>BLIMIT
// //             {{{0, 5, 1, 128}, {0, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key input1 empty
// //             {{{0, 5, 1, 128}, {0, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value input2 empty
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift input3
// //             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                   // atten_mask input4
// //             {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},         // actual_seq_lengths_q
// //             {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},        // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                    // quant_offset2 input9
// //         },
// //         {
// //             {{{0, 5, 1, 128}, {0, 5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 20;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // num_heads must be divisible by num_key_value_heads
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_20)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 40, 2048, 128}, {1, 40, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0 N=40
// //             {{{1, 7, 2048, 128}, {1, 7, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key input1 N=7
// //             {{{1, 7, 2048, 128}, {1, 7, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // value input2 N=7
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift input3
// //             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                   // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},         // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},        // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                    // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 40, 2048, 128}, {1, 40, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // FP32 attention mask type is not supported
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_21)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // query input0
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key input1
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                       // pse_shift input3
// //             {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},               // atten_mask DT_FLOAT (not supported)
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},               // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},              // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                        // deq_scale1
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                         // quant_scale1
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                        // deq_scale2
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                         // quant_scale2
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                          // quant_offset2
// //         },
// //         {
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // FP16 attention mask with isRowInvalid mode (inner_precise=3) is not supported
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_22)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // query input0
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key input1
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                       // pse_shift input3
// //             {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // atten_mask DT_FLOAT16 (not supported when isRowInvalid)
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},               // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},              // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                        // deq_scale1
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                         // quant_scale1
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                        // deq_scale2
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                         // quant_scale2
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                          // quant_offset2
// //         },
// //         {
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}
// //         },  // inner_precise=3: isRowInvalid=1, HIGH_PERFORMANCE
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // INT8 input with FP16 output must not have quant_scale2
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_23)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_INT8, ge::FORMAT_ND},           // query input0 INT8
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_INT8, ge::FORMAT_ND},           // key input1 INT8
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_INT8, ge::FORMAT_ND},           // value input2 INT8
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                      // pse_shift input3
// //             {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},                                         // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},              // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},             // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                       // deq_scale1
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                        // quant_scale1
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                       // deq_scale2
// //             {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // quant_scale2 non-null (invalid for INT8 input + FP16 output)
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                         // quant_offset2
// //         },
// //         {
// //             {{{1, 8, 2048, 128}, {1, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },          // FP16 output
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // num_heads / num_key_value_heads ratio exceeds the maximum allowed value of 64
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_24)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {64};
// //     int64_t actual_seq_kvlist[] = {64};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 128, 64, 128}, {1, 128, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0 N=128
// //             {{{1, 1, 64, 128}, {1, 1, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},      // key input1 N=1
// //             {{{1, 1, 64, 128}, {1, 1, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},      // value input2 N=1
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // pse_shift input3
// //             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                  // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},        // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},       // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                 // deq_scale1
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // quant_scale1
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                 // deq_scale2
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // quant_scale2
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                   // quant_offset2
// //         },
// //         {
// //             {{{1, 128, 64, 128}, {1, 128, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},         // ratio=128/1=128 > 64
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // TND layout does not support all-INT8 QKV data types
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_25)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
//     int64_t actual_seq_qlist[] = {8};
//     int64_t actual_seq_kvlist[] = {8};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{8, 8, 128}, {8, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},               // query input0 TND T=8,N=8,D=128 INT8
//             {{{8, 8, 128}, {8, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},               // key input1 INT8
//             {{{8, 8, 128}, {8, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},               // value input2 INT8
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                              // pse_shift input3
//             {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},                                 // atten_mask input4
//             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_q
//             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale1
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // quant_scale1
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                               // deq_scale2
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // quant_scale2
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                 // quant_offset2
//         },
//         {
//             {{{8, 8, 128}, {8, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 0;
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// }

// // // key and value tensor dimension count must match
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_26)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {2048};
// //     int64_t actual_seq_kvlist[] = {2048};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query input0 4D
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key input1 4D
// //             {{{5, 2048, 128}, {5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value input2 3D (dimNum mismatch)
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                // pse_shift input3
// //             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                                  // atten_mask input4
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},        // actual_seq_lengths_q
// //             {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},       // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                 // deq_scale1
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // quant_scale1
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                 // deq_scale2
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // quant_scale2
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                   // quant_offset2
// //         },
// //         {
// //             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // TND BF16 D=192, sparse_mode=0 (no attenMask) - BENCHMARK_TILING_KEY_1 path
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_27)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
//     int64_t actual_seq_qlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
//                                   4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
//                                   8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
//                                   12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
//     int64_t actual_seq_kvlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
//                                    4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
//                                    8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
//                                    12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // query input0
//             {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // key input1
//             {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // value input2
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                            // pse_shift input3
//             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                              // atten_mask input4 (null, sparse_mode=0 requires no mask)
//             {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
//             {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale1 input5
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale1 input6
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale2 input7
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale2 input8
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // quant_offset2 input9
//         },
//         {
//             {{{16384, 64, 192}, {16384, 64, 192}}, ge::DT_BF16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
//             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878364870323f)},
//             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
//             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
//             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
//             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 4000000000000000001;
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }

// // TND BF16 D=128, sparse_mode=3 (RIGHT_DOWN) with attenMask - BENCHMARK_TILING_KEY_2 path
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_28)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
//     int64_t actual_seq_qlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
//                                   4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
//                                   8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
//                                   12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
//     int64_t actual_seq_kvlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
//                                    4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
//                                    8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
//                                    12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{16384, 64, 128}, {16384, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // query input0
//             {{{16384, 64, 128}, {16384, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // key input1
//             {{{16384, 64, 128}, {16384, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // value input2
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                            // pse_shift input3
//             {{{1, 2048, 2048}, {1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},     // atten_mask input4
//             {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
//             {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale1 input5
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale1 input6
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale2 input7
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale2 input8
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // quant_offset2 input9
//         },
//         {
//             {{{16384, 64, 128}, {16384, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
//             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
//             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
//             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
//             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
//             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 4000000000000000002;
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }

// // TND BF16 D=128, sparse_mode=0 (no attenMask) - BENCHMARK_TILING_KEY_3 path
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_29)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
//     int64_t actual_seq_qlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
//                                   4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
//                                   8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
//                                   12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
//     int64_t actual_seq_kvlist[] = {409,   818,   1227,  1636,  2045,  2454,  2863,  3272,  3681,  4090,
//                                    4499,  4908,  5317,  5726,  6135,  6544,  6953,  7362,  7771,  8180,
//                                    8589,  8998,  9407,  9816,  10225, 10634, 11043, 11452, 11861, 12270,
//                                    12679, 13088, 13497, 13906, 14315, 14724, 15133, 15542, 15951, 16384};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{16384, 64, 128}, {16384, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // query input0
//             {{{16384, 64, 128}, {16384, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // key input1
//             {{{16384, 64, 128}, {16384, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // value input2
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                            // pse_shift input3
//             {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                              // atten_mask input4 (null, sparse_mode=0 requires no mask)
//             {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_q
//             {{{40}, {40}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale1 input5
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale1 input6
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale2 input7
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale2 input8
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // quant_offset2 input9
//         },
//         {
//             {{{16384, 64, 128}, {16384, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
//             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
//             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16384)},
//             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
//             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 4000000000000000003;
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }

// // // BNSD FP16 HIGH_PERFORMANCE D=128, sparse_mode=0, no mask - CVDIFF path, key=1012
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_30)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
// //             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
// //             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                  // pse_shift input3
// //             {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},                                     // atten_mask input4 (null)
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                    // actual_seq_lengths_q
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                    // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                   // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                   // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                     // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 1000000000000001012;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // BNSD FP16 HIGH_PRECISION D=128, sparse_mode=0, no mask - CVDIFF path, key=1612 (+600 for FP16 high precision)
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_31)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0
// //             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
// //             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                  // pse_shift input3
// //             {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},                                     // atten_mask input4 (null)
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                    // actual_seq_lengths_q
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                    // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                   // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                   // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                     // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 1000000000000001612;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // BSH FP16 HIGH_PERFORMANCE D=128, sparse_mode=0, no mask - CVDIFF path, key=101012 (+100000 for BSH)
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_32)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0 BSH B=1,S=2048,H=32*128
// //             {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
// //             {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                            // pse_shift input3
// //             {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},                               // atten_mask input4 (null)
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                              // actual_seq_lengths_q
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                              // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 1000000000000101012;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // BSND BF16 D=128, sparse_mode=0, no mask - CVDIFF path, key=111112 (+100 BF16 +10000 out_BF16 +100000 BSND)
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_33)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 2048, 32, 128}, {1, 2048, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // query input0 BSND B=1,S=2048,N=32,D=128
// //             {{{1, 2048, 32, 128}, {1, 2048, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key input1
// //             {{{1, 2048, 32, 128}, {1, 2048, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value input2
// //             {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                  // pse_shift input3
// //             {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},                                  // atten_mask input4 (null)
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_q
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 2048, 32, 128}, {1, 2048, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 1000000000000111112;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // BNSD BF16 D=128, sparse_mode=0, no mask - CVDIFF path, key=11112 (+100 BF16, +10000 outBF16)
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_34)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // query input0 BNSD
// //             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key input1
// //             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value input2
// //             {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                  // pse_shift input3
// //             {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},                                  // atten_mask input4 (null)
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_q
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                  // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_BF16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 1000000000000011112;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // BSH FP16 HIGH_PRECISION D=128, sparse_mode=0, no mask - CVDIFF path, key=101612 (+600 FP16+HP, +100000 BSH)
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_35)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query input0 BSH B=1,S=2048,H=32*128
// //             {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key input1
// //             {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                            // pse_shift input3
// //             {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},                               // atten_mask input4 (null)
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                              // actual_seq_lengths_q
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                              // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                             // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 1000000000000101612;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // // BNSD FP16 HIGH_PERFORMANCE D=128, S=16384, N=16 - SPLIT_ONEN_CUBE path, key=2001012 (+2000000 cube split)
// // // baseCond: D=128, FP16, no PSE, no actual_seq, seqQ==seqKV=16384, CVDIFF
// // // enableOneNByCubeSeqMode: S=16384>=16384, b*n=16>=12
// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_36)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 16, 16384, 128}, {1, 16, 16384, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query BNSD B=1,N=16,S=16384,D=128
// //             {{{1, 16, 16384, 128}, {1, 16, 16384, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key
// //             {{{1, 16, 16384, 128}, {1, 16, 16384, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                    // pse_shift (null)
// //             {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},                                       // atten_mask (null)
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                      // actual_seq_lengths_q (null)
// //             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                      // actual_seq_lengths_kv (null)
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                     // deq_scale1
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                      // quant_scale1
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                     // deq_scale2
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                      // quant_scale2
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                       // quant_offset2
// //         },
// //         {
// //             {{{1, 16, 16384, 128}, {1, 16, 16384, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 1000000000002001012;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_37)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {};
// //     int64_t actual_seq_kvlist[] = {};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 4096, 640}, {1, 4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // query input0
// //             {{{1, 4096, 640}, {1, 4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key input1
// //             {{{1, 4096, 640}, {1, 4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                       // pse_shift input3
// //             {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // atten_mask input4
// //             {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},             // actual_seq_lengths_q
// //             {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},            // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                      // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                      // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                        // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 4096, 640}, {1, 4096, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", 64, 262144, 16384);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_38)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     int64_t actual_seq_qlist[] = {};
// //     int64_t actual_seq_kvlist[] = {};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{1, 4, 2048, 256}, {1, 4, 2048, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // query input0
// //             {{{1, 4, 2048, 256}, {1, 4, 2048, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key input1
// //             {{{1, 4, 2048, 256}, {1, 4, 2048, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                       // pse_shift input3
// //             {{{1, 4, 2048, 2048}, {1, 4, 2048, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // atten_mask input4
// //             {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},             // actual_seq_lengths_q
// //             {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},            // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                      // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                      // deq_scale2 input7
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // quant_scale2 input8
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                        // quant_offset2 input9
// //         },
// //         {
// //             {{{1, 4, 2048, 256}, {1, 4, 2048, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0625)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo, "Ascend910B", PromptFlashAttention_A2SocInfo, 4096);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_39)
// // {
// //     optiling::PromptFlashAttentionCompileInfo compileInfo = {
// //         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432, platform_ascendc::SocVersion::ASCEND910B};
// //     // struct PromptFlashAttentionCompileInfo {} compileInfo;
// //     int64_t actual_seq_qlist[] = {32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 
// //         32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32};
// //     int64_t actual_seq_kvlist[] = {32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 
// //         32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32};
// //     gert::TilingContextPara tilingContextPara(
// //         "PromptFlashAttention",
// //         {
// //             {{{2048, 1280}, {2048, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // query input0
// //             {{{2048, 1280}, {2048, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key input1
// //             {{{2048, 1280}, {2048, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value input2
// //             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                       // pse_shift input3
// //             {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},          // atten_mask input4
// //             {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},             // actual_seq_lengths_q
// //             {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},            // actual_seq_lengths_kv
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                      // deq_scale1 input5
// //             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // quant_scale1 input6
// //             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                      // deq_scale2 input7
// //             {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                       // quant_scale2 input8
// //             {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}                                        // quant_offset2 input9
// //         },
// //         {
// //             {{{2048, 1280}, {2048, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND}
// //         },
// //         {
// //             {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
// //             {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0884)},
// //             {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
// //             {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SH")},
// //             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
// //             {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
// //             {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
// //         },
// //         &compileInfo);
// //     int64_t expectTilingKey = 0;
// //     std::string expectTilingData = "";
// //     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
// // }

// // BNSD FP16, MHA, no mask, HIGH_PERFORMANCE → GRAPH_SUCCESS
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_33)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432,
//         platform_ascendc::SocVersion::ASCEND910B};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query B=1,N=32,S=2048,D=128
//             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key
//             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // pse_shift: null
//             {{{}, {}}, ge::DT_BOOL,    ge::FORMAT_ND},   // atten_mask: null
//             {{{}, {}}, ge::DT_INT64,   ge::FORMAT_ND},   // actual_seq_lengths_q: null
//             {{{}, {}}, ge::DT_INT64,   ge::FORMAT_ND},   // actual_seq_lengths_kv: null
//             {{{}, {}}, ge::DT_UINT64,  ge::FORMAT_ND},   // deq_scale1
//             {{{}, {}}, ge::DT_FLOAT,   ge::FORMAT_ND},   // quant_scale1
//             {{{}, {}}, ge::DT_UINT64,  ge::FORMAT_ND},   // deq_scale2
//             {{{}, {}}, ge::DT_FLOAT,   ge::FORMAT_ND},   // quant_scale2
//             {{{}, {}}, ge::DT_FLOAT,   ge::FORMAT_ND}    // quant_offset2
//         },
//         {
//             {{{1, 32, 2048, 128}, {1, 32, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads",           Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
//             {"scale_value",         Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens",          Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"next_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"input_layout",        Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
//             {"sparse_mode",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"inner_precise",       Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 1000000000000001012; // CVDIFF, HIGH_PERFORMANCE, BNSD, D=128, no mask
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }

// // BNSD FP16, MHA, BOOL mask, HIGH_PRECISION (innerPrecise=0) → GRAPH_SUCCESS
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_34)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432,
//         platform_ascendc::SocVersion::ASCEND910B};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query B=2,N=8,S=1024,D=128
//             {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key
//             {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // pse_shift: null
//             {{{2, 1024, 1024}, {2, 1024, 1024}}, ge::DT_BOOL, ge::FORMAT_ND},        // atten_mask (B,S_q,S_kv)
//             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_q: null
//             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                 // actual_seq_lengths_kv: null
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND}
//         },
//         {
//             {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads",           Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"scale_value",         Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens",          Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"next_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"input_layout",        Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"sparse_mode",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"inner_precise",       Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)} // HIGH_PRECISION
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 1000000000000001612; // +600 for HIGH_PRECISION
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }

// // BNSD BF16, GQA ratio=8 (N_q=8, N_kv=1), no mask → GRAPH_SUCCESS
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_35)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432,
//         platform_ascendc::SocVersion::ASCEND910B};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // query N_q=8
//             {{{1, 1, 1024, 128}, {1, 1, 1024, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key   N_kv=1
//             {{{1, 1, 1024, 128}, {1, 1, 1024, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value N_kv=1
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_BOOL,    ge::FORMAT_ND}, // no mask
//             {{{}, {}}, ge::DT_INT64,   ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_INT64,   ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND}
//         },
//         {
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_BF16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads",           Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"scale_value",         Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens",          Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"next_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"input_layout",        Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, // GQA ratio=8
//             {"sparse_mode",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"inner_precise",       Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 0; // 待运行后填写实际值
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }

// // BSH FP16, MHA, no mask → GRAPH_SUCCESS
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_36)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432,
//         platform_ascendc::SocVersion::ASCEND910B};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query B=1,S=2048,H=32*128
//             {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key
//             {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_BOOL,    ge::FORMAT_ND}, // no mask
//             {{{}, {}}, ge::DT_INT64,   ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_INT64,   ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND}
//         },
//         {
//             {{{1, 2048, 4096}, {1, 2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads",           Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
//             {"scale_value",         Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens",          Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"next_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"input_layout",        Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
//             {"sparse_mode",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"inner_precise",       Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 1000000000000101012; // +100000 for BSH layout offset
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }

// // BSH FP16, GQA ratio=4 (N_q=8, N_kv=2), with actualSeqLengths → GRAPH_SUCCESS
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_37)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432,
//         platform_ascendc::SocVersion::ASCEND910B};
//     int64_t actual_seq_qlist[]  = {512, 1024};
//     int64_t actual_seq_kvlist[] = {512, 1024};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{2, 1024, 1024}, {2, 1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query B=2,S=1024,H_q=8*128=1024
//             {{{2, 1024,  256}, {2, 1024,  256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key   H_kv=2*128=256
//             {{{2, 1024,  256}, {2, 1024,  256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_BOOL,    ge::FORMAT_ND},
//             {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
//             {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND}
//         },
//         {
//             {{{2, 1024, 1024}, {2, 1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads",           Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"scale_value",         Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens",          Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"next_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"input_layout",        Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, // GQA ratio=4
//             {"sparse_mode",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"inner_precise",       Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 0; // 待运行后填写
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }

// // BSND BF16, MHA, BOOL mask, sparse_mode=3 (RIGHT_DOWN) → GRAPH_SUCCESS
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_38)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432,
//         platform_ascendc::SocVersion::ASCEND910B};
//     int64_t actual_seq_qlist[]  = {1024, 2048};
//     int64_t actual_seq_kvlist[] = {1024, 2048};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{2, 2048, 16, 128}, {2, 2048, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // query BSND B=2,S=2048,N=16,D=128
//             {{{2, 2048, 16, 128}, {2, 2048, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key
//             {{{2, 2048, 16, 128}, {2, 2048, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // value
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             {{{2, 1, 2048, 2048}, {2, 1, 2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND}, // atten_mask (B,1,S_q,S_kv)
//             {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
//             {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND}
//         },
//         {
//             {{{2, 2048, 16, 128}, {2, 2048, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads",           Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
//             {"scale_value",         Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens",          Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"next_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"input_layout",        Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
//             {"sparse_mode",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}, // RIGHT_DOWN
//             {"inner_precise",       Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 0; // 待运行后填写
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }

// // BNSD FP16, empty key/value → tilingKey=20, GRAPH_SUCCESS
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_39)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432,
//         platform_ascendc::SocVersion::ASCEND910B};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query non-empty
//             {{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key empty (B=0)
//             {{{0, 5, 2048, 128}, {0, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value empty
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT8,   ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_INT64,   ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_INT64,   ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND}
//         },
//         {
//             {{{1, 5, 2048, 128}, {1, 5, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads",           Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
//             {"scale_value",         Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens",          Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"next_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"input_layout",        Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
//             {"sparse_mode",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"inner_precise",       Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 20; // EMPTY_KV_TILING_KEY
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }

// // BNSD FP16, sparse_mode=22 (ALIBI) → Base API path → GRAPH_SUCCESS
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_40)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432,
//         platform_ascendc::SocVersion::ASCEND910B};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                               // pse_shift: null
//             {{{8, 1024, 1024}, {8, 1024, 1024}}, ge::DT_BOOL, ge::FORMAT_ND},        // attenMask for ALIBI
//             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND}
//         },
//         {
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads",           Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"scale_value",         Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens",          Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"next_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"input_layout",        Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"sparse_mode",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(22)}, // ALIBI → Base API
//             {"inner_precise",       Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 0; // 待运行后填写
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }

// // BNSD INT8 full quant, INT8 output, per-tensor scale → GRAPH_SUCCESS
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_41)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432,
//         platform_ascendc::SocVersion::ASCEND910B};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_INT8, ge::FORMAT_ND},   // query
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_INT8, ge::FORMAT_ND},   // key
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_INT8, ge::FORMAT_ND},   // value
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                              // pse_shift: null
//             {{{1, 1, 1024, 1024}, {1, 1, 1024, 1024}}, ge::DT_BOOL, ge::FORMAT_ND}, // atten_mask
//             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
//             {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},  // deq_scale1 (per-tensor)
//             {{{1}, {1}}, ge::DT_FLOAT,  ge::FORMAT_ND},  // quant_scale1
//             {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},  // deq_scale2
//             {{{1}, {1}}, ge::DT_FLOAT,  ge::FORMAT_ND},  // quant_scale2 (per-tensor, INT8 output)
//             {{{}, {}},   ge::DT_FLOAT,  ge::FORMAT_ND}   // quant_offset2: null
//         },
//         {
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_INT8, ge::FORMAT_ND}    // INT8 output
//         },
//         {
//             {"num_heads",           Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"scale_value",         Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens",          Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"next_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"input_layout",        Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"sparse_mode",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"inner_precise",       Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)} // INT8不生效
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 0; // 待运行后填写
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }

// // BNSD FP16, inner_precise=2 (isRowInvalid=1, HIGH_PRECISION), BOOL mask → GRAPH_SUCCESS
// TEST_F(PromptFlashAttentionTiling, PromptFlashAttention_910b_tiling_42)
// {
//     optiling::PromptFlashAttentionCompileInfo compileInfo = {
//         64, 32, 262144, 524288, 262144, 65536, 65536, 33554432,
//         platform_ascendc::SocVersion::ASCEND910B};
//     gert::TilingContextPara tilingContextPara(
//         "PromptFlashAttention",
//         {
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             {{{1, 1, 1024, 1024}, {1, 1, 1024, 1024}}, ge::DT_BOOL, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND},
//             {{{}, {}}, ge::DT_FLOAT,  ge::FORMAT_ND}
//         },
//         {
//             {{{1, 8, 1024, 128}, {1, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
//         },
//         {
//             {"num_heads",           Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"scale_value",         Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
//             {"pre_tokens",          Ops::Transformer::AnyValue::CreateFrom<int64_t>(65535)},
//             {"next_tokens",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"input_layout",        Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
//             {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
//             {"sparse_mode",         Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//             {"inner_precise",       Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)} // isRowInvalid=1, HIGH_PRECISION
//         },
//         &compileInfo, "Ascend910B", 64, 262144, 16384);
//     int64_t expectTilingKey = 0; // 待运行后填写
//     std::string expectTilingData = "";
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
// }