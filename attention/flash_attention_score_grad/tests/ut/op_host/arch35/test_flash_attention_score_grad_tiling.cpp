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
#include <vector>
#include <gtest/gtest.h>
#include "tiling/platform/platform_ascendc.h"
#include "../../../../common/include/op_host/tiling_base.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

std::string A5SocInfo = 
    "{\n"
    "  \"hardware_info\": {\n"
    "    \"BT_SIZE\": 0,\n"
    "    \"load3d_constraints\": \"1\",\n"
    "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
    "    \"Intrinsic_data_move_l12ub\": true,\n"
    "    \"Intrinsic_data_move_l0c2ub\": true,\n"
    "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
    "    \"UB_SIZE\": 262144,\n"
    "    \"L2_SIZE\": 134217728,\n"
    "    \"L1_SIZE\": 524288,\n"
    "    \"L0A_SIZE\": 65536,\n"
    "    \"L0B_SIZE\": 65536,\n"
    "    \"L0C_SIZE\": 262144,\n"
    "    \"CORE_NUM\": 32,\n"
    "    \"socVersion\": \"Ascend950\"\n"
    "  }\n"
    "}";


class FlashAttentionScoreGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScoreGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScoreGradTiling TearDown" << std::endl;
    }
};

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo MakeA5CompileInfo()
{
    return {
        64,                                      // aivNum
        32,                                      // aicNum
        196608,                                  // ubSize
        524288,                                  // l1Size
        65536,                                   // l0aSize
        65536,                                   // l0bSize
        131072,                                  // l0cSize
        33554432,                                // l2CacheSize
        32,                                      // coreNum
        platform_ascendc::SocVersion::ASCEND950  // socVersion
    };
}

std::vector<OpAttr> MakeAttrs(const std::string &layout = "BSH", int64_t sparseMode = 0,
                              float keepProb = 1.0f, int64_t preTokens = 65536, int64_t nextTokens = 65536,
                              int64_t headNum = 1, int64_t pseType = 1,
                              const std::string &softmaxInLayout = "")
{
    return {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
        {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(keepProb)},
        {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(preTokens)},
        {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(nextTokens)},
        {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(headNum)},
        {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
        {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode)},
        {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(pseType)},
        {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(softmaxInLayout)}
    };
}

std::vector<TensorDesc> MakeBshInputs(ge::DataType dtype = ge::DT_FLOAT16)
{
    return {
        {{{1, 128, 128}, {1, 128, 128}}, dtype, ge::FORMAT_ND},
        {{{1, 128, 128}, {1, 128, 128}}, dtype, ge::FORMAT_ND},
        {{{1, 128, 128}, {1, 128, 128}}, dtype, ge::FORMAT_ND},
        {{{1, 128, 128}, {1, 128, 128}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
        {{{1, 128, 128}, {1, 128, 128}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
    };
}

std::vector<TensorDesc> MakeBshOutputs(ge::DataType dtype = ge::DT_FLOAT16)
{
    return {
        {{{1, 128, 128}, {1, 128, 128}}, dtype, ge::FORMAT_ND},
        {{{1, 128, 128}, {1, 128, 128}}, dtype, ge::FORMAT_ND},
        {{{1, 128, 128}, {1, 128, 128}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
    };
}

std::vector<TensorDesc> MakeTndInputs(int64_t *actualSeqQ, int64_t *actualSeqKv, size_t seqSize,
                                      int64_t t1, int64_t t2, int64_t n = 2, int64_t d = 64,
                                      ge::DataType dtype = ge::DT_FLOAT16)
{
    return {
        {{{t1, n, d}, {t1, n, d}}, dtype, ge::FORMAT_ND},
        {{{t2, n, d}, {t2, n, d}}, dtype, ge::FORMAT_ND},
        {{{t2, n, d}, {t2, n, d}}, dtype, ge::FORMAT_ND},
        {{{t1, n, d}, {t1, n, d}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        {{{t1, n, 8}, {t1, n, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{t1, n, 8}, {t1, n, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
        {{{t1, n, d}, {t1, n, d}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{static_cast<int64_t>(seqSize)}, {static_cast<int64_t>(seqSize)}}, ge::DT_INT64, ge::FORMAT_ND, true,
         actualSeqQ},
        {{{static_cast<int64_t>(seqSize)}, {static_cast<int64_t>(seqSize)}}, ge::DT_INT64, ge::FORMAT_ND, true,
         actualSeqKv},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
    };
}

std::vector<TensorDesc> MakeTndOutputs(int64_t t1, int64_t t2, int64_t n = 2, int64_t d = 64,
                                       ge::DataType dtype = ge::DT_FLOAT16)
{
    return {
        {{{t1, n, d}, {t1, n, d}}, dtype, ge::FORMAT_ND},
        {{{t2, n, d}, {t2, n, d}}, dtype, ge::FORMAT_ND},
        {{{t2, n, d}, {t2, n, d}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
        {{{}, {}}, dtype, ge::FORMAT_ND},
    };
}

gert::TilingContextPara MakeA5Context(const std::vector<TensorDesc> &inputs, const std::vector<TensorDesc> &outputs,
                                      const std::vector<OpAttr> &attrs, void *compileInfo,
                                      int32_t deterministicInfo = 0)
{
    return gert::TilingContextPara(
        "FlashAttentionScoreGrad", inputs, outputs, attrs, compileInfo, "Ascend950", A5SocInfo, 4096,
        deterministicInfo);
}

void ExpectTilingFailed(const gert::TilingContextPara &tilingContextPara)
{
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

void ExpectTilingSuccess(const gert::TilingContextPara &tilingContextPara)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GT(tilingInfo.tilingKey, 0);
    ASSERT_GT(tilingInfo.workspaceSizes.size(), 0U);
}
} // namespace

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_0)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
    64,                                      // aivNum
    32,                                      // aicNum
    196608,                                   // ubSize
    524288,                                 // l1Size
    65536,                                   // l0aSize
    65536,                                // l0bSize
    131072,                                        // l0cSize
    33554432,                                        // l2CacheSize
    32,                                        // coreNum
    platform_ascendc::SocVersion::ASCEND950  // socVersion
};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q: S1=256, B=1, H1=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask: S1=256, S2=256
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1,1,256,8}, {1,1,256,8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1,1,256,8}, {1,1,256,8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{256,1,128}, {256,1,128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dk
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dv
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dq_rope
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dk_rope
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   
         },
        {
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577349537174544;
    std::string expectTilingData = "32 1 1 1 256 256 128 128 4575657222443697349 255 65536 65536 0 0 0 2 1 0 0 2 1099511627776 0 0 0 2 549755813952 549755813952 2 549755814016 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16384 8192 4 8192 8192 4 8192 8192 4 8192 0 0 0 4 61440 30720 114688 30720 1 16384 1 16384 1 0 0 0 0 0 0 0 0 0 0 1 32768 16384 16384 1 32768 16384 16384 1 32768 16384 16384 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_1)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
    64,                                      // aivNum
    32,                                      // aicNum
    196608,                                   // ubSize
    524288,                                 // l1Size
    65536,                                   // l0aSize
    65536,                                // l0bSize
    131072,                                        // l0cSize
    33554432,                                        // l2CacheSize
    32,                                        // coreNum
    platform_ascendc::SocVersion::ASCEND950  // socVersion
};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dy
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // attention_in
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dk
         {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dv
         {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dq_rope
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dk_rope
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   
         },
        {
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(45)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146416;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {21562880};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_2)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
    64,                                      // aivNum
    32,                                      // aicNum
    196608,                                   // ubSize
    524288,                                 // l1Size
    65536,                                   // l0aSize
    65536,                                // l0bSize
    131072,                                        // l0cSize
    33554432,                                        // l2CacheSize
    32,                                        // coreNum
    platform_ascendc::SocVersion::ASCEND950  // socVersion
};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{1, 1}, {1, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        // dk
         {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        // dv
         {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        // dq_rope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        // dk_rope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   
         },
        {
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953724962;
    std::string expectTilingData = "32 1 1 1 1 1 1 1 4575657222473777152 255 65536 0 2 0 0 2 1 0 0 2 4294967296 281474976710656 0 0 1 4294967360 4294967297 1 4294967424 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32 1 1 1 1 1 1 1 1 1 0 0 0 1 61440 30720 114688 30720 1 32 1 32 1 0 0 0 0 0 0 0 0 0 0 1 1 16384 1 1 1 16384 1 1 1 16384 1 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_3)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
    64,                                      // aivNum
    32,                                      // aicNum
    196608,                                   // ubSize
    524288,                                 // l1Size
    65536,                                   // l0aSize
    65536,                                // l0bSize
    131072,                                        // l0cSize
    33554432,                                        // l2CacheSize
    32,                                        // coreNum
    platform_ascendc::SocVersion::ASCEND950  // socVersion
};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dk
         {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dv
         {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dq_rope
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dk_rope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   
         },
        {
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08804509063256238f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577350074044432;
    std::string expectTilingData = "32 1 1 1 64 128 129 129 4575657222443651324 255 2147483647 2147483647 2 0 0 3 1 0 0 0 0 0 0 0 1 274877907008 274877907008 1 549755814016 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8192 8256 1 8256 16512 1 16512 16512 1 16512 0 0 0 1 61440 30720 114688 30720 1 8192 1 8192 1 0 0 0 0 0 0 0 0 0 0 1 8256 16384 8256 1 16512 16384 128 1 16512 16384 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_4)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
    64,                                      // aivNum
    32,                                      // aicNum
    196608,                                   // ubSize
    524288,                                 // l1Size
    65536,                                   // l0aSize
    65536,                                // l0bSize
    131072,                                        // l0cSize
    33554432,                                        // l2CacheSize
    32,                                        // coreNum
    platform_ascendc::SocVersion::ASCEND950  // socVersion
};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dy
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{2, 8, 2000, 8}, {2, 8, 2000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{2, 8, 2000, 8}, {2, 8, 2000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // attention_in
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{2, 2000, 8, 64}, {2, 2000, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{2, 2000, 8, 64}, {2, 2000, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dk
         {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dv
         {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dq_rope
         {{{2, 2000, 8, 64}, {2, 2000, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dk_rope
         {{{2, 2000, 8, 64}, {2, 2000, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   
         },
        {
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878364870323f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 20125462445953072;
    std::string expectTilingData = "32 2 8 1 2000 2000 192 128 4575657222441520442 255 2147483647 2147483647 2 0 256 1 1 0 0 0 0 72057598332895232 0 0 16 549755813952 343597383696 16 343597383808 549755813920 0 0 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 0 0 0 0 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096 0 0 0 0 2000000 96000 64 96000 96000 64 96000 64000 64 64000 0 0 0 64 61440 30720 114688 30720 66 3200 600479950314048 27776 1 242442313924672 20203526216832 147 4 4 59 59 500 500 32000 6 6144000 18432 6144 6 6144000 18432 6144 4 4096000 18432 4096 65536 24678912 49292288 0 65701376 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {87698944};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_5)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
    64,                                      // aivNum
    32,                                      // aicNum
    196608,                                   // ubSize
    524288,                                 // l1Size
    65536,                                   // l0aSize
    65536,                                // l0bSize
    131072,                                        // l0cSize
    33554432,                                        // l2CacheSize
    32,                                        // coreNum
    platform_ascendc::SocVersion::ASCEND950  // socVersion
};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{64, 123, 16, 389}, {64, 123, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{64, 64, 16, 389}, {64, 64, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{64, 64, 16, 389}, {64, 64, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{64, 123, 16, 389}, {64, 123, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{64, 16, 123, 8}, {64, 16, 123, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{64, 16, 123, 8}, {64, 16, 123, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{64, 123, 16, 389}, {64, 123, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{64, 123, 16, 389}, {64, 123, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
        // dk
         {{{64, 64, 16, 389}, {64, 64, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
        // dv
         {{{64, 64, 16, 389}, {64, 64, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        // dq_rope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        // dk_rope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   
         },
        {
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.05070201265633938f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140301101207586;
    std::string expectTilingData = "32 64 16 1 123 64 389 389 4575657222437055722 255 2147483647 2147483647 2 0 0 1 1 0 0 0 0 0 0 0 1 528280977472 528280977467 1 274877907072 137438953504 0 0 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 0 0 0 0 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 1024 0 0 0 0 251904 1531104 32 1531104 796672 32 796672 796672 32 796672 0 0 0 32 61440 30720 114688 30720 9 6144 9 6144 1 0 0 0 0 0 0 0 0 0 0 47 48995328 16384 7168 25 25493504 16384 16384 25 25493504 16384 16384 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {49349120};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_6)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
    64,                                      // aivNum
    32,                                      // aicNum
    196608,                                   // ubSize
    524288,                                 // l1Size
    65536,                                   // l0aSize
    65536,                                // l0bSize
    131072,                                        // l0cSize
    33554432,                                        // l2CacheSize
    32,                                        // coreNum
    platform_ascendc::SocVersion::ASCEND950  // socVersion
};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{3, 128 ,128}, {3, 128 ,128}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{3, 1, 128, 8}, {3, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{3, 1, 128, 8}, {3, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        // dk
         {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        // dv
         {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        // dq_rope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        // dk_rope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   
         },
        {
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140299490594850;
    std::string expectTilingData = "32 3 1 1 128 121 128 128 4575657222443697395 255 2147483647 2147483647 2 0 0 1 1 0 0 0 0 0 0 0 1 549755813952 549755813952 1 519691042944 4294967299 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 15488 16384 3 16384 15488 3 15488 15488 3 15488 0 0 0 3 61440 30720 114688 30720 1 15488 1 15488 1 0 0 0 0 0 0 0 0 0 0 1 49152 16384 16384 1 46464 16384 13696 1 46464 16384 13696 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {25756160};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_7)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
    64,                                      // aivNum
    32,                                      // aicNum
    196608,                                   // ubSize
    524288,                                 // l1Size
    65536,                                   // l0aSize
    65536,                                // l0bSize
    131072,                                        // l0cSize
    33554432,                                        // l2CacheSize
    32,                                        // coreNum
    platform_ascendc::SocVersion::ASCEND950  // socVersion
};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{1, 4096, 1, 200}, {1, 4096, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{1, 4096, 1, 200}, {1, 4096, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 73, 8}, {1, 1, 73, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 73, 8}, {1, 1, 73, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
        // dk
         {{{1, 4096, 1, 200}, {1, 4096, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
        // dv
         {{{1, 4096, 1, 200}, {1, 4096, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        // dq_rope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        // dk_rope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   
         },
        {
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07071067811865475f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.8f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140300564337952;
    std::string expectTilingData = "32 1 1 1 73 4096 200 200 4561245704492732611 204 2147483647 0 2 0 0 1 1 0 0 8589934594 8796093022209 72339069014638592 0 0 1 313532612672 313532612617 32 549755814016 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 299008 7300 2 7300 409600 2 409600 409600 2 409600 0 0 0 2 61440 30720 114688 30720 10 22528 0 30720 1099511627777 244091581366274 15255723892224 111 1 1 37 36 37 36 73 1 14600 16384 14600 25 819200 16384 16384 25 819200 16384 16384 65536 168448 3445760 0 6723072 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {27699200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_8)
{
    int64_t actual_seq_qlist[3] = {13, 14, 81};
    int64_t actual_seq_kvlist[3] = {13, 14, 81};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
    64,                                      // aivNum
    32,                                      // aicNum
    196608,                                   // ubSize
    524288,                                 // l1Size
    65536,                                   // l0aSize
    65536,                                // l0bSize
    131072,                                        // l0cSize
    33554432,                                        // l2CacheSize
    32,                                        // coreNum
    platform_ascendc::SocVersion::ASCEND950  // socVersion
};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // atten_mask
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // prefix
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dk
         {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dv
         {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dq_rope
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dk_rope
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   
         },
        {
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0625f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577350610917264;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_9)
{
    int64_t actual_seq_qlist[3] = {13, 14, 81};
    int64_t actual_seq_kvlist[3] = {13, 14, 81};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
    64,                                      // aivNum
    32,                                      // aicNum
    196608,                                   // ubSize
    524288,                                 // l1Size
    65536,                                   // l0aSize
    65536,                                // l0bSize
    131072,                                        // l0cSize
    33554432,                                        // l2CacheSize
    32,                                        // coreNum
    platform_ascendc::SocVersion::ASCEND950  // socVersion
};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dy
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // pse_shift
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // attention_in
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dk
         {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dv
         {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dq_rope
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dk_rope
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   
         },
        {
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(70)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703249444018098;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {25756160};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_10)
{
    int64_t actual_seq_qlist[3] = {13, 14, 81};
    int64_t actual_seq_kvlist[3] = {13, 14, 81};
    int64_t prefix_list[3] = {10, 12, 60};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
    64,                                      // aivNum
    32,                                      // aicNum
    196608,                                   // ubSize
    524288,                                 // l1Size
    65536,                                   // l0aSize
    65536,                                // l0bSize
    131072,                                        // l0cSize
    33554432,                                        // l2CacheSize
    32,                                        // coreNum
    platform_ascendc::SocVersion::ASCEND950  // socVersion
};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // atten_mask
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // prefix
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dk
         {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dv
         {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dq_rope
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dk_rope
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   
         },
        {
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0625f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577350610917264;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_attention_in_shape)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[11] = {{{1, 64, 128}, {1, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_softmax_max_dim)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[8] = {{{1, 128, 8}, {1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_softmax_sum_shape)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[9] = {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_softmax_dtype)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[8] = {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_dscale_for_non_quant)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[17] = {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_dropmask_without_dropout)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[5] = {{{2048}, {2048}}, ge::DT_UINT8, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_dropmask_dtype)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[5] = {{{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs("BSH", 0, 0.9f), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_dropmask_size)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[5] = {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs("BSH", 0, 0.9f), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_atten_mask_dim)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[7] = {{{1, 2, 3}, {1, 2, 3}}, ge::DT_UINT8, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_atten_mask_dtype)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[7] = {{{128, 128}, {128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_sparse_mode)
{
    auto compileInfo = MakeA5CompileInfo();
    ExpectTilingFailed(MakeA5Context(MakeBshInputs(), MakeBshOutputs(), MakeAttrs("BSH", 7), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_pse_type)
{
    auto compileInfo = MakeA5CompileInfo();
    ExpectTilingFailed(MakeA5Context(MakeBshInputs(), MakeBshOutputs(), MakeAttrs("BSH", 0, 1.0f, 65536, 65536, 1, 4),
                                    &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_pse_dtype)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[4] = {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_pse_dim)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[4] = {{{1, 1, 128}, {1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_rope_missing_pair)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[22] = {{{1, 128, 64}, {1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_rope_d)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[22] = {{{1, 128, 64}, {1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    inputs[23] = {{{1, 128, 32}, {1, 128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeBshOutputs(), MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_sink_dtype)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    auto outputs = MakeBshOutputs();
    inputs.push_back({{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    outputs.push_back({{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND});
    ExpectTilingFailed(MakeA5Context(inputs, outputs, MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_sink_query_dtype)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs(ge::DT_FLOAT);
    auto outputs = MakeBshOutputs(ge::DT_FLOAT);
    inputs.push_back({{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND});
    outputs.push_back({{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND});
    ExpectTilingFailed(MakeA5Context(inputs, outputs, MakeAttrs(), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_tnd_missing_actual_seq)
{
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeBshInputs();
    inputs[0] = {{{32, 2, 64}, {32, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    inputs[1] = {{{32, 2, 64}, {32, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    inputs[2] = {{{32, 2, 64}, {32, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    inputs[3] = {{{32, 2, 64}, {32, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    inputs[11] = {{{32, 2, 64}, {32, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeTndOutputs(32, 32), MakeAttrs("TND", 0, 1.0f, 65536, 65536, 2),
                                    &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_tnd_actual_seq_size)
{
    int64_t actualSeqQ[2] = {16, 40};
    int64_t actualSeqKv[3] = {16, 40, 64};
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeTndInputs(actualSeqQ, actualSeqKv, 2, 40, 64);
    inputs[14] = {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKv};
    ExpectTilingFailed(MakeA5Context(inputs, MakeTndOutputs(40, 64), MakeAttrs("TND", 0, 1.0f, 65536, 65536, 2),
                                    &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_tnd_eod)
{
    int64_t actualSeqQ[3] = {16, 0, 20};
    int64_t actualSeqKv[3] = {16, 0, 20};
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeTndInputs(actualSeqQ, actualSeqKv, 3, 20, 20);
    ExpectTilingFailed(MakeA5Context(inputs, MakeTndOutputs(20, 20), MakeAttrs("TND", 0, 1.0f, 65536, 65536, 2),
                                    &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_tnd_sparse_mode_prefix)
{
    int64_t actualSeqQ[2] = {16, 40};
    int64_t actualSeqKv[2] = {16, 48};
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeTndInputs(actualSeqQ, actualSeqKv, 2, 40, 48);
    ExpectTilingFailed(MakeA5Context(inputs, MakeTndOutputs(40, 48), MakeAttrs("TND", 5, 1.0f, 65536, 65536, 2),
                                    &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_invalid_tnd_softmax_same_as_input_shape)
{
    int64_t actualSeqQ[2] = {16, 32};
    int64_t actualSeqKv[2] = {16, 32};
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeTndInputs(actualSeqQ, actualSeqKv, 2, 32, 32);
    inputs[8] = {{{32, 2, 8, 1}, {32, 2, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND};
    ExpectTilingFailed(MakeA5Context(inputs, MakeTndOutputs(32, 32),
                                    MakeAttrs("TND", 0, 1.0f, 65536, 65536, 2, 1, "same_as_input"), &compileInfo));
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tnd_zero_seq_split_axis)
{
    int64_t actualSeqQ[2] = {0, 16};
    int64_t actualSeqKv[2] = {0, 16};
    auto compileInfo = MakeA5CompileInfo();
    auto inputs = MakeTndInputs(actualSeqQ, actualSeqKv, 2, 16, 16);
    ExpectTilingSuccess(MakeA5Context(inputs, MakeTndOutputs(16, 16), MakeAttrs("TND", 0, 1.0f, 65536, 65536, 2),
                                     &compileInfo));
}
