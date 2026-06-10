#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/flash_attention_score_tiling_common.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class FlashAttentionScoreTilingExtended : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScoreTilingExtended SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScoreTilingExtended TearDown" << std::endl;
    }
};

TEST_F(FlashAttentionScoreTilingExtended, EmptyInput_QueryZero)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{0, 1, 128}, {0, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
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
         },
        {
         {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 1, 128}, {0, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}
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
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, "", {104857600});
}

TEST_F(FlashAttentionScoreTilingExtended, EmptyInput_KeyZero)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 1, 128}, {0, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
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
         },
        {
         {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}
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
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, "", {104857600});
}

TEST_F(FlashAttentionScoreTilingExtended, CheckParams_BSH_BatchMismatch)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3, 256, 128}, {3, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3, 256, 128}, {3, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         {{{2, 1, 256, 8}, {2, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 1, 256, 8}, {2, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(FlashAttentionScoreTilingExtended, CheckParams_SBH_BatchMismatch)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{256, 2, 128}, {256, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{256, 3, 128}, {256, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{256, 3, 128}, {256, 3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         {{{1, 2, 256, 8}, {1, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 2, 256, 8}, {1, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{256, 2, 128}, {256, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(FlashAttentionScoreTilingExtended, CheckParams_TND_DMismatch)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{256, 4, 64}, {256, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{256, 4, 64}, {256, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         {{{256, 4, 8}, {256, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{256, 4, 8}, {256, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(FlashAttentionScoreTilingExtended, CheckParams_BNSD_BatchMismatch)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{2, 8, 128, 64}, {2, 8, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3, 8, 128, 64}, {3, 8, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3, 8, 128, 64}, {3, 8, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         {{{2, 8, 128, 8}, {2, 8, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 8, 128, 8}, {2, 8, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 8, 128, 64}, {2, 8, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(FlashAttentionScoreTilingExtended, CheckParams_BNSD_DMismatch)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{2, 8, 128, 64}, {2, 8, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 8, 128, 128}, {2, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 8, 128, 128}, {2, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         {{{2, 8, 128, 8}, {2, 8, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 8, 128, 8}, {2, 8, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 8, 128, 64}, {2, 8, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(FlashAttentionScoreTilingExtended, CheckParams_BSH_KeyDLessThanValueD)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 256, 64}, {2, 256, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         {{{2, 1, 256, 8}, {2, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 1, 256, 8}, {2, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(FlashAttentionScoreTilingExtended, CheckParams_BNSD_KeyDLessThanValueD)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{2, 8, 128, 64}, {2, 8, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 8, 128, 64}, {2, 8, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 8, 128, 128}, {2, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         {{{2, 8, 128, 8}, {2, 8, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 8, 128, 8}, {2, 8, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 8, 128, 128}, {2, 8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(FlashAttentionScoreTilingExtended, CheckParams_InvalidLayoutLength)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSHND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(FlashAttentionScoreTilingExtended, GQA_BNSD)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{1, 8, 128, 64}, {1, 8, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 4, 128, 64}, {1, 4, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 4, 128, 64}, {1, 4, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 128}, {128, 128}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         {{{1, 8, 128, 8}, {1, 8, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 8, 128, 8}, {1, 8, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 8, 128, 64}, {1, 8, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 8, 128, 64}, {1, 8, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, "", {});
}

TEST_F(FlashAttentionScoreTilingExtended, GQA_BSND)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{1, 128, 8, 64}, {1, 128, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 4, 64}, {1, 128, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 4, 64}, {1, 128, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 128}, {128, 128}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         {{{1, 8, 128, 8}, {1, 8, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 8, 128, 8}, {1, 8, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 128, 8, 64}, {1, 128, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 8, 64}, {1, 128, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, "", {});
}

TEST_F(FlashAttentionScoreTilingExtended, LargeD512_BNSD)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{1, 1, 128, 512}, {1, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 128, 512}, {1, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 128, 512}, {1, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 128}, {128, 128}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 1, 128, 512}, {1, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 128, 512}, {1, 1, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, "", {});
}

TEST_F(FlashAttentionScoreTilingExtended, InnerPrecise2_BNSD)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 128}, {128, 128}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.8f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo, "Ascend950", 64, 262144, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, "", {});
}
