#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FlashAttentionScoreProtoExtended : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScoreProtoExtended SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScoreProtoExtended TearDown" << std::endl;
    }
};

TEST_F(FlashAttentionScoreProtoExtended, infershape_bsh_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 1, 256, 8},
        {2, 1, 256, 8},
        {0, 0, 0, 0},
        {2, 256, 128},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProtoExtended, infershape_bsnd_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{2, 128, 4, 64}, {2, 128, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 4, 64}, {2, 256, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 4, 64}, {2, 256, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 256}, {128, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 128, 8},
        {2, 4, 128, 8},
        {0, 0, 0, 0},
        {2, 128, 4, 64},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProtoExtended, infershape_bnsd_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{2, 4, 128, 64}, {2, 4, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 256, 64}, {2, 4, 256, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 256, 64}, {2, 4, 256, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 256}, {128, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 128, 8},
        {2, 4, 128, 8},
        {0, 0, 0, 0},
        {2, 4, 128, 64},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProtoExtended, infershape_tnd_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
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
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {256, 4, 8},
        {256, 4, 8},
        {0, 0, 0, 0},
        {256, 4, 128},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProtoExtended, infershape_hifloat8_bsnd)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{1, 128, 8, 64}, {1, 128, 8, 64}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{1, 256, 8, 64}, {1, 256, 8, 64}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{1, 256, 8, 64}, {1, 256, 8, 64}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 256}, {128, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
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
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 8, 128, 1},
        {1, 8, 128, 1},
        {0, 0, 0, 0},
        {1, 128, 8, 64},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProtoExtended, infershape_case_insensitive_bsh)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("bsh")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 1, 256, 8},
        {2, 1, 256, 8},
        {0, 0, 0, 0},
        {2, 256, 128},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProtoExtended, infershape_invalid_layout)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("INVALID")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(FlashAttentionScoreProtoExtended, infershape_sbh_headnum_zero)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 0, 256, 8},
        {1, 0, 256, 8},
        {0, 0, 0, 0},
        {256, 1, 0},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProtoExtended, infershape_sbh_d1_zero)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{256, 1, 32}, {256, 1, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 32}, {256, 1, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 32}, {256, 1, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 64, 256, 8},
        {1, 64, 256, 8},
        {0, 0, 0, 0},
        {256, 1, 0},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProtoExtended, infershape_bsh_gqa)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScore",
        {
            {{{2, 256, 256}, {2, 256, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 128}, {2, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 64}, {2, 256, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0625f)},
            {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 256, 8},
        {2, 4, 256, 8},
        {0, 0, 0, 0},
        {2, 256, 128},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FlashAttentionScoreProtoExtended, inferdatatype_fp16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("FlashAttentionScore")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType dt_fp16 = ge::DT_FLOAT16;
    ge::DataType dt_int32 = ge::DT_INT32;
    ge::DataType dt_uint8 = ge::DT_UINT8;
    ge::DataType dt_int64 = ge::DT_INT64;

    auto context_holder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(17, 4)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&dt_fp16, &dt_fp16, &dt_fp16, &dt_fp16, &dt_fp16, &dt_int32, &dt_uint8,
                             &dt_int64, &dt_int64, &dt_int64, &dt_int64, &dt_int64,
                             &dt_fp16, &dt_fp16, &dt_fp16, &dt_fp16, &dt_fp16})
            .NodeAttrs({
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
                {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
            })
            .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT16);
}

TEST_F(FlashAttentionScoreProtoExtended, inferdatatype_bf16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("FlashAttentionScore")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType dt_bf16 = ge::DT_BF16;
    ge::DataType dt_int32 = ge::DT_INT32;
    ge::DataType dt_uint8 = ge::DT_UINT8;
    ge::DataType dt_int64 = ge::DT_INT64;

    auto context_holder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(17, 4)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&dt_bf16, &dt_bf16, &dt_bf16, &dt_bf16, &dt_bf16, &dt_int32, &dt_uint8,
                             &dt_int64, &dt_int64, &dt_int64, &dt_int64, &dt_int64,
                             &dt_bf16, &dt_bf16, &dt_bf16, &dt_bf16, &dt_bf16})
            .NodeAttrs({
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
                {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
            })
            .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_BF16);
}

TEST_F(FlashAttentionScoreProtoExtended, inferdatatype_fp8_e4m3fn)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("FlashAttentionScore")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType dt_fp8 = ge::DT_FLOAT8_E4M3FN;
    ge::DataType dt_int32 = ge::DT_INT32;
    ge::DataType dt_uint8 = ge::DT_UINT8;
    ge::DataType dt_int64 = ge::DT_INT64;

    auto context_holder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(17, 4)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&dt_fp8, &dt_fp8, &dt_fp8, &dt_fp8, &dt_fp8, &dt_int32, &dt_uint8,
                             &dt_int64, &dt_int64, &dt_int64, &dt_int64, &dt_int64,
                             &dt_fp8, &dt_fp8, &dt_fp8, &dt_fp8, &dt_fp8})
            .NodeAttrs({
                {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
                {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
            })
            .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_BF16);
}

TEST_F(FlashAttentionScoreProtoExtended, inferdatatype_hifloat8)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("FlashAttentionScore")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType dt_hifloat8 = ge::DT_HIFLOAT8;
    ge::DataType dt_int32 = ge::DT_INT32;
    ge::DataType dt_uint8 = ge::DT_UINT8;
    ge::DataType dt_int64 = ge::DT_INT64;

    auto context_holder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(17, 4)
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&dt_hifloat8, &dt_hifloat8, &dt_hifloat8, &dt_hifloat8, &dt_hifloat8, &dt_int32, &dt_uint8,
                             &dt_int64, &dt_int64, &dt_int64, &dt_int64, &dt_int64,
                             &dt_hifloat8, &dt_hifloat8, &dt_hifloat8, &dt_hifloat8, &dt_hifloat8})
            .NodeAttrs({
                {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
                {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}
            })
            .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_BF16);
}
