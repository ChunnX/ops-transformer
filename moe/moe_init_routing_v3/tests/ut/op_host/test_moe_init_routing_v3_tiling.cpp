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
#include "../../../op_host/moe_init_routing_v3_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

constexpr int64_t QUANT_MODE_NONE = -1;
constexpr int64_t QUANT_MODE_STATIC = 0;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
constexpr int64_t ROW_IDX_TYPE_DROPPAD = 0;
constexpr int64_t ROW_IDX_TYPE_DROPLESS = 1;
constexpr int64_t EXPERT_NUM = 256;

class MoeInitRoutingV3Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV3Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV3Tiling TearDown" << std::endl;
    }
};

void RunNormalCase(int64_t N, int64_t H, int64_t K, int64_t C, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCase1(int64_t N, int64_t H, int64_t K, int64_t C, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseNoQuant(int64_t N, int64_t H, int64_t K, int64_t C, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{N}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseKeyValue(int64_t N, int64_t H, int64_t K, int64_t C, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{expert_num, 2}, {expert_num, 2}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseFullload(int64_t N, int64_t H, int64_t K, int64_t C, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E, H}, {E, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{expert_num, 2}, {expert_num, 2}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核 + not quant + scale not None   1000000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_01)
{   
    string expectTilingData = "40 1 83 27 180 192 12 -1 0 0 0 256 1 1 0 1 27 0 0 1 27 1 27 27 27 1 27 27 1984 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 单核 + not quant + drop pad + scale None 1001000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_02)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 1 0 0 256 1 1 0 1 27 0 0 1 27 1 27 27 27 1 27 27 1984 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + scale None 1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_03)
{
    string expectTilingData = "40 160 96 1450 180 192 12 -1 0 0 0 256 1 1 0 0 232000 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23275856};
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_INT8, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1100000, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + drop pad + scale None 1101000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_04)
{
    string expectTilingData = "40 160 96 1450 180 192 12 -1 1 0 0 256 1 1 0 0 232000 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23275856};
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1101000, expectTilingData, expectWorkspaces);
}


// 单核 + dynamci quant + scale not None   1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_05)
{
    string expectTilingData = "40 1 83 27 180 192 12 1 0 0 0 256 1 1 0 0 27 0 0 1 27 1 27 27 27 1 27 27 1984 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16793892};
    RunNormalCase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1020000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + scale None 1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_06)
{
    string expectTilingData = "40 1 83 27 180 192 12 1 0 0 0 256 1 1 0 0 27 0 0 1 27 1 27 27 27 1 27 27 1984 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16793892};
    RunNormalCase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1020000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + drop mode + scale not None  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_07)
{
    string expectTilingData = "40 8 60 32 0 100 100 1 1 0 0 256 1 1 0 0 256 0 0 1 256 1 256 256 256 1 256 256 1984 0 1024 37 7 4 1 7 7 1 4 4 37 7 4 1 7 7 1 4 4 1 60 60 0 ";
    std::vector<size_t> expectWorkspaces = {16796976};
    RunNormalCase(8, 60, 32, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1021000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + drop mode + scale None  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_08)
{
    string expectTilingData = "40 8 60 32 0 100 100 1 1 0 0 256 1 1 0 0 256 0 0 1 256 1 256 256 256 1 256 256 1984 0 1024 37 7 4 1 7 7 1 4 4 37 7 4 1 7 7 1 4 4 1 60 60 0 ";
    std::vector<size_t> expectWorkspaces = {16796976};
    RunNormalCase(8, 60, 32, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1021000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + scale not None  11020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_09)
{
    string expectTilingData = "40 160 96 1450 180 192 12 1 0 0 0 256 1 1 0 0 232000 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291216};
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1120000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + scale None 11020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_10)
{
    string expectTilingData = "40 160 96 1450 180 192 12 1 0 0 0 256 1 1 0 0 232000 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291216};
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1120000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale not None  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_11)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 1 0 0 232000 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale None  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_12)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 1 0 0 232000 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale None + bfloat16  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_13)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 1 0 0 232000 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_BF16, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale None + float16 11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_14)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 1 0 0 232000 0 0 40 5824 3 1984 1856 4864 3 1632 1600 1984 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 2 3966 1834 2 3966 1834 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT16, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 单核 + static quant + drop mode + scale not None   1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_15)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 0 0 0 256 1 1 27 1 27 27 27 1 27 27 1984 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 ";
    std::vector<size_t> expectWorkspaces = {5329576};
    RunNormalCase1(1, 83, 27, 0, 0, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_FAILED, 1020000, expectTilingData, expectWorkspaces);
}

// full load
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_16)
{
    string expectTilingData = "40 1 7168 8 0 256 256 1 1 1 0 256 2 1 0 0 8 0 2 1 8 1 8 8 8 1 8 8 2016 0 1024 8 1 1 1 1 1 1 1 1 8 1 1 1 1 1 1 1 1 4 1792 1792 0 ";
    std::vector<size_t> expectWorkspaces = {17010432};
    RunNormalCaseFullload(1, 7168, 8, 0, 0, 2, true, QUANT_MODE_DYNAMIC, 1, ge::DT_BF16, {0, 256}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2000000, expectTilingData, expectWorkspaces);
}

// performance 单核gather
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_17)
{
    string expectTilingData = "40 1920 7168 8 0 8 8 -1 1 0 0 256 1 1 0 0 15360 0 0 16 960 1 960 960 960 1 960 960 1984 4 1024 40 384 384 1 384 384 1 384 384 40 384 384 1 384 384 1 384 384 1 7168 7168 0 ";
    std::vector<size_t> expectWorkspaces = {17209920};
    RunNormalCase(1920, 7168, 8, 0, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 8}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1201000, expectTilingData, expectWorkspaces);
}

// performance 多核gather
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_18)
{
    string expectTilingData = "40 4608 7168 8 0 8 8 -1 1 0 0 256 1 1 0 0 36864 0 0 40 928 1 928 928 672 1 672 672 1984 10 1024 40 922 906 1 922 922 1 906 906 40 922 906 1 922 922 1 906 906 1 7168 7168 0 ";
    std::vector<size_t> expectWorkspaces = {17812032};
    RunNormalCase(4608, 7168, 8, 0, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 8}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1301000, expectTilingData, expectWorkspaces);
}

// 多核排序
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_19)
{
    string expectTilingData = "40 4608 7168 10 0 8 8 -1 1 0 0 256 1 1 0 0 46080 0 0 40 1152 1 1152 1152 1152 1 1152 1152 1984 10 1024 40 1152 1152 1 1152 1152 1 1152 1152 40 1152 1152 2 1016 136 2 1016 136 1 7168 7168 0 ";
    std::vector<size_t> expectWorkspaces = {18070080};
    RunNormalCase(4608, 7168, 10, 0, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 8}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1101000, expectTilingData, expectWorkspaces);
}

// v3收编v2新增UT
// 非量化 fullload + scale None + count
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_20)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 0 0 0 256 1 1 0 1 27 0 0 1 27 1 27 27 27 1 27 27 1984 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 非量化 fullload + scale not None + count
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_21)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 0 1 0 256 1 1 0 1 27 0 0 1 27 1 27 27 27 1 27 27 1984 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCaseNoQuant(1, 83, 27, 0, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 非量化 fullload + scale None + key_value
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_22)
{
    string expectTilingData = "40 1 83 27 10 22 12 -1 0 0 0 256 2 1 0 1 27 0 0 1 27 1 27 27 27 1 27 27 1984 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCaseKeyValue(1, 83, 27, 0, 0, 2, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {10, 22}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 非量化 fullload + scale None + GatherFirst
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_23)
{
    string expectTilingData = "40 100 2880 4 180 192 12 -1 0 0 0 256 1 1 1 1 400 0 0 1 400 1 400 400 400 1 400 400 1984 0 1024 40 10 10 1 10 10 1 10 10 40 10 10 1 10 10 1 10 10 1 2880 2880 0 ";
    std::vector<size_t> expectWorkspaces = {16791056};
    RunNormalCase(100, 2880, 4, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

// 非量化 fullload + scale None + scatter
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_24)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 1 0 0 256 1 1 0 1 27 0 0 1 27 1 27 27 27 1 27 27 1984 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2100000, expectTilingData, expectWorkspaces);
}

constexpr int64_t EXPERT_NUM_REGBASE = 200;

void RunNormalCaseRegbase(int64_t N, int64_t H, int64_t K, int64_t C, int64_t dropPadMode, int64_t countFlag,
                          bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType,
                          std::vector<int64_t> aciveExpertRange, int64_t rowIdxType, ge::graphStatus result,
                          int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 196608, platform_ascendc::SocVersion::ASCEND910_95};
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

void RunNormalCaseRegbase1(int64_t N, int64_t H, int64_t K, int64_t C, int64_t dropPadMode, int64_t countFlag,
                          bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType,
                          std::vector<int64_t> aciveExpertRange, int64_t rowIdxType, ge::graphStatus result,
                          int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces)
{
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 196608, platform_ascendc::SocVersion::ASCEND910_95};
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, ge::DT_INT8, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核 + not quant + scale not None   1000000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_01)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 0 0 0 256 1 0 0 0 0 0 0 1 27 1 27 27 27 1 27 27 6144 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCaseRegbase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                         ge::GRAPH_SUCCESS, 1000000, expectTilingData, expectWorkspaces);
}

// 单核 + not quant + drop pad + scale None 1001000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_02)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 1 0 0 256 1 0 0 0 0 0 0 1 27 1 27 27 27 1 27 27 6144 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    RunNormalCaseRegbase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                         ge::GRAPH_SUCCESS, 1001000, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + scale None 1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_03)
{
    string expectTilingData = "40 160 96 1450 180 192 12 -1 0 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23275856};
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_INT8, {180, 192},
                         ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS, 1100000, expectTilingData, expectWorkspaces);
}

// 多核 + not quant + drop pad + scale None 1101000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_04)
{
    string expectTilingData = "40 160 96 1450 180 192 12 -1 1 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23275856};
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_DROPLESS, ge::GRAPH_SUCCESS, 1101000, expectTilingData, expectWorkspaces);
}


// 单核 + dynamci quant + scale not None   1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_05)
{
    string expectTilingData = "40 1 83 27 180 192 12 1 0 0 0 256 1 0 0 0 0 0 0 1 27 1 27 27 27 1 27 27 6144 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16793892};
    RunNormalCaseRegbase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS, 1020000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + scale None 1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_06)
{
    string expectTilingData = "40 1 83 27 180 192 12 1 0 0 0 256 1 0 0 0 0 0 0 1 27 1 27 27 27 1 27 27 6144 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 0 ";
    std::vector<size_t> expectWorkspaces = {16793892};
    RunNormalCaseRegbase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS, 1020000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + drop mode + scale not None  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_07)
{
    string expectTilingData = "40 8 60 32 0 100 100 1 1 0 0 256 1 0 0 0 0 0 0 1 256 1 256 256 256 1 256 256 6144 0 1024 37 7 4 1 7 7 1 4 4 37 7 4 1 7 7 1 4 4 1 60 60 0 ";
    std::vector<size_t> expectWorkspaces = {16796976};
    RunNormalCaseRegbase(8, 60, 32, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                         ge::GRAPH_SUCCESS, 1021000, expectTilingData, expectWorkspaces);
}

// 单核 + dynamci quant + drop mode + scale None  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_08)
{
    string expectTilingData = "40 8 60 32 0 100 100 1 1 0 0 256 1 0 0 0 0 0 0 1 256 1 256 256 256 1 256 256 6144 0 1024 37 7 4 1 7 7 1 4 4 37 7 4 1 7 7 1 4 4 1 60 60 0 ";
    std::vector<size_t> expectWorkspaces = {16796976};
    RunNormalCaseRegbase(8, 60, 32, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                         ge::GRAPH_SUCCESS, 1021000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + scale not None  11020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_09)
{
    string expectTilingData = "40 160 96 1450 180 192 12 1 0 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291216};
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS, 1120000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + scale None 11020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_10)
{
    string expectTilingData = "40 160 96 1450 180 192 12 1 0 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291216};
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS, 1120000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale not None  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_11)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100},
                         ROW_IDX_TYPE_DROPLESS, ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale None  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_12)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 100},
                         ROW_IDX_TYPE_DROPLESS, ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale None + bfloat16  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_13)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_BF16, {0, 100},
                         ROW_IDX_TYPE_DROPLESS, ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 多核 + dynamci quant + drop mode + scale None + float16 11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_14)
{
    string expectTilingData = "40 160 96 1450 0 100 100 1 1 0 0 256 1 0 0 0 0 0 0 40 5824 1 5824 5824 4864 1 4864 4864 6144 10 1024 40 5800 5800 1 5800 5800 1 5800 5800 40 5800 5800 1 5800 5800 1 5800 5800 1 96 96 0 ";
    std::vector<size_t> expectWorkspaces = {23291568};
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT16, {0, 100},
                         ROW_IDX_TYPE_DROPLESS, ge::GRAPH_SUCCESS, 1121000, expectTilingData, expectWorkspaces);
}

// 单核 + static quant + drop mode + scale not None   1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_15)
{
    string expectTilingData = "40 1 83 27 180 192 12 -1 0 0 0 256 1 1 27 1 27 27 27 1 27 27 1984 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 ";
    std::vector<size_t> expectWorkspaces = {5329576};
    RunNormalCaseRegbase1(1, 83, 27, 0, 0, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_DROPLESS, ge::GRAPH_FAILED, 1020000, expectTilingData, expectWorkspaces);
}