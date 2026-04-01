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
 * \file test_moe_distribute_combine_setup_tiling.cpp
 * \brief 算子tiling UT
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace MoeDistributeCombineSetupUT {

static const std::string OP_NAME = "MoeDistributeCombineSetup";

struct MoeDistributeCombineSetupTestParam {
    std::string caseName;
    // expand_x
    std::initializer_list<int64_t> expandXShape;
    ge::DataType expandXDtype;
    ge::Format expandXFormat;

    // expert_ids
    std::initializer_list<int64_t> expertIdsShape;
    ge::DataType expertIdsDtype;
    ge::Format expertIdsFormat;

    // assist_info_for_combine
    std::initializer_list<int64_t> assistInfoShape;
    ge::DataType assistInfoDtype;
    ge::Format assistInfoFormat;

    // quant_expand_x
    std::initializer_list<int64_t> quantExpandXShape;
    ge::DataType quantExpandXDtype;
    ge::Format quantExpandXFormat;

    // comm_cmd_info
    std::initializer_list<int64_t> commCmdInfoShape;
    ge::DataType commCmdInfoDtype;
    ge::Format commCmdInfoFormat;

    // attrs
    std::string groupAttr;
    int64_t epWorldSize;
    int64_t epRankId;
    int64_t moeExpertNum;
    int64_t expertShardType;
    int64_t sharedExpertNum;
    int64_t sharedExpertRankNum;
    int64_t globalBs;
    int64_t commQuantMode;
    int64_t commType;
    std::string commAlgAttr;

    // soc version
    std::string socVersion;
    // expert result
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t mc2TilingDataReservedLen;
};

inline std::ostream &operator<<(std::ostream &os, const MoeDistributeCombineSetupTestParam &param)
{
    return os << param.caseName;
}

// 用例列表集
static MoeDistributeCombineSetupTestParam g_testCases[] = {
    // 正常用例
    {"test_aclnn_moe_distribute_combine_setup_normal_1",
     {192, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {16, 6}, ge::DT_INT32, ge::FORMAT_ND,
     {24576}, ge::DT_INT32, ge::FORMAT_ND,
     {192, 6144}, ge::DT_INT8, ge::FORMAT_ND,
     {3104}, ge::DT_INT32, ge::FORMAT_ND,
     "group_ep", 2, 0, 32, 0, 0, 0, 0, 0, 2, "",
     "3510", ge::GRAPH_SUCCESS, 0UL, "", {16777216}, 0},
    {"test_aclnn_moe_distribute_combine_setup_normal_2",
     {512, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
     {16, 6}, ge::DT_INT32, ge::FORMAT_ND,
     {65536}, ge::DT_INT32, ge::FORMAT_ND,
     {512, 6144}, ge::DT_INT8, ge::FORMAT_ND,
     {8320}, ge::DT_INT32, ge::FORMAT_ND,
     "group_ep", 8, 0, 32, 0, 0, 0, 128, 0, 2, "",
     "3510", ge::GRAPH_SUCCESS, 0UL, "", {16777216}, 0},
    {"test_aclnn_moe_distribute_combine_setup_normal_3",
     {192, 4096}, ge::DT_BF16, ge::FORMAT_ND,
     {16, 6}, ge::DT_INT32, ge::FORMAT_ND,
     {24576}, ge::DT_INT32, ge::FORMAT_ND,
     {192, 6144}, ge::DT_INT8, ge::FORMAT_ND,
     {3104}, ge::DT_INT32, ge::FORMAT_ND,
     "group_ep", 2, 0, 32, 0, 0, 0, 32, 0, 2, "",
     "3510", ge::GRAPH_SUCCESS, 0UL, "", {16777216}, 0},
    {"test_aclnn_moe_distribute_combine_setup_normal_4",
     {512, 4096}, ge::DT_BF16, ge::FORMAT_ND,
     {16, 6}, ge::DT_INT32, ge::FORMAT_ND,
     {65536}, ge::DT_INT32, ge::FORMAT_ND,
     {512, 6144}, ge::DT_INT8, ge::FORMAT_ND,
     {8320}, ge::DT_INT32, ge::FORMAT_ND,
     "group_ep", 8, 0, 32, 0, 0, 0, 0, 0, 2, "",
     "3510", ge::GRAPH_SUCCESS, 0UL, "", {16777216}, 0},
    // 异常用例
    {"test_aclnn_moe_distribute_combine_setup_invalid_expand_x_dtype",
     {512, 4096}, ge::DT_INT32, ge::FORMAT_ND,
     {16, 6}, ge::DT_INT32, ge::FORMAT_ND,
     {65536}, ge::DT_INT32, ge::FORMAT_ND,
     {512, 6144}, ge::DT_INT8, ge::FORMAT_ND,
     {8320}, ge::DT_INT32, ge::FORMAT_ND,
     "group_ep", 8, 0, 32, 0, 0, 0, 0, 0, 2, "",
     "3510", ge::GRAPH_SUCCESS, 0UL, "", {16777216}, 0},
};

// setup & teardown
class MoeDistributeCombineSetupArch35TilingTest : public testing::TestWithParam<MoeDistributeCombineSetupTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineSetupArch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineSetupArch35TilingTest TearDown." << std::endl;
    }
};

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) {
        return gert::StorageShape{};
    }
    return gert::StorageShape{input_shape, input_shape};
}

static struct MoeDistributeCombineSetupCompileInfo {} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MoeDistributeCombineSetupTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;

    // 参数封装
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{make_shape(param.expandXShape), param.expandXDtype, param.expandXFormat},
         {make_shape(param.expertIdsShape), param.expertIdsDtype, param.expertIdsFormat},
         {make_shape(param.assistInfoShape), param.assistInfoDtype, param.assistInfoFormat}});

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{make_shape(param.quantExpandXShape), param.quantExpandXDtype, param.quantExpandXFormat},
         {make_shape(param.commCmdInfoShape), param.commCmdInfoDtype, param.commCmdInfoFormat}});

    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.groupAttr)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSize)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epRankId)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.moeExpertNum)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.expertShardType)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sharedExpertNum)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sharedExpertRankNum)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.globalBs)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantMode)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commType)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commAlgAttr)}});

    return gert::TilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                   param.socVersion);
}

static void ThreadFunction(const MoeDistributeCombineSetupTestParam *testCases, size_t caseNum, size_t threadIdx,
                           size_t threadNum)
{
    for (size_t idx = threadIdx; idx < caseNum; idx += threadNum) {
        auto param = testCases[idx];
        auto tilingContextPara = BuildTilingContextPara(param);
        ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                        param.expectWorkspaces, param.mc2TilingDataReservedLen);
    }
}

static void TestExecMultiThread(const MoeDistributeCombineSetupTestParam *testCases, size_t testCaseNum,
                                size_t threadNum)
{
    std::thread threads[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunction, testCases, testCaseNum, idx, threadNum);
    }
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

TEST_P(MoeDistributeCombineSetupArch35TilingTest, GeneralCases)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.epWorldSize}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.status, param.expectTilingKey,
                       param.expectTilingData, param.expectWorkspaces, param.mc2TilingDataReservedLen);
}

TEST_F(MoeDistributeCombineSetupArch35TilingTest, GeneralCasesMultiThread)
{
    TestExecMultiThread(g_testCases, sizeof(g_testCases) / sizeof(MoeDistributeCombineSetupTestParam), 3);
}

INSTANTIATE_TEST_CASE_P(MoeDistributeCombineSetupTilingUT, MoeDistributeCombineSetupArch35TilingTest,
                        testing::ValuesIn(g_testCases));

} // namespace MoeDistributeCombineSetupUT
