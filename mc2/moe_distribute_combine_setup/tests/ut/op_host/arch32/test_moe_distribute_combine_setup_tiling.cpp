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
 * \brief 算子tiling UT for arch32异常测试
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace MoeDistributeCombineSetupArch32UT {

static const std::string OP_NAME = "MoeDistributeCombineSetup";

struct MoeDistributeCombineSetupTestParam {
    std::string caseName;
    std::initializer_list<int64_t> expandXShape;
    ge::DataType expandXDtype;
    ge::Format expandXFormat;
    std::initializer_list<int64_t> expertIdsShape;
    ge::DataType expertIdsDtype;
    ge::Format expertIdsFormat;
    std::initializer_list<int64_t> assistInfoShape;
    ge::DataType assistInfoDtype;
    ge::Format assistInfoFormat;
    std::initializer_list<int64_t> quantExpandXShape;
    ge::DataType quantExpandXDtype;
    ge::Format quantExpandXFormat;
    std::initializer_list<int64_t> commCmdInfoShape;
    ge::DataType commCmdInfoDtype;
    ge::Format commCmdInfoFormat;
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
    std::string socVersion;
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

static MoeDistributeCombineSetupTestParam g_testCases[] = {
    {"test_arch32_invalid_ep_world_size_too_small",
     {512, 4096},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {16, 6},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {65536},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {512, 6144},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {8320},
     ge::DT_INT32,
     ge::FORMAT_ND,
     "group_ep",
     1,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     "910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {16777216},
     0},
    {"test_arch32_invalid_ep_world_size_too_large",
     {512, 4096},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {16, 6},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {65536},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {512, 6144},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {8320},
     ge::DT_INT32,
     ge::FORMAT_ND,
     "group_ep",
     400,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     "910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {16777216},
     0},
    {"test_arch32_invalid_moe_expert_num_0",
     {512, 4096},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {16, 6},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {65536},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {512, 6144},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {8320},
     ge::DT_INT32,
     ge::FORMAT_ND,
     "group_ep",
     8,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     "910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {16777216},
     0},
    {"test_arch32_invalid_moe_expert_num_513",
     {512, 4096},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {16, 6},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {65536},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {512, 6144},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {8320},
     ge::DT_INT32,
     ge::FORMAT_ND,
     "group_ep",
     8,
     0,
     513,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     "910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {16777216},
     0},
    {"test_arch32_invalid_expert_shard_type_1",
     {512, 4096},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {16, 6},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {65536},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {512, 6144},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {8320},
     ge::DT_INT32,
     ge::FORMAT_ND,
     "group_ep",
     8,
     0,
     32,
     1,
     0,
     0,
     0,
     0,
     2,
     "",
     "910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {16777216},
     0},
    {"test_arch32_invalid_shared_expert_num_5",
     {512, 4096},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {16, 6},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {65536},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {512, 6144},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {8320},
     ge::DT_INT32,
     ge::FORMAT_ND,
     "group_ep",
     8,
     0,
     32,
     0,
     5,
     1,
     0,
     0,
     2,
     "",
     "910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {16777216},
     0},
    {"test_arch32_invalid_shared_expert_rank_num_too_large",
     {512, 4096},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {16, 6},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {65536},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {512, 6144},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {8320},
     ge::DT_INT32,
     ge::FORMAT_ND,
     "group_ep",
     8,
     0,
     32,
     0,
     0,
     5,
     0,
     0,
     2,
     "",
     "910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {16777216},
     0},
};

class MoeDistributeCombineSetupArch32TilingTest : public testing::TestWithParam<MoeDistributeCombineSetupTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineSetupArch32TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineSetupArch32TilingTest TearDown." << std::endl;
    }
};

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) {
        return gert::StorageShape{};
    }
    return gert::StorageShape{input_shape, input_shape};
}

static struct MoeDistributeCombineSetupCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MoeDistributeCombineSetupTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;

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

TEST_P(MoeDistributeCombineSetupArch32TilingTest, GeneralCases)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.epWorldSize}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.status, param.expectTilingKey,
                       param.expectTilingData, param.expectWorkspaces, param.mc2TilingDataReservedLen);
}

INSTANTIATE_TEST_CASE_P(MoeDistributeCombineSetupArch32TilingUT, MoeDistributeCombineSetupArch32TilingTest,
                        testing::ValuesIn(g_testCases));

} // namespace MoeDistributeCombineSetupArch32UT