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
 * \file test_moe_distribute_dispatch_teardown_tiling.cpp
 * \brief tiling ut
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "csv_case_load_utils.h"
#include "mc2_tiling_case_executor.h"

namespace {

using namespace std;
using namespace ge;
using namespace gert;

const std::string OP_NAME = "MoeDistributeDispatchTeardown";

template <typename T>
auto build_from(const T &value)
{
    return Ops::Transformer::AnyValue::CreateFrom<T>(value);
}

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) {
        return gert::StorageShape{};
    }
    return gert::StorageShape{input_shape, input_shape};
}

// 定义用例信息结构体
struct MoeDistributeDispatchTeardownTilingTestParam {
    uint64_t inputTotalNum;
    uint64_t outputTotalNum;
    string caseName;
    string socVersion;

    // 输入信息shape
    std::initializer_list<int64_t> x;
    std::initializer_list<int64_t> y;
    std::initializer_list<int64_t> expert_ids;
    std::initializer_list<int64_t> comm_cmd_info;

    // 输入信息dtype
    ge::DataType x_dtype;
    ge::DataType y_dtype;
    ge::DataType expert_ids_dtype;
    ge::DataType comm_cmd_info_dtype;

    // 输入Attrs
    int64_t epWorldSize;
    int64_t epRankId;
    int64_t moeExpertNum;
    int64_t expertShardType;
    int64_t sharedExpertNum;
    int64_t sharedExpertRankNum;
    int64_t quantMode;
    int64_t globalBs;
    int64_t expertTokenNumsType;
    int64_t commType;
    uint64_t rankNum;

    // 输出信息shape
    std::initializer_list<int64_t> expandXOut;
    std::initializer_list<int64_t> dynamicScalesOut;
    std::initializer_list<int64_t> assistInfoForCombineOut;
    std::initializer_list<int64_t> expertTokenNumsOut;

    // 输出信息dtype
    ge::DataType expand_x_dtype;
    ge::DataType dynamic_scales_dtype;
    ge::DataType assist_info_for_combine_dtype;
    ge::DataType expert_token_nums_dtype;

    // 预期测试结果
    ge::graphStatus expectResult;
    uint64_t expectTilingKey;
};

// 重写 << 操作符，规避用例执行失败时打印入参乱码的问题
inline std::ostream &operator<<(std::ostream &os, const MoeDistributeDispatchTeardownTilingTestParam &param)
{
    return os << param.caseName;
}

// setup & teardown
class TestMoeDistributeTeardownTiling : public testing::TestWithParam<MoeDistributeDispatchTeardownTilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        setenv("HCCL_BUFFSIZE", "6000", 1);
        std::cout << "TestMoeDistributeTeardownTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        unsetenv("HCCL_BUFFSIZE");
        std::cout << "TestMoeDistributeTeardownTiling TearDown" << std::endl;
    }
};

// 用例列表集
static MoeDistributeDispatchTeardownTilingTestParam test_cases[] = {
    //===============================================典型shape====================================================
    {4,
     4,
     "critical_case_1_bs16_h4096_k6_ep8_quant0",
     "3510",
     {16, 4096},
     {16 * (6 + 0), 4096},
     {16, 6},
     {(16 * 6 + 8 * 4) * 16},
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     ge::DT_INT32,
     ge::DT_INT32,
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     1,
     2,
     8,
     {512, 4096},
     {512},
     {512 * 128},
     {4},
     ge::DT_FLOAT16,
     ge::DT_FLOAT,
     ge::DT_INT32,
     ge::DT_INT64,
     ge::GRAPH_SUCCESS,
     10000UL},
    {4,
     4,
     "critical_case_2_bs8_h4096_k6_ep8_quant0",
     "3510",
     {8, 4096},
     {8 * (6 + 0), 4096},
     {8, 6},
     {(8 * 6 + 8 * 4) * 16},
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     ge::DT_INT32,
     ge::DT_INT32,
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     1,
     2,
     8,
     {256, 4096},
     {256},
     {256 * 128},
     {4},
     ge::DT_FLOAT16,
     ge::DT_FLOAT,
     ge::DT_INT32,
     ge::DT_INT64,
     ge::GRAPH_SUCCESS,
     10000UL},
    {4,
     4,
     "critical_case_3_bs256_h4096_k6_ep8_quant0",
     "3510",
     {256, 4096},
     {256 * (6 + 0), 4096},
     {256, 6},
     {(256 * 6 + 8 * 4) * 16},
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     ge::DT_INT32,
     ge::DT_INT32,
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     1,
     2,
     8,
     {8192, 4096},
     {8192},
     {8192 * 128},
     {4},
     ge::DT_FLOAT16,
     ge::DT_FLOAT,
     ge::DT_INT32,
     ge::DT_INT64,
     ge::GRAPH_SUCCESS,
     10000UL},
    {4,
     4,
     "critical_case_4_bs16_h7168_k6_ep8_quant0",
     "3510",
     {16, 7168},
     {16 * (6 + 0), 7168},
     {16, 6},
     {(16 * 6 + 8 * 4) * 16},
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     ge::DT_INT32,
     ge::DT_INT32,
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     1,
     2,
     8,
     {512, 7168},
     {512},
     {512 * 128},
     {4},
     ge::DT_FLOAT16,
     ge::DT_FLOAT,
     ge::DT_INT32,
     ge::DT_INT64,
     ge::GRAPH_SUCCESS,
     10000UL},
    {4,
     4,
     "critical_case_5_bs16_h4096_k8_ep8_quant0",
     "3510",
     {16, 4096},
     {16 * (8 + 0), 4096},
     {16, 8},
     {(16 * 8 + 8 * 4) * 16},
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     ge::DT_INT32,
     ge::DT_INT32,
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     1,
     2,
     8,
     {512, 4096},
     {512},
     {512 * 128},
     {4},
     ge::DT_FLOAT16,
     ge::DT_FLOAT,
     ge::DT_INT32,
     ge::DT_INT64,
     ge::GRAPH_SUCCESS,
     10000UL},
    {4,
     4,
     "critical_case_6_bs16_h4096_k6_ep2_quant0",
     "3510",
     {16, 4096},
     {16 * (6 + 0), 4096},
     {16, 6},
     {(16 * 6 + 2 * 16) * 16},
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     ge::DT_INT32,
     ge::DT_INT32,
     2,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     1,
     2,
     2,
     {192, 4096},
     {192},
     {192 * 128},
     {16},
     ge::DT_FLOAT16,
     ge::DT_FLOAT,
     ge::DT_INT32,
     ge::DT_INT64,
     ge::GRAPH_SUCCESS,
     10000UL},
    {4,
     4,
     "critical_case_7_bs16_h4096_k6_ep8_quant4_mx_quant",
     "3510",
     {16, 4096},
     {16 * (6 + 0), ((4096 + 256 - 1) / 256 * 256 + ((4096 + 32 - 1) / 32 + 2 - 1) / 2 * 2 + 512 - 1) / 512 * 512},
     {16, 6},
     {(16 * 6 + 8 * 4) * 16},
     ge::DT_FLOAT16,
     ge::DT_INT8,
     ge::DT_INT32,
     ge::DT_INT32,
     8,
     0,
     32,
     0,
     0,
     0,
     4,
     0,
     1,
     2,
     8,
     {512, 4096},
     {512, ((4096 + 32 - 1) / 32 + 2 - 1) / 2 * 2},
     {512 * 128},
     {4},
     ge::DT_INT8,
     ge::DT_FLOAT,
     ge::DT_INT32,
     ge::DT_INT64,
     ge::GRAPH_SUCCESS,
     10004UL},
};

struct MoeDistributeDispatchTeardownCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MoeDistributeDispatchTeardownTilingTestParam &param)
{
    // 存储用例中输入信息
    std::vector<pair<std::initializer_list<int64_t>, ge::DataType>> inputshapeDtypeList = {
        {param.x, param.x_dtype},
        {param.y, param.y_dtype},
        {param.expert_ids, param.expert_ids_dtype},
        {param.comm_cmd_info, param.comm_cmd_info_dtype}};
    // 构造输入信息
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc;
    for (int i = 0; i < param.inputTotalNum; i++) {
        inputTensorDesc.push_back(
            {make_shape(inputshapeDtypeList[i].first), inputshapeDtypeList[i].second, ge::FORMAT_ND});
    }

    // 构造输入Attr信息
    std::vector<gert::TilingContextPara::OpAttr> attrs({
        {"group_ep", build_from<std::string>("ep_group")},
        {"ep_world_size", build_from<int64_t>(param.epWorldSize)},
        {"ep_rank_id", build_from<int64_t>(param.epRankId)},
        {"moe_expert_num", build_from<int64_t>(param.moeExpertNum)},
        {"expert_shard_type", build_from<int64_t>(param.expertShardType)},
        {"shared_expert_num", build_from<int64_t>(param.sharedExpertNum)},
        {"shared_expert_rank_num", build_from<int64_t>(param.sharedExpertRankNum)},
        {"quant_mode", build_from<int64_t>(param.quantMode)},
        {"global_bs", build_from<int64_t>(param.globalBs)},
        {"expert_token_nums_type", build_from<int64_t>(param.expertTokenNumsType)},
        {"comm_type", build_from<int64_t>(param.commType)},
        {"comm_alg", build_from<std::string>("")},
    });

    // 存储用例中输出信息
    std::vector<pair<std::initializer_list<int64_t>, ge::DataType>> outputshapeDtypeList = {
        {param.expandXOut, param.expand_x_dtype},
        {param.dynamicScalesOut, param.dynamic_scales_dtype},
        {param.assistInfoForCombineOut, param.assist_info_for_combine_dtype},
        {param.expertTokenNumsOut, param.expert_token_nums_dtype}};
    // 构造输出信息
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc;
    for (int i = 0; i < param.outputTotalNum; i++) {
        outputTensorDesc.push_back(
            {make_shape(outputshapeDtypeList[i].first), outputshapeDtypeList[i].second, ge::FORMAT_ND});
    }

    return gert::TilingContextPara(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo, param.socVersion);
}

TEST_P(TestMoeDistributeTeardownTiling, general_cases)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rankNum},
                                               {"cclBufferSize", 6000ULL * 1024ULL * 1024ULL}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey);
}

INSTANTIATE_TEST_CASE_P(MoeDistributeDispatchTeardownTilingUT, TestMoeDistributeTeardownTiling,
                        testing::ValuesIn(test_cases));
} // namespace