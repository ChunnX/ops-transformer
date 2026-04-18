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
#include <thread>
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
inline std::ostream& operator<<(std::ostream& os, const MoeDistributeDispatchTeardownTilingTestParam& param)
{
    return os << param.caseName;
}

// setup & teardown
class TestMoeDistributeTeardownTiling : public testing::TestWithParam<MoeDistributeDispatchTeardownTilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TestMoeDistributeTeardownTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TestMoeDistributeTeardownTiling TearDown" << std::endl;
    }
};

// 用例列表集
static MoeDistributeDispatchTeardownTilingTestParam test_cases[] = {
//===============================================典型shape====================================================
{4, 4, "critical_case_1", "3510", 
 {16, 4096},{16 * (6 + 0), 4096},{16, 6},{(16 * (6 + 0) + 16 * 16)* 16 }, 
 ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, 
 16, 0, 256, 0, 0, 0, 0, 0, 1, 2, 8,
 {1536, 4096},{1536},{1536 * 128},{16}, 
 ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, 
 ge::GRAPH_SUCCESS, 10000UL},

//===============================================quantMode边界值校验====================================================
// quantMode = -1 (下边界外，无效值，应返回失败)
{4, 4, "quantMode_boundary_invalid_minus1", "3510", 
 {16, 4096},{16 * (6 + 0), 4096},{16, 6},{(16 * (6 + 0) + 16 * 16)* 16 }, 
 ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, 
 16, 0, 256, 0, 0, 0, -1, 0, 1, 2, 8,
 {1536, 4096},{1536},{1536 * 128},{16}, 
 ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, 
 ge::GRAPH_FAILED, 0UL},

// quantMode = 0 (下边界，UNQUANT模式，应成功)
{4, 4, "quantMode_boundary_valid_0_unquant", "3510", 
 {16, 4096},{16 * (6 + 0), 4096},{16, 6},{(16 * (6 + 0) + 16 * 16)* 16 }, 
 ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, 
 16, 0, 256, 0, 0, 0, 0, 0, 1, 2, 8,
 {1536, 4096},{1536},{1536 * 128},{16}, 
 ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, 
 ge::GRAPH_SUCCESS, 10000UL},

// quantMode = 4 (上边界，MX_QUANT模式，应成功)
{4, 4, "quantMode_boundary_valid_4_mx_quant", "3510", 
 {16, 4096},{16 * (6 + 0), 4608},{16, 6},{(16 * (6 + 0) + 16 * 16)* 16 }, 
 ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, 
 16, 0, 256, 0, 0, 0, 4, 0, 1, 2, 8,
 {1536, 4096},{1536, 128},{1536 * 128},{16}, 
 ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, 
 ge::GRAPH_SUCCESS, 10004UL},

// quantMode = 5 (上边界外，无效值，应返回失败)
{4, 4, "quantMode_boundary_invalid_5", "3510", 
 {16, 4096},{16 * (6 + 0), 4096},{16, 6},{(16 * (6 + 0) + 16 * 16)* 16 }, 
 ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, 
 16, 0, 256, 0, 0, 0, 5, 0, 1, 2, 8,
 {1536, 4096},{1536},{1536 * 128},{16}, 
 ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, 
 ge::GRAPH_FAILED, 0UL},
};

struct MoeDistributeDispatchTeardownCompileInfo {} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MoeDistributeDispatchTeardownTilingTestParam &param) 
{
    // 存储用例中输入信息
    std::vector<pair<std::initializer_list<int64_t>, ge::DataType>> inputshapeDtypeList = {
    {param.x, param.x_dtype}, 
    {param.y, param.y_dtype}, 
    {param.expert_ids, param.expert_ids_dtype}, 
    {param.comm_cmd_info, param.comm_cmd_info_dtype}
    };
    // 构造输入信息
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc;
    for (int i = 0; i < param.inputTotalNum; i++) {
        inputTensorDesc.push_back({make_shape(inputshapeDtypeList[i].first), inputshapeDtypeList[i].second, ge::FORMAT_ND});
    }

    // 构造输入Attr信息
    std::vector<gert::TilingContextPara::OpAttr> attrs ({
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
    {param.expertTokenNumsOut, param.expert_token_nums_dtype}
    };
    // 构造输出信息
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc;
    for (int i = 0; i < param.outputTotalNum; i++) {
        outputTensorDesc.push_back({make_shape(outputshapeDtypeList[i].first), outputshapeDtypeList[i].second, ge::FORMAT_ND});
    }
    
    return gert::TilingContextPara(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo, param.socVersion);
}

// 多线程执行用例集
static void ThreadFunc(const MoeDistributeDispatchTeardownTilingTestParam *testCases, size_t testcase_num, size_t thread_idx, size_t thread_num) 
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        auto param = testCases[idx];
        auto tilingContextPara = BuildTilingContextPara(param);
        std::cout << "[TEST_CASE] " << param << std::endl;
        if (param.expectResult == ge::GRAPH_SUCCESS) 
        {
            // 正常用例分支
            ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey);
        }
        else {
            // 异常用例分支
            ExecuteTestCase(tilingContextPara);
        }
}
}

static void TestMultiThread(const MoeDistributeDispatchTeardownTilingTestParam *testCases, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx){
        threads[idx] = std::thread(ThreadFunc, testCases, testcase_num, idx, thread_num);
    }
    for (size_t idx = 0; idx < thread_num; ++idx){
        threads[idx].join();
    }
}

TEST_P(TestMoeDistributeTeardownTiling, general_cases) {
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rankNum}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey);
}

TEST_F(TestMoeDistributeTeardownTiling, general_cases_multi_thread) {
    size_t thread_num = 3;
    TestMultiThread(test_cases, sizeof(test_cases) / sizeof(MoeDistributeDispatchTeardownTilingTestParam), thread_num);
}

INSTANTIATE_TEST_CASE_P(MoeDistributeDispatchTeardownTilingUT, TestMoeDistributeTeardownTiling, testing::ValuesIn(test_cases));
} // namespace