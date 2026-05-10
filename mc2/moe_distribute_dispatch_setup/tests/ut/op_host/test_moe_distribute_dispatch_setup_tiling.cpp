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
 * \file test_moe_distribute_dispatch_setup_tiling.cpp
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

const std::string OP_NAME = "MoeDistributeDispatchSetup";

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

struct MoeDistributeDispatchSetupTilingTestParam {
    uint64_t inputTotalNum;
    uint64_t outputTotalNum;
    string caseName;
    string socVersion;

    std::initializer_list<int64_t> x;
    std::initializer_list<int64_t> expert_ids;

    ge::DataType x_dtype;
    ge::DataType expert_ids_dtype;

    int64_t epWorldSize;
    int64_t epRankId;
    int64_t moeExpertNum;
    int64_t expertShardType;
    int64_t sharedExpertNum;
    int64_t sharedExpertRankNum;
    int64_t quantMode;
    int64_t globalBs;
    int64_t commType;
    uint64_t rankNum;
    
    std::initializer_list<int64_t> yOut;
    std::initializer_list<int64_t> expandIdxOut;
    std::initializer_list<int64_t> commCmdInfoOut;

    ge::DataType y_out_dtype;
    ge::DataType expand_idx_out_dtype;
    ge::DataType comm_cmd_info_out_dtype;

    ge::graphStatus expectResult;
    uint64_t expectTilingKey;
};

inline std::ostream& operator<<(std::ostream& os, const MoeDistributeDispatchSetupTilingTestParam& param)
{
    return os << param.caseName;
}

class TestMoeDistributeDispatchSetupTiling : public testing::TestWithParam<MoeDistributeDispatchSetupTilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TestMoeDistributeDispatchSetupTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TestMoeDistributeDispatchSetupTiling TearDown" << std::endl;
    }
};

static MoeDistributeDispatchSetupTilingTestParam test_cases[] = {
//===============================================典型shape====================================================
{2, 3, "critical_case_1", "3510", 
  {16, 4096},{16, 6},
  ge::DT_FLOAT16, ge::DT_INT32, 
  16, 0, 256, 0, 0, 0, 0, 0, 2, 8,
  {16 * (6 + 0), 4096},{16 * 6},{(16 * (6 + 0) + 16 * 16) * 16}, 
  ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, 
  ge::GRAPH_SUCCESS, 1000UL},

//===============================================quantMode边界值校验====================================================
// quantMode = -1 (下边界外，无效值，应返回失败)
{2, 3, "quantMode_boundary_invalid_minus1", "3510", 
  {16, 4096},{16, 6},
  ge::DT_FLOAT16, ge::DT_INT32, 
  16, 0, 256, 0, 0, 0, -1, 0, 2, 8,
  {16 * (6 + 0), 4096},{16 * 6},{(16 * (6 + 0) + 16 * 16) * 16}, 
  ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, 
  ge::GRAPH_FAILED, 0UL},

// quantMode = 0 (下边界，UNQUANT模式，应成功)
{2, 3, "quantMode_boundary_valid_0_unquant", "3510", 
  {16, 4096},{16, 6},
  ge::DT_FLOAT16, ge::DT_INT32, 
  16, 0, 256, 0, 0, 0, 0, 0, 2, 8,
  {16 * (6 + 0), 4096},{16 * 6},{(16 * (6 + 0) + 16 * 16) * 16}, 
  ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, 
  ge::GRAPH_SUCCESS, 1000UL},

// quantMode = 4 (上边界，MX_QUANT模式，应成功)
// yOut dim1 = CeilAlign(CeilAlign(4096, 256) + CeilAlign(4096/32, 2), 512) = CeilAlign(4096 + 128, 512) = 4608
{2, 3, "quantMode_boundary_valid_4_mx_quant", "3510", 
  {16, 4096},{16, 6},
  ge::DT_FLOAT16, ge::DT_INT32, 
  16, 0, 256, 0, 0, 0, 4, 0, 2, 8,
  {16 * (6 + 0), 4608},{16 * 6},{(16 * (6 + 0) + 16 * 16) * 16}, 
  ge::DT_FLOAT8_E4M3FN, ge::DT_INT32, ge::DT_INT32, 
  ge::GRAPH_SUCCESS, 1004UL},

// quantMode = 5 (上边界外，无效值，应返回失败)
{2, 3, "quantMode_boundary_invalid_5", "3510", 
  {16, 4096},{16, 6},
  ge::DT_FLOAT16, ge::DT_INT32, 
  16, 0, 256, 0, 0, 0, 5, 0, 2, 8,
  {16 * (6 + 0), 4096},{16 * 6},{(16 * (6 + 0) + 16 * 16) * 16}, 
  ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, 
  ge::GRAPH_FAILED, 0UL},
};

struct MoeDistributeDispatchSetupCompileInfo {} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MoeDistributeDispatchSetupTilingTestParam &param) 
{
    std::vector<pair<std::initializer_list<int64_t>, ge::DataType>> inputshapeDtypeList = {
    {param.x, param.x_dtype}, 
    {param.expert_ids, param.expert_ids_dtype}
    };
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc;
    for (int i = 0; i < param.inputTotalNum; i++) {
        inputTensorDesc.push_back({make_shape(inputshapeDtypeList[i].first), inputshapeDtypeList[i].second, ge::FORMAT_ND});
    }

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
        {"comm_type", build_from<int64_t>(param.commType)},
        {"comm_alg", build_from<std::string>("")},
    });

    std::vector<pair<std::initializer_list<int64_t>, ge::DataType>> outputshapeDtypeList = {
    {param.yOut, param.y_out_dtype}, 
    {param.expandIdxOut, param.expand_idx_out_dtype}, 
    {param.commCmdInfoOut, param.comm_cmd_info_out_dtype}
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc;
    for (int i = 0; i < param.outputTotalNum; i++) {
        outputTensorDesc.push_back({make_shape(outputshapeDtypeList[i].first), outputshapeDtypeList[i].second, ge::FORMAT_ND});
    }
    
    return gert::TilingContextPara(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo, param.socVersion);
}

static void ThreadFunc(const MoeDistributeDispatchSetupTilingTestParam *testCases, size_t testcase_num, size_t thread_idx, size_t thread_num) 
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        auto param = testCases[idx];
        auto tilingContextPara = BuildTilingContextPara(param);
        std::cout << "[TEST_CASE] " << param << std::endl;
        if (param.expectResult == ge::GRAPH_SUCCESS) 
        {
            ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey);
        }
        else {
            ExecuteTestCase(tilingContextPara);
        }
}
}

static void TestMultiThread(const MoeDistributeDispatchSetupTilingTestParam *testCases, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx){
        threads[idx] = std::thread(ThreadFunc, testCases, testcase_num, idx, thread_num);
    }
    for (size_t idx = 0; idx < thread_num; ++idx){
        threads[idx].join();
    }
}

TEST_P(TestMoeDistributeDispatchSetupTiling, general_cases) {
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rankNum}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey);
}

TEST_F(TestMoeDistributeDispatchSetupTiling, general_cases_multi_thread) {
    size_t thread_num = 3;
    TestMultiThread(test_cases, sizeof(test_cases) / sizeof(MoeDistributeDispatchSetupTilingTestParam), thread_num);
}

INSTANTIATE_TEST_CASE_P(MoeDistributeDispatchSetupTilingUT, TestMoeDistributeDispatchSetupTiling, testing::ValuesIn(test_cases));
} // namespace