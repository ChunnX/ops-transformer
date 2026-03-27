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
 * \file test_moe_distribute_combine_teardown_tiling.cpp
 * \brief 算子tiling UT
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace {

using namespace std;
using namespace gert;
using namespace ge;

static const std::string OP_NAME = "MoeDistributeCombineTeardown";

struct MoeDistributeCombineTeardownTestParam {
    std::string caseName;
    std::string socVersion;

    // 输入Tensor
    uint64_t inputTotalNum;

    // expand_x
    std::initializer_list<int64_t> expandXShape;
    ge::DataType expandXDtype;
    ge::Format expandXFormat;

    // quant_expand_x
    std::initializer_list<int64_t> quantExpandXShape;
    ge::DataType quantExpandXDtype;
    ge::Format quantExpandXFormat;

    // expert_ids
    std::initializer_list<int64_t> expertIdsShape;
    ge::DataType expertIdsDtype;
    ge::Format expertIdsFormat;

    // expand_idx
    std::initializer_list<int64_t> expandIdxShape;
    ge::DataType expandIdxDtype;
    ge::Format expandIdxFormat;

    // expert_scales
    std::initializer_list<int64_t> expertScalesShape;
    ge::DataType expertScalesDtype;
    ge::Format expertScalesFormat;

    // comm_cmd_info
    std::initializer_list<int64_t> commCmdInfoShape;
    ge::DataType commCmdInfoDtype;
    ge::Format commCmdInfoFormat;

    // x_active_mask
    std::initializer_list<int64_t> xActiveMaskShape;
    ge::DataType xActiveMaskDtype;
    ge::Format xActiveMaskFormat;

    // shared_expert_x
    std::initializer_list<int64_t> sharedExpertXShape;
    ge::DataType sharedExpertXDtype;
    ge::Format sharedExpertXFormat;

    // 输出Tensor
    uint64_t outputTotalNum;

    // output_x
    std::initializer_list<int64_t> outputXShape;
    ge::DataType outputXDtype;
    ge::Format outputXFormat;

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

    // expert result
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t mc2TilingDataReservedLen;
};

inline std::ostream &operator<<(std::ostream &os, const MoeDistributeCombineTeardownTestParam &param)
{
    return os << param.caseName;
}

// 用例列表集
static MoeDistributeCombineTeardownTestParam test_cases[] = {
    // ===============================================交付shape===============================================
    {"moe_distribute_combine_teardown_critical_case_1", "Ascend910_95",
     6,
     {1536, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // expand_x
     {1536, 6144}, ge::DT_INT8, ge::FORMAT_ND,    // quant_expand_x
     {16, 6}, ge::DT_INT32, ge::FORMAT_ND,        // expert_ids
     {96}, ge::DT_INT32, ge::FORMAT_ND,           // expand_idx
     {16, 6}, ge::DT_FLOAT, ge::FORMAT_ND,        // expert_scales
     {24832}, ge::DT_INT32, ge::FORMAT_ND,        // comm_cmd_info
     {}, ge::DT_BOOL, ge::FORMAT_ND,              // x_active_mask
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,           // shared_expert_x
     1,
     {16, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,   // output_x
     "group_ep", 16, 0, 256, 0, 0, 0, 0, 0, 2, "", ge::GRAPH_SUCCESS, 1000UL,
     "16 0 1099511627776 16 25769803792 171798695936 0 262144 0 ", {33554432}, MC2_TILING_DATA_RESERVED_LEN},
    {"moe_distribute_combine_teardown_critical_case_2", "Ascend910_95",
     6,
     {1536, 4096}, ge::DT_BF16, ge::FORMAT_ND, // expand_x
     {1536, 6144}, ge::DT_INT8, ge::FORMAT_ND,    // quant_expand_x
     {16, 6}, ge::DT_INT32, ge::FORMAT_ND,        // expert_ids
     {96}, ge::DT_INT32, ge::FORMAT_ND,           // expand_idx
     {16, 6}, ge::DT_FLOAT, ge::FORMAT_ND,        // expert_scales
     {24832}, ge::DT_INT32, ge::FORMAT_ND,        // comm_cmd_info
     {}, ge::DT_BOOL, ge::FORMAT_ND,              // x_active_mask
     {}, ge::DT_BF16, ge::FORMAT_ND,           // shared_expert_x
     1,
     {16, 4096}, ge::DT_BF16, ge::FORMAT_ND,   // output_x
     "group_ep", 16, 0, 256, 0, 0, 0, 0, 0, 2, "", ge::GRAPH_SUCCESS, 1000UL,
     "16 0 1099511627776 16 25769803792 171798695936 0 262144 0 ", {33554432}, MC2_TILING_DATA_RESERVED_LEN},
    // ===============================================异常用例===============================================
    {"moe_distribute_combine_teardown_abuse_dim_case_1", "Ascend910_95",
     6,
     {1536, 4096, 0}, ge::DT_BF16, ge::FORMAT_ND, // expand_x
     {1536, 6144, 0}, ge::DT_INT8, ge::FORMAT_ND,    // quant_expand_x
     {16, 6, 0}, ge::DT_INT32, ge::FORMAT_ND,        // expert_ids
     {96, 0}, ge::DT_INT32, ge::FORMAT_ND,           // expand_idx
     {16, 6, 0}, ge::DT_FLOAT, ge::FORMAT_ND,        // expert_scales
     {24832, 0}, ge::DT_INT32, ge::FORMAT_ND,        // comm_cmd_info
     {}, ge::DT_BOOL, ge::FORMAT_ND,              // x_active_mask
     {1}, ge::DT_BF16, ge::FORMAT_ND,           // shared_expert_x
     1,
     {16, 4096, 0}, ge::DT_BF16, ge::FORMAT_ND,   // output_x
     "group_ep", 16, 0, 256, 0, 0, 0, 0, 0, 2, "",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"moe_distribute_combine_teardown_abuse_dim_case_2", "Ascend910_95",
     6,
     {1536}, ge::DT_BF16, ge::FORMAT_ND, // expand_x
     {1536}, ge::DT_INT8, ge::FORMAT_ND,    // quant_expand_x
     {16}, ge::DT_INT32, ge::FORMAT_ND,        // expert_ids
     {}, ge::DT_INT32, ge::FORMAT_ND,           // expand_idx
     {16}, ge::DT_FLOAT, ge::FORMAT_ND,        // expert_scales
     {}, ge::DT_INT32, ge::FORMAT_ND,        // comm_cmd_info
     {}, ge::DT_BOOL, ge::FORMAT_ND,              // x_active_mask
     {1, 2, 3}, ge::DT_BF16, ge::FORMAT_ND,           // shared_expert_x
     1,
     {16}, ge::DT_BF16, ge::FORMAT_ND,   // output_x
     "group_ep", 16, 0, 256, 0, 0, 0, 0, 0, 2, "",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"moe_distribute_combine_teardown_abuse_dtype_case_1", "Ascend910_95",
     6,
     {1536, 4096}, ge::DT_FLOAT, ge::FORMAT_ND, // expand_x
     {1536, 6144}, ge::DT_FLOAT, ge::FORMAT_ND,    // quant_expand_x
     {16, 6}, ge::DT_FLOAT, ge::FORMAT_ND,        // expert_ids
     {96}, ge::DT_FLOAT, ge::FORMAT_ND,           // expand_idx
     {16, 6}, ge::DT_FLOAT16, ge::FORMAT_ND,        // expert_scales
     {24832}, ge::DT_FLOAT, ge::FORMAT_ND,        // comm_cmd_info
     {}, ge::DT_FLOAT, ge::FORMAT_ND,              // x_active_mask
     {}, ge::DT_BOOL, ge::FORMAT_ND,           // shared_expert_x
     1,
     {16, 4096}, ge::DT_INT32, ge::FORMAT_ND,   // output_x
     "group_ep", 16, 0, 256, 0, 0, 0, 0, 0, 2, "",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"moe_distribute_combine_teardown_abuse_format_case_1", "Ascend910_95",
     6,
     {1536, 4096}, ge::DT_FLOAT16, ge::FORMAT_NCHW, // expand_x
     {1536, 6144}, ge::DT_INT8, ge::FORMAT_NCHW,    // quant_expand_x
     {16, 6}, ge::DT_INT32, ge::FORMAT_NCHW,        // expert_ids
     {96}, ge::DT_INT32, ge::FORMAT_NCHW,           // expand_idx
     {16, 6}, ge::DT_FLOAT, ge::FORMAT_NCHW,        // expert_scales
     {24832}, ge::DT_INT32, ge::FORMAT_NCHW,        // comm_cmd_info
     {}, ge::DT_BOOL, ge::FORMAT_NCHW,              // x_active_mask
     {}, ge::DT_FLOAT16, ge::FORMAT_NCHW,           // shared_expert_x
     1,
     {16, 4096}, ge::DT_FLOAT16, ge::FORMAT_NCHW,   // output_x
     "group_ep", 16, 0, 256, 0, 0, 0, 0, 0, 2, "",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"moe_distribute_combine_teardown_abuse_shape_case_1", "Ascend910_95",
     6,
     {1536, 1023}, ge::DT_FLOAT16, ge::FORMAT_ND, // expand_x (A, H)
     {1537, 1024}, ge::DT_INT8, ge::FORMAT_ND,    // quant_expand_x (A, tokenMsgSize)
     {0, 0}, ge::DT_INT32, ge::FORMAT_ND,        // expert_ids (Bs, K)
     {96}, ge::DT_INT32, ge::FORMAT_ND,           // expand_idx (Bs * K, )
     {16, 6}, ge::DT_FLOAT, ge::FORMAT_ND,        // expert_scales (Bs, K)
     {24832}, ge::DT_INT32, ge::FORMAT_ND,        // comm_cmd_info
     {}, ge::DT_BOOL, ge::FORMAT_ND,              // x_active_mask (Bs, )
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,           // shared_expert_x (Bs, H)
     1,
     {16, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,   // output_x (Bs, H)
     "", 0, 0, 0, 1, 5, 0, 0, 1, 1, "commAlg",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"moe_distribute_combine_teardown_abuse_shape_case_2", "Ascend910_95",
     6,
     {1536, 8197}, ge::DT_FLOAT16, ge::FORMAT_ND, // expand_x
     {1536, 6144}, ge::DT_INT8, ge::FORMAT_ND,    // quant_expand_x
     {513, 17}, ge::DT_INT32, ge::FORMAT_ND,        // expert_ids
     {96}, ge::DT_INT32, ge::FORMAT_ND,           // expand_idx
     {16, 6}, ge::DT_FLOAT, ge::FORMAT_ND,        // expert_scales
     {24832}, ge::DT_INT32, ge::FORMAT_ND,        // comm_cmd_info
     {}, ge::DT_BOOL, ge::FORMAT_ND,              // x_active_mask
     {}, ge::DT_FLOAT16, ge::FORMAT_ND,           // shared_expert_x
     1,
     {16, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,   // output_x
     "group_ep", 385, 0, 513, 0, 0, 0, 0, 3, 3, "",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"moe_distribute_combine_teardown_abuse_shape_case_3", "Ascend910_95",
     6,
     {1536, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // expand_x
     {1536, 6144}, ge::DT_INT8, ge::FORMAT_ND,    // quant_expand_x
     {16, 9}, ge::DT_INT32, ge::FORMAT_ND,        // expert_ids
     {96}, ge::DT_INT32, ge::FORMAT_ND,           // expand_idx
     {32, 6}, ge::DT_FLOAT, ge::FORMAT_ND,        // expert_scales
     {24832}, ge::DT_INT32, ge::FORMAT_ND,        // comm_cmd_info
     {64}, ge::DT_BOOL, ge::FORMAT_ND,              // x_active_mask
     {8, 2048}, ge::DT_FLOAT16, ge::FORMAT_ND,           // shared_expert_x
     1,
     {16, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND,   // output_x
     "group_ep", 17, 17, 8, 0, 0, 9, 1, 0, 2, "",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"moe_distribute_combine_teardown_abuse_shape_case_4", "Ascend910_95",
     6,
     {1536, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // expand_x
     {1536, 6144}, ge::DT_INT8, ge::FORMAT_ND,    // quant_expand_x
     {16, 6}, ge::DT_INT32, ge::FORMAT_ND,        // expert_ids
     {1}, ge::DT_INT32, ge::FORMAT_ND,           // expand_idx
     {16, 7}, ge::DT_FLOAT, ge::FORMAT_ND,        // expert_scales
     {1}, ge::DT_INT32, ge::FORMAT_ND,        // comm_cmd_info
     {}, ge::DT_BOOL, ge::FORMAT_ND,              // x_active_mask
     {1, 1, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,           // shared_expert_x
     1,
     {16, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,   // output_x
     "group_ep", 16, 0, 256, 0, 0, 0, 0, 0, 2, "",
     ge::GRAPH_FAILED, 0UL, "", {}, 0},
     // TODO HCCLBUFSIZE校验
};

// setup & teardown
class TestMoeDistributeCombineTeardownTiling
    : public testing::TestWithParam<MoeDistributeCombineTeardownTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TestMoeDistributeCombineTeardownTiling SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TestMoeDistributeCombineTeardownTiling TearDown." << std::endl;
    }
};

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) {
        return gert::StorageShape{};
    }
    return gert::StorageShape{input_shape, input_shape};
}

struct MoeDistributeCombineTeardownCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MoeDistributeCombineTeardownTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;

    // 参数封装
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{make_shape(param.expandXShape), param.expandXDtype, param.expandXFormat},
         {make_shape(param.quantExpandXShape), param.quantExpandXDtype, param.quantExpandXFormat},
         {make_shape(param.expertIdsShape), param.expertIdsDtype, param.expertIdsFormat},
         {make_shape(param.expandIdxShape), param.expandIdxDtype, param.expandIdxFormat},
         {make_shape(param.expertScalesShape), param.expertScalesDtype, param.expertScalesFormat},
         {make_shape(param.commCmdInfoShape), param.commCmdInfoDtype, param.commCmdInfoFormat},
         {make_shape(param.xActiveMaskShape), param.xActiveMaskDtype, param.xActiveMaskFormat},
         {make_shape(param.sharedExpertXShape), param.sharedExpertXDtype, param.sharedExpertXFormat}});

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{make_shape(param.outputXShape), param.outputXDtype, param.outputXFormat}});

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

static void ThreadFunction(const MoeDistributeCombineTeardownTestParam *testCases, size_t caseNum, size_t threadIdx,
                           size_t threadNum)
{
    for (size_t idx = threadIdx; idx < caseNum; idx += threadNum) {
        auto param = testCases[idx];
        auto tilingContextPara = BuildTilingContextPara(param);
        ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                        param.expectWorkspaces, param.mc2TilingDataReservedLen);
    }
}

static void TestExecMultiThread(const MoeDistributeCombineTeardownTestParam *testCases, size_t testCaseNum,
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

TEST_P(TestMoeDistributeCombineTeardownTiling, general_cases)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.epWorldSize}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.status, param.expectTilingKey,
                       param.expectTilingData, param.expectWorkspaces, param.mc2TilingDataReservedLen);
}

TEST_F(TestMoeDistributeCombineTeardownTiling, general_cases_multi_thread)
{
    TestExecMultiThread(test_cases, sizeof(test_cases) / sizeof(MoeDistributeCombineTeardownTestParam), 3);
}

INSTANTIATE_TEST_CASE_P(MoeDistributeCombineTeardownTilingUT, TestMoeDistributeCombineTeardownTiling,
                        testing::ValuesIn(test_cases));

} // namespace
