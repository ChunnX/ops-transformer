/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_distribute_combine_teardown_tiling.cpp
 * \brief host侧tiling ut
 */
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace MoeDistributeCombineTeardownUT {

const std::string OP_NAME = "MoeDistributeCombineTeardown";

struct MoeDistributeCombineTeardownTestParam {
    uint64_t inputTotalNum;
    uint64_t outputTotalNum;
    std::string caseName;
    std::string socVersion;
    
    // inputs
    std::initializer_list<int64_t> expandXShape;
    std::initializer_list<int64_t> quantExpandXShape;
    std::initializer_list<int64_t> expertIdsShape;
    std::initializer_list<int64_t> expandIdxShape;
    std::initializer_list<int64_t> expertScalesShape;
    std::initializer_list<int64_t> commCmdInfoShape;
    
    // input dtypes
    ge::DataType expandXDtype;
    ge::DataType quantExpandXDtype;
    ge::DataType expertIdsDtype;
    ge::DataType expandIdxDtype;
    ge::DataType expertScalesDtype;
    ge::DataType commCmdInfoDtype;
    
    // attrs
    int64_t epWorldSizeAttr;
    int64_t epRankIdAttr;
    int64_t moeExpertNumAttr;
    int64_t expertShardTypeAttr;
    int64_t sharedExpertNumAttr;
    int64_t sharedExpertRankNumAttr;
    int64_t globalBsAttr;
    int64_t quantModeAttr;
    int64_t commTypeAttr;
    uint64_t rankNum;
    
    // output
    std::initializer_list<int64_t> outputShape;
    ge::DataType outputDtype;
    
    // expert result
    ge::graphStatus status;
    uint64_t expectTilingKey;
};

inline std::ostream& operator<<(std::ostream& os, const MoeDistributeCombineTeardownTestParam& param)
{
    return os << param.caseName;
}

// 构造ut用例：这里可以按照正常用例/异常用例分开声明
static MoeDistributeCombineTeardownTestParam g_testCases[] = {
    {6, 1, "combine_teardown_1", "3510",
    // expandX: A, H (where A = Bs * epWorldSize * min(localMoeExpertNum, K) = 8 * 2 * min(16, 6) = 96)
    {96, 4096},
    // quantExpandX: A, tokenMsgSize (tokenMsgSize = CeilAlign(CeilAlign(4096, 32) + CeilAlign(4096, 8)/8*4, 512) = 4608)
    {96, 4608},
    // expertIds: Bs, K
    {8, 6},
    // expandIdx: Bs * K
    {48},
    // expertScales: Bs, K
    {8, 6},
    // commCmdInfo: (A + epWorldSize) * 16 = (96 + 2) * 16 = 1568
    {1568},
    // input dtypes
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT32,
    // attrs: epWorldSize, epRankId, moeExpertNum, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, quantMode, commType, rankNum
    2L, 0L, 32L, 0L, 0L, 0L, 0L, 1L, 2L, 2UL,
    // output: Bs, H
    {8, 4096}, ge::DT_FLOAT16,
    // expert result: status, expectTilingKey
    ge::GRAPH_SUCCESS, 0UL},
};

class MoeDistributeCombineTeardownTilingTest : public testing::TestWithParam<MoeDistributeCombineTeardownTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineTeardownTilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineTeardownTilingTest TearDown." << std::endl;
    }
};

static struct MoeDistributeCombineTeardownCompileInfo {} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MoeDistributeCombineTeardownTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    
    std::vector<std::pair<std::initializer_list<int64_t>, ge::DataType>> inputShapeDtypeList = {
        {param.expandXShape, param.expandXDtype},
        {param.quantExpandXShape, param.quantExpandXDtype},
        {param.expertIdsShape, param.expertIdsDtype},
        {param.expandIdxShape, param.expandIdxDtype},
        {param.expertScalesShape, param.expertScalesDtype},
        {param.commCmdInfoShape, param.commCmdInfoDtype}
    };
    
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc;
    for (size_t i = 0; i < param.inputTotalNum; i++) {
        gert::StorageShape shape = {inputShapeDtypeList[i].first, inputShapeDtypeList[i].first};
        inputTensorDesc.push_back({shape, inputShapeDtypeList[i].second, ge::FORMAT_ND});
    }
    
    gert::StorageShape outputShape = {param.outputShape, param.outputShape};
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc({
        {outputShape, param.outputDtype, ge::FORMAT_ND}
    });
    
    std::vector<gert::TilingContextPara::OpAttr> attrs({
        {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hccl_world_group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSizeAttr)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epRankIdAttr)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.moeExpertNumAttr)},
        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.expertShardTypeAttr)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sharedExpertNumAttr)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sharedExpertRankNumAttr)},
        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.globalBsAttr)},
        {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quantModeAttr)},
        {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commTypeAttr)}
    });
    
    return gert::TilingContextPara(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo, param.socVersion);
}

static void ThreadFunction(const MoeDistributeCombineTeardownTestParam *testCases, size_t caseNum, size_t threadIdx,
                            size_t threadNum)
{
    for (size_t idx = threadIdx; idx < caseNum; idx += threadNum) {
        auto param = testCases[idx];
        auto tilingContextPara = BuildTilingContextPara(param);
        Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rankNum}};
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.status, param.expectTilingKey, "", {}, MC2_TILING_DATA_RESERVED_LEN);
    }
}

static void TestExecMultiThread(const MoeDistributeCombineTeardownTestParam *testCases, size_t testCaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunction, testCases, testCaseNum, idx, threadNum);
    }
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

TEST_P(MoeDistributeCombineTeardownTilingTest, GeneralCasesTest)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rankNum}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.status, param.expectTilingKey, "", {}, MC2_TILING_DATA_RESERVED_LEN);
}

TEST_F(MoeDistributeCombineTeardownTilingTest, GeneralCasesMultiThreadTest)
{
    TestExecMultiThread(g_testCases, sizeof(g_testCases) / sizeof(MoeDistributeCombineTeardownTestParam), 3);
}

INSTANTIATE_TEST_CASE_P(MoeDistributeCombineTeardownTilingUT, MoeDistributeCombineTeardownTilingTest, testing::ValuesIn(g_testCases));

} // namespace
