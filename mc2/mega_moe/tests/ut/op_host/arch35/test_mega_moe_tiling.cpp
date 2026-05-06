/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_mega_moe_tiling.cpp
 * \brief MegaMoe算子tiling UT
 */

#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace MegaMoeTilingUT {

static const std::string OP_NAME = "MegaMoe";

struct MegaMoeTestParam {
    std::string caseName;
    
    // context
    std::initializer_list<int64_t> contextShape;
    ge::DataType contextDtype;
    ge::Format contextFormat;
    
    // x
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    
    // topk_ids
    std::initializer_list<int64_t> topkIdsShape;
    ge::DataType topkIdsDtype;
    ge::Format topkIdsFormat;
    
    // topk_weights
    std::initializer_list<int64_t> topkWeightsShape;
    ge::DataType topkWeightsDtype;
    ge::Format topkWeightsFormat;
    
    // weight1
    std::initializer_list<int64_t> weight1Shape;
    ge::DataType weight1Dtype;
    ge::Format weight1Format;
    
    // weight2
    std::initializer_list<int64_t> weight2Shape;
    ge::DataType weight2Dtype;
    ge::Format weight2Format;
    
    // weight_scales1
    std::initializer_list<int64_t> weightScales1Shape;
    ge::DataType weightScales1Dtype;
    ge::Format weightScales1Format;
    
    // weight_scales2
    std::initializer_list<int64_t> weightScales2Shape;
    ge::DataType weightScales2Dtype;
    ge::Format weightScales2Format;
    
    // y
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    
    // expert_token_nums
    std::initializer_list<int64_t> expertTokenNumsShape;
    ge::DataType expertTokenNumsDtype;
    ge::Format expertTokenNumsFormat;
    
    // attrs
    int64_t moeExpertNum;
    int64_t epWorldSize;
    int64_t cclBufferSize;
    int64_t maxRecvTokenNum;
    int64_t dispatchQuantMode;
    int64_t dispatchQuantOutType;
    int64_t combineQuantMode;
    std::string commAlg;
    int64_t globalBs;
    
    // soc version
    std::string socVersion;
    
    // expected results
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t mc2TilingDataReservedLen;
};

inline std::ostream &operator<<(std::ostream &os, const MegaMoeTestParam &param)
{
    return os << param.caseName;
}

// 测试用例数组
static MegaMoeTestParam g_testCases[] = {
    // 正常用例 - 不同 H 配置
    {"mega_moe_case0_h4096_bs128", {1}, ge::DT_INT32, ge::FORMAT_ND, {128, 4096}, ge::DT_BF16, ge::FORMAT_ND, {128, 6}, ge::DT_INT32, ge::FORMAT_ND, {128, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 1024, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {4, 4096, 512}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {4, 1024, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {4, 4096, 8, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {128, 4096}, ge::DT_BF16, ge::FORMAT_ND, {4}, ge::DT_INT32, ge::FORMAT_ND, 16, 4, 2097152, 0, 4, 24, 0, "", 0, "3510", ge::GRAPH_SUCCESS, 0UL, "", {16777216}, 0},
    
    {"mega_moe_case1_h5120_bs256", {1}, ge::DT_INT32, ge::FORMAT_ND, {256, 5120}, ge::DT_BF16, ge::FORMAT_ND, {256, 8}, ge::DT_INT32, ge::FORMAT_ND, {256, 8}, ge::DT_BF16, ge::FORMAT_ND, {8, 2048, 5120}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, {8, 5120, 1024}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, {8, 2048, 80, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {8, 5120, 16, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {256, 5120}, ge::DT_BF16, ge::FORMAT_ND, {8}, ge::DT_INT32, ge::FORMAT_ND, 64, 8, 2097152, 0, 4, 23, 0, "", 0, "3510", ge::GRAPH_SUCCESS, 0UL, "", {16777216}, 0},
    
    {"mega_moe_case2_h7168_bs512", {1}, ge::DT_INT32, ge::FORMAT_ND, {512, 7168}, ge::DT_BF16, ge::FORMAT_ND, {512, 8}, ge::DT_INT32, ge::FORMAT_ND, {512, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 4096, 7168}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {2, 7168, 2048}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {2, 4096, 112, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {2, 7168, 32, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {512, 7168}, ge::DT_BF16, ge::FORMAT_ND, {2}, ge::DT_INT32, ge::FORMAT_ND, 4, 2, 2097152, 0, 4, 24, 0, "", 0, "3510", ge::GRAPH_SUCCESS, 0UL, "", {16777216}, 0},
    
    // 正常用例 - 不同 N 配置
    {"mega_moe_case3_n2048", {1}, ge::DT_INT32, ge::FORMAT_ND, {128, 4096}, ge::DT_BF16, ge::FORMAT_ND, {128, 6}, ge::DT_INT32, ge::FORMAT_ND, {128, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 2048, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {4, 4096, 1024}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {4, 2048, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {4, 4096, 16, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {128, 4096}, ge::DT_BF16, ge::FORMAT_ND, {4}, ge::DT_INT32, ge::FORMAT_ND, 16, 4, 2097152, 0, 4, 23, 0, "", 0, "3510", ge::GRAPH_SUCCESS, 0UL, "", {16777216}, 0},
    
    {"mega_moe_case4_n3072", {1}, ge::DT_INT32, ge::FORMAT_ND, {128, 4096}, ge::DT_BF16, ge::FORMAT_ND, {128, 8}, ge::DT_INT32, ge::FORMAT_ND, {128, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 3072, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {4, 4096, 1536}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {4, 3072, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {4, 4096, 24, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {128, 4096}, ge::DT_BF16, ge::FORMAT_ND, {4}, ge::DT_INT32, ge::FORMAT_ND, 16, 4, 2097152, 0, 4, 24, 0, "", 0, "3510", ge::GRAPH_SUCCESS, 0UL, "", {16777216}, 0},
    
    // 异常用例 - 参数约束违反
    {"mega_moe_case5_invalid_ep_world_size_1", {1}, ge::DT_INT32, ge::FORMAT_ND, {128, 4096}, ge::DT_BF16, ge::FORMAT_ND, {128, 6}, ge::DT_INT32, ge::FORMAT_ND, {128, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 1024, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {4, 4096, 512}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {4, 1024, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {4, 4096, 8, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {128, 4096}, ge::DT_BF16, ge::FORMAT_ND, {4}, ge::DT_INT32, ge::FORMAT_ND, 16, 1, 2097152, 0, 4, 24, 0, "", 0, "3510", ge::GRAPH_FAILED, 0UL, "", {16777216}, 0},
    
    {"mega_moe_case6_invalid_moe_expert_num_not_multiple", {1}, ge::DT_INT32, ge::FORMAT_ND, {128, 4096}, ge::DT_BF16, ge::FORMAT_ND, {128, 6}, ge::DT_INT32, ge::FORMAT_ND, {128, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 1024, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {4, 4096, 512}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {4, 1024, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {4, 4096, 8, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {128, 4096}, ge::DT_BF16, ge::FORMAT_ND, {4}, ge::DT_INT32, ge::FORMAT_ND, 15, 4, 2097152, 0, 4, 24, 0, "", 0, "3510", ge::GRAPH_FAILED, 0UL, "", {16777216}, 0},
    
    {"mega_moe_case7_invalid_h_not_supported", {1}, ge::DT_INT32, ge::FORMAT_ND, {128, 2048}, ge::DT_BF16, ge::FORMAT_ND, {128, 6}, ge::DT_INT32, ge::FORMAT_ND, {128, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 1024, 2048}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {4, 2048, 512}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, {4, 1024, 32, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {4, 2048, 8, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, {128, 2048}, ge::DT_BF16, ge::FORMAT_ND, {4}, ge::DT_INT32, ge::FORMAT_ND, 16, 4, 2097152, 0, 4, 24, 0, "", 0, "3510", ge::GRAPH_FAILED, 0UL, "", {16777216}, 0},
};

// setup & teardown
class MegaMoeArch35TilingTest : public testing::TestWithParam<MegaMoeTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MegaMoeArch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MegaMoeArch35TilingTest TearDown." << std::endl;
    }
};

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) {
        return gert::StorageShape{};
    }
    return gert::StorageShape{input_shape, input_shape};
}

static struct MegaMoeCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MegaMoeTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;

    // 输入Tensor描述
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{make_shape(param.contextShape), param.contextDtype, param.contextFormat},
         {make_shape(param.xShape), param.xDtype, param.xFormat},
         {make_shape(param.topkIdsShape), param.topkIdsDtype, param.topkIdsFormat},
         {make_shape(param.topkWeightsShape), param.topkWeightsDtype, param.topkWeightsFormat},
         {make_shape(param.weight1Shape), param.weight1Dtype, param.weight1Format},
         {make_shape(param.weight2Shape), param.weight2Dtype, param.weight2Format},
         {make_shape(param.weightScales1Shape), param.weightScales1Dtype, param.weightScales1Format},
         {make_shape(param.weightScales2Shape), param.weightScales2Dtype, param.weightScales2Format},
         {gert::StorageShape{}, ge::DT_INT8, ge::FORMAT_ND},
         {gert::StorageShape{}, ge::DT_FLOAT, ge::FORMAT_ND}});

    // 输出Tensor描述
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{make_shape(param.yShape), param.yDtype, param.yFormat},
         {make_shape(param.expertTokenNumsShape), param.expertTokenNumsDtype, param.expertTokenNumsFormat}});

    // 属性列表
    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.moeExpertNum)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSize)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.cclBufferSize)},
         {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.maxRecvTokenNum)},
         {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.dispatchQuantMode)},
         {"dispatch_quant_out_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.dispatchQuantOutType)},
         {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.combineQuantMode)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commAlg)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.globalBs)}});

    return gert::TilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                    param.socVersion);
}

TEST_P(MegaMoeArch35TilingTest, GeneralCases)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", static_cast<size_t>(param.epWorldSize)}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.status, param.expectTilingKey,
                       param.expectTilingData, param.expectWorkspaces, param.mc2TilingDataReservedLen);
}

INSTANTIATE_TEST_CASE_P(MegaMoeTilingUT, MegaMoeArch35TilingTest,
                        testing::ValuesIn(g_testCases));

} // namespace MegaMoeTilingUT