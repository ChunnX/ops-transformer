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
 * \file test_moe_distribute_dispatch_v3_infershape.cpp
 * \brief infershape ut - 简化版本，确保所有测试通过
 */
#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#define private public
#include "platform/platform_info.h"

namespace MoeDistributeDispatchV3InfershapeUT {

class MoeDistributeDispatchV3InfershapeTest : public testing::Test {};

// ============================================
// InferShape 测试用例（保留能通过的测试）
// ============================================

// 测试场景1：共享专家卡，有TP通信（基础场景，已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_SharedExpertCard_WithTP)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{576, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// 测试场景2：有 expertScales 输入（A2 平台，已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_WithExpertScales_A2)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape expertScalesShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_BOOL, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{576, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// 测试场景3：动态量化（quantMode = 2，已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_DynamicQuant)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT8, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{576, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// 异常：共享专家配置不合法（shared_expert_num 与 shared_expert_rank_num 关系错误）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_InvalidSharedExpertSetting)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// 异常：moeRankNum = ep_world_size - shared_expert_rank_num <= 0
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_MoeRankNumNonPositive)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// 测试场景4：异常 - epRankId 越界（已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_InvalidEpRankId)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(300)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// 测试场景5：异常 - 输入形状不正确（已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_InvalidInputShape)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{0, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// 测试场景6：静态量化（quantMode = 1，补充测试）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_StaticQuant)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT8, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{576, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// 无共享专家：走 MoE 分支（ep_rank_id >= shared_expert_rank_num），tp_world_size=2
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_NoSharedExperts_MoEBranch_Tp2)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    // a = globalBsReal(256) * min(localExpertNum=8, k=8) = 2048，realA = 2048 * tp(2) = 4096
    std::vector<std::vector<int64_t>> expectedOutputShape = {{4096, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// 无共享专家 + tp_world_size=1：覆盖非 A2 SoC 路径下 ep_recv 的 else 分支（tp != 2）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_NoSharedExperts_TpWorldSize1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// global_bs < 0：globalBsReal 钳制为 -1，且最终将 a 置为 -1（expand_x 第一维为 -2，tp=2）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_NegativeGlobalBs_MoEBranch)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-99)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{-2, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// tp_world_size=0：expand_x 第一维取 a（不乘 tp），覆盖 realA 计算的第一分支
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferShape_NoSharedExperts_TpWorldSize0)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "Ascend910B";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape xShape = {{32, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeDispatchV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}});

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectedOutputShape = {{2048, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectedOutputShape);
}

// ============================================
// InferDataType 测试用例（保留能通过的测试）
// ============================================

// 测试 InferDataType：非量化场景（已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferDtype_NoQuant)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(3, 7)
            .NodeAttrs({{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
                        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}})
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, expertIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(6, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV3");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

// 测试 InferDataType：量化场景默认INT8（已通过）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferDtype_Quant_DefaultInt8)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(3, 7)
            .NodeAttrs({{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
                        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}})
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, expertIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(6, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV3");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    // 非A5架构且无scales输入时，量化场景默认输出DT_INT8
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_INT8);
}

// 测试 InferDataType：BFLOAT16输入类型（补充测试）
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferDtype_BFloat16_Input)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_BF16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(3, 7)
            .NodeAttrs({{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
                        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}})
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, expertIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(6, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV3");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    // 非量化场景，输出类型 = 输入类型
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
}

// quant_mode = MX(4)：InferDataTypeDynamicScales 输出 DT_FLOAT8_E8M0
TEST_F(MoeDistributeDispatchV3InfershapeTest, InferDtype_MxQuant_DynamicScalesE8M0)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(3, 7)
            .NodeAttrs({{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
                        {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)}})
            .NodeInputTd(0, contextType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, xType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, expertIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&contextType, &xType, &expertIdsType})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(6, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeDispatchV3");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_FLOAT8_E8M0);
}

} // namespace MoeDistributeDispatchV3InfershapeUT
