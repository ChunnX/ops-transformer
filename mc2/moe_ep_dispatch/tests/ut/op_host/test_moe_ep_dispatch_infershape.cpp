/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MoeEpDispatchInferShapeUT {

class MoeEpDispatchInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeEpDispatchInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeEpDispatchInfershape TearDown" << std::endl;
    }
};

// Normal case: non-cached, valid shapes
TEST_F(MoeEpDispatchInfershape, InferShape0)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{2048, 7168}, {}};
    gert::StorageShape topkIdxShape = {{2048, 8}, {}};
    gert::StorageShape topkWeightsShape = {{2048, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara("MoeEpDispatch",
        {
            {contextShape, ge::DT_INT32, ge::FORMAT_ND},
            {xShape, ge::DT_BF16, ge::FORMAT_ND},
            {topkIdxShape, ge::DT_INT32, ge::FORMAT_ND},
            {topkWeightsShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };
    int64_t numLocalExperts = 8;
    std::vector<std::vector<int64_t>> xOutputShape = {
        {8},                       // num_recv_tokens_per_rank
        {numLocalExperts},         // num_recv_tokens_per_expert
        {2048, 8},                 // dst_buffer_slot_idx
    };
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

// Normal case: cached mode (topk_idx/topk_weights null, cached_handle_dst_buffer_slot_idx present)
TEST_F(MoeEpDispatchInfershape, InferShapeCached)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{1024, 4096}, {}};
    gert::StorageShape cachedSlotShape = {{1024, 4}, {}};

    gert::InfershapeContextPara infershapeContextPara("MoeEpDispatch",
        {
            {contextShape, ge::DT_INT32, ge::FORMAT_ND},
            {xShape, ge::DT_BF16, ge::FORMAT_ND},
            // topk_idx: optional null
            // topk_weights: optional null
            {cachedSlotShape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };
    int64_t numLocalExperts = 8;
    std::vector<std::vector<int64_t>> xOutputShape = {
        {4},
        {numLocalExperts},
        {1024, 4},
    };
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

// InferDataType test: verify output dtypes
TEST_F(MoeEpDispatchInfershape, InferDtype0)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_BF16;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(2, 3)
        .InputDataTypes({&contextType, &xType})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeEpDispatch")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_INT32);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_INT64);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(2), ge::DT_INT32);
}

// InferShape with FP16 input
TEST_F(MoeEpDispatchInfershape, InferShapeFp16)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{512, 2048}, {}};
    gert::StorageShape topkIdxShape = {{512, 4}, {}};
    gert::StorageShape topkWeightsShape = {{512, 4}, {}};

    gert::InfershapeContextPara infershapeContextPara("MoeEpDispatch",
        {
            {contextShape, ge::DT_INT32, ge::FORMAT_ND},
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {topkIdxShape, ge::DT_INT32, ge::FORMAT_ND},
            {topkWeightsShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
            {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"do_cpu_sync", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };
    int64_t numLocalExperts = 8;
    std::vector<std::vector<int64_t>> xOutputShape = {
        {2},
        {numLocalExperts},
        {512, 4},
    };
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

} // namespace MoeEpDispatchInferShapeUT
