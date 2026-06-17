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

namespace MoeEpDispatchEpilogueInferShapeUT {

class MoeEpDispatchEpilogueInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeEpDispatchEpilogueInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeEpDispatchEpilogueInfershape TearDown" << std::endl;
    }
};

// Normal case: valid shapes with typical parameters
TEST_F(MoeEpDispatchEpilogueInfershape, InferShape0)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape dstSlotIdxShape = {{2048, 8}, {}};
    gert::StorageShape numRecvPerRankShape = {{8}, {}};
    gert::StorageShape numRecvPerExpertShape = {{8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeEpDispatchEpilogue",
        {
            {contextShape, ge::DT_INT32, ge::FORMAT_ND},
            {dstSlotIdxShape, ge::DT_INT32, ge::FORMAT_ND},
            {numRecvPerRankShape, ge::DT_INT32, ge::FORMAT_ND},
            {numRecvPerExpertShape, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    int64_t aAlloc = 8 * 2048;
    std::vector<std::vector<int64_t>> xOutputShape = {
        {aAlloc, -1},      // recv_x (hidden is -1, dynamic)
        {aAlloc, 8},       // recv_topk_idx
        {aAlloc},          // recv_topk_weights
        {aAlloc, 3},       // recv_src_metadata
    };
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

// InferDataType test
TEST_F(MoeEpDispatchEpilogueInfershape, InferDtype0)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType dstSlotIdxType = ge::DT_INT32;
    ge::DataType numRecvPerRankType = ge::DT_INT32;
    ge::DataType numRecvPerExpertType = ge::DT_INT64;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(4, 5)
                             .InputDataTypes({&contextType, &dstSlotIdxType, &numRecvPerRankType,
                                              &numRecvPerExpertType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeEpDispatchEpilogue")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_INT32);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_INT32);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(2), ge::DT_FLOAT);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(3), ge::DT_INT32);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(4), ge::DT_FLOAT);
}

// Small shape test
TEST_F(MoeEpDispatchEpilogueInfershape, InferShapeSmall)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape dstSlotIdxShape = {{512, 4}, {}};
    gert::StorageShape numRecvPerRankShape = {{4}, {}};
    gert::StorageShape numRecvPerExpertShape = {{8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeEpDispatchEpilogue",
        {
            {contextShape, ge::DT_INT32, ge::FORMAT_ND},
            {dstSlotIdxShape, ge::DT_INT32, ge::FORMAT_ND},
            {numRecvPerRankShape, ge::DT_INT32, ge::FORMAT_ND},
            {numRecvPerExpertShape, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)},
         {"expert_alignment", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    int64_t aAlloc = 4 * 512;
    std::vector<std::vector<int64_t>> xOutputShape = {
        {aAlloc, -1}, {aAlloc, 4}, {aAlloc}, {aAlloc, 3},
    };
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

} // namespace MoeEpDispatchEpilogueInferShapeUT
