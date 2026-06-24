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

namespace MoeEpCombineInferShapeUT {

class MoeEpCombineInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeEpCombineInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeEpCombineInfershape TearDown" << std::endl;
    }
};

// Normal case: without topk_weights (pure sum)
TEST_F(MoeEpCombineInfershape, InferShape0)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{8 * 2048, 7168}, {}};
    gert::StorageShape dstSlotIdxShape = {{2048, 8}, {}};
    gert::StorageShape recvSrcMetadataShape = {{8 * 2048, 3}, {}};
    gert::StorageShape numRecvPerRankShape = {{8}, {}};

    gert::InfershapeContextPara infershapeContextPara("MoeEpCombine",
        {
            {contextShape, ge::DT_INT32, ge::FORMAT_ND},
            {xShape, ge::DT_BF16, ge::FORMAT_ND},
            {dstSlotIdxShape, ge::DT_INT32, ge::FORMAT_ND},
            {recvSrcMetadataShape, ge::DT_INT32, ge::FORMAT_ND},
            {numRecvPerRankShape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_BF16, ge::FORMAT_ND},     // combined_x
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},     // combined_topk_weights (optional, empty)
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };
    std::vector<std::vector<int64_t>> xOutputShape = {
        {2048, 7168},   // combined_x
        {},              // combined_topk_weights (empty, no topk_weights input)
    };
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

// Normal case: with topk_weights
TEST_F(MoeEpCombineInfershape, InferShapeWithTopkWeights)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{8 * 2048, 7168}, {}};
    gert::StorageShape dstSlotIdxShape = {{2048, 8}, {}};
    gert::StorageShape recvSrcMetadataShape = {{8 * 2048, 3}, {}};
    gert::StorageShape numRecvPerRankShape = {{8}, {}};
    gert::StorageShape topkWeightsShape = {{2048, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara("MoeEpCombine",
        {
            {contextShape, ge::DT_INT32, ge::FORMAT_ND},
            {xShape, ge::DT_BF16, ge::FORMAT_ND},
            {dstSlotIdxShape, ge::DT_INT32, ge::FORMAT_ND},
            {recvSrcMetadataShape, ge::DT_INT32, ge::FORMAT_ND},
            {numRecvPerRankShape, ge::DT_INT32, ge::FORMAT_ND},
            {topkWeightsShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };
    std::vector<std::vector<int64_t>> xOutputShape = {
        {2048, 7168},
        {2048, 8},
    };
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

// InferDataType test: without topk_weights
TEST_F(MoeEpCombineInfershape, InferDtype0)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType xType = ge::DT_BF16;
    ge::DataType dstSlotIdxType = ge::DT_INT32;
    ge::DataType recvSrcMetaType = ge::DT_INT32;
    ge::DataType numRecvPerRankType = ge::DT_INT32;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(5, 2)
        .InputDataTypes({&contextType, &xType, &dstSlotIdxType,
                         &recvSrcMetaType, &numRecvPerRankType})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeEpCombine")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
}

// Small shape test
TEST_F(MoeEpCombineInfershape, InferShapeSmall)
{
    gert::StorageShape contextShape = {{1}, {}};
    gert::StorageShape xShape = {{2 * 256, 2048}, {}};
    gert::StorageShape dstSlotIdxShape = {{256, 4}, {}};
    gert::StorageShape recvSrcMetadataShape = {{2 * 256, 3}, {}};
    gert::StorageShape numRecvPerRankShape = {{2}, {}};

    gert::InfershapeContextPara infershapeContextPara("MoeEpCombine",
        {
            {contextShape, ge::DT_INT32, ge::FORMAT_ND},
            {xShape, ge::DT_BF16, ge::FORMAT_ND},
            {dstSlotIdxShape, ge::DT_INT32, ge::FORMAT_ND},
            {recvSrcMetadataShape, ge::DT_INT32, ge::FORMAT_ND},
            {numRecvPerRankShape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(200 * 1024 * 1024)}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };
    std::vector<std::vector<int64_t>> xOutputShape = {
        {256, 2048},
        {},
    };
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

} // namespace MoeEpCombineInferShapeUT
