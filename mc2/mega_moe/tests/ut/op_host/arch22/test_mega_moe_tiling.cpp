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
#include <map>
#include <vector>
#include <string>
#include <limits>

#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

#include "mc2_tiling_case_executor.h"

namespace MegaMoeUT {
class MegaMoeArch22TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MegaMoeArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MegaMoeArch22TilingTest TearDown" << std::endl;
    }
};

TEST_F(MegaMoeArch22TilingTest, Test0)
{
    struct MegaMoeCompileInfo {};
    MegaMoeCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    constexpr int64_t kHiddenSize = 7168;
    constexpr int64_t kN = 28672;
    constexpr int64_t kNHalf = 14336;
    constexpr uint32_t kExpertPerRank = 8;

    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.push_back({{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{32, kHiddenSize}, {32, kHiddenSize}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    inputs.push_back({{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND});
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kHiddenSize, kN}, {kHiddenSize, kN}}, ge::DT_INT4, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kNHalf, kHiddenSize}, {kNHalf, kHiddenSize}}, ge::DT_INT4, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kN}, {kN}}, ge::DT_UINT64, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kHiddenSize}, {kHiddenSize}}, ge::DT_UINT64, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kN}, {kN}}, ge::DT_FLOAT, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kHiddenSize}, {kHiddenSize}}, ge::DT_FLOAT, ge::FORMAT_ND});
    }

    gert::TilingContextPara tilingContextPara(
        "MegaMoe", inputs,
        {
            {{{32, kHiddenSize}, {32, kHiddenSize}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kExpertPerRank}, {kExpertPerRank}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"dispatch_quant_mode",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, // INT4/INT8 -> dispatch_quant_mode=2
         {"dispatch_quant_out_type",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, // INT4/INT8 -> dispatch_quant_out_type=INT8
         {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"num_max_token_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
         {"activation_clamp", Ops::Transformer::AnyValue::CreateFrom<float>(std::numeric_limits<float>::max())},
         {"activation_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        {1, 1, 1, 1, kExpertPerRank, kExpertPerRank, kExpertPerRank, kExpertPerRank, kExpertPerRank, kExpertPerRank},
        {1, 1}, &compileInfo, "Ascend910B", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, UINT64_MAX);
}

} // namespace MegaMoeUT