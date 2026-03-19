/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"
#include "../test_moe_distribute_combine_v2_host_ut_param.h"

namespace MoeDistributeCombineV2UT {

class MoeDistributeCombineV2Arch35TilingTest : public testing::TestWithParam<MoeDistributeCombineV2TilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineV2Arch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineV2Arch35TilingTest TearDown" << std::endl;
    }
};

TEST_P(MoeDistributeCombineV2Arch35TilingTest, param)
{
    auto param = GetParam();
    struct MoeDistributeCombineV2CompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombineV2",
        {
            param.expand_x,
            param.expert_ids,
            param.assist_info_for_combine,
            param.ep_send_counts,
            param.expert_scales,
            param.tp_send_counts,
            param.x_active_mask,
            param.activation_scale,
            param.weight_scale,
            param.group_list,
            param.expand_scales,
            param.shared_expert_x,
            param.elastic_info,
            param.ori_x,
            param.const_expert_alpha_1,
            param.const_expert_alpha_2,
            param.const_expert_v,
            param.performance_info
        },
        {
            param.x
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group_ep)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.ep_world_size)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.ep_rank_id)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.moe_expert_num)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group_tp)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.tp_world_size)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.tp_rank_id)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.expert_shard_type)},
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.shared_expert_num)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.shared_expert_rank_num)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.global_bs)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.out_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_quant_mode)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_list_type)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.comm_alg)},
            {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.zero_expert_num)},
            {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.copy_expert_num)},
            {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.const_expert_num)}
        },
        &compileInfo,
        param.soc, param.coreNum, param.ubsize
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", param.ranksize}
    };
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey);
}

INSTANTIATE_TEST_SUITE_P(
    MoeDistributeCombineV2,
    MoeDistributeCombineV2Arch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<MoeDistributeCombineV2TilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MoeDistributeCombineV2TilingUtParam>
);

} // namespace MoeDistributeCombineV2UT
