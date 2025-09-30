/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <gtest/gtest.h>
// #include "op_log.h"
// #include "register/op_tiling_registry.h"
// #include "test_common.h"
// #include "pad_ops.h"
// #include "array_ops.h"
// #include "common/utils/ut_op_util.h"
// #include "op_tiling/op_tiling_util.h"
// #include "common_unittest.h"
// #include "transformer/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling.h"
// #include "kernel_run_context_facker.h"
// #include "test_cube_util.h"
// #include "fusion_ops.h"
// #include "exe_graph/runtime/storage_format.h"
// #include "exe_graph/runtime/storage_shape.h"

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_v3_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

constexpr int64_t QUANT_MODE_NONE = -1;
constexpr int64_t QUANT_MODE_STATIC = 0;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
constexpr int64_t ROW_IDX_TYPE_DROPPAD = 0;
constexpr int64_t ROW_IDX_TYPE_DROPLESS = 1;
constexpr int64_t EXPERT_NUM = 256;

class MoeInitRoutingV3Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV3Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV3Tiling TearDown" << std::endl;
    }
};

void RunNormalCase(int64_t N, int64_t H, int64_t K, int64_t C, int64_t dropPadMode, int64_t countFlag, 
                   bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t expectTilingKey)
{   
    optiling::MoeInitRoutingV3CompileInfo compileInfo = {40, 65536};
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM;
    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3",
                                        {
                                            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
                                            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{N * K, H}, {N * K, H}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                        },
                                        {
                                            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(activeNum)},
                                            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(C)},
                                            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expert_num)},
                                            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                            {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(countFlag)},
                                            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tokenFlag)},
                                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                            {"acive_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                                            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rowIdxType)},
                                        },
                                        &compileInfo);
    
    string expectTilingData = "40 1 83 27 180 192 12 -1 0 0 0 256 1 1 27 1 27 27 27 1 27 27 1984 0 1024 27 1 1 1 1 1 1 1 1 27 1 1 1 1 1 1 1 1 1 83 83 ";
    std::vector<size_t> expectWorkspaces = {16780612};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核 + not quant + scale not None   1000000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_01)
{   
    RunNormalCase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1000000);
}