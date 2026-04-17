/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>

#include <gtest/gtest.h>
#include "op_log.h"
#include "kernel_run_context_faker.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../op_kernel/mega_moe_tiling_def.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class MegaMoeTilingTest : public testing::Test {
 public:
  static void SetUpTestCase() {
    std::cout << "MegaMoeTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MegaMoeTilingTest TearDown" << std::endl;
  }
};

TEST_F(MegaMoeTilingTest, test_tiling_basic) {
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 0,
            "UB_SIZE": 253952,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "CORE_NUM": 64,
            "socVersion": "Ascend950"
        }
    })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    std::string op_type("MegaMoe");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    ASSERT_NE(tiling_func, nullptr);

    auto param = gert::TilingData::CreateCap(4096 * 10);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    uint32_t M = 128;
    uint32_t K = 512;
    uint32_t N = 256;
    uint32_t topK = 2;
    uint32_t expertPerRank = 4;
    uint32_t EP = 1;
    uint32_t epRankId = 0;

    gert::StorageShape xShape = {{M, K}, {M, K}};
    gert::StorageShape expertIdsShape = {{M, topK}, {M, topK}};
    gert::StorageShape weight1Shape = {{expertPerRank, K, N}, {expertPerRank, K, N}};
    gert::StorageShape yShape = {{M, K}, {M, K}};
    gert::StorageShape contextShape = {{1024}, {1024}};
    
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(10, 1)
                      .InputShapes({&contextShape, &xShape, &expertIdsShape, nullptr, nullptr, nullptr, 
                                   &weight1Shape, nullptr, nullptr, nullptr})
                      .OutputShapes({&yShape})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"group_ep", ge::AnyValue::CreateFrom<string>("ep_group")},
                          {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(EP)},
                          {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(epRankId)},
                          {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(expertPerRank * EP)},
                          {"ccl_buffer_size", ge::AnyValue::CreateFrom<int64_t>(1024 * 1024 * 50)},
                          {"max_recv_token_num", ge::AnyValue::CreateFrom<int64_t>(M * topK)},
                          {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(1)},
                          {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"dispatch_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"combine_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"global_bs", ge::AnyValue::CreateFrom<int64_t>(M)}
                      })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}