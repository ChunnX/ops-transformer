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

#include "util/shape_util.h"
#include "kernel_run_context_faker.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

#include "../../op_kernel/mega_moe_tiling_def.h"
#include "../../../moe_init_routing_v3/op_kernel/arch35/moe_init_routing_v3_arch35_tiling_def.h"
#include "../../../moe_finalize_routing_v2/op_kernel/arch35/moe_finalize_routing_v2_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace optiling;

constexpr uint32_t NUM_128 = 128;
constexpr uint32_t NUM_256 = 256;
constexpr uint32_t NUM_384 = 384;
constexpr uint32_t NUM_512 = 512;

class MegaMoeTilingTest : public testing::Test {
 public:
  static void SetUpTestCase() {
    std::cout << "MegaMoeTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MegaMoeTilingTest TearDown" << std::endl;
  }
};

TEST_F(MegaMoeTilingTest, test_tiling_basic_M128_K512) {
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 0,
            "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": false,
            "Intrinsic_data_move_l12ub": true,
            "Intrinsic_data_move_l0c2ub": true,
            "Intrinsic_data_move_out2l1_nd2nz": false,
            "UB_SIZE": 253952,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
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
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096 * 10);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    uint32_t M = 128;
    uint32_t K = 512;
    uint32_t N = 256;
    uint32_t topK = 2;
    uint32_t expertPerRank = 4;
    uint32_t EP = 1;
    uint32_t epRankId = 0;

    gert::StorageShape contextShape = {{EP * 1024}, {EP * 1024}};
    gert::StorageShape xShape = {{M, K}, {M, K}};
    gert::StorageShape expertIdsShape = {{M, topK}, {M, topK}};
    gert::StorageShape scalesShape = {{M, topK}, {M, topK}};
    gert::StorageShape xActiveMaskShape = {{M}, {M}};
    gert::StorageShape expertScalesShape = {{M, topK}, {M, topK}};
    gert::StorageShape weight1Shape = {{expertPerRank, K, N}, {expertPerRank, K, N}};
    gert::StorageShape weightScales1Shape = {{expertPerRank, N, K / 64}, {expertPerRank, N, K / 64}};
    gert::StorageShape weight2Shape = {{expertPerRank, N / 2, K}, {expertPerRank, N / 2, K}};
    gert::StorageShape weightScales2Shape = {{expertPerRank, K, N / 2 / 64}, {expertPerRank, K, N / 2 / 64}};
    gert::StorageShape yShape = {{M, K}, {M, K}};
    
    uint64_t cclBufferSize = 1024 * 1024 * 50;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(10, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&contextShape, &xShape, &expertIdsShape, &scalesShape, 
                                   &xActiveMaskShape, &expertScalesShape, &weight1Shape, 
                                   &weightScales1Shape, &weight2Shape, &weightScales2Shape})
                      .OutputShapes({&yShape})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"group_ep", ge::AnyValue::CreateFrom<string>("ep_group")},
                          {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(EP)},
                          {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(epRankId)},
                          {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(expertPerRank * EP)},
                          {"ccl_buffer_size", ge::AnyValue::CreateFrom<int64_t>(cclBufferSize)},
                          {"max_recv_token_num", ge::AnyValue::CreateFrom<int64_t>(M * topK)},
                          {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(1)},
                          {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"dispatch_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"dispatch_quant_out_type", ge::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
                          {"combine_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"comm_alg", ge::AnyValue::CreateFrom<string>("")},
                          {"global_bs", ge::AnyValue::CreateFrom<int64_t>(M)}
                      })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    MegaMoeTilingData* tilingData = tiling_context->GetTilingData<MegaMoeTilingData>();
    ASSERT_NE(tilingData, nullptr);
    
    EXPECT_EQ(tilingData->m, M);
    EXPECT_EQ(tilingData->k, K);
    EXPECT_EQ(tilingData->n, N);
    EXPECT_EQ(tilingData->topK, topK);
    EXPECT_EQ(tilingData->expertPerRank, expertPerRank);
    EXPECT_EQ(tilingData->EP, EP);
    EXPECT_EQ(tilingData->rank, epRankId);
    EXPECT_EQ(tilingData->m0, NUM_128);
    EXPECT_EQ(tilingData->k0, NUM_512);
    EXPECT_EQ(tilingData->n0, NUM_256);
    EXPECT_EQ(tilingData->transX, false);
    EXPECT_EQ(tilingData->transW, true);
    EXPECT_EQ(tilingData->transW2, true);
    
    size_t workspaceSize = ws_size->at(0);
    EXPECT_GT(workspaceSize, 0);
}

TEST_F(MegaMoeTilingTest, test_tiling_basic_M256_K384) {
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 0,
            "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": false,
            "Intrinsic_data_move_l12ub": true,
            "Intrinsic_data_move_l0c2ub": true,
            "Intrinsic_data_move_out2l1_nd2nz": false,
            "UB_SIZE": 253952,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
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
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096 * 10);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    uint32_t M = 256;
    uint32_t K = 384;
    uint32_t N = 256;
    uint32_t topK = 2;
    uint32_t expertPerRank = 4;
    uint32_t EP = 1;
    uint32_t epRankId = 0;

    gert::StorageShape contextShape = {{EP * 1024}, {EP * 1024}};
    gert::StorageShape xShape = {{M, K}, {M, K}};
    gert::StorageShape expertIdsShape = {{M, topK}, {M, topK}};
    gert::StorageShape scalesShape = {{M, topK}, {M, topK}};
    gert::StorageShape xActiveMaskShape = {{M}, {M}};
    gert::StorageShape expertScalesShape = {{M, topK}, {M, topK}};
    gert::StorageShape weight1Shape = {{expertPerRank, K, N}, {expertPerRank, K, N}};
    gert::StorageShape weightScales1Shape = {{expertPerRank, N, K / 64}, {expertPerRank, N, K / 64}};
    gert::StorageShape weight2Shape = {{expertPerRank, N / 2, K}, {expertPerRank, N / 2, K}};
    gert::StorageShape weightScales2Shape = {{expertPerRank, K, N / 2 / 64}, {expertPerRank, K, N / 2 / 64}};
    gert::StorageShape yShape = {{M, K}, {M, K}};
    
    uint64_t cclBufferSize = 1024 * 1024 * 50;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(10, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&contextShape, &xShape, &expertIdsShape, &scalesShape, 
                                   &xActiveMaskShape, &expertScalesShape, &weight1Shape, 
                                   &weightScales1Shape, &weight2Shape, &weightScales2Shape})
                      .OutputShapes({&yShape})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"group_ep", ge::AnyValue::CreateFrom<string>("ep_group")},
                          {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(EP)},
                          {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(epRankId)},
                          {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(expertPerRank * EP)},
                          {"ccl_buffer_size", ge::AnyValue::CreateFrom<int64_t>(cclBufferSize)},
                          {"max_recv_token_num", ge::AnyValue::CreateFrom<int64_t>(M * topK)},
                          {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(1)},
                          {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"dispatch_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"dispatch_quant_out_type", ge::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
                          {"combine_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"comm_alg", ge::AnyValue::CreateFrom<string>("")},
                          {"global_bs", ge::AnyValue::CreateFrom<int64_t>(M)}
                      })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    MegaMoeTilingData* tilingData = tiling_context->GetTilingData<MegaMoeTilingData>();
    ASSERT_NE(tilingData, nullptr);
    
    EXPECT_EQ(tilingData->m, M);
    EXPECT_EQ(tilingData->k, K);
    EXPECT_EQ(tilingData->n, N);
    EXPECT_EQ(tilingData->topK, topK);
    EXPECT_EQ(tilingData->expertPerRank, expertPerRank);
    EXPECT_EQ(tilingData->EP, EP);
    EXPECT_EQ(tilingData->rank, epRankId);
    EXPECT_EQ(tilingData->m0, NUM_256);
    EXPECT_EQ(tilingData->k0, NUM_384);
    EXPECT_EQ(tilingData->n0, NUM_256);
}

TEST_F(MegaMoeTilingTest, test_tiling_ep_2cards) {
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 0,
            "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": false,
            "Intrinsic_data_move_l12ub": true,
            "Intrinsic_data_move_l0c2ub": true,
            "Intrinsic_data_move_out2l1_nd2nz": false,
            "UB_SIZE": 253952,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
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
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096 * 10);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    uint32_t M = 128;
    uint32_t K = 512;
    uint32_t N = 256;
    uint32_t topK = 2;
    uint32_t expertPerRank = 2;
    uint32_t EP = 2;
    uint32_t epRankId = 0;

    gert::StorageShape contextShape = {{EP * 1024}, {EP * 1024}};
    gert::StorageShape xShape = {{M, K}, {M, K}};
    gert::StorageShape expertIdsShape = {{M, topK}, {M, topK}};
    gert::StorageShape scalesShape = {{M, topK}, {M, topK}};
    gert::StorageShape xActiveMaskShape = {{M}, {M}};
    gert::StorageShape expertScalesShape = {{M, topK}, {M, topK}};
    gert::StorageShape weight1Shape = {{expertPerRank, K, N}, {expertPerRank, K, N}};
    gert::StorageShape weightScales1Shape = {{expertPerRank, N, K / 64}, {expertPerRank, N, K / 64}};
    gert::StorageShape weight2Shape = {{expertPerRank, N / 2, K}, {expertPerRank, N / 2, K}};
    gert::StorageShape weightScales2Shape = {{expertPerRank, K, N / 2 / 64}, {expertPerRank, K, N / 2 / 64}};
    gert::StorageShape yShape = {{M, K}, {M, K}};
    
    uint64_t cclBufferSize = 1024 * 1024 * 50;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(10, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&contextShape, &xShape, &expertIdsShape, &scalesShape, 
                                   &xActiveMaskShape, &expertScalesShape, &weight1Shape, 
                                   &weightScales1Shape, &weight2Shape, &weightScales2Shape})
                      .OutputShapes({&yShape})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"group_ep", ge::AnyValue::CreateFrom<string>("ep_group")},
                          {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(EP)},
                          {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(epRankId)},
                          {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(expertPerRank * EP)},
                          {"ccl_buffer_size", ge::AnyValue::CreateFrom<int64_t>(cclBufferSize)},
                          {"max_recv_token_num", ge::AnyValue::CreateFrom<int64_t>(M * topK * EP)},
                          {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(1)},
                          {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"dispatch_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"dispatch_quant_out_type", ge::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
                          {"combine_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"comm_alg", ge::AnyValue::CreateFrom<string>("")},
                          {"global_bs", ge::AnyValue::CreateFrom<int64_t>(M)}
                      })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    MegaMoeTilingData* tilingData = tiling_context->GetTilingData<MegaMoeTilingData>();
    ASSERT_NE(tilingData, nullptr);
    
    EXPECT_EQ(tilingData->EP, EP);
    EXPECT_EQ(tilingData->rank, epRankId);
    EXPECT_EQ(tilingData->expertPerRank, expertPerRank);
}

TEST_F(MegaMoeTilingTest, test_tiling_topK_1) {
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 0,
            "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": false,
            "Intrinsic_data_move_l12ub": true,
            "Intrinsic_data_move_l0c2ub": true,
            "Intrinsic_data_move_out2l1_nd2nz": false,
            "UB_SIZE": 253952,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
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

    auto param = gert::TilingData::CreateCap(4096 * 10);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    uint32_t M = 128;
    uint32_t K = 512;
    uint32_t N = 256;
    uint32_t topK = 1;
    uint32_t expertPerRank = 4;
    uint32_t EP = 1;

    gert::StorageShape contextShape = {{EP * 1024}, {EP * 1024}};
    gert::StorageShape xShape = {{M, K}, {M, K}};
    gert::StorageShape expertIdsShape = {{M, topK}, {M, topK}};
    gert::StorageShape scalesShape = {{M, topK}, {M, topK}};
    gert::StorageShape xActiveMaskShape = {{M}, {M}};
    gert::StorageShape expertScalesShape = {{M, topK}, {M, topK}};
    gert::StorageShape weight1Shape = {{expertPerRank, K, N}, {expertPerRank, K, N}};
    gert::StorageShape weightScales1Shape = {{expertPerRank, N, K / 64}, {expertPerRank, N, K / 64}};
    gert::StorageShape weight2Shape = {{expertPerRank, N / 2, K}, {expertPerRank, N / 2, K}};
    gert::StorageShape weightScales2Shape = {{expertPerRank, K, N / 2 / 64}, {expertPerRank, K, N / 2 / 64}};
    gert::StorageShape yShape = {{M, K}, {M, K}};
    
    uint64_t cclBufferSize = 1024 * 1024 * 50;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(10, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&contextShape, &xShape, &expertIdsShape, &scalesShape, 
                                   &xActiveMaskShape, &expertScalesShape, &weight1Shape, 
                                   &weightScales1Shape, &weight2Shape, &weightScales2Shape})
                      .OutputShapes({&yShape})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"group_ep", ge::AnyValue::CreateFrom<string>("ep_group")},
                          {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(EP)},
                          {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(expertPerRank * EP)},
                          {"ccl_buffer_size", ge::AnyValue::CreateFrom<int64_t>(cclBufferSize)},
                          {"max_recv_token_num", ge::AnyValue::CreateFrom<int64_t>(M * topK)},
                          {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(1)},
                          {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"dispatch_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"dispatch_quant_out_type", ge::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
                          {"combine_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"comm_alg", ge::AnyValue::CreateFrom<string>("")},
                          {"global_bs", ge::AnyValue::CreateFrom<int64_t>(M)}
                      })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    MegaMoeTilingData* tilingData = tiling_context->GetTilingData<MegaMoeTilingData>();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->topK, topK);
}

TEST_F(MegaMoeTilingTest, test_tiling_fp16_dtype) {
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 0,
            "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": false,
            "Intrinsic_data_move_l12ub": true,
            "Intrinsic_data_move_l0c2ub": true,
            "Intrinsic_data_move_out2l1_nd2nz": false,
            "UB_SIZE": 253952,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
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

    auto param = gert::TilingData::CreateCap(4096 * 10);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    uint32_t M = 128;
    uint32_t K = 512;
    uint32_t N = 256;
    uint32_t topK = 2;
    uint32_t expertPerRank = 4;
    uint32_t EP = 1;

    gert::StorageShape contextShape = {{EP * 1024}, {EP * 1024}};
    gert::StorageShape xShape = {{M, K}, {M, K}};
    gert::StorageShape expertIdsShape = {{M, topK}, {M, topK}};
    gert::StorageShape scalesShape = {{M, topK}, {M, topK}};
    gert::StorageShape xActiveMaskShape = {{M}, {M}};
    gert::StorageShape expertScalesShape = {{M, topK}, {M, topK}};
    gert::StorageShape weight1Shape = {{expertPerRank, K, N}, {expertPerRank, K, N}};
    gert::StorageShape weightScales1Shape = {{expertPerRank, N, K / 64}, {expertPerRank, N, K / 64}};
    gert::StorageShape weight2Shape = {{expertPerRank, N / 2, K}, {expertPerRank, N / 2, K}};
    gert::StorageShape weightScales2Shape = {{expertPerRank, K, N / 2 / 64}, {expertPerRank, K, N / 2 / 64}};
    gert::StorageShape yShape = {{M, K}, {M, K}};
    
    uint64_t cclBufferSize = 1024 * 1024 * 50;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(10, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&contextShape, &xShape, &expertIdsShape, &scalesShape, 
                                   &xActiveMaskShape, &expertScalesShape, &weight1Shape, 
                                   &weightScales1Shape, &weight2Shape, &weightScales2Shape})
                      .OutputShapes({&yShape})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"group_ep", ge::AnyValue::CreateFrom<string>("ep_group")},
                          {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(EP)},
                          {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(expertPerRank * EP)},
                          {"ccl_buffer_size", ge::AnyValue::CreateFrom<int64_t>(cclBufferSize)},
                          {"max_recv_token_num", ge::AnyValue::CreateFrom<int64_t>(M * topK)},
                          {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(1)},
                          {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"dispatch_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"dispatch_quant_out_type", ge::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
                          {"combine_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"comm_alg", ge::AnyValue::CreateFrom<string>("")},
                          {"global_bs", ge::AnyValue::CreateFrom<int64_t>(M)}
                      })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}