/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
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
#include "../../op_host/op_tiling/mega_moe.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace optiling {
class MegaMoeTilingTestArch35 : public testing::Test {
 public:
  static void SetUpTestCase() {
    std::cout << "MegaMoeTilingTestArch35 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MegaMoeTilingTestArch35 TearDown" << std::endl;
  }
};

TEST_F(MegaMoeTilingTestArch35, Arch35_Basic_BF16_H4096_N1024) {
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
    
    auto compile_info = GetCompileInfo(compile_info_string);
    
    gert::StorageShape xShape = {{128, 4096}, {}};
    gert::StorageShape topkIdsShape = {{128, 6}, {}};
    gert::StorageShape topkWeightsShape = {{128, 6}, {}};
    gert::StorageShape weight1Shape = {{4, 1024, 4096}, {}};
    gert::StorageShape weight2Shape = {{4, 4096, 512}, {}};
    gert::StorageShape weightScales1Shape = {{4, 1024, 64, 2}, {}};
    gert::StorageShape weightScales2Shape = {{4, 4096, 8, 2}, {}};
    gert::StorageShape yShape = {{128, 4096}, {}};
    gert::StorageShape expertTokenNumsShape = {{4}, {}};
    
    auto holder = gert::TilingContextFaker()
                     .SetOpType("MegaMoe")
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .InputShapes({&xShape, &topkIdsShape, &topkWeightsShape, &weight1Shape, &weight2Shape,
                                   &weightScales1Shape, &weightScales2Shape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(16)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(24)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(4, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(5, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(6, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(7, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .Build();
    
    auto tiling_func = holder.GetContext<gert::TilingContext>()->GetOpTilingFuncPtr();
    ASSERT_NE(tiling_func, nullptr);
    
    auto ret = tiling_func(holder.GetContext<gert::TilingContext>());
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    
    auto tiling_data = holder.GetContext<gert::TilingContext>()->GetRawTilingData();
    ASSERT_NE(tiling_data, nullptr);
    
    auto *tiling_def = reinterpret_cast<MegaMoeTilingDef *>(tiling_data->GetData());
    ASSERT_NE(tiling_def, nullptr);
    
    EXPECT_EQ(tiling_def->bs, 128);
    EXPECT_EQ(tiling_def->h, 4096);
    EXPECT_EQ(tiling_def->topk, 6);
    EXPECT_EQ(tiling_def->expertPerRank, 4);
    EXPECT_EQ(tiling_def->n, 1024);
}

TEST_F(MegaMoeTilingTestArch35, Arch35_Basic_BF16_H5120_N2048) {
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
    
    auto compile_info = GetCompileInfo(compile_info_string);
    
    gert::StorageShape xShape = {{256, 5120}, {}};
    gert::StorageShape topkIdsShape = {{256, 8}, {}};
    gert::StorageShape topkWeightsShape = {{256, 8}, {}};
    gert::StorageShape weight1Shape = {{8, 2048, 5120}, {}};
    gert::StorageShape weight2Shape = {{8, 5120, 1024}, {}};
    gert::StorageShape weightScales1Shape = {{8, 2048, 80, 2}, {}};
    gert::StorageShape weightScales2Shape = {{8, 5120, 16, 2}, {}};
    gert::StorageShape yShape = {{256, 5120}, {}};
    gert::StorageShape expertTokenNumsShape = {{8}, {}};
    
    auto holder = gert::TilingContextFaker()
                     .SetOpType("MegaMoe")
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 8, 8, 8, 8, 1, 1})
                     .InputShapes({&xShape, &topkIdsShape, &topkWeightsShape, &weight1Shape, &weight2Shape,
                                   &weightScales1Shape, &weightScales2Shape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(64)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(24)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(4, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(5, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(6, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(7, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .Build();
    
    auto tiling_func = holder.GetContext<gert::TilingContext>()->GetOpTilingFuncPtr();
    ASSERT_NE(tiling_func, nullptr);
    
    auto ret = tiling_func(holder.GetContext<gert::TilingContext>());
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeTilingTestArch35, Arch35_Basic_BF16_H7168_N7168) {
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
    
    auto compile_info = GetCompileInfo(compile_info_string);
    
    gert::StorageShape xShape = {{512, 7168}, {}};
    gert::StorageShape topkIdsShape = {{512, 8}, {}};
    gert::StorageShape topkWeightsShape = {{512, 8}, {}};
    gert::StorageShape weight1Shape = {{2, 7168, 7168}, {}};
    gert::StorageShape weight2Shape = {{2, 7168, 3584}, {}};
    gert::StorageShape weightScales1Shape = {{2, 7168, 112, 2}, {}};
    gert::StorageShape weightScales2Shape = {{2, 7168, 56, 2}, {}};
    gert::StorageShape yShape = {{512, 7168}, {}};
    gert::StorageShape expertTokenNumsShape = {{2}, {}};
    
    auto holder = gert::TilingContextFaker()
                     .SetOpType("MegaMoe")
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 2, 2, 2, 2, 1, 1})
                     .InputShapes({&xShape, &topkIdsShape, &topkWeightsShape, &weight1Shape, &weight2Shape,
                                   &weightScales1Shape, &weightScales2Shape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(24)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(4, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(5, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(6, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(7, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .Build();
    
    auto tiling_func = holder.GetContext<gert::TilingContext>()->GetOpTilingFuncPtr();
    ASSERT_NE(tiling_func, nullptr);
    
    auto ret = tiling_func(holder.GetContext<gert::TilingContext>());
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeTilingTestArch35, Arch35_FP8_E5M2) {
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
    
    auto compile_info = GetCompileInfo(compile_info_string);
    
    gert::StorageShape xShape = {{128, 4096}, {}};
    gert::StorageShape topkIdsShape = {{128, 6}, {}};
    gert::StorageShape topkWeightsShape = {{128, 6}, {}};
    gert::StorageShape weight1Shape = {{4, 1024, 4096}, {}};
    gert::StorageShape weight2Shape = {{4, 4096, 512}, {}};
    gert::StorageShape weightScales1Shape = {{4, 1024, 64, 2}, {}};
    gert::StorageShape weightScales2Shape = {{4, 4096, 8, 2}, {}};
    gert::StorageShape yShape = {{128, 4096}, {}};
    gert::StorageShape expertTokenNumsShape = {{4}, {}};
    
    auto holder = gert::TilingContextFaker()
                     .SetOpType("MegaMoe")
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .InputShapes({&xShape, &topkIdsShape, &topkWeightsShape, &weight1Shape, &weight2Shape,
                                   &weightScales1Shape, &weightScales2Shape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(16)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(23)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(4, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(5, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(6, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(7, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .Build();
    
    auto tiling_func = holder.GetContext<gert::TilingContext>()->GetOpTilingFuncPtr();
    ASSERT_NE(tiling_func, nullptr);
    
    auto ret = tiling_func(holder.GetContext<gert::TilingContext>());
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeTilingTestArch35, Arch35_Invalid_H) {
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
    
    auto compile_info = GetCompileInfo(compile_info_string);
    
    gert::StorageShape xShape = {{128, 2048}, {}};
    gert::StorageShape topkIdsShape = {{128, 6}, {}};
    gert::StorageShape topkWeightsShape = {{128, 6}, {}};
    gert::StorageShape weight1Shape = {{4, 1024, 2048}, {}};
    gert::StorageShape weight2Shape = {{4, 2048, 512}, {}};
    gert::StorageShape weightScales1Shape = {{4, 1024, 32, 2}, {}};
    gert::StorageShape weightScales2Shape = {{4, 2048, 8, 2}, {}};
    gert::StorageShape yShape = {{128, 2048}, {}};
    gert::StorageShape expertTokenNumsShape = {{4}, {}};
    
    auto holder = gert::TilingContextFaker()
                     .SetOpType("MegaMoe")
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .InputShapes({&xShape, &topkIdsShape, &topkWeightsShape, &weight1Shape, &weight2Shape,
                                   &weightScales1Shape, &weightScales2Shape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(16)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(24)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(4, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(5, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(6, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(7, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .Build();
    
    auto tiling_func = holder.GetContext<gert::TilingContext>()->GetOpTilingFuncPtr();
    ASSERT_NE(tiling_func, nullptr);
    
    auto ret = tiling_func(holder.GetContext<gert::TilingContext>());
    EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeTilingTestArch35, Arch35_Invalid_N) {
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
    
    auto compile_info = GetCompileInfo(compile_info_string);
    
    gert::StorageShape xShape = {{128, 4096}, {}};
    gert::StorageShape topkIdsShape = {{128, 6}, {}};
    gert::StorageShape topkWeightsShape = {{128, 6}, {}};
    gert::StorageShape weight1Shape = {{4, 512, 4096}, {}};
    gert::StorageShape weight2Shape = {{4, 4096, 256}, {}};
    gert::StorageShape weightScales1Shape = {{4, 512, 64, 2}, {}};
    gert::StorageShape weightScales2Shape = {{4, 4096, 4, 2}, {}};
    gert::StorageShape yShape = {{128, 4096}, {}};
    gert::StorageShape expertTokenNumsShape = {{4}, {}};
    
    auto holder = gert::TilingContextFaker()
                     .SetOpType("MegaMoe")
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .InputShapes({&xShape, &topkIdsShape, &topkWeightsShape, &weight1Shape, &weight2Shape,
                                   &weightScales1Shape, &weightScales2Shape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(16)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(24)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(4, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(5, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(6, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(7, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .Build();
    
    auto tiling_func = holder.GetContext<gert::TilingContext>()->GetOpTilingFuncPtr();
    ASSERT_NE(tiling_func, nullptr);
    
    auto ret = tiling_func(holder.GetContext<gert::TilingContext>());
    EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeTilingTestArch35, Arch35_Invalid_TopK) {
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
    
    auto compile_info = GetCompileInfo(compile_info_string);
    
    gert::StorageShape xShape = {{128, 4096}, {}};
    gert::StorageShape topkIdsShape = {{128, 10}, {}};
    gert::StorageShape topkWeightsShape = {{128, 10}, {}};
    gert::StorageShape weight1Shape = {{4, 1024, 4096}, {}};
    gert::StorageShape weight2Shape = {{4, 4096, 512}, {}};
    gert::StorageShape weightScales1Shape = {{4, 1024, 64, 2}, {}};
    gert::StorageShape weightScales2Shape = {{4, 4096, 8, 2}, {}};
    gert::StorageShape yShape = {{128, 4096}, {}};
    gert::StorageShape expertTokenNumsShape = {{4}, {}};
    
    auto holder = gert::TilingContextFaker()
                     .SetOpType("MegaMoe")
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .InputShapes({&xShape, &topkIdsShape, &topkWeightsShape, &weight1Shape, &weight2Shape,
                                   &weightScales1Shape, &weightScales2Shape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(16)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(24)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(4, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(5, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(6, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(7, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .Build();
    
    auto tiling_func = holder.GetContext<gert::TilingContext>()->GetOpTilingFuncPtr();
    ASSERT_NE(tiling_func, nullptr);
    
    auto ret = tiling_func(holder.GetContext<gert::TilingContext>());
    EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeTilingTestArch35, Arch35_Invalid_BS) {
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
    
    auto compile_info = GetCompileInfo(compile_info_string);
    
    gert::StorageShape xShape = {{1024, 4096}, {}};
    gert::StorageShape topkIdsShape = {{1024, 6}, {}};
    gert::StorageShape topkWeightsShape = {{1024, 6}, {}};
    gert::StorageShape weight1Shape = {{4, 1024, 4096}, {}};
    gert::StorageShape weight2Shape = {{4, 4096, 512}, {}};
    gert::StorageShape weightScales1Shape = {{4, 1024, 64, 2}, {}};
    gert::StorageShape weightScales2Shape = {{4, 4096, 8, 2}, {}};
    gert::StorageShape yShape = {{1024, 4096}, {}};
    gert::StorageShape expertTokenNumsShape = {{4}, {}};
    
    auto holder = gert::TilingContextFaker()
                     .SetOpType("MegaMoe")
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .InputShapes({&xShape, &topkIdsShape, &topkWeightsShape, &weight1Shape, &weight2Shape,
                                   &weightScales1Shape, &weightScales2Shape})
                     .OutputShapes({&yShape, &expertTokenNumsShape})
                     .NodeAttrs({{"moe_expert_num", Ops::NN::AnyValue::CreateFrom<int64_t>(16)},
                                 {"ep_world_size", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"ccl_buffer_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2097152)},
                                 {"max_recv_token_num", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"dispatch_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                 {"dispatch_quant_out_type", Ops::NN::AnyValue::CreateFrom<int64_t>(24)},
                                 {"combine_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                 {"comm_alg", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                                 {"global_bs", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(4, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(5, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(6, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(7, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .Build();
    
    auto tiling_func = holder.GetContext<gert::TilingContext>()->GetOpTilingFuncPtr();
    ASSERT_NE(tiling_func, nullptr);
    
    auto ret = tiling_func(holder.GetContext<gert::TilingContext>());
    EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}
}