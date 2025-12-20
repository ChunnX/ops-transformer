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
#include <thread>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace ge;

class MatmulAllReduceTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduceTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduceTiling TearDown" << std::endl;
    }
};

//===============================================Ascend310P====================================================
TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_1_310P)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_1_310P_weightnz_antiquant)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 768, 16, 32}, {48, 768, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_1)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{5512, 11136}, {5512, 11136}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{11136, 4096}, {11136, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{5512, 4096}, {5512, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2000000UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_1_k_zero)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{5512, 0}, {5512, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{0, 4096}, {0, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{5512, 4096}, {5512, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2100000UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_2)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8198, 2536}, {8198, 2536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2536, 12288}, {2536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8198, 12288}, {8198, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2000001UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_3)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{15356, 5120}, {15356, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 8192}, {5120, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{15356, 8192}, {15356, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2000001UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_4)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 30720}, {8192, 30720}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{30720, 2048}, {30720, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 2048}, {8192, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2000000UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_5)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{128, 40960}, {128, 40960}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{40960, 128}, {40960, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{128, 128}, {128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2000000UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_6)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 3072}, {8192, 3072}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3072, 8192}, {3072, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 8192}, {8192, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2000001UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_Weight_NZ)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{5512, 11136}, {5512, 11136}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{11136, 4096}, {11136, 4096}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{5512, 4096}, {5512, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 67536UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_a8w8_310p)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            {{{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 1UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_empty_k_310p_nz)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{256, 0}, {256, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{0, 8192}, {0, 8192}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2100000UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

namespace {
template <typename T>
auto build_from(const T& value){
    return Ops::Transformer::AnyValue::CreateFrom<T>(value);
}

// 定义用例信息结构体
struct MatmulAllReduceTilingTestParam {
    // 平台信息
    uint64_t inputTotalNum;
    string case_name;
    string compile_info;
    string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;
    uint64_t tilingDataSize;

    // 输入信息shape
    std::initializer_list<int64_t> x1_shape;
    std::initializer_list<int64_t> x2_shape;
    std::initializer_list<int64_t> bias_shape;
    std::initializer_list<int64_t> x3_shape;
    std::initializer_list<int64_t> antiquant_scale_shape;
    std::initializer_list<int64_t> antiquant_offset_shape;
    std::initializer_list<int64_t> dequant_scale_shape;
    std::initializer_list<int64_t> pertoken_scale_shape;
    std::initializer_list<int64_t> comm_quant_scale_1_shape;
    std::initializer_list<int64_t> comm_quant_scale_2_shape;
    std::initializer_list<int64_t> output_shape; // 输出信息

    // 输入信息类型
    ge::DataType x1_dtype;
    ge::DataType x2_dtype;
    ge::DataType bias_dtype;
    ge::DataType x3_dtype;
    ge::DataType antiquant_scale_dtype;
    ge::DataType antiquant_offset_dtype;
    ge::DataType dequant_scale_dtype;
    ge::DataType pertoken_scale_dtype;
    ge::DataType comm_quant_scale_1_dtype;
    ge::DataType comm_quant_scale_2_dtype;
    ge::DataType output_dtype; // 输出信息

    bool is_trans_a;
    bool is_trans_b;

    // 结果
    uint64_t expectTilingKey;
};

gert::StorageShape make_shape(const std::initializer_list<int64_t>& input_shape){
    if (input_shape.size() == 0){
        return gert::StorageShape{};
    }
    return gert::StorageShape{input_shape, input_shape};
}

void TestOneParamCase(const MatmulAllReduceTilingTestParam& param){
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;

    // 存取用户输入的用例信息
    std::vector<pair<std::initializer_list<int64_t>, ge::DataType>> shapeDtypeList = {
    {param.x1_shape, param.x1_dtype}, 
    {param.x2_shape, param.x2_dtype}, 
    {param.bias_shape, param.bias_dtype}, 
    {param.x3_shape, param.x3_dtype}, 
    {param.antiquant_scale_shape, param.antiquant_scale_dtype}, 
    {param.antiquant_offset_shape, param.antiquant_offset_dtype}, 
    {param.dequant_scale_shape, param.dequant_scale_dtype}, 
    {param.pertoken_scale_shape, param.pertoken_scale_dtype}, 
    {param.comm_quant_scale_1_shape, param.comm_quant_scale_1_dtype}, 
    {param.comm_quant_scale_2_shape, param.comm_quant_scale_2_dtype}
    };

    // 按需提取后传入构造
    std::vector<gert::TilingContextPara::TensorDescription> inputList;
    for (int i = 0; i < param.inputTotalNum; i++){
        inputList.push_back({make_shape(shapeDtypeList[i].first), shapeDtypeList[i].second, ge::FORMAT_ND});
    }

    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputList,
        {
            {{param.output_shape, param.output_shape}, param.output_dtype, ge::FORMAT_ND},
        },
        {
            {"group", build_from<std::string>("group")},
            {"reduce_op", build_from<std::string>("sum")},
            {"is_trans_a", build_from<bool>(param.is_trans_a)},
            {"is_trans_b", build_from<bool>(param.is_trans_b)},
            {"comm_turn", build_from<int64_t>(0)},
            {"antiquant_group_size", build_from<int64_t>(0)},
            {"group_size", build_from<int64_t>(0)},
            {"y_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)}
        },
        &compileInfo, param.soc_version, param.compile_info, param.tilingDataSize);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, param.expectTilingKey);
}

const string COMPILE_INFO = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20, "socVersion": "Ascend910B"}})";

// 用例列表集
MatmulAllReduceTilingTestParam cases_params[] = {
{4,"matmul_all_reduce_test_tiling_float16_empty_k",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 0},{0, 8192},{},{},{},{},{},{},{},{},{256, 8192},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000000009UL},
{4,"matmul_all_reduce_test_tiling_bfloat16",COMPILE_INFO,"Ascend910B",20,196608,4096,{8192, 1536},{1536, 12288},{12288},{},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_support_3_dim",COMPILE_INFO,"Ascend910B",20,196608,4096,{1, 8192, 1536},{1536, 12288},{12288},{},{},{},{},{},{},{},{1, 8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_5",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 1536},{1536, 8192},{},{},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_4",COMPILE_INFO,"Ascend910B",20,196608,4096,{1024, 1536},{1536, 8192},{},{},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_3",COMPILE_INFO,"Ascend910B",20,196608,4096,{128, 1536},{1536, 8192},{},{},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_2",COMPILE_INFO,"Ascend910B",20,196608,4096,{8192, 1536},{1536, 12288},{},{},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,true,true,10000000000000001100UL},
{4,"matmul_all_reduce_test_mcut_float16_910B_win2win",COMPILE_INFO,"Ascend910B",20,196608,4096,{12290, 15360},{15360, 12288},{},{},{},{},{},{},{},{},{12290, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_big_K",COMPILE_INFO,"Ascend910B",20,196608,4096,{8192, 0xFFFFFFF},{0xFFFFFFF, 12288},{},{8192, 12288},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,65536UL},
{4,"matmul_all_reduce_test_tiling_big_N",COMPILE_INFO,"Ascend910B",20,196608,4096,{8192, 1536},{1536, 0xFFFFFFF},{},{8192, 0xFFFFFFF},{},{},{},{},{},{},{8192, 0xFFFFFFF},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,65536UL},
{4,"matmul_all_reduce_test_tiling_float16_unaligned",COMPILE_INFO,"Ascend910B",20,196608,4096,{1, 65536},{65536, 128},{},{},{},{},{},{},{},{},{1, 128},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_1_cube",COMPILE_INFO,"Ascend910B",20,196608,4096,{8192, 1536},{1536, 12288},{},{},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_1",COMPILE_INFO,"Ascend910B",20,196608,4096,{8192, 1536},{1536, 12288},{},{8192, 12288},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,65536UL},
{8,"matmul_all_reduce_test_tiling_int8_bf16",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 1536},{1536, 8192},{},{},{},{},{8192},{},{},{},{256, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_BF16,false,false,0UL},
{8,"matmul_all_reduce_test_tiling_int8_1",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 1536},{1536, 8192},{},{},{},{},{8192},{},{},{},{256, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_UINT64,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,0UL},
{9,"matmul_all_reduce_test_tiling_int8_2",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 1536},{1536, 8192},{},{},{},{},{1},{256},{},{},{256, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,16UL},
{10,"matmul_all_reduce_test_tiling_a8w8_910b_mCut_2",COMPILE_INFO,"Ascend910B",20,196608,4096,{4096, 1024},{1024, 8192},{},{},{},{},{8192},{},{8192},{8192},{4096, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_INT32,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_UINT64,ge::DT_UINT64,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,false,false,10UL},
{10,"matmul_all_reduce_test_tiling_a8w8_910b_mCut_1",COMPILE_INFO,"Ascend910B",20,196608,4096,{4096, 6272},{6272, 8192},{},{},{},{},{8192},{},{8192},{8192},{4096, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_INT32,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_UINT64,ge::DT_UINT64,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,false,false,10UL},
{10,"matmul_all_reduce_test_tiling_a8w8_scaleDimNum2_910b",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 1536},{1536, 8192},{},{},{},{},{1,8192},{},{1,8192},{1,8192},{256, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_INT32,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_UINT64,ge::DT_UINT64,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,false,false,10UL},
{10,"matmul_all_reduce_test_tiling_a8w8_910b",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 1536},{1536, 8192},{},{},{},{},{8192},{},{8192},{8192},{256, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_INT32,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_UINT64,ge::DT_UINT64,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,false,false,10UL},
};

// 多线程执行用例集
static void ThreadFunc(const MatmulAllReduceTilingTestParam* params, size_t testcase_num, size_t thread_idx, size_t thread_num){
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num){
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const MatmulAllReduceTilingTestParam* params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx){
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx){
        threads[idx].join();
    }
}

TEST_F(MatmulAllReduceTiling, general_cases_params_multi_thread){
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    TestMultiThread(cases_params, sizeof(cases_params) / sizeof(MatmulAllReduceTilingTestParam), 1);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
}

}