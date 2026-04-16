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
#include <gtest/gtest.h>
#include "../../../../op_kernel/all_gather_matmul_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace AllGatherMatmulUT {

class AllGatherMatmulArch32TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmulArch32TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmulArch32TilingTest TearDown" << std::endl;
    }
};

TEST_F(AllGatherMatmulArch32TilingTest, Float16Test1)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{512, 12288}, {512, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288, 3904}, {12288, 3904}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{4096, 3904}, {4096, 3904}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 12288}, {4096, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulArch32TilingTest, Float16Test2)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{16384, 1536}, {16384, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 4096}, {16384, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulArch32TilingTest, Float16Test3)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{327680, 15360}, {327680, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{15360, 10240}, {15360, 10240}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{2621440, 10240}, {2621440, 10240}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2621440, 15360}, {2621440, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulArch32TilingTest, Bfloat16)
{
    // tilingFunc simulate
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{16384, 1536}, {16384, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 4096}, {16384, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 7UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulArch32TilingTest, Float16TestL2cache)
{
    // tilingFunc simulate
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{8192, 5120}, {8192, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 12288}, {5120, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{65536, 12288}, {65536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{65536, 5120}, {65536, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 7UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulArch32TilingTest, N0)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{1024, 256}, {1024, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 0}, {256, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{8192, 0}, {8192, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 256}, {8192, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 测试场景：极小M + 大K + 中等N，通信远超计算2倍(strongTpBound_) + reduceAlignLen
// reduceAlignLen = N>2048 && M<=512，strongTpBound_需要通信时间 >> 计算时间
// 覆盖 reduceAlignLen 对齐逻辑
TEST_F(AllGatherMatmulArch32TilingTest, TestReduceAlignLenWithStrongTpBound)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{128, 8192}, {128, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 3072}, {8192, 3072}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{1024, 3072}, {1024, 3072}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 测试场景：超大K值(>32768)，触发 noCutFlag_ = false
// 覆盖超大K值时禁用noCutFlag的切分逻辑
TEST_F(AllGatherMatmulArch32TilingTest, TestHugeKValue)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{1024, 40000}, {1024, 40000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{40000, 2048}, {40000, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{8192, 2048}, {8192, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 40000}, {8192, 40000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 测试场景：大K大N，触发 bwGrowthByShape 分支
// 覆盖带宽按shape增长的切分策略
TEST_F(AllGatherMatmulArch32TilingTest, TestBwGrowthByLargeKN)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 6000}, {8192, 6000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{32768, 6000}, {32768, 6000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32768, 8192}, {32768, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 测试场景：极小M+大K，通信强bound(strongTpBound_)，短块前置
// 覆盖通信bound场景下短块前置的切分策略
TEST_F(AllGatherMatmulArch32TilingTest, TestStrongTpBoundCommBoundFront)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{64, 16384}, {64, 16384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 2048}, {16384, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 16384}, {512, 16384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}


// 测试场景：大M(>4096)+大K+大N，进入 bwGrowthByShape 分支且 !medianMFlag
// 覆盖大M场景下非medianM的带宽增长切分策略
TEST_F(AllGatherMatmulArch32TilingTest, TestBwGrowthByShapeLargeM)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{8192, 8192}, {8192, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 6000}, {8192, 6000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{65536, 6000}, {65536, 6000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{65536, 8192}, {65536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 测试场景：rankNum=2，超大K(>32768)强制noCutFlag_=false，计算Bound，小N
// 覆盖计算bound下的ShortAtEndCalcBoundBalancing和小维度的smallDimAlignUp对齐
TEST_F(AllGatherMatmulArch32TilingTest, TestRankDim2CalcBoundSmallN)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{4096, 40000}, {4096, 40000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{40000, 1024}, {40000, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{8192, 1024}, {8192, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 40000}, {8192, 40000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}


// 测试场景：topoType=COMM_MESH，isA3=0，走SOC910_B分支
// 覆盖 A2芯片(MESH拓扑)的AllGatherPlusMM构造分支
TEST_F(AllGatherMatmulArch32TilingTest, TestSoc910BMeshTopo)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{16384, 1536}, {16384, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 4096}, {16384, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}, {"topoType", 1}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}


// 测试场景：isStorageGather=false(gather_out为空)，gather_index=0
// 覆盖 gatherIndex==0 且 isStorageGather==false 时的gatherLen计算逻辑
TEST_F(AllGatherMatmulArch32TilingTest, TestStorageGatherFalseGatherIdx0)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{16384, 1536}, {16384, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 测试场景：gather_index=1，参数校验应失败
TEST_F(AllGatherMatmulArch32TilingTest, TestInvalidGatherIndex1)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{16384, 1536}, {16384, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}


// 测试场景：B矩阵存储为 [N, K] 格式，导致 A.dim1(K) != B.dim0(N)
// 覆盖 B矩阵shape不匹配时的N值修正逻辑
TEST_F(AllGatherMatmulArch32TilingTest, TestBShapeMismatchTransB)
{
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 4096}, {1536, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{16384, 1536}, {16384, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

} // AllGatherMatmulUT