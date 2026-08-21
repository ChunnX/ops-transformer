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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/ffn_worker_batching_tiling.h"

using namespace ge;
using namespace optiling;

class FfnWorkerBatchingTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FfnWorkerBatchingTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FfnWorkerBatchingTilingTest TearDown" << std::endl;
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// A5(arch35/ascend950) 核心路径 UT
//
// 路由：socVersion="Ascend950" -> DAV_3510 -> IsRegbaseSocVersion()==true
//   -> TilingRegistry 命中 FfnWorkerBatchingTilingArch35（1000 档）。
//
// TilingData：FfnWorkerBatchingArch35TilingData（22×int64），字段顺序见
//   op_kernel/arch35/ffn_worker_batching_arch35_tiling_def.h。
// coreNum：按工作量 (Y+31)/32 与 aivNum 取小，不再使用 A2 的 RECV 固定 32 限核。
// ubSize：GetPlatformInfo 经 ReserveLocalMemory(32K) 后 GetCoreMemSize(UB) 的返回值。
// ---------------------------------------------------------------------------------------------------------------------

namespace {
// 构造 8 输出 desc（shape 仅供 faker 建 tensor，tiling 不读输出 shape，dtype 按 def 顺序）。
std::vector<gert::TilingContextPara::TensorDescription> MakeA5OutputDesc()
{
    static gert::StorageShape y_shape = {{1152, 4096}, {1152, 4096}};
    static gert::StorageShape group_list_shape = {{8, 2}, {8, 2}};
    static gert::StorageShape idx_shape = {{1152}, {1152}};
    static gert::StorageShape scale_shape = {{1152}, {1152}};
    static gert::StorageShape actual_token_num_shape = {{1}, {1}};
    return {
        {y_shape, ge::DT_INT8, ge::FORMAT_ND},                 // y
        {group_list_shape, ge::DT_INT64, ge::FORMAT_ND},       // group_list
        {idx_shape, ge::DT_INT32, ge::FORMAT_ND},              // session_ids
        {idx_shape, ge::DT_INT32, ge::FORMAT_ND},              // micro_batch_ids
        {idx_shape, ge::DT_INT32, ge::FORMAT_ND},              // token_ids
        {idx_shape, ge::DT_INT32, ge::FORMAT_ND},              // expert_offsets
        {scale_shape, ge::DT_FLOAT, ge::FORMAT_ND},            // dynamic_scale
        {actual_token_num_shape, ge::DT_INT64, ge::FORMAT_ND}, // actual_token_num
    };
}

constexpr uint64_t kArch35Ub248K = 253952; // GetCoreMemSize 注入值（扣 32K 预留前），实际 ubSize_=221184
constexpr uint64_t kAivNum64 = 64;

// FfnWorkerBatchingArch35TilingData 字段下标（与 tiling_def.h 一致）
constexpr size_t kIdxY = 0;
constexpr size_t kIdxCoreNum = 4;
constexpr size_t kIdxUbSize = 5;
constexpr size_t kIdxFlatElements = 6;
constexpr size_t kIdxMergeOneLoop = 13;
constexpr size_t kIdxExpertNum = 3;
constexpr size_t kIdxGlRows = 15;

std::vector<gert::TilingContextPara::OpAttr> MakeArch35Attrs(int64_t tokenDtype, int64_t needSchedule, int64_t A,
                                                             int64_t BS, int64_t K, int64_t H, int64_t expertNum = 8)
{
    return {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertNum)},
            {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({A, BS, K, H})},
            {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenDtype)},
            {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(needSchedule)},
            {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}};
}

std::string TilingDataToStr(const TilingInfo &info)
{
    std::string s;
    const int64_t *d = reinterpret_cast<const int64_t *>(info.tilingData.get());
    for (size_t i = 0; i < info.tilingDataSize / sizeof(int64_t); ++i) {
        s += std::to_string(d[i]);
        s += " ";
    }
    return s;
}
} // namespace

// P0｜Ascend950 + need_schedule=0(NORM) -> TilingKey=100；coreNum=min(64,(1152+31)/32)=36。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_arch35_ascend950_norm_key100)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching",
        {{schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950");

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, info));
    std::cout << "[arch35][NORM] tilingKey=" << info.tilingKey << " blockDim=" << info.blockNum
              << " workspace=" << (info.workspaceSizes.empty() ? -1 : info.workspaceSizes[0])
              << " tilingData=" << TilingDataToStr(info) << std::endl;

    // faker 默认 UB=262144，扣 32K 后 ubSize=229376；Y=1152 -> coreNum=36
    std::string expectTilingData =
        "1152 4096 0 8 36 229376 1152 16 36 32 64 1000000 3 3552 1152 8 128 1280 3584 5888 6176 7328 ";
    std::vector<size_t> expectWorkspaces = {16811136};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 100, expectTilingData, expectWorkspaces);
}

// P0｜Ascend950 + need_schedule=1(RECV) -> TilingKey=101；同 shape 下 coreNum 同样按工作量=36。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_arch35_ascend950_recv_key101)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching", {{schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950");

    std::string expectTilingData =
        "1152 4096 0 8 36 229376 1152 16 36 32 64 1000000 3 3552 1152 8 128 1280 3584 5888 6176 7328 ";
    std::vector<size_t> expectWorkspaces = {16811136};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 101, expectTilingData, expectWorkspaces);
}

// 规格：BF16(token_dtype=1) + NORM
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_arch35_ascend950_bf16_norm)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching", {{schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950");

    std::string expectTilingData =
        "1152 4096 1 8 36 229376 1152 16 36 32 64 1000000 3 3552 1152 8 128 1280 3584 5888 6176 7328 ";
    std::vector<size_t> expectWorkspaces = {16811136};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 100, expectTilingData, expectWorkspaces);
}

// 规格：dynamic_quant_int8(token_dtype=2) + RECV
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_arch35_ascend950_int8_recv)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching", {{schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950");

    std::string expectTilingData =
        "1152 4096 2 8 36 229376 1152 16 36 32 64 1000000 3 3552 1152 8 128 1280 3584 5888 6176 7328 ";
    std::vector<size_t> expectWorkspaces = {16811136};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 101, expectTilingData, expectWorkspaces);
}

// A1｜248KB UB 注入：扣 32K 后 ubSize=221184，coreNum 仍按工作量=36。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_norm_key100_ub248k_singlecore)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 16, /*BS*/ 8, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const int64_t *d = reinterpret_cast<const int64_t *>(info.tilingData.get());
    std::cout << "[A1][NORM][ub248k] tilingKey=" << info.tilingKey << " blockDim=" << info.blockNum << " Y=" << d[kIdxY]
              << " coreNum=" << d[kIdxCoreNum] << " ubSize=" << d[kIdxUbSize] << std::endl;
    EXPECT_EQ(info.tilingKey, 100);
    EXPECT_EQ(info.blockNum, 36u);
    EXPECT_EQ(d[kIdxUbSize], 221184);
    EXPECT_EQ(d[kIdxCoreNum], 36);

    std::string expectTilingData =
        "1152 4096 0 8 36 221184 1152 16 36 32 64 1000000 3 3424 1152 8 128 1280 3584 5888 6176 7328 ";
    std::vector<size_t> expectWorkspaces = {16811136};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 100, expectTilingData, expectWorkspaces);
}

// A2｜RECV + 248KB UB：TilingKey=101，coreNum 同样=36（工作量口径，非固定 32 限核）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_recv_key101_ub248k_corelimit)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 1, /*A*/ 16, /*BS*/ 8, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);

    std::string expectTilingData =
        "1152 4096 0 8 36 221184 1152 16 36 32 64 1000000 3 3424 1152 8 128 1280 3584 5888 6176 7328 ";
    std::vector<size_t> expectWorkspaces = {16811136};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 101, expectTilingData, expectWorkspaces);
}

// A3｜UB 运行时派生：仅 ubSize 不同时，mergeOneLoopElements（依赖 UB）随之变化。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_sortloopmax_runtime_derived)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};

    gert::TilingContextPara para248("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
                                    MakeArch35Attrs(0, 0, 16, 8, 9, 4096), &compileInfo, "Ascend950", kAivNum64,
                                    253952);
    gert::TilingContextPara para256("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
                                    MakeArch35Attrs(0, 0, 16, 8, 9, 4096), &compileInfo, "Ascend950", kAivNum64,
                                    262144);

    TilingInfo info248;
    TilingInfo info256;
    ASSERT_TRUE(ExecuteTiling(para248, info248));
    ASSERT_TRUE(ExecuteTiling(para256, info256));
    const int64_t *d248 = reinterpret_cast<const int64_t *>(info248.tilingData.get());
    const int64_t *d256 = reinterpret_cast<const int64_t *>(info256.tilingData.get());
    std::cout << "[A3][runtime-derived] ub248=" << d248[kIdxUbSize] << " mergeLoop=" << d248[kIdxMergeOneLoop]
              << " ub256=" << d256[kIdxUbSize] << " mergeLoop=" << d256[kIdxMergeOneLoop] << std::endl;
    EXPECT_EQ(d248[kIdxUbSize], 221184);
    EXPECT_EQ(d256[kIdxUbSize], 229376);
    EXPECT_EQ(d248[kIdxMergeOneLoop], 3424);
    EXPECT_EQ(d256[kIdxMergeOneLoop], 3552);
    EXPECT_NE(d248[kIdxMergeOneLoop], d256[kIdxMergeOneLoop]);
}

// B1｜大 shape：Y=65536 -> coreNum=min(64,2048)=64，flatElements/workspace 随 Y 增长。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_norm_multicore_largeshape)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 1024, /*BS*/ 1, /*K*/ 64, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const int64_t *d = reinterpret_cast<const int64_t *>(info.tilingData.get());
    std::cout << "[B1][NORM][largeshape] Y=" << d[kIdxY] << " coreNum=" << d[kIdxCoreNum]
              << " flatElements=" << d[kIdxFlatElements] << std::endl;
    EXPECT_EQ(info.tilingKey, 100);
    EXPECT_EQ(d[kIdxY], 65536);
    EXPECT_EQ(d[kIdxCoreNum], 64);
    EXPECT_EQ(d[kIdxFlatElements], 65536);

    std::string expectTilingData =
        "65536 4096 0 8 64 221184 65536 768 64 1024 2048 1000000 3 3392 13824 8 128 65664 196736 327808 328320 393856 ";
    std::vector<size_t> expectWorkspaces = {18614784};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 100, expectTilingData, expectWorkspaces);
}

// C1｜小 Y=4032：coreByWork=126，受 aivNum 封顶 coreNum=64；RECV 不再固定限核 32。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_recv_corelimit_small_y_32)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 1, /*A*/ 63, /*BS*/ 1, /*K*/ 64, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const int64_t *d = reinterpret_cast<const int64_t *>(info.tilingData.get());
    std::cout << "[C1][RECV][Y=4032] coreNum=" << d[kIdxCoreNum] << " blockDim=" << info.blockNum << std::endl;
    EXPECT_EQ(info.tilingKey, 101);
    EXPECT_EQ(d[kIdxCoreNum], 64);
    EXPECT_EQ(info.blockNum, 64u);

    std::string expectTilingData =
        "4032 4096 0 8 64 221184 4032 63 64 63 128 1000000 3 3392 4032 8 128 4160 12352 20544 21056 25088 ";
    std::vector<size_t> expectWorkspaces = {16893696};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 101, expectTilingData, expectWorkspaces);
}

// C2｜Y=4096：coreByWork=128，仍封顶 coreNum=64。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_recv_corelimit_large_y_64)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 1, /*A*/ 64, /*BS*/ 1, /*K*/ 64, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const int64_t *d = reinterpret_cast<const int64_t *>(info.tilingData.get());
    std::cout << "[C2][RECV][Y=4096] coreNum=" << d[kIdxCoreNum] << " blockDim=" << info.blockNum << std::endl;
    EXPECT_EQ(info.tilingKey, 101);
    EXPECT_EQ(d[kIdxCoreNum], 64);
    EXPECT_EQ(info.blockNum, 64u);

    std::string expectTilingData =
        "4096 4096 0 8 64 221184 4096 64 64 64 128 1000000 3 3392 4096 8 128 4224 12416 20608 21120 25216 ";
    std::vector<size_t> expectWorkspaces = {16894464};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 101, expectTilingData, expectWorkspaces);
}

// C3｜大 expert_num=5120：expertNum/glRowsPerLoop 随专家数正确派生（glRows 封顶到 UB 拼装容量）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_recv_large_expert_num_5120)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
                                 MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 1, /*A*/ 16, /*BS*/ 8, /*K*/ 9,
                                                 /*H*/ 4096, /*expertNum*/ 5120),
                                 &compileInfo, "Ascend950", kAivNum64, kArch35Ub248K);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const int64_t *d = reinterpret_cast<const int64_t *>(info.tilingData.get());
    std::cout << "[C3][RECV][expert_num=5120] expertNum=" << d[kIdxExpertNum] << " glRows=" << d[kIdxGlRows]
              << " coreNum=" << d[kIdxCoreNum] << std::endl;
    EXPECT_EQ(info.tilingKey, 101);
    EXPECT_EQ(d[kIdxExpertNum], 5120);
    EXPECT_EQ(d[kIdxGlRows], 864);
    EXPECT_EQ(d[kIdxCoreNum], 36);

    std::string expectTilingData =
        "1152 4096 0 5120 36 221184 1152 16 36 32 64 1000000 3 3424 1152 864 128 1280 3584 5888 6176 7328 ";
    std::vector<size_t> expectWorkspaces = {16811136};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 101, expectTilingData, expectWorkspaces);
}

// =====================================================================================================================
// arch35 硬校验错误路径 UT
// =====================================================================================================================

TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_expert_num_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
                                 MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 16, /*BS*/ 8, /*K*/ 9,
                                                 /*H*/ 4096, /*expertNum*/ 8193),
                                 &compileInfo, "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_topk_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 16, /*BS*/ 8, /*K*/ 65, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_session_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 1025, /*BS*/ 1, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_max_out_shape_len_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_token_dtype_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 3, /*needSchedule*/ 0, /*A*/ 16, /*BS*/ 8, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_need_schedule_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 2, /*A*/ 16, /*BS*/ 8, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
