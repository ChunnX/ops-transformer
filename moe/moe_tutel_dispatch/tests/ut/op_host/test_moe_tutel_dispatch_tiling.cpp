/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_tutel_dispatch_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>

#include "../../../op_host/moe_tutel_dispatch_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
constexpr uint64_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024UL;
constexpr uint32_t CORE_NUM = 64;
constexpr uint64_t UB_SIZE = 253952;

optiling::MoeTutelDispatchCompileInfo BuildCompileInfo()
{
    optiling::MoeTutelDispatchCompileInfo compileInfo{};
    compileInfo.coreNum = CORE_NUM;
    compileInfo.ubSize = UB_SIZE;
    return compileInfo;
}

gert::TilingContextPara BuildCase(optiling::MoeTutelDispatchCompileInfo &compileInfo, ge::DataType dtype,
                                  int64_t samples, int64_t hidden, int64_t expertNum, int64_t capacity,
                                  ge::DataType indexDtype = ge::DT_INT32)
{
    gert::StorageShape xShape = {{samples, hidden}, {samples, hidden}};
    gert::StorageShape routeShape = {{expertNum, samples}, {expertNum, samples}};
    gert::StorageShape yShape = {{expertNum, capacity, hidden}, {expertNum, capacity, hidden}};

    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {xShape, dtype, ge::FORMAT_ND},
        {routeShape, dtype, ge::FORMAT_ND},
        {routeShape, indexDtype, ge::FORMAT_ND},
        {routeShape, indexDtype, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {yShape, dtype, ge::FORMAT_ND},
    };

    return gert::TilingContextPara("MoeTutelDispatch", inputs, outputs,
                                   {
                                       {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(capacity)},
                                   },
                                   &compileInfo, "Ascend950", compileInfo.coreNum, compileInfo.ubSize);
}
} // namespace

class MoeTutelDispatchTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTutelDispatchTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTutelDispatchTiling TearDown" << std::endl;
    }
};

// 整行驻留 UB：TilingKey 0
TEST_F(MoeTutelDispatchTilingTest, FullRowFloat32)
{
    auto compileInfo = BuildCompileInfo();
    auto para = BuildCase(compileInfo, ge::DT_FLOAT, 1024, 256, 2, 512);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, "", {SYS_WORKSPACE_SIZE});
}

TEST_F(MoeTutelDispatchTilingTest, FullRowFloat16)
{
    auto compileInfo = BuildCompileInfo();
    auto para = BuildCase(compileInfo, ge::DT_FLOAT16, 1024, 256, 2, 512);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, "", {SYS_WORKSPACE_SIZE});
}

TEST_F(MoeTutelDispatchTilingTest, FullRowBFloat16)
{
    auto compileInfo = BuildCompileInfo();
    auto para = BuildCase(compileInfo, ge::DT_BF16, 1024, 256, 2, 512);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, "", {SYS_WORKSPACE_SIZE});
}

// H 非 32B 对齐，仍走整行分支
TEST_F(MoeTutelDispatchTilingTest, FullRowUnalignedHidden)
{
    auto compileInfo = BuildCompileInfo();
    auto para = BuildCase(compileInfo, ge::DT_FLOAT, 37, 17, 3, 11);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, "", {SYS_WORKSPACE_SIZE});
}

// 单行放不进 UB：TilingKey 1
TEST_F(MoeTutelDispatchTilingTest, SplitHugeHidden)
{
    auto compileInfo = BuildCompileInfo();
    auto para = BuildCase(compileInfo, ge::DT_FLOAT, 8, 131072, 2, 4);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 1, "", {SYS_WORKSPACE_SIZE});
}

// capacity = 0 -> y 为空张量：TilingKey 2
TEST_F(MoeTutelDispatchTilingTest, EmptyCapacityZero)
{
    auto compileInfo = BuildCompileInfo();
    auto para = BuildCase(compileInfo, ge::DT_FLOAT, 32, 64, 2, 0);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 2, "", {SYS_WORKSPACE_SIZE});
}

// H = 0 -> y 为空张量：TilingKey 2
TEST_F(MoeTutelDispatchTilingTest, EmptyHiddenZero)
{
    auto compileInfo = BuildCompileInfo();
    auto para = BuildCase(compileInfo, ge::DT_FLOAT, 32, 0, 2, 8);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 2, "", {SYS_WORKSPACE_SIZE});
}

// B = 0 -> y 为空张量：TilingKey 2
TEST_F(MoeTutelDispatchTilingTest, EmptyExpertZero)
{
    auto compileInfo = BuildCompileInfo();
    auto para = BuildCase(compileInfo, ge::DT_FLOAT, 32, 64, 0, 8);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 2, "", {SYS_WORKSPACE_SIZE});
}

// S = 0 -> 无路由但 y 非空，仍走整行分支只做清零
TEST_F(MoeTutelDispatchTilingTest, EmptySamplesStillInitsOutput)
{
    auto compileInfo = BuildCompileInfo();
    auto para = BuildCase(compileInfo, ge::DT_FLOAT, 0, 64, 2, 8);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, "", {SYS_WORKSPACE_SIZE});
}

// capacity 为负：拒收
TEST_F(MoeTutelDispatchTilingTest, NegativeCapacityRejected)
{
    auto compileInfo = BuildCompileInfo();
    gert::StorageShape xShape = {{8, 16}, {8, 16}};
    gert::StorageShape routeShape = {{2, 8}, {2, 8}};
    gert::StorageShape yShape = {{2, 4, 16}, {2, 4, 16}};
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {xShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {routeShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {routeShape, ge::DT_INT32, ge::FORMAT_ND},
        {routeShape, ge::DT_INT32, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}};
    gert::TilingContextPara para("MoeTutelDispatch", inputs, outputs,
                                 {
                                     {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                 },
                                 &compileInfo, "Ascend950", compileInfo.coreNum, compileInfo.ubSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// indices 为 int64：拒收
TEST_F(MoeTutelDispatchTilingTest, IndexDtypeInt64Rejected)
{
    auto compileInfo = BuildCompileInfo();
    auto para = BuildCase(compileInfo, ge::DT_FLOAT, 8, 16, 2, 4, ge::DT_INT64);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// x 为 int32：拒收
TEST_F(MoeTutelDispatchTilingTest, XDtypeInt32Rejected)
{
    auto compileInfo = BuildCompileInfo();
    auto para = BuildCase(compileInfo, ge::DT_INT32, 8, 16, 2, 4);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// gates 与 indices shape 不一致：拒收
TEST_F(MoeTutelDispatchTilingTest, RouteShapeMismatchRejected)
{
    auto compileInfo = BuildCompileInfo();
    gert::StorageShape xShape = {{8, 16}, {8, 16}};
    gert::StorageShape gatesShape = {{2, 8}, {2, 8}};
    gert::StorageShape badShape = {{3, 8}, {3, 8}};
    gert::StorageShape yShape = {{2, 4, 16}, {2, 4, 16}};
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {xShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {gatesShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {badShape, ge::DT_INT32, ge::FORMAT_ND},
        {gatesShape, ge::DT_INT32, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}};
    gert::TilingContextPara para("MoeTutelDispatch", inputs, outputs,
                                 {
                                     {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                 },
                                 &compileInfo, "Ascend950", compileInfo.coreNum, compileInfo.ubSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// y 的专家维与 gates.dim(0) 不一致：拒收
TEST_F(MoeTutelDispatchTilingTest, OutputExpertDimMismatchRejected)
{
    auto compileInfo = BuildCompileInfo();
    gert::StorageShape xShape = {{8, 16}, {8, 16}};
    gert::StorageShape routeShape = {{2, 8}, {2, 8}};
    gert::StorageShape yShape = {{5, 4, 16}, {5, 4, 16}};
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {xShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {routeShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {routeShape, ge::DT_INT32, ge::FORMAT_ND},
        {routeShape, ge::DT_INT32, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}};
    gert::TilingContextPara para("MoeTutelDispatch", inputs, outputs,
                                 {
                                     {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                 },
                                 &compileInfo, "Ascend950", compileInfo.coreNum, compileInfo.ubSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// x.dim(0) 与 gates.dim(1) 不一致：拒收
TEST_F(MoeTutelDispatchTilingTest, SamplesMismatchRejected)
{
    auto compileInfo = BuildCompileInfo();
    gert::StorageShape xShape = {{9, 16}, {9, 16}};
    gert::StorageShape routeShape = {{2, 8}, {2, 8}};
    gert::StorageShape yShape = {{2, 4, 16}, {2, 4, 16}};
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {xShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {routeShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {routeShape, ge::DT_INT32, ge::FORMAT_ND},
        {routeShape, ge::DT_INT32, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}};
    gert::TilingContextPara para("MoeTutelDispatch", inputs, outputs,
                                 {
                                     {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                 },
                                 &compileInfo, "Ascend950", compileInfo.coreNum, compileInfo.ubSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 负维（动态 shape 的 -1 漏到 tiling）：拒收，不进入后续切分
TEST_F(MoeTutelDispatchTilingTest, NegativeDimRejected)
{
    auto compileInfo = BuildCompileInfo();
    gert::StorageShape xShape = {{-1, 16}, {-1, 16}};
    gert::StorageShape routeShape = {{2, -1}, {2, -1}};
    gert::StorageShape yShape = {{2, 4, 16}, {2, 4, 16}};
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {xShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {routeShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {routeShape, ge::DT_INT32, ge::FORMAT_ND},
        {routeShape, ge::DT_INT32, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}};
    gert::TilingContextPara para("MoeTutelDispatch", inputs, outputs,
                                 {
                                     {"capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                 },
                                 &compileInfo, "Ascend950", compileInfo.coreNum, compileInfo.ubSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
