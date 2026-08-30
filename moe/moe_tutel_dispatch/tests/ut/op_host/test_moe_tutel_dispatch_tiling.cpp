/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_tutel_dispatch_tiling.cpp
 * \brief Ascend 950 Host tiling contract tests. K and caller-provided E are independent.
 */

#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_host/arch35/moe_tutel_dispatch_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
constexpr uint64_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024UL;
constexpr uint32_t CORE_NUM = 64;
constexpr uint64_t UB_SIZE = 253952;

struct CaseParams {
    std::vector<int64_t> xShape{8, 16};
    std::vector<int64_t> gatesShape{2, 8};
    std::vector<int64_t> indicesShape{2, 8};
    std::vector<int64_t> locationsShape{2, 8};
    std::vector<int64_t> yShape{5, 4, 16};
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType gatesDtype = ge::DT_FLOAT;
    ge::DataType indicesDtype = ge::DT_INT32;
    ge::DataType locationsDtype = ge::DT_INT32;
    ge::DataType yDtype = ge::DT_FLOAT;
    ge::Format xFormat = ge::FORMAT_ND;
    ge::Format gatesFormat = ge::FORMAT_ND;
    ge::Format indicesFormat = ge::FORMAT_ND;
    ge::Format locationsFormat = ge::FORMAT_ND;
    ge::Format yFormat = ge::FORMAT_ND;
    bool hasCapacity = true;
    int64_t capacity = 4;
    uint64_t coreNum = CORE_NUM;
    uint64_t ubSize = UB_SIZE;
};

optiling::MoeTutelDispatchCompileInfo BuildCompileInfo(const CaseParams &params)
{
    optiling::MoeTutelDispatchCompileInfo compileInfo{};
    compileInfo.aivCoreLimit = static_cast<int64_t>(params.coreNum);
    compileInfo.ubSize = static_cast<int64_t>(params.ubSize);
    return compileInfo;
}

gert::StorageShape MakeStorageShape(const std::vector<int64_t> &dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::TilingContextPara BuildCase(const CaseParams &params, optiling::MoeTutelDispatchCompileInfo &compileInfo)
{
    const gert::StorageShape xShape = MakeStorageShape(params.xShape);
    const gert::StorageShape gatesShape = MakeStorageShape(params.gatesShape);
    const gert::StorageShape indicesShape = MakeStorageShape(params.indicesShape);
    const gert::StorageShape locationsShape = MakeStorageShape(params.locationsShape);
    const gert::StorageShape yShape = MakeStorageShape(params.yShape);

    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {xShape, params.xDtype, params.xFormat},
        {gatesShape, params.gatesDtype, params.gatesFormat},
        {indicesShape, params.indicesDtype, params.indicesFormat},
        {locationsShape, params.locationsDtype, params.locationsFormat},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {yShape, params.yDtype, params.yFormat},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    if (params.hasCapacity) {
        attrs.emplace_back("capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(params.capacity));
    }

    return gert::TilingContextPara("MoeTutelDispatch", inputs, outputs, attrs, &compileInfo, "Ascend950",
                                   params.coreNum, params.ubSize);
}

void ExpectTiling(const CaseParams &params, ge::graphStatus result, uint64_t key = 0)
{
    auto compileInfo = BuildCompileInfo(params);
    auto testcase = BuildCase(params, compileInfo);
    if (result == ge::GRAPH_SUCCESS) {
        ExecuteTestCase(testcase, result, GET_TPL_TILING_KEY(key), "", {SYS_WORKSPACE_SIZE});
    } else {
        ExecuteTestCase(testcase, result);
    }
}

CaseParams DtypeCase(ge::DataType dtype)
{
    CaseParams params;
    params.xDtype = dtype;
    params.gatesDtype = dtype;
    params.yDtype = dtype;
    return params;
}
} // namespace

class MoeTutelDispatchTiling : public testing::Test {};

TEST_F(MoeTutelDispatchTiling, FullRowFloat32EGreaterThanK)
{
    ExpectTiling(DtypeCase(ge::DT_FLOAT), ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTutelDispatchTiling, FullRowFloat16ELessThanK)
{
    auto params = DtypeCase(ge::DT_FLOAT16);
    params.gatesShape = params.indicesShape = params.locationsShape = {5, 8};
    params.yShape = {2, 4, 16};
    ExpectTiling(params, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTutelDispatchTiling, FullRowBFloat16UnalignedHidden)
{
    auto params = DtypeCase(ge::DT_BF16);
    params.xShape = {37, 17};
    params.gatesShape = params.indicesShape = params.locationsShape = {3, 37};
    params.yShape = {7, 11, 17};
    params.capacity = 11;
    ExpectTiling(params, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTutelDispatchTiling, SplitHugeHidden)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.xShape = {8, 131072};
    params.gatesShape = params.indicesShape = params.locationsShape = {2, 8};
    params.yShape = {5, 4, 131072};
    ExpectTiling(params, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTutelDispatchTiling, EmptyExpertZero)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.yShape = {0, 4, 16};
    ExpectTiling(params, ge::GRAPH_SUCCESS, 2);
}

TEST_F(MoeTutelDispatchTiling, EmptyCapacityZero)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.capacity = 0;
    params.yShape = {5, 0, 16};
    ExpectTiling(params, ge::GRAPH_SUCCESS, 2);
}

TEST_F(MoeTutelDispatchTiling, EmptyHiddenZero)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.xShape = {8, 0};
    params.yShape = {5, 4, 0};
    ExpectTiling(params, ge::GRAPH_SUCCESS, 2);
}

TEST_F(MoeTutelDispatchTiling, ZeroOnlyRouteKZero)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.gatesShape = params.indicesShape = params.locationsShape = {0, 8};
    ExpectTiling(params, ge::GRAPH_SUCCESS, 3);
}

TEST_F(MoeTutelDispatchTiling, ZeroOnlySamplesZero)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.xShape = {0, 16};
    params.gatesShape = params.indicesShape = params.locationsShape = {2, 0};
    ExpectTiling(params, ge::GRAPH_SUCCESS, 3);
}

TEST_F(MoeTutelDispatchTiling, MissingX)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.xShape = {};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, MissingGates)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.gatesShape = {};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, MissingIndices)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.indicesShape = {};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, MissingLocations)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.locationsShape = {};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, MissingCapacity)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.hasCapacity = false;
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, MissingCallerYDesc)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.yShape = {};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, XRank1)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.xShape = {8};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, EmptyRank1Rejected)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.xShape = {0};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, GatesRank3)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.gatesShape = {2, 8, 1};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, IndicesShapeMismatch)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.indicesShape = {3, 8};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, LocationsShapeMismatch)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.locationsShape = {2, 7};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, SMismatch)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.xShape = {9, 16};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, YRank2)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.yShape = {5, 4};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, YCapacityMismatch)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.yShape = {5, 3, 16};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, YHiddenMismatch)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.yShape = {5, 4, 15};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, XFloat64)
{
    auto params = DtypeCase(ge::DT_DOUBLE);
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, GatesDtypeMismatch)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.gatesDtype = ge::DT_FLOAT16;
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, IndicesInt64)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.indicesDtype = ge::DT_INT64;
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, LocationsInt64)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.locationsDtype = ge::DT_INT64;
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, YDtypeMismatch)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.yDtype = ge::DT_FLOAT16;
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, CapacityNegative)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.capacity = -1;
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, CapacityGtInt32)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.capacity = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, DimensionGtInt32)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.yShape[0] = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, SignedInt64ProductOverflow)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    const int64_t maxDim = std::numeric_limits<int32_t>::max();
    params.xShape = {1, maxDim};
    params.gatesShape = params.indicesShape = params.locationsShape = {1, 1};
    params.yShape = {maxDim, maxDim, maxDim};
    params.capacity = maxDim;
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, NonNdFormat)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.xFormat = ge::FORMAT_NCHW;
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, NegativeDynamicDimensionRejectedAtRuntime)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.xShape = {-1, 16};
    params.gatesShape = params.indicesShape = params.locationsShape = {2, -1};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, DynamicRank)
{
    auto params = DtypeCase(ge::DT_FLOAT);
    params.xShape = {-2};
    params.gatesShape = params.indicesShape = params.locationsShape = {-2};
    params.yShape = {-2};
    ExpectTiling(params, ge::GRAPH_FAILED);
}

TEST_F(MoeTutelDispatchTiling, MultiAxisEmptyOutput)
{
    auto params = DtypeCase(ge::DT_FLOAT16);
    params.xShape = {0, 0};
    params.gatesShape = params.indicesShape = params.locationsShape = {2, 0};
    params.yShape = {5, 4, 0};
    ExpectTiling(params, ge::GRAPH_SUCCESS, MOE_TUTEL_DISPATCH_EMPTY_OUTPUT);
}
