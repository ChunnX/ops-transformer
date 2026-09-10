/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/graph.h"
#define private public
#include "platform/platform_info.h"
#undef private
#include "register/register_custom_pass.h"

#include "../../../op_graph/fusion_pass/grouped_matmul_finalize_routing_transpose_fusion_pass.h"

using namespace ge;

namespace {
constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT_INDEX = 1;
constexpr int64_t WEIGHT_SCALE_INDEX = 2;

struct TensorWithDesc {
    ge::es::EsTensorHolder tensor;
    TensorDesc desc;
};

struct FusionCase {
    std::string caseName;
    DataType xDtype;
    DataType weightDtype;
    DataType weightScaleDtype;
    bool hasBitcast;
    bool useWeightReshape;
    bool validWeightPattern;
    bool hasMxWeightScaleTranspose;
    Status expectedStatus;
    int expectedTransposeCount;
    int expectedReshapeCount;
    int expectedBitcastCount;
    bool expectedTransposeWeight;
    bool nonUnitWeightReshape = false;
};

class TestGroupedMatmulFinalizeRoutingTransposeFusionPass
    : public ops::GroupedMatmulFinalizeRoutingTransposeFusionPass {
public:
    Status RunForTest(GraphPtr &graph, CustomPassContext &passContext)
    {
        return Run(graph, passContext);
    }
};

static std::vector<std::pair<int64_t, int64_t>> MakeShapeRange(const std::vector<int64_t> &shape)
{
    std::vector<std::pair<int64_t, int64_t>> range;
    range.reserve(shape.size());
    for (auto dim : shape) {
        range.emplace_back(dim, dim);
    }
    return range;
}

static void SetPlatformSupport()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optionalInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
    fe::PlatformInfoManager::Instance().platform_info_map_[optionalInfo.soc_version] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

static void ClearPlatformSupport()
{
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
}

static TensorDesc MakeTensorDesc(const std::vector<int64_t> &shape, DataType dtype, Format format = FORMAT_ND)
{
    TensorDesc desc(Shape(shape), format, dtype);
    desc.SetOriginFormat(format);
    desc.SetOriginShape(Shape(shape));
    desc.SetShapeRange(MakeShapeRange(shape));
    return desc;
}

static std::vector<int64_t> TransposeLastTwoDims(const std::vector<int64_t> &shape)
{
    auto transposedShape = shape;
    if (transposedShape.size() >= 2) {
        std::swap(transposedShape[transposedShape.size() - 1], transposedShape[transposedShape.size() - 2]);
    }
    return transposedShape;
}

static GNode BuildUnaryNode(Graph *graph, const char *opType, const char *nodeName, const TensorDesc &outputDesc)
{
    return ge::es::CompliantNodeBuilder(graph)
        .OpType(opType)
        .Name(nodeName)
        .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .InstanceOutputDataType("y", outputDesc.GetDataType())
        .InstanceOutputShape("y", outputDesc.GetShape().GetDims())
        .InstanceOutputFormat("y", FORMAT_ND)
        .Build();
}

static void LinkUnaryNode(Graph *graph, ge::es::EsGraphBuilder &graphBuilder, TensorWithDesc &input, GNode &node,
                          const TensorDesc &outputDesc)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(), node, 0);
    node.UpdateInputDesc(0, input.desc);
    node.UpdateOutputDesc(0, outputDesc);
    input.tensor = ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
    input.desc = outputDesc;
}

static void AddTranspose(Graph *graph, ge::es::EsGraphBuilder &graphBuilder, TensorWithDesc &input,
                         const char *nodeName, bool validPerm = true)
{
    auto outputDims = input.desc.GetShape().GetDims();
    std::vector<int64_t> perm;
    if (input.desc.GetDataType() == DT_FLOAT8_E8M0) {
        std::swap(outputDims[1], outputDims[2]);
        perm = {0, 2, 1, 3};
    } else {
        outputDims = TransposeLastTwoDims(outputDims);
        perm = {0, 2, 1};
    }
    auto outputDesc = MakeTensorDesc(outputDims, input.desc.GetDataType());
    auto transposeNode = BuildUnaryNode(graph, "TransposeD", nodeName, outputDesc);
    if (!validPerm) {
        perm = {0, 1, 2};
    }
    transposeNode.SetAttr("perm", perm);
    LinkUnaryNode(graph, graphBuilder, input, transposeNode, outputDesc);
}

static void AddReshape(Graph *graph, ge::es::EsGraphBuilder &graphBuilder, TensorWithDesc &input, const char *nodeName,
                       bool validShape)
{
    auto outputDims = TransposeLastTwoDims(input.desc.GetShape().GetDims());
    if (!validShape) {
        outputDims = {2, 16, 128};
    }
    auto outputDesc = MakeTensorDesc(outputDims, input.desc.GetDataType());
    auto reshapeNode = BuildUnaryNode(graph, "Reshape", nodeName, outputDesc);
    LinkUnaryNode(graph, graphBuilder, input, reshapeNode, outputDesc);
}

static void AddBitcast(Graph *graph, ge::es::EsGraphBuilder &graphBuilder, TensorWithDesc &input, const char *nodeName)
{
    auto outputDesc = input.desc;
    outputDesc.SetDataType(DT_HIFLOAT8);
    auto bitcastNode = BuildUnaryNode(graph, "Bitcast", nodeName, outputDesc);
    LinkUnaryNode(graph, graphBuilder, input, bitcastNode, outputDesc);
    DataType bitcastType = DT_HIFLOAT8;
    bitcastNode.SetAttr("type", bitcastType);
}

static void LinkGmmInput(Graph *graph, GNode &gmmNode, const TensorWithDesc &input, int64_t inputIndex)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(), gmmNode,
                                     inputIndex);
    gmmNode.UpdateInputDesc(inputIndex, input.desc);
}

static GNode BuildGmmNode(Graph *graph)
{
    return ge::es::CompliantNodeBuilder(graph)
        .OpType("GroupedMatmulFinalizeRouting")
        .Name("GroupedMatmulFinalizeRouting")
        .IrDefInputs({
            {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"w", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"bias", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"pertoken_scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"group_list", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"shared_input", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"logit", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"row_index", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"offset", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
        })
        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .InstanceOutputDataType("y", DT_BF16)
        .InstanceOutputShape("y", {4, 32})
        .InstanceOutputFormat("y", FORMAT_ND)
        .Build();
}

static GraphPtr BuildGraph(const FusionCase &param)
{
    auto graphBuilder = ge::es::EsGraphBuilder(param.caseName.c_str());
    auto *graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto xDesc = MakeTensorDesc({4, 64}, param.xDtype);
    const auto weightShape = param.useWeightReshape && param.validWeightPattern && !param.nonUnitWeightReshape ?
                                                                 std::vector<int64_t>{2, 1, 64} :
                                                                                  std::vector<int64_t>{2, 32, 64};
    auto weightDesc = MakeTensorDesc(weightShape, param.weightDtype);
    const auto weightScaleShape =
        param.weightScaleDtype == DT_FLOAT8_E8M0 ? std::vector<int64_t>{2, 32, 1, 2} : std::vector<int64_t>{2, 1};
    auto weightScaleDesc = MakeTensorDesc(weightScaleShape, param.weightScaleDtype);

    auto dataX = graphBuilder.CreateInput(0, "dataX", param.xDtype, FORMAT_ND, xDesc.GetShape().GetDims());
    auto dataWeight =
        graphBuilder.CreateInput(1, "dataWeight", param.weightDtype, FORMAT_ND, weightDesc.GetShape().GetDims());
    auto dataWeightScale = graphBuilder.CreateInput(2, "dataWeightScale", param.weightScaleDtype, FORMAT_ND,
                                                    weightScaleDesc.GetShape().GetDims());
    dataX.GetProducer()->UpdateOutputDesc(0, xDesc);
    dataWeight.GetProducer()->UpdateOutputDesc(0, weightDesc);
    dataWeightScale.GetProducer()->UpdateOutputDesc(0, weightScaleDesc);

    TensorWithDesc xInput = {dataX, xDesc};
    TensorWithDesc weightInput = {dataWeight, weightDesc};
    TensorWithDesc weightScaleInput = {dataWeightScale, weightScaleDesc};

    if (param.useWeightReshape) {
        AddReshape(graph, graphBuilder, weightInput, "reshape_weight", param.validWeightPattern);
    } else {
        AddTranspose(graph, graphBuilder, weightInput, "transpose_weight", param.validWeightPattern);
    }
    if (param.hasBitcast) {
        AddBitcast(graph, graphBuilder, weightInput, "bitcast_weight");
    }
    if (param.hasMxWeightScaleTranspose) {
        AddTranspose(graph, graphBuilder, weightScaleInput, "transpose_weight_scale");
    }

    auto gmmNode = BuildGmmNode(graph);
    LinkGmmInput(graph, gmmNode, xInput, X_INDEX);
    LinkGmmInput(graph, gmmNode, weightInput, WEIGHT_INDEX);
    LinkGmmInput(graph, gmmNode, weightScaleInput, WEIGHT_SCALE_INDEX);

    auto outputDesc = MakeTensorDesc({4, 32}, DT_BF16);
    gmmNode.UpdateOutputDesc(0, outputDesc);
    bool transposeWeight = false;
    gmmNode.SetAttr("transpose_w", transposeWeight);

    auto output = ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(gmmNode, 0));
    return graphBuilder.BuildAndReset({output});
}

static int CountNodes(const GraphPtr &graph, const char *nodeType)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && type == nodeType) {
            ++count;
        }
    }
    return count;
}

static GNode FindGmmNode(const GraphPtr &graph)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && type == "GroupedMatmulFinalizeRouting") {
            return node;
        }
    }
    ADD_FAILURE() << "GroupedMatmulFinalizeRouting node is not found.";
    return GNode();
}

class GroupedMatmulFinalizeRoutingTransposeFusionPassTest : public testing::TestWithParam<FusionCase> {};

TEST_P(GroupedMatmulFinalizeRoutingTransposeFusionPassTest, RunFusionPass)
{
    const auto &param = GetParam();
    auto graph = BuildGraph(param);

    CustomPassContext passContext;
    passContext.SetPassName("GroupedMatmulFinalizeRoutingTransposeFusionPass");
    TestGroupedMatmulFinalizeRoutingTransposeFusionPass pass;
    SetPlatformSupport();
    auto status = pass.RunForTest(graph, passContext);
    ClearPlatformSupport();

    EXPECT_EQ(status, param.expectedStatus);
    EXPECT_EQ(CountNodes(graph, "TransposeD"), param.expectedTransposeCount);
    EXPECT_EQ(CountNodes(graph, "Reshape"), param.expectedReshapeCount);
    EXPECT_EQ(CountNodes(graph, "Bitcast"), param.expectedBitcastCount);

    auto gmmNode = FindGmmNode(graph);
    bool transposeWeight = false;
    ASSERT_EQ(gmmNode.GetAttr("transpose_w", transposeWeight), GRAPH_SUCCESS);
    EXPECT_EQ(transposeWeight, param.expectedTransposeWeight);
}

std::string TestName(const testing::TestParamInfo<FusionCase> &info)
{
    return info.param.caseName;
}

INSTANTIATE_TEST_CASE_P(GroupedMatmulFinalizeRoutingTransposeFusionPass,
                        GroupedMatmulFinalizeRoutingTransposeFusionPassTest,
                        testing::Values(FusionCase{"mx_direct_weight_scale_transpose", DT_INT8, DT_INT8, DT_FLOAT8_E8M0,
                                                   false, false, true, true, SUCCESS, 0, 0, 0, true},
                                        FusionCase{"mx_bitcast_weight_transpose", DT_INT8, DT_FLOAT4_E2M1,
                                                   DT_FLOAT8_E8M0, true, false, true, true, SUCCESS, 0, 0, 1, true},
                                        FusionCase{"mx_weight_scale_transpose", DT_INT8, DT_INT8, DT_FLOAT8_E8M0, false,
                                                   false, true, true, SUCCESS, 0, 0, 0, true},
                                        FusionCase{"unsupported_x_dtype", DT_FLOAT16, DT_INT8, DT_FLOAT8_E8M0, false,
                                                   false, true, true, GRAPH_NOT_CHANGED, 2, 0, 0, false},
                                        FusionCase{"invalid_weight_perm", DT_INT8, DT_INT8, DT_FLOAT8_E8M0, false,
                                                   false, false, true, GRAPH_NOT_CHANGED, 2, 0, 0, false},
                                        FusionCase{"mx_missing_scale_transpose", DT_INT8, DT_INT8, DT_FLOAT8_E8M0,
                                                   false, false, true, false, GRAPH_NOT_CHANGED, 1, 0, 0, false},
                                        FusionCase{"non_mx_scale", DT_INT8, DT_INT8, DT_FLOAT, false, false, true,
                                                   false, GRAPH_NOT_CHANGED, 1, 0, 0, false},
                                        FusionCase{"unit_axis_reshape", DT_INT8, DT_INT8, DT_FLOAT8_E8M0, false, true,
                                                   true, true, SUCCESS, 0, 0, 0, true},
                                        FusionCase{"regular_reshape_without_unit_axis", DT_INT8, DT_INT8, DT_FLOAT8_E8M0,
                                                   false, true, true, true, SUCCESS, 0, 0, 0, true, true},
                                        FusionCase{"invalid_reshape", DT_INT8, DT_INT8, DT_FLOAT8_E8M0, false, true,
                                                   false, true, GRAPH_NOT_CHANGED, 1, 1, 0, false}),
                        TestName);
} // namespace
