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
#include "register/register_custom_pass.h"

#include "../../../op_graph/fusion_pass/grouped_matmul_swiglu_quant_transpose_fusion_pass.h"

using namespace ge;

namespace {
constexpr int64_t X_INDEX = 0;
constexpr int64_t X_SCALE_INDEX = 1;
constexpr int64_t GROUP_LIST_INDEX = 2;
constexpr int64_t WEIGHT_INDEX = 3;
constexpr int64_t WEIGHT_SCALE_INDEX = 4;

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
};

class TestGroupedMatmulSwigluQuantTransposeFusionPass : public ops::GroupedMatmulSwigluQuantTransposeFusionPass {
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
        .OpType("GroupedMatmulSwigluQuantV2")
        .Name("GroupedMatmulSwigluQuantV2")
        .IrDefInputs({
            {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x_scale", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"group_list", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"weight", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
            {"weight_scale", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
            {"weight_assist_matrix", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
            {"bias", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"smooth_scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
        })
        .IrDefOutputs({
            {"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"y_scale", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
        })
        .InstanceOutputDataType("y", DT_INT8)
        .InstanceOutputShape("y", {4, 32})
        .InstanceOutputFormat("y", FORMAT_ND)
        .InstanceOutputDataType("y_scale", DT_FLOAT)
        .InstanceOutputShape("y_scale", {4, 32})
        .InstanceOutputFormat("y_scale", FORMAT_ND)
        .Build();
}

static GraphPtr BuildGraph(const FusionCase &param)
{
    auto graphBuilder = ge::es::EsGraphBuilder(param.caseName.c_str());
    auto *graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto xDesc = MakeTensorDesc({4, 64}, param.xDtype);
    auto xScaleDesc = MakeTensorDesc({4, 1}, DT_FLOAT);
    auto groupListDesc = MakeTensorDesc({2}, DT_INT64);
    const auto weightShape = param.useWeightReshape && param.validWeightPattern ? std::vector<int64_t>{2, 1, 64} :
                                                                                  std::vector<int64_t>{2, 32, 64};
    auto weightDesc = MakeTensorDesc(weightShape, param.weightDtype);
    const auto weightScaleShape =
        param.weightScaleDtype == DT_FLOAT8_E8M0 ? std::vector<int64_t>{2, 32, 1, 2} : std::vector<int64_t>{2, 1};
    auto weightScaleDesc = MakeTensorDesc(weightScaleShape, param.weightScaleDtype);

    auto dataX = graphBuilder.CreateInput(0, "dataX", param.xDtype, FORMAT_ND, xDesc.GetShape().GetDims());
    auto dataXScale = graphBuilder.CreateInput(1, "dataXScale", DT_FLOAT, FORMAT_ND, xScaleDesc.GetShape().GetDims());
    auto dataGroupList =
        graphBuilder.CreateInput(2, "dataGroupList", DT_INT64, FORMAT_ND, groupListDesc.GetShape().GetDims());
    auto dataWeight =
        graphBuilder.CreateInput(3, "dataWeight", param.weightDtype, FORMAT_ND, weightDesc.GetShape().GetDims());
    auto dataWeightScale = graphBuilder.CreateInput(4, "dataWeightScale", param.weightScaleDtype, FORMAT_ND,
                                                    weightScaleDesc.GetShape().GetDims());
    dataX.GetProducer()->UpdateOutputDesc(0, xDesc);
    dataXScale.GetProducer()->UpdateOutputDesc(0, xScaleDesc);
    dataGroupList.GetProducer()->UpdateOutputDesc(0, groupListDesc);
    dataWeight.GetProducer()->UpdateOutputDesc(0, weightDesc);
    dataWeightScale.GetProducer()->UpdateOutputDesc(0, weightScaleDesc);

    TensorWithDesc xInput = {dataX, xDesc};
    TensorWithDesc xScaleInput = {dataXScale, xScaleDesc};
    TensorWithDesc groupListInput = {dataGroupList, groupListDesc};
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
    LinkGmmInput(graph, gmmNode, xScaleInput, X_SCALE_INDEX);
    LinkGmmInput(graph, gmmNode, groupListInput, GROUP_LIST_INDEX);
    LinkGmmInput(graph, gmmNode, weightInput, WEIGHT_INDEX);
    LinkGmmInput(graph, gmmNode, weightScaleInput, WEIGHT_SCALE_INDEX);

    auto outputDesc = MakeTensorDesc({4, 32}, DT_INT8);
    auto outputScaleDesc = MakeTensorDesc({4, 32}, DT_FLOAT);
    gmmNode.UpdateOutputDesc(0, outputDesc);
    gmmNode.UpdateOutputDesc(1, outputScaleDesc);
    bool transposeWeight = false;
    gmmNode.SetAttr("transpose_weight", transposeWeight);

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
        if (node.GetType(type) == GRAPH_SUCCESS && type == "GroupedMatmulSwigluQuantV2") {
            return node;
        }
    }
    ADD_FAILURE() << "GroupedMatmulSwigluQuantV2 node is not found.";
    return GNode();
}

class GroupedMatmulSwigluQuantTransposeFusionPassTest : public testing::TestWithParam<FusionCase> {};

TEST_P(GroupedMatmulSwigluQuantTransposeFusionPassTest, RunFusionPass)
{
    const auto &param = GetParam();
    auto graph = BuildGraph(param);

    CustomPassContext passContext;
    passContext.SetPassName("GroupedMatmulSwigluQuantTransposeFusionPass");
    TestGroupedMatmulSwigluQuantTransposeFusionPass pass;
    auto status = pass.RunForTest(graph, passContext);

    EXPECT_EQ(status, param.expectedStatus);
    EXPECT_EQ(CountNodes(graph, "TransposeD"), param.expectedTransposeCount);
    EXPECT_EQ(CountNodes(graph, "Reshape"), param.expectedReshapeCount);
    EXPECT_EQ(CountNodes(graph, "Bitcast"), param.expectedBitcastCount);

    auto gmmNode = FindGmmNode(graph);
    bool transposeWeight = false;
    ASSERT_EQ(gmmNode.GetAttr("transpose_weight", transposeWeight), GRAPH_SUCCESS);
    EXPECT_EQ(transposeWeight, param.expectedTransposeWeight);
}

std::string TestName(const testing::TestParamInfo<FusionCase> &info)
{
    return info.param.caseName;
}

INSTANTIATE_TEST_CASE_P(GroupedMatmulSwigluQuantTransposeFusionPass, GroupedMatmulSwigluQuantTransposeFusionPassTest,
                        testing::Values(FusionCase{"direct_weight_transpose", DT_INT8, DT_INT8, DT_FLOAT, false, false,
                                                   true, false, SUCCESS, 0, 0, 0, true},
                                        FusionCase{"bitcast_weight_transpose", DT_INT8, DT_FLOAT4_E2M1, DT_FLOAT, true,
                                                   false, true, false, SUCCESS, 0, 0, 1, true},
                                        FusionCase{"mx_weight_scale_transpose", DT_INT8, DT_INT8, DT_FLOAT8_E8M0, false,
                                                   false, true, true, SUCCESS, 0, 0, 0, true},
                                        FusionCase{"unsupported_x_dtype", DT_FLOAT16, DT_INT8, DT_FLOAT, false, false,
                                                   true, false, GRAPH_NOT_CHANGED, 1, 0, 0, false},
                                        FusionCase{"invalid_weight_perm", DT_INT8, DT_INT8, DT_FLOAT, false, false,
                                                   false, false, GRAPH_NOT_CHANGED, 1, 0, 0, false},
                                        FusionCase{"mx_missing_scale_transpose", DT_INT8, DT_INT8, DT_FLOAT8_E8M0,
                                                   false, false, true, false, GRAPH_NOT_CHANGED, 1, 0, 0, false},
                                        FusionCase{"unit_axis_reshape", DT_INT8, DT_INT8, DT_FLOAT, false, true, true,
                                                   false, SUCCESS, 0, 0, 0, true},
                                        FusionCase{"invalid_reshape", DT_INT8, DT_INT8, DT_FLOAT, false, true, false,
                                                   false, GRAPH_NOT_CHANGED, 0, 1, 0, false}),
                        TestName);
} // namespace
