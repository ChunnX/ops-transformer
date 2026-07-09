/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include "acl/acl_rt.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/graph.h"
#include "register/register_custom_pass.h"

#include "../../../op_graph/fusion_pass/grouped_matmul_transpose_fusion_pass.h"

using namespace ge;
using namespace ge::fusion;

namespace {
constexpr int32_t kGraphFusionSupportVersion = 90000000;
constexpr int32_t kXIndex = 0;
constexpr int32_t kWeightIndex = 1;
constexpr char kGroupedMatmulType[] = "GroupedMatmul";
constexpr char kTransposeDType[] = "TransposeD";

struct TensorWithDesc {
    ge::es::EsTensorHolder tensor;
    TensorDesc desc;
};

class TestGroupedMatmulTransposeFusionPass : public ops::GroupedMatmulTransposeFusionPass {
public:
    Status RunForTest(GraphPtr& graph, CustomPassContext& passContext)
    {
        return Run(graph, passContext);
    }
};

std::vector<std::pair<int64_t, int64_t>> MakeShapeRange(const std::vector<int64_t>& shape)
{
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    shapeRange.reserve(shape.size());
    for (const auto dim : shape) {
        shapeRange.emplace_back(dim, dim);
    }
    return shapeRange;
}

TensorDesc MakeTensorDesc(const std::vector<int64_t>& shape, DataType dtype)
{
    TensorDesc desc(Shape(shape), FORMAT_ND, dtype);
    desc.SetOriginFormat(FORMAT_ND);
    desc.SetOriginShape(Shape(shape));
    desc.SetShapeRange(MakeShapeRange(shape));
    return desc;
}

GNode BuildTransposeDNode(Graph* graph, const char* name, const TensorDesc& outputDesc)
{
    auto transposeNode = ge::es::CompliantNodeBuilder(graph)
                             .OpType(kTransposeDType)
                             .Name(name)
                             .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .InstanceOutputDataType("y", outputDesc.GetDataType())
                             .InstanceOutputShape("y", outputDesc.GetShape().GetDims())
                             .InstanceOutputFormat("y", FORMAT_ND)
                             .Build();
    std::vector<int64_t> perm = {0, 2, 1};
    transposeNode.SetAttr("perm", perm);
    return transposeNode;
}

void LinkUnaryNode(
    Graph* graph, ge::es::EsGraphBuilder& graphBuilder, TensorWithDesc& input, GNode& node, const TensorDesc& outputDesc)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(), node, 0);
    node.UpdateInputDesc(0, input.desc);
    node.UpdateOutputDesc(0, outputDesc);
    input.tensor = ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
    input.desc = outputDesc;
}

void LinkGroupedMatmulInput(Graph* graph, GNode& groupedMatmulNode, const TensorWithDesc& input, int32_t inputIndex)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(),
        groupedMatmulNode, inputIndex);
    groupedMatmulNode.UpdateInputDesc(inputIndex, input.desc);
}

GraphPtr BuildWeightTransposeGraph()
{
    auto graphBuilder = ge::es::EsGraphBuilder("grouped_matmul_transpose_fusion_pass_test");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    const std::vector<int64_t> xShape = {32, 16};
    const std::vector<int64_t> weightSourceShape = {2, 64, 16};
    const std::vector<int64_t> weightTransposedShape = {2, 16, 64};
    const std::vector<int64_t> yShape = {32, 64};

    auto xDesc = MakeTensorDesc(xShape, DT_FLOAT16);
    auto weightSourceDesc = MakeTensorDesc(weightSourceShape, DT_FLOAT16);
    auto weightTransposedDesc = MakeTensorDesc(weightTransposedShape, DT_FLOAT16);
    auto yDesc = MakeTensorDesc(yShape, DT_FLOAT16);

    auto dataX = graphBuilder.CreateInput(0, "dataX", DT_FLOAT16, FORMAT_ND, xShape);
    auto dataWeight = graphBuilder.CreateInput(1, "dataWeight", DT_FLOAT16, FORMAT_ND, weightSourceShape);
    dataX.GetProducer()->UpdateOutputDesc(0, xDesc);
    dataWeight.GetProducer()->UpdateOutputDesc(0, weightSourceDesc);

    TensorWithDesc xInput = {dataX, xDesc};
    TensorWithDesc weightInput = {dataWeight, weightSourceDesc};
    auto transposeWeightNode = BuildTransposeDNode(graph, "transposeWeight", weightTransposedDesc);
    LinkUnaryNode(graph, graphBuilder, weightInput, transposeWeightNode, weightTransposedDesc);

    auto groupedMatmulNode = ge::es::CompliantNodeBuilder(graph)
                                 .OpType(kGroupedMatmulType)
                                 .Name("groupedMatmul")
                                 .IrDefInputs({
                                     {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                     {"weight", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                     {"bias", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                     {"scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                     {"offset", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                     {"antiquant_scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                     {"antiquant_offset", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                     {"group_list", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                     {"per_token_scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                 })
                                 .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                 .InstanceOutputDataType("y", DT_FLOAT16)
                                 .InstanceOutputShape("y", yShape)
                                 .InstanceOutputFormat("y", FORMAT_ND)
                                 .Build();
    LinkGroupedMatmulInput(graph, groupedMatmulNode, xInput, kXIndex);
    LinkGroupedMatmulInput(graph, groupedMatmulNode, weightInput, kWeightIndex);
    groupedMatmulNode.UpdateOutputDesc(0, yDesc);

    bool transposeX = false;
    bool transposeWeightAttr = false;
    groupedMatmulNode.SetAttr("transpose_x", transposeX);
    groupedMatmulNode.SetAttr("transpose_weight", transposeWeightAttr);

    auto output = ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(groupedMatmulNode, 0));
    return graphBuilder.BuildAndReset({output});
}

int CountNodes(const GraphPtr& graph, const char* type)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == type) {
            ++count;
        }
    }
    return count;
}

GNode FindGroupedMatmulNode(const GraphPtr& graph)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == kGroupedMatmulType) {
            return node;
        }
    }
    throw std::runtime_error("GroupedMatmul node is not found.");
}
} // namespace

extern "C" aclError aclsysGetVersionNum(const char*, int32_t* version)
{
    if (version != nullptr) {
        *version = kGraphFusionSupportVersion;
    }
    return ACL_SUCCESS;
}

extern "C" rtError_t rtGetSocSpec(const char*, const char*, char* value, const uint32_t len)
{
    if (value == nullptr || len == 0) {
        return static_cast<rtError_t>(1);
    }
    const char socVersion[] = "3510";
    std::strncpy(value, socVersion, len - 1);
    value[len - 1] = '\0';
    return RT_ERROR_NONE;
}

TEST(GroupedMatmulTransposeFusionPassTest, FuseWeightTransposeDAndSetTransposeWeight)
{
    auto graph = BuildWeightTransposeGraph();
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(CountNodes(graph, kTransposeDType), 1);

    CustomPassContext passContext;
    TestGroupedMatmulTransposeFusionPass pass;
    ASSERT_EQ(pass.RunForTest(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graph, kTransposeDType), 0);
    auto groupedMatmulNode = FindGroupedMatmulNode(graph);

    bool transposeWeight = false;
    ASSERT_EQ(groupedMatmulNode.GetAttr("transpose_weight", transposeWeight), GRAPH_SUCCESS);
    EXPECT_TRUE(transposeWeight);

    TensorDesc weightDesc;
    ASSERT_EQ(groupedMatmulNode.GetInputDesc(kWeightIndex, weightDesc), GRAPH_SUCCESS);
    EXPECT_EQ(weightDesc.GetShape().GetDims(), std::vector<int64_t>({2, 64, 16}));
}
