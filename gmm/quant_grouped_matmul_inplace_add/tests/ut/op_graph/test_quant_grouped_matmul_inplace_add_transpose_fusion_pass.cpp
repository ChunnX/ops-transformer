/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"

#include "../../../op_graph/fusion_pass/quant_grouped_matmul_inplace_add_transpose_fusion_pass.h"

namespace {
using ge::DataType;
using ge::GNode;
using ge::Graph;
using ge::GraphPtr;
using ge::Shape;
using ge::TensorDesc;

constexpr char kTargetType[] = "QuantGroupedMatmulInplaceAdd";
constexpr char kTransposeDType[] = "TransposeD";
constexpr char kReshapeType[] = "Reshape";

struct TensorWithDesc {
    ge::es::EsTensorHolder tensor;
    TensorDesc desc;
};

class TestPass : public ops::QuantGroupedMatmulInplaceAddTransposeFusionPass {
public:
    ge::Status RunForTest(GraphPtr& graph, ge::CustomPassContext& context)
    {
        return Run(graph, context);
    }
};

TensorDesc MakeDesc(const std::vector<int64_t>& shape, DataType dtype)
{
    TensorDesc desc(Shape(shape), ge::FORMAT_ND, dtype);
    desc.SetOriginShape(Shape(shape));
    desc.SetOriginFormat(ge::FORMAT_ND);
    return desc;
}

void Link(Graph* graph, TensorWithDesc& input, GNode& destination, int32_t inputIndex)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(),
        destination, inputIndex);
    destination.UpdateInputDesc(inputIndex, input.desc);
}

void AddUnary(Graph* graph, ge::es::EsGraphBuilder& builder, TensorWithDesc& input, const char* type,
    const char* name, const std::vector<int64_t>& outputShape, const std::vector<int64_t>& perm = {})
{
    auto outputDesc = MakeDesc(outputShape, input.desc.GetDataType());
    auto nodeBuilder = ge::es::CompliantNodeBuilder(graph);
    nodeBuilder.OpType(type)
        .Name(name)
        .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .InstanceOutputDataType("y", outputDesc.GetDataType())
        .InstanceOutputShape("y", outputShape)
        .InstanceOutputFormat("y", ge::FORMAT_ND);
    auto node = nodeBuilder.Build();
    if (!perm.empty()) {
        auto attr = perm;
        node.SetAttr("perm", attr);
    }
    Link(graph, input, node, 0);
    node.UpdateOutputDesc(0, outputDesc);
    input.tensor = ge::es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
    input.desc = outputDesc;
}

GraphPtr BuildGraph(bool mxMode, bool transposeScale, bool reshapeX, bool validReshape)
{
    ge::es::EsGraphBuilder builder("qgmmInplaceAddTransposePass");
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    int64_t inputIndex = 0;
    auto createInput = [&](const char* name, const std::vector<int64_t>& shape, DataType dtype) {
        auto desc = MakeDesc(shape, dtype);
        auto tensor = builder.CreateInput(inputIndex++, name, dtype, ge::FORMAT_ND, shape);
        tensor.GetProducer()->UpdateOutputDesc(0, desc);
        return TensorWithDesc{tensor, desc};
    };

    auto x1 = createInput("x1", {64, 32}, ge::DT_HIFLOAT8);
    auto x2 = createInput("x2", {2, 32, 48}, ge::DT_HIFLOAT8);
    auto scale2 = createInput("scale2", {2, 1}, ge::DT_FLOAT);
    auto groupList = createInput("groupList", {2}, ge::DT_INT64);
    auto y = createInput("y", {64, 48}, ge::DT_FLOAT);
    auto scale1 = createInput("scale1", {4, 64, 2}, mxMode ? ge::DT_FLOAT8_E8M0 : ge::DT_FLOAT);

    if (reshapeX) {
        AddUnary(graph, builder, x1, kReshapeType, "reshapeX1", validReshape ?
            std::vector<int64_t>{32, 64} : std::vector<int64_t>{16, 128});
    } else {
        AddUnary(graph, builder, x1, kTransposeDType, "transposeX1", {32, 64}, {1, 0});
    }
    if (mxMode && transposeScale) {
        AddUnary(graph, builder, scale1, kTransposeDType, "transposeScale1", {64, 4, 2}, {1, 0, 2});
    }

    auto target = ge::es::CompliantNodeBuilder(graph)
        .OpType(kTargetType)
        .Name("quantGroupedMatmulInplaceAdd")
        .IrDefInputs({
            {"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"scale2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"group_list", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"y", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"scale1", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
        })
        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .InstanceOutputDataType("y", ge::DT_FLOAT)
        .InstanceOutputShape("y", {64, 48})
        .InstanceOutputFormat("y", ge::FORMAT_ND)
        .Build();
    Link(graph, x1, target, 0);
    Link(graph, x2, target, 1);
    Link(graph, scale2, target, 2);
    Link(graph, groupList, target, 3);
    Link(graph, y, target, 4);
    Link(graph, scale1, target, 5);
    target.UpdateOutputDesc(0, MakeDesc({64, 48}, ge::DT_FLOAT));
    auto output = ge::es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(target, 0));
    return builder.BuildAndReset({output});
}

int CountType(const GraphPtr& graph, const char* expected)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        ge::AscendString type;
        if (node.GetType(type) == ge::GRAPH_SUCCESS && type == expected) {
            ++count;
        }
    }
    return count;
}

ge::Status RunPass(GraphPtr& graph)
{
    ge::CustomPassContext context;
    TestPass pass;
    return pass.RunForTest(graph, context);
}
} // namespace

TEST(QuantGroupedMatmulInplaceAddTransposeFusionPass, FuseNormalX1Transpose)
{
    auto graph = BuildGraph(false, false, false, false);
    ASSERT_EQ(RunPass(graph), ge::SUCCESS);
    EXPECT_EQ(CountType(graph, kTransposeDType), 0);
}

TEST(QuantGroupedMatmulInplaceAddTransposeFusionPass, FuseMxX1AndScale1Atomically)
{
    auto graph = BuildGraph(true, true, false, false);
    ASSERT_EQ(RunPass(graph), ge::SUCCESS);
    EXPECT_EQ(CountType(graph, kTransposeDType), 0);
}

TEST(QuantGroupedMatmulInplaceAddTransposeFusionPass, KeepMxGraphWhenScale1TransposeIsMissing)
{
    auto graph = BuildGraph(true, false, false, false);
    ASSERT_EQ(RunPass(graph), ge::GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, kTransposeDType), 1);
}

TEST(QuantGroupedMatmulInplaceAddTransposeFusionPass, RejectNonEquivalentReshape)
{
    auto graph = BuildGraph(false, false, true, false);
    ASSERT_EQ(RunPass(graph), ge::GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, kReshapeType), 1);
}
