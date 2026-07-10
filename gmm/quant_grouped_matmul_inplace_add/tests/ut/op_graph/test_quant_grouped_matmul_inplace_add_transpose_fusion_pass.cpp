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
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/graph.h"
#include "register/register_custom_pass.h"

#include "../../../op_graph/fusion_pass/quant_grouped_matmul_inplace_add_transpose_fusion_pass.h"

using namespace ge;

namespace {
constexpr char PASS_NAME[] = "QuantGroupedMatmulInplaceAddTransposeFusionPass";
constexpr char TEST_CSV_FILE[] = "test_quant_grouped_matmul_inplace_add_transpose_fusion_pass.csv";
#ifndef QUANT_GROUPED_MATMUL_INPLACE_ADD_TRANSPOSE_FUSION_PASS_CSV_PATH
#define QUANT_GROUPED_MATMUL_INPLACE_ADD_TRANSPOSE_FUSION_PASS_CSV_PATH ""
#endif
constexpr size_t CASE_FIELD_NUM = 21;
constexpr int64_t X1_INDEX = 0;
constexpr int64_t X2_INDEX = 1;
constexpr int64_t SCALE2_INDEX = 2;
constexpr int64_t GROUP_LIST_INDEX = 3;
constexpr int64_t Y_INDEX = 4;
constexpr int64_t SCALE1_INDEX = 5;

struct TensorWithDesc {
    ge::es::EsTensorHolder tensor;
    TensorDesc desc;
};

struct QuantGroupedMatmulInplaceAddTransposeFusionPassParam {
    std::string caseName;
    std::vector<int64_t> x1Shape;
    DataType x1Dtype;
    std::vector<int64_t> x2Shape;
    DataType x2Dtype;
    std::vector<int64_t> scale2Shape;
    DataType scale2Dtype;
    std::vector<int64_t> groupListShape;
    std::vector<int64_t> yShape;
    std::vector<int64_t> scale1Shape;
    DataType scale1Dtype;
    std::vector<int64_t> outShape;
    DataType outDtype;
    bool transposeX1;
    bool bitcastX1;
    bool transposeScale1;
    Status expectedStatus;
    int expectedTransposeCount;
    int expectedBitcastCount;
    std::vector<int64_t> expectedX1Shape;
    std::vector<int64_t> expectedScale1Shape;
};

class TestQuantGroupedMatmulInplaceAddTransposeFusionPass
    : public ops::QuantGroupedMatmulInplaceAddTransposeFusionPass {
public:
    Status RunForTest(GraphPtr& graph, CustomPassContext& passContext)
    {
        return Run(graph, passContext);
    }
};

static std::string Trim(const std::string& value)
{
    auto begin = value.begin();
    while (begin != value.end() && std::isspace(static_cast<unsigned char>(*begin))) {
        ++begin;
    }
    auto end = value.end();
    while (end != begin && std::isspace(static_cast<unsigned char>(*(end - 1)))) {
        --end;
    }
    return std::string(begin, end);
}

static std::vector<std::string> SplitCsvLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.emplace_back(Trim(field));
    }
    if (!line.empty() && line.back() == ',') {
        fields.emplace_back("");
    }
    return fields;
}

static std::string GetDirName(const std::string& path)
{
    auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

static std::ifstream OpenCsvData()
{
    const std::vector<std::string> casePaths = {
        QUANT_GROUPED_MATMUL_INPLACE_ADD_TRANSPOSE_FUSION_PASS_CSV_PATH,
        GetDirName(__FILE__) + "/" + TEST_CSV_FILE,
        TEST_CSV_FILE,
        "../../../../gmm/quant_grouped_matmul_inplace_add/tests/ut/op_graph/" + std::string(TEST_CSV_FILE),
    };
    std::string triedPaths;
    for (const auto& casePath : casePaths) {
        if (casePath.empty()) {
            continue;
        }
        std::ifstream csvData(casePath, std::ios::in);
        if (csvData.is_open()) {
            return csvData;
        }
        if (!triedPaths.empty()) {
            triedPaths += ", ";
        }
        triedPaths += casePath;
    }
    throw std::runtime_error("Open csv file failed, tried: " + triedPaths);
}

static std::vector<int64_t> ParseInt64Vec(const std::string& value)
{
    std::vector<int64_t> result;
    std::stringstream stream(value);
    int64_t item = 0;
    while (stream >> item) {
        result.emplace_back(item);
    }
    return result;
}

static bool ParseBool(const std::string& value)
{
    return value == "1" || value == "true" || value == "TRUE";
}

static int ParseIntOrDefault(const std::string& value, int defaultValue)
{
    return value.empty() ? defaultValue : std::stoi(value);
}

static Status ToStatus(const std::string& value)
{
    if (value == "SUCCESS") {
        return SUCCESS;
    }
    if (value == "GRAPH_NOT_CHANGED") {
        return GRAPH_NOT_CHANGED;
    }
    if (value == "GRAPH_FAILED") {
        return GRAPH_FAILED;
    }
    if (value == "FAILED") {
        return FAILED;
    }
    throw std::runtime_error("Unsupported status: " + value);
}

static DataType ParseDtype(const std::string& value)
{
    if (value == "DT_FLOAT") {
        return DT_FLOAT;
    }
    if (value == "DT_FLOAT16") {
        return DT_FLOAT16;
    }
    if (value == "DT_INT64") {
        return DT_INT64;
    }
    if (value == "DT_FLOAT8_E5M2") {
        return DT_FLOAT8_E5M2;
    }
    if (value == "DT_FLOAT8_E4M3FN") {
        return DT_FLOAT8_E4M3FN;
    }
    if (value == "DT_FLOAT8_E8M0") {
        return DT_FLOAT8_E8M0;
    }
    if (value == "DT_HIFLOAT8") {
        return DT_HIFLOAT8;
    }
    throw std::runtime_error("Unsupported dtype: " + value);
}

static QuantGroupedMatmulInplaceAddTransposeFusionPassParam ParseParam(const std::vector<std::string>& fields)
{
    if (fields.size() != CASE_FIELD_NUM) {
        throw std::runtime_error("Invalid csv column size.");
    }
    size_t index = 0;
    QuantGroupedMatmulInplaceAddTransposeFusionPassParam param;
    param.caseName = fields[index++];
    param.x1Shape = ParseInt64Vec(fields[index++]);
    param.x1Dtype = ParseDtype(fields[index++]);
    param.x2Shape = ParseInt64Vec(fields[index++]);
    param.x2Dtype = ParseDtype(fields[index++]);
    param.scale2Shape = ParseInt64Vec(fields[index++]);
    param.scale2Dtype = ParseDtype(fields[index++]);
    param.groupListShape = ParseInt64Vec(fields[index++]);
    param.yShape = ParseInt64Vec(fields[index++]);
    param.scale1Shape = ParseInt64Vec(fields[index++]);
    param.scale1Dtype = ParseDtype(fields[index++]);
    param.outShape = ParseInt64Vec(fields[index++]);
    param.outDtype = ParseDtype(fields[index++]);
    param.transposeX1 = ParseBool(fields[index++]);
    param.bitcastX1 = ParseBool(fields[index++]);
    param.transposeScale1 = ParseBool(fields[index++]);
    param.expectedStatus = ToStatus(fields[index++]);
    param.expectedTransposeCount = ParseIntOrDefault(fields[index++], 0);
    param.expectedBitcastCount = ParseIntOrDefault(fields[index++], 0);
    param.expectedX1Shape = ParseInt64Vec(fields[index++]);
    param.expectedScale1Shape = ParseInt64Vec(fields[index++]);
    return param;
}

static std::vector<QuantGroupedMatmulInplaceAddTransposeFusionPassParam> GetParams()
{
    std::ifstream csvData = OpenCsvData();

    std::vector<QuantGroupedMatmulInplaceAddTransposeFusionPassParam> params;
    std::string line;
    while (std::getline(csvData, line)) {
        if (Trim(line).empty()) {
            continue;
        }
        auto fields = SplitCsvLine(line);
        if (!fields.empty() && fields[0] == "case_name") {
            continue;
        }
        params.emplace_back(ParseParam(fields));
    }
    return params;
}

static std::vector<std::pair<int64_t, int64_t>> MakeShapeRange(const std::vector<int64_t>& shape)
{
    std::vector<std::pair<int64_t, int64_t>> range;
    range.reserve(shape.size());
    for (auto dim : shape) {
        range.emplace_back(dim, dim);
    }
    return range;
}

static TensorDesc MakeTensorDesc(const std::vector<int64_t>& shape, DataType dtype, Format format = FORMAT_ND)
{
    TensorDesc desc(Shape(shape), format, dtype);
    desc.SetOriginFormat(format);
    desc.SetOriginShape(Shape(shape));
    desc.SetShapeRange(MakeShapeRange(shape));
    return desc;
}

static std::vector<int64_t> TransposeLastTwoDims(const std::vector<int64_t>& shape)
{
    auto transposedShape = shape;
    if (transposedShape.size() >= 2) {
        std::swap(transposedShape[transposedShape.size() - 1], transposedShape[transposedShape.size() - 2]);
    }
    return transposedShape;
}

static GNode BuildUnaryNode(Graph* graph, const char* opType, const char* nodeName, const TensorDesc& outputDesc)
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

static void LinkUnaryNode(
    Graph* graph, ge::es::EsGraphBuilder& graphBuilder, TensorWithDesc& input, GNode& node, const TensorDesc& outputDesc)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(), node, 0);
    node.UpdateInputDesc(0, input.desc);
    node.UpdateOutputDesc(0, outputDesc);
    input.tensor = ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
    input.desc = outputDesc;
}

static void AddTransposeIfNeeded(
    Graph* graph, ge::es::EsGraphBuilder& graphBuilder, TensorWithDesc& input, bool enabled, const char* nodeName)
{
    if (!enabled) {
        return;
    }
    auto outputDesc = MakeTensorDesc(TransposeLastTwoDims(input.desc.GetShape().GetDims()), input.desc.GetDataType());
    auto transposeNode = BuildUnaryNode(graph, "Transpose", nodeName, outputDesc);
    LinkUnaryNode(graph, graphBuilder, input, transposeNode, outputDesc);
}

static void AddBitcastIfNeeded(
    Graph* graph, ge::es::EsGraphBuilder& graphBuilder, TensorWithDesc& input, bool enabled, const char* nodeName)
{
    if (!enabled) {
        return;
    }
    auto outputDesc = input.desc;
    outputDesc.SetDataType(DT_HIFLOAT8);
    auto bitcastNode = BuildUnaryNode(graph, "Bitcast", nodeName, outputDesc);
    LinkUnaryNode(graph, graphBuilder, input, bitcastNode, outputDesc);
    DataType bitcastType = DT_HIFLOAT8;
    bitcastNode.SetAttr("type", bitcastType);
}

static void LinkQgmInput(Graph* graph, GNode& qgmNode, const TensorWithDesc& input, int64_t inputIndex)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(),
        qgmNode, inputIndex);
    qgmNode.UpdateInputDesc(inputIndex, input.desc);
}

static GNode BuildQgmNode(Graph* graph, const QuantGroupedMatmulInplaceAddTransposeFusionPassParam& param)
{
    return ge::es::CompliantNodeBuilder(graph)
        .OpType("QuantGroupedMatmulInplaceAdd")
        .Name("QuantGroupedMatmulInplaceAdd")
        .IrDefInputs({
            {"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"scale2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"group_list", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"y", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"scale1", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
        })
        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .InstanceOutputDataType("y", param.outDtype)
        .InstanceOutputShape("y", param.outShape)
        .InstanceOutputFormat("y", FORMAT_ND)
        .Build();
}

static GraphPtr BuildGraph(const QuantGroupedMatmulInplaceAddTransposeFusionPassParam& param)
{
    auto graphBuilder = ge::es::EsGraphBuilder(param.caseName.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc(param.x1Shape, param.x1Dtype);
    auto x2Desc = MakeTensorDesc(param.x2Shape, param.x2Dtype);
    auto scale2Desc = MakeTensorDesc(param.scale2Shape, param.scale2Dtype);
    auto groupListDesc = MakeTensorDesc(param.groupListShape, DT_INT64);
    auto yDesc = MakeTensorDesc(param.yShape, DT_FLOAT);
    auto scale1Desc = MakeTensorDesc(param.scale1Shape, param.scale1Dtype);
    auto outDesc = MakeTensorDesc(param.outShape, param.outDtype);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", param.x1Dtype, FORMAT_ND, param.x1Shape);
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", param.x2Dtype, FORMAT_ND, param.x2Shape);
    auto dataScale2 = graphBuilder.CreateInput(2, "dataScale2", param.scale2Dtype, FORMAT_ND, param.scale2Shape);
    auto dataGroupList = graphBuilder.CreateInput(3, "dataGroupList", DT_INT64, FORMAT_ND, param.groupListShape);
    auto dataY = graphBuilder.CreateInput(4, "dataY", DT_FLOAT, FORMAT_ND, param.yShape);
    auto dataScale1 = graphBuilder.CreateInput(5, "dataScale1", param.scale1Dtype, FORMAT_ND, param.scale1Shape);
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataScale2.GetProducer()->UpdateOutputDesc(0, scale2Desc);
    dataGroupList.GetProducer()->UpdateOutputDesc(0, groupListDesc);
    dataY.GetProducer()->UpdateOutputDesc(0, yDesc);
    dataScale1.GetProducer()->UpdateOutputDesc(0, scale1Desc);

    TensorWithDesc x1Input = {dataX1, x1Desc};
    TensorWithDesc x2Input = {dataX2, x2Desc};
    TensorWithDesc scale2Input = {dataScale2, scale2Desc};
    TensorWithDesc groupListInput = {dataGroupList, groupListDesc};
    TensorWithDesc yInput = {dataY, yDesc};
    TensorWithDesc scale1Input = {dataScale1, scale1Desc};

    AddTransposeIfNeeded(graph, graphBuilder, x1Input, param.transposeX1, "transpose_x1");
    AddBitcastIfNeeded(graph, graphBuilder, x1Input, param.bitcastX1, "bitcast_x1");
    AddTransposeIfNeeded(graph, graphBuilder, scale1Input, param.transposeScale1, "transpose_scale1");

    auto qgmNode = BuildQgmNode(graph, param);
    LinkQgmInput(graph, qgmNode, x1Input, X1_INDEX);
    LinkQgmInput(graph, qgmNode, x2Input, X2_INDEX);
    LinkQgmInput(graph, qgmNode, scale2Input, SCALE2_INDEX);
    LinkQgmInput(graph, qgmNode, groupListInput, GROUP_LIST_INDEX);
    LinkQgmInput(graph, qgmNode, yInput, Y_INDEX);
    LinkQgmInput(graph, qgmNode, scale1Input, SCALE1_INDEX);
    qgmNode.UpdateOutputDesc(0, outDesc);

    auto output = ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(qgmNode, 0));
    return graphBuilder.BuildAndReset({output});
}

static int CountNodes(const GraphPtr& graph, const char* nodeType)
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

static GNode FindQgmNode(const GraphPtr& graph)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && type == "QuantGroupedMatmulInplaceAdd") {
            return node;
        }
    }
    throw std::runtime_error("QuantGroupedMatmulInplaceAdd node is not found.");
}

static void ExpectFusedDesc(const QuantGroupedMatmulInplaceAddTransposeFusionPassParam& param, GNode& qgmNode)
{
    if (param.expectedStatus != SUCCESS) {
        return;
    }

    TensorDesc x1Desc;
    ASSERT_EQ(qgmNode.GetInputDesc(X1_INDEX, x1Desc), GRAPH_SUCCESS);
    EXPECT_EQ(x1Desc.GetShape().GetDims(), param.expectedX1Shape);

    if (!param.expectedScale1Shape.empty()) {
        TensorDesc scale1Desc;
        ASSERT_EQ(qgmNode.GetInputDesc(SCALE1_INDEX, scale1Desc), GRAPH_SUCCESS);
        EXPECT_EQ(scale1Desc.GetShape().GetDims(), param.expectedScale1Shape);
    }
}

static std::string TestName(
    const testing::TestParamInfo<QuantGroupedMatmulInplaceAddTransposeFusionPassParam>& info)
{
    std::string name = info.param.caseName;
    for (auto& ch : name) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) {
            ch = '_';
        }
    }
    return name;
}
} // namespace

class QuantGroupedMatmulInplaceAddTransposeFusionPassTest
    : public testing::TestWithParam<QuantGroupedMatmulInplaceAddTransposeFusionPassParam> {};

TEST_P(QuantGroupedMatmulInplaceAddTransposeFusionPassTest, RunFusionPass)
{
    const auto& param = GetParam();
    auto graph = BuildGraph(param);

    CustomPassContext passContext;
    passContext.SetPassName(PASS_NAME);
    TestQuantGroupedMatmulInplaceAddTransposeFusionPass pass;
    auto status = pass.RunForTest(graph, passContext);

    EXPECT_EQ(status, param.expectedStatus);
    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), param.expectedTransposeCount);
    EXPECT_EQ(CountNodes(graph, "Bitcast"), param.expectedBitcastCount);

    auto qgmNode = FindQgmNode(graph);
    ExpectFusedDesc(param, qgmNode);
}

INSTANTIATE_TEST_CASE_P(QuantGroupedMatmulInplaceAddTransposeFusionPass,
    QuantGroupedMatmulInplaceAddTransposeFusionPassTest, testing::ValuesIn(GetParams()), TestName);
