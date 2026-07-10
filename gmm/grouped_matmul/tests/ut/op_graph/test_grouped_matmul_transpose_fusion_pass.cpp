/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cctype>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "acl/acl_rt.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/graph.h"
#define private public
#include "platform/platform_info.h"
#undef private
#include "register/register_custom_pass.h"

#include "../../../op_graph/fusion_pass/grouped_matmul_transpose_fusion_pass.h"
#include "gmm_csv_parse_utils.h"

using namespace ge;
using namespace ge::fusion;

namespace {
constexpr int32_t kGraphFusionSupportVersion = 90000000;
constexpr int32_t kXIndex = 0;
constexpr int32_t kWeightIndex = 1;
constexpr char kGroupedMatmulType[] = "GroupedMatmul";
constexpr char kTransposeDType[] = "TransposeD";
constexpr char kTestCsvFile[] = "test_grouped_matmul_transpose_fusion_pass.csv";
constexpr size_t kCaseFieldNum = 19;
#ifndef GROUPED_MATMUL_TRANSPOSE_FUSION_PASS_CSV_PATH
#define GROUPED_MATMUL_TRANSPOSE_FUSION_PASS_CSV_PATH ""
#endif

struct GroupedMatmulTransposeFusionPassParam {
    std::string caseName;
    std::vector<int64_t> xShape;
    DataType xDtype;
    std::vector<int64_t> weightShape;
    DataType weightDtype;
    std::vector<int64_t> outShape;
    DataType outDtype;
    bool transposeX;
    bool transposeWeight;
    std::vector<int64_t> xPerm;
    std::vector<int64_t> weightPerm;
    bool initialTransposeX;
    bool initialTransposeWeight;
    Status expectedStatus;
    int expectedTransposeCount;
    bool expectedTransposeX;
    bool expectedTransposeWeight;
    std::vector<int64_t> expectedXShape;
    std::vector<int64_t> expectedWeightShape;
};

struct TensorWithDesc {
    ge::es::EsTensorHolder tensor;
    TensorDesc desc;
};

class TestGroupedMatmulTransposeFusionPass : public ops::GroupedMatmulTransposeFusionPass {
public:
    Status RunForTest(GraphPtr &graph, CustomPassContext &passContext)
    {
        return Run(graph, passContext);
    }
};

std::vector<std::string> SplitCsvLine(const std::string &line)
{
    std::vector<std::string> fields;
    ops::ut::SplitStr2Vec(line, ",", fields);
    for (auto &field : fields) {
        field = ops::ut::Trim(field);
    }
    return fields;
}

DataType ParseDataType(const std::string &value)
{
    if (value == "DT_FLOAT16") {
        return DT_FLOAT16;
    }
    if (value == "DT_BF16") {
        return DT_BF16;
    }
    if (value == "DT_INT8") {
        return DT_INT8;
    }
    throw std::runtime_error("Unsupported data type: " + value);
}

Status ParseStatus(const std::string &value)
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
    throw std::runtime_error("Unsupported status: " + value);
}

GroupedMatmulTransposeFusionPassParam ParseParam(const std::vector<std::string> &fields)
{
    if (fields.size() != kCaseFieldNum) {
        throw std::runtime_error("Invalid csv column size: " + std::to_string(fields.size()));
    }

    size_t index = 0;
    GroupedMatmulTransposeFusionPassParam param;
    param.caseName = fields[index++];
    param.xShape = ops::ut::ParseDims(fields[index++]);
    param.xDtype = ParseDataType(fields[index++]);
    param.weightShape = ops::ut::ParseDims(fields[index++]);
    param.weightDtype = ParseDataType(fields[index++]);
    param.outShape = ops::ut::ParseDims(fields[index++]);
    param.outDtype = ParseDataType(fields[index++]);
    param.transposeX = ops::ut::ParseBool(fields[index++]);
    param.transposeWeight = ops::ut::ParseBool(fields[index++]);
    param.xPerm = ops::ut::ParseDims(fields[index++]);
    param.weightPerm = ops::ut::ParseDims(fields[index++]);
    param.initialTransposeX = ops::ut::ParseBool(fields[index++]);
    param.initialTransposeWeight = ops::ut::ParseBool(fields[index++]);
    param.expectedStatus = ParseStatus(fields[index++]);
    param.expectedTransposeCount = std::stoi(fields[index++]);
    param.expectedTransposeX = ops::ut::ParseBool(fields[index++]);
    param.expectedTransposeWeight = ops::ut::ParseBool(fields[index++]);
    param.expectedXShape = ops::ut::ParseDims(fields[index++]);
    param.expectedWeightShape = ops::ut::ParseDims(fields[index++]);
    return param;
}

std::vector<GroupedMatmulTransposeFusionPassParam> GetParams()
{
    std::string csvPath = GROUPED_MATMUL_TRANSPOSE_FUSION_PASS_CSV_PATH;
    if (csvPath.empty()) {
        csvPath = ops::ut::ResolveCsvPath(kTestCsvFile, "gmm/grouped_matmul/tests/ut/op_graph", __FILE__);
    }
    std::ifstream csvData(csvPath, std::ios::in);
    if (!csvData.is_open()) {
        throw std::runtime_error("Open csv file failed: " + csvPath);
    }

    std::vector<GroupedMatmulTransposeFusionPassParam> params;
    std::string line;
    size_t lineNo = 0;
    while (std::getline(csvData, line)) {
        ++lineNo;
        if (ops::ut::Trim(line).empty()) {
            continue;
        }
        auto fields = SplitCsvLine(line);
        if (!fields.empty() && fields[0] == "case_name") {
            continue;
        }
        try {
            params.emplace_back(ParseParam(fields));
        } catch (const std::exception &error) {
            throw std::runtime_error(
                ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, fields.empty() ? "" : fields[0], error));
        }
    }
    return params;
}

void SetPlatformSupport()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    optionalInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
    fe::PlatformInfoManager::Instance().platform_info_map_[optionalInfo.soc_version] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

void ClearPlatformSupport()
{
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
}

std::vector<std::pair<int64_t, int64_t>> MakeShapeRange(const std::vector<int64_t> &shape)
{
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    shapeRange.reserve(shape.size());
    for (const auto dim : shape) {
        shapeRange.emplace_back(dim, dim);
    }
    return shapeRange;
}

TensorDesc MakeTensorDesc(const std::vector<int64_t> &shape, DataType dtype)
{
    TensorDesc desc(Shape(shape), FORMAT_ND, dtype);
    desc.SetOriginFormat(FORMAT_ND);
    desc.SetOriginShape(Shape(shape));
    desc.SetShapeRange(MakeShapeRange(shape));
    return desc;
}

std::vector<int64_t> PermuteShape(const std::vector<int64_t> &shape, const std::vector<int64_t> &perm)
{
    if (shape.size() != perm.size()) {
        throw std::runtime_error("Shape and perm ranks are different.");
    }
    std::vector<int64_t> outputShape;
    outputShape.reserve(shape.size());
    for (const auto index : perm) {
        if (index < 0 || static_cast<size_t>(index) >= shape.size()) {
            throw std::runtime_error("Perm index is out of range.");
        }
        outputShape.emplace_back(shape[static_cast<size_t>(index)]);
    }
    return outputShape;
}

GNode BuildTransposeDNode(Graph *graph, const char *name, const TensorDesc &outputDesc,
                          const std::vector<int64_t> &perm)
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
    transposeNode.SetAttr("perm", perm);
    return transposeNode;
}

void LinkUnaryNode(Graph *graph, ge::es::EsGraphBuilder &graphBuilder, TensorWithDesc &input, GNode &node,
                   const TensorDesc &outputDesc)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(), node, 0);
    node.UpdateInputDesc(0, input.desc);
    node.UpdateOutputDesc(0, outputDesc);
    input.tensor = ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
    input.desc = outputDesc;
}

void AddTransposeIfNeeded(Graph *graph, ge::es::EsGraphBuilder &graphBuilder, TensorWithDesc &input, bool enabled,
                          const char *name, const std::vector<int64_t> &perm)
{
    if (!enabled) {
        return;
    }
    auto outputDesc = MakeTensorDesc(PermuteShape(input.desc.GetShape().GetDims(), perm), input.desc.GetDataType());
    auto transposeNode = BuildTransposeDNode(graph, name, outputDesc, perm);
    LinkUnaryNode(graph, graphBuilder, input, transposeNode, outputDesc);
}

void LinkGroupedMatmulInput(Graph *graph, GNode &groupedMatmulNode, const TensorWithDesc &input, int32_t inputIndex)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(),
                                     groupedMatmulNode, inputIndex);
    groupedMatmulNode.UpdateInputDesc(inputIndex, input.desc);
}

GraphPtr BuildGraph(const GroupedMatmulTransposeFusionPassParam &param)
{
    auto graphBuilder = ge::es::EsGraphBuilder(param.caseName.c_str());
    auto *graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto xDesc = MakeTensorDesc(param.xShape, param.xDtype);
    auto weightDesc = MakeTensorDesc(param.weightShape, param.weightDtype);
    auto outDesc = MakeTensorDesc(param.outShape, param.outDtype);

    auto dataX = graphBuilder.CreateInput(0, "dataX", param.xDtype, FORMAT_ND, param.xShape);
    auto dataWeight = graphBuilder.CreateInput(1, "dataWeight", param.weightDtype, FORMAT_ND, param.weightShape);
    dataX.GetProducer()->UpdateOutputDesc(0, xDesc);
    dataWeight.GetProducer()->UpdateOutputDesc(0, weightDesc);

    TensorWithDesc xInput = {dataX, xDesc};
    TensorWithDesc weightInput = {dataWeight, weightDesc};
    AddTransposeIfNeeded(graph, graphBuilder, xInput, param.transposeX, "transposeX", param.xPerm);
    AddTransposeIfNeeded(graph, graphBuilder, weightInput, param.transposeWeight, "transposeWeight", param.weightPerm);

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
                                 .InstanceOutputDataType("y", param.outDtype)
                                 .InstanceOutputShape("y", param.outShape)
                                 .InstanceOutputFormat("y", FORMAT_ND)
                                 .Build();
    LinkGroupedMatmulInput(graph, groupedMatmulNode, xInput, kXIndex);
    LinkGroupedMatmulInput(graph, groupedMatmulNode, weightInput, kWeightIndex);
    groupedMatmulNode.UpdateOutputDesc(0, outDesc);
    groupedMatmulNode.SetAttr("transpose_x", param.initialTransposeX);
    groupedMatmulNode.SetAttr("transpose_weight", param.initialTransposeWeight);

    auto output =
        ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(groupedMatmulNode, 0));
    return graphBuilder.BuildAndReset({output});
}

int CountNodes(const GraphPtr &graph, const char *type)
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

GNode FindGroupedMatmulNode(const GraphPtr &graph)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == kGroupedMatmulType) {
            return node;
        }
    }
    throw std::runtime_error("GroupedMatmul node is not found.");
}

std::string TestName(const testing::TestParamInfo<GroupedMatmulTransposeFusionPassParam> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}
} // namespace

extern "C" aclError aclsysGetVersionNum(const char *, int32_t *version)
{
    if (version != nullptr) {
        *version = kGraphFusionSupportVersion;
    }
    return ACL_SUCCESS;
}

extern "C" rtError_t rtGetSocSpec(const char *, const char *, char *value, const uint32_t len)
{
    if (value == nullptr || len == 0) {
        return static_cast<rtError_t>(1);
    }
    const char socVersion[] = "3510";
    std::strncpy(value, socVersion, len - 1);
    value[len - 1] = '\0';
    return RT_ERROR_NONE;
}

class GroupedMatmulTransposeFusionPassTest : public testing::TestWithParam<GroupedMatmulTransposeFusionPassParam> {};

TEST_P(GroupedMatmulTransposeFusionPassTest, RunFusionPass)
{
    const auto &param = GetParam();
    auto graph = BuildGraph(param);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestGroupedMatmulTransposeFusionPass pass;
    SetPlatformSupport();
    const auto status = pass.RunForTest(graph, passContext);
    ClearPlatformSupport();

    ASSERT_EQ(status, param.expectedStatus);
    EXPECT_EQ(CountNodes(graph, kTransposeDType), param.expectedTransposeCount);

    auto groupedMatmulNode = FindGroupedMatmulNode(graph);
    bool transposeX = false;
    bool transposeWeight = false;
    ASSERT_EQ(groupedMatmulNode.GetAttr("transpose_x", transposeX), GRAPH_SUCCESS);
    ASSERT_EQ(groupedMatmulNode.GetAttr("transpose_weight", transposeWeight), GRAPH_SUCCESS);
    EXPECT_EQ(transposeX, param.expectedTransposeX);
    EXPECT_EQ(transposeWeight, param.expectedTransposeWeight);

    TensorDesc xDesc;
    TensorDesc weightDesc;
    ASSERT_EQ(groupedMatmulNode.GetInputDesc(kXIndex, xDesc), GRAPH_SUCCESS);
    ASSERT_EQ(groupedMatmulNode.GetInputDesc(kWeightIndex, weightDesc), GRAPH_SUCCESS);
    EXPECT_EQ(xDesc.GetShape().GetDims(), param.expectedXShape);
    EXPECT_EQ(weightDesc.GetShape().GetDims(), param.expectedWeightShape);
}

INSTANTIATE_TEST_CASE_P(GroupedMatmulTransposeFusionPass, GroupedMatmulTransposeFusionPassTest,
                        testing::ValuesIn(GetParams()), TestName);
