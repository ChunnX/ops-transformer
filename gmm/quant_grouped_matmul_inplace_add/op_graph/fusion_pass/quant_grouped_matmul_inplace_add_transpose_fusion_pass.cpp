/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_grouped_matmul_inplace_add_transpose_fusion_pass.h"

#if CANN_VERSION_NUM >= QGMM_INPLACE_ADD_GRAPH_FUSION_SUPPORT_VERSION
#include <algorithm>
#include <cstdint>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"

namespace ops {
namespace {
constexpr char kPassName[] = "QuantGroupedMatmulInplaceAddTransposeFusionPass";
constexpr char kTargetType[] = "QuantGroupedMatmulInplaceAdd";
constexpr char kTransposeType[] = "Transpose";
constexpr char kTransposeDType[] = "TransposeD";
constexpr char kReshapeType[] = "Reshape";
constexpr char kBitcastType[] = "Bitcast";
constexpr char kConstType[] = "Const";
constexpr int32_t kX1Index = 0;
constexpr int32_t kX2Index = 1;
constexpr int32_t kScale1Index = 5;
constexpr int32_t kPermIndex = 1;
const std::vector<int64_t> kX1Perm = {1, 0};
const std::vector<int64_t> kScale1Perm = {1, 0, 2};

struct InputPattern {
    ge::GNodePtr bitcast = nullptr;
    ge::GNodePtr transposeLike = nullptr;
    ge::GNodePtr source = nullptr;
    ge::GNodePtr oldSource = nullptr;
    int32_t sourcePort = 0;
};

bool IsType(const ge::GNodePtr& node, const char* expected)
{
    if (node == nullptr) {
        return false;
    }
    ge::AscendString type;
    return node->GetType(type) == ge::GRAPH_SUCCESS && type == expected;
}

bool IsType(const ge::GNode& node, const char* expected)
{
    ge::AscendString type;
    return node.GetType(type) == ge::GRAPH_SUCCESS && type == expected;
}

ge::GNodePtr GetInputNode(const ge::GNode& node, int32_t inputIndex, int32_t* outputPort = nullptr)
{
    auto [inputNode, inputPort] = node.GetInDataNodesAndPortIndexs(inputIndex);
    if (outputPort != nullptr) {
        *outputPort = inputPort;
    }
    return inputNode;
}

bool BuildInputPattern(ge::GNode& target, int32_t inputIndex, InputPattern& pattern)
{
    auto immediateInput = GetInputNode(target, inputIndex);
    if (IsType(immediateInput, kBitcastType)) {
        pattern.bitcast = immediateInput;
        pattern.transposeLike = GetInputNode(*immediateInput, 0);
    } else {
        pattern.transposeLike = immediateInput;
    }
    if (pattern.transposeLike == nullptr) {
        return false;
    }
    pattern.source = GetInputNode(*pattern.transposeLike, 0, &pattern.sourcePort);
    pattern.oldSource = pattern.transposeLike;
    return pattern.source != nullptr;
}

bool GetTransposePerm(const ge::GNodePtr& node, std::vector<int64_t>& perm)
{
    if (IsType(node, kTransposeDType)) {
        return node->GetAttr("perm", perm) == ge::GRAPH_SUCCESS;
    }
    if (!IsType(node, kTransposeType)) {
        return false;
    }
    ge::Tensor value;
    ge::TensorDesc desc;
    if (node->GetInputConstData(kPermIndex, value) != ge::GRAPH_SUCCESS ||
        node->GetInputDesc(kPermIndex, desc) != ge::GRAPH_SUCCESS || value.GetData() == nullptr) {
        return false;
    }
    const auto count = desc.GetDataType() == ge::DT_INT32 ? value.GetSize() / sizeof(int32_t) :
        (desc.GetDataType() == ge::DT_INT64 ? value.GetSize() / sizeof(int64_t) : 0U);
    for (std::size_t i = 0; i < count; ++i) {
        perm.emplace_back(desc.GetDataType() == ge::DT_INT32 ?
            static_cast<int64_t>(reinterpret_cast<const int32_t*>(value.GetData())[i]) :
            reinterpret_cast<const int64_t*>(value.GetData())[i]);
    }
    return count != 0U;
}

bool IsStaticEquivalentReshape(const ge::GNodePtr& node, const std::vector<int64_t>& perm)
{
    if (!IsType(node, kReshapeType)) {
        return false;
    }
    ge::TensorDesc inputDesc;
    ge::TensorDesc outputDesc;
    if (node->GetInputDesc(0, inputDesc) != ge::GRAPH_SUCCESS ||
        node->GetOutputDesc(0, outputDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    auto inputDims = inputDesc.GetShape().GetDims();
    const auto outputDims = outputDesc.GetShape().GetDims();
    if (inputDims.size() != perm.size() || outputDims.size() != perm.size() ||
        std::any_of(inputDims.begin(), inputDims.end(), [](int64_t dim) { return dim < 0; }) ||
        std::any_of(outputDims.begin(), outputDims.end(), [](int64_t dim) { return dim < 0; })) {
        return false;
    }
    std::vector<int64_t> expected(inputDims.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] < 0 || static_cast<std::size_t>(perm[i]) >= inputDims.size()) {
            return false;
        }
        expected[i] = inputDims[static_cast<std::size_t>(perm[i])];
    }
    const bool hasUnitTransposeAxis = inputDims[0] == 1 || inputDims[1] == 1;
    return hasUnitTransposeAxis && expected == outputDims;
}

bool CheckPattern(const InputPattern& pattern, const std::vector<int64_t>& expectedPerm)
{
    if (pattern.bitcast != nullptr && pattern.bitcast->GetOutDataNodesAndPortIndexs(0).size() != 1U) {
        return false;
    }
    if (pattern.transposeLike->GetOutDataNodesAndPortIndexs(0).size() != 1U) {
        return false;
    }
    if (IsType(pattern.transposeLike, kReshapeType)) {
        return IsStaticEquivalentReshape(pattern.transposeLike, expectedPerm);
    }
    std::vector<int64_t> actualPerm;
    return GetTransposePerm(pattern.transposeLike, actualPerm) && actualPerm == expectedPerm;
}

bool CheckTarget(ge::GNode& target)
{
    ge::TensorDesc x1Desc;
    ge::TensorDesc x2Desc;
    ge::TensorDesc outputDesc;
    if (target.GetInputDesc(kX1Index, x1Desc) != ge::GRAPH_SUCCESS ||
        target.GetInputDesc(kX2Index, x2Desc) != ge::GRAPH_SUCCESS ||
        target.GetOutputDesc(0, outputDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    const std::vector<ge::DataType> supported = {
        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8};
    return std::find(supported.begin(), supported.end(), x1Desc.GetDataType()) != supported.end() &&
        std::find(supported.begin(), supported.end(), x2Desc.GetDataType()) != supported.end() &&
        outputDesc.GetDataType() == ge::DT_FLOAT && x1Desc.GetShape().GetDimNum() >= 2U &&
        x2Desc.GetShape().GetDimNum() >= 2U;
}

bool IsMxMode(ge::GNode& target)
{
    ge::TensorDesc scale1Desc;
    return target.GetInputDesc(kScale1Index, scale1Desc) == ge::GRAPH_SUCCESS &&
        scale1Desc.GetDataType() == ge::DT_FLOAT8_E8M0;
}

bool UpdateBitcast(ge::GNode& bitcast, const ge::TensorDesc& sourceDesc, ge::TensorDesc& targetDesc)
{
    ge::TensorDesc bitcastOutput;
    if (bitcast.GetOutputDesc(0, bitcastOutput) != ge::GRAPH_SUCCESS ||
        bitcast.UpdateInputDesc(0, sourceDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    bitcastOutput.SetShape(sourceDesc.GetShape());
    bitcastOutput.SetOriginShape(sourceDesc.GetOriginShape());
    bitcastOutput.SetFormat(sourceDesc.GetFormat());
    bitcastOutput.SetOriginFormat(sourceDesc.GetOriginFormat());
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    if (sourceDesc.GetShapeRange(shapeRange) == ge::GRAPH_SUCCESS) {
        (void)bitcastOutput.SetShapeRange(shapeRange);
    }
    if (bitcast.UpdateOutputDesc(0, bitcastOutput) != ge::GRAPH_SUCCESS) {
        return false;
    }
    targetDesc = bitcastOutput;
    return true;
}

bool RelinkInput(ge::GraphPtr& graph, ge::GNode& target, int32_t inputIndex, const InputPattern& pattern)
{
    ge::GNode& destination = pattern.bitcast == nullptr ? target : *pattern.bitcast;
    const int32_t destinationPort = pattern.bitcast == nullptr ? inputIndex : 0;
    if (graph->RemoveEdge(*pattern.oldSource, 0, destination, destinationPort) != ge::GRAPH_SUCCESS ||
        graph->AddDataEdge(*pattern.source, pattern.sourcePort, destination, destinationPort) != ge::GRAPH_SUCCESS) {
        return false;
    }
    ge::TensorDesc sourceDesc;
    if (pattern.source->GetOutputDesc(pattern.sourcePort, sourceDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    ge::TensorDesc targetDesc = sourceDesc;
    if (pattern.bitcast != nullptr && !UpdateBitcast(*pattern.bitcast, sourceDesc, targetDesc)) {
        return false;
    }
    return target.UpdateInputDesc(inputIndex, targetDesc) == ge::GRAPH_SUCCESS;
}

bool RemoveNodeIfUnused(ge::GraphPtr& graph, const ge::GNodePtr& node)
{
    if (node == nullptr || !node->GetOutDataNodesAndPortIndexs(0).empty()) {
        return true;
    }
    std::vector<ge::GNodePtr> constNodes;
    for (std::size_t i = 0; i < node->GetInputsSize(); ++i) {
        int32_t sourcePort = 0;
        auto source = GetInputNode(*node, static_cast<int32_t>(i), &sourcePort);
        if (source == nullptr) {
            continue;
        }
        if (graph->RemoveEdge(*source, sourcePort, *node, static_cast<int32_t>(i)) != ge::GRAPH_SUCCESS) {
            return false;
        }
        if (IsType(source, kConstType)) {
            constNodes.emplace_back(source);
        }
    }
    if (graph->RemoveNode(*node) != ge::GRAPH_SUCCESS) {
        return false;
    }
    for (const auto& constNode : constNodes) {
        if (!RemoveNodeIfUnused(graph, constNode)) {
            return false;
        }
    }
    return true;
}

ge::Status FuseNode(ge::GraphPtr& graph, ge::GNode& target)
{
    if (!CheckTarget(target)) {
        return ge::GRAPH_NOT_CHANGED;
    }
    InputPattern x1Pattern;
    if (!BuildInputPattern(target, kX1Index, x1Pattern) || !CheckPattern(x1Pattern, kX1Perm)) {
        return ge::GRAPH_NOT_CHANGED;
    }
    InputPattern scale1Pattern;
    const bool isMx = IsMxMode(target);
    if (isMx && (!BuildInputPattern(target, kScale1Index, scale1Pattern) ||
        !CheckPattern(scale1Pattern, kScale1Perm))) {
        return ge::GRAPH_NOT_CHANGED;
    }
    if (!RelinkInput(graph, target, kX1Index, x1Pattern) ||
        (isMx && !RelinkInput(graph, target, kScale1Index, scale1Pattern))) {
        return ge::GRAPH_FAILED;
    }
    if (!RemoveNodeIfUnused(graph, x1Pattern.transposeLike) ||
        (isMx && !RemoveNodeIfUnused(graph, scale1Pattern.transposeLike))) {
        return ge::GRAPH_FAILED;
    }
    return ge::SUCCESS;
}

ge::Status RunFusion(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    if (graph == nullptr || !graph->IsValid()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    passContext.SetPassName(kPassName);
    std::vector<ge::GNode> targets;
    for (auto& node : graph->GetDirectNode()) {
        if (IsType(node, kTargetType)) {
            targets.emplace_back(node);
        }
    }
    bool changed = false;
    for (auto& target : targets) {
        const auto status = FuseNode(graph, target);
        if (status == ge::SUCCESS) {
            changed = true;
        } else if (status != ge::GRAPH_NOT_CHANGED) {
            return status;
        }
    }
    return changed ? ge::SUCCESS : ge::GRAPH_NOT_CHANGED;
}
} // namespace

ge::Status QuantGroupedMatmulInplaceAddTransposeFusionPass::Run(
    ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    return RunFusion(graph, passContext);
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(QuantGroupedMatmulInplaceAddTransposeFusionPass)
    .Stage(ge::fusion::CustomPassStage::kCompatibleInherited);
#endif
} // namespace ops
#endif
