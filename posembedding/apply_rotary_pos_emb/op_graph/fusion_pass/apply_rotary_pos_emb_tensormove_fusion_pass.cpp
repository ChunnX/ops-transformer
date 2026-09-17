/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_tensormove_fusion_pass.cpp
 * \brief Remove TensorMove before ApplyRotaryPosEmb when it is only used by that op.
 *
 *       q           k       cos  sin           q     k    cos  sin
 *       |           |        /   /              \    |    /   /
 *   tensormove  tensormove  /   /      ->       ApplyRotaryPosEmb
 *            \      |      /   /                        |
 *             \     |     /   /                      output
 *             ApplyRotaryPosEmb
 *                    |
 *                 output
 */

#include "apply_rotary_pos_emb_tensormove_fusion_pass.h"

#if CANN_VERSION_NUM >= APPLY_ROTARY_POS_EMB_GRAPH_FUSION_SUPPORT_VERSION
#include <cstdint>
#include <string>
#include <vector>

#include "acl/acl_rt.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"

using namespace ge;
using namespace ge::fusion;

namespace ops {
namespace {
constexpr char kPassName[] = "ApplyRotaryPosEmbTensorMoveFusionPass";
constexpr char kOpTypeApplyRotaryPosEmb[] = "ApplyRotaryPosEmb";
constexpr char kOpTypeTensorMove[] = "TensorMove";
constexpr int32_t kQIndex = 0;
constexpr int32_t kKIndex = 1;
constexpr int32_t kCompatibleInheritedSupportVersion = 90100000;

bool IsGraphFusionRuntimeSupported()
{
    int32_t version = 0;
    if (aclsysGetVersionNum("ge_compiler", &version) != ACL_SUCCESS) {
        OP_LOGW(kPassName, "Failed to get ge compiler version.");
        return false;
    }
    return version >= kCompatibleInheritedSupportVersion;
}

bool IsType(const GNode &node, const char *type)
{
    AscendString nodeType;
    return node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == type;
}

bool IsType(const GNodePtr &node, const char *type)
{
    if (node == nullptr) {
        return false;
    }
    return IsType(*node, type);
}

GNodePtr GetInputNode(const GNode &dstNode, int32_t dstInputPort, int32_t *srcOutputPort = nullptr)
{
    auto [srcNode, resolvedSrcOutputPort] = dstNode.GetInDataNodesAndPortIndexs(dstInputPort);
    if (srcOutputPort != nullptr) {
        *srcOutputPort = resolvedSrcOutputPort;
    }
    return srcNode;
}

std::size_t GetOutputConsumerNum(const GNodePtr &node, int32_t outputPort = 0)
{
    if (node == nullptr) {
        return 0;
    }
    return node->GetOutDataNodesAndPortIndexs(outputPort).size();
}

bool RemoveNodeIfUnused(GraphPtr &graph, const GNodePtr &node)
{
    if (graph == nullptr || !graph->IsValid() || node == nullptr) {
        return true;
    }
    if (!node->GetOutDataNodesAndPortIndexs(0).empty()) {
        return true;
    }

    const auto inputSize = node->GetInputsSize();
    for (std::size_t inputIndex = 0; inputIndex < inputSize; ++inputIndex) {
        int32_t srcOutputPort = 0;
        auto srcNode = GetInputNode(*node, static_cast<int32_t>(inputIndex), &srcOutputPort);
        if (srcNode == nullptr) {
            continue;
        }
        if (graph->RemoveEdge(*srcNode, srcOutputPort, *node, static_cast<int32_t>(inputIndex)) != GRAPH_SUCCESS) {
            OP_LOGW(kPassName, "Failed to remove input edge before removing TensorMove.");
            return false;
        }
    }

    if (graph->RemoveNode(*node) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to remove unused TensorMove node.");
        return false;
    }
    return true;
}

Status RemoveTensorMoveOnInput(GraphPtr &graph, GNode &arpeNode, int32_t inputIndex)
{
    int32_t tmSrcOutputPort = 0;
    auto tmNode = GetInputNode(arpeNode, inputIndex, &tmSrcOutputPort);
    if (tmNode == nullptr) {
        OP_LOGD(kPassName, "ApplyRotaryPosEmb input %d is null.", inputIndex);
        return GRAPH_NOT_CHANGED;
    }
    if (!IsType(tmNode, kOpTypeTensorMove)) {
        OP_LOGD(kPassName, "Node before ApplyRotaryPosEmb(input %d) is not TensorMove.", inputIndex);
        return GRAPH_NOT_CHANGED;
    }
    if (GetOutputConsumerNum(tmNode) != 1) {
        OP_LOGD(kPassName, "ApplyRotaryPosEmb(input %d) TensorMove is used by %zu nodes.", inputIndex,
                GetOutputConsumerNum(tmNode));
        return GRAPH_NOT_CHANGED;
    }

    int32_t sourceOutputPort = 0;
    auto sourceNode = GetInputNode(*tmNode, 0, &sourceOutputPort);
    if (sourceNode == nullptr) {
        OP_LOGD(kPassName, "TensorMove input before ApplyRotaryPosEmb(input %d) is null.", inputIndex);
        return GRAPH_NOT_CHANGED;
    }
    if (GetOutputConsumerNum(sourceNode, sourceOutputPort) != 1) {
        OP_LOGD(kPassName, "Input(%d) data for TensorMove is used by %zu nodes.", inputIndex,
                GetOutputConsumerNum(sourceNode, sourceOutputPort));
        return GRAPH_NOT_CHANGED;
    }

    if (graph->RemoveEdge(*tmNode, tmSrcOutputPort, arpeNode, inputIndex) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to remove edge between TensorMove and ApplyRotaryPosEmb(input %d).", inputIndex);
        return GRAPH_FAILED;
    }
    if (graph->AddDataEdge(*sourceNode, sourceOutputPort, arpeNode, inputIndex) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to add edge from source to ApplyRotaryPosEmb(input %d).", inputIndex);
        return GRAPH_FAILED;
    }

    TensorDesc sourceOutputDesc;
    if (sourceNode->GetOutputDesc(sourceOutputPort, sourceOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to get source output desc for ApplyRotaryPosEmb(input %d).", inputIndex);
        return GRAPH_FAILED;
    }
    if (arpeNode.UpdateInputDesc(inputIndex, sourceOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to update ApplyRotaryPosEmb input desc at index %d.", inputIndex);
        return GRAPH_FAILED;
    }

    if (!RemoveNodeIfUnused(graph, tmNode)) {
        return GRAPH_FAILED;
    }
    return SUCCESS;
}

Status FusionApplyRotaryPosEmbNode(GraphPtr &graph, GNode &arpeNode)
{
    if (!IsType(arpeNode, kOpTypeApplyRotaryPosEmb)) {
        return GRAPH_NOT_CHANGED;
    }

    auto resultInput0 = RemoveTensorMoveOnInput(graph, arpeNode, kQIndex);
    if (resultInput0 == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }
    auto resultInput1 = RemoveTensorMoveOnInput(graph, arpeNode, kKIndex);
    if (resultInput1 == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }
    if (resultInput0 == GRAPH_NOT_CHANGED && resultInput1 == GRAPH_NOT_CHANGED) {
        OP_LOGD(kPassName, "ApplyRotaryPosEmb TensorMove fusion does not match.");
        return GRAPH_NOT_CHANGED;
    }
    OP_LOGI(kPassName, "ApplyRotaryPosEmb TensorMove fusion success.");
    return SUCCESS;
}

Status RunApplyRotaryPosEmbTensorMoveFusion(GraphPtr &graph, CustomPassContext &passContext)
{
    OP_LOGD(kPassName, "Enter ApplyRotaryPosEmb TensorMove fusion pass.");
    if (!IsGraphFusionRuntimeSupported()) {
        OP_LOGD(kPassName, "Skip pass because graph fusion runtime is unsupported.");
        return GRAPH_NOT_CHANGED;
    }
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(kPassName, "Graph is null or invalid.");
        return GRAPH_NOT_CHANGED;
    }

    std::vector<GNode> arpeNodes;
    for (auto &node : graph->GetDirectNode()) {
        if (IsType(node, kOpTypeApplyRotaryPosEmb)) {
            arpeNodes.emplace_back(node);
        }
    }
    if (arpeNodes.empty()) {
        return GRAPH_NOT_CHANGED;
    }

    passContext.SetPassName(kPassName);
    bool changed = false;
    for (auto &arpeNode : arpeNodes) {
        auto status = FusionApplyRotaryPosEmbNode(graph, arpeNode);
        if (status == SUCCESS) {
            changed = true;
            continue;
        }
        if (status != GRAPH_NOT_CHANGED) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(kPassName, "graph",
                                                     "ApplyRotaryPosEmb TensorMove fusion failed, status = " +
                                                         std::to_string(static_cast<unsigned int>(status)));
            return status;
        }
    }
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}
} // namespace

Status ApplyRotaryPosEmbTensorMoveFusionPass::Run(GraphPtr &graph, CustomPassContext &passContext)
{
    return RunApplyRotaryPosEmbTensorMoveFusion(graph, passContext);
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(ApplyRotaryPosEmbTensorMoveFusionPass).Stage(CustomPassStage::kCompatibleInherited);
#endif
} // namespace ops
#endif // CANN_VERSION_NUM >= APPLY_ROTARY_POS_EMB_GRAPH_FUSION_SUPPORT_VERSION
