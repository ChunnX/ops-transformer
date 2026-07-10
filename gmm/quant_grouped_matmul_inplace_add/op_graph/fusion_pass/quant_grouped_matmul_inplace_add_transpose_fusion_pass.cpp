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
 * \file quant_grouped_matmul_inplace_add_transpose_fusion_pass.cpp
 * \brief Delete transpose-like nodes before QuantGroupedMatmulInplaceAdd x1 and scale1 inputs.
 */

#include "quant_grouped_matmul_inplace_add_transpose_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include <algorithm>
#include <iostream>
#include <vector>

#include "acl/acl_rt.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/register_custom_pass.h"

using namespace ge;
using namespace ge::fusion;

namespace ops {
namespace {
constexpr char PASS_NAME[] = "QuantGroupedMatmulInplaceAddTransposeFusionPass";
constexpr char OP_TYPE_QGMM_INPLACE_ADD[] = "QuantGroupedMatmulInplaceAdd";
constexpr char OP_TYPE_TRANSPOSE[] = "Transpose";
constexpr char OP_TYPE_TRANSPOSED[] = "TransposeD";
constexpr char OP_TYPE_RESHAPE[] = "Reshape";
constexpr char OP_TYPE_BITCAST[] = "Bitcast";

constexpr int64_t X1_INDEX = 0;
constexpr int64_t X2_INDEX = 1;
constexpr int64_t X1_SCALE_INDEX = 5;
constexpr int64_t MINI_SHAPE_LEN = 2;
constexpr int64_t THREE_DIMS = 3;

struct TransposeChain {
    GNodePtr transLikeNode = nullptr;
    GNodePtr sourceNode = nullptr;
    int64_t sourceOutputPort = 0;
    int64_t inputIndex = X1_INDEX;
    GNodePtr bitcastNode = nullptr;
    bool relinkToBitcast = false;
    std::vector<GNodePtr> nodesToRemove;
};

bool IsTargetVersion()
{
    int32_t version = 0;
    (void)aclsysGetVersionNum("ge_compiler", &version);
    return version >= GRAPH_FUSION_SUPPORT_VERSION;
}

ge::CustomPassStage GetFusionPassStage()
{
    return IsTargetVersion() ? CustomPassStage::kCompatibleInherited : CustomPassStage::kAfterInferShape;
}

bool IsType(const GNodePtr& node, const char* type)
{
    if (node == nullptr) {
        return false;
    }
    AscendString nodeType;
    return node->GetType(nodeType) == GRAPH_SUCCESS && nodeType == type;
}

bool IsType(const GNode& node, const char* type)
{
    AscendString nodeType;
    return node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == type;
}

GNodePtr GetInputNode(const GNode& dstNode, int64_t dstInputPort, int64_t* srcOutputPort = nullptr)
{
    auto [srcNode, resolvedSrcOutputPort] = dstNode.GetInDataNodesAndPortIndexs(dstInputPort);
    if (srcOutputPort != nullptr) {
        *srcOutputPort = resolvedSrcOutputPort;
    }
    return srcNode;
}

bool IsLegalDataType(DataType dtype, const std::vector<DataType>& legalDtypes)
{
    return std::find(legalDtypes.begin(), legalDtypes.end(), dtype) != legalDtypes.end();
}

bool IsDynamicDesc(const TensorDesc& desc)
{
    for (auto dim : desc.GetShape().GetDims()) {
        if (dim < 0) {
            return true;
        }
    }
    return false;
}

bool IsReshapeTrans(const TensorDesc& inputDesc, bool isDynamic)
{
    if (isDynamic) {
        return true;
    }

    auto dims = inputDesc.GetShape().GetDims();
    if (dims.size() < static_cast<size_t>(MINI_SHAPE_LEN)) {
        return false;
    }
    if (dims.size() == static_cast<size_t>(MINI_SHAPE_LEN)) {
        return dims[dims.size() - 1] == 1 || dims[dims.size() - MINI_SHAPE_LEN] == 1;
    }
    if (dims.size() == static_cast<size_t>(THREE_DIMS)) {
        return dims[0] == 1 || dims[1] == 1;
    }
    return false;
}

bool IsTransposeLikeNode(const GNodePtr& node, const TensorDesc& inputDesc)
{
    if (IsType(node, OP_TYPE_TRANSPOSE) || IsType(node, OP_TYPE_TRANSPOSED)) {
        return true;
    }
    return IsType(node, OP_TYPE_RESHAPE) && IsReshapeTrans(inputDesc, IsDynamicDesc(inputDesc));
}

bool CheckQuantGroupedMatmulInplaceAdd(const GNode& nodeQgm)
{
    return IsType(nodeQgm, OP_TYPE_QGMM_INPLACE_ADD);
}

bool CheckNodeDtype(const GNode& nodeQgm)
{
    static const std::vector<DataType> legalInputDtypes = {DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8};
    static const std::vector<DataType> legalOutputDtypes = {DT_FLOAT};

    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc outputDesc;
    if (nodeQgm.GetInputDesc(X1_INDEX, x1Desc) != GRAPH_SUCCESS ||
        nodeQgm.GetInputDesc(X2_INDEX, x2Desc) != GRAPH_SUCCESS ||
        nodeQgm.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get QuantGroupedMatmulInplaceAdd input or output desc.");
        return false;
    }
    if (!IsLegalDataType(x1Desc.GetDataType(), legalInputDtypes)) {
        OP_LOGW(PASS_NAME, "X1 dtype is %s, which does not support transpose fusion.",
            ge::TypeUtils::DataTypeToSerialString(x1Desc.GetDataType()).c_str());
        return false;
    }
    if (!IsLegalDataType(x2Desc.GetDataType(), legalInputDtypes)) {
        OP_LOGW(PASS_NAME, "X2 dtype is %s, which does not support transpose fusion.",
            ge::TypeUtils::DataTypeToSerialString(x2Desc.GetDataType()).c_str());
        return false;
    }
    if (!IsLegalDataType(outputDesc.GetDataType(), legalOutputDtypes)) {
        OP_LOGW(PASS_NAME, "Output dtype is %s, which does not support transpose fusion.",
            ge::TypeUtils::DataTypeToSerialString(outputDesc.GetDataType()).c_str());
        return false;
    }
    return true;
}

bool CheckNodeShape(const GNode& nodeQgm)
{
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    if (nodeQgm.GetInputDesc(X1_INDEX, x1Desc) != GRAPH_SUCCESS ||
        nodeQgm.GetInputDesc(X2_INDEX, x2Desc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get shape desc before fusion.");
        return false;
    }
    if (x1Desc.GetOriginShape().GetDimNum() < MINI_SHAPE_LEN ||
        x2Desc.GetOriginShape().GetDimNum() < MINI_SHAPE_LEN) {
        OP_LOGW(PASS_NAME, "X1 dim num or X2 dim num does not support transpose fusion.");
        return false;
    }
    return true;
}

bool IsMxScale1(const GNode& nodeQgm)
{
    TensorDesc scale1Desc;
    return nodeQgm.GetInputDesc(X1_SCALE_INDEX, scale1Desc) == GRAPH_SUCCESS &&
           scale1Desc.GetDataType() == DT_FLOAT8_E8M0;
}

bool FillChainFromSource(TransposeChain& chain, const GNodePtr& transLikeNode)
{
    int64_t srcOutputPort = 0;
    auto sourceNode = GetInputNode(*transLikeNode, 0, &srcOutputPort);
    if (sourceNode == nullptr) {
        OP_LOGD(PASS_NAME, "Failed to get source node before transpose-like node.");
        return false;
    }
    chain.transLikeNode = transLikeNode;
    chain.sourceNode = sourceNode;
    chain.sourceOutputPort = srcOutputPort;
    return true;
}

bool ResolveTransposeChain(const GNode& nodeQgm, int64_t inputIndex, TransposeChain& chain)
{
    TensorDesc inputDesc;
    if (nodeQgm.GetInputDesc(inputIndex, inputDesc) != GRAPH_SUCCESS) {
        return false;
    }

    auto transLikeNode = GetInputNode(nodeQgm, inputIndex);
    if (transLikeNode == nullptr) {
        return false;
    }
    if (IsType(transLikeNode, OP_TYPE_BITCAST)) {
        chain.relinkToBitcast = true;
        chain.bitcastNode = transLikeNode;
        transLikeNode = GetInputNode(*transLikeNode, 0);
    }
    if (!IsTransposeLikeNode(transLikeNode, inputDesc) || !FillChainFromSource(chain, transLikeNode)) {
        return false;
    }
    chain.inputIndex = inputIndex;
    chain.nodesToRemove = {transLikeNode};
    return true;
}

bool RemoveNode(const GraphPtr& graph, const GNodePtr& dstNode)
{
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(PASS_NAME, "Graph is null or invalid when removing node.");
        return false;
    }
    if (dstNode == nullptr || !dstNode->GetOutDataNodesAndPortIndexs(0).empty()) {
        return true;
    }

    const auto inputSize = dstNode->GetInputsSize();
    for (size_t dstInputPort = 0; dstInputPort < inputSize; ++dstInputPort) {
        int64_t srcOutputPort = 0;
        auto srcNode = GetInputNode(*dstNode, static_cast<int64_t>(dstInputPort), &srcOutputPort);
        if (srcNode == nullptr) {
            continue;
        }
        if (graph->RemoveEdge(*srcNode, srcOutputPort, *dstNode, dstInputPort) != GRAPH_SUCCESS) {
            OP_LOGW(PASS_NAME, "Failed to remove input edge before removing transpose-like node.");
            return false;
        }
    }
    if (graph->RemoveNode(*dstNode) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to remove unused transpose-like node.");
        return false;
    }
    return true;
}

bool UpdateInputDesc(GNode& nodeQgm, const TransposeChain& chain, int64_t inputIndex, const TensorDesc& sourceDesc)
{
    if (!chain.relinkToBitcast) {
        if (nodeQgm.UpdateInputDesc(inputIndex, sourceDesc) != GRAPH_SUCCESS) {
            OP_LOGW(PASS_NAME, "Failed to update QuantGroupedMatmulInplaceAdd input desc.");
            return false;
        }
        return true;
    }
    if (chain.bitcastNode == nullptr) {
        OP_LOGW(PASS_NAME, "Bitcast node is null when updating desc.");
        return false;
    }

    TensorDesc qgmInputDesc;
    if (nodeQgm.GetInputDesc(inputIndex, qgmInputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get QuantGroupedMatmulInplaceAdd input desc.");
        return false;
    }
    if (chain.bitcastNode->UpdateInputDesc(0, sourceDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update Bitcast input desc.");
        return false;
    }

    TensorDesc bitcastOutputDesc;
    if (chain.bitcastNode->GetOutputDesc(0, bitcastOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get Bitcast output desc.");
        return false;
    }
    bitcastOutputDesc.SetShape(sourceDesc.GetShape());
    bitcastOutputDesc.SetOriginShape(sourceDesc.GetOriginShape());
    if (chain.bitcastNode->UpdateOutputDesc(0, bitcastOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update Bitcast output desc.");
        return false;
    }
    bitcastOutputDesc.SetDataType(qgmInputDesc.GetDataType());
    if (nodeQgm.UpdateInputDesc(inputIndex, bitcastOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update QuantGroupedMatmulInplaceAdd input desc from Bitcast.");
        return false;
    }
    return true;
}

bool RelinkNode(const GraphPtr& graph, GNode& nodeQgm, const TransposeChain& chain, int64_t inputIndex)
{
    if (chain.transLikeNode == nullptr || chain.sourceNode == nullptr) {
        OP_LOGW(PASS_NAME, "Transpose-like chain is invalid.");
        return false;
    }
    TensorDesc sourceDesc;
    if (chain.sourceNode->GetOutputDesc(chain.sourceOutputPort, sourceDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get source output desc.");
        return false;
    }

    auto& dstNode = chain.relinkToBitcast ? *chain.bitcastNode : nodeQgm;
    const int64_t dstInputPort = chain.relinkToBitcast ? 0 : inputIndex;
    if (graph->RemoveEdge(*chain.transLikeNode, 0, dstNode, dstInputPort) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to remove edge from transpose-like node.");
        return false;
    }
    if (graph->AddDataEdge(*chain.sourceNode, chain.sourceOutputPort, dstNode, dstInputPort) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to add edge from source node.");
        return false;
    }
    return UpdateInputDesc(nodeQgm, chain, inputIndex, sourceDesc);
}

void AddNodesToReport(const GNode& nodeQgm, const std::vector<TransposeChain>& chains,
    std::vector<GNode>& nodesBeforeFuse, std::vector<GNodePtr>& nodesToRemove)
{
    nodesBeforeFuse = {nodeQgm};
    nodesToRemove.clear();
    for (const auto& chain : chains) {
        for (const auto& node : chain.nodesToRemove) {
            if (node == nullptr || std::find(nodesToRemove.begin(), nodesToRemove.end(), node) != nodesToRemove.end()) {
                continue;
            }
            nodesBeforeFuse.emplace_back(*node);
            nodesToRemove.emplace_back(node);
        }
    }
}

void ReportTransposeFusion(const std::vector<GNode>& nodesBeforeFuse, const GNode& nodeQgm,
    CustomPassContext& passContext)
{
    auto reportFuse = GraphFuseInspectorUtils::ReportFuse;
    if (reportFuse == nullptr) {
        return;
    }
    if (reportFuse(nodesBeforeFuse, {nodeQgm}, passContext) != SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to report fusion result.");
    }
}

bool RemoveFusionNodes(const GraphPtr& graph, const std::vector<GNodePtr>& nodesToRemove)
{
    for (const auto& nodeToRemove : nodesToRemove) {
        if (!RemoveNode(graph, nodeToRemove)) {
            return false;
        }
    }
    return true;
}

Status PrepareFusion(const GNode& nodeQgm, std::vector<TransposeChain>& chains)
{
    if (!CheckQuantGroupedMatmulInplaceAdd(nodeQgm)) {
        return GRAPH_NOT_CHANGED;
    }
    if (!CheckNodeDtype(nodeQgm) || !CheckNodeShape(nodeQgm)) {
        return GRAPH_NOT_CHANGED;
    }

    TransposeChain x1Chain;
    if (!ResolveTransposeChain(nodeQgm, X1_INDEX, x1Chain)) {
        return GRAPH_NOT_CHANGED;
    }
    chains.emplace_back(x1Chain);

    TransposeChain scale1Chain;
    if (ResolveTransposeChain(nodeQgm, X1_SCALE_INDEX, scale1Chain)) {
        chains.emplace_back(scale1Chain);
        return SUCCESS;
    }
    return IsMxScale1(nodeQgm) ? GRAPH_FAILED : SUCCESS;
}

Status CommitFusion(const GraphPtr& graph, GNode& nodeQgm, const std::vector<TransposeChain>& chains,
    CustomPassContext& passContext)
{
    std::vector<GNode> nodesBeforeFuse;
    std::vector<GNodePtr> nodesToRemove;
    AddNodesToReport(nodeQgm, chains, nodesBeforeFuse, nodesToRemove);

    for (const auto& chain : chains) {
        if (!RelinkNode(graph, nodeQgm, chain, chain.inputIndex)) {
            return GRAPH_FAILED;
        }
    }
    ReportTransposeFusion(nodesBeforeFuse, nodeQgm, passContext);
    if (!RemoveFusionNodes(graph, nodesToRemove)) {
        return GRAPH_FAILED;
    }

    AscendString nodeName;
    const auto getNameRet = nodeQgm.GetName(nodeName);
    std::cout << "[QuantGroupedMatmulInplaceAddTransposeFusionPass] fusion applied, node="
              << (getNameRet == GRAPH_SUCCESS ? nodeName.GetString() : "<unknown>")
              << ", chains=" << chains.size() << std::endl;
    return SUCCESS;
}

Status Fusion(const GraphPtr& graph, GNode& nodeQgm, CustomPassContext& passContext)
{
    std::vector<TransposeChain> chains;
    const auto status = PrepareFusion(nodeQgm, chains);
    if (status != SUCCESS) {
        return status;
    }
    return CommitFusion(graph, nodeQgm, chains, passContext);
}

Status RunQuantGroupedMatmulInplaceAddTransposeFusion(const GraphPtr& graph, CustomPassContext& passContext)
{
    OP_LOGI(PASS_NAME, "==== Enter rebuilt QuantGroupedMatmulInplaceAdd transpose fusion pass ====");
    std::cout << "[QuantGroupedMatmulInplaceAddTransposeFusionPass] enter rebuilt transpose fusion pass" << std::endl;
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(PASS_NAME, "Graph is null or invalid, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }

    std::vector<GNode> nodeQgmList;
    for (auto& node : graph->GetDirectNode()) {
        if (CheckQuantGroupedMatmulInplaceAdd(node)) {
            nodeQgmList.emplace_back(node);
        }
    }
    if (nodeQgmList.empty()) {
        OP_LOGD(PASS_NAME, "No QuantGroupedMatmulInplaceAdd node, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }

    passContext.SetPassName(PASS_NAME);
    bool changed = false;
    for (auto& nodeQgm : nodeQgmList) {
        auto status = Fusion(graph, nodeQgm, passContext);
        if (status == SUCCESS) {
            changed = true;
            continue;
        }
        if (status != GRAPH_NOT_CHANGED) {
            return status;
        }
    }
    OP_LOGI(PASS_NAME, "==== Exit rebuilt QuantGroupedMatmulInplaceAdd transpose fusion pass ====");
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}
} // namespace

Status QuantGroupedMatmulInplaceAddTransposeFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    if (!IsTargetVersion()) {
        return GRAPH_NOT_CHANGED;
    }
    return RunQuantGroupedMatmulInplaceAddTransposeFusion(graph, passContext);
}

REG_FUSION_PASS(QuantGroupedMatmulInplaceAddTransposeFusionPass).Stage(GetFusionPassStage());
} // namespace ops
#endif // CANN_VERSION_NUM
