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
 * \file grouped_matmul_finalize_routing_transpose_fusion_pass.cpp
 * \brief Delete transpose-like nodes before GroupedMatmulFinalizeRouting weight inputs and set transpose_w.
 */

#include "grouped_matmul_finalize_routing_transpose_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

using namespace ge;
using namespace ge::fusion;

namespace ge {
namespace fusion {
class GraphFuseInspectorUtils {
public:
    static Status ReportFuse(const std::vector<GNode> &nodesBeforeFuse, const std::vector<GNode> &nodesAfterFuse,
                             CustomPassContext &ctx) __attribute__((weak));
};
} // namespace fusion
} // namespace ge

namespace ops {
namespace {
constexpr char PASS_NAME[] = "GroupedMatmulFinalizeRoutingTransposeFusionPass";
constexpr char OP_TYPE_GMMFR[] = "GroupedMatmulFinalizeRouting";
constexpr char OP_TYPE_TRANSPOSE[] = "Transpose";
constexpr char OP_TYPE_TRANSPOSED[] = "TransposeD";
constexpr char OP_TYPE_RESHAPE[] = "Reshape";
constexpr char OP_TYPE_BITCAST[] = "Bitcast";
constexpr char ATTR_TRANSPOSE_WEIGHT[] = "transpose_w";
constexpr char ASCEND_950_SOC_VERSION[] = "Ascend950";

constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT_INDEX = 1;
constexpr int64_t WEIGHT_SCALE_INDEX = 2;
constexpr int64_t MINI_SHAPE_LEN = 2;
constexpr int64_t WEIGHT_MINI_SHAPE_LEN = 3;
constexpr size_t INDEX_THREE = 2;
constexpr int64_t DOUBLE_SHAPE = 2;
constexpr int64_t PERM_INPUT_INDEX = 1;
const std::vector<int64_t> PERM_WEIGHT = {0, 2, 1};
const std::vector<int64_t> PERM_WEIGHT_RESHAPE = {1, 0};
const std::vector<int64_t> PERM_WEIGHT_SCALE = {0, 2, 1, 3};
const std::vector<int64_t> PERM_WEIGHT_SCALE_RESHAPE = {1, 0, 2};

struct TransposeChain {
    GNodePtr transLikeNode = nullptr;
    GNodePtr sourceNode = nullptr;
    int64_t sourceOutputPort = 0;
    int64_t inputIndex = WEIGHT_INDEX;
    GNodePtr bitcastNode = nullptr;
    GNodePtr tailReshapeNode = nullptr;
    bool relinkToBitcast = false;
    bool specialFp4Path = false;
    std::vector<GNodePtr> nodesToRemove;
};

bool IsType(const GNodePtr &node, const char *type)
{
    if (node == nullptr) {
        return false;
    }
    AscendString nodeType;
    return node->GetType(nodeType) == GRAPH_SUCCESS && nodeType == type;
}

bool IsType(const GNode &node, const char *type)
{
    AscendString nodeType;
    return node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == type;
}

bool IsTransposeType(const GNodePtr &node)
{
    return IsType(node, OP_TYPE_TRANSPOSE) || IsType(node, OP_TYPE_TRANSPOSED);
}

bool IsTransposeLikeType(const GNodePtr &node)
{
    return IsTransposeType(node) || IsType(node, OP_TYPE_RESHAPE);
}

GNodePtr GetInputNode(const GNode &dstNode, int64_t dstInputPort, int64_t *srcOutputPort = nullptr)
{
    auto [srcNode, resolvedSrcOutputPort] = dstNode.GetInDataNodesAndPortIndexs(dstInputPort);
    if (srcOutputPort != nullptr) {
        *srcOutputPort = resolvedSrcOutputPort;
    }
    return srcNode;
}

std::size_t GetOutputConsumerNum(const GNodePtr &node)
{
    return node == nullptr ? 0 : node->GetOutDataNodesAndPortIndexs(0).size();
}

bool IsUnknownDim(int64_t dim)
{
    return dim < 0;
}

bool IsSameOrBothUnknown(int64_t lhs, int64_t rhs)
{
    return lhs == rhs || (IsUnknownDim(lhs) && IsUnknownDim(rhs));
}

bool GetTransposePerm(const GNodePtr &transposeNode, std::vector<int64_t> &perm)
{
    if (IsType(transposeNode, OP_TYPE_TRANSPOSED)) {
        return transposeNode->GetAttr("perm", perm) == GRAPH_SUCCESS;
    }
    if (!IsType(transposeNode, OP_TYPE_TRANSPOSE)) {
        return false;
    }
    Tensor permTensor;
    TensorDesc permDesc;
    if (transposeNode->GetInputConstData(PERM_INPUT_INDEX, permTensor) != GRAPH_SUCCESS ||
        transposeNode->GetInputDesc(PERM_INPUT_INDEX, permDesc) != GRAPH_SUCCESS || permTensor.GetData() == nullptr) {
        return false;
    }
    const auto *data = permTensor.GetData();
    if (permDesc.GetDataType() == DT_INT32) {
        const auto size = permTensor.GetSize() / sizeof(int32_t);
        for (std::size_t i = 0; i < size; ++i) {
            perm.emplace_back(static_cast<int64_t>(reinterpret_cast<const int32_t *>(data)[i]));
        }
        return true;
    }
    if (permDesc.GetDataType() == DT_INT64) {
        const auto size = permTensor.GetSize() / sizeof(int64_t);
        for (std::size_t i = 0; i < size; ++i) {
            perm.emplace_back(reinterpret_cast<const int64_t *>(data)[i]);
        }
        return true;
    }
    return false;
}

std::vector<int64_t> GetExpectedPerm(int64_t inputIndex, bool reshapePattern)
{
    if (inputIndex == WEIGHT_SCALE_INDEX) {
        return reshapePattern ? PERM_WEIGHT_SCALE_RESHAPE : PERM_WEIGHT_SCALE;
    }
    return reshapePattern ? PERM_WEIGHT_RESHAPE : PERM_WEIGHT;
}

bool CheckTransposeNode(const GNodePtr &transposeNode, int64_t inputIndex, bool reshapePattern)
{
    std::vector<int64_t> perm;
    return GetOutputConsumerNum(transposeNode) == 1 && GetTransposePerm(transposeNode, perm) &&
           perm == GetExpectedPerm(inputIndex, reshapePattern);
}

bool GetShapeDims(const GNodePtr &node, bool input, std::vector<int64_t> &dims)
{
    TensorDesc desc;
    const auto status = input ? node->GetInputDesc(0, desc) : node->GetOutputDesc(0, desc);
    if (status != GRAPH_SUCCESS) {
        return false;
    }
    dims = desc.GetShape().GetDims();
    return true;
}

bool IsExpectedSwap(const std::vector<int64_t> &inputDims, const std::vector<int64_t> &outputDims, int64_t inputIndex,
                    bool requireUnitAxis)
{
    if (inputDims.size() != outputDims.size() || inputDims.size() < 3) {
        return false;
    }
    std::size_t firstAxis = inputDims.size() - 2;
    std::size_t secondAxis = inputDims.size() - 1;
    if (inputIndex == WEIGHT_SCALE_INDEX) {
        if (inputDims.size() != 4) {
            return false;
        }
        firstAxis = 1;
        secondAxis = 2;
    }
    if (requireUnitAxis && inputDims[firstAxis] != 1 && inputDims[secondAxis] != 1) {
        return false;
    }
    auto expectedDims = inputDims;
    std::swap(expectedDims[firstAxis], expectedDims[secondAxis]);
    for (std::size_t i = 0; i < expectedDims.size(); ++i) {
        if (!IsSameOrBothUnknown(expectedDims[i], outputDims[i])) {
            return false;
        }
    }
    return true;
}

bool CheckDirectReshape(const GNodePtr &reshapeNode, int64_t inputIndex)
{
    std::vector<int64_t> inputDims;
    std::vector<int64_t> outputDims;
    return GetOutputConsumerNum(reshapeNode) == 1 && GetShapeDims(reshapeNode, true, inputDims) &&
           GetShapeDims(reshapeNode, false, outputDims) && IsExpectedSwap(inputDims, outputDims, inputIndex, true);
}

bool IsLegalDataType(DataType dtype, const std::vector<DataType> &legalDtypes)
{
    return std::find(legalDtypes.begin(), legalDtypes.end(), dtype) != legalDtypes.end();
}

bool IsSpecialFp4Path(const GNodePtr &node, int64_t inputIndex)
{
    if (inputIndex != WEIGHT_INDEX || !IsType(node, OP_TYPE_RESHAPE)) {
        return false;
    }
    TensorDesc outputDesc;
    if (node->GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS) {
        return false;
    }
    auto dtype = outputDesc.GetDataType();
    return dtype == DT_FLOAT4_E2M1;
}

bool CheckPlatformIsAscend950()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get platform information.");
        return false;
    }
    return platformInfo.str_info.short_soc_version == ASCEND_950_SOC_VERSION;
}

bool CheckGroupedMatmulFinalizeRouting(const GNode &nodeGmmFr)
{
    return IsType(nodeGmmFr, OP_TYPE_GMMFR);
}

bool CheckNodeDtype(const GNode &nodeGmmFr)
{
    static const std::vector<DataType> legalInputDtypes = {DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1, DT_INT8,
                                                           DT_HIFLOAT8};
    TensorDesc xDesc;
    TensorDesc weightDesc;
    if (nodeGmmFr.GetInputDesc(X_INDEX, xDesc) != GRAPH_SUCCESS ||
        nodeGmmFr.GetInputDesc(WEIGHT_INDEX, weightDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get GroupedMatmulFinalizeRouting input desc.");
        return false;
    }
    if (!IsLegalDataType(xDesc.GetDataType(), legalInputDtypes)) {
        OP_LOGW(PASS_NAME, "X dtype is %s, which does not support transpose fusion.",
                ge::TypeUtils::DataTypeToSerialString(xDesc.GetDataType()).c_str());
        return false;
    }
    if (!IsLegalDataType(weightDesc.GetDataType(), legalInputDtypes)) {
        OP_LOGW(PASS_NAME, "Weight dtype is %s, which does not support transpose fusion.",
                ge::TypeUtils::DataTypeToSerialString(weightDesc.GetDataType()).c_str());
        return false;
    }
    return true;
}

bool CheckNodeShape(const GNode &nodeGmmFr)
{
    TensorDesc xDesc;
    TensorDesc weightDesc;
    if (nodeGmmFr.GetInputDesc(X_INDEX, xDesc) != GRAPH_SUCCESS ||
        nodeGmmFr.GetInputDesc(WEIGHT_INDEX, weightDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get shape desc before fusion.");
        return false;
    }
    if (xDesc.GetOriginShape().GetDimNum() < MINI_SHAPE_LEN ||
        weightDesc.GetOriginShape().GetDimNum() < WEIGHT_MINI_SHAPE_LEN) {
        OP_LOGW(PASS_NAME, "X dim num or weight dim num does not support transpose fusion.");
        return false;
    }
    return true;
}

bool IsMxWeightScale(const GNode &nodeGmmFr)
{
    TensorDesc weightScaleDesc;
    return nodeGmmFr.GetInputDesc(WEIGHT_SCALE_INDEX, weightScaleDesc) == GRAPH_SUCCESS &&
           weightScaleDesc.GetDataType() == DT_FLOAT8_E8M0;
}

bool FillChainFromSource(TransposeChain &chain, const GNodePtr &transLikeNode)
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

bool TryResolveTwoReshapeChain(const GNode &nodeGmmFr, int64_t inputIndex, TransposeChain &chain)
{
    auto bottomNode = GetInputNode(nodeGmmFr, inputIndex);
    if (bottomNode == nullptr) {
        return false;
    }
    if (IsSpecialFp4Path(bottomNode, inputIndex)) {
        chain.specialFp4Path = true;
        chain.tailReshapeNode = bottomNode;
        bottomNode = GetInputNode(*bottomNode, 0);
    }
    if (IsType(bottomNode, OP_TYPE_BITCAST)) {
        chain.relinkToBitcast = true;
        chain.bitcastNode = bottomNode;
        bottomNode = GetInputNode(*bottomNode, 0);
    }
    if (!IsType(bottomNode, OP_TYPE_RESHAPE)) {
        return false;
    }
    auto transposeNode = GetInputNode(*bottomNode, 0);
    if (!IsTransposeType(transposeNode)) {
        return false;
    }
    auto topReshapeNode = GetInputNode(*transposeNode, 0);
    if (!IsType(topReshapeNode, OP_TYPE_RESHAPE)) {
        return false;
    }
    int64_t srcOutputPort = 0;
    auto sourceNode = GetInputNode(*topReshapeNode, 0, &srcOutputPort);
    if (sourceNode == nullptr) {
        return false;
    }

    TensorDesc sourceDesc;
    TensorDesc targetDesc;
    if (sourceNode->GetOutputDesc(srcOutputPort, sourceDesc) != GRAPH_SUCCESS ||
        nodeGmmFr.GetInputDesc(inputIndex, targetDesc) != GRAPH_SUCCESS ||
        !IsExpectedSwap(sourceDesc.GetShape().GetDims(), targetDesc.GetShape().GetDims(), inputIndex, false) ||
        !CheckTransposeNode(transposeNode, inputIndex, true) || GetOutputConsumerNum(topReshapeNode) != 1 ||
        GetOutputConsumerNum(bottomNode) != 1 ||
        (chain.bitcastNode != nullptr && GetOutputConsumerNum(chain.bitcastNode) != 1) ||
        (chain.tailReshapeNode != nullptr && GetOutputConsumerNum(chain.tailReshapeNode) != 1)) {
        return false;
    }
    chain.inputIndex = inputIndex;
    chain.transLikeNode = bottomNode;
    chain.sourceNode = sourceNode;
    chain.sourceOutputPort = srcOutputPort;
    chain.nodesToRemove = {bottomNode, transposeNode, topReshapeNode};
    return true;
}

bool TryResolveDirectChain(const GNode &nodeGmmFr, int64_t inputIndex, TransposeChain &chain)
{
    auto transLikeNode = GetInputNode(nodeGmmFr, inputIndex);
    if (transLikeNode == nullptr) {
        return false;
    }
    if (IsSpecialFp4Path(transLikeNode, inputIndex)) {
        chain.specialFp4Path = true;
        chain.tailReshapeNode = transLikeNode;
        transLikeNode = GetInputNode(*transLikeNode, 0);
    }
    if (IsType(transLikeNode, OP_TYPE_BITCAST)) {
        chain.relinkToBitcast = true;
        chain.bitcastNode = transLikeNode;
        transLikeNode = GetInputNode(*transLikeNode, 0);
    }
    if (!IsTransposeLikeType(transLikeNode) || !FillChainFromSource(chain, transLikeNode)) {
        return false;
    }
    if ((IsTransposeType(transLikeNode) && !CheckTransposeNode(transLikeNode, inputIndex, false)) ||
        (IsType(transLikeNode, OP_TYPE_RESHAPE) && !CheckDirectReshape(transLikeNode, inputIndex)) ||
        (chain.bitcastNode != nullptr && GetOutputConsumerNum(chain.bitcastNode) != 1) ||
        (chain.tailReshapeNode != nullptr && GetOutputConsumerNum(chain.tailReshapeNode) != 1)) {
        return false;
    }
    chain.inputIndex = inputIndex;
    chain.nodesToRemove = {transLikeNode};
    return true;
}

bool ResolveTransposeChain(const GNode &nodeGmmFr, int64_t inputIndex, TransposeChain &chain)
{
    if (TryResolveTwoReshapeChain(nodeGmmFr, inputIndex, chain)) {
        return true;
    }
    chain = TransposeChain();
    return TryResolveDirectChain(nodeGmmFr, inputIndex, chain);
}

bool RemoveNode(const GraphPtr &graph, const GNodePtr &dstNode)
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

bool UpdateSpecialFp4Desc(GNode &nodeGmmFr, const TransposeChain &chain, int64_t inputIndex,
                          const TensorDesc &sourceDesc)
{
    if (chain.bitcastNode == nullptr || chain.tailReshapeNode == nullptr) {
        OP_LOGW(PASS_NAME, "Special FP4 path misses Bitcast or tail Reshape node.");
        return false;
    }
    auto dims = sourceDesc.GetShape().GetDims();
    if (dims.size() <= INDEX_THREE) {
        OP_LOGW(PASS_NAME, "Special FP4 path source shape dims are too small.");
        return false;
    }

    TensorDesc bitcastOutputDesc;
    if (chain.bitcastNode->GetOutputDesc(0, bitcastOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get Bitcast output desc.");
        return false;
    }
    Shape bitcastShape({dims[0], dims[1], dims[INDEX_THREE], DOUBLE_SHAPE});
    bitcastOutputDesc.SetShape(bitcastShape);
    bitcastOutputDesc.SetOriginShape(bitcastShape);
    if (chain.bitcastNode->UpdateOutputDesc(0, bitcastOutputDesc) != GRAPH_SUCCESS ||
        chain.tailReshapeNode->UpdateInputDesc(0, bitcastOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update Bitcast or tail Reshape desc for special FP4 path.");
        return false;
    }

    TensorDesc tailReshapeOutputDesc;
    if (chain.tailReshapeNode->GetOutputDesc(0, tailReshapeOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get tail Reshape output desc.");
        return false;
    }
    Shape reshapeShape({dims[0], dims[1], dims[INDEX_THREE] * DOUBLE_SHAPE});
    tailReshapeOutputDesc.SetShape(reshapeShape);
    tailReshapeOutputDesc.SetOriginShape(reshapeShape);
    if (chain.tailReshapeNode->UpdateOutputDesc(0, tailReshapeOutputDesc) != GRAPH_SUCCESS ||
        nodeGmmFr.UpdateInputDesc(inputIndex, tailReshapeOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update GroupedMatmulFinalizeRouting desc for special FP4 path.");
        return false;
    }
    return true;
}

bool UpdateInputDesc(GNode &nodeGmmFr, const TransposeChain &chain, int64_t inputIndex, const TensorDesc &sourceDesc)
{
    if (!chain.relinkToBitcast) {
        if (nodeGmmFr.UpdateInputDesc(inputIndex, sourceDesc) != GRAPH_SUCCESS) {
            OP_LOGW(PASS_NAME, "Failed to update GroupedMatmulFinalizeRouting input desc.");
            return false;
        }
        return true;
    }
    if (chain.bitcastNode == nullptr) {
        OP_LOGW(PASS_NAME, "Bitcast node is null when updating desc.");
        return false;
    }
    TensorDesc gmmInputDesc;
    if (nodeGmmFr.GetInputDesc(inputIndex, gmmInputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get GroupedMatmulFinalizeRouting input desc.");
        return false;
    }
    if (chain.bitcastNode->UpdateInputDesc(0, sourceDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update Bitcast input desc.");
        return false;
    }
    if (chain.specialFp4Path) {
        return UpdateSpecialFp4Desc(nodeGmmFr, chain, inputIndex, sourceDesc);
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
    bitcastOutputDesc.SetDataType(gmmInputDesc.GetDataType());
    if (nodeGmmFr.UpdateInputDesc(inputIndex, bitcastOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update GroupedMatmulFinalizeRouting input desc from Bitcast.");
        return false;
    }
    return true;
}

bool RelinkNode(const GraphPtr &graph, GNode &nodeGmmFr, const TransposeChain &chain, int64_t inputIndex)
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

    auto &dstNode = chain.relinkToBitcast ? *chain.bitcastNode : nodeGmmFr;
    const int64_t dstInputPort = chain.relinkToBitcast ? 0 : inputIndex;
    if (graph->RemoveEdge(*chain.transLikeNode, 0, dstNode, dstInputPort) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to remove edge from transpose-like node.");
        return false;
    }
    if (graph->AddDataEdge(*chain.sourceNode, chain.sourceOutputPort, dstNode, dstInputPort) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to add edge from source node.");
        return false;
    }
    return UpdateInputDesc(nodeGmmFr, chain, inputIndex, sourceDesc);
}

void AddNodesToReport(const GNode &nodeGmmFr, const std::vector<TransposeChain> &chains,
                      std::vector<GNode> &nodesBeforeFuse, std::vector<GNodePtr> &nodesToRemove)
{
    nodesBeforeFuse = {nodeGmmFr};
    nodesToRemove.clear();
    for (const auto &chain : chains) {
        for (const auto &node : chain.nodesToRemove) {
            if (node == nullptr || std::find(nodesToRemove.begin(), nodesToRemove.end(), node) != nodesToRemove.end()) {
                continue;
            }
            nodesBeforeFuse.emplace_back(*node);
            nodesToRemove.emplace_back(node);
        }
    }
}

void ReportTransposeFusion(const std::vector<GNode> &nodesBeforeFuse, const GNode &nodeGmmFr,
                           CustomPassContext &passContext)
{
    auto reportFuse = GraphFuseInspectorUtils::ReportFuse;
    if (reportFuse == nullptr) {
        return;
    }
    if (reportFuse(nodesBeforeFuse, {nodeGmmFr}, passContext) != SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to report fusion result.");
    }
}

bool RemoveFusionNodes(const GraphPtr &graph, const std::vector<GNodePtr> &nodesToRemove)
{
    for (const auto &nodeToRemove : nodesToRemove) {
        if (!RemoveNode(graph, nodeToRemove)) {
            return false;
        }
    }
    return true;
}

Status PrepareFusion(const GNode &nodeGmmFr, std::vector<TransposeChain> &chains)
{
    if (!CheckGroupedMatmulFinalizeRouting(nodeGmmFr) || !CheckPlatformIsAscend950()) {
        return GRAPH_NOT_CHANGED;
    }
    if (!CheckNodeDtype(nodeGmmFr) || !CheckNodeShape(nodeGmmFr)) {
        return GRAPH_NOT_CHANGED;
    }
    bool transposeWeight = false;
    if (nodeGmmFr.GetAttr(ATTR_TRANSPOSE_WEIGHT, transposeWeight) != GRAPH_SUCCESS || transposeWeight) {
        return GRAPH_NOT_CHANGED;
    }

    TransposeChain weightChain;
    if (!ResolveTransposeChain(nodeGmmFr, WEIGHT_INDEX, weightChain)) {
        return GRAPH_NOT_CHANGED;
    }
    chains.emplace_back(weightChain);

    if (!IsMxWeightScale(nodeGmmFr)) {
        return GRAPH_NOT_CHANGED;
    }
    TransposeChain weightScaleChain;
    if (!ResolveTransposeChain(nodeGmmFr, WEIGHT_SCALE_INDEX, weightScaleChain)) {
        return GRAPH_NOT_CHANGED;
    }
    chains.emplace_back(weightScaleChain);
    return SUCCESS;
}

Status CommitFusion(const GraphPtr &graph, GNode &nodeGmmFr, const std::vector<TransposeChain> &chains,
                    CustomPassContext &passContext)
{
    std::vector<GNode> nodesBeforeFuse;
    std::vector<GNodePtr> nodesToRemove;
    AddNodesToReport(nodeGmmFr, chains, nodesBeforeFuse, nodesToRemove);

    bool transposeWeight = true;
    if (nodeGmmFr.SetAttr(ATTR_TRANSPOSE_WEIGHT, transposeWeight) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to set transpose_w attr.");
        return GRAPH_NOT_CHANGED;
    }
    for (const auto &chain : chains) {
        if (!RelinkNode(graph, nodeGmmFr, chain, chain.inputIndex)) {
            return GRAPH_FAILED;
        }
    }
    ReportTransposeFusion(nodesBeforeFuse, nodeGmmFr, passContext);
    if (!RemoveFusionNodes(graph, nodesToRemove)) {
        return GRAPH_FAILED;
    }
    return SUCCESS;
}

Status Fusion(const GraphPtr &graph, GNode &nodeGmmFr, CustomPassContext &passContext)
{
    std::vector<TransposeChain> chains;
    const auto status = PrepareFusion(nodeGmmFr, chains);
    if (status != SUCCESS) {
        return status;
    }
    return CommitFusion(graph, nodeGmmFr, chains, passContext);
}

Status RunGroupedMatmulFinalizeRoutingTransposeFusion(const GraphPtr &graph, CustomPassContext &passContext)
{
    OP_LOGD(PASS_NAME, "Enter GroupedMatmulFinalizeRouting transpose fusion pass.");
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(PASS_NAME, "Graph is null or invalid, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }

    std::vector<GNode> nodeGmmFrList;
    for (auto &node : graph->GetDirectNode()) {
        if (CheckGroupedMatmulFinalizeRouting(node)) {
            nodeGmmFrList.emplace_back(node);
        }
    }
    if (nodeGmmFrList.empty()) {
        OP_LOGD(PASS_NAME, "No GroupedMatmulFinalizeRouting node, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }

    passContext.SetPassName(PASS_NAME);
    bool changed = false;
    for (auto &nodeGmmFr : nodeGmmFrList) {
        auto status = Fusion(graph, nodeGmmFr, passContext);
        if (status == SUCCESS) {
            changed = true;
            continue;
        }
        if (status != GRAPH_NOT_CHANGED) {
            return status;
        }
    }
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}
} // namespace

Status GroupedMatmulFinalizeRoutingTransposeFusionPass::Run(GraphPtr &graph, CustomPassContext &passContext)
{
    return RunGroupedMatmulFinalizeRoutingTransposeFusion(graph, passContext);
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(GroupedMatmulFinalizeRoutingTransposeFusionPass).Stage(CustomPassStage::kCompatibleInherited);
#endif
} // namespace ops
#endif // CANN_VERSION_NUM
