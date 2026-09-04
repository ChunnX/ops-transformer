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
 * \file moe_distribute_combine_add_rms_norm_fusion_pass.cpp
 * \brief Fuse the chain below into one MoeDistributeCombineAddRmsNorm node.
 *
 *  expand_x, expert_ids, assist_info, ep_send_counts, expert_scales
 *                |
 *      MoeDistributeCombineV2
 *                |                x2(shared expert output)   x1(residual)   gamma
 *            x2 reshape                  \                        \            \
 *                \  |                     \                        \            \
 *                  Add           ---->    MoeDistributeCombineAddRmsNorm
 *                    \                   /         |            \
 *          x1, gamma  \                  y         rstd          x
 *                     InplaceAddRmsNorm
 *                       |     |     |
 *                      x1   rstd   x2
 */

#include "moe_distribute_combine_add_rms_norm_fusion_pass.h"

#if CANN_VERSION_NUM >= MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_GRAPH_FUSION_SUPPORT_VERSION
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl_rt.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "graph/graph.h"
#include "graph/operator_factory.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"

using namespace ge;
using namespace ge::fusion;

namespace ops {
namespace {
constexpr char kPassName[] = "MoeDistributeCombineAddRmsNormFusionPass";
constexpr char kOpTypeCombineV2[] = "MoeDistributeCombineV2";
constexpr char kOpTypeFused[] = "MoeDistributeCombineAddRmsNorm";
constexpr char kOpTypeReshape[] = "Reshape";
constexpr char kOpTypeAdd[] = "Add";
constexpr char kOpTypeInplaceAddRmsNorm[] = "InplaceAddRmsNorm";
constexpr char kOpTypeConst[] = "Const";

constexpr int32_t kIdx0 = 0;
constexpr int32_t kIdx1 = 1;
constexpr int32_t kIdx2 = 2;
constexpr int32_t kCombineRequiredInputNum = 5; // expand_x ~ expert_scales
constexpr int32_t kCombineOptionalInputNum = 6; // tp_send_counts ~ expand_scales
constexpr int32_t kCombineMaxSupportedInputNum = kCombineRequiredInputNum + kCombineOptionalInputNum;
constexpr int32_t kAddInputNum = 2;
constexpr int32_t kArnInputNum = 3;
constexpr int32_t kArnOutputNum = 3;
constexpr int32_t kFusedSharedExpertInputIdx = 13;
constexpr int64_t kMaxHSize = 7168;
constexpr int64_t kMaxBsSize = 192;
constexpr int64_t kMaxEpRankSize = 288;
constexpr int32_t kCompatibleInheritedSupportVersion = 90100000;
constexpr float32_t kDefaultEpsilon = 1e-6f;

struct MdcarnFuseScene {
    GNodePtr combineNode;
    GNodePtr reshapeNode;
    GNodePtr addNode;
    GNodePtr residualNode = nullptr;
    GNodePtr gammaNode = nullptr;
    GNodePtr sharedExpertNode = nullptr;
    int32_t residualPort = 0;
    int32_t gammaPort = 0;
    int32_t sharedExpertPort = 0;
};

bool IsGraphFusionRuntimeSupported()
{
    int32_t version = 0;
    if (aclsysGetVersionNum("ge_compiler", &version) != ACL_SUCCESS) {
        OP_LOGW(kPassName, "Failed to get ge compiler version.");
        return false;
    }
    return version >= kCompatibleInheritedSupportVersion;
}

bool IsType(const GNodePtr &node, const char *type)
{
    if (node == nullptr) {
        return false;
    }
    AscendString nodeType;
    return node->GetType(nodeType) == GRAPH_SUCCESS && nodeType == type;
}

bool IsType(GNode &node, const char *type)
{
    AscendString nodeType;
    return node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == type;
}

GNodePtr GetInputNode(GNode &dstNode, int32_t dstInputPort, int32_t *srcOutputPort = nullptr)
{
    auto [srcNode, resolvedSrcOutputPort] = dstNode.GetInDataNodesAndPortIndexs(dstInputPort);
    if (srcOutputPort != nullptr) {
        *srcOutputPort = resolvedSrcOutputPort;
    }
    return srcNode;
}

size_t GetOutputConsumerNum(const GNodePtr &node, int32_t outputPort)
{
    if (node == nullptr) {
        return 0;
    }
    return node->GetOutDataNodesAndPortIndexs(outputPort).size();
}

bool BuildScene(GNode &arnNode, MdcarnFuseScene &scene)
{
    if (!IsType(arnNode, kOpTypeInplaceAddRmsNorm)) {
        return false;
    }
    if (arnNode.GetInputsSize() != static_cast<size_t>(kArnInputNum)) {
        OP_LOGD(kPassName, "InplaceAddRmsNorm input num is not %d.", kArnInputNum);
        return false;
    }
    auto [residualNode, residualPort] = arnNode.GetInDataNodesAndPortIndexs(kIdx0);
    auto [addNode, addOutPort] = arnNode.GetInDataNodesAndPortIndexs(kIdx1);
    auto [gammaNode, gammaPort] = arnNode.GetInDataNodesAndPortIndexs(kIdx2);
    if (residualNode == nullptr || addNode == nullptr || gammaNode == nullptr) {
        OP_LOGD(kPassName, "InplaceAddRmsNorm inputs are not fully connected.");
        return false;
    }
    if (!IsType(addNode, kOpTypeAdd)) {
        OP_LOGD(kPassName, "The input x2 of InplaceAddRmsNorm is not produced by Add.");
        return false;
    }
    if (addNode->GetInputsSize() != static_cast<size_t>(kAddInputNum)) {
        OP_LOGD(kPassName, "Add input num is not %d.", kAddInputNum);
        return false;
    }
    auto [reshapeNode, reshapeOutPort] = addNode->GetInDataNodesAndPortIndexs(kIdx0);
    auto [sharedExpertNode, sharedExpertPort] = addNode->GetInDataNodesAndPortIndexs(kIdx1);
    if (reshapeNode == nullptr || sharedExpertNode == nullptr) {
        OP_LOGD(kPassName, "Add inputs are not fully connected.");
        return false;
    }
    if (!IsType(reshapeNode, kOpTypeReshape)) {
        OP_LOGD(kPassName, "The first input of Add is not produced by Reshape.");
        return false;
    }
    auto [combineNode, combineOutPort] = reshapeNode->GetInDataNodesAndPortIndexs(kIdx0);
    if (combineNode == nullptr) {
        OP_LOGD(kPassName, "The input of Reshape is not connected.");
        return false;
    }
    if (!IsType(combineNode, kOpTypeCombineV2)) {
        OP_LOGD(kPassName, "The input of Reshape is not produced by MoeDistributeCombineV2.");
        return false;
    }

    scene.combineNode = combineNode;
    scene.reshapeNode = reshapeNode;
    scene.addNode = addNode;
    scene.residualNode = residualNode;
    scene.residualPort = residualPort;
    scene.gammaNode = gammaNode;
    scene.gammaPort = gammaPort;
    scene.sharedExpertNode = sharedExpertNode;
    scene.sharedExpertPort = sharedExpertPort;
    return true;
}

bool CheckNodeEdges(const MdcarnFuseScene &scene)
{
    const size_t combineInputSize = scene.combineNode->GetInputsSize();
    for (int32_t i = kIdx0; i < kCombineRequiredInputNum; ++i) {
        if (GetInputNode(*scene.combineNode, i) == nullptr) {
            OP_LOGD(kPassName, "MoeDistributeCombineV2 required input %d is not connected.", i);
            return false;
        }
    }
    for (size_t i = kCombineMaxSupportedInputNum; i < combineInputSize; ++i) {
        if (GetInputNode(*scene.combineNode, static_cast<int32_t>(i)) != nullptr) {
            OP_LOGD(kPassName, "MoeDistributeCombineV2 input %zu is out of fusion support.", i);
            return false;
        }
    }
    if (GetOutputConsumerNum(scene.combineNode, kIdx0) != 1) {
        OP_LOGD(kPassName, "MoeDistributeCombineV2 output is not used by only one node.");
        return false;
    }
    if (GetOutputConsumerNum(scene.reshapeNode, kIdx0) != 1) {
        OP_LOGD(kPassName, "Reshape output is not used by only one node.");
        return false;
    }
    if (GetOutputConsumerNum(scene.addNode, kIdx0) != 1) {
        OP_LOGD(kPassName, "Add output is not used by only one node.");
        return false;
    }
    return true;
}

bool CheckShapesAndDtypes(GNode &arnNode, const MdcarnFuseScene &scene)
{
    TensorDesc expandXDesc;
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    if (scene.combineNode->GetInputDesc(kIdx0, expandXDesc) != GRAPH_SUCCESS ||
        arnNode.GetInputDesc(kIdx0, x1Desc) != GRAPH_SUCCESS ||
        scene.addNode->GetInputDesc(kIdx1, x2Desc) != GRAPH_SUCCESS) {
        OP_LOGD(kPassName, "Failed to get input desc of fusion scene.");
        return false;
    }

    const auto expandXDims = expandXDesc.GetShape().GetDims();
    const auto x1Dims = x1Desc.GetShape().GetDims();
    const auto x2Dims = x2Desc.GetShape().GetDims();
    const int64_t dimTwo = 2;
    const int64_t dimThree = 3;
    if (static_cast<int64_t>(expandXDims.size()) != dimTwo) {
        OP_LOGD(kPassName, "Expand_x shape is not 2 dims.");
        return false;
    }
    if (static_cast<int64_t>(x2Dims.size()) != dimThree || x2Dims[kIdx0] > kMaxBsSize) {
        OP_LOGD(kPassName, "X2 shape is not right.");
        return false;
    }
    if (static_cast<int64_t>(x1Dims.size()) != dimThree || x1Dims[kIdx0] > kMaxBsSize) {
        OP_LOGD(kPassName, "X1 shape is not right.");
        return false;
    }
    if (x1Dims != x2Dims) {
        OP_LOGD(kPassName, "X2 is not the same as x1.");
        return false;
    }
    if (expandXDims[kIdx1] != x1Dims[kIdx2]) {
        OP_LOGD(kPassName, "Expand_x is not matched to x1.");
        return false;
    }
    if (expandXDims[kIdx0] > kMaxBsSize * kMaxEpRankSize || expandXDims[kIdx1] > kMaxHSize) {
        OP_LOGD(kPassName, "Expand_x[%ld, %ld] is too huge to merge.", expandXDims[kIdx0], expandXDims[kIdx1]);
        return false;
    }

    if (expandXDesc.GetDataType() != DT_BF16 || x1Desc.GetDataType() != DT_BF16 || x2Desc.GetDataType() != DT_BF16) {
        OP_LOGD(kPassName, "Expand_x, x1 or x2 dtype is not bfloat16.");
        return false;
    }
    return true;
}

bool CopyStringAttr(GNode &srcNode, GNode &dstNode, const char *attrName)
{
    AscendString attrValue;
    if (srcNode.GetAttr(attrName, attrValue) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Get attr %s failed.", attrName);
        return false;
    }
    return dstNode.SetAttr(attrName, attrValue) == GRAPH_SUCCESS;
}

void CopyStringAttrIfPresent(GNode &srcNode, GNode &dstNode, const char *attrName)
{
    AscendString attrValue;
    if (srcNode.GetAttr(attrName, attrValue) == GRAPH_SUCCESS) {
        (void)dstNode.SetAttr(attrName, attrValue);
    }
}

bool CopyIntAttr(GNode &srcNode, GNode &dstNode, const char *attrName)
{
    int64_t attrValue = 0;
    if (srcNode.GetAttr(attrName, attrValue) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Get attr %s failed.", attrName);
        return false;
    }
    return dstNode.SetAttr(attrName, attrValue) == GRAPH_SUCCESS;
}

void CopyIntAttrIfPresent(GNode &srcNode, GNode &dstNode, const char *attrName)
{
    int64_t attrValue = 0;
    if (srcNode.GetAttr(attrName, attrValue) == GRAPH_SUCCESS) {
        (void)dstNode.SetAttr(attrName, attrValue);
    }
}

bool SetFusedNodeAttrs(GNode &arnNode, GNode &combineNode, GNode &fusedNode)
{
    if (!CopyStringAttr(combineNode, fusedNode, "group_ep") || !CopyIntAttr(combineNode, fusedNode, "ep_world_size") ||
        !CopyIntAttr(combineNode, fusedNode, "ep_rank_id") || !CopyIntAttr(combineNode, fusedNode, "moe_expert_num")) {
        OP_LOGW(kPassName, "Copy required attrs from MoeDistributeCombineV2 failed.");
        return false;
    }

    CopyStringAttrIfPresent(combineNode, fusedNode, "group_tp");
    CopyStringAttrIfPresent(combineNode, fusedNode, "comm_alg");
    const char *optionalIntAttrs[] = {
        "tp_world_size",          "tp_rank_id",      "expert_shard_type", "shared_expert_num",
        "shared_expert_rank_num", "global_bs",       "out_dtype",         "comm_quant_mode",
        "group_list_type",        "zero_expert_num", "copy_expert_num",   "const_expert_num"};
    for (const char *attrName : optionalIntAttrs) {
        CopyIntAttrIfPresent(combineNode, fusedNode, attrName);
    }

    float32_t epsilon = kDefaultEpsilon;
    if (arnNode.GetAttr("epsilon", epsilon) != GRAPH_SUCCESS) {
        epsilon = kDefaultEpsilon;
    }
    return fusedNode.SetAttr("norm_eps", epsilon) == GRAPH_SUCCESS;
}

struct FusedInputPlan {
    int32_t fusedPort = 0;
    GNodePtr srcNode = nullptr;
    int32_t srcPort = 0;
};

bool CollectInputPlans(const MdcarnFuseScene &scene, std::vector<FusedInputPlan> &plans)
{
    for (int32_t i = kIdx0; i < kCombineRequiredInputNum; ++i) {
        int32_t srcPort = 0;
        auto srcNode = GetInputNode(*scene.combineNode, i, &srcPort);
        if (srcNode == nullptr) {
            OP_LOGW(kPassName, "Get source of MoeDistributeCombineV2 input %d failed.", i);
            return false;
        }
        plans.push_back({i, srcNode, srcPort});
    }
    plans.push_back({5, scene.residualNode, scene.residualPort});
    plans.push_back({6, scene.gammaNode, scene.gammaPort});
    for (int32_t i = kCombineRequiredInputNum; i < kCombineMaxSupportedInputNum; ++i) {
        int32_t srcPort = 0;
        auto srcNode = GetInputNode(*scene.combineNode, i, &srcPort);
        if (srcNode != nullptr) {
            plans.push_back({i + 2, srcNode, srcPort});
        }
    }
    plans.push_back({kFusedSharedExpertInputIdx, scene.sharedExpertNode, scene.sharedExpertPort});
    return true;
}

bool CreateFusedNode(GraphPtr &graph, GNode &arnNode, const MdcarnFuseScene &scene, GNode &fusedNode)
{
    AscendString combineName;
    if (scene.combineNode->GetName(combineName) != GRAPH_SUCCESS || combineName.GetString() == nullptr) {
        OP_LOGW(kPassName, "Get MoeDistributeCombineV2 node name failed.");
        return false;
    }
    const std::string fusedName = std::string(combineName.GetString()) + "_add_rms_norm";
    Operator fusedOp = OperatorFactory::CreateOperator(fusedName.c_str(), kOpTypeFused);
    if (fusedOp.IsEmpty()) {
        OP_LOGW(kPassName, "Create operator %s failed.", kOpTypeFused);
        return false;
    }
    fusedNode = graph->AddNodeByOp(fusedOp);

    std::vector<FusedInputPlan> plans;
    if (!CollectInputPlans(scene, plans)) {
        return false;
    }
    for (const auto &plan : plans) {
        if (graph->AddDataEdge(*plan.srcNode, plan.srcPort, fusedNode, plan.fusedPort) != GRAPH_SUCCESS) {
            OP_LOGW(kPassName, "Add data edge for fused input %d failed.", plan.fusedPort);
            return false;
        }
        TensorDesc srcOutputDesc;
        if (plan.srcNode->GetOutputDesc(plan.srcPort, srcOutputDesc) != GRAPH_SUCCESS ||
            fusedNode.UpdateInputDesc(plan.fusedPort, srcOutputDesc) != GRAPH_SUCCESS) {
            OP_LOGW(kPassName, "Update input desc for fused input %d failed.", plan.fusedPort);
            return false;
        }
    }
    for (int32_t i = kIdx0; i < kArnOutputNum; ++i) {
        TensorDesc arnOutputDesc;
        if (arnNode.GetOutputDesc(i, arnOutputDesc) != GRAPH_SUCCESS ||
            fusedNode.UpdateOutputDesc(i, arnOutputDesc) != GRAPH_SUCCESS) {
            OP_LOGW(kPassName, "Update output desc %d for fused node failed.", i);
            return false;
        }
    }
    return true;
}

bool RewireArnOutputs(GraphPtr &graph, GNode &arnNode, GNode &fusedNode)
{
    for (int32_t port = kIdx0; port < kArnOutputNum; ++port) {
        auto consumers = arnNode.GetOutDataNodesAndPortIndexs(port);
        for (auto &[consumer, consumerPort] : consumers) {
            if (consumer == nullptr) {
                continue;
            }
            if (graph->RemoveEdge(arnNode, port, *consumer, consumerPort) != GRAPH_SUCCESS) {
                OP_LOGW(kPassName, "Remove edge of InplaceAddRmsNorm output %d failed.", port);
                return false;
            }
            if (graph->AddDataEdge(fusedNode, port, *consumer, consumerPort) != GRAPH_SUCCESS) {
                OP_LOGW(kPassName, "Relink consumer of InplaceAddRmsNorm output %d failed.", port);
                return false;
            }
        }
    }
    return true;
}

bool RemoveNodeIfUnused(GraphPtr &graph, GNode &node)
{
    if (graph == nullptr || !graph->IsValid()) {
        return true;
    }
    if (!node.GetOutDataNodesAndPortIndexs(kIdx0).empty()) {
        return true;
    }

    std::vector<GNodePtr> inputConstNodes;
    const auto inputSize = node.GetInputsSize();
    for (std::size_t inputIndex = 0; inputIndex < inputSize; ++inputIndex) {
        int32_t srcOutputPort = 0;
        auto srcNode = GetInputNode(node, static_cast<int32_t>(inputIndex), &srcOutputPort);
        if (srcNode == nullptr) {
            continue;
        }
        if (graph->RemoveEdge(*srcNode, srcOutputPort, node, static_cast<int32_t>(inputIndex)) != GRAPH_SUCCESS) {
            OP_LOGW(kPassName, "Failed to remove input edge before removing node.");
            return false;
        }
        if (IsType(srcNode, kOpTypeConst)) {
            inputConstNodes.emplace_back(srcNode);
        }
    }

    if (graph->RemoveNode(node) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to remove unused node.");
        return false;
    }

    for (const auto &constNode : inputConstNodes) {
        if (!RemoveNodeIfUnused(graph, *constNode)) {
            return false;
        }
    }
    return true;
}

bool RemoveFusionNodes(GraphPtr &graph, GNode &arnNode, const MdcarnFuseScene &scene)
{
    // InplaceAddRmsNorm must be removed first: its rewired outputs are empty, and removing it releases the
    // Add consumer, then Reshape and MoeDistributeCombineV2 in order.
    return RemoveNodeIfUnused(graph, arnNode) && RemoveNodeIfUnused(graph, *scene.addNode) &&
           RemoveNodeIfUnused(graph, *scene.reshapeNode) && RemoveNodeIfUnused(graph, *scene.combineNode);
}

Status FuseOneScene(GraphPtr &graph, GNode &arnNode)
{
    MdcarnFuseScene scene;
    if (!BuildScene(arnNode, scene)) {
        return GRAPH_NOT_CHANGED;
    }
    if (!CheckNodeEdges(scene) || !CheckShapesAndDtypes(arnNode, scene)) {
        return GRAPH_NOT_CHANGED;
    }

    GNode fusedNode;
    if (!CreateFusedNode(graph, arnNode, scene, fusedNode)) {
        OP_LOGE(kPassName, "Create fused node failed.");
        return GRAPH_FAILED;
    }
    if (!SetFusedNodeAttrs(arnNode, *scene.combineNode, fusedNode)) {
        OP_LOGE(kPassName, "Set attrs for fused node failed.");
        return GRAPH_FAILED;
    }
    if (!RewireArnOutputs(graph, arnNode, fusedNode)) {
        OP_LOGE(kPassName, "Rewire outputs of InplaceAddRmsNorm failed.");
        return GRAPH_FAILED;
    }
    if (!RemoveFusionNodes(graph, arnNode, scene)) {
        OP_LOGE(kPassName, "Remove fusion source nodes failed.");
        return GRAPH_FAILED;
    }
    OP_LOGI(kPassName, "MoeDistributeCombineAddRmsNorm graph fusion success!");
    return SUCCESS;
}

Status RunMoeDistributeCombineAddRmsNormFusion(GraphPtr &graph, CustomPassContext &passContext)
{
    OP_LOGD(kPassName, "Enter MoeDistributeCombineAddRmsNormFusionPass.");
    if (!IsGraphFusionRuntimeSupported()) {
        OP_LOGD(kPassName, "Skip pass because graph fusion runtime is unsupported.");
        return GRAPH_NOT_CHANGED;
    }
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(kPassName, "Graph is null or invalid.");
        return GRAPH_NOT_CHANGED;
    }

    std::vector<GNode> arnNodes;
    for (auto &node : graph->GetDirectNode()) {
        if (IsType(node, kOpTypeInplaceAddRmsNorm)) {
            arnNodes.emplace_back(node);
        }
    }
    if (arnNodes.empty()) {
        return GRAPH_NOT_CHANGED;
    }

    passContext.SetPassName(kPassName);
    bool changed = false;
    for (auto &arnNode : arnNodes) {
        auto status = FuseOneScene(graph, arnNode);
        if (status == SUCCESS) {
            changed = true;
            continue;
        }
        if (status != GRAPH_NOT_CHANGED) {
            OP_LOGE(kPassName, "MoeDistributeCombineAddRmsNorm fusion failed, status is %u.", status);
            return status;
        }
    }
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}
} // namespace

Status MoeDistributeCombineAddRmsNormFusionPass::Run(GraphPtr &graph, CustomPassContext &passContext)
{
    return RunMoeDistributeCombineAddRmsNormFusion(graph, passContext);
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(MoeDistributeCombineAddRmsNormFusionPass).Stage(CustomPassStage::kCompatibleInherited);
#endif
} // namespace ops
#endif // CANN_VERSION_NUM >= MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_GRAPH_FUSION_SUPPORT_VERSION
