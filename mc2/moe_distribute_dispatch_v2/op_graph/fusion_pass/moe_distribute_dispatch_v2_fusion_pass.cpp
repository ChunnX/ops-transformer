/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "moe_distribute_dispatch_v2_fusion_pass.h"
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "graph/graph.h"
#include "graph/gnode.h"
#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "graph/ge_error_codes.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "mc2_common_log.h"
#include "mc2_platform_info.h"
#include "op_graph/mc2_moe_graph_context.h"

namespace ops {
namespace {
const std::string PASS_NAME = "MoeDistributeDispatchV2FusionPass";
const std::string FUSED_OP_TYPE2 = "MoeDistributeDispatchV2";
const std::string FUSED_OP_TYPE3 = "MoeDistributeDispatchV3";
const std::string MOE_CONTEXT_TAG = "MOE_DISPATCH_COMBINE_V2_CONTEXT";
const std::string MOE_EP_NAME = "group_ep";

const uint32_t IDX_0 = 0;
constexpr size_t kV2TotalOutputs = 7;
constexpr size_t kV2TotalInputs = 7;

const std::vector<std::string> REQUIRED_INT_ATTRS = {
    "ep_world_size", "ep_rank_id", "moe_expert_num"
};
const std::vector<std::string> OPTIONAL_INT_ATTRS = {
    "tp_world_size", "tp_rank_id", "expert_shard_type", "shared_expert_num",
    "shared_expert_rank_num", "quant_mode", "global_bs", "expert_token_nums_type",
    "zero_expert_num", "copy_expert_num", "const_expert_num", "y_dtype"
};
const std::vector<std::string> OPTIONAL_STR_ATTRS = {"comm_alg"};
}  // namespace

ge::graphStatus MoeDistributeDispatchV2FusionPass::AddAttr(ge::GNode &moeDistributeDispatchNode,
                                                           ge::GNode &fusionNode,
                                                           int64_t &hccl_buff_size) const
{
    ge::AscendString nameStr;
    moeDistributeDispatchNode.GetName(nameStr);
    int64_t attrInt = 0;
    for (const auto &attrName : REQUIRED_INT_ATTRS) {
        if (moeDistributeDispatchNode.GetAttr(ge::AscendString(attrName.c_str()), attrInt) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "node %s: Get required attr %s failed", nameStr.GetString(), attrName.c_str());
            return ge::GRAPH_FAILED;
        }
        fusionNode.SetAttr(ge::AscendString(attrName.c_str()), attrInt);
    }
    fusionNode.SetAttr(ge::AscendString("ccl_buffer_size"), hccl_buff_size);
    for (const auto &attrName : OPTIONAL_INT_ATTRS) {
        if (moeDistributeDispatchNode.GetAttr(ge::AscendString(attrName.c_str()), attrInt) == ge::GRAPH_SUCCESS) {
            fusionNode.SetAttr(ge::AscendString(attrName.c_str()), attrInt);
        }
    }
    ge::AscendString strVal;
    for (const auto &attrName : OPTIONAL_STR_ATTRS) {
        if (moeDistributeDispatchNode.GetAttr(ge::AscendString(attrName.c_str()), strVal) == ge::GRAPH_SUCCESS) {
            fusionNode.SetAttr(ge::AscendString(attrName.c_str()), strVal);
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 创建 V3 节点，拷贝 V2 的输入/输出描述，添加 context 输入（port 0），拷贝属性
ge::graphStatus MoeDistributeDispatchV2FusionPass::CreateFusionNode(
    ge::Graph &graph, ge::GNode &moeDistributeDispatchNode,
    int64_t &hccl_buff_size, const ge::TensorDesc &tensorDesc, ge::GNode &fusionNode) const
{
    ge::AscendString nameStr;
    moeDistributeDispatchNode.GetName(nameStr);
    auto fusionOp = ge::OperatorFactory::CreateOperator(std::string(nameStr.GetString()) + "_V3", FUSED_OP_TYPE3);
    if (fusionOp.IsEmpty()) {
        OPS_LOG_E(PASS_NAME.c_str(), "node %s: %s op not found in factory",
                  nameStr.GetString(), FUSED_OP_TYPE3.c_str());
        return ge::GRAPH_FAILED;
    }
    fusionOp.UpdateInputDesc(static_cast<uint32_t>(IDX_0), tensorDesc);
    for (size_t v2Port = 0; v2Port < kV2TotalInputs; v2Port++) {
        auto srcPair = moeDistributeDispatchNode.GetInDataNodesAndPortIndexs(static_cast<int32_t>(v2Port));
        if (srcPair.first == nullptr) { continue; }
        ge::TensorDesc srcDesc;
        if (srcPair.first->GetOutputDesc(srcPair.second, srcDesc) != ge::GRAPH_SUCCESS) { continue; }
        fusionOp.UpdateInputDesc(static_cast<uint32_t>(v2Port + 1), srcDesc);
    }
    size_t v2OutputNum = moeDistributeDispatchNode.GetOutputsSize();
    for (size_t v2Port = 0; v2Port < v2OutputNum; v2Port++) {
        ge::TensorDesc v2OutDesc;
        if (moeDistributeDispatchNode.GetOutputDesc(static_cast<int32_t>(v2Port), v2OutDesc) == ge::GRAPH_SUCCESS) {
            fusionOp.UpdateOutputDesc(static_cast<uint32_t>(v2Port), v2OutDesc);
        }
    }
    fusionNode = graph.AddNodeByOp(fusionOp);
    return AddAttr(moeDistributeDispatchNode, fusionNode, hccl_buff_size);
}

// 将 V2 节点的所有边重连到 V3：输入边（port+1 偏移）→ 删除 V2 → 输出边 → context→V3.0
ge::graphStatus MoeDistributeDispatchV2FusionPass::AddEdge(
    ge::Graph &graph, ge::GNode &moeNodePtr, ge::GNode &fusionNode,
    ge::GNode &contextNodePtr) const
{
    ge::graphStatus edgeRet;
    for (size_t v2Port = 0; v2Port < kV2TotalInputs; v2Port++) {
        auto srcPair = moeNodePtr.GetInDataNodesAndPortIndexs(static_cast<int32_t>(v2Port));
        if (srcPair.first == nullptr) { continue; }
        edgeRet = graph.AddDataEdge(*srcPair.first, srcPair.second, fusionNode, static_cast<int32_t>(v2Port) + 1);
        if (edgeRet != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    std::vector<std::vector<std::pair<ge::GNodePtr, int32_t>>> outputConsumers(kV2TotalOutputs);
    for (size_t i = IDX_0; i < kV2TotalOutputs; i++) {
        outputConsumers[i] = moeNodePtr.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
    }
    graph.RemoveNode(moeNodePtr);
    for (size_t i = IDX_0; i < kV2TotalOutputs; i++) {
        for (auto &outPair : outputConsumers[i]) {
            edgeRet = graph.AddDataEdge(fusionNode, static_cast<int32_t>(i), *outPair.first, outPair.second);
            if (edgeRet != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    return graph.AddDataEdge(contextNodePtr, static_cast<int32_t>(IDX_0),
                             fusionNode, static_cast<int32_t>(IDX_0));
}

// 单节点融合：创建 V3 节点 + 重连所有边
ge::graphStatus MoeDistributeDispatchV2FusionPass::FusionNode(
    ge::Graph &graph, ge::GNode &moeDistributeDispatchNode,
    ge::GNode &contextNode, int64_t &hccl_buff_size, const ge::TensorDesc &tensorDesc) const
{
    ge::GNode fusionNode;
    if (CreateFusionNode(graph, moeDistributeDispatchNode, hccl_buff_size, tensorDesc, fusionNode) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return AddEdge(graph, moeDistributeDispatchNode, fusionNode, contextNode);
}

// 获取或创建 Mc2MoeContext Const 节点，构造 TensorDesc，调用 FusionNode
ge::graphStatus MoeDistributeDispatchV2FusionPass::Fusion(
    ge::Graph &graph, ge::GNode &moeDistributeDispatchNode, ge::GNode &sharedCtxNode) const
{
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    int64_t hccl_buff_size = 0;
    if (Mc2MoeGraphContext::GetMc2ContextA5(graph, MOE_CONTEXT_TAG, MOE_EP_NAME,
                                            moeDistributeDispatchNode,
                                             sharedCtxNode, hccl_buff_size) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "GetMc2ContextA5 failed, V2 node kept as-is");
        return ge::GRAPH_SUCCESS;
    }
    ge::Shape shape({static_cast<int64_t>(sizeof(Mc2MoeContext) / sizeof(int32_t))});
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    return FusionNode(graph, moeDistributeDispatchNode, sharedCtxNode, hccl_buff_size, tensorDesc);
#else
    return ge::GRAPH_SUCCESS;
#endif
}

// A5 + non-ccu 才融合；遍历图中所有 DispatchV2 节点，逐个调用 Fusion
ge::graphStatus MoeDistributeDispatchV2FusionPass::Run(ge::GraphPtr &graph, ge::CustomPassContext &pass_context)
{
    (void)pass_context;
    if (IsTargetPlatformSocVersion(PASS_NAME.c_str(), PLATFORM_A2)) {
        return ge::GRAPH_SUCCESS;
    }
    if (!IsTargetPlatformNpuArch(PASS_NAME.c_str(), NPUARCH_A5)) {
        return ge::GRAPH_SUCCESS;
    }
    ge::GNode sharedCtxNode;
    for (auto &gNode : graph->GetDirectNode()) {
        ge::AscendString nodeType;
        if ((gNode.GetType(nodeType) != ge::GRAPH_SUCCESS) || (nodeType != FUSED_OP_TYPE2.c_str())) {
            continue;
        }
        ge::AscendString commAlg;
        if (gNode.GetAttr(ge::AscendString("comm_alg"), commAlg) == ge::GRAPH_SUCCESS &&
            commAlg.GetString() != nullptr && std::string(commAlg.GetString()) == "ccu") {
            continue;
        }
        ge::AscendString nameStr;
        gNode.GetName(nameStr);
        auto ret = Fusion(*graph, gNode, sharedCtxNode);
        if (ret != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "node %s: V2->V3 fusion FAILED", nameStr.GetString());
        }
    }
    return ge::GRAPH_SUCCESS;
}

REG_FUSION_PASS(MoeDistributeDispatchV2FusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
}  // namespace ops
#endif  // GRAPH_FUSION_SUPPORT_VERSION
