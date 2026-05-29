/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * MoeDistributeDispatchV2 -> MoeDistributeDispatchV3 fusion pass
 *
 *   x expert_ids scales x_active_mask expert_scales ...
 *                         |
 *                MoeDistributeDispatchV2
 *                         |
 *   expand_x dynamic_scales assist_info_for_combine expert_token_nums
 *   ep_recv_counts tp_recv_counts expand_scales
 *
 *   After fusion (A5 only, non-ccu):
 *   context x expert_ids scales x_active_mask expert_scales ...
 *                         |
 *                MoeDistributeDispatchV3
 *                         |
 *   (same outputs)
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
                                                           int64_t &hccl_buff_size)
{
    OPS_LOG_D(PASS_NAME.c_str(), "start dispatch fusion AddAttr");
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
    OPS_LOG_D(PASS_NAME.c_str(), "end dispatch fusion AddAttr");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2FusionPass::CreateFusionNode(
    ge::Graph &graph, ge::GNode &moeDistributeDispatchNode,
    int64_t &hccl_buff_size, const ge::TensorDesc &tensorDesc, ge::GNode &fusionNode)
{
    ge::AscendString nameStr;
    moeDistributeDispatchNode.GetName(nameStr);
    OPS_LOG_D(PASS_NAME.c_str(), "start dispatch fusion CreateFusionNode");

    std::string v3NodeName = std::string(nameStr.GetString()) + "_V3";
    auto fusionOp = ge::OperatorFactory::CreateOperator(v3NodeName, FUSED_OP_TYPE3);
    if (fusionOp.IsEmpty()) {
        OPS_LOG_E(PASS_NAME.c_str(), "node %s: %s op not found in factory", nameStr.GetString(),
                  FUSED_OP_TYPE3.c_str());
        return ge::GRAPH_FAILED;
    }

    // context 输入描述（端口 0，与 Const 输出一致）
    fusionOp.UpdateInputDesc(static_cast<uint32_t>(IDX_0), tensorDesc);

    // 遍历所有 V2 proto 输入（按索引），有边连接的就从源节点输出取类型
    size_t copiedCount = 0;
    for (size_t v2Port = 0; v2Port < kV2TotalInputs; v2Port++) {
        auto srcPair = moeDistributeDispatchNode.GetInDataNodesAndPortIndexs(static_cast<int32_t>(v2Port));
        if (srcPair.first == nullptr) {
            OPS_LOG_D(PASS_NAME.c_str(), "V2.port%zu: no edge, skip", v2Port);
            continue;
        }
        ge::TensorDesc srcDesc;
        if (srcPair.first->GetOutputDesc(srcPair.second, srcDesc) == ge::GRAPH_SUCCESS) {
            ge::AscendString srcName;
            srcPair.first->GetName(srcName);
            OPS_LOG_I(PASS_NAME.c_str(), "V2.port%zu(src=%s:%d, dtype=%d) → V3.port%zu",
                      v2Port, srcName.GetString(), srcPair.second,
                      static_cast<int>(srcDesc.GetDataType()), v2Port + 1);
            fusionOp.UpdateInputDesc(static_cast<uint32_t>(v2Port + 1), srcDesc);
            copiedCount++;
        } else {
            OPS_LOG_W(PASS_NAME.c_str(), "V2.port%zu: has edge but GetOutputDesc failed", v2Port);
        }
    }
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: %zu V2 input descs copied to V3", nameStr.GetString(), copiedCount);

    // 从 V2 拷贝所有输出描述到 V3
    size_t v2OutputNum = moeDistributeDispatchNode.GetOutputsSize();
    for (size_t v2Port = 0; v2Port < v2OutputNum; v2Port++) {
        ge::TensorDesc v2OutDesc;
        if (moeDistributeDispatchNode.GetOutputDesc(static_cast<int32_t>(v2Port), v2OutDesc) == ge::GRAPH_SUCCESS) {
            fusionOp.UpdateOutputDesc(static_cast<uint32_t>(v2Port), v2OutDesc);
        }
    }
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: %zu V2 output descs copied to V3", nameStr.GetString(), v2OutputNum);

    fusionNode = graph.AddNodeByOp(fusionOp);
    ge::AscendString v3Name;
    fusionNode.GetName(v3Name);
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: V3 node added as %s", nameStr.GetString(), v3Name.GetString());

    // 验证 V3 节点输入/输出描述
    size_t v3InputNum = fusionNode.GetInputsSize();
    OPS_LOG_I(PASS_NAME.c_str(), "V3[%s]: v3InputNum=%zu", v3Name.GetString(), v3InputNum);
    for (size_t p = 0; p < v3InputNum; p++) {
        ge::TensorDesc desc;
        if (fusionNode.GetInputDesc(static_cast<int32_t>(p), desc) == ge::GRAPH_SUCCESS) {
            OPS_LOG_I(PASS_NAME.c_str(), "  V3.port%zu: dtype=%d, format=%d, shape_dim=%zu",
                      p, static_cast<int>(desc.GetDataType()), static_cast<int>(desc.GetFormat()),
                      desc.GetShape().GetDimNum());
        } else {
            OPS_LOG_D(PASS_NAME.c_str(), "  V3.port%zu: GetInputDesc failed (unconnected optional)", p);
        }
    }
    size_t v3OutputNum = fusionNode.GetOutputsSize();
    OPS_LOG_I(PASS_NAME.c_str(), "V3[%s]: v3OutputNum=%zu", v3Name.GetString(), v3OutputNum);
    for (size_t p = 0; p < v3OutputNum; p++) {
        ge::TensorDesc desc;
        if (fusionNode.GetOutputDesc(static_cast<int32_t>(p), desc) == ge::GRAPH_SUCCESS) {
            OPS_LOG_I(PASS_NAME.c_str(), "  V3.out%zu: dtype=%d, format=%d, shape_dim=%zu",
                      p, static_cast<int>(desc.GetDataType()), static_cast<int>(desc.GetFormat()),
                      desc.GetShape().GetDimNum());
        }
    }

    if (AddAttr(moeDistributeDispatchNode, fusionNode, hccl_buff_size) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "node %s: AddAttr failed", nameStr.GetString());
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_D(PASS_NAME.c_str(), "end dispatch fusion CreateFusionNode");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2FusionPass::AddEdge(
    ge::Graph &graph, ge::GNode &moeNodePtr, ge::GNode &fusionNode,
    ge::GNode &contextNodePtr)
{
    OPS_LOG_D(PASS_NAME.c_str(), "start to add dispatch edge for fusion node");

    ge::graphStatus edgeRet;

    size_t wiredInputs = 0;
    for (size_t v2Port = 0; v2Port < kV2TotalInputs; v2Port++) {
        auto srcPair = moeNodePtr.GetInDataNodesAndPortIndexs(static_cast<int32_t>(v2Port));
        if (srcPair.first == nullptr) {
            OPS_LOG_D(PASS_NAME.c_str(), "AddEdge: V2.port%zu no edge, skip", v2Port);
            continue;
        }
        int32_t v3Port = static_cast<int32_t>(v2Port) + 1;
        ge::AscendString srcName;
        srcPair.first->GetName(srcName);
        OPS_LOG_I(PASS_NAME.c_str(), "AddEdge: %s:%d → V3.port%d", srcName.GetString(), srcPair.second, v3Port);
        edgeRet = graph.AddDataEdge(*srcPair.first, srcPair.second, fusionNode, v3Port);
        if (edgeRet != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "dispatch AddDataEdge from %s.port%d→V3.port%d failed",
                      srcName.GetString(), srcPair.second, v3Port);
            return ge::GRAPH_FAILED;
        }
        wiredInputs++;
    }

    // 收集 V2 输出消费者 → 删除 V2 → 重连 V3 输出到消费者
    std::vector<std::vector<std::pair<ge::GNodePtr, int32_t>>> outputConsumers(kV2TotalOutputs);
    for (size_t i = IDX_0; i < kV2TotalOutputs; i++) {
        outputConsumers[i] = moeNodePtr.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
    }

    graph.RemoveNode(moeNodePtr);
    OPS_LOG_I(PASS_NAME.c_str(), "dispatch: V2 node removed");

    size_t wiredOutputs = 0;
    for (size_t i = IDX_0; i < kV2TotalOutputs; i++) {
        for (auto &outPair : outputConsumers[i]) {
            edgeRet = graph.AddDataEdge(fusionNode, static_cast<int32_t>(i),
                                        *outPair.first, outPair.second);
            if (edgeRet != ge::GRAPH_SUCCESS) {
                ge::AscendString consumerName;
                outPair.first->GetName(consumerName);
                OPS_LOG_E(PASS_NAME.c_str(), "dispatch AddEdge V3.port%zu→%s.port%d failed",
                          i, consumerName.GetString(), outPair.second);
                return ge::GRAPH_FAILED;
            }
            wiredOutputs++;
        }
    }

    // contextNode → V3.0（最后加，和 MultiInOne 后手动补一致）
    edgeRet = graph.AddDataEdge(contextNodePtr, static_cast<int32_t>(IDX_0),
                                fusionNode, static_cast<int32_t>(IDX_0));
    if (edgeRet != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "dispatch AddEdge for context failed");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(PASS_NAME.c_str(), "dispatch: %zu input edges wired, %zu output edges wired, context→V3.0",
              wiredInputs, wiredOutputs);
    OPS_LOG_D(PASS_NAME.c_str(), "dispatch end to add edge for fusion node");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2FusionPass::FusionNode(
    ge::Graph &graph, ge::GNode &moeDistributeDispatchNode,
    ge::GNode &contextNode, int64_t &hccl_buff_size, const ge::TensorDesc &tensorDesc)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Using MoeDistributeDispatchV2 FusionPass, Fussion Node start !");
    ge::AscendString v2NodeName;
    moeDistributeDispatchNode.GetName(v2NodeName);
    OPS_LOG_I(PASS_NAME.c_str(), "=== Fusing node: %s ===", v2NodeName.GetString());
    ge::GNode fusionNode;
    if (CreateFusionNode(graph, moeDistributeDispatchNode, hccl_buff_size, tensorDesc, fusionNode) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "dispatch CreateFusionNode failed");
        return ge::GRAPH_FAILED;
    }
    if (AddEdge(graph, moeDistributeDispatchNode, fusionNode, contextNode) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "dispatch fusion AddEdge failed");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(PASS_NAME.c_str(), "MoeDistributeDispatchV2FusionPass fun end !");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2FusionPass::Fusion(
    ge::Graph &graph, ge::GNode &moeDistributeDispatchNode, ge::GNode &sharedCtxNode)
{
    OPS_LOG_I(PASS_NAME.c_str(), "MoeDistributeDispatchV2FusionPass fun start !");
#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    int64_t hccl_buff_size = 0;

    if (Mc2MoeGraphContext::GetMc2ContextA5(graph, MOE_CONTEXT_TAG, MOE_EP_NAME,
                                           moeDistributeDispatchNode,
                                           sharedCtxNode, hccl_buff_size) != ge::GRAPH_SUCCESS) {
        OPS_LOG_I(PASS_NAME.c_str(), "GetMc2ContextA5 doesn't support current case, dispatch Failed");
        return ge::GRAPH_SUCCESS;
    }

    ge::Shape shape({static_cast<int64_t>(sizeof(Mc2MoeContext) / sizeof(int32_t))});
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);

    if (FusionNode(graph, moeDistributeDispatchNode, sharedCtxNode, hccl_buff_size, tensorDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "Dispatch FusionNode failed");
        return ge::GRAPH_FAILED;
    }
#endif
    OPS_LOG_I(PASS_NAME.c_str(), "MoeDistributeDispatchV2FusionPass fun end !");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2FusionPass::Run(ge::GraphPtr &graph, ge::CustomPassContext &pass_context)
{
    OPS_LOG_I(PASS_NAME.c_str(), "MoeDistributeDispatchV2FusionPass fun start !");
    (void)pass_context;

    if (IsTargetPlatformSocVersion(PASS_NAME.c_str(), PLATFORM_A2)) {
        OPS_LOG_I(PASS_NAME.c_str(), "A2 platform detected, skip V3 fusion");
        return ge::GRAPH_SUCCESS;
    }

    if (!IsTargetPlatformNpuArch(PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_I(PASS_NAME.c_str(), "Non-A5 platform detected, skip V3 fusion");
        return ge::GRAPH_SUCCESS;
    }

    ge::GNode sharedCtxNode;
    for (auto &gNode : graph->GetDirectNode()) {
        ge::AscendString nodeType;
        if (gNode.GetType(nodeType) != ge::GRAPH_SUCCESS || nodeType != FUSED_OP_TYPE2.c_str()) {
            continue;
        }

        ge::AscendString nameStr;
        gNode.GetName(nameStr);
        OPS_LOG_I(PASS_NAME.c_str(), "found MoeDistributeDispatchV2 node: %s", nameStr.GetString());

        ge::AscendString commAlg;
        if (gNode.GetAttr(ge::AscendString("comm_alg"), commAlg) == ge::GRAPH_SUCCESS &&
            std::string(commAlg.GetString()) == "ccu") {
            OPS_LOG_I(PASS_NAME.c_str(), "comm_alg is ccu, skip V3 fusion (delegated to V1)");
            continue;
        }

        auto ret = Fusion(*graph, gNode, sharedCtxNode);
        if (ret == ge::GRAPH_SUCCESS) {
            OPS_LOG_I(PASS_NAME.c_str(), "node %s: V2->V3 fusion SUCCESS", nameStr.GetString());
        } else {
            OPS_LOG_I(PASS_NAME.c_str(), "node %s: V2->V3 fusion FAILED (ret=%d)", nameStr.GetString(), ret);
        }
    }

    OPS_LOG_I(PASS_NAME.c_str(), "MoeDistributeDispatchV2FusionPass fun end !");
    return ge::GRAPH_SUCCESS;
}

REG_FUSION_PASS(MoeDistributeDispatchV2FusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
}  // namespace ops
#endif  // GRAPH_FUSION_SUPPORT_VERSION
