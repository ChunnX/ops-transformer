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
 * MoeDistributeCombineV2 -> MoeDistributeCombineV3 fusion pass
 *
 *   expand_x expert_ids expand_idx ep_send_counts expert_scales tp_send_counts ...
 *                                   |
 *                          MoeDistributeCombineV2
 *                                   |
 *                                   x
 *
 *   After fusion (A5 only, non-ccu):
 *   context expand_x expert_ids expand_idx ep_send_counts expert_scales tp_send_counts ...
 *                                   |
 *                          MoeDistributeCombineV3
 *                                   |
 *                                   x
 */
#include "moe_distribute_combine_v2_fusion_pass.h"
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "graph/graph.h"
#include "graph/gnode.h"
#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "graph/ge_error_codes.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "mc2_common_log.h"
#include "mc2_platform_info.h"
#include "op_graph/mc2_moe_context.h"

namespace ops {
namespace {
const std::string PASS_NAME = "MoeDistributeCombineV2FusionPass";
const std::string FUSED_OP_TYPE2 = "MoeDistributeCombineV2";
const std::string FUSED_OP_TYPE3 = "MoeDistributeCombineV3";
const std::string MOE_CONTEXT_TAG = "MOE_DISPATCH_COMBINE_V2_CONTEXT";
const std::string MOE_EP_NAME = "group_ep";

const uint32_t IDX_0 = 0;

const std::vector<std::string> REQUIRED_INT_ATTRS = {
    "ep_world_size", "ep_rank_id", "moe_expert_num"
};

const std::vector<std::string> OPTIONAL_INT_ATTRS = {
    "tp_world_size", "tp_rank_id", "expert_shard_type", "shared_expert_num",
    "shared_expert_rank_num", "global_bs", "out_dtype", "comm_quant_mode",
    "group_list_type", "zero_expert_num", "copy_expert_num", "const_expert_num"
};

const std::vector<std::string> OPTIONAL_STR_ATTRS = {"comm_alg"};
}  // namespace

ge::graphStatus MoeDistributeCombineV2FusionPass::AddAttr(ge::GNode &moeDistributeCombineNode,
                                                          ge::GNode &fusionNode,
                                                          int64_t &hccl_buff_size)
{
    OPS_LOG_D(PASS_NAME.c_str(), "start combine fusion AddAttr");
    ge::AscendString nameStr;
    moeDistributeCombineNode.GetName(nameStr);

    int64_t attrInt = 0;
    for (const auto &attrName : REQUIRED_INT_ATTRS) {
        if (moeDistributeCombineNode.GetAttr(ge::AscendString(attrName.c_str()), attrInt) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "node %s: Get required attr %s failed", nameStr.GetString(), attrName.c_str());
            return ge::GRAPH_FAILED;
        }
        fusionNode.SetAttr(ge::AscendString(attrName.c_str()), attrInt);
    }

    fusionNode.SetAttr(ge::AscendString("ccl_buffer_size"), hccl_buff_size);

    for (const auto &attrName : OPTIONAL_INT_ATTRS) {
        if (moeDistributeCombineNode.GetAttr(ge::AscendString(attrName.c_str()), attrInt) == ge::GRAPH_SUCCESS) {
            fusionNode.SetAttr(ge::AscendString(attrName.c_str()), attrInt);
        }
    }

    ge::AscendString strVal;
    for (const auto &attrName : OPTIONAL_STR_ATTRS) {
        if (moeDistributeCombineNode.GetAttr(ge::AscendString(attrName.c_str()), strVal) == ge::GRAPH_SUCCESS) {
            fusionNode.SetAttr(ge::AscendString(attrName.c_str()), strVal);
        }
    }
    OPS_LOG_D(PASS_NAME.c_str(), "end combine fusion AddAttr");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineV2FusionPass::CreateFusionNode(
    ge::Graph &graph, ge::GNode &moeDistributeCombineNode,
    int64_t &hccl_buff_size, const ge::TensorDesc &tensorDesc, ge::GNode &fusionNode)
{
    ge::AscendString nameStr;
    moeDistributeCombineNode.GetName(nameStr);
    OPS_LOG_D(PASS_NAME.c_str(), "start combine fusion CreateFusionNode");

    std::string v3NodeName = std::string(nameStr.GetString()) + "_V3";
    auto fusionOp = ge::OperatorFactory::CreateOperator(v3NodeName, FUSED_OP_TYPE3);
    if (fusionOp.IsEmpty()) {
        OPS_LOG_E(PASS_NAME.c_str(), "node %s: %s op not found in factory", nameStr.GetString(),
                  FUSED_OP_TYPE3.c_str());
        return ge::GRAPH_FAILED;
    }

    // context 输入描述（端口 0，与 Const 输出一致）
    fusionOp.UpdateInputDesc(static_cast<uint32_t>(IDX_0), tensorDesc);

    // 从 V2 拷贝所有输入描述（V3.port = V2.port + 1）
    size_t v2InputNum = moeDistributeCombineNode.GetInputsSize();
    size_t copiedCount = 0;
    for (size_t v2Port = 0; v2Port < v2InputNum; v2Port++) {
        ge::TensorDesc v2Desc;
        if (moeDistributeCombineNode.GetInputDesc(static_cast<int32_t>(v2Port), v2Desc) == ge::GRAPH_SUCCESS) {
            fusionOp.UpdateInputDesc(static_cast<uint32_t>(v2Port + 1), v2Desc);
            copiedCount++;
        }
    }
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: %zu/%zu V2 input descs copied to V3", nameStr.GetString(), copiedCount, v2InputNum);

    // 从 V2 拷贝所有输出描述到 V3
    size_t v2OutputNum = moeDistributeCombineNode.GetOutputsSize();
    for (size_t v2Port = 0; v2Port < v2OutputNum; v2Port++) {
        ge::TensorDesc v2OutDesc;
        if (moeDistributeCombineNode.GetOutputDesc(static_cast<int32_t>(v2Port), v2OutDesc) == ge::GRAPH_SUCCESS) {
            fusionOp.UpdateOutputDesc(static_cast<uint32_t>(v2Port), v2OutDesc);
        }
    }
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: %zu V2 output descs copied to V3", nameStr.GetString(), v2OutputNum);

    fusionNode = graph.AddNodeByOp(fusionOp);
    ge::AscendString v3Name;
    fusionNode.GetName(v3Name);
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: V3 node added as %s", nameStr.GetString(), v3Name.GetString());

    if (AddAttr(moeDistributeCombineNode, fusionNode, hccl_buff_size) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "node %s: AddAttr failed", nameStr.GetString());
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_D(PASS_NAME.c_str(), "end combine fusion CreateFusionNode");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineV2FusionPass::AddEdge(
    ge::Graph &graph, ge::GNode &moeNodePtr, ge::GNode &fusionNode,
    ge::GNode &contextNodePtr)
{
    OPS_LOG_D(PASS_NAME.c_str(), "combine start to add edge for fusion node");

    size_t v2TotalInputs = moeNodePtr.GetInputsSize();
    size_t v2TotalOutputs = moeNodePtr.GetOutputsSize();
    OPS_LOG_I(PASS_NAME.c_str(), "combine: V2 inputs=%zu outputs=%zu", v2TotalInputs, v2TotalOutputs);
    ge::graphStatus edgeRet;

    // V2 输入源 → V3 (V3.port = V2.port + 1)
    size_t wiredInputs = 0;
    for (size_t i = IDX_0; i < v2TotalInputs; i++) {
        auto srcPair = moeNodePtr.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        if (srcPair.first == nullptr) continue;
        size_t v3Port = i + 1;
        edgeRet = graph.AddDataEdge(*srcPair.first, srcPair.second,
                                    fusionNode, static_cast<int32_t>(v3Port));
        if (edgeRet != ge::GRAPH_SUCCESS) {
            ge::AscendString srcName;
            srcPair.first->GetName(srcName);
            OPS_LOG_E(PASS_NAME.c_str(), "combine AddDataEdge from %s.port%d→V3.port%zu failed",
                      srcName.GetString(), srcPair.second, v3Port);
            return ge::GRAPH_FAILED;
        }
        wiredInputs++;
    }
    OPS_LOG_I(PASS_NAME.c_str(), "combine: %zu input edges wired", wiredInputs);

    // 收集 V2 输出消费者 → 删除 V2 → 重连 V3 输出到消费者
    std::vector<std::vector<std::pair<ge::GNodePtr, int32_t>>> outputConsumers(v2TotalOutputs);
    for (size_t i = IDX_0; i < v2TotalOutputs; i++) {
        outputConsumers[i] = moeNodePtr.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
    }

    graph.RemoveNode(moeNodePtr);
    OPS_LOG_I(PASS_NAME.c_str(), "combine: V2 node removed");

    size_t wiredOutputs = 0;
    for (size_t i = IDX_0; i < v2TotalOutputs; i++) {
        for (auto &outPair : outputConsumers[i]) {
            edgeRet = graph.AddDataEdge(fusionNode, static_cast<int32_t>(i),
                                        *outPair.first, outPair.second);
            if (edgeRet != ge::GRAPH_SUCCESS) {
                ge::AscendString consumerName;
                outPair.first->GetName(consumerName);
                OPS_LOG_E(PASS_NAME.c_str(), "combine AddEdge V3.port%zu→%s.port%d failed",
                          i, consumerName.GetString(), outPair.second);
                return ge::GRAPH_FAILED;
            }
            wiredOutputs++;
        }
    }
    OPS_LOG_I(PASS_NAME.c_str(), "combine: %zu output edges wired", wiredOutputs);

    // contextNode → V3.0（最后加，和 MultiInOne 后手动补一致）
    edgeRet = graph.AddDataEdge(contextNodePtr, static_cast<int32_t>(IDX_0),
                                fusionNode, static_cast<int32_t>(IDX_0));
    if (edgeRet != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "combine AddEdge for context failed");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_D(PASS_NAME.c_str(), "combine end to add edge for fusion node");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineV2FusionPass::FusionNode(
    ge::Graph &graph, ge::GNode &moeDistributeCombineNode,
    ge::GNode &contextNode, int64_t &hccl_buff_size, const ge::TensorDesc &tensorDesc)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Using MoeDistributeCombineV2 FusionPass, Fussion Node start !");
    ge::GNode fusionNode;
    if (CreateFusionNode(graph, moeDistributeCombineNode, hccl_buff_size, tensorDesc, fusionNode) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "combine CreateFusionNode failed");
        return ge::GRAPH_FAILED;
    }
    if (AddEdge(graph, moeDistributeCombineNode, fusionNode, contextNode) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "combine fusion AddEdge failed");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(PASS_NAME.c_str(), "MoeDistributeCombineV2FusionPass fun end !");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineV2FusionPass::Fusion(
    ge::Graph &graph, ge::GNode &moeDistributeCombineNode)
{
    OPS_LOG_I(PASS_NAME.c_str(), "MoeDistributeCombineV2FusionPass fun start !");
#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    ge::GNode contextNode;
    int64_t hccl_buff_size = 0;

    Mc2MoeContext mc2Ctx;
    if (MoeContextManager::GetMc2ContextA5(MOE_CONTEXT_TAG, MOE_EP_NAME, moeDistributeCombineNode,
                                           mc2Ctx, hccl_buff_size) != ge::GRAPH_SUCCESS) {
        OPS_LOG_I(PASS_NAME.c_str(), "GetMc2ContextA5 doesn't support current case, combine Failed");
        return ge::GRAPH_SUCCESS;  // 回退（无变化）
    }
    OPS_LOG_I(PASS_NAME.c_str(), "GetMc2ContextA5 OK, cclBufferSize=%ld", hccl_buff_size);

    ge::Shape shape({static_cast<int64_t>(sizeof(Mc2MoeContext) / sizeof(int32_t))});
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    ge::Tensor tensorValue(tensorDesc, reinterpret_cast<const uint8_t *>(&mc2Ctx), sizeof(Mc2MoeContext));

    auto constOp = ge::OperatorFactory::CreateOperator("mc2_moe_context", "Const");
    if (constOp.IsEmpty()) {
        OPS_LOG_E(PASS_NAME.c_str(), "Const op not found in factory");
        return ge::GRAPH_SUCCESS;
    }
    constOp.UpdateOutputDesc(static_cast<uint32_t>(IDX_0), tensorDesc);
    constOp.SetAttr("value", tensorValue);
    OPS_LOG_I(PASS_NAME.c_str(), "Const op: UpdateOutputDesc(idx=0, DT_INT32) + SetAttr(value) done");
    contextNode = graph.AddNodeByOp(constOp);
    ge::AscendString ctxNodeName;
    contextNode.GetName(ctxNodeName);
    OPS_LOG_I(PASS_NAME.c_str(), "Context Const node added as %s", ctxNodeName.GetString());
    {
        ge::TensorDesc outDesc;
        if (contextNode.GetOutputDesc(static_cast<int32_t>(IDX_0), outDesc) == ge::GRAPH_SUCCESS) {
            OPS_LOG_I(PASS_NAME.c_str(), "Context Const output dtype=%d fmt=%d shape=%zu",
                      static_cast<int>(outDesc.GetDataType()), static_cast<int>(outDesc.GetFormat()),
                      outDesc.GetShape().GetDimNum());
        } else {
            OPS_LOG_E(PASS_NAME.c_str(), "Context Const output desc INVALID!");
        }
    }

    if (FusionNode(graph, moeDistributeCombineNode, contextNode, hccl_buff_size, tensorDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "Combine FusionNode failed");
        return ge::GRAPH_FAILED;
    }
#endif
    OPS_LOG_I(PASS_NAME.c_str(), "MoeDistributeCombineV2FusionPass fun end !");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineV2FusionPass::Run(ge::GraphPtr &graph, ge::CustomPassContext &pass_context)
{
    OPS_LOG_I(PASS_NAME.c_str(), "MoeDistributeCombineV2FusionPass fun start !");
    (void)pass_context;

    if (IsTargetPlatformSocVersion(PASS_NAME.c_str(), PLATFORM_A2)) {
        OPS_LOG_I(PASS_NAME.c_str(), "A2 platform detected, skip V3 fusion");
        return ge::GRAPH_SUCCESS;
    }

    if (!IsTargetPlatformNpuArch(PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_I(PASS_NAME.c_str(), "Non-A5 platform detected, skip V3 fusion");
        return ge::GRAPH_SUCCESS;
    }

    for (auto &gNode : graph->GetDirectNode()) {
        ge::AscendString nodeType;
        if (gNode.GetType(nodeType) != ge::GRAPH_SUCCESS || nodeType != FUSED_OP_TYPE2.c_str()) {
            continue;
        }

        ge::AscendString nameStr;
        gNode.GetName(nameStr);
        OPS_LOG_I(PASS_NAME.c_str(), "found MoeDistributeCombineV2 node: %s", nameStr.GetString());

        ge::AscendString commAlg;
        if (gNode.GetAttr(ge::AscendString("comm_alg"), commAlg) == ge::GRAPH_SUCCESS &&
            std::string(commAlg.GetString()) == "ccu") {
            OPS_LOG_I(PASS_NAME.c_str(), "comm_alg is ccu, skip V3 fusion (delegated to V1)");
            continue;
        }

        auto ret = Fusion(*graph, gNode);
        if (ret == ge::GRAPH_SUCCESS) {
            OPS_LOG_I(PASS_NAME.c_str(), "node %s: V2->V3 fusion SUCCESS", nameStr.GetString());
        } else {
            OPS_LOG_I(PASS_NAME.c_str(), "node %s: V2->V3 fusion FAILED (ret=%d)", nameStr.GetString(), ret);
        }
    }

    OPS_LOG_I(PASS_NAME.c_str(), "MoeDistributeCombineV2FusionPass fun end !");
    return ge::GRAPH_SUCCESS;
}

REG_FUSION_PASS(MoeDistributeCombineV2FusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
}  // namespace ops
#endif  // GRAPH_FUSION_SUPPORT_VERSION
