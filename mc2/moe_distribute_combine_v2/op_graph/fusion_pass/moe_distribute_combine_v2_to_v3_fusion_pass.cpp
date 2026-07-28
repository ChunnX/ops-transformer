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
 * 将图中的 MoeDistributeCombineV2 节点 1:1 替换为 MoeDistributeCombineV3（仅 A5，comm_alg != "ccu"）。
 * V3 相比 V2：头部新增 context 输入（Mc2MoeContext 序列化数据，Const 节点承载），
 * 属性删除 group_ep/group_tp，新增 ccl_buffer_size。通信域 window buffer 地址在编译期通过
 * Mc2MoeContextManager 查询 HCCL 获得，与单算子路径（mc2_context.cpp）语义一致。
 *
 * 输入/输出按原型索引 1:1 映射（V2 port i → V3 port i+1），可选输入槽位按
 * GetInDataNodesAndPortIndexs 是否有生产者判定连边，未连接槽位只保留 desc 占位——
 * 与 torchair "空输入 + 无效描述占位" 构图语义一致，不依赖任何按名查索引接口。
 */
#include "moe_distribute_combine_v2_to_v3_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include <map>
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "op_graph/mc2_moe_context_manager.h"
#include "graph/operator_factory.h"

namespace ops {
namespace {
const std::string FUSION_PASS_NAME = "MoeDistributeCombineV2FusionPass";
const std::string FUSED_OP_TYPE2 = "MoeDistributeCombineV2";
const std::string FUSED_OP_TYPE3 = "MoeDistributeCombineV3";
constexpr size_t kV2TotalInputs = 18; // 5 个必选 + 13 个可选输入
constexpr size_t kV2TotalOutputs = 1; // x

const std::vector<std::string> REQUIRED_INT_ATTRS = {"ep_world_size", "ep_rank_id", "moe_expert_num"};
const std::vector<std::string> OPTIONAL_INT_ATTRS = {
    "tp_world_size",          "tp_rank_id",      "expert_shard_type", "shared_expert_num",
    "shared_expert_rank_num", "global_bs",       "out_dtype",         "comm_quant_mode",
    "group_list_type",        "zero_expert_num", "copy_expert_num",   "const_expert_num"};
const std::vector<std::string> OPTIONAL_STR_ATTRS = {"comm_alg"};
} // namespace

ge::graphStatus MoeDistributeCombineV2FusionPass::AddAttr(ge::GNode &moeNode, ge::GNode &fusionNode,
                                                          int64_t hcclBuffSize)
{
    ge::AscendString nameStr;
    moeNode.GetName(nameStr);
    int64_t attrInt = 0;
    for (const auto &attrName : REQUIRED_INT_ATTRS) {
        if (moeNode.GetAttr(ge::AscendString(attrName.c_str()), attrInt) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "node %s: get required attr %s failed.", nameStr.GetString(),
                      attrName.c_str());
            return ge::GRAPH_FAILED;
        }
        fusionNode.SetAttr(ge::AscendString(attrName.c_str()), attrInt);
    }
    // V3 新增必选属性 ccl_buffer_size
    fusionNode.SetAttr(ge::AscendString("ccl_buffer_size"), hcclBuffSize);
    // 可选属性存在才拷贝；group_ep/group_tp 在 V3 已删除，不拷贝
    for (const auto &attrName : OPTIONAL_INT_ATTRS) {
        if (moeNode.GetAttr(ge::AscendString(attrName.c_str()), attrInt) == ge::GRAPH_SUCCESS) {
            fusionNode.SetAttr(ge::AscendString(attrName.c_str()), attrInt);
        }
    }
    ge::AscendString strVal;
    for (const auto &attrName : OPTIONAL_STR_ATTRS) {
        if (moeNode.GetAttr(ge::AscendString(attrName.c_str()), strVal) == ge::GRAPH_SUCCESS) {
            fusionNode.SetAttr(ge::AscendString(attrName.c_str()), strVal);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineV2FusionPass::CreateFusionNode(ge::Graph &graph, ge::GNode &moeNode,
                                                                   int64_t hcclBuffSize,
                                                                   const ge::TensorDesc &contextDesc,
                                                                   ge::GNode &fusionNode)
{
    ge::AscendString nameStr;
    moeNode.GetName(nameStr);
    auto fusionOp = ge::OperatorFactory::CreateOperator(std::string(nameStr.GetString()) + "_V3", FUSED_OP_TYPE3);
    if (fusionOp.IsEmpty()) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "node %s: %s op not found in factory.", nameStr.GetString(),
                  FUSED_OP_TYPE3.c_str());
        return ge::GRAPH_FAILED;
    }
    // port 0 为新增的 context 输入
    fusionOp.UpdateInputDesc(0U, contextDesc);
    // 输入 desc 按原型索引全量拷贝（含未连接的占位槽位），V2 port i → V3 port i+1
    for (size_t v2Port = 0; v2Port < kV2TotalInputs; ++v2Port) {
        ge::TensorDesc srcDesc;
        if (moeNode.GetInputDesc(static_cast<int32_t>(v2Port), srcDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        fusionOp.UpdateInputDesc(static_cast<uint32_t>(v2Port + 1), srcDesc);
    }
    // 输出 desc 索引不变
    for (size_t v2Port = 0; v2Port < kV2TotalOutputs; ++v2Port) {
        ge::TensorDesc outDesc;
        if (moeNode.GetOutputDesc(static_cast<int32_t>(v2Port), outDesc) == ge::GRAPH_SUCCESS) {
            fusionOp.UpdateOutputDesc(static_cast<uint32_t>(v2Port), outDesc);
        }
    }
    fusionNode = graph.AddNodeByOp(fusionOp);
    return AddAttr(moeNode, fusionNode, hcclBuffSize);
}

ge::graphStatus MoeDistributeCombineV2FusionPass::AddEdge(ge::Graph &graph, ge::GNode &moeNode, ge::GNode &fusionNode,
                                                          ge::GNode &contextNode)
{
    // 输入边：有真实生产者的槽位按 V2 port i → V3 port i+1 重连，未连接槽位跳过
    for (size_t v2Port = 0; v2Port < kV2TotalInputs; ++v2Port) {
        auto srcPair = moeNode.GetInDataNodesAndPortIndexs(static_cast<int32_t>(v2Port));
        if (srcPair.first == nullptr) {
            continue;
        }
        if (graph.AddDataEdge(*srcPair.first, srcPair.second, fusionNode, static_cast<int32_t>(v2Port) + 1) !=
            ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "add input edge for port %zu failed.", v2Port);
            return ge::GRAPH_FAILED;
        }
    }
    // context 边：Const 节点 → V3 port 0
    if (graph.AddDataEdge(contextNode, 0, fusionNode, 0) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "add context edge failed.");
        return ge::GRAPH_FAILED;
    }
    // 输出边：先收集消费者，删除 V2 节点后重连（输出索引不变）
    std::vector<std::vector<std::pair<ge::GNodePtr, int32_t>>> outputConsumers(kV2TotalOutputs);
    for (size_t i = 0; i < kV2TotalOutputs; ++i) {
        outputConsumers[i] = moeNode.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
    }
    graph.RemoveNode(moeNode);
    for (size_t i = 0; i < kV2TotalOutputs; ++i) {
        for (auto &outPair : outputConsumers[i]) {
            if (graph.AddDataEdge(fusionNode, static_cast<int32_t>(i), *outPair.first, outPair.second) !=
                ge::GRAPH_SUCCESS) {
                OPS_LOG_E(FUSION_PASS_NAME.c_str(), "add output edge for port %zu failed.", i);
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineV2FusionPass::FusionNode(ge::Graph &graph, ge::GNode &moeNode,
                                                             ge::GNode &contextNode, int64_t hcclBuffSize,
                                                             const ge::TensorDesc &contextDesc)
{
    ge::GNode fusionNode;
    if (CreateFusionNode(graph, moeNode, hcclBuffSize, contextDesc, fusionNode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return AddEdge(graph, moeNode, fusionNode, contextNode);
}

ge::graphStatus MoeDistributeCombineV2FusionPass::Run(ge::GraphPtr &graph, ge::CustomPassContext &passContext)
{
    (void)passContext;
    if (graph == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 仅 A5 支持 MoeDistributeCombineV3
    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "target platform is not A5, skip.");
        return ge::GRAPH_SUCCESS;
    }
    // 按 groupEp 缓存 context Const 节点与 buffer 大小，同图同通信域的多个 V2 节点复用
    std::map<std::string, std::pair<ge::GNode, int64_t>> contextNodes;
    for (auto &gNode : graph->GetDirectNode()) {
        ge::AscendString nodeType;
        if ((gNode.GetType(nodeType) != ge::GRAPH_SUCCESS) || (nodeType != FUSED_OP_TYPE2.c_str())) {
            continue;
        }
        // ccu 通信算法不支持 mc2Context，保持 V2 不变
        ge::AscendString commAlg;
        if (gNode.GetAttr(ge::AscendString("comm_alg"), commAlg) == ge::GRAPH_SUCCESS &&
            commAlg.GetString() != nullptr && std::string(commAlg.GetString()) == "ccu") {
            continue;
        }
        ge::AscendString groupEp;
        if (gNode.GetAttr(ge::AscendString("group_ep"), groupEp) != ge::GRAPH_SUCCESS ||
            groupEp.GetString() == nullptr) {
            continue;
        }
        // 防御：老 torchair 未占位的 compact 构图若未归位到全 IR 槽位，索引映射不可靠，安全回退 V2
        if (gNode.GetInputsSize() != kV2TotalInputs) {
            OPS_LOG_I(FUSION_PASS_NAME.c_str(), "node input size %zu != expected %zu, skip V2 to V3 fusion for safety.",
                      gNode.GetInputsSize(), kV2TotalInputs);
            continue;
        }
        std::string groupEpStr = groupEp.GetString();
        if (contextNodes.find(groupEpStr) == contextNodes.end()) {
            std::vector<int32_t> contextData;
            int64_t hcclBuffSize = 0;
            if (!ops::Mc2MoeContextManager::GetContextData(groupEpStr, contextData, hcclBuffSize)) {
                OPS_LOG_E(FUSION_PASS_NAME.c_str(), "get mc2 context data failed, V2 node kept as-is.");
                continue;
            }
            ge::GNode contextNode;
            if (ops::Mc2MoeContextManager::CreateContextConstNode(*graph, contextData, contextNode) !=
                ge::GRAPH_SUCCESS) {
                continue;
            }
            contextNodes.emplace(groupEpStr, std::make_pair(contextNode, hcclBuffSize));
        }
        ge::GNode contextNode = contextNodes[groupEpStr].first;
        int64_t hcclBuffSize = contextNodes[groupEpStr].second;
        ge::TensorDesc contextDesc;
        if (contextNode.GetOutputDesc(0, contextDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        ge::AscendString nameStr;
        gNode.GetName(nameStr);
        if (FusionNode(*graph, gNode, contextNode, hcclBuffSize, contextDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "node %s: V2->V3 fusion failed.", nameStr.GetString());
            continue;
        }
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "replace %s with %s success.", nameStr.GetString(), FUSED_OP_TYPE3.c_str());
    }
    return ge::GRAPH_SUCCESS;
}

REG_FUSION_PASS(MoeDistributeCombineV2FusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
} // namespace ops
#endif // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
