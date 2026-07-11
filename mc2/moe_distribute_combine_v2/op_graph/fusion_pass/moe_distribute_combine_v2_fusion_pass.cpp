/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_distribute_combine_v2_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION

#include <string>
#include <vector>
#include "graph/operator_factory.h"
#include "ge/es_graph_builder.h"
#include "mc2_common_log.h"
#include "mc2_platform_info.h"
#include "op_graph/mc2_context_graph.h"
#include "ge/ge_utils.h"

namespace ops {
namespace {
const std::string PASS_NAME = "MoeDistributeCombineV2FusionPass";
const std::string V3_OP_TYPE = "MoeDistributeCombineV3";
const std::string MOE_OP_NAME = "moe_distribute_v2";
const std::string ATTR_GROUP_EP = "group_ep";

const std::vector<std::string> REQUIRED_INT_ATTRS = {"ep_world_size", "ep_rank_id", "moe_expert_num"};
const std::vector<std::string> OPTIONAL_INT_ATTRS = {
    "tp_world_size", "tp_rank_id", "expert_shard_type", "shared_expert_num",
    "shared_expert_rank_num", "global_bs", "out_dtype", "comm_quant_mode",
    "group_list_type", "zero_expert_num", "copy_expert_num", "const_expert_num"};
const std::vector<std::string> OPTIONAL_STR_ATTRS = {"comm_alg"};
const int32_t V3_CONTEXT_INPUT_OFFSET = 1;
}  // namespace

bool MoeDistributeCombineV2FusionPass::MeetRequirements(const ge::GNode &matched_node)
{
    if (IsTargetPlatformSocVersion(PASS_NAME.c_str(), PLATFORM_A2)) {
        return false;
    }
    if (!IsTargetPlatformNpuArch(PASS_NAME.c_str(), NPUARCH_A5)) {
        return false;
    }
    ge::AscendString commAlgStr;
    if (matched_node.GetAttr(ge::AscendString("comm_alg"), commAlgStr) == ge::GRAPH_SUCCESS &&
        commAlgStr.GetString() != nullptr && std::string(commAlgStr.GetString()) == "ccu") {
        return false;
    }
    return true;
}

ge::fusion::GraphUniqPtr MoeDistributeCombineV2FusionPass::Replacement(const ge::GNode &matched_node)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Enter Replacement for MoeDistributeCombineV2FusionPass");

    // 1. Get group_ep from V2 node
    ge::AscendString groupEpStr;
    if (matched_node.GetAttr(ge::AscendString(ATTR_GROUP_EP.c_str()), groupEpStr) != ge::GRAPH_SUCCESS ||
        groupEpStr.GetString() == nullptr) {
        OPS_LOG_E(PASS_NAME.c_str(), "Failed to get group_ep attr");
        return nullptr;
    }
    std::string groupEp(groupEpStr.GetString());

    // 2. Get context data + hcclBuffSize (shared with dispatch via same opName/tag)
    Mc2Aclnn::Mc2MoeContext context{0};
    int64_t hcclBuffSize = 0;
    if (Mc2ContextGraph::GetMc2ContextData(groupEp, MOE_OP_NAME, context, hcclBuffSize) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "GetMc2ContextData failed");
        return nullptr;
    }

    // 3. Collect V2 connected inputs
    auto v2InputNum = matched_node.GetInputsSize();
    std::vector<ge::TensorDesc> connectedInputDescs;
    for (size_t v2Port = 0; v2Port < v2InputNum; ++v2Port) {
        auto [srcNode, srcPort] = matched_node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(v2Port));
        if (srcNode == nullptr) {
            continue;
        }
        ge::TensorDesc v2InputDesc;
        if (matched_node.GetInputDesc(static_cast<int32_t>(v2Port), v2InputDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        connectedInputDescs.push_back(v2InputDesc);
    }

    // 4. Build replacement graph
    auto builder = ge::es::EsGraphBuilder("replacement_combine_v3");

    // 4a. Create Data input nodes
    std::vector<ge::es::EsTensorHolder> inputHolders;
    for (size_t i = 0; i < connectedInputDescs.size(); ++i) {
        auto holder = builder.CreateInput(static_cast<int64_t>(i), nullptr,
                                          connectedInputDescs[i].GetDataType(),
                                          connectedInputDescs[i].GetOriginFormat(),
                                          connectedInputDescs[i].GetShape().GetDims());
        inputHolders.push_back(holder);
    }

    // 4b. Create Const context node
    int64_t ctxElementNum = static_cast<int64_t>(sizeof(Mc2Aclnn::Mc2MoeContext) / sizeof(uint32_t));
    std::vector<uint32_t> ctxVec(reinterpret_cast<const uint32_t *>(&context),
                                 reinterpret_cast<const uint32_t *>(&context) + ctxElementNum);
    auto ctxHolder = builder.CreateConst(ctxVec, std::vector<int64_t>{ctxElementNum});

    // 4c. Get inner graph, create V3 node via OperatorFactory
    auto *innerGraph = builder.GetCGraphBuilder()->GetGraph();
    if (innerGraph == nullptr) {
        OPS_LOG_E(PASS_NAME.c_str(), "Failed to get inner graph from EsGraphBuilder");
        return nullptr;
    }

    ge::AscendString nodeName;
    matched_node.GetName(nodeName);
    auto v3Op = ge::OperatorFactory::CreateOperator(std::string(nodeName.GetString()) + "_V3", V3_OP_TYPE);
    if (v3Op.IsEmpty()) {
        OPS_LOG_E(PASS_NAME.c_str(), "V3 op not found in factory");
        return nullptr;
    }

    // 4d. Set V3 input descs: port 0 = context, port 1..N = V2 inputs
    ge::TensorDesc ctxOutDesc;
    ctxHolder.GetProducer()->GetOutputDesc(0, ctxOutDesc);
    v3Op.UpdateInputDesc(static_cast<uint32_t>(0), ctxOutDesc);
    for (size_t i = 0; i < connectedInputDescs.size(); ++i) {
        v3Op.UpdateInputDesc(static_cast<uint32_t>(i + V3_CONTEXT_INPUT_OFFSET), connectedInputDescs[i]);
    }

    // 4e. Set V3 output descs (combine has 1 output: x)
    auto v2OutputNum = matched_node.GetOutputsSize();
    for (size_t v2Port = 0; v2Port < v2OutputNum; ++v2Port) {
        ge::TensorDesc v2OutputDesc;
        if (matched_node.GetOutputDesc(static_cast<int32_t>(v2Port), v2OutputDesc) == ge::GRAPH_SUCCESS) {
            v3Op.UpdateOutputDesc(static_cast<uint32_t>(v2Port), v2OutputDesc);
        }
    }

    // 4f. Copy attrs
    int64_t intVal = 0;
    for (const auto &attrName : REQUIRED_INT_ATTRS) {
        if (matched_node.GetAttr(ge::AscendString(attrName.c_str()), intVal) == ge::GRAPH_SUCCESS) {
            v3Op.SetAttr(attrName.c_str(), intVal);
        }
    }
    v3Op.SetAttr("ccl_buffer_size", hcclBuffSize);
    for (const auto &attrName : OPTIONAL_INT_ATTRS) {
        if (matched_node.GetAttr(ge::AscendString(attrName.c_str()), intVal) == ge::GRAPH_SUCCESS) {
            v3Op.SetAttr(attrName.c_str(), intVal);
        }
    }
    ge::AscendString strVal;
    for (const auto &attrName : OPTIONAL_STR_ATTRS) {
        if (matched_node.GetAttr(ge::AscendString(attrName.c_str()), strVal) == ge::GRAPH_SUCCESS) {
            v3Op.SetAttr(attrName.c_str(), strVal.GetString());
        }
    }

    // 4g. Add V3 node and connect edges
    auto v3Node = innerGraph->AddNodeByOp(v3Op);
    innerGraph->AddDataEdge(*ctxHolder.GetProducer(), 0, v3Node, 0);
    for (size_t i = 0; i < inputHolders.size(); ++i) {
        innerGraph->AddDataEdge(*inputHolders[i].GetProducer(), 0, v3Node,
                                static_cast<int32_t>(i + V3_CONTEXT_INPUT_OFFSET));
    }

    // 4h. Collect V3 outputs and build
    std::vector<ge::es::EsTensorHolder> outputHolders;
    for (size_t v2Port = 0; v2Port < v2OutputNum; ++v2Port) {
        auto *cHolder = builder.GetCGraphBuilder()->GetTensorHolderFromNode(v3Node, static_cast<int32_t>(v2Port));
        outputHolders.emplace_back(cHolder);
    }

    auto repGraph = builder.BuildAndReset(outputHolders);
    if (repGraph == nullptr) {
        OPS_LOG_E(PASS_NAME.c_str(), "BuildAndReset failed");
        return nullptr;
    }

    // 5. InferShape
    std::vector<ge::Shape> inputShapes;
    for (const auto &desc : connectedInputDescs) {
        inputShapes.emplace_back(desc.GetShape());
    }
    if (ge::GeUtils::InferShape(*repGraph, inputShapes) != ge::SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "InferShape failed for replacement graph");
        return nullptr;
    }

    return repGraph;
}

REG_DECOMPOSE_PASS(MoeDistributeCombineV2FusionPass, {"MoeDistributeCombineV2"})
    .Stage(ge::CustomPassStage::kAfterInferShape);

}  // namespace ops
#endif  // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
