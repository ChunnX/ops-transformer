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
#include "op_graph/mc2_moe_context.h"

namespace ops {
namespace {

const std::string PASS_NAME = "MoeDistributeDispatchV2FusionPass";
const std::string FUSED_OP_TYPE_V2 = "MoeDistributeDispatchV2";
const std::string FUSED_OP_TYPE_V3 = "MoeDistributeDispatchV3";
const std::string MOE_CONTEXT_TAG = "MOE_DISPATCH_COMBINE_V2_CONTEXT";
const std::string MOE_EP_NAME = "group_ep";

const std::vector<std::string> REQUIRED_INT_ATTRS = {
    "ep_world_size", "ep_rank_id", "moe_expert_num"
};

const std::vector<std::string> OPTIONAL_INT_ATTRS = {
    "tp_world_size", "tp_rank_id", "expert_shard_type", "shared_expert_num",
    "shared_expert_rank_num", "quant_mode", "global_bs", "expert_token_nums_type",
    "zero_expert_num", "copy_expert_num", "const_expert_num", "y_dtype"
};

const std::vector<std::string> OPTIONAL_STR_ATTRS = {"comm_alg"};

ge::graphStatus CopyAttrToV3Node(ge::GNode &v2Node, ge::Operator &v3Op)
{
    int64_t intVal = 0;
    ge::AscendString nameStr;
    v2Node.GetName(nameStr);

    for (const auto &attrName : REQUIRED_INT_ATTRS) {
        if (v2Node.GetAttr(ge::AscendString(attrName.c_str()), intVal) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "node %s: Get required attr %s failed", nameStr.GetString(), attrName.c_str());
            return ge::GRAPH_FAILED;
        }
        v3Op.SetAttr(attrName, intVal);
        OPS_LOG_D(PASS_NAME.c_str(), "node %s: set required attr %s=%ld", nameStr.GetString(), attrName.c_str(), intVal);
    }

    int optionalCopied = 0;
    for (const auto &attrName : OPTIONAL_INT_ATTRS) {
        if (v2Node.GetAttr(ge::AscendString(attrName.c_str()), intVal) == ge::GRAPH_SUCCESS) {
            v3Op.SetAttr(attrName, intVal);
            optionalCopied++;
        }
    }

    ge::AscendString strVal;
    for (const auto &attrName : OPTIONAL_STR_ATTRS) {
        if (v2Node.GetAttr(ge::AscendString(attrName.c_str()), strVal) == ge::GRAPH_SUCCESS) {
            v3Op.SetAttr(attrName, std::string(strVal.GetString()));
            optionalCopied++;
        }
    }
    OPS_LOG_D(PASS_NAME.c_str(), "node %s: %d optional attrs copied", nameStr.GetString(), optionalCopied);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReplaceDispatchV2WithV3(ge::Graph &graph, ge::GNode &v2Node)
{
#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    ge::AscendString nameStr;
    v2Node.GetName(nameStr);
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: start V2->V3 replacement", nameStr.GetString());

    Mc2MoeContext mc2Ctx;
    int64_t cclBufferSize = 0;
    if (MoeContextManager::GetMc2ContextA5(MOE_CONTEXT_TAG, MOE_EP_NAME, v2Node,
                                           mc2Ctx, cclBufferSize) != ge::GRAPH_SUCCESS) {
        OPS_LOG_I(PASS_NAME.c_str(), "node %s: GetMc2ContextA5 failed, keep V2", nameStr.GetString());
        return ge::GRAPH_SUCCESS;
    }
    OPS_LOG_D(PASS_NAME.c_str(), "node %s: context obtained, cclBufferSize=%ld", nameStr.GetString(), cclBufferSize);

    ge::Shape shape({static_cast<int64_t>(sizeof(Mc2MoeContext) / sizeof(int32_t))});
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    ge::Tensor tensorValue(tensorDesc, reinterpret_cast<const uint8_t *>(&mc2Ctx), sizeof(Mc2MoeContext));

    auto constOp = ge::OperatorFactory::CreateOperator("mc2_moe_context", "Const");
    if (constOp.IsEmpty()) {
        OPS_LOG_E(PASS_NAME.c_str(), "node %s: Const op not found in factory", nameStr.GetString());
        return ge::GRAPH_SUCCESS;
    }
    constOp.SetAttr("value", tensorValue);

    std::string v3NodeName = std::string(nameStr.GetString()) + "_V3";
    auto v3Op = ge::OperatorFactory::CreateOperator(v3NodeName, FUSED_OP_TYPE_V3);
    if (v3Op.IsEmpty()) {
        OPS_LOG_E(PASS_NAME.c_str(), "node %s: %s op not found in factory", nameStr.GetString(),
                  FUSED_OP_TYPE_V3.c_str());
        return ge::GRAPH_SUCCESS;
    }

    if (CopyAttrToV3Node(v2Node, v3Op) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "node %s: CopyAttrToV3Node failed", nameStr.GetString());
        return ge::GRAPH_FAILED;
    }
    v3Op.SetAttr("ccl_buffer_size", cclBufferSize);
    OPS_LOG_D(PASS_NAME.c_str(), "node %s: attrs set, adding nodes to graph...", nameStr.GetString());

    auto constGNode = graph.AddNodeByOp(constOp);
    auto v3GNode = graph.AddNodeByOp(v3Op);
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: Const + V3 nodes added to graph", nameStr.GetString());

    size_t v2InputSize = v2Node.GetInputsSize();
    size_t v2OutputSize = v2Node.GetOutputsSize();
    size_t v3InputSize = v3GNode.GetInputsSize();
    size_t v3OutputSize = v3GNode.GetOutputsSize();
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: V2 has %zu inputs + %zu outputs, V3 has %zu inputs + %zu outputs",
              nameStr.GetString(), v2InputSize, v2OutputSize, v3InputSize, v3OutputSize);
    ge::graphStatus edgeRet;

    OPS_LOG_I(PASS_NAME.c_str(), "node %s: [1/5] wiring Const→V3.0 ...", nameStr.GetString());
    edgeRet = graph.AddDataEdge(constGNode, 0, v3GNode, 0);
    if (edgeRet != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "node %s: AddDataEdge Const→V3.0 FAILED (ret=%d)", nameStr.GetString(), edgeRet);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_I(PASS_NAME.c_str(), "node %s: [2/5] wiring V2 inputs→V3 ...", nameStr.GetString());
    int skippedInputs = 0;
    int v3MissingInputs = 0;
    int wiredInputs = 0;
    for (size_t i = 0; i < v2InputSize; i++) {
        auto srcPair = v2Node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        size_t v3Port = i + 1;
        if (srcPair.first == nullptr) {
            OPS_LOG_D(PASS_NAME.c_str(), "node %s: V2.port%zu→V3.port%zu: src is nullptr (optional unfed), skip",
                      nameStr.GetString(), i, v3Port);
            skippedInputs++;
            continue;
        }
        ge::AscendString srcName;
        srcPair.first->GetName(srcName);
        if (v3Port >= v3InputSize) {
            OPS_LOG_D(PASS_NAME.c_str(), "node %s: V2.port%zu[%s]→V3.port%zu: V3 only has %zu inputs, skip",
                      nameStr.GetString(), i, srcName.GetString(), v3Port, v3InputSize);
            v3MissingInputs++;
            continue;
        }
        edgeRet = graph.AddDataEdge(*srcPair.first, srcPair.second, v3GNode, static_cast<int32_t>(v3Port));
        if (edgeRet != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "node %s: AddDataEdge V2.port%zu[%s].%d→V3.port%zu FAILED (ret=%d)",
                      nameStr.GetString(), i, srcName.GetString(), srcPair.second, v3Port, edgeRet);
            return ge::GRAPH_FAILED;
        }
        OPS_LOG_D(PASS_NAME.c_str(), "node %s: V2.port%zu[%s].%d→V3.port%zu OK", nameStr.GetString(), i,
                  srcName.GetString(), srcPair.second, v3Port);
        wiredInputs++;
    }
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: input wiring done - %d OK, %d null-src-skipped, %d anchor-missing-skipped",
              nameStr.GetString(), wiredInputs, skippedInputs, v3MissingInputs);

    OPS_LOG_I(PASS_NAME.c_str(), "node %s: [3/5] collecting V2 output consumers ...", nameStr.GetString());
    std::vector<std::vector<std::pair<ge::GNodePtr, int32_t>>> v2OutputConsumers(v2OutputSize);
    int totalConsumers = 0;
    for (size_t i = 0; i < v2OutputSize; i++) {
        v2OutputConsumers[i] = v2Node.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
        totalConsumers += static_cast<int>(v2OutputConsumers[i].size());
        for (auto &outPair : v2OutputConsumers[i]) {
            ge::AscendString consumerName;
            outPair.first->GetName(consumerName);
            OPS_LOG_D(PASS_NAME.c_str(), "node %s: V2.port%zu→%s.port%d", nameStr.GetString(), i,
                      consumerName.GetString(), outPair.second);
        }
    }
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: %d output consumers recorded across %zu ports",
              nameStr.GetString(), totalConsumers, v2OutputSize);

    OPS_LOG_I(PASS_NAME.c_str(), "node %s: [4/5] removing V2 node ...", nameStr.GetString());
    graph.RemoveNode(v2Node);
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: V2 removed from graph", nameStr.GetString());

    OPS_LOG_I(PASS_NAME.c_str(), "node %s: [5/5] wiring V3 outputs→consumers ...", nameStr.GetString());
    int v3MissingOutputs = 0;
    int totalOutputEdges = 0;
    for (size_t i = 0; i < v2OutputSize; i++) {
        if (i >= v3OutputSize) {
            OPS_LOG_D(PASS_NAME.c_str(), "node %s: V3.port%zu not exist, V3 only %zu outputs, skip %zu consumers",
                      nameStr.GetString(), i, v3OutputSize, v2OutputConsumers[i].size());
            v3MissingOutputs += static_cast<int>(v2OutputConsumers[i].size());
            continue;
        }
        for (auto &outPair : v2OutputConsumers[i]) {
            edgeRet = graph.AddDataEdge(v3GNode, static_cast<int32_t>(i), *outPair.first, outPair.second);
            if (edgeRet != ge::GRAPH_SUCCESS) {
                ge::AscendString consumerName;
                outPair.first->GetName(consumerName);
                OPS_LOG_E(PASS_NAME.c_str(), "node %s: AddDataEdge V3.port%zu→%s.port%d FAILED (ret=%d)",
                          nameStr.GetString(), i, consumerName.GetString(), outPair.second, edgeRet);
                return ge::GRAPH_FAILED;
            }
            totalOutputEdges++;
        }
    }
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: output wiring done - %d OK, %d anchor-missing-skipped. fusion COMPLETE!",
              nameStr.GetString(), totalOutputEdges, v3MissingOutputs);
#else
    OPS_LOG_I(PASS_NAME.c_str(), "HCOMM not available, skip V3 fusion");
#endif
    return ge::GRAPH_SUCCESS;
}

}  // namespace

ge::graphStatus MoeDistributeDispatchV2FusionPass::Run(ge::GraphPtr &graph, ge::CustomPassContext &pass_context)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Run start! total direct nodes = %zu", graph->GetDirectNode().size());

    if (IsTargetPlatformSocVersion(PASS_NAME.c_str(), PLATFORM_A2)) {
        OPS_LOG_I(PASS_NAME.c_str(), "A2 platform detected, skip V3 fusion");
        return ge::GRAPH_SUCCESS;
    }

    if (!IsTargetPlatformNpuArch(PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_I(PASS_NAME.c_str(), "Non-A5 platform detected, skip V3 fusion");
        return ge::GRAPH_SUCCESS;
    }
    OPS_LOG_I(PASS_NAME.c_str(), "A5 platform confirmed, scanning for V2 nodes...");

    for (auto &gNode : graph->GetDirectNode()) {
        ge::AscendString nodeType;
        if (gNode.GetType(nodeType) != ge::GRAPH_SUCCESS || nodeType != FUSED_OP_TYPE_V2.c_str()) {
            continue;
        }

        ge::AscendString nameStr;
        gNode.GetName(nameStr);
        OPS_LOG_I(PASS_NAME.c_str(), "found MoeDistributeDispatchV2 node: %s", nameStr.GetString());

        ge::AscendString commAlg;
        if (gNode.GetAttr(ge::AscendString("comm_alg"), commAlg) == ge::GRAPH_SUCCESS &&
            std::string(commAlg.GetString()) == "ccu") {
            OPS_LOG_I(PASS_NAME.c_str(), "node %s: comm_alg=ccu, skip V3 fusion (delegated to V1)", nameStr.GetString());
            continue;
        }

        auto ret = ReplaceDispatchV2WithV3(*graph, gNode);
        if (ret == ge::GRAPH_SUCCESS) {
            OPS_LOG_I(PASS_NAME.c_str(), "node %s: V2->V3 fusion SUCCESS", nameStr.GetString());
        } else {
            OPS_LOG_I(PASS_NAME.c_str(), "node %s: V2->V3 fusion FAILED (ret=%d), V2 kept", nameStr.GetString(), ret);
        }
    }

    OPS_LOG_I(PASS_NAME.c_str(), "Run end!");
    return ge::GRAPH_SUCCESS;
}

REG_FUSION_PASS(MoeDistributeDispatchV2FusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
}  // namespace ops
#endif  // GRAPH_FUSION_SUPPORT_VERSION
