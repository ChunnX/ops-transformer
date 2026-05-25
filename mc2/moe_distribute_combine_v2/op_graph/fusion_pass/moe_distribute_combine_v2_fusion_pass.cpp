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
const std::string FUSED_OP_TYPE_V2 = "MoeDistributeCombineV2";
const std::string FUSED_OP_TYPE_V3 = "MoeDistributeCombineV3";
const std::string MOE_CONTEXT_TAG = "MOE_DISPATCH_COMBINE_V2_CONTEXT";
const std::string MOE_EP_NAME = "group_ep";

const std::vector<std::string> REQUIRED_INT_ATTRS = {
    "ep_world_size", "ep_rank_id", "moe_expert_num"
};

const std::vector<std::string> OPTIONAL_INT_ATTRS = {
    "tp_world_size", "tp_rank_id", "expert_shard_type", "shared_expert_num",
    "shared_expert_rank_num", "global_bs", "out_dtype", "comm_quant_mode",
    "group_list_type", "zero_expert_num", "copy_expert_num", "const_expert_num"
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

ge::graphStatus ReplaceCombineV2WithV3(ge::Graph &graph, ge::GNode &v2Node)
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

    // Mark connected OPTIONAL_INPUT descriptors valid for FE_TransInsert.
    static const struct { size_t v2Port; const char *v3Name; ge::DataType dtype; } kOptMapping[] = {
        {5, "tp_send_counts", ge::DT_INT32},         {6, "x_active_mask", ge::DT_BOOL},
        {7, "activation_scale", ge::DT_FLOAT},       {8, "weight_scale", ge::DT_FLOAT},
        {9, "group_list", ge::DT_INT64},             {10, "expand_scales", ge::DT_FLOAT},
        {11, "shared_expert_x", ge::DT_BF16},        {12, "elastic_info", ge::DT_INT32},
        {13, "ori_x", ge::DT_BF16},                  {14, "const_expert_alpha_1", ge::DT_BF16},
        {15, "const_expert_alpha_2", ge::DT_BF16},   {16, "const_expert_v", ge::DT_BF16},
        {17, "performance_info", ge::DT_INT64},
    };
    for (auto &m : kOptMapping) {
        auto srcPair = v2Node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(m.v2Port));
        if (srcPair.first == nullptr) continue;
        ge::TensorDesc validDesc(ge::Shape({1}), ge::FORMAT_ND, m.dtype);
        v3Op.UpdateInputDesc(m.v3Name, validDesc);
        OPS_LOG_D(PASS_NAME.c_str(), "node %s: validated V3 input '%s' (V2.port%zu connected)",
                  nameStr.GetString(), m.v3Name, m.v2Port);
    }

    OPS_LOG_D(PASS_NAME.c_str(), "node %s: attrs set, adding nodes to graph (ccl_buffer_size=%ld)...",
              nameStr.GetString(), static_cast<long>(cclBufferSize));

    auto constGNode = graph.AddNodeByOp(constOp);
    auto v3GNode = graph.AddNodeByOp(v3Op);
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: Const + V3 nodes added to graph", nameStr.GetString());

    size_t v2ValidInputs = v2Node.GetInputsSize();
    size_t v2ValidOutputs = v2Node.GetOutputsSize();
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: [SIZE] V2(valid_in=%zu,valid_out=%zu)",
              nameStr.GetString(), v2ValidInputs, v2ValidOutputs);

    // V2 total inputs from REG_OP: 5 INPUT + 13 OPTIONAL_INPUT = 18
    constexpr size_t kV2TotalInputs = 18;
    ge::graphStatus edgeRet;

    OPS_LOG_I(PASS_NAME.c_str(), "node %s: [1/5] wiring Const→V3.0 ...", nameStr.GetString());
    edgeRet = graph.AddDataEdge(constGNode, 0, v3GNode, 0);
    if (edgeRet != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "node %s: AddDataEdge Const→V3.0 FAILED (ret=%d)", nameStr.GetString(), edgeRet);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_I(PASS_NAME.c_str(), "node %s: [2/5] wiring V2 inputs→V3 (proto V2=%zu inputs) ...",
              nameStr.GetString(), kV2TotalInputs);
    int skippedInputs = 0;
    int wiredInputs = 0;
    for (size_t v2Port = 0; v2Port < kV2TotalInputs; v2Port++) {
        auto srcPair = v2Node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(v2Port));
        if (srcPair.first == nullptr) {
            OPS_LOG_D(PASS_NAME.c_str(), "node %s: V2.port%zu src null, skip", nameStr.GetString(), v2Port);
            skippedInputs++;
            continue;
        }
        size_t v3Port = v2Port + 1;
        ge::AscendString srcName;
        srcPair.first->GetName(srcName);
        OPS_LOG_I(PASS_NAME.c_str(), "node %s: wiring %s.port%d → V3.port%zu (from V2.port%zu) ...",
                  nameStr.GetString(), srcName.GetString(), srcPair.second, v3Port, v2Port);
        edgeRet = graph.AddDataEdge(*srcPair.first, srcPair.second, v3GNode, static_cast<int32_t>(v3Port));
        if (edgeRet != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "node %s: AddDataEdge %s.port%d→V3.port%zu FAILED (ret=%d)",
                      nameStr.GetString(), srcName.GetString(), srcPair.second, v3Port, edgeRet);
            return ge::GRAPH_FAILED;
        }
        OPS_LOG_I(PASS_NAME.c_str(), "node %s:   %s.port%d → V3.port%zu OK", nameStr.GetString(),
                  srcName.GetString(), srcPair.second, v3Port);
        wiredInputs++;
    }
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: input wiring done - %d OK, %d skipped (V2_proto=%zu)",
              nameStr.GetString(), wiredInputs, skippedInputs, kV2TotalInputs);

    OPS_LOG_I(PASS_NAME.c_str(), "node %s: [3/5] collecting V2 output consumers (V2 has 1 OUTPUT) ...",
              nameStr.GetString());
    std::vector<std::vector<std::pair<ge::GNodePtr, int32_t>>> v2OutputConsumers(1);
    int totalConsumers = 0;
    for (size_t i = 0; i < 1; i++) {
        v2OutputConsumers[i] = v2Node.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
        totalConsumers += static_cast<int>(v2OutputConsumers[i].size());
        for (auto &outPair : v2OutputConsumers[i]) {
            ge::AscendString consumerName;
            outPair.first->GetName(consumerName);
            OPS_LOG_D(PASS_NAME.c_str(), "node %s: V2.port%zu→%s.port%d", nameStr.GetString(), i,
                      consumerName.GetString(), outPair.second);
        }
    }
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: %d output consumers recorded",
              nameStr.GetString(), totalConsumers);

    OPS_LOG_I(PASS_NAME.c_str(), "node %s: [4/5] removing V2 node ...", nameStr.GetString());
    graph.RemoveNode(v2Node);
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: V2 removed from graph", nameStr.GetString());

    OPS_LOG_I(PASS_NAME.c_str(), "node %s: [5/5] wiring V3 outputs→%d consumers ...",
              nameStr.GetString(), totalConsumers);
    int totalOutputEdges = 0;
    for (size_t i = 0; i < 1; i++) {
        for (auto &outPair : v2OutputConsumers[i]) {
            ge::AscendString consumerName;
            outPair.first->GetName(consumerName);
            OPS_LOG_I(PASS_NAME.c_str(), "node %s: wiring V3.port%zu → %s.port%d ...",
                      nameStr.GetString(), i, consumerName.GetString(), outPair.second);
            edgeRet = graph.AddDataEdge(v3GNode, static_cast<int32_t>(i), *outPair.first, outPair.second);
            if (edgeRet != ge::GRAPH_SUCCESS) {
                OPS_LOG_E(PASS_NAME.c_str(), "node %s: AddDataEdge V3.port%zu→%s.port%d FAILED (ret=%d)",
                          nameStr.GetString(), i, consumerName.GetString(), outPair.second, edgeRet);
                return ge::GRAPH_FAILED;
            }
            OPS_LOG_I(PASS_NAME.c_str(), "node %s:   V3.port%zu → %s.port%d OK", nameStr.GetString(),
                      i, consumerName.GetString(), outPair.second);
            totalOutputEdges++;
        }
    }
    OPS_LOG_I(PASS_NAME.c_str(), "node %s: output wiring done - %d OK (expected %d). fusion COMPLETE!",
              nameStr.GetString(), totalOutputEdges, totalConsumers);
#else
    OPS_LOG_I(PASS_NAME.c_str(), "HCOMM not available, skip V3 fusion");
#endif
    return ge::GRAPH_SUCCESS;
}

}  // namespace

ge::graphStatus MoeDistributeCombineV2FusionPass::Run(ge::GraphPtr &graph, ge::CustomPassContext &pass_context)
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
        OPS_LOG_I(PASS_NAME.c_str(), "found MoeDistributeCombineV2 node: %s", nameStr.GetString());

        ge::AscendString commAlg;
        if (gNode.GetAttr(ge::AscendString("comm_alg"), commAlg) == ge::GRAPH_SUCCESS &&
            std::string(commAlg.GetString()) == "ccu") {
            OPS_LOG_I(PASS_NAME.c_str(), "node %s: comm_alg=ccu, skip V3 fusion (delegated to V1)", nameStr.GetString());
            continue;
        }

        auto ret = ReplaceCombineV2WithV3(*graph, gNode);
        if (ret == ge::GRAPH_SUCCESS) {
            OPS_LOG_I(PASS_NAME.c_str(), "node %s: V2->V3 fusion SUCCESS", nameStr.GetString());
        } else {
            OPS_LOG_I(PASS_NAME.c_str(), "node %s: V2->V3 fusion FAILED (ret=%d), V2 kept", nameStr.GetString(), ret);
        }
    }

    OPS_LOG_I(PASS_NAME.c_str(), "Run end!");
    return ge::GRAPH_SUCCESS;
}

REG_FUSION_PASS(MoeDistributeCombineV2FusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
}  // namespace ops
#endif  // GRAPH_FUSION_SUPPORT_VERSION
