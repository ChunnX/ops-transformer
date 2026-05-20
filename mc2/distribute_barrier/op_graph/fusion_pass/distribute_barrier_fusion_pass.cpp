/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * ...
 */

#include "distribute_barrier_fusion_pass.h"
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "graph/graph.h"
#include "graph/gnode.h"
#include "graph/operator.h"
#include "graph/tensor.h"
#include "graph/ge_error_codes.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "mc2_common_log.h"
#include "mc2_platform_info.h"
#include "op_graph/mc2_moe_context.h"

namespace ops {
namespace {
const std::string PASS_NAME = "DistributeBarrierFusionPass";

ge::GNode CreateContextConstNode(ge::Graph &graph, const ops::Mc2MoeContext &context)
{
    ge::Shape shape({static_cast<int64_t>(sizeof(ops::Mc2MoeContext) / sizeof(int32_t))});
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    ge::Tensor tensorValue(tensorDesc, reinterpret_cast<const uint8_t*>(&context), sizeof(ops::Mc2MoeContext));
    ge::Operator constOp((PASS_NAME + "_context").c_str(), "Const");
    ge::AttrValue attr;
    attr.SetAttrValue(tensorValue);
    constOp.SetAttr("value", std::move(attr));
    return graph.AddNodeByOp(constOp);
}

ge::graphStatus ReplaceBarrierWithExtend(ge::Graph &graph, ge::GNode &v2Node)
{
#if defined(HCOMM_VERSION_NUM) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    ops::Mc2MoeContext mc2Ctx;
    int64_t cclBufferSize = 0;
    if (ops::MoeContextManager::GetMc2ContextA5(
            "DistributeBarrier", "group", mc2Ctx, cclBufferSize) != ge::GRAPH_SUCCESS) {
        OPS_LOG_I(PASS_NAME.c_str(), "GetMc2ContextA5 failed, skip");
        return ge::GRAPH_SUCCESS;
    }
    auto contextNode = CreateContextConstNode(graph, mc2Ctx);
    ge::AscendString nameStr;
    v2Node.GetName(nameStr);
    ge::Operator v3Op((std::string(nameStr.GetString()) + "_Extend").c_str(), "DistributeBarrierExtend");
    // copy attrs
    int64_t worldSize = 0;
    ge::AscendString group;
    if (v2Node.GetAttr(ge::AscendString("world_size"), worldSize) == ge::GRAPH_SUCCESS) {
        v3Op.SetAttr("world_size", worldSize);
    }
    if (v2Node.GetAttr(ge::AscendString("group"), group) == ge::GRAPH_SUCCESS) {
        v3Op.SetAttr("group", std::string(group.GetString()));
    }
    auto v3Node = graph.AddNodeByOp(v3Op);
    graph.AddDataEdge(contextNode, 0, v3Node, 0);
    size_t inputSize = v2Node.GetInputsSize();
    for (size_t i = 0; i < inputSize; i++) {
        auto srcPair = v2Node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        graph.AddDataEdge(*srcPair.first, srcPair.second, v3Node, static_cast<int32_t>(i + 1));
    }
    size_t outputSize = v2Node.GetOutputsSize();
    for (size_t i = 0; i < outputSize; i++) {
        auto outNodes = v2Node.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
        for (auto &outPair : outNodes) {
            graph.AddDataEdge(v3Node, static_cast<int32_t>(i), *outPair.first, outPair.second);
        }
    }
    graph.RemoveNode(v2Node);
#endif
    return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus DistributeBarrierFusionPass::Run(ge::GraphPtr &graph, ge::CustomPassContext &pass_context)
{
    OPS_LOG_I(PASS_NAME.c_str(), "DistributeBarrierFusionPass Run start!");
    for (auto &gNode : graph->GetDirectNode()) {
        ge::AscendString nodeType;
        if (gNode.GetType(nodeType) != ge::GRAPH_SUCCESS || nodeType != "DistributeBarrier") continue;
        if (!ops::IsTargetPlatformNpuArch(PASS_NAME.c_str(), ops::NPUARCH_A5)) continue;
        OPS_LOG_I(PASS_NAME.c_str(), "found DistributeBarrier node, replacing with Extend...");
        ReplaceBarrierWithExtend(*graph, gNode);
    }
    OPS_LOG_I(PASS_NAME.c_str(), "DistributeBarrierFusionPass Run end!");
    return ge::GRAPH_SUCCESS;
}

REG_FUSION_PASS(DistributeBarrierFusionPass)
    .Stage(ge::CustomPassStage::kCompatibleInherited);
}  // namespace ops
#endif // GRAPH_FUSION_SUPPORT_VERSION
