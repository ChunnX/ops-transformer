/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * ...
 */

#include "moe_distribute_dispatch_v2_fusion_pass.h"
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include <unordered_map>
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
constexpr uint32_t DISPATCH_V3_OPTIONAL_ATTR_NUM = 13;
const std::string PASS_NAME = "MoeDistributeDispatchV2FusionPass";

const std::unordered_map<int, std::pair<std::string, std::string>> dispatchV3OptionalAttr = {
    {0, {"int", "tp_world_size"}}, {1, {"int", "tp_rank_id"}}, {2, {"int", "expert_shard_type"}},
    {3, {"int", "shared_expert_num"}}, {4, {"int", "shared_expert_rank_num"}}, {5, {"int", "quant_mode"}},
    {6, {"int", "global_bs"}}, {7, {"int", "expert_token_nums_type"}}, {8, {"string", "comm_alg"}},
    {9, {"int", "zero_expert_num"}}, {10, {"int", "copy_expert_num"}}, {11, {"int", "const_expert_num"}},
    {12, {"int", "y_dtype"}}};

void CopyRequiredAttr(ge::Operator &dst, const ge::GNode &src)
{
    auto copyIntAttr = [&](const std::string &name) {
        int64_t val = 0;
        if (src.GetAttr(ge::AscendString(name.c_str()), val) == ge::GRAPH_SUCCESS) {
            dst.SetAttr(name, val);
        }
    };
    copyIntAttr("ep_world_size");
    copyIntAttr("ep_rank_id");
    copyIntAttr("moe_expert_num");
}

void CopyOptionalAttrs(ge::Operator &dst, const ge::GNode &src, int64_t cclBufferSize)
{
    dst.SetAttr("ccl_buffer_size", cclBufferSize);
    for (uint32_t i = 0; i < DISPATCH_V3_OPTIONAL_ATTR_NUM; i++) {
        const auto &opt = dispatchV3OptionalAttr.at(i);
        if (opt.first == "string") {
            ge::AscendString val;
            if (src.GetAttr(ge::AscendString(opt.second.c_str()), val) == ge::GRAPH_SUCCESS) {
                dst.SetAttr(opt.second, std::string(val.GetString()));
            }
        } else {
            int64_t val = 0;
            if (src.GetAttr(ge::AscendString(opt.second.c_str()), val) == ge::GRAPH_SUCCESS) {
                dst.SetAttr(opt.second, val);
            }
        }
    }
}

ge::GNode CreateContextConstNode(ge::Graph &graph, const ops::Mc2MoeContext &context)
{
    std::string name = PASS_NAME + "_context";
    ge::Shape shape({static_cast<int64_t>(sizeof(ops::Mc2MoeContext) / sizeof(int32_t))});
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    ge::Tensor tensorValue(tensorDesc, reinterpret_cast<const uint8_t*>(&context), sizeof(ops::Mc2MoeContext));
    ge::Operator constOp(name.c_str(), "Const");
    ge::AttrValue attr;
    attr.SetAttrValue(tensorValue);
    constOp.SetAttr("value", std::move(attr));
    return graph.AddNodeByOp(constOp);
}

bool IsA5Platform()
{
    return ops::IsTargetPlatformNpuArch(PASS_NAME.c_str(), ops::NPUARCH_A5);
}

bool IsCCUComm(const ge::GNode &node)
{
    ge::AscendString commAlg;
    if (node.GetAttr(ge::AscendString("comm_alg"), commAlg) != ge::GRAPH_SUCCESS) {
        return false;
    }
    return (std::string(commAlg.GetString()) == "ccu");
}

ge::graphStatus ReplaceDispatchV2WithV3(ge::Graph &graph, ge::GNode &v2Node)
{
    // 1. Get HCCL context data
#if defined(HCOMM_VERSION_NUM) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    ops::Mc2MoeContext mc2Ctx;
    int64_t cclBufferSize = 0;
    if (ops::MoeContextManager::GetMc2ContextA5(
            "MOE_DISPATCH_COMBINE_V2_CONTEXT", "group_ep", mc2Ctx, cclBufferSize) != ge::GRAPH_SUCCESS) {
        OPS_LOG_I(PASS_NAME.c_str(), "GetMc2ContextA5 failed, skip");
        return ge::GRAPH_SUCCESS;
    }
    // 2. Create context Const node
    auto contextNode = CreateContextConstNode(graph, mc2Ctx);
    (void)contextNode;
    // 3. Create V3 node
    std::string v3Name;
    {
        ge::AscendString nameStr;
        v2Node.GetName(nameStr);
        v3Name = std::string(nameStr.GetString()) + "_V3";
    }
    ge::Operator v3Op(v3Name.c_str(), "MoeDistributeDispatchV3");
    CopyRequiredAttr(v3Op, v2Node);
    CopyOptionalAttrs(v3Op, v2Node, cclBufferSize);
    auto v3Node = graph.AddNodeByOp(v3Op);
    // 4. Re-wire: context input → v3 input0, v2 inputs → v3 inputs[1..N]
    graph.AddDataEdge(contextNode, 0, v3Node, 0);
    size_t inputSize = v2Node.GetInputsSize();
    for (size_t i = 0; i < inputSize; i++) {
        auto srcPair = v2Node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        graph.AddDataEdge(*srcPair.first, srcPair.second, v3Node, static_cast<int32_t>(i + 1));
    }
    // 5. Re-wire outputs: v2 consumers → v3
    size_t outputSize = v2Node.GetOutputsSize();
    for (size_t i = 0; i < outputSize; i++) {
        auto outNodes = v2Node.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
        for (auto &outPair : outNodes) {
            graph.AddDataEdge(v3Node, static_cast<int32_t>(i), *outPair.first, outPair.second);
        }
    }
    // 6. Remove old V2 node
    graph.RemoveNode(v2Node);
#endif
    return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus MoeDistributeDispatchV2FusionPass::Run(ge::GraphPtr &graph, ge::CustomPassContext &pass_context)
{
    OPS_LOG_I(PASS_NAME.c_str(), "MoeDistributeDispatchV2FusionPass Run start!");
    for (auto &gNode : graph->GetDirectNode()) {
        ge::AscendString nodeType;
        if (gNode.GetType(nodeType) != ge::GRAPH_SUCCESS || nodeType != "MoeDistributeDispatchV2") {
            continue;
        }
        if (ops::IsTargetPlatformSocVersion(PASS_NAME.c_str(), ops::PLATFORM_A2)) continue;
        if (!IsA5Platform()) continue;
        if (IsCCUComm(gNode)) {
            OPS_LOG_I(PASS_NAME.c_str(), "comm_alg is ccu, skip V2 fusion");
            continue;
        }
        OPS_LOG_I(PASS_NAME.c_str(), "found MoeDistributeDispatchV2 node, replacing with V3...");
        ReplaceDispatchV2WithV3(*graph, gNode);
    }
    OPS_LOG_I(PASS_NAME.c_str(), "MoeDistributeDispatchV2FusionPass Run end!");
    return ge::GRAPH_SUCCESS;
}

REG_FUSION_PASS(MoeDistributeDispatchV2FusionPass)
    .Stage(ge::CustomPassStage::kCompatibleInherited);
}  // namespace ops
#endif // GRAPH_FUSION_SUPPORT_VERSION
