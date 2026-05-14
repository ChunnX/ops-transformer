/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "distribute_barrier_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION

#include "es_DistributeBarrier.h"
#include "es_DistributeBarrierExtend.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/match_result.h"
#include "ge/fusion/subgraph_boundary.h"
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "../../common/op_graph/mc2_moe_context.h"
#include <vector>
#include <string>
#include <cstring>

namespace ops {

constexpr size_t MOE_CONTEXT_SIZE_INT32 = 2052;
const std::string FUSION_PASS_NAME = "DistributeBarrierFusionPass";

static bool IsTargetPlatformA5()
{
    return IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5);
}

std::vector<ge::fusion::PatternUniqPtr> DistributeBarrierFusionPass::Patterns()
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Patterns");
    std::vector<ge::fusion::PatternUniqPtr> patterns;

    auto graphBuilder = ge::es::EsGraphBuilder("DistributeBarrierPattern");
    auto inputs = graphBuilder.CreateInputs(3);

    std::string group = "";
    int64_t worldSize = 0;

    auto output = ge::es::DistributeBarrier(inputs[0], inputs[1], inputs[2], group.c_str(), worldSize);

    std::vector<ge::es::EsTensorHolder> outputTensors = {output};
    auto graph = graphBuilder.BuildAndReset(outputTensors);
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*output.GetProducer(), 0});

    patterns.emplace_back(std::move(pattern));
    return patterns;
}

bool DistributeBarrierFusionPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements");
    return IsTargetPlatformA5();
}

ge::fusion::GraphUniqPtr
DistributeBarrierFusionPass::Replacement(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Replacement");

    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "GetAllInputs failed");
        return nullptr;
    }

    ge::fusion::NodeIo barrierNodeIo;
    matchResult->GetCapturedTensor(0, barrierNodeIo);
    ge::GNode barrierNode = barrierNodeIo.node;

    Mc2MoeContextLite context{};
    int64_t hcclBuffSize = 0;
    ge::AscendString group;
    barrierNode.GetAttr("group", group);
    std::string nodeName;
    ge::AscendString nodeNameStr;
    barrierNode.GetName(nodeNameStr);
    nodeName = nodeNameStr.GetString();

    auto status = GetMc2ContextDataLite(nodeName, std::string(group.GetString()), context, hcclBuffSize);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetMc2ContextDataLite failed, use zero context");
        hcclBuffSize = 0;
    }

    auto replaceGraphBuilder = ge::es::EsGraphBuilder("DistributeBarrierReplacement");

    std::vector<int32_t> contextData(MOE_CONTEXT_SIZE_INT32);
    std::memcpy(contextData.data(), &context, sizeof(context));
    auto contextTensor = replaceGraphBuilder.CreateConst(contextData, {MOE_CONTEXT_SIZE_INT32}, ge::DT_INT32);

    std::vector<ge::es::EsTensorHolder> inputs;
    for (size_t i = 0; i < subgraphInputs.size(); ++i) {
        inputs.emplace_back(replaceGraphBuilder.CreateInput(static_cast<int64_t>(i)));
    }

    int64_t worldSize = 0;
    barrierNode.GetAttr("world_size", worldSize);

    ge::es::EsTensorHolder emptyTensor;
    auto output = ge::es::DistributeBarrierExtend(
        contextTensor, inputs.size() > 0 ? inputs[0] : replaceGraphBuilder.CreateInput(0),
        inputs.size() > 1 ? inputs[1] : emptyTensor, inputs.size() > 2 ? inputs[2] : emptyTensor, group.GetString(),
        worldSize);

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement done");
    std::vector<ge::es::EsTensorHolder> outputTensors = {output};
    return replaceGraphBuilder.BuildAndReset(outputTensors);
}

REG_FUSION_PASS(DistributeBarrierFusionPass);

} // namespace ops

#endif // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION