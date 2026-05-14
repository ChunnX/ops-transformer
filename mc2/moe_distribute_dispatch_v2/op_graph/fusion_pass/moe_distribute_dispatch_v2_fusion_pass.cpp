/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_distribute_dispatch_v2_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION

#include "es_MoeDistributeDispatchV2.h"
#include "es_MoeDistributeDispatchV3.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/match_result.h"
#include "ge/fusion/subgraph_boundary.h"
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "../../common/op_graph/mc2_moe_context_lite.h"
#include <vector>
#include <string>
#include <cstring>

namespace ops {

constexpr size_t MOE_CONTEXT_SIZE_INT32 = 2052;
const std::string FUSION_PASS_NAME = "MoeDistributeDispatchV2FusionPass";

static bool IsTargetPlatformA5()
{
    return IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5);
}

std::vector<ge::fusion::PatternUniqPtr> MoeDistributeDispatchV2FusionPass::Patterns()
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Patterns");
    std::vector<ge::fusion::PatternUniqPtr> patterns;

    auto graphBuilder = ge::es::EsGraphBuilder("MoeDistributeDispatchV2Pattern");
    auto inputs = graphBuilder.CreateInputs(7);

    std::string groupEp = "";
    int64_t epWorldSize = 0;
    int64_t epRankId = 0;
    int64_t moeExpertNum = 0;

    auto outputs = ge::es::MoeDistributeDispatchV2(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                                                   inputs[6], groupEp.c_str(), epWorldSize, epRankId, moeExpertNum);

    std::vector<ge::es::EsTensorHolder> outputTensors = {outputs.expand_x};
    auto graph = graphBuilder.BuildAndReset(outputTensors);
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*outputs.expand_x.GetProducer(), 0});

    patterns.emplace_back(std::move(pattern));
    return patterns;
}

bool MoeDistributeDispatchV2FusionPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements");

    if (!IsTargetPlatformA5()) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Not A5 platform");
        return false;
    }

    ge::fusion::NodeIo dispatchV2NodeIo;
    if (!matchResult->GetCapturedTensor(0, dispatchV2NodeIo)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetCapturedTensor failed");
        return false;
    }

    ge::GNode dispatchV2Node = dispatchV2NodeIo.node;
    ge::AscendString commAlg;
    if (!dispatchV2Node.GetAttr("comm_alg", commAlg)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetAttr comm_alg failed");
        return false;
    }

    std::string commAlgStr(commAlg.GetString());
    if (commAlgStr == "ccu") {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "comm_alg is ccu, skip");
        return false;
    }

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MeetRequirements passed");
    return true;
}

ge::fusion::GraphUniqPtr
MoeDistributeDispatchV2FusionPass::Replacement(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Replacement");

    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "GetAllInputs failed");
        return nullptr;
    }

    ge::fusion::NodeIo dispatchV2NodeIo;
    matchResult->GetCapturedTensor(0, dispatchV2NodeIo);
    ge::GNode dispatchV2Node = dispatchV2NodeIo.node;

    Mc2MoeContextLite context{};
    int64_t hcclBuffSize = 0;
    ge::AscendString groupEp;
    dispatchV2Node.GetAttr("group_ep", groupEp);
    std::string nodeName;
    ge::AscendString nodeNameStr;
    dispatchV2Node.GetName(nodeNameStr);
    nodeName = nodeNameStr.GetString();

    auto status = GetMc2ContextDataLite(nodeName, std::string(groupEp.GetString()), context, hcclBuffSize);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetMc2ContextDataLite failed, use zero context");
        hcclBuffSize = 0;
    }

    auto replaceGraphBuilder = ge::es::EsGraphBuilder("MoeDistributeDispatchV2Replacement");

    std::vector<int32_t> contextData(MOE_CONTEXT_SIZE_INT32);
    std::memcpy(contextData.data(), &context, sizeof(context));
    auto contextTensor = replaceGraphBuilder.CreateConst(contextData, {MOE_CONTEXT_SIZE_INT32}, ge::DT_INT32);

    std::vector<ge::es::EsTensorHolder> inputs;
    for (size_t i = 0; i < subgraphInputs.size(); ++i) {
        inputs.emplace_back(replaceGraphBuilder.CreateInput(static_cast<int64_t>(i)));
    }

    int64_t epWorldSize = 0;
    dispatchV2Node.GetAttr("ep_world_size", epWorldSize);
    int64_t epRankId = 0;
    dispatchV2Node.GetAttr("ep_rank_id", epRankId);
    int64_t moeExpertNum = 0;
    dispatchV2Node.GetAttr("moe_expert_num", moeExpertNum);
    int64_t tpWorldSize = 0;
    dispatchV2Node.GetAttr("tp_world_size", tpWorldSize);
    int64_t tpRankId = 0;
    dispatchV2Node.GetAttr("tp_rank_id", tpRankId);
    int64_t expertShardType = 0;
    dispatchV2Node.GetAttr("expert_shard_type", expertShardType);
    int64_t sharedExpertNum = 1;
    dispatchV2Node.GetAttr("shared_expert_num", sharedExpertNum);
    int64_t sharedExpertRankNum = 0;
    dispatchV2Node.GetAttr("shared_expert_rank_num", sharedExpertRankNum);
    int64_t quantMode = 0;
    dispatchV2Node.GetAttr("quant_mode", quantMode);
    int64_t globalBs = 0;
    dispatchV2Node.GetAttr("global_bs", globalBs);
    int64_t expertTokenNumsType = 1;
    dispatchV2Node.GetAttr("expert_token_nums_type", expertTokenNumsType);
    ge::AscendString commAlg;
    dispatchV2Node.GetAttr("comm_alg", commAlg);
    int64_t zeroExpertNum = 0;
    dispatchV2Node.GetAttr("zero_expert_num", zeroExpertNum);
    int64_t copyExpertNum = 0;
    dispatchV2Node.GetAttr("copy_expert_num", copyExpertNum);
    int64_t constExpertNum = 0;
    dispatchV2Node.GetAttr("const_expert_num", constExpertNum);
    int64_t yDtype = 28;
    dispatchV2Node.GetAttr("y_dtype", yDtype);

    ge::es::EsTensorHolder emptyTensor;
    auto outputs = ge::es::MoeDistributeDispatchV3(
        contextTensor, inputs.size() > 0 ? inputs[0] : replaceGraphBuilder.CreateInput(0),
        inputs.size() > 1 ? inputs[1] : replaceGraphBuilder.CreateInput(1), inputs.size() > 2 ? inputs[2] : emptyTensor,
        inputs.size() > 3 ? inputs[3] : emptyTensor, inputs.size() > 4 ? inputs[4] : emptyTensor,
        inputs.size() > 5 ? inputs[5] : emptyTensor, inputs.size() > 6 ? inputs[6] : emptyTensor, epWorldSize, epRankId,
        moeExpertNum, hcclBuffSize, tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum,
        quantMode, globalBs, expertTokenNumsType, commAlg.GetString(), zeroExpertNum, copyExpertNum, constExpertNum,
        yDtype);

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Replacement done");
    std::vector<ge::es::EsTensorHolder> outputTensors = {outputs.expand_x};
    return replaceGraphBuilder.BuildAndReset(outputTensors);
}

REG_FUSION_PASS(MoeDistributeDispatchV2FusionPass);

} // namespace ops

#endif // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION