/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_distribute_dispatch_v2_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_MoeDistributeDispatchV2.h"
#include "es_MoeDistributeDispatchV3.h"
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"

#include "ge/es_graph_builder.h"
#include "ge/fusion/match_result.h"
#include "ge/fusion/subgraph_boundary.h"

#include "../../common/op_graph/mc2_moe_context.h"
#include <vector>
#include <string>
#include <cstring>
#include <dlfcn.h> // dlopen 动态加载

namespace ops {
const std::string FUSION_PASS_NAME = "MoeDistributeDispatchV2FusionPass";

static bool IsMTECommAlg(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(0, mc2NodeIo) != ge::SUCCESS, false, FUSION_PASS_NAME.c_str(),
               "Get MoeDistributeDispatchV2 node failed.");
    ge::GNode mc2Node = mc2NodeIo.node;

    ge::AscendString commAlg;
    if (mc2Node.GetAttr("comm_alg", commAlg) != ge::GRAPH_SUCCESS) {
        OPS_LOG_W(FUSION_PASS_NAME.c_str(), "Get comm_alg Attr failed.");
        return false;
    }

    std::string commAlgStr(commAlg.GetString());
    if (commAlgStr == "ccu") {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "comm_alg is ccu, skip");
        return false;
    }
    return true;
}

std::vector<ge::fusion::PatternUniqPtr> MoeDistributeDispatchV2FusionPass::Patterns()
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Patterns for MoeDistributeDispatchV2FusionPass");
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;

    auto graphBuilder = ge::es::EsGraphBuilder(FUSION_PASS_NAME.c_str());
    auto inputs = graphBuilder.CreateInputs<7>();
    // 必选attr 设置默认值
    std::string groupEp = "";
    int64_t epWorldSize = 0;
    int64_t epRankId = 0;
    int64_t moeExpertNum = 0;
    // 有输入的data节点传入默认, 无输入的可选节点传入nullptr
    auto mc2 = ge::es::MoeDistributeDispatchV2(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                                               inputs[6], groupEp.c_str(), epWorldSize, epRankId, moeExpertNum);

    auto graph = graphBuilder.BuildAndReset({mc2});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.expand_x.GetProducer(), 0});

    patternGraphs.emplace_back(std::move(pattern));
    return patternGraphs;
}

bool MoeDistributeDispatchV2FusionPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements for MoeDistributeDispatchV2FusionPass");

    // 是否ascend950
    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Check target platform fail!");
        return false;
    }

    // 是否MTE通信，CCU则不融合
    if (IsMTECommAlg(matchResult)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Commalg is ccu, fusion will be skipped.");
        return false;
    }

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Found One pattern that meets requirements");
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

    Mc2MoeContext context{};
    int64_t hcclBuffSize = 0;
    ge::AscendString groupEp;
    dispatchV2Node.GetAttr("group_ep", groupEp);
    std::string nodeName;
    ge::AscendString nodeNameStr;
    dispatchV2Node.GetName(nodeNameStr);
    nodeName = nodeNameStr.GetString();

    auto status =
        Mc2MoeContextManager::GetMc2ContextData(nodeName, std::string(groupEp.GetString()), context, hcclBuffSize);
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetMc2ContextData failed, use zero context");
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