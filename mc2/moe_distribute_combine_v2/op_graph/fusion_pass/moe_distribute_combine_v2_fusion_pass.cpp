/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_distribute_combine_v2_fusion_pass.h"
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_MoeDistributeCombineV2.h"
#include "es_MoeDistributeCombineV3.h"
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "op_graph/mc2_moe_context.h"
#include "ge/ge_utils.h"

namespace ops {
const std::string FUSION_PASS_NAME = "MoeDistributeCombineV2FusionPass";
const int64_t COMBINE_CAPTURE_IDX = 0l;
const size_t V2_INPUT_NUM = 18;
const std::string COMM_ALG_CCU = "ccu";

static ge::fusion::PatternUniqPtr MakePattern()
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Start to build pattern for MoeDistributeCombineV2");
    auto graphBuilder = ge::es::EsGraphBuilder(FUSION_PASS_NAME.c_str());
    auto inputs = graphBuilder.CreateInputs<V2_INPUT_NUM>();

    std::string groupEp = "";
    int64_t epWorldSize = 0;
    int64_t epRankId = 0;
    int64_t moeExpertNum = 0;

    auto v2 = ge::es::MoeDistributeCombineV2(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
        inputs[5], inputs[6], inputs[7], inputs[8], inputs[9],
        inputs[10], inputs[11], inputs[12], inputs[13], inputs[14],
        inputs[15], inputs[16], inputs[17],
        groupEp.c_str(), epWorldSize, epRankId, moeExpertNum);

    auto graph = graphBuilder.BuildAndReset({v2});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*v2.GetProducer(), 0});
    return pattern;
}

std::vector<ge::fusion::PatternUniqPtr> MoeDistributeCombineV2FusionPass::Patterns()
{
    OPS_LOG_E(FUSION_PASS_NAME.c_str(), "=== [PASS LOADED] Patterns() called ===");
    OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Enter Patterns for MoeDistributeCombineV2FusionPass");
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    patternGraphs.emplace_back(MakePattern());
    return patternGraphs;
}

bool MoeDistributeCombineV2FusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements");

    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Not A5 platform, skip fusion.");
        return false;
    }

    ge::fusion::NodeIo v2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(COMBINE_CAPTURE_IDX, v2NodeIo) != ge::GRAPH_SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get captured V2 node failed.");

    ge::AscendString commAlgStr("");
    if (v2NodeIo.node.GetAttr("comm_alg", commAlgStr) != ge::GRAPH_SUCCESS) {
        return true;
    }
    if (std::string(commAlgStr.GetString()) == COMM_ALG_CCU) {
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "comm_alg is ccu, skip V2 fusion (use V1 fusion).");
        return false;
    }

    return true;
}

static bool GetInputsInfo(const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                          std::vector<ge::Shape> &inputShapes,
                          std::vector<ge::DataType> &inputDTypes,
                          std::vector<ge::Format> &inputFormats)
{
    for (const auto &subgraphInput : subgraphInputs) {
        auto matchNodes = subgraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            continue;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
        inputDTypes.emplace_back(tmpDesc.GetDataType());
        inputFormats.emplace_back(tmpDesc.GetOriginFormat());
    }
    return true;
}

static bool InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr &replaceGraph,
                                   const std::vector<ge::fusion::SubgraphInput> &subgraphInputs)
{
    std::vector<ge::Shape> inputShapes;
    for (const auto &subgraphInput : subgraphInputs) {
        auto matchNodes = subgraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            continue;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
    }
    return (ge::GeUtils::InferShape(*replaceGraph, inputShapes) == ge::SUCCESS);
}

static bool CreateReplaceGraphInputs(std::vector<ge::es::EsTensorHolder> &v3Inputs,
                                     ge::es::EsGraphBuilder &replaceGraphBuilder,
                                     const std::vector<ge::fusion::SubgraphInput> &subgraphInputs)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Start to create inputs in Replacement.");
    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats);

    if (inputShapes.size() < V2_INPUT_NUM) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums %zu < %zu.", inputShapes.size(), V2_INPUT_NUM);
        return false;
    }

    for (size_t i = 0; i < V2_INPUT_NUM; ++i) {
        int64_t idx = static_cast<int64_t>(i);
        v3Inputs.push_back(replaceGraphBuilder.CreateInput(
            idx, "", inputDTypes[i], inputFormats[i], inputShapes[i].GetDims()));
    }
    return true;
}

static bool ReadAttrsFromNode(
    const ge::GNode &v2Node,
    std::string &groupEp, int64_t &epWorldSize, int64_t &epRankId, int64_t &moeExpertNum,
    int64_t &tpWorldSize, int64_t &tpRankId, int64_t &expertShardType,
    int64_t &sharedExpertNum, int64_t &sharedExpertRankNum,
    int64_t &globalBs, int64_t &outDtype, int64_t &commQuantMode, int64_t &groupListType,
    std::string &commAlg,
    int64_t &zeroExpertNum, int64_t &copyExpertNum, int64_t &constExpertNum)
{
    ge::AscendString groupEpAscend("");
    OP_LOGE_IF(v2Node.GetAttr("group_ep", groupEpAscend) != ge::GRAPH_SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get group_ep attr failed.");
    groupEp = groupEpAscend.GetString();

    OP_LOGE_IF(v2Node.GetAttr("ep_world_size", epWorldSize) != ge::GRAPH_SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get ep_world_size attr failed.");

    OP_LOGE_IF(v2Node.GetAttr("ep_rank_id", epRankId) != ge::GRAPH_SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get ep_rank_id attr failed.");

    OP_LOGE_IF(v2Node.GetAttr("moe_expert_num", moeExpertNum) != ge::GRAPH_SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get moe_expert_num attr failed.");

    v2Node.GetAttr("tp_world_size", tpWorldSize);
    v2Node.GetAttr("tp_rank_id", tpRankId);
    v2Node.GetAttr("expert_shard_type", expertShardType);
    v2Node.GetAttr("shared_expert_num", sharedExpertNum);
    v2Node.GetAttr("shared_expert_rank_num", sharedExpertRankNum);
    v2Node.GetAttr("global_bs", globalBs);
    v2Node.GetAttr("out_dtype", outDtype);
    v2Node.GetAttr("comm_quant_mode", commQuantMode);
    v2Node.GetAttr("group_list_type", groupListType);

    ge::AscendString commAlgAscend("");
    v2Node.GetAttr("comm_alg", commAlgAscend);
    commAlg = commAlgAscend.GetString();

    v2Node.GetAttr("zero_expert_num", zeroExpertNum);
    v2Node.GetAttr("copy_expert_num", copyExpertNum);
    v2Node.GetAttr("const_expert_num", constExpertNum);
    return true;
}

ge::fusion::GraphUniqPtr
MoeDistributeCombineV2FusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Replacement.");

    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed in Replacement.");
        return nullptr;
    }
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Subgraph input size is %zu.", subgraphInputs.size());

    ge::fusion::NodeIo v2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(COMBINE_CAPTURE_IDX, v2NodeIo) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get captured V2 node failed.");
    ge::GNode v2Node = v2NodeIo.node;

    std::string groupEp;
    int64_t epWorldSize = 0, epRankId = 0, moeExpertNum = 0;
    int64_t tpWorldSize = 0, tpRankId = 0, expertShardType = 0;
    int64_t sharedExpertNum = 1, sharedExpertRankNum = 0;
    int64_t globalBs = 0, outDtype = 0, commQuantMode = 0, groupListType = 0;
    std::string commAlg;
    int64_t zeroExpertNum = 0, copyExpertNum = 0, constExpertNum = 0;
    if (!ReadAttrsFromNode(v2Node, groupEp, epWorldSize, epRankId, moeExpertNum,
                            tpWorldSize, tpRankId, expertShardType,
                            sharedExpertNum, sharedExpertRankNum,
                            globalBs, outDtype, commQuantMode, groupListType, commAlg,
                            zeroExpertNum, copyExpertNum, constExpertNum)) {
        return nullptr;
    }

    ge::AscendString v2NodeNameAscend("");
    v2Node.GetName(v2NodeNameAscend);
    std::string v2NodeName = v2NodeNameAscend.GetString();

#if defined(HCOMM_VERSION_NUM) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    ops::Mc2MoeContext moeContext{0};
    int64_t hcclBuffSize = 0;
    std::string contextTag = FUSION_PASS_NAME + "_" + v2NodeName + "_" + groupEp;

    auto contextStatus = ops::MoeContextManager::GetMc2ContextA5(contextTag, groupEp, moeContext, hcclBuffSize);
    if (contextStatus != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetMc2ContextA5 failed, skip fusion.");
        return nullptr;
    }

    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");

    std::vector<ge::es::EsTensorHolder> v3BoundaryInputs;
    if (!CreateReplaceGraphInputs(v3BoundaryInputs, replaceGraphBuilder, subgraphInputs)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CreateReplaceGraphInputs failed.");
        return nullptr;
    }

    size_t contextInt32Count = sizeof(ops::Mc2MoeContext) / sizeof(int32_t);
    std::vector<int32_t> contextData(
        reinterpret_cast<const int32_t*>(&moeContext),
        reinterpret_cast<const int32_t*>(&moeContext) + contextInt32Count);
    auto contextConst = replaceGraphBuilder.CreateConst<int32_t>(
        contextData, {static_cast<int64_t>(contextInt32Count)}, ge::DT_INT32);

    auto v3 = ge::es::MoeDistributeCombineV3(
        contextConst,
        v3BoundaryInputs[0], v3BoundaryInputs[1], v3BoundaryInputs[2],
        v3BoundaryInputs[3], v3BoundaryInputs[4], v3BoundaryInputs[5],
        v3BoundaryInputs[6], v3BoundaryInputs[7], v3BoundaryInputs[8],
        v3BoundaryInputs[9], v3BoundaryInputs[10], v3BoundaryInputs[11],
        v3BoundaryInputs[12], v3BoundaryInputs[13], v3BoundaryInputs[14],
        v3BoundaryInputs[15], v3BoundaryInputs[16], v3BoundaryInputs[17],
        epWorldSize, epRankId, moeExpertNum, hcclBuffSize,
        tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum,
        globalBs, outDtype, commQuantMode, groupListType, commAlg.c_str(),
        zeroExpertNum, copyExpertNum, constExpertNum);

    auto replaceGraph = replaceGraphBuilder.BuildAndReset({v3});

    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "InferShapeReplaceGraph failed.");
        return nullptr;
    }

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "MoeDistributeCombineV2 -> V3 fusion succeeded.");
    return std::move(replaceGraph);
#else
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "HCOMM_VERSION not sufficient, skip fusion.");
    return nullptr;
#endif
}

REG_FUSION_PASS(MoeDistributeCombineV2FusionPass)
    .Stage(ge::CustomPassStage::kCompatibleInherited);
} // namespace ops
#endif // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
