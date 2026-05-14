/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "distribute_barrier_fusion_pass.h"
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_DistributeBarrier.h"
#include "es_DistributeBarrierExtend.h"
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "op_graph/mc2_moe_context.h"
#include "ge/ge_utils.h"

namespace ops {
const std::string FUSION_PASS_NAME = "DistributeBarrierFusionPass";
const int64_t BARRIER_CAPTURE_IDX = 0l;
const size_t V1_INPUT_NUM = 3;

static ge::fusion::PatternUniqPtr MakePattern()
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Start to build pattern for DistributeBarrier");
    auto graphBuilder = ge::es::EsGraphBuilder(FUSION_PASS_NAME.c_str());
    auto inputs = graphBuilder.CreateInputs<V1_INPUT_NUM>();

    std::string group = "";
    int64_t worldSize = 0;

    auto barrier = ge::es::DistributeBarrier(
        inputs[0], inputs[1], inputs[2],
        group.c_str(), worldSize);

    auto graph = graphBuilder.BuildAndReset({barrier});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*barrier.GetProducer(), 0});
    return pattern;
}

std::vector<ge::fusion::PatternUniqPtr> DistributeBarrierFusionPass::Patterns()
{
    OPS_LOG_E(FUSION_PASS_NAME.c_str(), "=== [PASS LOADED] Patterns() called ===");
    OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Enter Patterns for DistributeBarrierFusionPass");
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    patternGraphs.emplace_back(MakePattern());
    return patternGraphs;
}

bool DistributeBarrierFusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements");

    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Not A5 platform, skip fusion.");
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

static bool CreateReplaceGraphInputs(std::vector<ge::es::EsTensorHolder> &extendInputs,
                                     ge::es::EsGraphBuilder &replaceGraphBuilder,
                                     const std::vector<ge::fusion::SubgraphInput> &subgraphInputs)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Start to create inputs in Replacement.");
    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats);

    if (inputShapes.size() < V1_INPUT_NUM) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums %zu < %zu.", inputShapes.size(), V1_INPUT_NUM);
        return false;
    }

    for (size_t i = 0; i < V1_INPUT_NUM; ++i) {
        int64_t idx = static_cast<int64_t>(i);
        extendInputs.push_back(replaceGraphBuilder.CreateInput(
            idx, "", inputDTypes[i], inputFormats[i], inputShapes[i].GetDims()));
    }
    return true;
}

ge::fusion::GraphUniqPtr
DistributeBarrierFusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Replacement.");

    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed in Replacement.");
        return nullptr;
    }
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Subgraph input size is %zu.", subgraphInputs.size());

    ge::fusion::NodeIo barrierNodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(BARRIER_CAPTURE_IDX, barrierNodeIo) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get captured Barrier node failed.");
    ge::GNode barrierNode = barrierNodeIo.node;

    ge::AscendString groupAscend("");
    OP_LOGE_IF(barrierNode.GetAttr("group", groupAscend) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get group attr failed.");
    std::string group = groupAscend.GetString();

    int64_t worldSize = 0;
    OP_LOGE_IF(barrierNode.GetAttr("world_size", worldSize) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get world_size attr failed.");

    ge::AscendString barrierNodeNameAscend("");
    barrierNode.GetName(barrierNodeNameAscend);
    std::string barrierNodeName = barrierNodeNameAscend.GetString();

#if defined(HCOMM_VERSION_NUM) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    ops::Mc2MoeContext moeContext{0};
    int64_t hcclBuffSize = 0;
    std::string contextTag = FUSION_PASS_NAME + "_" + barrierNodeName + "_" + group;

    auto contextStatus = ops::MoeContextManager::GetMc2ContextA5(contextTag, group, moeContext, hcclBuffSize);
    if (contextStatus != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetMc2ContextA5 failed, skip fusion.");
        return nullptr;
    }

    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");

    std::vector<ge::es::EsTensorHolder> extendBoundaryInputs;
    if (!CreateReplaceGraphInputs(extendBoundaryInputs, replaceGraphBuilder, subgraphInputs)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CreateReplaceGraphInputs failed.");
        return nullptr;
    }

    size_t contextInt32Count = sizeof(ops::Mc2MoeContext) / sizeof(int32_t);
    std::vector<int32_t> contextData(
        reinterpret_cast<const int32_t*>(&moeContext),
        reinterpret_cast<const int32_t*>(&moeContext) + contextInt32Count);
    auto contextConst = replaceGraphBuilder.CreateConst<int32_t>(
        contextData, {static_cast<int64_t>(contextInt32Count)}, ge::DT_INT32);

    auto extend = ge::es::DistributeBarrierExtend(
        contextConst,
        extendBoundaryInputs[0], extendBoundaryInputs[1], extendBoundaryInputs[2],
        group.c_str(), worldSize);

    auto replaceGraph = replaceGraphBuilder.BuildAndReset({extend});

    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "InferShapeReplaceGraph failed.");
        return nullptr;
    }

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "DistributeBarrier -> Extend fusion succeeded.");
    return std::move(replaceGraph);
#else
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "HCOMM_VERSION not sufficient, skip fusion.");
    return nullptr;
#endif
}

REG_FUSION_PASS(DistributeBarrierFusionPass)
    .Stage(ge::CustomPassStage::kCompatibleInherited);
} // namespace ops
#endif // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
